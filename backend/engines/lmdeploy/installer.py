import asyncio
import os
import shutil
from typing import Any, Dict, Optional

from backend.data_store import get_store
from backend.engines.python.installer import PythonVenvInstaller, unique_version_name, utcnow
from backend.proxy.llama_swap.manager import mark_swap_config_stale
from backend.logging_config import get_logger


logger = get_logger(__name__)

_manager_instance: Optional["LMDeployInstaller"] = None


def get_lmdeploy_manager() -> "LMDeployInstaller":
    """Singleton accessor, mirroring the llama manager pattern."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = LMDeployInstaller()
    return _manager_instance


class LMDeployInstaller(PythonVenvInstaller):
    """
    Manage LMDeploy installation into its own venv, similar in spirit to LlamaManager.

    Responsibilities:
    - Create a dedicated venv under data/lmdeploy
    - Install LMDeploy from PyPI (release) or from a git source checkout
    - Track install status, version, binary path and venv path
    - Emit progress events so the UI can show logs and status
    """

    MANAGER_NAME = "lmdeploy"

    OPERATION_DESCRIPTIONS = {
        "install": "Install LMDeploy",
        "install_source": "Install LMDeploy from Source",
        "sync_source": "Sync LMDeploy Source",
        "remove": "Remove LMDeploy",
    }

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        self.distribution_names = ("lmdeploy",)
        super().__init__(
            engine_id="lmdeploy",
            label="LMDeploy",
            root_name="lmdeploy",
            log_name="lmdeploy_install.log",
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
            state_name="lmdeploy_manager.json",
        )

    def _resolve_binary_path(self) -> Optional[str]:
        override = os.getenv("LMDEPLOY_BIN")
        if override:
            override_path = os.path.abspath(os.path.expanduser(override))
            if os.path.exists(override_path):
                return override_path
            resolved_override = shutil.which(override)
            if resolved_override:
                return resolved_override

        candidate = self._venv_bin("lmdeploy")
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)

        return shutil.which("lmdeploy")

    # --- Public interface -----------------------------------------------------------

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install LMDeploy from PyPI into its own venv."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another LMDeploy operation is already running")
            await self._start_operation("install")
            dir_name = self._bind_install_dir(reuse_dir, "pip")
            pending_version = existing_version or dir_name
            self._register_pending_version(
                pending_version,
                {"install_type": "pip", "package_version": version},
            )
            args = ["install", "--upgrade"]
            if force_reinstall:
                args.append("--force-reinstall")
            package = "lmdeploy"
            if version:
                package = f"lmdeploy=={version}"
            args.append(package)

            async def _runner():
                try:
                    code = await self._run_pip(args, "install")
                    if code != 0:
                        raise RuntimeError(f"pip exited with status {code}")
                    detected_version = self._detect_installed_version()
                    self._update_installed_state(True, detected_version)
                    try:
                        store = get_store()
                        if existing_version:
                            version_name = pending_version
                        else:
                            base = detected_version or pending_version
                            version_name = (
                                pending_version
                                if base == pending_version
                                else unique_version_name(store, "lmdeploy", base)
                            )
                        meta: Dict[str, Any] = {
                            "version": version_name,
                            "install_type": "pip",
                            "package_version": detected_version,
                            "venv_path": self._venv_path,
                            "install_dir": self._base_dir,
                            "installed_at": utcnow(),
                        }
                        self._ready_pending_version(pending_version, meta)
                        store.set_active_engine_version("lmdeploy", version_name)
                        try:
                            from backend.engines.scan.scanner import scan_engine_version

                            scan_engine_version(store, "lmdeploy", meta)
                        except Exception as scan_e:
                            logger.warning(
                                "LMDeploy param scan after pip install: %s", scan_e
                            )
                        mark_swap_config_stale()
                    except Exception as exc:
                        logger.debug(
                            f"Failed to persist LMDeploy engine metadata: {exc}"
                        )
                    await self._finish_operation(True, "LMDeploy installed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(pending_version, str(exc), {"install_type": "pip"})
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response("LMDeploy installation started")

    async def install_from_source(
        self,
        repo_url: str = "https://github.com/InternLM/lmdeploy.git",
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install LMDeploy from a git repo and branch (for development)."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another LMDeploy operation is already running")
            await self._start_operation("install_source")
            dir_name = self._bind_install_dir(reuse_dir, "source")
            pending_version = existing_version or dir_name
            from backend.repo_identity import source_build_type_labels_for_engine

            type_labels = source_build_type_labels_for_engine("lmdeploy", repo_url)
            self._register_pending_version(
                pending_version,
                {
                    "type": type_labels["type"],
                    "install_type": type_labels["install_type"],
                    "is_fork": type_labels["is_fork"],
                    "source_repo": repo_url,
                    "source_branch": branch,
                },
            )
            clone_dir = os.path.join(self._base_dir, "source")

            async def _runner():
                workspace = None
                source_checkout = clone_dir
                try:
                    from backend.engines.build_workspace import (
                        BuildWorkspace,
                        ccache_environment,
                    )

                    self._ensure_venv()
                    reuse = bool(existing_version or reuse_dir)
                    if reuse:
                        if os.path.exists(source_checkout):
                            shutil.rmtree(source_checkout)
                        os.makedirs(source_checkout, exist_ok=True)
                        proc = await asyncio.create_subprocess_exec(
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            branch,
                            repo_url,
                            source_checkout,
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        await proc.wait()
                        if proc.returncode != 0:
                            raise RuntimeError(
                                f"git clone failed with code {proc.returncode}"
                            )
                    else:
                        workspace = BuildWorkspace.open("lmdeploy", repo_url, {"kind": "source"})
                        await asyncio.to_thread(workspace.acquire)
                        source_checkout = workspace.checkout_dir
                        await asyncio.to_thread(workspace.sync_git, repo_url, branch)
                    install_env = os.environ.copy()
                    install_env.update(
                        ccache_environment(
                            os.path.dirname(source_checkout),
                            launchers=True,
                            cuda=True,
                        )
                    )
                    code = await self._run_pip(
                        ["install", "-v", "-e", "."],
                        "install_source",
                        cwd=source_checkout,
                        env=install_env,
                    )
                    if code != 0:
                        raise RuntimeError(
                            f"pip install -e -v . failed with code {code}"
                        )
                    if workspace is not None:
                        source_checkout = await asyncio.to_thread(
                            workspace.seal_installed_source,
                            os.path.join(self._base_dir, "source"),
                            self._venv_path,
                        )
                    detected = self._detect_installed_version()
                    self._update_installed_state(True, detected)
                    try:
                        store = get_store()
                        if existing_version:
                            version_name = pending_version
                        else:
                            base_version = detected or branch or pending_version
                            version_name = unique_version_name(
                                store, "lmdeploy", f"{base_version}-{utcnow()}"
                            )
                        from backend.repo_identity import (
                            source_build_type_labels_for_engine,
                        )

                        type_labels = source_build_type_labels_for_engine(
                            "lmdeploy", repo_url
                        )
                        meta: Dict[str, Any] = {
                            "version": version_name,
                            "type": type_labels["type"],
                            "install_type": type_labels["install_type"],
                            "is_fork": type_labels["is_fork"],
                            "source_repo": repo_url,
                            "source_branch": branch,
                            "package_version": detected,
                            "venv_path": self._venv_path,
                            "install_dir": self._base_dir,
                            "installed_at": utcnow(),
                        }
                        self._ready_pending_version(pending_version, meta)
                        store.set_active_engine_version("lmdeploy", version_name)
                        try:
                            from backend.engines.scan.scanner import scan_engine_version

                            scan_engine_version(store, "lmdeploy", meta)
                        except Exception as scan_e:
                            logger.warning(
                                "LMDeploy param scan after source install: %s", scan_e
                            )
                        mark_swap_config_stale()
                    except Exception as exc:
                        logger.debug(
                            f"Failed to persist LMDeploy engine metadata (source): {exc}"
                        )
                    await self._finish_operation(True, f"Installed from {branch}")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(
                        pending_version,
                        str(exc),
                        {
                            "source_repo": repo_url,
                            "source_branch": branch,
                            "install_type": "source",
                        },
                    )
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))
                finally:
                    from backend.engines.build_workspace import release_held

                    release_held(workspace)

            self._create_task(_runner())
            return self._started_response(
                "LMDeploy install from source started",
                repo=repo_url,
                branch=branch,
            )

    async def retry_existing_install(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Retry a failed LMDeploy install using its existing directory."""
        version_entry = version_entry or {}
        version_name = str(version_entry.get("version") or "").strip()
        install_dir = str(version_entry.get("install_dir") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        if not install_dir and venv_path:
            install_dir = os.path.dirname(os.path.abspath(venv_path))
        if not version_name or not install_dir:
            raise ValueError("This LMDeploy version does not have enough metadata to retry")
        kind = str(
            version_entry.get("install_type") or version_entry.get("type") or ""
        ).strip().lower()
        if kind in {"source", "fork", "patched", "local"}:
            repo = str(version_entry.get("source_repo") or "").strip() or (
                "https://github.com/InternLM/lmdeploy.git"
            )
            branch = str(
                version_entry.get("source_branch")
                or version_entry.get("source_ref")
                or "main"
            ).strip()
            return await self.install_from_source(
                repo, branch, reuse_dir=install_dir, existing_version=version_name
            )
        pip_version = version_entry.get("package_version")
        return await self.install_release(
            version=str(pip_version) if pip_version else None,
            force_reinstall=True,
            reuse_dir=install_dir,
            existing_version=version_name,
        )

    async def sync_source_version(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Pull and reinstall an existing branch-based LMDeploy source install."""
        version_entry = version_entry or {}
        branch = str(version_entry.get("source_branch") or "").strip()
        version_name = str(version_entry.get("version") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        kind = str(
            version_entry.get("install_type") or version_entry.get("type") or ""
        ).strip().lower()
        if kind not in {"source", "fork"}:
            raise RuntimeError("Only LMDeploy source installs can be synced")
        if not branch:
            raise RuntimeError("LMDeploy source install is missing source_branch")
        if not version_name or not venv_path:
            raise RuntimeError("LMDeploy source install metadata is incomplete")

        async with self._lock:
            if self._operation:
                raise RuntimeError("Another LMDeploy operation is already running")

            self._venv_path = os.path.abspath(venv_path)
            self._base_dir = os.path.dirname(self._venv_path)
            self._ensure_directories()
            clone_dir = os.path.join(self._base_dir, "source")

            await self._start_operation(
                "sync_source",
                {"version": version_name, "branch": branch, "sync": True},
            )

            async def _runner():
                try:
                    self._ensure_venv()
                    await self._sync_git_checkout(clone_dir, branch)
                    code = await self._run_pip(
                        ["install", "-v", "-e", "."],
                        "sync_source",
                        cwd=clone_dir,
                        append=True,
                    )
                    if code != 0:
                        raise RuntimeError(
                            f"pip install -e -v . failed with code {code}"
                        )
                    detected = self._detect_installed_version()
                    self._update_installed_state(True, detected)
                    try:
                        store = get_store()
                        updated = store.update_engine_version(
                            "lmdeploy",
                            version_name,
                            {
                                "source_commit": await self._git_head(clone_dir),
                                "source_branch": branch,
                                "source_repo": version_entry.get("source_repo"),
                                "venv_path": self._venv_path,
                                "updated_at": utcnow(),
                            },
                        )
                        if updated:
                            try:
                                from backend.engines.scan.scanner import (
                                    scan_engine_version,
                                )

                                scan_engine_version(store, "lmdeploy", updated)
                            except Exception as scan_e:
                                logger.warning(
                                    "LMDeploy param scan after source sync: %s",
                                    scan_e,
                                )
                        mark_swap_config_stale()
                    except Exception as exc:
                        logger.debug(
                            f"Failed to update LMDeploy metadata after sync: {exc}"
                        )
                    await self._finish_operation(True, f"Synced from {branch}")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response(
                "LMDeploy source sync started",
                version=version_name,
                branch=branch,
            )

    async def remove(self, retire_references: bool = False) -> Dict[str, Any]:
        """Remove LMDeploy from its venv and clean up state."""
        from backend.utils.fs_ops import release_launch_hold, robust_rmtree

        async with self._lock:
            if self._operation:
                raise RuntimeError("Another LMDeploy operation is already running")
            from backend.data_store import get_store

            store = get_store()
            active = store.get_active_engine_version("lmdeploy")
            venv_path = active.get("venv_path") if active else self._venv_path
            if venv_path and os.path.exists(venv_path):
                release_launch_hold(venv_path, retire_references=retire_references)
            await self._start_operation("remove")
            args = ["uninstall", "-y", "lmdeploy"]

            async def _runner():
                try:
                    python_exists = os.path.exists(self._venv_python())
                    if python_exists:
                        code = await self._run_pip(args, "remove", ensure_venv=False)
                        if code != 0:
                            raise RuntimeError(f"pip exited with status {code}")
                    if venv_path and os.path.exists(venv_path):
                        robust_rmtree(venv_path, retire_references=retire_references)
                    if active and active.get("version"):
                        try:
                            store.delete_engine_version("lmdeploy", active["version"])
                        except Exception as exc:  # pragma: no cover
                            logger.debug(
                                f"Failed to delete LMDeploy engine version metadata: {exc}"
                            )
                    self._update_installed_state(False, None)
                    mark_swap_config_stale()
                    await self._finish_operation(True, "LMDeploy removed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response("LMDeploy removal started")

    # --- Introspection --------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        return self._stateful_status(resolve_binary_path=self._resolve_binary_path)


LMDeployManager = LMDeployInstaller
