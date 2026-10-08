"""Vanilla vLLM installer. Sibling of SglangInstaller, not a subclass of it."""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Any, Dict, Optional

from backend.data_store import get_store
from backend.engines.python.installer import PythonVenvInstaller, unique_version_name, utcnow
from backend.proxy.llama_swap.manager import mark_swap_config_stale
from backend.logging_config import get_logger


logger = get_logger(__name__)

VLLM_REPOSITORY = "https://github.com/vllm-project/vllm.git"

_manager_instance: Optional["VllmInstaller"] = None


def get_vllm_manager() -> "VllmInstaller":
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = VllmInstaller()
    return _manager_instance


class VllmInstaller(PythonVenvInstaller):
    """Versioned installer for upstream vLLM."""

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        self.distribution_names = ("vllm",)
        super().__init__(
            engine_id="vllm",
            label="vLLM",
            root_name="vllm",
            log_name="vllm_install.log",
            log_path=log_path,
            base_dir=base_dir,
        )

    def _ensure_venv(self) -> None:
        running = sys.version_info[:2]
        if running < (3, 10) or running > (3, 13):
            raise RuntimeError(
                f"{self.label} requires Python 3.10–3.13; "
                f"Studio is running Python {sys.version_info.major}.{sys.version_info.minor}"
            )
        super()._ensure_venv()

    def _cuda_build_env(self) -> tuple[Dict[str, str], Dict[str, Any]]:
        from backend.cuda_installer import get_cuda_installer

        cuda_installer = get_cuda_installer()
        cuda = cuda_installer.status()
        build_env = dict(os.environ)
        build_env.update(cuda_installer.get_cuda_env())
        build_env["VIRTUAL_ENV"] = self._venv_path
        self._prepend_path(build_env, os.path.dirname(self._venv_python()))
        runtime_meta = {
            "cuda_version": cuda.get("version"),
            "cuda_path": cuda.get("cuda_path"),
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        }
        return build_env, runtime_meta

    async def _clone(self, repo_url: str, branch: str, clone_dir: str) -> None:
        import shutil

        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        await self._update_progress_task(3, "Cloning source")
        code = await self._run_logged(
            ["git", "clone", "--depth", "1", "--branch", branch, repo_url, clone_dir],
            "clone",
            append=True,
        )
        if code != 0:
            raise RuntimeError(f"git clone failed with status {code}")
        await self._update_progress_task(8, "Source checkout ready")

    async def _install_source_checkout(
        self, clone_dir: str, build_env: Optional[Dict[str, str]] = None
    ) -> None:
        code = await self._run_pip(
            ["install", "--upgrade", "pip"], "install_source", append=True
        )
        if code != 0:
            raise RuntimeError(f"pip upgrade failed with status {code}")
        if not os.path.isdir(clone_dir):
            raise RuntimeError(f"{self.label} source package directory is unavailable")
        code = await self._run_pip(
            ["install", "-v", "-e", clone_dir],
            "install_source",
            cwd=clone_dir,
            env=build_env,
        )
        if code != 0:
            raise RuntimeError(f"{self.label} source install failed with status {code}")

    async def _finalize_install(
        self,
        pending_version: str,
        meta: Dict[str, Any],
        success_message: str,
    ) -> None:
        await self._update_progress_task(98, f"Validating {self.label} installation")
        detected = self._detect_installed_version()
        if not detected:
            raise RuntimeError(f"Installed {self.label} package could not be imported")
        store = get_store()
        final_base = str(meta.get("version") or detected or pending_version)
        final_version = (
            pending_version
            if meta.get("reuse_existing")
            else unique_version_name(store, self.engine_id, final_base)
        )
        meta = {
            **meta,
            "version": final_version,
            "package_version": detected,
            "installed_at": utcnow(),
            "venv_path": self._venv_path,
            "install_dir": self._base_dir,
        }
        meta.pop("reuse_existing", None)
        final_version = self._ready_pending_version(pending_version, meta)
        from backend.engines.activation import activate_engine_version

        await activate_engine_version(
            self.engine_id,
            final_version,
            covered_by=self.progress_task_id,
            row=meta,
        )
        await self._finish_operation(True, success_message)

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._lock:
            self._raise_if_busy()
            await self._start_operation("install")
            dirname = self._bind_install_dir(reuse_dir, "pip")
            pending = existing_version or dirname
            extra = {"install_type": "pip", "package_version": version}
            self._register_pending_version(pending, extra)

            async def runner() -> None:
                try:
                    args = ["install", "--upgrade"]
                    if force_reinstall:
                        args.append("--force-reinstall")
                    args.append(f"vllm=={version}" if version else "vllm")
                    code = await self._run_pip(args, "install", append=False)
                    if code != 0:
                        raise RuntimeError(f"pip install failed with status {code}")
                    await self._finalize_install(
                        pending,
                        {
                            **extra,
                            "version": existing_version or pending,
                            "reuse_existing": bool(existing_version),
                        },
                        f"{self.label} installed",
                    )
                except asyncio.CancelledError:
                    self._fail_pending_version(
                        pending, "Operation cancelled by user", extra, cancelled=True
                    )
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(pending, str(exc), extra)
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(f"{self.label} installation started")

    async def install_from_source(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        repo_url = str(repo_url or VLLM_REPOSITORY).strip()
        branch = str(branch or "main").strip()
        async with self._lock:
            self._raise_if_busy()
            await self._start_operation("install_source")
            dirname = self._bind_install_dir(reuse_dir, "source")
            pending = existing_version or dirname
            from backend.repo_identity import source_build_type_labels_for_engine

            labels = source_build_type_labels_for_engine(self.engine_id, repo_url)
            extra = {
                "type": labels["type"],
                "install_type": labels["install_type"],
                "is_fork": labels["is_fork"],
                "source_repo": repo_url,
                "source_branch": branch,
            }
            self._register_pending_version(pending, extra)
            clone_dir = os.path.join(self._base_dir, "source")

            async def runner() -> None:
                runtime_meta: Dict[str, Any] = {}
                workspace = None
                try:
                    build_env, runtime_meta = self._cuda_build_env()
                    source_checkout = clone_dir
                    reuse = bool(existing_version or reuse_dir)
                    if reuse:
                        await self._clone(repo_url, branch, source_checkout)
                    else:
                        from backend.engines.build_workspace import BuildWorkspace

                        workspace = BuildWorkspace.open(self.engine_id, repo_url, {"kind": "source"})
                        await asyncio.to_thread(workspace.acquire)
                        source_checkout = workspace.checkout_dir
                        await asyncio.to_thread(workspace.sync_git, repo_url, branch)
                    from backend.engines.build_workspace import (
                        ccache_base_dir,
                        ccache_environment,
                    )

                    build_env.update(
                        ccache_environment(
                            ccache_base_dir(source_checkout),
                            launchers=True,
                            cuda=True,
                        )
                    )
                    await self._install_source_checkout(source_checkout, build_env)
                    if workspace is not None:
                        source_checkout = await asyncio.to_thread(
                            workspace.seal_installed_source,
                            os.path.join(self._base_dir, "source"),
                            self._venv_path,
                        )
                    commit = await self._git_head(source_checkout)
                    detected = self._detect_installed_version()
                    base = f"{detected or branch}-{(commit or '')[:8]}".rstrip("-")
                    await self._finalize_install(
                        pending,
                        {
                            **extra,
                            "version": existing_version or base or pending,
                            "source_commit": commit,
                            "reuse_existing": bool(existing_version),
                            **runtime_meta,
                        },
                        f"{self.label} installed from {branch}",
                    )
                except asyncio.CancelledError:
                    self._fail_pending_version(
                        pending,
                        "Operation cancelled by user",
                        {**extra, **runtime_meta},
                        cancelled=True,
                    )
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(pending, str(exc), {**extra, **runtime_meta})
                    await self._finish_operation(False, str(exc))
                finally:
                    from backend.engines.build_workspace import release_held

                    release_held(workspace)

            self._create_task(runner())
            return self._started_response(
                f"{self.label} source installation started",
                repo=repo_url,
                branch=branch,
            )

    async def retry_existing_install(self, row: Dict[str, Any]) -> Dict[str, Any]:
        install_dir = str(row.get("install_dir") or "").strip()
        version = str(row.get("version") or "").strip()
        if not install_dir or not version:
            raise ValueError("Version is missing its install directory or name")
        kind = str(row.get("install_type") or row.get("type") or "").lower()
        if kind in {"source", "fork", "patched", "local"}:
            return await self.install_from_source(
                row.get("source_repo") or VLLM_REPOSITORY,
                row.get("source_branch") or "main",
                reuse_dir=install_dir,
                existing_version=version,
            )
        return await self.install_release(
            row.get("package_version"),
            force_reinstall=True,
            reuse_dir=install_dir,
            existing_version=version,
        )

    async def sync_source_version(self, row: Dict[str, Any]) -> Dict[str, Any]:
        install_dir = str(row.get("install_dir") or "").strip()
        version = str(row.get("version") or "").strip()
        branch = str(row.get("source_branch") or "").strip()
        if not install_dir or not version or not branch:
            raise ValueError("Source version is missing install directory, name, or branch")
        async with self._lock:
            self._raise_if_busy()
            self._base_dir = os.path.abspath(install_dir)
            self._venv_path = os.path.join(self._base_dir, "venv")
            clone_dir = os.path.join(self._base_dir, "source")
            await self._start_operation("sync_source")
            self._register_pending_version(version, dict(row))

            async def runner() -> None:
                runtime_meta: Dict[str, Any] = {}
                try:
                    build_env, runtime_meta = self._cuda_build_env()
                    from backend.engines.build_workspace import (
                        ccache_base_dir,
                        ccache_environment,
                    )

                    build_env.update(
                        ccache_environment(
                            ccache_base_dir(clone_dir),
                            launchers=True,
                            cuda=True,
                        )
                    )
                    await self._sync_git_checkout(clone_dir, branch)
                    await self._install_source_checkout(clone_dir, build_env)
                    commit = await self._git_head(clone_dir)
                    await self._finalize_install(
                        version,
                        {
                            **row,
                            "version": version,
                            "source_commit": commit,
                            "reuse_existing": True,
                            **runtime_meta,
                        },
                        f"{self.label} source synchronized",
                    )
                except asyncio.CancelledError:
                    self._fail_pending_version(
                        version,
                        "Operation cancelled by user",
                        {**row, **runtime_meta},
                        cancelled=True,
                    )
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(version, str(exc), {**row, **runtime_meta})
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(
                f"{self.label} source sync started", version=version
            )

    async def remove(self, retire_references: bool = False) -> Dict[str, Any]:
        from backend.utils.fs_ops import release_launch_hold, robust_rmtree

        async with self._lock:
            self._raise_if_busy()
            active = get_store().get_active_engine_version(self.engine_id)
            raw_install = str((active or {}).get("install_dir") or "")
            install_dir = os.path.realpath(raw_install) if raw_install else ""
            root_dir = os.path.realpath(self._root_dir)
            if (
                install_dir
                and os.path.dirname(install_dir) == root_dir
                and os.path.exists(install_dir)
            ):
                release_launch_hold(install_dir, retire_references=retire_references)
            await self._start_operation("remove")

            async def runner() -> None:
                try:
                    if (
                        install_dir
                        and os.path.dirname(install_dir) == root_dir
                        and os.path.exists(install_dir)
                    ):
                        robust_rmtree(install_dir, retire_references=retire_references)
                    if active and active.get("version"):
                        get_store().delete_engine_version(
                            self.engine_id, str(active["version"])
                        )
                    mark_swap_config_stale()
                    await self._finish_operation(True, f"{self.label} removed")
                except Exception as exc:
                    self._last_error = str(exc)
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(f"{self.label} removal started")

    def status(self) -> Dict[str, Any]:
        active = get_store().get_active_engine_version(self.engine_id)
        venv_path = str((active or {}).get("venv_path") or self._venv_path)
        version = self._detect_installed_version(venv_path)
        python_bin = os.path.join(
            venv_path,
            "Scripts" if os.name == "nt" else "bin",
            "python.exe" if os.name == "nt" else "python",
        )
        return {
            "engine": self.engine_id,
            "installed": bool(version and os.path.isfile(python_bin)),
            "version": version,
            "binary_path": python_bin if os.path.isfile(python_bin) else None,
            "venv_path": venv_path,
            "installed_at": (active or {}).get("installed_at"),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "progress_task_id": self._progress_task_id,
            "last_error": self._last_error,
            "log_path": self._log_path,
            "install_type": (active or {}).get("install_type"),
            "source_repo": (active or {}).get("source_repo"),
            "source_branch": (active or {}).get("source_branch"),
            "source_commit": (active or {}).get("source_commit"),
            "cuda_version": (active or {}).get("cuda_version"),
            "cuda_path": (active or {}).get("cuda_path"),
            "python_version": (active or {}).get("python_version"),
        }


VllmManager = VllmInstaller
