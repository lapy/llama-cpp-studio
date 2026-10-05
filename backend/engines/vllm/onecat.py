"""Install/manage the 1Cat-vLLM engine (vLLM fork for Tesla V100 / SM70).

Mirrors :mod:`backend.lmdeploy_manager`: each install lives in its own versioned
venv under ``data/1cat-vllm`` and is registered in ``engines.yaml`` under the
``1cat_vllm`` key so the rest of the app (llama-swap config, param catalog, UI)
treats it as a first-class engine.

Unlike LMDeploy (published on PyPI), 1Cat-vLLM ships prebuilt wheels through
GitHub releases. Current releases publish a single ``1cat_vllm`` wheel that
already bundles Flash-V100. Older 0.0.x releases still ship the two-wheel layout
(``flash_attn_v100`` + ``vllm``). Both are pip-installed against the CUDA 12.8
PyTorch index. A source build path is also provided for kernel development;
that path needs a Rust toolchain (``cargo`` / ``rustc``) because the wheel uses
``setuptools-rust``.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import time
from asyncio.subprocess import PIPE, STDOUT
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, List, Optional, Tuple

import httpx

from backend.data_store import get_store
from backend.engines.python.installer import PythonVenvInstaller, unique_version_name, utcnow
from backend.proxy.llama_swap.manager import mark_swap_config_stale
from backend.logging_config import get_logger


ENGINE_ID = "1cat_vllm"
GITHUB_REPO = "1CatAI/1Cat-vLLM"
DEFAULT_SOURCE_REPO = "https://github.com/1CatAI/1Cat-vLLM.git"
# 1Cat-vLLM validates against the CUDA 12.8 PyTorch runtime wheels.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
# Public wheels target SM70 (Tesla V100) only.
DEFAULT_TORCH_CUDA_ARCH_LIST = "7.0"
# setuptools-rust / vllm-rs need cargo+rustc on PATH during ``python -m build``.
RUSTUP_INIT_URL = "https://sh.rustup.rs"
# Current 1Cat-vLLM (vLLM) layout uses requirements/build/cuda.txt; older
# checkouts still ship a flat requirements/build.txt.
_SOURCE_REQUIREMENT_RELS = (
    os.path.join("requirements", "build", "cuda.txt"),
    os.path.join("requirements", "build.txt"),
    os.path.join("requirements", "cuda.txt"),
    os.path.join("requirements", "common.txt"),
)
_SOURCE_BUILD_PY_PACKAGES = (
    "cmake",
    "ninja",
    "build",
    "setuptools-rust",
    "patchelf",
)


logger = get_logger(__name__)

_manager_instance: Optional["OneCatVllmInstaller"] = None


def get_onecat_vllm_manager() -> "OneCatVllmInstaller":
    """Singleton accessor, mirroring the LMDeploy manager pattern."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = OneCatVllmInstaller()
    return _manager_instance


class OneCatVllmInstaller(PythonVenvInstaller):
    """
    Manage 1Cat-vLLM installation into its own venv, similar in spirit to LMDeployManager.

    Responsibilities:
    - Create a dedicated venv under data/1cat-vllm
    - Install 1Cat-vLLM from GitHub release wheels or from a git source build
    - Track install status, version, binary path and venv path
    - Emit progress events so the UI can show logs and status
    """

    MANAGER_NAME = "onecat_vllm"

    OPERATION_DESCRIPTIONS = {
        "install": "Install 1Cat-vLLM",
        "install_source": "Build 1Cat-vLLM from Source",
        "sync_source": "Sync 1Cat-vLLM Source",
        "remove": "Remove 1Cat-vLLM",
    }

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        self.distribution_names = ("1cat-vllm", "1cat_vllm", "vllm")
        super().__init__(
            engine_id=ENGINE_ID,
            label="1Cat-vLLM",
            root_name="1cat-vllm",
            log_name="onecat_vllm_install.log",
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
            state_name="onecat_vllm_manager.json",
        )
        self.MANAGER_NAME = "onecat_vllm"

        # --- State persistence -----------------------------------------------------------

    def _resolve_binary_path(self) -> Optional[str]:
        """1Cat-vLLM is served via ``python -m vllm...``; report the venv python."""
        override = os.getenv("ONECAT_VLLM_BIN")
        if override:
            override_path = os.path.abspath(os.path.expanduser(override))
            if os.path.exists(override_path):
                return override_path
            resolved_override = shutil.which(override)
            if resolved_override:
                return resolved_override

        candidate = self._venv_python()
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return os.path.abspath(candidate)
        return None

    # --- Subprocess helpers and progress broadcasting -------------------------------

    def _managed_rust_root(self) -> str:
        """Persistent rustup prefix: ``<data>/tools/rust`` (sibling of ``1cat-vllm/``)."""
        return os.path.join(os.path.dirname(self._root_dir), "tools", "rust")

    @staticmethod
    def _rust_exe(name: str) -> str:
        return f"{name}.exe" if os.name == "nt" else name

    @staticmethod
    def _bin_dir_has_rust(bin_dir: str) -> bool:
        if not bin_dir or not os.path.isdir(bin_dir):
            return False
        rustc = os.path.join(bin_dir, OneCatVllmInstaller._rust_exe("rustc"))
        cargo = os.path.join(bin_dir, OneCatVllmInstaller._rust_exe("cargo"))
        return all(
            os.path.isfile(path) and os.access(path, os.X_OK)
            for path in (rustc, cargo)
        )

    @staticmethod
    def _dir_writable(path: str) -> bool:
        return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK)

    def _retarget_unwritable_rust_homes(self, env: Dict[str, str]) -> None:
        """Move rustup/cargo off a read-only prefix.

        The image toolchain lives in ``/usr/local/rustup``. rustup has to
        create temp files and may download another channel (for example
        1.95) during ``build_rust``. A root-owned tree fails with EACCES.
        """
        rust_root = self._managed_rust_root()
        current_rustup = env.get("RUSTUP_HOME") or ""
        current_cargo = env.get("CARGO_HOME") or ""
        if current_rustup and os.path.isdir(current_rustup) and not self._dir_writable(
            current_rustup
        ):
            rustup_home = os.path.join(rust_root, "rustup")
            os.makedirs(rustup_home, exist_ok=True)
            env["RUSTUP_HOME"] = rustup_home
        if current_cargo and os.path.isdir(current_cargo) and not self._dir_writable(
            current_cargo
        ):
            cargo_home = os.path.join(rust_root, "cargo")
            os.makedirs(cargo_home, exist_ok=True)
            env["CARGO_HOME"] = cargo_home

    def _apply_rust_bin_dir(self, env: Dict[str, str], bin_dir: str) -> None:
        self._prepend_path(env, bin_dir)
        cargo_home = os.path.dirname(os.path.abspath(bin_dir))
        if self._dir_writable(cargo_home):
            env.setdefault("CARGO_HOME", cargo_home)
        rustup_home = os.path.join(os.path.dirname(cargo_home), "rustup")
        if self._dir_writable(rustup_home):
            env.setdefault("RUSTUP_HOME", rustup_home)

    def _rust_bin_candidates(self, env: Dict[str, str]) -> List[str]:
        home = env.get("HOME") or os.path.expanduser("~")
        cargo_home = env.get("CARGO_HOME") or ""
        return [
            os.path.join(self._managed_rust_root(), "cargo", "bin"),
            os.path.join(cargo_home, "bin") if cargo_home else "",
            os.path.join(home, ".cargo", "bin"),
            "/usr/local/cargo/bin",
            "/root/.cargo/bin",
        ]

    def _discover_rust_bin_dir(self, env: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Return a directory that contains both ``rustc`` and ``cargo``."""
        env = env if env is not None else os.environ
        which_rustc = shutil.which(self._rust_exe("rustc"), path=env.get("PATH", ""))
        if which_rustc:
            which_dir = os.path.dirname(os.path.abspath(which_rustc))
            if self._bin_dir_has_rust(which_dir):
                return which_dir
        for candidate in self._rust_bin_candidates(env):
            if self._bin_dir_has_rust(candidate):
                return candidate
        return None

    @staticmethod
    def _source_requirement_files(clone_dir: str) -> List[str]:
        """Requirement files to pip-install before a no-isolation wheel build."""
        return [
            os.path.join(clone_dir, rel)
            for rel in _SOURCE_REQUIREMENT_RELS
            if os.path.isfile(os.path.join(clone_dir, rel))
        ]

    def _build_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Environment for source builds: force SM70 + CUDA 12.8 + Rust toolchain."""
        env = os.environ.copy()
        self._retarget_unwritable_rust_homes(env)
        cuda_home = (
            os.getenv("ONECAT_VLLM_CUDA_HOME")
            or env.get("CUDA_HOME")
            or "/usr/local/cuda-12.8"
        )
        env["CUDA_HOME"] = cuda_home
        self._prepend_path(env, os.path.join(cuda_home, "bin"))
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{os.path.join(cuda_home, 'lib64')}:{ld}"
        env.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_TORCH_CUDA_ARCH_LIST)
        env.setdefault("MAX_JOBS", os.getenv("ONECAT_VLLM_MAX_JOBS", "12"))
        env.setdefault("NVCC_THREADS", "1")
        rust_bin = self._discover_rust_bin_dir(env)
        if rust_bin:
            self._apply_rust_bin_dir(env, rust_bin)
        else:
            cargo_home = os.path.join(self._managed_rust_root(), "cargo")
            rustup_home = os.path.join(self._managed_rust_root(), "rustup")
            env.setdefault("CARGO_HOME", cargo_home)
            env.setdefault("RUSTUP_HOME", rustup_home)
            self._prepend_path(env, os.path.join(cargo_home, "bin"))
        if extra:
            env.update(extra)
        return env

    async def _ensure_rust(self, env: Dict[str, str]) -> None:
        """Make cargo/rustc available, installing rustup into data/tools/rust if needed."""
        self._retarget_unwritable_rust_homes(env)
        rust_bin = self._discover_rust_bin_dir(env)
        if rust_bin:
            self._apply_rust_bin_dir(env, rust_bin)
            await self._broadcast_log_line(f"Using Rust toolchain at {rust_bin}")
            return

        rust_root = self._managed_rust_root()
        cargo_home = os.path.join(rust_root, "cargo")
        rustup_home = os.path.join(rust_root, "rustup")
        os.makedirs(cargo_home, exist_ok=True)
        os.makedirs(rustup_home, exist_ok=True)
        env["CARGO_HOME"] = cargo_home
        env["RUSTUP_HOME"] = rustup_home
        self._prepend_path(env, os.path.join(cargo_home, "bin"))

        await self._broadcast_log_line(
            "Rust toolchain not found; installing rustup "
            "(required to build the 1Cat-vLLM wheel)."
        )
        installer = os.path.join(rust_root, "rustup-init.sh")
        download_code = await self._run_logged(
            [
                "curl",
                "--proto",
                "=https",
                "--tlsv1.2",
                "-sSfL",
                RUSTUP_INIT_URL,
                "-o",
                installer,
            ],
            "install_rust",
            env=env,
        )
        if download_code != 0 or not os.path.isfile(installer):
            raise RuntimeError(
                "Failed to download rustup. 1Cat-vLLM source builds need "
                "cargo/rustc (https://rustup.rs)."
            )
        os.chmod(installer, 0o755)
        install_code = await self._run_logged(
            [
                "sh",
                installer,
                "-y",
                "--profile",
                "minimal",
                "--default-toolchain",
                "stable",
                "--no-modify-path",
            ],
            "install_rust",
            env=env,
        )
        rust_bin = self._discover_rust_bin_dir(env)
        if install_code != 0 or not rust_bin:
            raise RuntimeError(
                "Rust toolchain is required to build 1Cat-vLLM from source "
                "(cargo/rustc not found). Install rustup from https://rustup.rs "
                "and retry."
            )
        self._apply_rust_bin_dir(env, rust_bin)
        await self._broadcast_log_line(f"Rust toolchain installed at {rust_bin}")

    async def _install_source_build_deps(
        self,
        clone_dir: str,
        build_env: Dict[str, str],
        operation: str,
    ) -> None:
        """Install Python + Rust build deps used by ``python -m build --no-isolation``."""
        # Dependency setup is the short slice. Wheel compilation owns the bar.
        self._active_build_window = (4, 16)
        await self._ensure_rust(build_env)
        await self._run_pip(
            ["install", "--upgrade", "pip", "setuptools", "wheel"],
            operation,
            cwd=clone_dir,
            env=build_env,
        )
        for req_path in self._source_requirement_files(clone_dir):
            rel = os.path.relpath(req_path, clone_dir)
            code = await self._run_pip(
                [
                    "install",
                    "--extra-index-url",
                    TORCH_CUDA_INDEX,
                    "-r",
                    req_path,
                ],
                operation,
                cwd=clone_dir,
                env=build_env,
            )
            if code != 0:
                raise RuntimeError(f"pip install -r {rel} failed ({code})")
        code = await self._run_pip(
            ["install", *_SOURCE_BUILD_PY_PACKAGES],
            operation,
            cwd=clone_dir,
            env=build_env,
        )
        if code != 0:
            raise RuntimeError(
                f"pip install {' '.join(_SOURCE_BUILD_PY_PACKAGES)} failed ({code})"
            )

    async def _build_source_wheels(
        self,
        clone_dir: str,
        dist_dir: str,
        build_env: Dict[str, str],
        *,
        empty_error: str = "Source build produced no wheels",
    ) -> List[str]:
        python_exe = self._venv_python()
        fa_dir = os.path.join(clone_dir, "flash-attention-v100")
        compile_floor = 16
        if os.path.isdir(fa_dir):
            self._active_build_window = (16, 36)
            code = await self._run_logged(
                [
                    python_exe,
                    "-m",
                    "build",
                    "--wheel",
                    "--no-isolation",
                    "--outdir",
                    dist_dir,
                ],
                "build_flash_attn",
                cwd=fa_dir,
                env=build_env,
            )
            if code != 0:
                raise RuntimeError(
                    f"flash-attention-v100 wheel build failed ({code})"
                )
            compile_floor = 36
        self._active_build_window = (compile_floor, 92)
        code = await self._run_logged(
            [
                python_exe,
                "-m",
                "build",
                "--wheel",
                "--no-isolation",
                "--outdir",
                dist_dir,
            ],
            "build_vllm",
            cwd=clone_dir,
            env=build_env,
        )
        if code != 0:
            rust_hint = ""
            if not self._discover_rust_bin_dir(build_env):
                rust_hint = "; Rust toolchain (cargo/rustc) was not found"
            raise RuntimeError(f"vllm wheel build failed ({code}){rust_hint}")
        self._active_build_window = (92, 98)
        wheels = [
            os.path.join(dist_dir, f)
            for f in sorted(os.listdir(dist_dir))
            if f.endswith(".whl")
        ]
        if not wheels:
            raise RuntimeError(empty_error)
        return wheels

    async def _fetch_release(self, version: Optional[str]) -> Dict[str, Any]:
        """Return the GitHub release JSON for a tag (or the latest release)."""
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
            if version:
                tag = version if version.startswith("v") else f"v{version}"
                url = f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{tag}"
            else:
                url = (
                    f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
                )
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def _select_release_wheels(release: Dict[str, Any]) -> Tuple[str, List[str]]:
        """Pick installable wheel URLs from a GitHub release.

        1.x+ releases ship a bundled ``1cat_vllm-*.whl``. Older 0.0.x releases
        ship ``flash_attn_v100-*.whl`` plus ``vllm-*.whl``.
        """
        assets = release.get("assets") or []
        bundled_url = None
        flash_url = None
        vllm_url = None
        for asset in assets:
            name = (asset.get("name") or "").lower()
            if not name.endswith(".whl"):
                continue
            url = asset.get("browser_download_url")
            if name.startswith("1cat_vllm") or name.startswith("1cat-vllm"):
                bundled_url = url
            elif name.startswith("flash_attn_v100"):
                flash_url = url
            elif name.startswith("vllm"):
                vllm_url = url
        if bundled_url:
            wheels = [bundled_url]
        else:
            wheels = [u for u in (flash_url, vllm_url) if u]
        if not wheels or not (bundled_url or vllm_url):
            raise RuntimeError(
                "1Cat-vLLM release does not contain a 1cat_vllm or vllm wheel asset"
            )
        tag = release.get("tag_name") or ""
        return tag, wheels

    # --- Public interface -----------------------------------------------------------

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Install 1Cat-vLLM from prebuilt GitHub release wheels into its own venv."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another 1Cat-vLLM operation is already running")
            await self._start_operation("install")
            dir_name = self._bind_install_dir(reuse_dir, "release")
            pending_version = existing_version or dir_name
            self._register_pending_version(
                pending_version,
                {"install_type": "release", "release_tag": version},
            )

            async def _runner():
                try:
                    release = await self._fetch_release(version)
                    tag, wheels = self._select_release_wheels(release)
                    if not wheels:
                        raise RuntimeError("No installable wheels found in release")
                    self._ensure_venv()
                    await self._run_pip(
                        ["install", "--upgrade", "pip", "setuptools", "wheel"],
                        "install",
                        append=False,
                    )
                    args = [
                        "install",
                        "--prefer-binary",
                        "--no-cache-dir",
                        "--extra-index-url",
                        TORCH_CUDA_INDEX,
                    ]
                    if force_reinstall:
                        args.append("--force-reinstall")
                    args.extend(wheels)
                    code = await self._run_pip(args, "install")
                    if code != 0:
                        raise RuntimeError(f"pip exited with status {code}")
                    detected_version = self._detect_installed_version()
                    self._update_installed_state(True, detected_version)
                    try:
                        store = get_store()
                        release_tag = tag.lstrip("v") if tag else None
                        if existing_version:
                            version_name = pending_version
                        else:
                            base = detected_version or release_tag or pending_version
                            version_name = (
                                pending_version
                                if base == pending_version
                                else unique_version_name(store, ENGINE_ID, base)
                            )
                        meta: Dict[str, Any] = {
                            "version": version_name,
                            "install_type": "release",
                            "release_tag": tag,
                            "package_version": detected_version,
                            "venv_path": self._venv_path,
                            "install_dir": self._base_dir,
                            "installed_at": utcnow(),
                        }
                        self._ready_pending_version(pending_version, meta)
                        store.set_active_engine_version(ENGINE_ID, version_name)
                        try:
                            from backend.engines.scan.scanner import (
                                scan_engine_version,
                            )

                            scan_engine_version(store, ENGINE_ID, meta)
                        except Exception as scan_e:
                            logger.warning(
                                "1Cat-vLLM param scan after release install: %s",
                                scan_e,
                            )
                        mark_swap_config_stale()
                    except Exception as exc:
                        logger.debug(
                            f"Failed to persist 1Cat-vLLM engine metadata: {exc}"
                        )
                    await self._finish_operation(True, "1Cat-vLLM installed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending_version(
                        pending_version, str(exc), {"install_type": "release"}
                    )
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response("1Cat-vLLM installation started")

    async def install_from_source(
        self,
        repo_url: str = DEFAULT_SOURCE_REPO,
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Build and install 1Cat-vLLM from a git checkout (SM70 / CUDA 12.8 build)."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another 1Cat-vLLM operation is already running")
            await self._start_operation("install_source")
            dir_name = self._bind_install_dir(reuse_dir, "source")
            pending_version = existing_version or dir_name
            from backend.repo_identity import source_build_type_labels_for_engine

            type_labels = source_build_type_labels_for_engine(ENGINE_ID, repo_url)
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
            dist_dir = os.path.join(self._base_dir, "dist")

            async def _runner():
                workspace = None
                source_checkout = clone_dir
                try:
                    self._ensure_venv()
                    build_env = self._build_env()
                    os.makedirs(dist_dir, exist_ok=True)
                    reuse = bool(existing_version or reuse_dir)
                    if reuse:
                        if os.path.exists(source_checkout):
                            shutil.rmtree(source_checkout)
                        os.makedirs(source_checkout, exist_ok=True)
                        clone_code = await self._run_logged(
                            [
                                "git",
                                "clone",
                                "--depth",
                                "1",
                                "--branch",
                                branch,
                                repo_url,
                                source_checkout,
                            ],
                            "clone",
                            append=False,
                        )
                        if clone_code != 0:
                            raise RuntimeError(f"git clone failed with code {clone_code}")
                    else:
                        from backend.engines.build_workspace import (
                            BuildWorkspace,
                            ccache_environment,
                        )

                        workspace = BuildWorkspace.open(ENGINE_ID, repo_url, {"kind": "source"})
                        await asyncio.to_thread(workspace.acquire)
                        source_checkout = workspace.checkout_dir
                        await asyncio.to_thread(workspace.sync_git, repo_url, branch)
                        build_env.update(
                            ccache_environment(workspace.path, launchers=True, cuda=True)
                        )

                    await self._install_source_build_deps(
                        source_checkout, build_env, "install_source"
                    )
                    wheels = await self._build_source_wheels(
                        source_checkout, dist_dir, build_env
                    )
                    code = await self._run_pip(
                        [
                            "install",
                            "--prefer-binary",
                            "--no-cache-dir",
                            "--extra-index-url",
                            TORCH_CUDA_INDEX,
                            *wheels,
                        ],
                        "install_source",
                        cwd=source_checkout,
                        env=build_env,
                    )
                    if code != 0:
                        raise RuntimeError(f"pip install of built wheels failed ({code})")
                    if workspace is not None:
                        await asyncio.to_thread(
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
                                store, ENGINE_ID, f"{base_version}-{utcnow()}"
                            )
                        from backend.repo_identity import (
                            source_build_type_labels_for_engine,
                        )

                        type_labels = source_build_type_labels_for_engine(
                            ENGINE_ID, repo_url
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
                        store.set_active_engine_version(ENGINE_ID, version_name)
                        try:
                            from backend.engines.scan.scanner import (
                                scan_engine_version,
                            )

                            scan_engine_version(store, ENGINE_ID, meta)
                        except Exception as scan_e:
                            logger.warning(
                                "1Cat-vLLM param scan after source install: %s",
                                scan_e,
                            )
                        mark_swap_config_stale()
                    except Exception as exc:
                        logger.debug(
                            f"Failed to persist 1Cat-vLLM engine metadata (source): {exc}"
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
                "1Cat-vLLM install from source started",
                repo=repo_url,
                branch=branch,
            )

    async def retry_existing_install(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Retry a failed 1Cat-vLLM install using its existing directory."""
        version_entry = version_entry or {}
        version_name = str(version_entry.get("version") or "").strip()
        install_dir = str(version_entry.get("install_dir") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        if not install_dir and venv_path:
            install_dir = os.path.dirname(os.path.abspath(venv_path))
        if not version_name or not install_dir:
            raise ValueError(
                "This 1Cat-vLLM version does not have enough metadata to retry"
            )
        kind = str(
            version_entry.get("install_type") or version_entry.get("type") or ""
        ).strip().lower()
        if kind in {"source", "fork", "patched", "local"}:
            repo = str(version_entry.get("source_repo") or "").strip() or DEFAULT_SOURCE_REPO
            branch = str(
                version_entry.get("source_branch")
                or version_entry.get("source_ref")
                or "main"
            ).strip()
            return await self.install_from_source(
                repo, branch, reuse_dir=install_dir, existing_version=version_name
            )
        tag = version_entry.get("release_tag") or version_entry.get("package_version")
        return await self.install_release(
            version=str(tag) if tag else None,
            force_reinstall=True,
            reuse_dir=install_dir,
            existing_version=version_name,
        )

    async def sync_source_version(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Pull and rebuild an existing branch-based 1Cat-vLLM source install."""
        version_entry = version_entry or {}
        branch = str(version_entry.get("source_branch") or "").strip()
        version_name = str(version_entry.get("version") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        kind = str(
            version_entry.get("install_type") or version_entry.get("type") or ""
        ).strip().lower()
        if kind not in {"source", "fork"}:
            raise RuntimeError("Only 1Cat-vLLM source installs can be synced")
        if not branch:
            raise RuntimeError("1Cat-vLLM source install is missing source_branch")
        if not version_name or not venv_path:
            raise RuntimeError("1Cat-vLLM source install metadata is incomplete")

        async with self._lock:
            if self._operation:
                raise RuntimeError("Another 1Cat-vLLM operation is already running")

            self._venv_path = os.path.abspath(venv_path)
            self._base_dir = os.path.dirname(self._venv_path)
            self._ensure_directories()
            clone_dir = os.path.join(self._base_dir, "source")
            dist_dir = os.path.join(self._base_dir, "dist")

            await self._start_operation(
                "sync_source",
                {"version": version_name, "branch": branch, "sync": True},
            )

            async def _runner():
                try:
                    self._ensure_venv()
                    build_env = self._build_env()
                    os.makedirs(dist_dir, exist_ok=True)
                    await self._sync_git_checkout(clone_dir, branch)

                    await self._install_source_build_deps(
                        clone_dir, build_env, "sync_source"
                    )
                    for filename in os.listdir(dist_dir):
                        if filename.endswith(".whl"):
                            try:
                                os.remove(os.path.join(dist_dir, filename))
                            except OSError:
                                pass
                    wheels = await self._build_source_wheels(
                        clone_dir,
                        dist_dir,
                        build_env,
                        empty_error="Source sync produced no wheels",
                    )
                    code = await self._run_pip(
                        [
                            "install",
                            "--prefer-binary",
                            "--no-cache-dir",
                            "--extra-index-url",
                            TORCH_CUDA_INDEX,
                            *wheels,
                        ],
                        "sync_source",
                        cwd=clone_dir,
                        env=build_env,
                    )
                    if code != 0:
                        raise RuntimeError(f"pip install of built wheels failed ({code})")

                    detected = self._detect_installed_version()
                    self._update_installed_state(True, detected)
                    try:
                        store = get_store()
                        updated = store.update_engine_version(
                            ENGINE_ID,
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

                                scan_engine_version(store, ENGINE_ID, updated)
                            except Exception as scan_e:
                                logger.warning(
                                    "1Cat-vLLM param scan after source sync: %s",
                                    scan_e,
                                )
                        mark_swap_config_stale()
                    except Exception as exc:
                        logger.debug(
                            f"Failed to update 1Cat-vLLM metadata after sync: {exc}"
                        )
                    await self._finish_operation(True, f"Synced from {branch}")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response(
                "1Cat-vLLM source sync started",
                version=version_name,
                branch=branch,
            )

    async def remove(self) -> Dict[str, Any]:
        """Remove 1Cat-vLLM from its venv and clean up state."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another 1Cat-vLLM operation is already running")
            await self._start_operation("remove")
            args = ["uninstall", "-y", "1cat-vllm", "1cat_vllm", "vllm", "flash_attn_v100"]

            async def _runner():
                try:
                    store = get_store()
                    active = store.get_active_engine_version(ENGINE_ID)
                    venv_path = active.get("venv_path") if active else self._venv_path
                    if venv_path:
                        self._venv_path = venv_path

                    python_exists = os.path.exists(self._venv_python())
                    if python_exists:
                        code = await self._run_pip(
                            args, "remove", ensure_venv=False, append=False
                        )
                        if code != 0:
                            raise RuntimeError(f"pip exited with status {code}")
                    if venv_path:
                        shutil.rmtree(venv_path, ignore_errors=True)
                    if active and active.get("version"):
                        try:
                            store.delete_engine_version(
                                ENGINE_ID, active["version"]
                            )
                        except Exception as exc:  # pragma: no cover
                            logger.debug(
                                f"Failed to delete 1Cat-vLLM engine version metadata: {exc}"
                            )
                    self._update_installed_state(False, None)
                    mark_swap_config_stale()
                    await self._finish_operation(True, "1Cat-vLLM removed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response("1Cat-vLLM removal started")

    # --- Introspection --------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        active = get_store().get_active_engine_version(ENGINE_ID)
        return self._stateful_status(
            resolve_binary_path=self._resolve_binary_path,
            extra={"release_tag": active.get("release_tag") if active else None},
        )


OneCatVllmManager = OneCatVllmInstaller
