"""Install/manage the 1Cat-vLLM engine (vLLM fork for Tesla V100 / SM70).

Mirrors :mod:`backend.lmdeploy_manager`: each install lives in its own versioned
venv under ``data/1cat-vllm`` and is registered in ``engines.yaml`` under the
``1cat_vllm`` key so the rest of the app (llama-swap config, param catalog, UI)
treats it as a first-class engine.

Unlike LMDeploy (published on PyPI), 1Cat-vLLM ships prebuilt wheels through
GitHub releases. The recommended install path therefore downloads the two release
wheels (``flash_attn_v100`` + ``vllm``) and pip-installs them against the CUDA 12.8
PyTorch index. A source build path is also provided for kernel development.
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from asyncio.subprocess import PIPE, STDOUT
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, List, Optional, Tuple

import httpx

from backend.cancellable_operation_manager import CancellableOperationManager
from backend.logging_config import get_logger
from backend.progress_manager import get_progress_manager
from backend.data_store import get_store
from backend.llama_swap_manager import mark_swap_config_stale


ENGINE_ID = "1cat_vllm"
GITHUB_REPO = "1CatAI/1Cat-vLLM"
DEFAULT_SOURCE_REPO = "https://github.com/1CatAI/1Cat-vLLM.git"
# 1Cat-vLLM validates against the CUDA 12.8 PyTorch runtime wheels.
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
# Public wheels target SM70 (Tesla V100) only.
DEFAULT_TORCH_CUDA_ARCH_LIST = "7.0"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


logger = get_logger(__name__)

_manager_instance: Optional["OneCatVllmManager"] = None


def get_onecat_vllm_manager() -> "OneCatVllmManager":
    """Singleton accessor, mirroring the LMDeploy manager pattern."""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = OneCatVllmManager()
    return _manager_instance


def _unique_version_name(store, base: str) -> str:
    """Ensure engines.yaml can hold multiple installs without duplicate version ids."""
    existing = {
        str(v.get("version"))
        for v in store.get_engine_versions(ENGINE_ID)
        if v.get("version")
    }
    if base not in existing:
        return base
    t = int(time.time())
    candidate = f"{base}-{t}"
    for n in range(1, 10000):
        if candidate not in existing:
            return candidate
        candidate = f"{base}-{t}-{n}"
    return f"{base}-{t}-x"


class OneCatVllmManager(CancellableOperationManager):
    """
    Manage 1Cat-vLLM installation into its own venv, similar in spirit to LMDeployManager.

    Responsibilities:
    - Create a dedicated venv under data/1cat-vllm
    - Install 1Cat-vLLM from GitHub release wheels or from a git source build
    - Track install status, version, binary path and venv path
    - Emit progress events so the UI can show logs and status
    """

    MANAGER_NAME = "onecat_vllm"
    LEGACY_STATUS_EVENT = "onecat_vllm_install_status"
    LEGACY_LOG_EVENT = "onecat_vllm_install_log"

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
        super().__init__()

        data_root = os.path.abspath("data")
        base_path = base_dir or os.path.join(data_root, "1cat-vllm")
        # Root directory under which versioned 1Cat-vLLM environments are created.
        self._root_dir = os.path.abspath(base_path)
        # Default venv path (used only as a fallback when no versioned install exists).
        self._base_dir = self._root_dir
        self._venv_path = os.path.join(self._base_dir, "venv")
        log_path = log_path or os.path.join(
            data_root, "logs", "onecat_vllm_install.log"
        )
        state_path = state_path or os.path.join(
            data_root, "config", "onecat_vllm_manager.json"
        )
        self._log_path = os.path.abspath(log_path)
        self._state_path = os.path.abspath(state_path)
        self._ensure_directories()

    # --- Venv and filesystem helpers -------------------------------------------------

    def _ensure_directories(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)

    def _venv_bin(self, executable: str) -> str:
        if os.name == "nt":
            exe = (
                executable
                if executable.lower().endswith(".exe")
                else f"{executable}.exe"
            )
            return os.path.join(self._venv_path, "Scripts", exe)
        return os.path.join(self._venv_path, "bin", executable)

    def _venv_python(self) -> str:
        return self._venv_bin("python")

    def _prepare_versioned_paths(self, label: str = "") -> str:
        """
        Prepare a new versioned install directory under the 1Cat-vLLM root.

        Returns:
          A version directory name component (e.g. '20250309-123456-release').
        """
        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        suffix = f"-{label}" if label else ""
        version_dir = f"{ts}{suffix}"
        self._base_dir = os.path.join(self._root_dir, version_dir)
        self._venv_path = os.path.join(self._base_dir, "venv")
        self._ensure_directories()
        return version_dir

    def _ensure_venv(self) -> None:
        python_path = self._venv_python()
        if os.path.exists(python_path):
            return
        os.makedirs(self._base_dir, exist_ok=True)
        try:
            # 1Cat-vLLM wheels are built for Python 3.12; create the venv with a
            # matching interpreter when the host runs a different default Python.
            subprocess.run([sys.executable, "-m", "venv", self._venv_path], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Failed to create 1Cat-vLLM virtual environment: {exc}"
            ) from exc

    # --- State persistence -----------------------------------------------------------

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self._state_path):
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Failed to load 1Cat-vLLM manager state: {exc}")
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        tmp_path = f"{self._state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        os.replace(tmp_path, self._state_path)

    def _detect_installed_version(self) -> Optional[str]:
        python_exe = self._venv_python()
        if not os.path.exists(python_exe):
            return None
        # The fork installs as the ``vllm`` distribution.
        script = (
            "import importlib, sys\n"
            "try:\n"
            "    from importlib import metadata\n"
            "except ImportError:\n"
            "    import importlib_metadata as metadata\n"
            "try:\n"
            "    print(metadata.version('vllm'))\n"
            "except metadata.PackageNotFoundError:\n"
            "    sys.exit(1)\n"
        )
        try:
            output = subprocess.check_output(
                [python_exe, "-c", script], text=True
            ).strip()
            return output or None
        except subprocess.CalledProcessError:
            return None
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Unable to determine 1Cat-vLLM version: {exc}")
            return None

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

    def _update_installed_state(self, installed: bool, version: Optional[str]) -> None:
        state = self._load_state()
        if installed:
            state["installed_at"] = _utcnow()
            state["installed_version"] = version
            state["venv_path"] = self._venv_path
        else:
            state["installed_version"] = None
            state["installed_at"] = None
            state["removed_at"] = _utcnow()
            state["venv_path"] = self._venv_path
        self._save_state(state)

    def _refresh_state_from_environment(self) -> None:
        state = self._load_state()
        version = self._detect_installed_version()
        state["installed_version"] = version
        if version is None:
            state["removed_at"] = _utcnow()
        state["venv_path"] = self._venv_path
        self._save_state(state)

    # --- Subprocess helpers and progress broadcasting -------------------------------

    def _build_env(self, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Environment for source builds: force SM70 + CUDA 12.8 toolchain."""
        env = os.environ.copy()
        cuda_home = (
            os.getenv("ONECAT_VLLM_CUDA_HOME")
            or env.get("CUDA_HOME")
            or "/usr/local/cuda-12.8"
        )
        env["CUDA_HOME"] = cuda_home
        env["PATH"] = f"{os.path.join(cuda_home, 'bin')}:{env.get('PATH', '')}"
        ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{os.path.join(cuda_home, 'lib64')}:{ld}"
        env.setdefault("TORCH_CUDA_ARCH_LIST", DEFAULT_TORCH_CUDA_ARCH_LIST)
        env.setdefault("MAX_JOBS", os.getenv("ONECAT_VLLM_MAX_JOBS", "12"))
        env.setdefault("NVCC_THREADS", "1")
        if extra:
            env.update(extra)
        return env

    async def _run_logged(
        self,
        argv: List[str],
        operation: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        """Run a command, streaming combined output to the log + SSE."""
        mode = "a" if append else "w"
        header = f"[{_utcnow()}] 1Cat-vLLM {operation}: {' '.join(argv)}\n"
        with open(self._log_path, mode, encoding="utf-8") as log_file:
            log_file.write(header)
        await self._broadcast_log_line(f"$ {' '.join(argv)}")

        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=PIPE,
            stderr=STDOUT,
            cwd=cwd,
            env=env,
        )
        self._active_process = process

        async def _stream_output() -> None:
            if process.stdout is None:
                return
            with open(self._log_path, "a", encoding="utf-8", buffering=1) as log_file:
                while True:
                    chunk = await process.stdout.readline()
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    log_file.write(text)
                    await self._broadcast_log_line(text.rstrip("\n"))

        await asyncio.gather(process.wait(), _stream_output())
        self._clear_active_process()
        return process.returncode or 0

    async def _run_pip(
        self,
        args: List[str],
        operation: str,
        *,
        ensure_venv: bool = True,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        if ensure_venv:
            self._ensure_venv()
        python_exe = self._venv_python()
        if not os.path.exists(python_exe):
            raise RuntimeError(
                "1Cat-vLLM virtual environment is missing; cannot run pip."
            )
        return await self._run_logged(
            [python_exe, "-m", "pip", *args],
            operation,
            cwd=cwd,
            env=env,
            append=append,
        )

    async def _sync_git_checkout(self, clone_dir: str, branch: str) -> None:
        branch = str(branch or "").strip()
        if not branch:
            raise RuntimeError("A source branch is required for sync")
        if not os.path.isdir(os.path.join(clone_dir, ".git")):
            raise RuntimeError(f"Source checkout not found: {clone_dir}")

        code = await self._run_logged(
            ["git", "fetch", "--prune", "origin", branch],
            "sync_source",
            cwd=clone_dir,
            append=False,
        )
        if code != 0:
            raise RuntimeError(f"git fetch failed with code {code}")

        code = await self._run_logged(
            ["git", "checkout", "-B", branch, "FETCH_HEAD"],
            "sync_source",
            cwd=clone_dir,
        )
        if code != 0:
            await self._broadcast_log_line(
                "Checkout had local conflicts; cleaning untracked source files while keeping build caches."
            )
            clean_code = await self._run_logged(
                [
                    "git",
                    "clean",
                    "-fd",
                    "-e",
                    "build/",
                    "-e",
                    "build",
                    "-e",
                    "dist/",
                    "-e",
                    "dist",
                    "-e",
                    ".cache/",
                    "-e",
                    ".cache",
                ],
                "sync_source",
                cwd=clone_dir,
            )
            if clean_code != 0:
                raise RuntimeError(f"git clean failed with code {clean_code}")
            code = await self._run_logged(
                ["git", "checkout", "-B", branch, "FETCH_HEAD"],
                "sync_source",
                cwd=clone_dir,
            )
            if code != 0:
                raise RuntimeError(f"git checkout failed with code {code}")

        code = await self._run_logged(
            ["git", "reset", "--hard", "FETCH_HEAD"],
            "sync_source",
            cwd=clone_dir,
        )
        if code != 0:
            raise RuntimeError(f"git reset failed with code {code}")

    async def _git_head(self, clone_dir: str) -> Optional[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                stdout=PIPE,
                stderr=STDOUT,
                cwd=clone_dir,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode != 0:
                return None
            return stdout.decode("utf-8", errors="replace").strip() or None
        except Exception as exc:
            logger.debug("Could not read 1Cat-vLLM source HEAD: %s", exc)
            return None

    async def _broadcast_log_line(self, line: str) -> None:
        try:
            await self._append_task_log(line)
            await self._emit_legacy_log(line)
            if self._progress_task_id:
                existing = get_progress_manager().get_task(self._progress_task_id) or {}
                log_count = int((existing.get("metadata") or {}).get("log_count", 0)) + 1
                progress = min(
                    90.0,
                    max(float(existing.get("progress") or 10), 10 + log_count * 2),
                )
                await self._update_progress_task(
                    progress,
                    line,
                    metadata_update={"log_count": log_count},
                )
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Failed to broadcast 1Cat-vLLM log line: {exc}")

    async def _start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        description = self.OPERATION_DESCRIPTIONS.get(operation, "Install 1Cat-vLLM")
        return await self._begin_operation(operation, description, metadata)

    def _on_task_error(self, exc: Exception) -> None:
        logger.error(f"1Cat-vLLM manager task error: {exc}")

    # --- GitHub release resolution --------------------------------------------------

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
        """Pick the flash_attn_v100 + vllm wheel download URLs from a release."""
        assets = release.get("assets") or []
        flash_url = None
        vllm_url = None
        for asset in assets:
            name = (asset.get("name") or "").lower()
            if not name.endswith(".whl"):
                continue
            url = asset.get("browser_download_url")
            if name.startswith("flash_attn_v100"):
                flash_url = url
            elif name.startswith("vllm"):
                vllm_url = url
        wheels = [u for u in (flash_url, vllm_url) if u]
        if not vllm_url:
            raise RuntimeError(
                "1Cat-vLLM release does not contain a vllm wheel asset"
            )
        tag = release.get("tag_name") or ""
        return tag, wheels

    # --- Public interface -----------------------------------------------------------

    async def install_release(
        self, version: Optional[str] = None, force_reinstall: bool = False
    ) -> Dict[str, Any]:
        """Install 1Cat-vLLM from prebuilt GitHub release wheels into its own venv."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another 1Cat-vLLM operation is already running")
            await self._start_operation("install")
            self._prepare_versioned_paths(label="release")

            async def _runner():
                try:
                    release = await self._fetch_release(version)
                    tag, wheels = self._select_release_wheels(release)
                    if not wheels:
                        raise RuntimeError("No installable wheels found in release")
                    self._ensure_venv()
                    # Keep build tooling current; wheels are platform-specific.
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
                        base = detected_version or release_tag or f"release-{_utcnow()}"
                        version_name = _unique_version_name(store, base)
                        meta: Dict[str, Any] = {
                            "version": version_name,
                            "install_type": "release",
                            "release_tag": tag,
                            "venv_path": self._venv_path,
                            "installed_at": _utcnow(),
                        }
                        store.add_engine_version(ENGINE_ID, meta)
                        store.set_active_engine_version(ENGINE_ID, version_name)
                        try:
                            from backend.engine_param_scanner import (
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
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response("1Cat-vLLM installation started")

    async def install_from_source(
        self,
        repo_url: str = DEFAULT_SOURCE_REPO,
        branch: str = "main",
    ) -> Dict[str, Any]:
        """Build and install 1Cat-vLLM from a git checkout (SM70 / CUDA 12.8 build)."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another 1Cat-vLLM operation is already running")
            await self._start_operation("install_source")
            self._prepare_versioned_paths(label="source")
            clone_dir = os.path.join(self._base_dir, "source")
            dist_dir = os.path.join(self._base_dir, "dist")

            async def _runner():
                try:
                    self._ensure_venv()
                    build_env = self._build_env()
                    if os.path.exists(clone_dir):
                        shutil.rmtree(clone_dir)
                    os.makedirs(clone_dir, exist_ok=True)
                    os.makedirs(dist_dir, exist_ok=True)

                    clone_code = await self._run_logged(
                        [
                            "git",
                            "clone",
                            "--depth",
                            "1",
                            "--branch",
                            branch,
                            repo_url,
                            clone_dir,
                        ],
                        "clone",
                        append=False,
                    )
                    if clone_code != 0:
                        raise RuntimeError(f"git clone failed with code {clone_code}")

                    await self._run_pip(
                        ["install", "--upgrade", "pip", "setuptools", "wheel"],
                        "install_source",
                    )
                    # Build dependencies (mirrors the 1Cat-vLLM source build docs).
                    for req in ("build.txt", "cuda.txt", "common.txt"):
                        req_path = os.path.join(clone_dir, "requirements", req)
                        if os.path.exists(req_path):
                            code = await self._run_pip(
                                [
                                    "install",
                                    "--extra-index-url",
                                    TORCH_CUDA_INDEX,
                                    "-r",
                                    req_path,
                                ],
                                "install_source",
                                cwd=clone_dir,
                                env=build_env,
                            )
                            if code != 0:
                                raise RuntimeError(
                                    f"pip install -r requirements/{req} failed ({code})"
                                )
                    code = await self._run_pip(
                        ["install", "cmake", "build"],
                        "install_source",
                        cwd=clone_dir,
                        env=build_env,
                    )
                    if code != 0:
                        raise RuntimeError(f"pip install cmake build failed ({code})")

                    python_exe = self._venv_python()
                    # Build the V100 FlashAttention wheel first, then the vllm wheel.
                    fa_dir = os.path.join(clone_dir, "flash-attention-v100")
                    if os.path.isdir(fa_dir):
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
                        raise RuntimeError(f"vllm wheel build failed ({code})")

                    wheels = [
                        os.path.join(dist_dir, f)
                        for f in sorted(os.listdir(dist_dir))
                        if f.endswith(".whl")
                    ]
                    if not wheels:
                        raise RuntimeError("Source build produced no wheels")
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
                        cwd=clone_dir,
                        env=build_env,
                    )
                    if code != 0:
                        raise RuntimeError(f"pip install of built wheels failed ({code})")

                    detected = self._detect_installed_version()
                    self._update_installed_state(True, detected)
                    try:
                        store = get_store()
                        base_version = detected or branch or "source"
                        base = f"{base_version}-{_utcnow()}"
                        version_name = _unique_version_name(store, base)
                        meta: Dict[str, Any] = {
                            "version": version_name,
                            "install_type": "source",
                            "source_repo": repo_url,
                            "source_branch": branch,
                            "venv_path": self._venv_path,
                            "installed_at": _utcnow(),
                        }
                        store.add_engine_version(ENGINE_ID, meta)
                        store.set_active_engine_version(ENGINE_ID, version_name)
                        try:
                            from backend.engine_param_scanner import (
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
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return self._started_response(
                "1Cat-vLLM install from source started",
                repo=repo_url,
                branch=branch,
            )

    async def sync_source_version(self, version_entry: Dict[str, Any]) -> Dict[str, Any]:
        """Pull and rebuild an existing branch-based 1Cat-vLLM source install."""
        version_entry = version_entry or {}
        branch = str(version_entry.get("source_branch") or "").strip()
        version_name = str(version_entry.get("version") or "").strip()
        venv_path = str(version_entry.get("venv_path") or "").strip()
        if (version_entry.get("install_type") or version_entry.get("type")) != "source":
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

                    await self._run_pip(
                        ["install", "--upgrade", "pip", "setuptools", "wheel"],
                        "sync_source",
                        cwd=clone_dir,
                        env=build_env,
                    )
                    for req in ("build.txt", "cuda.txt", "common.txt"):
                        req_path = os.path.join(clone_dir, "requirements", req)
                        if os.path.exists(req_path):
                            code = await self._run_pip(
                                [
                                    "install",
                                    "--extra-index-url",
                                    TORCH_CUDA_INDEX,
                                    "-r",
                                    req_path,
                                ],
                                "sync_source",
                                cwd=clone_dir,
                                env=build_env,
                            )
                            if code != 0:
                                raise RuntimeError(
                                    f"pip install -r requirements/{req} failed ({code})"
                                )
                    code = await self._run_pip(
                        ["install", "cmake", "build"],
                        "sync_source",
                        cwd=clone_dir,
                        env=build_env,
                    )
                    if code != 0:
                        raise RuntimeError(f"pip install cmake build failed ({code})")

                    for filename in os.listdir(dist_dir):
                        if filename.endswith(".whl"):
                            try:
                                os.remove(os.path.join(dist_dir, filename))
                            except OSError:
                                pass

                    python_exe = self._venv_python()
                    fa_dir = os.path.join(clone_dir, "flash-attention-v100")
                    if os.path.isdir(fa_dir):
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
                        raise RuntimeError(f"vllm wheel build failed ({code})")

                    wheels = [
                        os.path.join(dist_dir, f)
                        for f in sorted(os.listdir(dist_dir))
                        if f.endswith(".whl")
                    ]
                    if not wheels:
                        raise RuntimeError("Source sync produced no wheels")
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
                                "updated_at": _utcnow(),
                            },
                        )
                        if updated:
                            try:
                                from backend.engine_param_scanner import (
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
            args = ["uninstall", "-y", "vllm", "flash_attn_v100"]

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
        store = get_store()
        active = store.get_active_engine_version(ENGINE_ID)
        saved_venv = self._venv_path
        try:
            if active and active.get("venv_path"):
                self._venv_path = active["venv_path"]
            version = self._detect_installed_version()
            binary_path = self._resolve_binary_path()
            installed = version is not None and binary_path is not None
            state = self._load_state()
            venv_display = (
                (active.get("venv_path") if active else None)
                or state.get("venv_path")
                or self._venv_path
            )
            return {
                "installed": installed,
                "version": version,
                "binary_path": binary_path,
                "venv_path": venv_display,
                "installed_at": (active.get("installed_at") if active else None)
                or state.get("installed_at"),
                "removed_at": state.get("removed_at"),
                "operation": self._operation,
                "operation_started_at": self._operation_started_at,
                "progress_task_id": self._progress_task_id,
                "last_error": self._last_error,
                "log_path": self._log_path,
                "install_type": (active.get("install_type") if active else None),
                "release_tag": active.get("release_tag") if active else None,
                "source_repo": active.get("source_repo") if active else None,
                "source_branch": active.get("source_branch") if active else None,
            }
        finally:
            self._venv_path = saved_venv

    def read_log_tail(self, max_bytes: int = 8192) -> str:
        if not os.path.exists(self._log_path):
            return ""
        with open(self._log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            data = log_file.read().decode("utf-8", errors="replace")
            if size > max_bytes:
                data = data.split("\n", 1)[-1]
            return data.strip()
