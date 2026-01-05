import asyncio
import json
import os
import shutil
import subprocess
import sys
from asyncio.subprocess import PIPE, STDOUT
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Optional

from backend.logging_config import get_logger
from backend.websocket_manager import websocket_manager


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


logger = get_logger(__name__)

_installer_instance: Optional["LMDeployInstaller"] = None


def get_lmdeploy_installer() -> "LMDeployInstaller":
    global _installer_instance
    if _installer_instance is None:
        _installer_instance = LMDeployInstaller()
    return _installer_instance


class LMDeployInstaller:
    """Install or remove LMDeploy inside the runtime environment on demand."""

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._operation: Optional[str] = None
        self._operation_started_at: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._last_error: Optional[str] = None
        data_root = os.path.abspath("data")
        base_path = base_dir or os.path.join(data_root, "lmdeploy")
        self._base_dir = os.path.abspath(base_path)
        self._venv_path = os.path.join(self._base_dir, "venv")
        log_path = log_path or os.path.join(data_root, "logs", "lmdeploy_install.log")
        state_path = state_path or os.path.join(data_root, "configs", "lmdeploy_installer.json")
        self._log_path = os.path.abspath(log_path)
        self._state_path = os.path.abspath(state_path)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)

    def _venv_bin(self, executable: str) -> str:
        if os.name == "nt":
            exe = executable if executable.lower().endswith(".exe") else f"{executable}.exe"
            return os.path.join(self._venv_path, "Scripts", exe)
        return os.path.join(self._venv_path, "bin", executable)

    def _venv_python(self) -> str:
        return self._venv_bin("python")

    def _ensure_venv(self) -> None:
        python_path = self._venv_python()
        if os.path.exists(python_path):
            return
        os.makedirs(self._base_dir, exist_ok=True)
        try:
            subprocess.run([sys.executable, "-m", "venv", self._venv_path], check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Failed to create LMDeploy virtual environment: {exc}") from exc

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self._state_path):
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Failed to load LMDeploy installer state: {exc}")
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
        script = (
            "import importlib, sys\n"
            "try:\n"
            "    from importlib import metadata\n"
            "except ImportError:\n"
            "    import importlib_metadata as metadata\n"
            "try:\n"
            "    print(metadata.version('lmdeploy'))\n"
            "except metadata.PackageNotFoundError:\n"
            "    sys.exit(1)\n"
        )
        try:
            output = subprocess.check_output([python_exe, "-c", script], text=True).strip()
            return output or None
        except subprocess.CalledProcessError:
            return None
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Unable to determine LMDeploy version: {exc}")
            return None

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

        resolved = shutil.which("lmdeploy")
        return resolved

    def _update_installed_state(self, installed: bool, version: Optional[str] = None) -> None:
        state = self._load_state()
        if installed:
            state["installed_at"] = _utcnow()
            if version:
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

    async def _run_pip(self, args: list[str], operation: str, ensure_venv: bool = True) -> int:
        if ensure_venv:
            self._ensure_venv()
        python_exe = self._venv_python()
        if not os.path.exists(python_exe):
            raise RuntimeError("LMDeploy virtual environment is missing; cannot run pip.")
        header = f"[{_utcnow()}] Starting LMDeploy {operation} via pip {' '.join(args)}\n"
        with open(self._log_path, "w", encoding="utf-8") as log_file:
            log_file.write(header)
        process = await asyncio.create_subprocess_exec(
            python_exe,
            "-m",
            "pip",
            *args,
            stdout=PIPE,
            stderr=STDOUT,
        )

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
        return process.returncode or 0

    async def _broadcast_log_line(self, line: str) -> None:
        try:
            await websocket_manager.broadcast(
                {
                    "type": "lmdeploy_install_log",
                    "line": line,
                    "timestamp": _utcnow(),
                }
            )
        except Exception as exc:  # pragma: no cover
            logger.debug(f"Failed to broadcast LMDeploy log line: {exc}")

    async def _set_operation(self, operation: str) -> None:
        self._operation = operation
        self._operation_started_at = _utcnow()
        self._last_error = None
        await websocket_manager.broadcast(
            {
                "type": "lmdeploy_install_status",
                "status": operation,
                "started_at": self._operation_started_at,
            }
        )

    async def _finish_operation(self, success: bool, message: str = "") -> None:
        payload = {
            "type": "lmdeploy_install_status",
            "status": "completed" if success else "failed",
            "operation": self._operation,
            "message": message,
            "ended_at": _utcnow(),
        }
        await websocket_manager.broadcast(payload)
        self._operation = None
        self._operation_started_at = None

    def _create_task(self, coro: Awaitable[Any]) -> None:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        self._current_task = task

        def _cleanup(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception as exc:  # pragma: no cover - surfaced via status
                logger.error(f"LMDeploy installer task error: {exc}")
            finally:
                self._current_task = None

        task.add_done_callback(_cleanup)

    async def install(self, version: Optional[str] = None, force_reinstall: bool = False) -> Dict[str, Any]:
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another LMDeploy installer operation is already running")
            await self._set_operation("install")
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
                    await self._finish_operation(True, "LMDeploy installed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return {"message": "LMDeploy installation started"}

    async def remove(self) -> Dict[str, Any]:
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another LMDeploy installer operation is already running")
            await self._set_operation("remove")
            args = ["uninstall", "-y", "lmdeploy"]

            async def _runner():
                try:
                    python_exists = os.path.exists(self._venv_python())
                    if python_exists:
                        code = await self._run_pip(args, "remove", ensure_venv=False)
                        if code != 0:
                            raise RuntimeError(f"pip exited with status {code}")
                    shutil.rmtree(self._venv_path, ignore_errors=True)
                    self._update_installed_state(False)
                    await self._finish_operation(True, "LMDeploy removed")
                except Exception as exc:
                    self._last_error = str(exc)
                    self._refresh_state_from_environment()
                    await self._finish_operation(False, str(exc))

            self._create_task(_runner())
            return {"message": "LMDeploy removal started"}

    def status(self) -> Dict[str, Any]:
        version = self._detect_installed_version()
        binary_path = self._resolve_binary_path()
        installed = version is not None and binary_path is not None
        state = self._load_state()
        return {
            "installed": installed,
            "version": version,
            "binary_path": binary_path,
            "venv_path": state.get("venv_path") or self._venv_path,
            "installed_at": state.get("installed_at"),
            "removed_at": state.get("removed_at"),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "last_error": self._last_error,
            "log_path": self._log_path,
        }

    def is_operation_running(self) -> bool:
        return self._operation is not None

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

