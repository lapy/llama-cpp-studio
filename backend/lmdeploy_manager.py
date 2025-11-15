import asyncio
import os
import shlex
import shutil
from datetime import datetime
from typing import Optional, Dict, Any

import httpx
from asyncio.subprocess import Process, STDOUT

from backend.logging_config import get_logger

logger = get_logger(__name__)

_lmdeploy_manager_instance: Optional["LMDeployManager"] = None


def get_lmdeploy_manager() -> "LMDeployManager":
    """Return singleton LMDeploy manager."""
    global _lmdeploy_manager_instance
    if _lmdeploy_manager_instance is None:
        _lmdeploy_manager_instance = LMDeployManager()
    return _lmdeploy_manager_instance


class LMDeployManager:
    """Manage LMDeploy TurboMind runtime lifecycle."""

    def __init__(
        self,
        binary_path: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 2001,
    ):
        self.binary_path = binary_path or os.getenv("LMDEPLOY_BIN", "lmdeploy")
        self.host = host
        self.port = int(os.getenv("LMDEPLOY_PORT", port))
        self._process: Optional[Process] = None
        self._log_file = None
        self._lock = asyncio.Lock()
        self._current_instance: Optional[Dict[str, Any]] = None
        self._started_at: Optional[str] = None
        self._log_path = os.path.join("data", "logs", "lmdeploy.log")
        self._health_timeout = 180  # seconds
        self._last_health_status: Optional[Dict[str, Any]] = None

    async def start(self, model_entry: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Start LMDeploy serving the provided model. Only one model may run at once."""
        async with self._lock:
            if self._process and self._process.returncode is None:
                raise RuntimeError("LMDeploy runtime is already running")

            model_path = model_entry.get("file_path")
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")
            model_dir = model_entry.get("model_dir") or os.path.dirname(model_path)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Model directory not found at {model_dir}")
            model_dir_abs = os.path.abspath(model_dir)

            binary = self._resolve_binary()
            command = self._build_command(binary, model_dir_abs, config)
            env = os.environ.copy()
            env.setdefault("LMDEPLOY_LOG_DIR", os.path.dirname(self._log_path))
            os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
            self._log_file = open(self._log_path, "ab", buffering=0)

            logger.info(f"Starting LMDeploy with command: {' '.join(command)}")
            self._process = await asyncio.create_subprocess_exec(
                *command,
                stdout=self._log_file,
                stderr=STDOUT,
                cwd=model_dir_abs,
                env=env,
            )
            self._started_at = datetime.utcnow().isoformat() + "Z"
            self._current_instance = {
                "model_id": model_entry.get("model_id"),
                "huggingface_id": model_entry.get("huggingface_id"),
                "filename": model_entry.get("filename"),
                "file_path": model_path,
                "config": config,
                "pid": self._process.pid,
            }

        try:
            await self._wait_for_ready()
        except Exception as exc:
            await self.stop(force=True)
            raise exc

        return self.status()

    async def stop(self, force: bool = False) -> None:
        """Stop LMDeploy process if running."""
        async with self._lock:
            if not self._process:
                return
            if self._process.returncode is None:
                try:
                    self._process.terminate()
                    await asyncio.wait_for(self._process.wait(), timeout=30)
                except asyncio.TimeoutError:
                    logger.warning("LMDeploy did not terminate gracefully; killing process")
                    self._process.kill()
                    await self._process.wait()
                except ProcessLookupError:
                    logger.debug("LMDeploy process already stopped")
            elif force:
                try:
                    self._process.kill()
                except ProcessLookupError:
                    pass
            self._cleanup_process_state()

    async def restart(self, model_entry: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """Restart LMDeploy with a new model/config."""
        await self.stop()
        return await self.start(model_entry, config)

    def status(self) -> Dict[str, Any]:
        """Return status payload describing the running instance."""
        running = bool(self._process and self._process.returncode is None)
        return {
            "running": running,
            "port": self.port,
            "host": self.host,
            "process_id": self._process.pid if running else None,
            "started_at": self._started_at,
            "current_instance": self._current_instance if running else None,
            "health": self._last_health_status,
            "binary_path": self._current_binary_path(),
            "log_path": self._log_path,
        }

    def _current_binary_path(self) -> Optional[str]:
        try:
            return self._resolve_binary()
        except FileNotFoundError:
            return None

    def _resolve_binary(self) -> str:
        try:
            from backend.lmdeploy_installer import get_lmdeploy_installer

            installer_binary = get_lmdeploy_installer().status().get("binary_path")
            if installer_binary and os.path.exists(installer_binary):
                return installer_binary
        except Exception as exc:
            logger.debug(f"Failed to resolve LMDeploy binary via installer status: {exc}")

        resolved = shutil.which(self.binary_path)
        if resolved:
            return resolved

        candidate = os.path.expanduser(self.binary_path)
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            "LMDeploy binary not found in PATH. Install LMDeploy from the LMDeploy page or set LMDEPLOY_BIN."
        )

    def _build_command(self, binary: str, model_dir: str, config: Dict[str, Any]) -> list:
        """Convert stored config into lmdeploy CLI arguments."""
        tensor_parallel = max(1, int(config.get("tensor_parallel") or 1))
        pipeline_parallel = max(1, int(config.get("pipeline_parallel") or 1))
        context_length = max(1024, int(config.get("context_length") or 4096))
        max_batch_size = max(1, int(config.get("max_batch_size") or 4))
        max_batch_tokens = max(
            context_length,
            int(config.get("max_batch_tokens") or (context_length * 2)),
        )

        command = [
            binary,
            "serve",
            "api_server",
            model_dir,
            "--backend",
            "turbomind",
            "--server-name",
            self.host,
            "--server-port",
            str(self.port),
            "--tp",
            str(tensor_parallel),
            "--session-len",
            str(context_length),
            "--max-batch-size",
            str(max_batch_size),
        ]

        # newer CLI exposes --tp-split for tensor splitting
        tensor_split = config.get("tensor_split") or []
        if isinstance(tensor_split, str):
            tensor_split = [part.strip() for part in tensor_split.split(",") if part.strip()]
        if isinstance(tensor_split, list) and tensor_split:
            command.extend(["--tp-split", ",".join(str(part) for part in tensor_split)])

        if pipeline_parallel > 1:
            logger.warning(
                "Pipeline parallel is not supported in lmdeploy serve api_server; "
                "launch will continue with tensor parallel only."
            )

        # Optional inference settings
        if max_batch_tokens:
            command.extend(["--max-prefill-token-num", str(max_batch_tokens)])

        tensor_split = config.get("tensor_split") or []
        if isinstance(tensor_split, str):
            tensor_split = [part.strip() for part in tensor_split.split(",") if part.strip()]
        if isinstance(tensor_split, list) and tensor_split:
            command.extend(["--tp-split", ",".join(str(part) for part in tensor_split)])

        additional_args = config.get("additional_args")
        if isinstance(additional_args, str) and additional_args.strip():
            command.extend(shlex.split(additional_args.strip()))

        return command

    async def _wait_for_ready(self) -> None:
        """Poll LMDeploy server until healthy or timeout."""
        start_time = asyncio.get_event_loop().time()
        url = f"http://{self.host}:{self.port}/v1/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self._process and self._process.returncode not in (None, 0):
                    self._raise_with_logs(
                        f"LMDeploy exited unexpectedly with code {self._process.returncode}"
                    )
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        self._last_health_status = {
                            "status": "ready",
                            "checked_at": datetime.utcnow().isoformat() + "Z",
                        }
                        return
                except Exception as exc:
                    logger.debug(f"LMDeploy health check pending: {exc}")
                if asyncio.get_event_loop().time() - start_time > self._health_timeout:
                    self._raise_with_logs("Timed out waiting for LMDeploy server to become ready")
                await asyncio.sleep(2)

    def _cleanup_process_state(self) -> None:
        if self._log_file:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None
        self._process = None
        self._current_instance = None
        self._started_at = None
        self._last_health_status = {
            "status": "stopped",
            "checked_at": datetime.utcnow().isoformat() + "Z",
        }

    def _read_log_tail(self, max_bytes: int = 8192) -> str:
        """Return the tail of the lmdeploy log file for debugging."""
        try:
            with open(self._log_path, "rb") as log_file:
                log_file.seek(0, os.SEEK_END)
                file_size = log_file.tell()
                seek_pos = max(0, file_size - max_bytes)
                log_file.seek(seek_pos)
                data = log_file.read().decode("utf-8", errors="replace")
                if seek_pos > 0:
                    # Remove potential partial first line
                    data = data.split("\n", 1)[-1]
                return data.strip()
        except Exception as exc:
            logger.error(f"Failed to read LMDeploy log tail: {exc}")
            return ""

    def _raise_with_logs(self, message: str) -> None:
        """Raise a runtime error that includes the recent LMDeploy logs."""
        log_tail = self._read_log_tail()
        if log_tail:
            logger.error(f"{message}\n--- LMDeploy log tail ---\n{log_tail}\n--- end ---")
            raise RuntimeError(f"{message}. See logs for details.\n{log_tail}")
        raise RuntimeError(message)

