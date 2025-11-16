import asyncio
import json
import os
import shlex
import shutil
from datetime import datetime
from typing import Optional, Dict, Any, List

import httpx
import psutil
from asyncio.subprocess import Process, STDOUT

from backend.logging_config import get_logger
from backend.database import SessionLocal, Model, RunningInstance
from backend.huggingface import DEFAULT_LMDEPLOY_CONTEXT

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
        host: str = "0.0.0.0",
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
        self._last_detected_external: Optional[Dict[str, Any]] = None

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

            # Derive a stable model name for LMDeploy's --model-name flag.
            # Preference order:
            # 1) Explicit model_name passed in model_entry
            # 2) Base model / display name from model_entry
            # 3) Hugging Face repo id
            # 4) Directory name
            model_name = (
                model_entry.get("model_name")
                or model_entry.get("display_name")
                or model_entry.get("huggingface_id")
                or os.path.basename(model_dir_abs.rstrip(os.sep))
            )

            # Inject model_name into config passed to LMDeploy so the command builder
            # can add --model-name and we persist it in status/config reflection.
            effective_config = dict(config or {})
            if model_name and not effective_config.get("model_name"):
                effective_config["model_name"] = model_name

            binary = self._resolve_binary()
            command = self._build_command(binary, model_dir_abs, effective_config)
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
                "config": effective_config,
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
        detection = None
        if not running:
            detection = self._detect_external_process()
            if detection:
                running = True
                self._last_detected_external = detection
                if not self._current_instance:
                    self._current_instance = detection.get("instance")
                if not self._started_at:
                    self._started_at = detection.get("started_at")
            else:
                self._last_detected_external = None
        else:
            self._last_detected_external = None

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
            "auto_detected": bool(detection),
            "detection": detection,
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
        session_len = max(
            1024,
            int(
                config.get("session_len")
                or config.get("context_length")
                or DEFAULT_LMDEPLOY_CONTEXT
            ),
        )
        max_batch_size = max(1, int(config.get("max_batch_size") or 4))
        max_prefill_token_num = max(
            session_len,
            int(
                config.get("max_prefill_token_num")
                or config.get("max_batch_tokens")
                or (session_len * 2)
            ),
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
            str(session_len),
            "--max-batch-size",
            str(max_batch_size),
        ]

        # Optional model identity for OpenAI-style /v1/models listing
        model_name = config.get("model_name")
        if model_name and str(model_name).strip():
            command.extend(["--model-name", str(model_name).strip()])

        # Optional inference settings
        dtype = config.get("dtype")
        if dtype and str(dtype).strip():
            command.extend(["--dtype", str(dtype).strip()])
        if max_prefill_token_num:
            command.extend(["--max-prefill-token-num", str(max_prefill_token_num)])
        cache_max_entry_count = config.get("cache_max_entry_count")
        if cache_max_entry_count is not None:
            command.extend(["--cache-max-entry-count", str(cache_max_entry_count)])
        cache_block_seq_len = config.get("cache_block_seq_len")
        if cache_block_seq_len:
            command.extend(["--cache-block-seq-len", str(cache_block_seq_len)])
        if config.get("enable_prefix_caching"):
            command.append("--enable-prefix-caching")
        quant_policy = config.get("quant_policy")
        if quant_policy is not None:
            command.extend(["--quant-policy", str(quant_policy)])
        model_format = config.get("model_format")
        if model_format and str(model_format).strip():
            command.extend(["--model-format", str(model_format).strip()])
        hf_overrides = config.get("hf_overrides")
        if hf_overrides:
            command.extend(["--hf-overrides", hf_overrides if isinstance(hf_overrides, str) else str(hf_overrides)])
        if config.get("enable_metrics"):
            command.append("--enable-metrics")
        rope_scaling_factor = config.get("rope_scaling_factor")
        if rope_scaling_factor is not None:
            command.extend(["--rope-scaling-factor", str(rope_scaling_factor)])
        num_tokens_per_iter = config.get("num_tokens_per_iter")
        if num_tokens_per_iter:
            command.extend(["--num-tokens-per-iter", str(num_tokens_per_iter)])
        max_prefill_iters = config.get("max_prefill_iters")
        if max_prefill_iters:
            command.extend(["--max-prefill-iters", str(max_prefill_iters)])
        communicator = config.get("communicator")
        if communicator and str(communicator).strip():
            command.extend(["--communicator", str(communicator).strip()])

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

    def _detect_external_process(self) -> Optional[Dict[str, Any]]:
        """Scan system processes for an LMDeploy server launched outside the manager."""
        try:
            for proc in psutil.process_iter(attrs=["pid", "cmdline", "create_time"]):
                cmdline: List[str] = proc.info.get("cmdline") or []
                if not cmdline:
                    continue
                lowered = " ".join(cmdline).lower()
                if "lmdeploy" not in lowered:
                    continue
                if "serve" not in lowered or "api_server" not in lowered:
                    continue

                try:
                    api_server_idx = cmdline.index("api_server")
                except ValueError:
                    continue
                model_dir = cmdline[api_server_idx + 1] if len(cmdline) > api_server_idx + 1 else None
                detection = {
                    "pid": proc.info["pid"],
                    "cmdline": cmdline,
                    "model_dir": model_dir,
                    "detected_at": datetime.utcnow().isoformat() + "Z",
                }

                config = self._config_from_cmdline(cmdline)
                model_entry = self._lookup_model_by_dir(model_dir) if model_dir else None
                if model_entry:
                    self._ensure_running_instance_record(model_entry.id, config)
                    detection["instance"] = {
                        "model_id": model_entry.id,
                        "huggingface_id": model_entry.huggingface_id,
                        "filename": os.path.basename(model_entry.file_path),
                        "file_path": model_entry.file_path,
                        "config": config,
                        "pid": proc.info["pid"],
                        "auto_detected": True,
                    }
                    detection["model_id"] = model_entry.id
                    detection["huggingface_id"] = model_entry.huggingface_id
                else:
                    detection["instance"] = {
                        "model_id": None,
                        "huggingface_id": None,
                        "filename": None,
                        "file_path": model_dir,
                        "config": config,
                        "pid": proc.info["pid"],
                        "auto_detected": True,
                    }

                started_at = proc.info.get("create_time")
                if started_at:
                    detection["started_at"] = datetime.utcfromtimestamp(started_at).isoformat() + "Z"
                else:
                    detection["started_at"] = datetime.utcnow().isoformat() + "Z"
                return detection
        except Exception as exc:
            logger.debug(f"LMDeploy external scan failed: {exc}")
        return None

    def _config_from_cmdline(self, cmdline: List[str]) -> Dict[str, Any]:
        """Reconstruct a minimal config dict from lmdeploy CLI arguments."""
        def _extract(flag: str, cast, default=None):
            if flag in cmdline:
                idx = cmdline.index(flag)
                if idx + 1 < len(cmdline):
                    try:
                        return cast(cmdline[idx + 1])
                    except (ValueError, TypeError):
                        return default
            return default

        session_len = _extract("--session-len", int, DEFAULT_LMDEPLOY_CONTEXT)
        max_prefill = _extract("--max-prefill-token-num", int, session_len)
        # Note: --max-context-token-num doesn't exist in LMDeploy, so derive from session_len
        max_context = session_len

        config = {
            "session_len": session_len,
            "tensor_parallel": _extract("--tp", int, 1),
            "max_batch_size": _extract("--max-batch-size", int, 4),
            "max_prefill_token_num": max_prefill,
            "max_context_token_num": max_context,
            "dtype": _extract("--dtype", str, "auto"),
            "cache_max_entry_count": _extract("--cache-max-entry-count", float, 0.8),
            "cache_block_seq_len": _extract("--cache-block-seq-len", int, 64),
            "enable_prefix_caching": "--enable-prefix-caching" in cmdline,
            "quant_policy": _extract("--quant-policy", int, 0),
            "model_format": _extract("--model-format", str, ""),
            "hf_overrides": _extract("--hf-overrides", str, ""),
            "enable_metrics": "--enable-metrics" in cmdline,
            "rope_scaling_factor": _extract("--rope-scaling-factor", float, 0.0),
            "num_tokens_per_iter": _extract("--num-tokens-per-iter", int, 0),
            "max_prefill_iters": _extract("--max-prefill-iters", int, 1),
            "communicator": _extract("--communicator", str, "nccl"),
            "model_name": _extract("--model-name", str, ""),
            "additional_args": "",
        }

        if "--tp-split" in cmdline:
            idx = cmdline.index("--tp-split")
            if idx + 1 < len(cmdline):
                parts = [float(part) for part in cmdline[idx + 1].split(",") if part.strip()]
                config["tensor_split"] = parts
        else:
            config["tensor_split"] = []
        return config

    def _lookup_model_by_dir(self, model_dir: Optional[str]) -> Optional[Model]:
        if not model_dir:
            return None
        db = SessionLocal()
        try:
            candidates = db.query(Model).filter(Model.model_format == "safetensors").all()
            for candidate in candidates:
                if candidate.file_path and os.path.dirname(candidate.file_path) == model_dir:
                    return candidate
        finally:
            db.close()
        return None

    def _ensure_running_instance_record(self, model_id: Optional[int], config: Dict[str, Any]) -> None:
        if not model_id:
            return
        db = SessionLocal()
        try:
            existing = (
                db.query(RunningInstance)
                .filter(
                    RunningInstance.model_id == model_id,
                    RunningInstance.runtime_type == "lmdeploy",
                )
                .first()
            )
            if existing:
                return
            instance = RunningInstance(
                model_id=model_id,
                llama_version="lmdeploy",
                proxy_model_name=f"lmdeploy::{model_id}",
                started_at=datetime.utcnow(),
                config=json.dumps({"lmdeploy": config}),
                runtime_type="lmdeploy",
            )
            db.add(instance)
            model = db.query(Model).filter(Model.id == model_id).first()
            if model:
                model.is_active = True
            db.commit()
        except Exception as exc:
            logger.warning(f"Failed to create LMDeploy running instance record: {exc}")
            db.rollback()
        finally:
            db.close()

