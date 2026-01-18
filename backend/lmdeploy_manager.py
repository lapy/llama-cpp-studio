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
from backend.huggingface import DEFAULT_LMDEPLOY_CONTEXT, MAX_LMDEPLOY_CONTEXT
from backend.websocket_manager import websocket_manager

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
        self._last_broadcast_log_position = 0

    async def start(
        self, model_entry: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
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
                    logger.warning(
                        "LMDeploy did not terminate gracefully; killing process"
                    )
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

    async def restart(
        self, model_entry: Dict[str, Any], config: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            logger.debug(
                f"Failed to resolve LMDeploy binary via installer status: {exc}"
            )

        resolved = shutil.which(self.binary_path)
        if resolved:
            return resolved

        candidate = os.path.expanduser(self.binary_path)
        if os.path.isabs(candidate) and os.path.exists(candidate):
            return candidate
        raise FileNotFoundError(
            "LMDeploy binary not found in PATH. Install LMDeploy from the LMDeploy page or set LMDEPLOY_BIN."
        )

    def _build_command(
        self, binary: str, model_dir: str, config: Dict[str, Any]
    ) -> list:
        """Convert stored config into lmdeploy CLI arguments."""
        tensor_parallel = max(1, int(config.get("tensor_parallel") or 1))
        base_session_len = max(
            1024,
            int(
                config.get("session_len")
                or config.get("context_length")
                or DEFAULT_LMDEPLOY_CONTEXT
            ),
        )
        rope_scaling_mode = str(config.get("rope_scaling_mode") or "disabled").lower()
        rope_scaling_factor = float(config.get("rope_scaling_factor") or 1.0)
        scaling_enabled = (
            rope_scaling_mode not in {"", "none", "disabled"}
            and rope_scaling_factor > 1.0
        )
        effective_session_len = base_session_len
        if scaling_enabled:
            scaled = int(base_session_len * rope_scaling_factor)
            effective_session_len = max(
                base_session_len, min(scaled, MAX_LMDEPLOY_CONTEXT)
            )
        max_batch_size = max(1, int(config.get("max_batch_size") or 4))
        base_prefill = int(
            config.get("max_prefill_token_num")
            or config.get("max_batch_tokens")
            or (base_session_len * 2)
        )
        if scaling_enabled:
            scaled_prefill = int(base_prefill * rope_scaling_factor)
            max_prefill_token_num = scaled_prefill
        else:
            max_prefill_token_num = base_prefill

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
            str(effective_session_len),
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
        if isinstance(hf_overrides, dict) and hf_overrides:

            def _flatten(prefix: str, value: Any):
                if isinstance(value, dict):
                    for key, nested in value.items():
                        if not isinstance(key, str) or not key:
                            continue
                        new_prefix = f"{prefix}.{key}" if prefix else key
                        yield from _flatten(new_prefix, nested)
                else:
                    yield prefix, value

            def _format_override_value(val: Any) -> str:
                if isinstance(val, bool):
                    return "true" if val else "false"
                if val is None:
                    return "null"
                return str(val)

            for path, value in _flatten("", hf_overrides):
                if not path:
                    continue
                command.extend(
                    [f"--hf-overrides.{path}", _format_override_value(value)]
                )
        elif isinstance(hf_overrides, str) and hf_overrides.strip():
            command.extend(["--hf-overrides", hf_overrides.strip()])
        # LMDeploy uses --disable-metrics (inverted logic)
        # When enable_metrics=false, send --disable-metrics
        # When enable_metrics=true (default), don't send anything (metrics enabled by default)
        if not config.get("enable_metrics", True):
            command.append("--disable-metrics")
        if scaling_enabled:
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

        # Server configuration parameters
        allow_origins = config.get("allow_origins")
        if allow_origins:
            if isinstance(allow_origins, list):
                command.extend(
                    ["--allow-origins"] + [str(origin) for origin in allow_origins]
                )
            elif isinstance(allow_origins, str):
                command.extend(["--allow-origins", allow_origins])
        if config.get("allow_credentials"):
            command.append("--allow-credentials")
        allow_methods = config.get("allow_methods")
        if allow_methods:
            if isinstance(allow_methods, list):
                command.extend(
                    ["--allow-methods"] + [str(method) for method in allow_methods]
                )
            elif isinstance(allow_methods, str):
                command.extend(["--allow-methods", allow_methods])
        allow_headers = config.get("allow_headers")
        if allow_headers:
            if isinstance(allow_headers, list):
                command.extend(
                    ["--allow-headers"] + [str(header) for header in allow_headers]
                )
            elif isinstance(allow_headers, str):
                command.extend(["--allow-headers", allow_headers])
        proxy_url = config.get("proxy_url")
        if proxy_url and str(proxy_url).strip():
            command.extend(["--proxy-url", str(proxy_url).strip()])
        max_concurrent_requests = config.get("max_concurrent_requests")
        if max_concurrent_requests is not None:
            command.extend(
                ["--max-concurrent-requests", str(int(max_concurrent_requests))]
            )
        log_level = config.get("log_level")
        if log_level and str(log_level).strip():
            command.extend(["--log-level", str(log_level).strip()])
        api_keys = config.get("api_keys")
        if api_keys:
            if isinstance(api_keys, list):
                command.extend(["--api-keys"] + [str(key) for key in api_keys])
            elif isinstance(api_keys, str):
                command.extend(["--api-keys", api_keys])
        if config.get("ssl"):
            command.append("--ssl")
        max_log_len = config.get("max_log_len")
        if max_log_len is not None:
            command.extend(["--max-log-len", str(int(max_log_len))])
        if config.get("disable_fastapi_docs"):
            command.append("--disable-fastapi-docs")
        if config.get("allow_terminate_by_client"):
            command.append("--allow-terminate-by-client")
        if config.get("enable_abort_handling"):
            command.append("--enable-abort-handling")

        # Model configuration parameters
        chat_template = config.get("chat_template")
        if chat_template and str(chat_template).strip():
            command.extend(["--chat-template", str(chat_template).strip()])
        tool_call_parser = config.get("tool_call_parser")
        if tool_call_parser and str(tool_call_parser).strip():
            command.extend(["--tool-call-parser", str(tool_call_parser).strip()])
        reasoning_parser = config.get("reasoning_parser")
        if reasoning_parser and str(reasoning_parser).strip():
            command.extend(["--reasoning-parser", str(reasoning_parser).strip()])
        revision = config.get("revision")
        if revision and str(revision).strip():
            command.extend(["--revision", str(revision).strip()])
        download_dir = config.get("download_dir")
        if download_dir and str(download_dir).strip():
            command.extend(["--download-dir", str(download_dir).strip()])
        adapters = config.get("adapters")
        if adapters:
            if isinstance(adapters, list):
                command.extend(["--adapters"] + [str(adapter) for adapter in adapters])
            elif isinstance(adapters, str):
                command.extend(["--adapters", adapters])
        device = config.get("device")
        if device and str(device).strip():
            command.extend(["--device", str(device).strip()])
        if config.get("eager_mode"):
            command.append("--eager-mode")
        if config.get("disable_vision_encoder"):
            command.append("--disable-vision-encoder")
        logprobs_mode = config.get("logprobs_mode")
        if logprobs_mode is not None:
            command.extend(["--logprobs-mode", str(logprobs_mode)])

        # DLLM parameters
        dllm_block_length = config.get("dllm_block_length")
        if dllm_block_length is not None:
            command.extend(["--dllm-block-length", str(int(dllm_block_length))])
        dllm_unmasking_strategy = config.get("dllm_unmasking_strategy")
        if dllm_unmasking_strategy and str(dllm_unmasking_strategy).strip():
            command.extend(
                ["--dllm-unmasking-strategy", str(dllm_unmasking_strategy).strip()]
            )
        dllm_denoising_steps = config.get("dllm_denoising_steps")
        if dllm_denoising_steps is not None:
            command.extend(["--dllm-denoising-steps", str(int(dllm_denoising_steps))])
        dllm_confidence_threshold = config.get("dllm_confidence_threshold")
        if dllm_confidence_threshold is not None:
            command.extend(
                ["--dllm-confidence-threshold", str(float(dllm_confidence_threshold))]
            )

        # Distributed/Multi-node parameters
        dp = config.get("dp")
        if dp is not None:
            command.extend(["--dp", str(int(dp))])
        ep = config.get("ep")
        if ep is not None:
            command.extend(["--ep", str(int(ep))])
        if config.get("enable_microbatch"):
            command.append("--enable-microbatch")
        if config.get("enable_eplb"):
            command.append("--enable-eplb")
        role = config.get("role")
        if role and str(role).strip():
            command.extend(["--role", str(role).strip()])
        migration_backend = config.get("migration_backend")
        if migration_backend and str(migration_backend).strip():
            command.extend(["--migration-backend", str(migration_backend).strip()])
        node_rank = config.get("node_rank")
        if node_rank is not None:
            command.extend(["--node-rank", str(int(node_rank))])
        nnodes = config.get("nnodes")
        if nnodes is not None:
            command.extend(["--nnodes", str(int(nnodes))])
        cp = config.get("cp")
        if cp is not None:
            command.extend(["--cp", str(int(cp))])
        if config.get("enable_return_routed_experts"):
            command.append("--enable-return-routed-experts")
        distributed_executor_backend = config.get("distributed_executor_backend")
        if distributed_executor_backend and str(distributed_executor_backend).strip():
            command.extend(
                [
                    "--distributed-executor-backend",
                    str(distributed_executor_backend).strip(),
                ]
            )

        # Vision parameters
        vision_max_batch_size = config.get("vision_max_batch_size")
        if vision_max_batch_size is not None:
            command.extend(["--vision-max-batch-size", str(int(vision_max_batch_size))])

        # Speculative decoding parameters
        speculative_algorithm = config.get("speculative_algorithm")
        if speculative_algorithm and str(speculative_algorithm).strip():
            command.extend(
                ["--speculative-algorithm", str(speculative_algorithm).strip()]
            )
        speculative_draft_model = config.get("speculative_draft_model")
        if speculative_draft_model and str(speculative_draft_model).strip():
            command.extend(
                ["--speculative-draft-model", str(speculative_draft_model).strip()]
            )
        speculative_num_draft_tokens = config.get("speculative_num_draft_tokens")
        if speculative_num_draft_tokens is not None:
            command.extend(
                [
                    "--speculative-num-draft-tokens",
                    str(int(speculative_num_draft_tokens)),
                ]
            )

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
                    self._raise_with_logs(
                        "Timed out waiting for LMDeploy server to become ready"
                    )
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

    def read_log_tail(self, max_bytes: int = 8192) -> str:
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

    async def _broadcast_runtime_logs(self) -> None:
        """Broadcast new runtime log lines via WebSocket."""
        try:
            if not os.path.exists(self._log_path):
                return
            
            # Read new content since last broadcast
            current_size = os.path.getsize(self._log_path)
            if current_size <= self._last_broadcast_log_position:
                return  # No new content
            
            # Read only new content
            with open(self._log_path, "rb") as log_file:
                log_file.seek(self._last_broadcast_log_position)
                new_content = log_file.read().decode("utf-8", errors="replace")
                self._last_broadcast_log_position = current_size
            
            if new_content:
                # Split into lines and broadcast each non-empty line
                lines = new_content.split('\n')
                for line in lines:
                    if line.strip():  # Only send non-empty lines
                        await websocket_manager.send_lmdeploy_runtime_log(line.strip())
        except Exception as exc:
            logger.debug(f"Failed to broadcast LMDeploy runtime logs: {exc}")

    def _read_log_tail(self, max_bytes: int = 8192) -> str:
        """Private alias for backward compatibility."""
        return self.read_log_tail(max_bytes)

    def _raise_with_logs(self, message: str) -> None:
        """Raise a runtime error that includes the recent LMDeploy logs."""
        log_tail = self.read_log_tail()
        if log_tail:
            logger.error(
                f"{message}\n--- LMDeploy log tail ---\n{log_tail}\n--- end ---"
            )
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
                model_dir = (
                    cmdline[api_server_idx + 1]
                    if len(cmdline) > api_server_idx + 1
                    else None
                )
                detection = {
                    "pid": proc.info["pid"],
                    "cmdline": cmdline,
                    "model_dir": model_dir,
                    "detected_at": datetime.utcnow().isoformat() + "Z",
                }

                config = self._config_from_cmdline(cmdline)
                model_entry = (
                    self._lookup_model_by_dir(model_dir) if model_dir else None
                )
                if model_entry:
                    self._ensure_running_instance_record(model_entry.id, config)
                    detection["instance"] = {
                        "model_id": model_entry.id,
                        "huggingface_id": model_entry.huggingface_id,
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
                        "file_path": model_dir,
                        "config": config,
                        "pid": proc.info["pid"],
                        "auto_detected": True,
                    }

                started_at = proc.info.get("create_time")
                if started_at:
                    detection["started_at"] = (
                        datetime.utcfromtimestamp(started_at).isoformat() + "Z"
                    )
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

        def _extract_list(flag: str, default=None):
            """Extract list of values for flags that accept multiple arguments."""
            if flag not in cmdline:
                return default
            idx = cmdline.index(flag)
            result = []
            i = idx + 1
            while i < len(cmdline) and not cmdline[i].startswith("--"):
                result.append(cmdline[i])
                i += 1
            return result if result else default

        session_len = _extract("--session-len", int, DEFAULT_LMDEPLOY_CONTEXT)
        max_prefill = _extract("--max-prefill-token-num", int, session_len)
        # Note: --max-context-token-num doesn't exist in LMDeploy, so derive from session_len
        max_context = session_len

        rope_scaling_factor = _extract("--rope-scaling-factor", float, 1.0)
        rope_scaling_mode = "disabled"
        if rope_scaling_factor and rope_scaling_factor > 1.0:
            rope_scaling_mode = "detected"

        hf_overrides: Dict[str, Any] = {}

        def _assign_nested(target: Dict[str, Any], path: List[str], value: Any) -> None:
            current = target
            for segment in path[:-1]:
                current = current.setdefault(segment, {})
            current[path[-1]] = value

        def _coerce_override_value(raw: str) -> Any:
            lowered = raw.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
            if lowered == "null":
                return None
            try:
                if "." in raw:
                    return float(raw)
                return int(raw)
            except ValueError:
                return raw

        i = 0
        while i < len(cmdline):
            token = cmdline[i]
            if token.startswith("--hf-overrides."):
                path_str = token[len("--hf-overrides.") :]
                if path_str and i + 1 < len(cmdline):
                    value = _coerce_override_value(cmdline[i + 1])
                    _assign_nested(hf_overrides, path_str.split("."), value)
                    i += 2
                    continue
            i += 1

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
            "hf_overrides": hf_overrides or _extract("--hf-overrides", str, ""),
            # LMDeploy uses --disable-metrics, so enable_metrics=True when flag is NOT present
            "enable_metrics": "--disable-metrics" not in cmdline,
            "rope_scaling_factor": rope_scaling_factor,
            "rope_scaling_mode": rope_scaling_mode,
            "num_tokens_per_iter": _extract("--num-tokens-per-iter", int, 0),
            "max_prefill_iters": _extract("--max-prefill-iters", int, 1),
            "communicator": _extract("--communicator", str, "nccl"),
            "model_name": _extract("--model-name", str, ""),
            # Server configuration
            "allow_origins": _extract_list("--allow-origins"),
            "allow_credentials": "--allow-credentials" in cmdline,
            "allow_methods": _extract_list("--allow-methods"),
            "allow_headers": _extract_list("--allow-headers"),
            "proxy_url": _extract("--proxy-url", str, ""),
            "max_concurrent_requests": _extract("--max-concurrent-requests", int),
            "log_level": _extract("--log-level", str, ""),
            "api_keys": _extract_list("--api-keys"),
            "ssl": "--ssl" in cmdline,
            "max_log_len": _extract("--max-log-len", int),
            "disable_fastapi_docs": "--disable-fastapi-docs" in cmdline,
            "allow_terminate_by_client": "--allow-terminate-by-client" in cmdline,
            "enable_abort_handling": "--enable-abort-handling" in cmdline,
            # Model configuration
            "chat_template": _extract("--chat-template", str, ""),
            "tool_call_parser": _extract("--tool-call-parser", str, ""),
            "reasoning_parser": _extract("--reasoning-parser", str, ""),
            "revision": _extract("--revision", str, ""),
            "download_dir": _extract("--download-dir", str, ""),
            "adapters": _extract_list("--adapters"),
            "device": _extract("--device", str, ""),
            "eager_mode": "--eager-mode" in cmdline,
            "disable_vision_encoder": "--disable-vision-encoder" in cmdline,
            "logprobs_mode": _extract("--logprobs-mode", str),
            # DLLM parameters
            "dllm_block_length": _extract("--dllm-block-length", int),
            "dllm_unmasking_strategy": _extract("--dllm-unmasking-strategy", str, ""),
            "dllm_denoising_steps": _extract("--dllm-denoising-steps", int),
            "dllm_confidence_threshold": _extract("--dllm-confidence-threshold", float),
            # Distributed/Multi-node parameters
            "dp": _extract("--dp", int),
            "ep": _extract("--ep", int),
            "enable_microbatch": "--enable-microbatch" in cmdline,
            "enable_eplb": "--enable-eplb" in cmdline,
            "role": _extract("--role", str, ""),
            "migration_backend": _extract("--migration-backend", str, ""),
            "node_rank": _extract("--node-rank", int),
            "nnodes": _extract("--nnodes", int),
            "cp": _extract("--cp", int),
            "enable_return_routed_experts": "--enable-return-routed-experts" in cmdline,
            "distributed_executor_backend": _extract(
                "--distributed-executor-backend", str, ""
            ),
            # Vision parameters
            "vision_max_batch_size": _extract("--vision-max-batch-size", int),
            # Speculative decoding parameters
            "speculative_algorithm": _extract("--speculative-algorithm", str, ""),
            "speculative_draft_model": _extract("--speculative-draft-model", str, ""),
            "speculative_num_draft_tokens": _extract(
                "--speculative-num-draft-tokens", int
            ),
            "additional_args": "",
        }

        return config

    def _lookup_model_by_dir(self, model_dir: Optional[str]) -> Optional[Model]:
        if not model_dir:
            return None
        db = SessionLocal()
        try:
            candidates = (
                db.query(Model).filter(Model.model_format == "safetensors").all()
            )
            for candidate in candidates:
                if (
                    candidate.file_path
                    and os.path.dirname(candidate.file_path) == model_dir
                ):
                    return candidate
        finally:
            db.close()
        return None

    def _ensure_running_instance_record(
        self, model_id: Optional[int], config: Dict[str, Any]
    ) -> None:
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
