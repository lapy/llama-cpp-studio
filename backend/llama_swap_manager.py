import subprocess
import asyncio
import copy
import json
import os
import re
import tempfile
import threading
import shlex
import yaml
import httpx
from typing import Any, Dict, List, Optional, Tuple
from backend.llama_swap_config import generate_llama_swap_config
from backend.data_store import get_store
from backend.logging_config import get_logger

logger = get_logger(__name__)

_SIDECAR_REVISION = re.compile(r"\.r(\d+)\.json")
RELOAD_CONFIRM_SECONDS = 7.0

# Global singleton instance
_llama_swap_manager_instance = None


def get_llama_swap_manager() -> "LlamaSwapManager":
    """Get the global llama-swap manager instance"""
    global _llama_swap_manager_instance
    if _llama_swap_manager_instance is None:
        from backend.llama_swap_client import get_proxy_port

        _llama_swap_manager_instance = LlamaSwapManager(proxy_port=get_proxy_port())
    return _llama_swap_manager_instance


def mark_swap_config_stale() -> None:
    """Mark that llama-swap YAML may be out of sync (studio DB or engine state changed)."""
    get_llama_swap_manager().mark_swap_config_stale()


def _json_norm(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def _flag_argv_to_pairs(tokens: List[str]) -> List[Tuple[str, Optional[str]]]:
    """Split a flat argv list into (flag, value_or_none) pairs for canonical ordering."""
    pairs: List[Tuple[str, Optional[str]]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if not t.startswith("-"):
            i += 1
            continue
        if i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
            pairs.append((t, tokens[i + 1]))
            i += 2
        else:
            pairs.append((t, None))
            i += 1
    return pairs


def _pairs_to_argv(pairs: List[Tuple[str, Optional[str]]]) -> List[str]:
    out: List[str] = []
    for flag, val in pairs:
        out.append(flag)
        if val is not None:
            out.append(val)
    return out


def _normalize_cmd_after_port_marker(cmd: str) -> Optional[str]:
    """
    If ``cmd`` contains ``--port ${PORT}`` or ``--server-port ${PORT}``, return the same
    string with argv tokens after that marker sorted for stable comparison.
    """
    for marker in ("--port ${PORT}", "--server-port ${PORT}"):
        pos = cmd.find(marker)
        if pos >= 0:
            break
    else:
        return None
    head = cmd[: pos + len(marker)].rstrip()
    rest = cmd[pos + len(marker) :].lstrip()
    if not rest:
        return cmd
    try:
        tokens = shlex.split(rest)
    except ValueError:
        return None
    pairs = _flag_argv_to_pairs(tokens)
    pairs.sort(key=lambda p: (p[0], p[1] if p[1] is not None else ""))
    new_parts = _pairs_to_argv(pairs)
    try:
        new_rest = shlex.join(new_parts)
    except AttributeError:  # pragma: no cover — Python 3.7 and older
        new_rest = " ".join(shlex.quote(p) for p in new_parts)
    return f"{head} {new_rest}"


def _normalize_bash_c_cmd_after_port_marker(cmd: str) -> str:
    """
    Reorder argv after --port ${PORT} or --server-port ${PORT} so semantically identical
    commands compare equal regardless of flag emission order.
    Supports ``bash -c '…'`` wrappers and plain one-line cmds.
    """
    prefix = "bash -c '"
    if cmd.startswith(prefix) and cmd.endswith("'"):
        inner = cmd[len(prefix) : -1]
        normalized = _normalize_cmd_after_port_marker(inner)
        if normalized is None:
            return cmd
        return f"{prefix}{normalized}'"
    normalized = _normalize_cmd_after_port_marker(cmd)
    return normalized if normalized is not None else cmd


def _canonicalize_llama_swap_doc_for_compare(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-copy and normalize structures where YAML / dict iteration order must not affect equality:
    - groups.*.members (unordered set of model names)
    - models.*.cmd (llama-swap shell one-liners; flag order is not semantically meaningful)
    """
    out = copy.deepcopy(doc)
    groups = out.get("groups")
    if isinstance(groups, dict):
        for gv in groups.values():
            if not isinstance(gv, dict):
                continue
            m = gv.get("members")
            if isinstance(m, list) and m and all(isinstance(x, str) for x in m):
                gv["members"] = sorted(m)
    models = out.get("models")
    if isinstance(models, dict):
        for mv in models.values():
            if isinstance(mv, dict):
                c = mv.get("cmd")
                if isinstance(c, str):
                    mv["cmd"] = _normalize_bash_c_cmd_after_port_marker(c)
    return out


def _norm_config_text(s: str) -> str:
    if not s:
        return ""
    return "\n".join(s.replace("\r\n", "\n").strip().splitlines())


def _configs_semantically_equal(disk_raw: str, desired_raw: str) -> bool:
    """True if parsed YAML documents are structurally the same (key order ignored)."""
    try:
        disk_doc = yaml.safe_load(disk_raw) if (disk_raw or "").strip() else {}
    except Exception:
        return False
    try:
        desired_doc = yaml.safe_load(desired_raw)
    except Exception:
        return False
    if not isinstance(disk_doc, dict):
        disk_doc = {}
    if not isinstance(desired_doc, dict):
        desired_doc = {}
    disk_doc = _canonicalize_llama_swap_doc_for_compare(disk_doc)
    desired_doc = _canonicalize_llama_swap_doc_for_compare(desired_doc)
    return _json_norm(disk_doc) == _json_norm(desired_doc)


def summarize_llama_swap_yaml_diff(disk_raw: str, desired_raw: str) -> List[str]:
    """
    Build short human-readable bullets comparing on-disk config vs what would be written.
    """
    try:
        disk_doc = yaml.safe_load(disk_raw) if (disk_raw or "").strip() else {}
    except Exception:
        disk_doc = {}
    if not isinstance(disk_doc, dict):
        disk_doc = {}
    try:
        desired_doc = yaml.safe_load(desired_raw)
    except Exception:
        return ["Generated config could not be parsed for comparison"]

    if not isinstance(desired_doc, dict):
        desired_doc = {}

    disk_doc = _canonicalize_llama_swap_doc_for_compare(disk_doc)
    desired_doc = _canonicalize_llama_swap_doc_for_compare(desired_doc)

    lines: List[str] = []
    dk, dv = disk_doc, desired_doc

    dm = dk.get("models") if isinstance(dk.get("models"), dict) else {}
    dvm = dv.get("models") if isinstance(dv.get("models"), dict) else {}

    for name in sorted(set(dvm.keys()) - set(dm.keys())):
        lines.append(f"Add model «{name}»")
    for name in sorted(set(dm.keys()) - set(dvm.keys())):
        lines.append(f"Remove model «{name}»")
    for name in sorted(set(dm.keys()) & set(dvm.keys())):
        if _json_norm(dm.get(name)) != _json_norm(dvm.get(name)):
            lines.append(f"Update model «{name}»")

    for section, label in (("profiles", "profile"), ("selectors", "selector")):
        disk_section = dk.get(section) if isinstance(dk.get(section), dict) else {}
        desired_section = dv.get(section) if isinstance(dv.get(section), dict) else {}
        for name in sorted(set(desired_section.keys()) - set(disk_section.keys())):
            lines.append(f"Add {label} «{name}»")
        for name in sorted(set(disk_section.keys()) - set(desired_section.keys())):
            lines.append(f"Remove {label} «{name}»")
        for name in sorted(set(disk_section.keys()) & set(desired_section.keys())):
            if _json_norm(disk_section.get(name)) != _json_norm(desired_section.get(name)):
                lines.append(f"Update {label} «{name}»")

    skip_global = {"models", "profiles", "selectors"}
    all_keys = set(dk.keys()) | set(dv.keys())
    for key in sorted(k for k in all_keys if k not in skip_global):
        if _json_norm(dk.get(key)) != _json_norm(dv.get(key)):
            lines.append(f"Change global option «{key}»")

    max_lines = 14
    if len(lines) > max_lines:
        extra = len(lines) - (max_lines - 1)
        lines = lines[: max_lines - 1] + [f"…and {extra} more changes"]
    return lines


class LlamaSwapManager:
    def __init__(self, proxy_port: int = None, config_path: str = None):
        from backend.llama_swap_client import DEFAULT_PROXY_PORT, get_proxy_port

        self.proxy_port = (
            int(proxy_port)
            if proxy_port is not None
            else get_proxy_port() or DEFAULT_PROXY_PORT
        )
        # Use absolute path to avoid permission issues with relative paths
        if config_path is None:
            config_path = "/app/data/llama-swap-config.yaml"
        self.config_path = (
            os.path.abspath(config_path)
            if not os.path.isabs(config_path)
            else config_path
        )
        self.process: Optional[subprocess.Popen] = None
        self.running_models: Dict[
            str, Dict[str, Any]
        ] = {}  # {proxy_model_name: {model_path, config}}
        self.proxy_url = f"http://localhost:{self.proxy_port}"
        self.admin_url = f"http://localhost:{self.proxy_port}/admin"
        self.monitor_task: Optional[asyncio.Task] = None
        self._should_restart = True  # Flag to control auto-restart
        self._swap_stale_lock = threading.Lock()
        self._swap_config_stale = False
        self._stale_epoch = 0
        self._apply_lock = asyncio.Lock()
        self.config_revision = 0
        self._previous_config_text = ""
        self._reload_watch = False
        self._reload_outcome: Optional[str] = None
        self._reload_confirm_timeout = RELOAD_CONFIRM_SECONDS

    def _client(self):
        """HTTP client bound to this manager's proxy port."""
        from backend.llama_swap_client import LlamaSwapClient

        return LlamaSwapClient(base_url=self.proxy_url)

    def mark_swap_config_stale(self) -> None:
        with self._swap_stale_lock:
            self._swap_config_stale = True
            self._stale_epoch += 1

    def clear_swap_config_stale(self) -> None:
        with self._swap_stale_lock:
            self._swap_config_stale = False

    def _is_swap_config_applicable_sync(self) -> bool:
        """True when any registered runtime has an active runnable installation."""
        from backend.llama_swap_config import any_active_runtime_in_db

        return any_active_runtime_in_db()

    def get_swap_config_stale_state(self) -> Dict[str, Any]:
        """
        Cheap snapshot for the UI badge: whether to prompt for “apply” without diffing YAML.
        """
        with self._swap_stale_lock:
            stale_flag = self._swap_config_stale
        applicable = self._is_swap_config_applicable_sync()
        return {
            "applicable": applicable,
            "stale": bool(applicable and stale_flag),
        }

    @staticmethod
    def _write_audio_sidecars(sidecars: Dict[str, dict]) -> None:
        """Atomically replace each generated audio.cpp server JSON sidecar."""
        from backend.audio_cpp_manager import get_audio_cpp_manager

        root = os.path.realpath(get_audio_cpp_manager().server_configs_dir)
        os.makedirs(root, exist_ok=True)
        for raw_path, payload in sidecars.items():
            path = os.path.realpath(raw_path)
            if (
                os.path.commonpath([root, path]) != root
                or os.path.dirname(path) != root
                or not path.endswith(".json")
            ):
                raise ValueError(f"Invalid audio.cpp sidecar path: {raw_path}")
            temp_path = f"{path}.tmp.{os.getpid()}"
            try:
                with open(temp_path, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, path)
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    @staticmethod
    def _remove_orphan_audio_sidecars(expected_paths: set[str]) -> None:
        from backend.audio_cpp_manager import get_audio_cpp_manager

        root = os.path.realpath(get_audio_cpp_manager().server_configs_dir)
        expected = {os.path.realpath(path) for path in expected_paths}
        if not os.path.isdir(root):
            return
        for filename in os.listdir(root):
            if not filename.endswith(".json"):
                continue
            path = os.path.realpath(os.path.join(root, filename))
            if path in expected or os.path.dirname(path) != root:
                continue
            try:
                os.remove(path)
            except OSError as exc:
                logger.warning("Could not remove orphan audio.cpp sidecar %s: %s", path, exc)

    async def _write_config(self) -> None:
        """Write config under the apply lock and clear stale when nothing changed mid-write."""
        async with self._apply_lock:
            epoch = self._stale_epoch
            await self._write_config_unlocked()
            if self._stale_epoch == epoch:
                self.clear_swap_config_stale()

    async def _compose_config(self) -> Tuple[str, Dict[str, dict]]:
        from backend.llama_swap_config import any_active_runtime_in_db

        if not any_active_runtime_in_db():
            logger.error(
                "Cannot write config: no registered inference runtime is active on disk"
            )
            raise ValueError(
                "No inference runtime available: activate a compatible engine build"
            )
        store = get_store()
        all_models = store.list_models()
        audio_sidecars: Dict[str, dict] = {}
        config_content = generate_llama_swap_config(
            self.running_models,
            all_models,
            sidecar_payloads=audio_sidecars,
        )
        self._validate_swap_yaml(config_content)
        return config_content, audio_sidecars

    @staticmethod
    def _validate_swap_yaml(content: str) -> None:
        try:
            parsed = yaml.safe_load(content)
        except yaml.YAMLError as exc:
            raise ValueError(f"llama-swap config is not valid YAML: {exc}") from exc
        if not isinstance(parsed, dict):
            raise ValueError("llama-swap config must be a YAML mapping")

    async def _write_config_unlocked(
        self,
        content: Optional[str] = None,
        sidecars: Optional[Dict[str, dict]] = None,
    ) -> None:
        if content is None or sidecars is None:
            content, sidecars = await self._compose_config()
        else:
            self._validate_swap_yaml(content)
        await asyncio.to_thread(self._publish_config_files, content, sidecars)
        logger.debug("Successfully wrote config to %s", self.config_path)

    def _read_config_text(self) -> str:
        if not os.path.exists(self.config_path):
            return ""
        try:
            with open(self.config_path, "r", encoding="utf-8") as handle:
                return handle.read()
        except OSError:
            return ""

    def _known_sidecar_revisions(self) -> set[int]:
        """Revisions already used on disk or by the working YAML, including after a restart."""
        found = {self.config_revision}
        for match in _SIDECAR_REVISION.finditer(self._read_config_text()):
            found.add(int(match.group(1)))
        revision_path = self.config_path + ".revision"
        try:
            with open(revision_path, "r", encoding="utf-8") as handle:
                found.add(int(handle.read().strip()))
        except (OSError, ValueError):
            pass
        root = self._sidecar_root()
        if root and os.path.isdir(root):
            for name in os.listdir(root):
                match = _SIDECAR_REVISION.search(name)
                if match:
                    found.add(int(match.group(1)))
        return found

    def _sidecar_root(self) -> str:
        try:
            from backend.audio_cpp_manager import get_audio_cpp_manager

            return os.path.realpath(get_audio_cpp_manager().server_configs_dir)
        except Exception:
            return ""

    def _allocate_revision(self) -> int:
        revision = max(self._known_sidecar_revisions()) + 1
        self.config_revision = revision
        return revision

    def _persist_revision(self, revision: int) -> None:
        path = self.config_path + ".revision"
        directory = os.path.dirname(path) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".llama-swap-config.", suffix=".revision", dir=directory
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(f"{revision}\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
            tmp_path = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    def _publish_config_files(self, content: str, sidecars: Dict[str, dict]) -> None:
        """Stage sidecars, then atomically replace the YAML without unlinking it first."""
        revision = self._allocate_revision()
        staged: List[str] = []
        rewritten = content
        published = False
        tmp_path = ""
        try:
            for raw_path, payload in sidecars.items():
                generation_path = self._stage_sidecar_generation(
                    raw_path, payload, revision
                )
                if raw_path not in rewritten:
                    raise ValueError(
                        f"Generated config does not reference sidecar {raw_path}"
                    )
                rewritten = rewritten.replace(raw_path, generation_path)
                staged.append(generation_path)
            config_dir = os.path.dirname(self.config_path) or "."
            os.makedirs(config_dir, exist_ok=True)
            self._previous_config_text = self._read_config_text()
            if self._previous_config_text:
                previous_path = self.config_path + ".prev"
                with open(previous_path, "w", encoding="utf-8") as handle:
                    handle.write(self._previous_config_text)
                    handle.flush()
                    os.fsync(handle.fileno())
            fd, tmp_path = tempfile.mkstemp(
                prefix=".llama-swap-config.", suffix=".tmp", dir=config_dir
            )
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(rewritten)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.config_path)
            tmp_path = ""
            published = True
            self._persist_revision(revision)
            from backend.ops_metrics import set_config_revision

            set_config_revision(revision)
            logger.info(
                "published llama-swap config",
                extra={"config_revision": revision},
            )
        except Exception:
            if not published:
                for path in staged:
                    try:
                        os.remove(path)
                    except OSError:
                        pass
            raise
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    def _stage_sidecar_generation(
        self, raw_path: str, payload: dict, revision: int
    ) -> str:
        from backend.audio_cpp_manager import get_audio_cpp_manager

        root = os.path.realpath(get_audio_cpp_manager().server_configs_dir)
        os.makedirs(root, exist_ok=True)
        path = os.path.realpath(raw_path)
        if (
            os.path.commonpath([root, path]) != root
            or os.path.dirname(path) != root
            or not path.endswith(".json")
        ):
            raise ValueError(f"Invalid audio.cpp sidecar path: {raw_path}")
        stem, ext = os.path.splitext(os.path.basename(path))
        generation_path = os.path.join(root, f"{stem}.r{revision}{ext}")
        if os.path.lexists(generation_path):
            raise FileExistsError(
                f"Refusing to replace sidecar still on disk: {generation_path}"
            )
        fd, temp_path = tempfile.mkstemp(
            prefix=f".{stem}.", suffix=".tmp", dir=root
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, generation_path)
            temp_path = ""
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        return generation_path

    def restore_previous_config(self) -> None:
        """Put the last published YAML back. Sidecars from that generation are left in place."""
        previous = self._previous_config_text
        if previous == "" and os.path.exists(self.config_path + ".prev"):
            with open(self.config_path + ".prev", "r", encoding="utf-8") as handle:
                previous = handle.read()
        if previous == "" and not os.path.exists(self.config_path + ".prev"):
            return
        config_dir = os.path.dirname(self.config_path) or "."
        os.makedirs(config_dir, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=".llama-swap-config.", suffix=".restore", dir=config_dir
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(previous)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, self.config_path)
            tmp_path = ""
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    async def sync_running_models(self):
        """Sync running_models with actual state from llama-swap"""
        client = self._client()
        try:
            running_models_data = await client.get_running_models()

            # Clear current running_models
            self.running_models.clear()

            # The response format is {"running": [{"model": "...", "state": "..."}]}
            if (
                isinstance(running_models_data, dict)
                and "running" in running_models_data
            ):
                running_list = running_models_data["running"]
            else:
                running_list = running_models_data

            # Populate with actual running models from llama-swap
            for model_data in running_list:
                if isinstance(model_data, dict):
                    proxy_model_name = model_data.get("model", "")
                    if proxy_model_name:
                        # We don't need to store the full config here since it's in the database
                        self.running_models[proxy_model_name] = {
                            "model_path": "",  # Will be loaded from database when needed
                            "config": {},  # Will be loaded from database when needed
                        }

            logger.info(
                f"Synced running_models with llama-swap state: {len(self.running_models)} models"
            )

        except Exception as e:
            logger.warning(f"Failed to sync running_models with llama-swap: {e}")
            # Keep existing running_models if sync fails

    async def start_proxy(self):
        if self.process and self.process.poll() is None:
            logger.info("llama-swap is already running")
            return

        await self._do_start_proxy()

        # Only start monitoring task on first start, not on restart
        if self.monitor_task is None or self.monitor_task.done():
            self.monitor_task = asyncio.create_task(self._monitor_process())

        # Wait for llama-swap to become ready
        await self._wait_for_proxy_ready()

    async def _ensure_config_file_for_proxy(self) -> None:
        """
        If the config file is missing or empty, write a minimal stub so llama-swap can start.
        Full YAML from the database is only written when the user applies configuration.
        """
        config_dir = os.path.dirname(self.config_path)
        os.makedirs(config_dir, exist_ok=True)
        if os.path.exists(self.config_path):
            try:
                if os.path.getsize(self.config_path) > 0:
                    return
            except OSError:
                pass
        content = (
            "healthCheckTimeout: 1200\n"
            'logTimeFormat: "2006-01-02 15:04:05"\n'
            "sendLoadingState: true\n"
            "includeAliasesInList: true\n"
            "models: {}\n"
        )
        tmp = os.path.join(config_dir, f".llama-swap-config.stub.tmp.{os.getpid()}")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.write(content)
            os.replace(tmp, self.config_path)
        except OSError as exc:
            logger.error("Failed to write minimal llama-swap config: %s", exc)
            raise
        logger.info(
            "Wrote minimal llama-swap config (empty models). "
            "Use Apply configuration in the UI to generate from the database."
        )

    async def _do_start_proxy(self):
        """Internal method to actually start the process"""
        await self._ensure_config_file_for_proxy()

        from backend.access_policy import proxy_listen_address

        cmd = [
            "llama-swap",
            "--config",
            self.config_path,
            "--listen",
            proxy_listen_address(self.proxy_port),
            "--watch-config",
        ]

        # Get CUDA environment variables and merge with current environment
        env = os.environ.copy()
        # CUDA_VISIBLE_DEVICES=all is not valid CUDA syntax (NVIDIA expects indices/UUIDs).
        # Older images set it by mistake; drop it so unset means "all devices".
        cvd = str(env.get("CUDA_VISIBLE_DEVICES", "")).strip().lower()
        if cvd in ("all", "*"):
            env.pop("CUDA_VISIBLE_DEVICES", None)
        try:
            from backend.cuda_installer import get_cuda_installer

            cuda_installer = get_cuda_installer()
            cuda_env = cuda_installer.get_cuda_env()
            if cuda_env:
                env.update(cuda_env)
                logger.debug(
                    f"Added CUDA environment variables to llama-swap process: {list(cuda_env.keys())}"
                )
        except Exception as e:
            logger.warning(f"Failed to get CUDA environment variables: {e}")

        # Docker uses /app; on Windows dev that path usually does not exist and Popen cwd would fail.
        swap_cwd = "/app" if os.path.isdir("/app") else None
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,
            cwd=swap_cwd,
            env=env,
        )

        # Start background task to stream llama-swap logs
        asyncio.create_task(self._stream_llama_swap_logs())

    async def _stream_llama_swap_logs(self):
        """Stream llama-swap stdout/stderr to our logger"""
        if not self.process or not self.process.stdout:
            return

        async def read_loop():
            try:
                while self.process and self.process.poll() is None:
                    # Read a line asynchronously
                    line = await asyncio.to_thread(self.process.stdout.readline)
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        self._observe_proxy_log(line)
                        logger.debug(f"[llama-swap] {line}")
            except Exception as e:
                logger.debug(f"Stopped reading llama-swap logs: {e}")

        # Start the reading loop
        asyncio.create_task(read_loop())

    async def _monitor_process(self):
        """Monitor llama-swap process and restart if it dies"""
        try:
            while self._should_restart:
                if self.process:
                    # Check if process is still alive
                    poll_result = self.process.poll()
                    if poll_result is not None:
                        # Process has terminated
                        exit_code = poll_result
                        logger.warning(
                            f"llama-swap process died with exit code {exit_code}"
                        )

                        if self._should_restart:
                            logger.info("Attempting to restart llama-swap...")
                            try:
                                # Clear the dead process
                                self.process = None

                                # Restart it using internal method to avoid re-creating monitor task
                                await self._do_start_proxy()

                                # Wait for it to become ready
                                await self._wait_for_proxy_ready()

                                logger.info("llama-swap restarted successfully")
                            except Exception as e:
                                logger.error(f"Failed to restart llama-swap: {e}")
                                # Wait before retrying
                                await asyncio.sleep(5)

                # Check every 2 seconds
                await asyncio.sleep(2)

        except asyncio.CancelledError:
            logger.debug("Monitor task cancelled")
        except Exception as e:
            logger.error(f"Error in monitor task: {e}")

    async def _wait_for_proxy_ready(self, timeout: int = 30):
        """Waits until the llama-swap proxy is responsive."""
        client = httpx.AsyncClient(timeout=httpx.Timeout(1.0))
        start_time = asyncio.get_event_loop().time()
        try:
            while asyncio.get_event_loop().time() - start_time < timeout:
                try:
                    response = await client.get(f"{self.proxy_url}/health", timeout=1)
                    if response.status_code == 200:
                        return
                except httpx.ConnectError:
                    pass
                await asyncio.sleep(0.5)
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
        raise Exception("llama-swap proxy did not become ready in time.")

    async def stop_proxy(self):
        """Stops the llama-swap proxy server and all managed models."""
        # Disable auto-restart
        self._should_restart = False

        # Stop the monitor task
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass

        if self.process:
            logger.info("Stopping llama-swap proxy...")
            self.process.terminate()
            try:
                await asyncio.wait_for(asyncio.to_thread(self.process.wait), timeout=10)
                logger.info("llama-swap proxy stopped gracefully")
            except asyncio.TimeoutError:
                logger.warning(
                    "llama-swap did not terminate gracefully, killing process"
                )
                self.process.kill()
                await asyncio.to_thread(self.process.wait)
            self.process = None
            self.running_models = {}  # Clear registered models
        else:
            logger.info("llama-swap is not running")

    async def restart_proxy(self):
        """Restarts the llama-swap proxy server to pick up new environment variables (e.g., after CUDA installation)."""
        logger.info("Restarting llama-swap proxy to pick up new environment...")
        was_running = self.process is not None and self.process.poll() is None

        if was_running:
            # Temporarily disable auto-restart to prevent the monitor from interfering
            original_should_restart = self._should_restart
            self._should_restart = False

            # Stop the proxy
            await self.stop_proxy()

            # Re-enable auto-restart
            self._should_restart = original_should_restart

        # Start the proxy (will use new environment variables)
        await self.start_proxy()
        logger.info("llama-swap proxy restarted successfully")

    async def register_model(self, model: Any, config: Dict[str, Any]) -> str:
        """
        Registers a model with llama-swap by storing its configuration.
        Returns the proxy_model_name used by llama-swap.
        model can be a dict or an object with proxy_name, file_path, display_name/name.
        """
        proxy_name = (
            model.get("proxy_name")
            if isinstance(model, dict)
            else getattr(model, "proxy_name", None)
        )
        file_path = (
            model.get("file_path")
            if isinstance(model, dict)
            else getattr(model, "file_path", None)
        )
        name = (
            model.get("display_name") or model.get("name")
            if isinstance(model, dict)
            else (getattr(model, "display_name", None) or getattr(model, "name", None))
        )

        if not proxy_name:
            raise ValueError(f"Model '{name}' does not have a proxy_name set")

        if proxy_name in self.running_models:
            raise ValueError(
                f"Model '{proxy_name}' is already registered with llama-swap."
            )

        self.running_models[proxy_name] = {
            "model_path": file_path,
            "config": config,
        }

        logger.info(
            f"Model '{name}' registered in memory as '{proxy_name}' with llama-swap "
            "(config file is only updated when the user applies configuration)"
        )
        return proxy_name

    def _detect_correct_binary_path(self, version_dir: str) -> str:
        """
        Automatically detects the correct binary path for llama-server.
        Prioritizes llama-server over server binary for better compatibility.
        """
        import os

        # Priority order: llama-server first (newer, works better), then server (older)
        possible_paths = [
            os.path.join(
                version_dir, "build", "bin", "llama-server"
            ),  # New location (preferred)
            os.path.join(version_dir, "bin", "llama-server"),  # Alternative location
            os.path.join(version_dir, "llama-server"),  # Direct location
            os.path.join(version_dir, "server"),  # Old location (fallback)
        ]

        for path in possible_paths:
            if os.path.exists(path) and os.access(path, os.X_OK):
                logger.info(f"Found executable llama-server at: {path}")
                return path

        # If no executable found, return the most likely path (new location)
        logger.warning(
            f"No executable llama-server found in {version_dir}, using default path"
        )
        return os.path.join(version_dir, "build", "bin", "llama-server")

    async def _ensure_correct_binary_path(self):
        """
        Ensures the active llama-cpp version has the correct binary path.
        Automatically detects and updates if needed.
        """
        store = get_store()
        for engine in ("llama_cpp", "ik_llama"):
            active_version = store.get_active_engine_version(engine)
            if not active_version:
                continue
            version_dir = active_version.get("binary_path")
            if not version_dir:
                continue
            if not os.path.isabs(version_dir):
                version_dir = os.path.join("/app", version_dir)
            binary_dir = os.path.dirname(version_dir)
            correct_binary_path = self._detect_correct_binary_path(binary_dir)
            relative_path = os.path.relpath(correct_binary_path, "/app")
            if active_version.get("binary_path") != relative_path:
                logger.info(
                    f"Updating binary path from '{active_version.get('binary_path')}' to '{relative_path}'"
                )
                store.update_engine_version(
                    engine,
                    str(active_version.get("version")),
                    {"binary_path": relative_path},
                )
                logger.info("Binary path updated successfully")
            else:
                logger.debug(f"Binary path is already correct: {relative_path}")
            return
        logger.warning("No active llama-cpp version found")

    async def regenerate_config_with_active_version(self, *, sync_running: bool = True):
        """
        Regenerate llama-swap YAML from the DB (per-model engine binaries).
        Syncs ``running_models`` with llama-swap, fixes active binary paths when possible,
        writes config, then tries to start the proxy.
        """
        async with self._apply_lock:
            epoch = self._stale_epoch
            await self._regenerate_unlocked(sync_running=sync_running, require_proxy=False)
            if self._stale_epoch == epoch:
                self.clear_swap_config_stale()

    async def _regenerate_unlocked(
        self, *, sync_running: bool, require_proxy: bool
    ) -> None:
        await self._ensure_correct_binary_path()

        from backend.llama_swap_config import any_active_runtime_in_db

        if not any_active_runtime_in_db():
            logger.warning(
                "No active registered runtime on disk, skipping config regeneration"
            )
            return

        if sync_running:
            await self.sync_running_models()
        await self._write_config_unlocked()
        logger.info(
            "Regenerated llama-swap config (%s running models)",
            len(self.running_models),
        )
        try:
            await self.start_proxy()
            logger.info("Ensured llama-swap is running after config regeneration")
        except Exception as exc:
            if require_proxy:
                raise RuntimeError(
                    f"llama-swap did not accept the published configuration: {exc}"
                ) from exc
            logger.warning(
                "Failed to start llama-swap after config regeneration: %s", exc
            )

    async def _unload_before_apply(self) -> None:
        """Unload models before publish. A down proxy is not fatal; a refusal is."""
        try:
            await self._client().unload_all_models()
            logger.info("Stopped all running models before applying llama-swap config")
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in (404, 405):
                logger.info("Unload endpoint missing (%s); continuing apply", status)
                return
            raise RuntimeError(f"Proxy refused to unload models: {exc}") from exc
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning(
                "Proxy unreachable before apply; publishing the candidate anyway: %s",
                exc,
            )

    def _arm_reload_watch(self) -> None:
        """Accept the next publish only after the running proxy logs a successful reload."""
        self._reload_outcome = None
        self._reload_watch = True

    def _observe_proxy_log(self, line: str) -> None:
        if not self._reload_watch or self._reload_outcome:
            return
        if "failed to reload config" in line:
            self._reload_outcome = "rejected"
        elif "configuration reloaded" in line:
            self._reload_outcome = "reloaded"

    async def _confirm_proxy_accepted(self) -> None:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("llama-swap is not running after configuration publish")
        if self._reload_watch:
            deadline = asyncio.get_event_loop().time() + self._reload_confirm_timeout
            while self._reload_outcome is None and asyncio.get_event_loop().time() < deadline:
                if self.process.poll() is not None:
                    self._reload_watch = False
                    raise RuntimeError("llama-swap exited while reloading configuration")
                await asyncio.sleep(0.05)
            outcome = self._reload_outcome
            self._reload_watch = False
            if outcome == "rejected":
                raise RuntimeError(
                    "llama-swap rejected the candidate and kept the previous configuration"
                )
            if outcome != "reloaded":
                raise RuntimeError(
                    "llama-swap did not confirm the published configuration"
                )
        health = await self._client().check_health()
        if not isinstance(health, dict) or not health.get("healthy"):
            raise RuntimeError("llama-swap health check failed after configuration publish")

    async def unregister_model(self, proxy_model_name: str):
        """
        Unregisters a model from llama-swap by unloading the specific model.
        Works for both app-registered and externally loaded models.
        """
        try:
            logger.info(
                f"unregister_model called with proxy_model_name: {proxy_model_name}"
            )
            logger.info(f"Starting unregister process for model '{proxy_model_name}'")

            # Unload the specific model from llama-swap (works regardless of how it was loaded)
            client = self._client()
            try:
                logger.info(f"Calling unload_model for '{proxy_model_name}'...")
                result = await client.unload_model(proxy_model_name)
                logger.info(
                    f"Unloaded model '{proxy_model_name}' from llama-swap, result: {result}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to unload model '{proxy_model_name}' from llama-swap: {e}"
                )
                raise

            # Remove the model from running_models if it exists there
            if proxy_model_name in self.running_models:
                del self.running_models[proxy_model_name]
                logger.info(f"Removed '{proxy_model_name}' from running_models")
            else:
                logger.info(
                    f"Model '{proxy_model_name}' was not in running_models (loaded externally)"
                )

            # Sync with actual llama-swap state to ensure consistency
            await self.sync_running_models()

            logger.info(f"Model '{proxy_model_name}' unregistered from llama-swap")

        except Exception as e:
            logger.error(f"Error in unregister_model: {e}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    async def compute_desired_config_content(self) -> Optional[str]:
        """
        YAML that would be written after sync. ``None`` when no registered runtime is active.
        """
        from backend.llama_swap_config import any_active_runtime_in_db

        await self._ensure_correct_binary_path()
        if not any_active_runtime_in_db():
            return None
        await self.sync_running_models()
        store = get_store()
        all_models = store.list_models()
        return generate_llama_swap_config(self.running_models, all_models)

    async def get_config_pending_state(self) -> Dict[str, Any]:
        """Compare on-disk llama-swap config to freshly generated YAML."""
        try:
            desired = await self.compute_desired_config_content()
        except Exception as exc:
            logger.warning("compute_desired_config_content failed: %s", exc)
            return {
                "applicable": False,
                "pending": False,
                "changes": [],
                "reason": f"Could not compute desired config: {exc}",
            }

        if desired is None:
            return {
                "applicable": False,
                "pending": False,
                "changes": [],
                "reason": "No active registered engine has a runnable inference server.",
            }

        disk_raw = ""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as fh:
                    disk_raw = fh.read()
            except OSError as exc:
                logger.warning("Could not read llama-swap config: %s", exc)
                disk_raw = ""

        if _configs_semantically_equal(disk_raw, desired):
            self.clear_swap_config_stale()
            return {"applicable": True, "pending": False, "changes": []}

        changes = summarize_llama_swap_yaml_diff(disk_raw, desired)
        return {"applicable": True, "pending": True, "changes": changes}

    async def user_apply_regenerate_config(self) -> None:
        """Render and validate a candidate, then publish it without dropping the last good file."""
        async with self._apply_lock:
            epoch = self._stale_epoch
            previous_running = self.running_models
            self.running_models = {}
            try:
                content, sidecars = await self._compose_config()
            finally:
                self.running_models = previous_running
            await self._unload_before_apply()
            self.running_models.clear()
            previous = self._read_config_text()
            self._previous_config_text = previous
            already_running = self.process is not None and self.process.poll() is None
            if already_running:
                self._arm_reload_watch()
            try:
                await self._write_config_unlocked(content, sidecars)
                await self._regenerate_start_only(require_proxy=True)
                await self._confirm_proxy_accepted()
            except Exception:
                self._reload_watch = False
                self.restore_previous_config()
                raise
            if self._stale_epoch == epoch:
                self.clear_swap_config_stale()
            else:
                logger.info(
                    "llama-swap apply finished with a newer edit still pending"
                )

    async def _regenerate_start_only(self, *, require_proxy: bool) -> None:
        try:
            await self.start_proxy()
        except Exception as exc:
            if require_proxy:
                raise RuntimeError(
                    f"llama-swap did not accept the published configuration: {exc}"
                ) from exc
            logger.warning("Failed to start llama-swap after config regeneration: %s", exc)
