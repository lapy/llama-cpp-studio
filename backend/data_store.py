"""YAML-backed data store replacing SQLite.

Each document is replaced on its own. ``settings.yaml``, ``models.yaml``,
``model_config_templates.yaml``, and ``operations.yaml`` do not share a
commit. A crash or error between two of those writes can leave one document
new and another old. That is not an atomic restore. Multi-document restore
has to record its own recovery state; a sequence of these writes is not a
transaction. ``backend/config_backup.py`` is that protocol: a journal is
replaced before any configuration document, and startup returns every
document to the pre-import snapshot unless all of them already match the
completed restore.

``os.replace`` is atomic replacement of one file: a reader sees the complete
previous document or the complete new one, never a torn mix of the two.
Syncing the temporary file before that replace makes the new bytes
crash-durable before the name changes. Syncing the directory afterward makes
the directory entry crash-durable. Those fsyncs are not the same guarantee as
the replace. A failed directory sync does not mean the replacement is unknown:
this process already sees the new file, and a later crash might still lose the
unsynced directory entry.

A save response uses ``committed`` for replacement only. ``True`` means this
process observed the replace return. ``False`` means it observed that the
replace did not happen. ``"unknown"`` means it cannot establish which, and the
caller must read the document again before repeating a side effect. Restore
can depend on those three outcomes and on the per-document boundary above. It
cannot treat several of these writes as one commit.
"""

import copy
import os
import re
import shutil
import tempfile
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import yaml

try:
    import fcntl
except ImportError:  # pragma: no cover - non-Unix
    fcntl = None

from backend.engines.registry import ENGINE_REGISTRY
from backend.store_io import StoreDurabilityError
from backend.logging_config import get_logger
from backend.models.config import (
    effective_model_config,
    normalize_model_config,
)
from backend.models.schema import compatible_engines_for_record, normalize_model_record
from backend.paths import studio_data_dir
from backend.utils.coercion import coerce_json_dict

logger = get_logger(__name__)
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_YAML_SAFE_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_MODELS_DOCUMENT_SCHEMA = 3
_write_checkpoint: Optional[Callable[[str, str], None]] = None


def set_write_checkpoint(hook: Optional[Callable[[str, str], None]]) -> None:
    """Install a test hook invoked at durability boundaries.

    The hook receives ``(name, path)``. Names are ``temp_opened``,
    ``before_temp_write``, ``before_fsync``, ``after_fsync``,
    ``before_replace``, ``after_replace``, and ``after_acknowledge``.
    Raising ``OSError`` is reported as a durability failure for that phase.
    """
    global _write_checkpoint
    _write_checkpoint = hook


def _reach_write_checkpoint(name: str, path: str) -> None:
    hook = _write_checkpoint
    if hook is not None:
        hook(name, path)


def _discard_temp_write(tmp_path: str) -> None:
    if tmp_path and os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except OSError:
            pass


class StorageCorruptionError(RuntimeError):
    """Raised when a config document is unreadable and must not be overwritten."""


class DuplicateIdentifierError(ValueError):
    """Raised when a model or engine version identifier is already stored."""


class _SkipWrite(Exception):
    """Abort a mutation without replacing the document."""

    def __init__(self, result: Any = None):
        self.result = result


def _get_config_dir() -> str:
    """Return config directory (Docker: /app/data/config, local: data/config)."""
    return os.path.join(studio_data_dir(), "config")


def generate_proxy_name(huggingface_id: str, quantization: Optional[str] = None) -> str:
    """
    Generate a proxy name for llama-swap using HuggingFace ID and optional quantization.
    """
    huggingface_slug = (
        huggingface_id.replace("/", "-").replace(" ", "-").replace(".", "-").lower()
    )
    if quantization:
        quantization_slug = quantization.replace(" ", "-").lower()
        return f"{huggingface_slug}.{quantization_slug}"
    return huggingface_slug


def _coerce_config(config_value: Optional[Any]) -> Dict[str, Any]:
    return coerce_json_dict(config_value, copy=False)


def _model_value(model: Any, key: str, default: Any = None) -> Any:
    if isinstance(model, dict):
        return model.get(key, default)
    return getattr(model, key, default)


def normalize_proxy_alias(alias: Optional[str]) -> str:
    """Normalize a user-provided model alias into a safe exposed engine ID."""
    if alias is None:
        return ""

    normalized = str(alias).strip().lower()
    if not normalized:
        return ""

    normalized = normalized.replace("/", "-").replace("\\", "-")
    normalized = re.sub(r"\s+", "-", normalized)
    normalized = re.sub(r"[^a-z0-9._-]", "-", normalized)
    normalized = re.sub(r"-{2,}", "-", normalized)
    normalized = normalized.strip("._-")
    return normalized


def resolve_llama_swap_id(model: Any) -> str:
    """
    Stable llama-swap YAML model key for a catalog row.

    Never derived from ``model_alias`` so alias changes do not remap running state
    or collide across models.
    """
    existing = normalize_proxy_alias(_model_value(model, "proxy_name"))
    if existing:
        return existing

    source = _model_value(model, "source", {})
    source_id = source.get("id") if isinstance(source, dict) else None
    identity = (
        _model_value(model, "huggingface_id")
        or source_id
        or _model_value(model, "id", "")
    )
    return generate_proxy_name(
        identity,
        _model_value(model, "quantization"),
    )


def resolve_proxy_name(model: Any) -> str:
    """Stable llama-swap model id (YAML ``models`` key). Same as :func:`resolve_llama_swap_id`."""
    return resolve_llama_swap_id(model)


def _effective_config_for_model(model: Any) -> Dict[str, Any]:
    raw = _coerce_config(_model_value(model, "config"))
    return effective_model_config(normalize_model_config(raw))


def resolve_routing_name(
    model: Any, config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Preferred id clients send in API ``model`` requests.

    Uses ``model_alias`` when set, otherwise the stable llama-swap id.
    """
    stable = resolve_llama_swap_id(model)
    if config is None:
        config = _effective_config_for_model(model)
    alias = normalize_proxy_alias(config.get("model_alias"))
    return alias or stable


def normalize_swap_aliases(raw: Any) -> List[str]:
    """Normalize ``swap_aliases`` list from per-engine config."""
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        alias = normalize_proxy_alias(item if isinstance(item, str) else str(item))
        if not alias or alias in seen:
            continue
        seen.add(alias)
        out.append(alias)
    return out


def collect_config_swap_aliases(
    config: Dict[str, Any], stable_id: str
) -> List[str]:
    """
    Extra llama-swap ``aliases`` (excluding the stable YAML key).

    Includes primary ``model_alias`` and ``swap_aliases`` when they differ from
    ``stable_id``.
    """
    stable = normalize_proxy_alias(stable_id)
    routing = resolve_routing_name_from_config(config, stable or stable_id)
    aliases: List[str] = []
    seen = {stable} if stable else set()
    if routing and routing not in seen:
        aliases.append(routing)
        seen.add(routing)
    for alias in normalize_swap_aliases(config.get("swap_aliases")):
        if alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return aliases


def resolve_routing_name_from_config(
    config: Dict[str, Any], stable_id: str
) -> str:
    alias = normalize_proxy_alias(config.get("model_alias"))
    stable = normalize_proxy_alias(stable_id)
    return alias or stable or ""


def collect_claimed_swap_names(
    model: Any, config: Optional[Dict[str, Any]] = None
) -> set[str]:
    """All swap names this model registers (stable id, routing id, aliases, sub-ids)."""
    if config is None:
        config = _effective_config_for_model(model)
    stable = resolve_llama_swap_id(model)
    routing = resolve_routing_name_from_config(config, stable)
    names = {stable}
    if routing:
        names.add(routing)
    names.update(collect_config_swap_aliases(config, stable))
    raw_variants = config.get("set_params_by_id")
    if isinstance(raw_variants, list):
        for item in raw_variants:
            if not isinstance(item, dict):
                continue
            sub_id = str(item.get("sub_id", "")).strip()
            if not sub_id:
                continue
            names.add(f"{routing}:{sub_id}")
    return names


def find_swap_name_conflicts(
    store: "DataStore",
    model_id: str,
    config: Dict[str, Any],
) -> List[str]:
    """Return swap names claimed by *model_id* that another catalog row already uses."""
    model = store.get_model(model_id)
    if not model:
        return []
    claimed = collect_claimed_swap_names(model, config)
    conflicts: List[str] = []
    for other in store.list_models():
        other_id = other.get("id")
        if not other_id or other_id == model_id:
            continue
        other_config = _effective_config_for_model(other)
        other_names = collect_claimed_swap_names(other, other_config)
        for name in sorted(claimed):
            if name in other_names:
                conflicts.append(name)
    return conflicts


class DataStore:
    """Thread-safe YAML-backed data store replacing SQLite."""

    def __init__(self, config_dir: Optional[str] = None):
        self._config_dir = os.path.abspath(config_dir or _get_config_dir())
        self._lock = threading.RLock()
        self._ipc_depth = 0
        self._lock_fd: Optional[int] = None
        # path -> (stamp, document). The stamp is the revision whose bytes were
        # read, not a stat taken after the read returned.
        self._doc_cache: Dict[str, tuple[tuple[int, int, int], dict]] = {}
        self._ensure_files_exist()

    def _default_document(self, filename: str) -> dict:
        if filename == "models.yaml":
            return {"schema_version": _MODELS_DOCUMENT_SCHEMA, "models": []}
        if filename == "engines.yaml":
            engine_defaults = {
                engine_id: {"active_version": None, "versions": []}
                for engine_id in ENGINE_REGISTRY
            }
            engine_defaults["cuda"] = {
                "installed_version": None,
                "install_path": None,
            }
            return engine_defaults
        if filename == "settings.yaml":
            return {"huggingface_token": "", "proxy_port": 2000}
        if filename == "engine_params_catalog.yaml":
            return {"schema_version": 2, "engines": {}}
        if filename == "model_config_templates.yaml":
            return {"templates": []}
        if filename == "llama_swap_routing.yaml":
            return {"profiles": {}, "selectors": {}}
        if filename == "operations.yaml":
            return {"schema_version": 1, "operations": []}
        if filename == "config_restore.yaml":
            return {"schema_version": 1, "phase": "idle"}
        return {}

    def _ensure_files_exist(self) -> None:
        """Create config dir and default YAML files if they don't exist."""
        os.makedirs(self._config_dir, exist_ok=True)
        for filename in (
            "models.yaml",
            "engines.yaml",
            "settings.yaml",
            "engine_params_catalog.yaml",
            "model_config_templates.yaml",
            "llama_swap_routing.yaml",
        ):
            path = os.path.join(self._config_dir, filename)
            if not os.path.exists(path):
                self._write_yaml(path, self._default_document(filename))

    def _ipc_enter(self) -> None:
        if fcntl is None:
            return
        if self._ipc_depth == 0:
            if self._lock_fd is None:
                lock_path = os.path.join(self._config_dir, ".store.lock")
                self._lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
        self._ipc_depth += 1

    def _ipc_exit(self) -> None:
        if fcntl is None:
            return
        self._ipc_depth = max(0, self._ipc_depth - 1)
        if self._ipc_depth == 0 and self._lock_fd is not None:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _load_document(self, path: str) -> tuple[str, Optional[dict]]:
        """Return ``(absent|ok|corrupt, data)``. Absent is not the same as corrupt."""
        if not os.path.exists(path):
            return "absent", None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = yaml.load(handle, Loader=_YAML_SAFE_LOADER)
        except Exception as exc:
            logger.error("Refusing to use unreadable YAML %s: %s", path, exc)
            return "corrupt", None
        if loaded is None:
            return "ok", {}
        if not isinstance(loaded, dict):
            logger.error("Refusing to use non-mapping YAML %s", path)
            return "corrupt", None
        return "ok", loaded

    def _preserve_corrupt_copy(self, path: str) -> None:
        """Keep a timestamped copy so a bad document can be recovered by hand."""
        stamp = time.strftime("%Y%m%d%H%M%S")
        dest = f"{path}.corrupt-{stamp}"
        try:
            shutil.copy2(path, dest)
            os.chmod(dest, 0o600)
        except OSError as exc:
            logger.error("Could not preserve corrupt YAML %s: %s", path, exc)

    def _file_stamp(self, path: str) -> Optional[tuple[int, int, int]]:
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return stat.st_mtime_ns, stat.st_size, stat.st_ino

    def _cached_document(self, path: str) -> Optional[dict]:
        stamp = self._file_stamp(path)
        cached = self._doc_cache.get(path)
        if stamp is None or cached is None or cached[0] != stamp:
            return None
        return cached[1]

    def _remember_document(
        self,
        path: str,
        data: dict,
        stamp: Optional[tuple[int, int, int]],
    ) -> None:
        """Cache ``data`` only with the revision those bytes were read from."""
        if stamp is None:
            self._doc_cache.pop(path, None)
            return
        self._doc_cache[path] = (stamp, copy.deepcopy(data))

    def _load_consistent_document(
        self, path: str
    ) -> tuple[str, Optional[dict], Optional[tuple[int, int, int]], bool]:
        """Read bytes and the file revision that produced them.

        A writer can replace the file after the read and before a later stat.
        Caching that pair would let the next locked mutation write the stale
        bytes back. Retry until the stamp is unchanged across the read. The
        last result is returned unbound when the file keeps changing, and
        callers must not cache it.
        """
        status, data = "absent", None
        for _ in range(5):
            before = self._file_stamp(path)
            status, data = self._load_document(path)
            after = self._file_stamp(path)
            if before == after:
                return status, data, after, True
        return status, data, None, False

    def _forget_document(self, path: str) -> None:
        self._doc_cache.pop(path, None)

    def exclusive_documents(self):
        """Hold the store locks across a multi-document read or restore."""
        return _ExclusiveDocuments(self)

    def document_revision(self, filename: str) -> str:
        """Revision of one document. ``absent`` when the file does not exist."""
        stamp = self._file_stamp(os.path.join(self._config_dir, filename))
        if stamp is None:
            return "absent"
        return f"{stamp[0]}:{stamp[1]}:{stamp[2]}"

    def write_document(self, filename: str, data: dict, *, require_directory_sync: bool = False) -> None:
        """Replace one document. Callers inside ``exclusive_documents`` stay serialized.

        ``require_directory_sync`` makes the directory entry part of the write.
        Mode ``0600`` is only the permission on the replaced file.
        """
        path = os.path.join(self._config_dir, filename)
        payload = copy.deepcopy(data)
        with self._lock:
            self._ipc_enter()
            try:
                self._validate_document(filename, payload)
                self._write_yaml(path, payload, require_directory_sync=require_directory_sync)
                self._remember_document(path, payload, self._file_stamp(path))
            finally:
                self._ipc_exit()

    def _read_yaml(self, filename: str) -> dict:
        """Read a YAML document. Missing files are empty; corrupt files raise."""
        path = os.path.join(self._config_dir, filename)
        with self._lock:
            cached = self._cached_document(path)
            if cached is not None:
                return copy.deepcopy(cached)
            status, data, stamp, confirmed = self._load_consistent_document(path)
            if status == "absent":
                self._forget_document(path)
                return {}
            if status == "corrupt" or data is None:
                self._forget_document(path)
                raise StorageCorruptionError(
                    f"{filename} is corrupt or unreadable; mutation and silent reset are refused"
                )
            if confirmed and stamp is not None:
                self._remember_document(path, data, stamp)
            else:
                self._forget_document(path)
            return copy.deepcopy(data)

    def _write_yaml(self, path: str, data: dict, *, require_directory_sync: bool = False) -> None:
        """Atomic write via a unique temp file. The previous file is left in place until replace.

        This replaces one document. It does not make a later write of another
        document atomic with this one.
        """
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            dir=directory,
        )
        phase = "temp_write"
        committed = False
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                _reach_write_checkpoint("temp_opened", path)
                _reach_write_checkpoint("before_temp_write", path)
                yaml.dump(
                    data,
                    handle,
                    Dumper=_YAML_SAFE_DUMPER,
                    default_flow_style=False,
                    sort_keys=False,
                )
                handle.flush()
                phase = "fsync"
                _reach_write_checkpoint("before_fsync", path)
                os.fsync(handle.fileno())
                _reach_write_checkpoint("after_fsync", path)
            if os.path.exists(path):
                backup = path + ".bak"
                shutil.copy2(path, backup)
                if os.path.basename(path) in {"settings.yaml", "config_restore.yaml"}:
                    os.chmod(backup, 0o600)
            phase = "replace"
            _reach_write_checkpoint("before_replace", path)
            os.replace(tmp_path, path)
            tmp_path = ""
            committed = True
            phase = "acknowledge"
            if os.path.basename(path) in {"settings.yaml", "config_restore.yaml"}:
                # Permissions for the replaced file. This is not the durability barrier.
                os.chmod(path, 0o600)
            _reach_write_checkpoint("after_replace", path)
            self._sync_directory(directory, path, required=require_directory_sync)
            _reach_write_checkpoint("after_acknowledge", path)
        except StoreDurabilityError:
            _discard_temp_write(tmp_path)
            raise
        except OSError as exc:
            _discard_temp_write(tmp_path)
            raise StoreDurabilityError(str(exc), phase=phase, committed=committed) from exc
        except Exception:
            _discard_temp_write(tmp_path)
            raise

    def _migrate_document(self, filename: str, data: dict) -> dict:
        if filename == "models.yaml":
            version = data.get("schema_version")
            if isinstance(version, int) and version > _MODELS_DOCUMENT_SCHEMA:
                raise StorageCorruptionError(
                    "models.yaml schema_version is newer than this build; refusing to rewrite it"
                )
            if not isinstance(version, int) or version < 2:
                data.setdefault("models", [])
            if not isinstance(version, int) or version < _MODELS_DOCUMENT_SCHEMA:
                data["schema_version"] = _MODELS_DOCUMENT_SCHEMA
        if filename == "operations.yaml":
            data.setdefault("schema_version", 1)
            data.setdefault("operations", [])
        return data

    def _validate_document(self, filename: str, data: dict) -> None:
        if not isinstance(data, dict):
            raise StorageCorruptionError(f"{filename} must be a mapping")
        if filename == "models.yaml":
            models = data.get("models", [])
            if not isinstance(models, list):
                raise StorageCorruptionError("models.yaml models must be a list")
            seen: set[str] = set()
            for model in models:
                if not isinstance(model, dict):
                    raise StorageCorruptionError("each model must be a mapping")
                model_id = model.get("id")
                if not model_id:
                    continue
                model_id = str(model_id)
                if model_id in seen:
                    raise DuplicateIdentifierError(
                        f"duplicate model id {model_id}"
                    )
                seen.add(model_id)
        if filename == "engines.yaml":
            for engine, payload in data.items():
                if not isinstance(payload, dict):
                    continue
                versions = payload.get("versions")
                if versions is None:
                    continue
                if not isinstance(versions, list):
                    raise StorageCorruptionError(
                        f"{engine} versions must be a list"
                    )
                seen_versions: set[str] = set()
                for row in versions:
                    if not isinstance(row, dict):
                        raise StorageCorruptionError(
                            f"{engine} version rows must be mappings"
                        )
                    version = row.get("version")
                    if version is None or version == "":
                        continue
                    version = str(version)
                    if version in seen_versions:
                        raise DuplicateIdentifierError(
                            f"duplicate {engine} version {version}"
                        )
                    seen_versions.add(version)

    def _sync_directory(self, directory: str, path: str, *, required: bool) -> None:
        """Fsync the directory entry. A required sync is part of journal durability."""
        _reach_write_checkpoint("before_directory_sync", path)
        try:
            dir_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError as exc:
            if required:
                raise StoreDurabilityError(
                    "Could not make the directory entry durable.",
                    phase="directory_sync",
                    committed=True,
                ) from exc
            logger.debug(
                "Could not fsync config directory %s after replacing %s",
                directory,
                os.path.basename(path),
            )
            return
        _reach_write_checkpoint("after_directory_sync", path)

    def _mutate(
        self,
        filename: str,
        mutator: Callable[[dict], Any],
    ) -> Any:
        """Hold the process and interprocess locks across read, validate, mutate, and write."""
        _wait_until_restore_released()
        path = os.path.join(self._config_dir, filename)
        with self._lock:
            self._ipc_enter()
            try:
                configuration_documents = {
                    "settings.yaml",
                    "models.yaml",
                    "model_config_templates.yaml",
                    "llama_swap_routing.yaml",
                }
                restore_journal = os.path.join(self._config_dir, "config_restore.yaml")
                if filename in configuration_documents and os.path.exists(restore_journal):
                    raise StorageCorruptionError(
                        "Configuration restore requires reconciliation before another save"
                    )
                cached = self._cached_document(path)
                if cached is not None:
                    status, loaded = "ok", cached
                else:
                    status, loaded, _stamp, confirmed = self._load_consistent_document(
                        path
                    )
                    if not confirmed:
                        self._forget_document(path)
                        raise StorageCorruptionError(
                            f"{filename} changed while it was being read; the file was left unchanged"
                        )
                if status == "corrupt":
                    self._preserve_corrupt_copy(path)
                    self._forget_document(path)
                    raise StorageCorruptionError(
                        f"{filename} is corrupt or unreadable; the file was left unchanged"
                    )
                if status == "absent" or loaded is None:
                    data = self._default_document(filename)
                else:
                    data = copy.deepcopy(loaded)
                data = self._migrate_document(filename, data)
                previous = copy.deepcopy(data)
                try:
                    result = mutator(data)
                except _SkipWrite as skipped:
                    return skipped.result
                self._validate_document(filename, data)
                if data != previous:
                    from backend.config_history import record_document_snapshot

                    record_document_snapshot(store=self, filename=filename, snapshot=previous)
                self._write_yaml(path, data)
                self._remember_document(path, data, self._file_stamp(path))
                return result
            finally:
                self._ipc_exit()

    def _save_yaml(self, filename: str, data: dict) -> None:
        """Thread-safe full-document replace. Refuses to overwrite a corrupt file."""

        def mutator(current: dict) -> None:
            current.clear()
            current.update(copy.deepcopy(data))

        self._mutate(filename, mutator)

    # --- Models ---

    def list_models(self) -> List[dict]:
        return self._read_yaml("models.yaml").get("models", [])

    def get_model(self, model_id: str) -> Optional[dict]:
        for m in self.list_models():
            if m.get("id") == model_id:
                return m
        return None

    def add_model(self, model: dict) -> dict:
        def mutator(data: dict) -> dict:
            data["schema_version"] = _MODELS_DOCUMENT_SCHEMA
            normalized = self._record_for_models_yaml(model)
            data.setdefault("models", []).append(normalized)
            return normalized

        return self._mutate("models.yaml", mutator)

    def update_model(self, model_id: str, updates: dict) -> Optional[dict]:
        def mutator(data: dict) -> dict:
            for model in data.get("models", []):
                if model.get("id") == model_id:
                    model.update(updates)
                    normalized = self._record_for_models_yaml(model)
                    model.clear()
                    model.update(normalized)
                    data["schema_version"] = _MODELS_DOCUMENT_SCHEMA
                    return model
            raise _SkipWrite(None)

        return self._mutate("models.yaml", mutator)

    @staticmethod
    def _record_for_models_yaml(model: dict) -> dict:
        """Normalize a record and persist the live engine-compatibility snapshot."""
        record = normalize_model_record(model)
        record["compatible_engines"] = compatible_engines_for_record(record)
        return record

    def delete_model(self, model_id: str) -> bool:
        def mutator(data: dict) -> bool:
            models = data.get("models", [])
            new_models = [model for model in models if model.get("id") != model_id]
            if len(new_models) == len(models):
                raise _SkipWrite(False)
            data["models"] = new_models
            return True

        return self._mutate("models.yaml", mutator)

    # --- Engines ---

    def get_engine_versions(self, engine: str) -> List[dict]:
        """Return installed versions for any registered engine."""
        return self._read_yaml("engines.yaml").get(engine, {}).get("versions", [])

    def get_active_engine_version(self, engine: str) -> Optional[dict]:
        data = self._read_yaml("engines.yaml").get(engine, {})
        active = data.get("active_version")
        if not active:
            return None
        for v in data.get("versions", []):
            if v.get("version") == active:
                return v
        return None

    def add_engine_version(self, engine: str, version_data: dict) -> None:
        def mutator(data: dict) -> None:
            data.setdefault(engine, {}).setdefault("versions", []).append(version_data)

        self._mutate("engines.yaml", mutator)

    def update_engine_version(
        self, engine: str, version: str, updates: Dict[str, Any]
    ) -> Optional[dict]:
        """Merge updates into an existing engine version row."""
        if not isinstance(updates, dict):
            updates = {}

        def mutator(data: dict) -> dict:
            engine_data = data.setdefault(engine, {})
            for row in engine_data.setdefault("versions", []):
                if str(row.get("version")) == str(version):
                    row.update(updates)
                    return row
            raise _SkipWrite(None)

        return self._mutate("engines.yaml", mutator)

    def set_active_engine_version(self, engine: str, version: str) -> None:
        def mutator(data: dict) -> None:
            data.setdefault(engine, {})["active_version"] = version

        self._mutate("engines.yaml", mutator)

    def delete_engine_version(self, engine: str, version: str) -> bool:
        def mutator(data: dict) -> bool:
            engine_data = data.get(engine, {})
            versions = engine_data.get("versions", [])
            new_versions = [row for row in versions if row.get("version") != version]
            if len(new_versions) == len(versions):
                raise _SkipWrite(False)
            engine_data["versions"] = new_versions
            if engine_data.get("active_version") == version:
                engine_data["active_version"] = None
            return True

        return self._mutate("engines.yaml", mutator)

    def get_engine_build_settings(self, engine: str) -> Dict[str, Any]:
        """Return persisted build settings for the given engine (or empty dict)."""
        data = self._read_yaml("engines.yaml")
        return data.get(engine, {}).get("build_settings", {}) or {}

    def update_engine_build_settings(
        self, engine: str, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge and persist build settings for the given engine. Returns the stored settings."""
        if not isinstance(settings, dict):
            settings = {}

        def mutator(data: dict) -> Dict[str, Any]:
            engine_data = data.setdefault(engine, {})
            existing = engine_data.get("build_settings") or {}
            merged = {**existing, **settings}
            engine_data["build_settings"] = merged
            return merged

        return self._mutate("engines.yaml", mutator)

    def replace_engine_build_settings(
        self, engine: str, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Replace persisted build settings for the given engine. Returns the stored settings."""
        if not isinstance(settings, dict):
            settings = {}

        def mutator(data: dict) -> Dict[str, Any]:
            stored = dict(settings)
            data.setdefault(engine, {})["build_settings"] = stored
            return stored

        return self._mutate("engines.yaml", mutator)

    # --- CUDA ---

    def get_cuda_status(self) -> dict:
        return self._read_yaml("engines.yaml").get("cuda", {})

    def update_cuda(self, updates: dict) -> None:
        def mutator(data: dict) -> None:
            data.setdefault("cuda", {}).update(updates)

        self._mutate("engines.yaml", mutator)

    # --- Model config templates ---

    def list_config_templates(self) -> List[dict]:
        return list(self._read_yaml("model_config_templates.yaml").get("templates", []))

    def get_config_template(self, template_id: str) -> Optional[dict]:
        for item in self.list_config_templates():
            if item.get("id") == template_id:
                return item
        return None

    def add_config_template(self, template: dict) -> dict:
        def mutator(data: dict) -> dict:
            data.setdefault("templates", []).append(template)
            return template

        return self._mutate("model_config_templates.yaml", mutator)

    def update_config_template(
        self, template_id: str, updates: dict
    ) -> Optional[dict]:
        def mutator(data: dict) -> dict:
            for item in data.get("templates", []):
                if item.get("id") == template_id:
                    item.update(updates)
                    return item
            raise _SkipWrite(None)

        return self._mutate("model_config_templates.yaml", mutator)

    def delete_config_template(self, template_id: str) -> bool:
        def mutator(data: dict) -> bool:
            templates = data.get("templates", [])
            kept = [item for item in templates if item.get("id") != template_id]
            if len(kept) == len(templates):
                raise _SkipWrite(False)
            data["templates"] = kept
            return True

        return self._mutate("model_config_templates.yaml", mutator)

    # --- Settings ---

    def get_settings(self) -> dict:
        return self._read_yaml("settings.yaml")

    def update_settings(self, updates: dict) -> None:
        def mutator(data: dict) -> None:
            data.update(updates)

        self._mutate("settings.yaml", mutator)

    # --- llama-swap routing (profiles / selectors) ---

    def get_llama_swap_routing(self) -> dict:
        data = self._read_yaml("llama_swap_routing.yaml")
        if not isinstance(data, dict):
            return {"profiles": {}, "selectors": {}}
        profiles = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
        selectors = (
            data.get("selectors") if isinstance(data.get("selectors"), dict) else {}
        )
        return {"profiles": profiles, "selectors": selectors}

    def set_llama_swap_routing(self, routing: dict) -> dict:
        if not isinstance(routing, dict):
            routing = {}
        payload = {
            "profiles": routing.get("profiles")
            if isinstance(routing.get("profiles"), dict)
            else {},
            "selectors": routing.get("selectors")
            if isinstance(routing.get("selectors"), dict)
            else {},
        }

        def mutator(data: dict) -> dict:
            data.clear()
            data.update(payload)
            return payload

        return self._mutate("llama_swap_routing.yaml", mutator)

    def list_operations(self) -> List[dict]:
        data = self._read_yaml("operations.yaml")
        operations = data.get("operations", []) if isinstance(data, dict) else []
        return list(operations) if isinstance(operations, list) else []

    def _store_operation_rows(self, data: dict, rows: list) -> list:
        from backend.operations.retention import prune_operation_rows

        data["schema_version"] = 1
        kept = prune_operation_rows(rows)
        data["operations"] = kept
        return kept

    def upsert_operation(self, operation: dict) -> dict:
        """Insert or replace a durable operation record by ``operation_id``.

        Terminal history is pruned in the same write. The returned record is
        the one submitted, even when retention drops a finished success.
        """
        operation_id = str(operation.get("operation_id") or "").strip()
        if not operation_id:
            raise ValueError("operation_id is required")

        def mutator(data: dict) -> dict:
            rows = data.setdefault("operations", [])
            stored = dict(operation)
            stored["operation_id"] = operation_id
            replaced = False
            for index, row in enumerate(rows):
                if str(row.get("operation_id") or "") == operation_id:
                    rows[index] = stored
                    replaced = True
                    break
            if not replaced:
                rows.append(stored)
            self._store_operation_rows(data, rows)
            return stored

        return self._mutate("operations.yaml", mutator)

    def replace_operations(self, operations: list) -> list:
        """Replace the durable operation list in one fsync, then apply retention."""

        def mutator(data: dict) -> list:
            rows = [dict(row) for row in operations if isinstance(row, dict)]
            return self._store_operation_rows(data, rows)

        return self._mutate("operations.yaml", mutator)

    def delete_operation(self, operation_id: str) -> bool:
        """Drop a durable operation so a dismissed activity is not restored."""
        operation_id = str(operation_id or "").strip()
        if not operation_id:
            return False

        def mutator(data: dict) -> bool:
            rows = data.get("operations", [])
            if not isinstance(rows, list):
                raise _SkipWrite(False)
            kept = [
                row
                for row in rows
                if str(row.get("operation_id") or "") != operation_id
            ]
            if len(kept) == len(rows):
                raise _SkipWrite(False)
            data["operations"] = kept
            return True

        return bool(self._mutate("operations.yaml", mutator))


_RESTORE_RELEASED = threading.Event()
_RESTORE_RELEASED.set()


def hold_configuration_writes() -> None:
    """Block ordinary configuration writes until restore recovery has finished."""
    _RESTORE_RELEASED.clear()


def release_configuration_writes() -> None:
    """Allow configuration writes after restore recovery has finished."""
    _RESTORE_RELEASED.set()


def _wait_until_restore_released() -> None:
    _RESTORE_RELEASED.wait()


_store: Optional[DataStore] = None


def get_store() -> DataStore:
    global _store
    if _store is None:
        _store = DataStore()
    return _store


class _ExclusiveDocuments:
    def __init__(self, store: DataStore) -> None:
        self._store = store

    def __enter__(self) -> DataStore:
        self._store._lock.acquire()
        self._store._ipc_enter()
        return self._store

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            self._store._ipc_exit()
        finally:
            self._store._lock.release()
