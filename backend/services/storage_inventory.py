"""Bounded, read-only inventory of the disk Studio owns.

Storage governance needs one truthful answer to "what is using disk, and what
may Studio reclaim without touching something another surface owns?".  This
module scans the known roots once, classifies them into seven disjoint
categories, and marks the units a cleanup may consider.  It never mutates
anything.

Design rules (single-operator hardening, WS2):

* **Bounded.** A scan stops after ``MAX_SCAN_ENTRIES`` filesystem entries or
  ``MAX_UNITS_PER_CATEGORY`` units per category.  A truncated scan is reported
  as partial quality, never silently padded with guessed bytes.
* **No symlink following.** Directory symlinks are skipped while measuring and
  every unit path is re-validated with ``realpath`` containment before it can
  be offered for cleanup.
* **No path input.** Units are identified by an opaque hash of their identity
  inside the category; the API never accepts a filesystem path.
* **Report-only where another surface owns the data.** Catalog model payloads,
  engine installs, CUDA toolkits and the proxy accounting database are measured
  but cannot be cleaned here; they carry a ``manage_path`` instead.
* **Conservative refusals.** A referenced model directory, a running model, a
  retained launch generation, an active download/build/install and an
  interrupted configuration restore all block the affected units.
* **Off the event loop.** The scan is synchronous by design; routes call it
  through :func:`backend.concurrency.run_blocking`.  A 10 second TTL cache keeps
  repeated UI reads from re-walking a multi-GB cache tree.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import stat
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from backend.logging_config import get_logger
from backend.paths import studio_data_dir
from backend.utils.fs_ops import running_model_ids

logger = get_logger(__name__)

INVENTORY_SCHEMA_VERSION = 1
INVENTORY_TTL_SECONDS = 10.0
MAX_SCAN_ENTRIES = 100_000
MAX_UNITS_PER_CATEGORY = 200
DEFAULT_HISTORY_KEEP = 50
MAX_HISTORY_ENTRIES = 100

CATEGORY_ORDER: Tuple[str, ...] = (
    "models",
    "hf_cache",
    "engine_installs",
    "cuda_toolkits",
    "jit_caches",
    "history_backups",
    "accounting_db",
)

CLEANABLE_CATEGORIES = frozenset({"models", "hf_cache", "jit_caches", "history_backups"})
REPORT_ONLY_CATEGORIES = frozenset({"engine_installs", "cuda_toolkits", "accounting_db"})

CATEGORY_LABELS: Dict[str, str] = {
    "models": "Model files",
    "hf_cache": "Hugging Face cache",
    "engine_installs": "Engine installs",
    "cuda_toolkits": "CUDA toolkits",
    "jit_caches": "Build and JIT caches",
    "history_backups": "Configuration history",
    "accounting_db": "Proxy accounting",
}

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "models": "Payload directories in Studio's model store. Directories with a catalog row are managed from Models.",
    "hf_cache": "Hugging Face hub downloads. Repositories used by a catalog model are reported, not deleted.",
    "engine_installs": "Versioned engine installations. Remove versions from Engines.",
    "cuda_toolkits": "Installed CUDA toolkits. Manage them from System hardware.",
    "jit_caches": "Regenerable compiled-kernel, compile-workspace and installer caches.",
    "history_backups": "Local pre-change configuration revisions. Older revisions can be pruned.",
    "accounting_db": "The proxy request accounting database. Manage it from Proxy.",
}

CATEGORY_MANAGE_PATHS: Dict[str, str] = {
    "models": "/models",
    "engine_installs": "/engines",
    "cuda_toolkits": "/system#hardware",
    "accounting_db": "/proxy",
}

# Stable refusal codes shared with the cleanup service and the frontend.
CODE_REPORT_ONLY = "REPORT_ONLY"
CODE_CATALOG_MODEL = "CATALOG_MODEL"
CODE_SHARED_DATA = "SHARED_DATA"
CODE_RUNNING_MODEL = "RUNNING_MODEL"
CODE_RETAINED_LAUNCH_REFERENCE = "RETAINED_LAUNCH_REFERENCE"
CODE_DOWNLOAD_IN_PROGRESS = "DOWNLOAD_IN_PROGRESS"
CODE_BUILD_IN_PROGRESS = "BUILD_IN_PROGRESS"
CODE_CUDA_INSTALL_IN_PROGRESS = "CUDA_INSTALL_IN_PROGRESS"
CODE_CONFIG_RECOVERY_IN_PROGRESS = "CONFIG_RECOVERY_IN_PROGRESS"
CODE_STORE_UNAVAILABLE = "STORE_UNAVAILABLE"
CODE_RECENT_HISTORY = "RECENT_HISTORY"

_ACTIVE_OPERATION_STATUSES = frozenset({"queued", "running", "cancelling"})
_BUILD_OPERATION_KINDS = frozenset(
    {
        "build",
        "install",
        "install_source",
        "update",
        "sync_source",
        "remove",
        "uninstall",
    }
)
_MODEL_INSTALL_OPERATION_KINDS = frozenset({"audio_model_install", "audio_model_import"})

_INVENTORY_CACHE: Dict[str, Tuple[float, dict, dict]] = {}
_INVENTORY_LOCK = threading.Lock()


class _ScanBudget:
    """Shared entry budget and truncation flag for one inventory scan."""

    def __init__(self, limit: Optional[int] = None) -> None:
        self.limit = max(1, int(MAX_SCAN_ENTRIES if limit is None else limit))
        self.entries = 0
        self.truncated = False

    def spend(self, count: int = 1) -> bool:
        if self.entries + count > self.limit:
            self.truncated = True
            return False
        self.entries += count
        return True


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_realpath(path: str) -> str:
    text = str(path or "")
    if not text:
        return ""
    try:
        return os.path.realpath(text)
    except (OSError, ValueError):
        return ""


def _is_within(root: str, path: str) -> bool:
    """True when ``path`` is ``root`` or lives under it (both realpaths)."""
    if not root or not path:
        return False
    try:
        return os.path.commonpath([root, path]) == root
    except ValueError:
        return False


def _paths_overlap(first: str, second: str) -> bool:
    if not first or not second:
        return False
    if first == second:
        return True
    return first.startswith(second + os.sep) or second.startswith(first + os.sep)


def _unit_id(category: str, key: str) -> str:
    digest = hashlib.sha256(f"{category}\x00{key}".encode()).hexdigest()
    return digest[:20]


def _blocked(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _mtime_ns(path: str) -> int:
    try:
        return int(os.stat(path, follow_symlinks=False).st_mtime_ns)
    except OSError:
        return 0


def _file_bytes(path: str) -> int:
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        return 0
    return int(info.st_size) if stat.S_ISREG(info.st_mode) else 0


def _measure_tree(
    path: str,
    budget: _ScanBudget,
    *,
    skip_realpaths: Optional[Iterable[str]] = None,
) -> Tuple[int, int, bool]:
    """Return ``(bytes, latest_mtime_ns, hit_budget)`` without following links."""
    skip = {value for value in (skip_realpaths or ()) if value}
    try:
        info = os.stat(path, follow_symlinks=False)
    except OSError:
        return 0, 0, False
    if not stat.S_ISDIR(info.st_mode):
        return _file_bytes(path), int(info.st_mtime_ns), False

    total = 0
    latest = int(info.st_mtime_ns)
    stack = [path]
    while stack:
        current = stack.pop()
        try:
            entries = list(os.scandir(current))
        except OSError:
            continue
        for entry in entries:
            if not budget.spend():
                return total, latest, True
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    real = _safe_realpath(entry.path)
                    if real and real in skip:
                        continue
                    latest = max(latest, _mtime_ns(entry.path))
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    total += int(entry.stat(follow_symlinks=False).st_size)
                    latest = max(latest, _mtime_ns(entry.path))
            except OSError:
                continue
    return total, latest, False


def _direct_children(path: str) -> List[os.DirEntry]:
    try:
        return list(os.scandir(path))
    except OSError:
        return []


def _child_dirs(path: str) -> List[os.DirEntry]:
    return [
        entry
        for entry in _direct_children(path)
        if not entry.is_symlink() and entry.is_dir(follow_symlinks=False)
    ]


def _store_or_none():
    try:
        from backend.data_store import get_store

        return get_store()
    except Exception as exc:  # pragma: no cover - a broken store degrades the scan
        logger.debug("Storage inventory could not read the data store: %s", exc)
        return None


def _config_dir(store) -> str:
    if store is not None:
        directory = getattr(store, "_config_dir", None)
        if directory:
            return _safe_realpath(str(directory))
    return _safe_realpath(os.path.join(studio_data_dir(), "config"))


def _model_rows(store) -> Optional[List[dict]]:
    if store is None:
        return None
    try:
        return [row for row in store.list_models() if isinstance(row, dict)]
    except Exception as exc:  # pragma: no cover - degraded store
        logger.debug("Storage inventory could not list models: %s", exc)
        return None


def _iter_string_paths(value: Any, depth: int = 0) -> Iterable[str]:
    """Yield absolute path-looking strings from a model row, safely bounded."""
    if depth > 5:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"filename", "id", "name", "display_name", "reason"}:
                continue
            yield from _iter_string_paths(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_string_paths(item, depth + 1)
    elif isinstance(value, str):
        text = value.strip()
        if text and os.path.isabs(text):
            yield text


def _referenced_model_paths(rows: Iterable[dict]) -> Set[str]:
    referenced: Set[str] = set()
    for row in rows:
        explicit: List[Any] = [
            row.get("local_path"),
            row.get("path"),
            row.get("bundle_path"),
        ]
        artifact = row.get("artifact")
        if isinstance(artifact, dict):
            explicit.extend(
                [
                    artifact.get("path"),
                    artifact.get("bundle_path"),
                    artifact.get("runtime_path"),
                ]
            )
        for value in explicit:
            text = str(value or "").strip()
            if text and os.path.isabs(text):
                real = _safe_realpath(text)
                if real:
                    referenced.add(real)
        for text in _iter_string_paths(row.get("files")):
            real = _safe_realpath(text)
            if real:
                referenced.add(real)
    return referenced


def _referenced_hf_folders(rows: Iterable[dict]) -> Set[str]:
    folders: Set[str] = set()
    for row in rows:
        repo_id = str(row.get("huggingface_id") or "").strip()
        if not repo_id:
            continue
        try:
            from backend.models.hub import _hf_repo_folder_name

            folders.add(_hf_repo_folder_name(repo_id))
        except Exception:
            folders.add("models--" + repo_id.replace("/", "--"))
    return folders


def _protected_model_dirs(data_root: str) -> Set[str]:
    """Directories under the model store owned by a shared Studio feature."""
    protected: Set[str] = set()
    try:
        from backend.reference_audio import REFERENCE_AUDIO_DATA_SUBDIR

        protected.add(_safe_realpath(os.path.join(data_root, REFERENCE_AUDIO_DATA_SUBDIR)))
    except Exception:
        pass
    try:
        from backend.audio.community_voices import community_voice_dir

        protected.add(_safe_realpath(community_voice_dir()))
    except Exception:
        pass
    return {path for path in protected if path}


def _hf_cache_root() -> str:
    env_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
    if env_cache:
        return _safe_realpath(env_cache)
    try:
        from backend.models import hub

        return _safe_realpath(hub._hf_hub_cache_root())
    except Exception as exc:  # pragma: no cover - huggingface_hub import issues
        logger.debug("Storage inventory could not resolve the HF cache root: %s", exc)
        home = os.getenv("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        return _safe_realpath(os.path.join(home, "hub"))


def _engine_roots() -> List[Tuple[str, str]]:
    """Distinct ``(engine, root_realpath)`` install roots from existing owners."""
    try:
        from backend.engines.lifecycle import discover_engine_install_roots

        roots = discover_engine_install_roots()
    except Exception as exc:  # pragma: no cover - engine package unavailable
        logger.debug("Storage inventory could not resolve engine roots: %s", exc)
        return []
    seen: Dict[str, str] = {}
    for engine, root in (roots or {}).items():
        real = _safe_realpath(str(root or ""))
        if real:
            seen.setdefault(real, str(engine))
    return [(engine, real) for real, engine in seen.items()]


def _active_operation_rows(store) -> List[dict]:
    if store is None:
        return []
    try:
        rows = store.list_operations()
    except Exception as exc:  # pragma: no cover - degraded store
        logger.debug("Storage inventory could not list operations: %s", exc)
        return []
    return [
        dict(row)
        for row in rows
        if isinstance(row, dict) and str(row.get("status") or "") in _ACTIVE_OPERATION_STATUSES
    ]


def _row_keys(row: dict) -> List[str]:
    keys = [str(row.get("resource_key") or "")]
    detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
    depends_on = detail.get("depends_on")
    if isinstance(depends_on, str):
        keys.append(depends_on)
    elif isinstance(depends_on, (list, tuple)):
        keys.extend(str(item) for item in depends_on)
    return [key for key in keys if key]


def _has_active_operation(
    rows: Iterable[dict], *, kinds: Iterable[str], keys: Iterable[str]
) -> bool:
    from backend.operations.action_recovery import keys_overlap

    wanted_kinds = set(kinds)
    wanted_keys = [key for key in keys if key]
    for row in rows:
        if wanted_kinds and str(row.get("kind") or "") in wanted_kinds:
            return True
        if wanted_keys and any(
            keys_overlap(key, wanted) for key in wanted_keys for wanted in _row_keys(row)
        ):
            return True
    return False


def _launch_block(path: str) -> Optional[dict]:
    """Refusal for a running model or a retained launch generation at ``path``."""
    try:
        from backend.proxy.manifests import deletion_block_reason
    except Exception:
        return None
    try:
        if not deletion_block_reason(path):
            return None
        live = running_model_ids(path)
    except Exception as exc:  # pragma: no cover - manifest store unavailable
        logger.debug("Storage inventory could not check launch holds: %s", exc)
        return _blocked(
            CODE_RETAINED_LAUNCH_REFERENCE,
            "A launch generation may still depend on this path.",
        )
    if live:
        names = ", ".join(str(name) for name in live)
        return _blocked(
            CODE_RUNNING_MODEL,
            f"Stop {names} before deleting this path. A running launch still uses it.",
        )
    return _blocked(
        CODE_RETAINED_LAUNCH_REFERENCE,
        "A published or retained launch generation still references this path.",
    )


def _unit(
    category: str,
    *,
    budget: _ScanBudget,
    path: str,
    root: str,
    label: str,
    kind: str,
    key: str,
    data_root: str,
    manage_path: Optional[str],
    blocked: Optional[dict] = None,
    detail: Optional[dict] = None,
    skip_realpaths: Optional[Iterable[str]] = None,
) -> dict:
    real = _safe_realpath(path)
    root_real = _safe_realpath(root)
    bytes_used, latest, _hit = _measure_tree(real, budget, skip_realpaths=skip_realpaths)
    return {
        "id": _unit_id(category, key),
        "label": label,
        "kind": kind,
        "bytes": int(bytes_used),
        "mtime_ns": int(latest),
        "path": real,
        "root": root_real,
        "cleanable": blocked is None,
        "blocked": blocked,
        "manage_path": manage_path if blocked is not None else None,
        "outside_data_root": bool(real and not _is_within(data_root, real)),
        "detail": detail or {},
    }


def _public_unit(unit: dict) -> dict:
    """Project an internal unit onto the API contract (no filesystem paths)."""
    return {
        "id": unit["id"],
        "label": unit["label"],
        "kind": unit["kind"],
        "bytes": int(unit["bytes"]),
        "cleanable": bool(unit["cleanable"]),
        "blocked": unit["blocked"],
        "manage_path": unit["manage_path"],
        "outside_data_root": bool(unit["outside_data_root"]),
        "detail": dict(unit.get("detail") or {}),
    }


def _category_payload(
    category: str,
    *,
    units: List[dict],
    unmanaged_bytes: int,
    available: bool,
    truncated: bool,
    blocked: Optional[dict] = None,
    notes: Optional[List[str]] = None,
) -> dict:
    scan_blocked = blocked
    if scan_blocked is None and category in REPORT_ONLY_CATEGORIES:
        scan_blocked = _blocked(
            CODE_REPORT_ONLY,
            "This storage is managed by another Studio surface and cannot be deleted here.",
        )
    if scan_blocked is not None:
        for unit in units:
            if unit["blocked"] is None:
                unit["blocked"] = scan_blocked
            unit["cleanable"] = False
            if unit.get("manage_path") is None:
                unit["manage_path"] = CATEGORY_MANAGE_PATHS.get(category)
    else:
        for unit in units:
            unit["cleanable"] = unit["blocked"] is None
    cleanable = category in CLEANABLE_CATEGORIES and scan_blocked is None
    cleanable_bytes = sum(unit["bytes"] for unit in units if unit["cleanable"])
    total_bytes = sum(unit["bytes"] for unit in units) + max(0, int(unmanaged_bytes))
    outside = any(unit["outside_data_root"] for unit in units)
    return {
        "id": category,
        "label": CATEGORY_LABELS[category],
        "description": CATEGORY_DESCRIPTIONS[category],
        "bytes": int(total_bytes),
        "unmanaged_bytes": max(0, int(unmanaged_bytes)),
        "unit_count": len(units),
        "cleanable": cleanable,
        "cleanable_bytes": int(cleanable_bytes),
        "blocked": scan_blocked,
        "manage_path": CATEGORY_MANAGE_PATHS.get(category),
        "outside_data_root": outside,
        "available": bool(available),
        "truncated": bool(truncated),
        "notes": list(notes or []),
        "units": [
            _public_unit(unit)
            for unit in sorted(units, key=lambda item: (-int(item["bytes"]), item["label"].lower()))
        ],
    }


def _has_payload_marker(path: str) -> bool:
    """True when a directory directly contains obvious model payload files."""
    markers = {
        "config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
        "manifest.json",
        "model.yaml",
    }
    weights = {".gguf", ".safetensors", ".bin", ".pt", ".pth", ".onnx"}
    for entry in _direct_children(path):
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                continue
        except OSError:
            continue
        if entry.name in markers or os.path.splitext(entry.name)[1].lower() in weights:
            return True
    return False


def _unmanaged_bytes(path: str, unit_paths: Set[str]) -> int:
    """Sizes of files immediately under ``path`` that are not part of a unit."""
    total = 0
    for entry in _direct_children(path):
        try:
            if entry.is_symlink():
                continue
            real = _safe_realpath(entry.path)
            if real in unit_paths:
                continue
            if entry.is_dir(follow_symlinks=False):
                continue
            total += int(entry.stat(follow_symlinks=False).st_size)
        except OSError:
            continue
    return total


def _cap_units(units: List[dict], notes: List[str]) -> List[dict]:
    if len(units) <= MAX_UNITS_PER_CATEGORY:
        return units
    ordered = sorted(units, key=lambda item: (-int(item["bytes"]), item["label"].lower()))
    notes.append(f"Only the largest {MAX_UNITS_PER_CATEGORY} units are listed.")
    return ordered[:MAX_UNITS_PER_CATEGORY]


def _scan_models(
    data_root: str,
    store,
    budget: _ScanBudget,
    *,
    active_rows: List[dict],
) -> Tuple[dict, Dict[str, dict]]:
    root = _safe_realpath(os.path.join(data_root, "models"))
    rows = _model_rows(store)
    references = _referenced_model_paths(rows or [])
    protected = _protected_model_dirs(data_root)
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    unmanaged = 0
    notes: List[str] = []
    blocked: Optional[dict] = None
    available = bool(root) and os.path.isdir(root)

    if rows is None:
        blocked = _blocked(
            CODE_STORE_UNAVAILABLE,
            "The model catalog could not be read, so model directories cannot be classified safely.",
        )
    elif _has_active_operation(active_rows, kinds=_MODEL_INSTALL_OPERATION_KINDS, keys=()):
        blocked = _blocked(
            CODE_BUILD_IN_PROGRESS,
            "A model install is in progress. Model storage is not cleaned until it finishes.",
        )

    def add(path: str, label: str, *, key: str, shared: bool = False) -> None:
        real = _safe_realpath(path)
        unit_blocked = blocked
        if unit_blocked is None and shared:
            unit_blocked = _blocked(
                CODE_SHARED_DATA,
                "This directory holds shared reference voices or audio owned by another feature.",
            )
        if unit_blocked is None and any(_paths_overlap(real, ref) for ref in references):
            unit_blocked = _blocked(
                CODE_CATALOG_MODEL,
                "A catalog model uses this directory. Remove the model from Models.",
            )
        if unit_blocked is None:
            unit_blocked = _launch_block(real)
        unit = _unit(
            "models",
            budget=budget,
            path=real,
            root=root,
            label=label,
            kind="model_payload",
            key=key,
            data_root=data_root,
            manage_path=CATEGORY_MANAGE_PATHS["models"],
            blocked=unit_blocked,
        )
        if unit["id"] not in unit_map:
            unit_map[unit["id"]] = unit
            units.append(unit)

    if available:
        for namespace in _child_dirs(root):
            if namespace.name.startswith("."):
                continue
            if _has_payload_marker(namespace.path):
                add(namespace.path, namespace.name, key=namespace.name)
                continue
            for child in _child_dirs(namespace.path):
                if child.name.startswith("."):
                    continue
                child_real = _safe_realpath(child.path)
                shared = child_real in protected
                add(
                    child.path,
                    f"{namespace.name}/{child.name}",
                    key=f"{namespace.name}/{child.name}",
                    shared=shared,
                )
            unmanaged += _unmanaged_bytes(namespace.path, {unit["path"] for unit in units})
        unmanaged += _unmanaged_bytes(root, {unit["path"] for unit in units})
    units = _cap_units(units, notes)
    category = _category_payload(
        "models",
        units=units,
        unmanaged_bytes=unmanaged,
        available=available,
        truncated=budget.truncated,
        blocked=blocked,
        notes=notes,
    )
    return category, unit_map


def _scan_hf_cache(
    data_root: str,
    store,
    budget: _ScanBudget,
    *,
    active_rows: List[dict],
) -> Tuple[dict, Dict[str, dict]]:
    root = _hf_cache_root()
    rows = _model_rows(store)
    referenced = _referenced_hf_folders(rows or [])
    referenced_paths = _referenced_model_paths(rows or [])
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    unmanaged = 0
    notes: List[str] = []
    blocked: Optional[dict] = None
    available = bool(root) and os.path.isdir(root)
    if rows is None:
        blocked = _blocked(
            CODE_STORE_UNAVAILABLE,
            "The model catalog could not be read, so cached repositories cannot be classified safely.",
        )
    if available:
        for entry in _direct_children(root):
            try:
                if entry.is_symlink():
                    continue
                if not entry.is_dir(follow_symlinks=False):
                    if entry.is_file(follow_symlinks=False):
                        unmanaged += int(entry.stat(follow_symlinks=False).st_size)
                    continue
            except OSError:
                continue
            if not entry.name.startswith("models--"):
                unmanaged += _measure_tree(entry.path, budget)[0]
                continue
            unit_blocked = blocked
            repo_label = entry.name[len("models--") :].replace("--", "/")
            entry_real = _safe_realpath(entry.path)
            if unit_blocked is None and (
                entry.name in referenced
                or any(_paths_overlap(entry_real, ref) for ref in referenced_paths)
            ):
                unit_blocked = _blocked(
                    CODE_CATALOG_MODEL,
                    "A catalog model uses this cached repository. Remove the model from Models.",
                )
            if unit_blocked is None:
                unit_blocked = _launch_block(entry.path)
            if unit_blocked is None and _has_active_operation(
                active_rows,
                kinds=("download",),
                keys=(f"hf:{repo_label}",),
            ):
                unit_blocked = _blocked(
                    CODE_DOWNLOAD_IN_PROGRESS,
                    "A download for this repository is in progress.",
                )
            unit = _unit(
                "hf_cache",
                budget=budget,
                path=entry.path,
                root=root,
                label=repo_label,
                kind="hf_repository",
                key=entry.name,
                data_root=data_root,
                manage_path=CATEGORY_MANAGE_PATHS["models"],
                blocked=unit_blocked,
            )
            if unit["id"] not in unit_map:
                unit_map[unit["id"]] = unit
                units.append(unit)
    units = _cap_units(units, notes)
    category = _category_payload(
        "hf_cache",
        units=units,
        unmanaged_bytes=unmanaged,
        available=available,
        truncated=budget.truncated,
        blocked=blocked,
        notes=notes,
    )
    return category, unit_map


def _scan_engine_installs(
    data_root: str,
    budget: _ScanBudget,
    *,
    roots: List[Tuple[str, str]],
) -> Tuple[dict, Dict[str, dict], List[Tuple[str, str, str]]]:
    """Scan engine version installs, excluding cache subtrees owned by jit_caches."""
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    cache_roots: List[Tuple[str, str, str]] = []  # (engine, install_dir, caches_dir)
    unmanaged = 0
    notes: List[str] = []
    available = bool(roots)
    for engine, root in roots:
        if not os.path.isdir(root):
            continue
        for version in _child_dirs(root):
            if version.name == "caches":
                unmanaged += _measure_tree(version.path, budget)[0]
                continue
            if version.name.startswith("."):
                continue
            install_dir = _safe_realpath(version.path)
            caches_real = _safe_realpath(os.path.join(install_dir, "caches"))
            if os.path.isdir(caches_real):
                cache_roots.append((engine, install_dir, caches_real))
            unit = _unit(
                "engine_installs",
                budget=budget,
                path=install_dir,
                root=root,
                label=f"{engine}/{version.name}",
                kind="engine_install",
                key=f"{engine}/{version.name}",
                data_root=data_root,
                manage_path=CATEGORY_MANAGE_PATHS["engine_installs"],
                detail={"engine": engine, "version": version.name},
                skip_realpaths={caches_real} if caches_real else None,
            )
            if unit["id"] not in unit_map:
                unit_map[unit["id"]] = unit
                units.append(unit)
        for entry in _direct_children(root):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False) and entry.name != "caches":
                    continue
                unmanaged += _measure_tree(entry.path, budget)[0]
            except OSError:
                continue
    units = _cap_units(units, notes)
    category = _category_payload(
        "engine_installs",
        units=units,
        unmanaged_bytes=unmanaged,
        available=available,
        truncated=budget.truncated,
        notes=notes,
    )
    return category, unit_map, cache_roots


def _scan_cuda_toolkits(data_root: str, budget: _ScanBudget) -> Tuple[dict, Dict[str, dict]]:
    root = _safe_realpath(os.path.join(data_root, "cuda"))
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    unmanaged = 0
    available = bool(root) and os.path.isdir(root)
    if available:
        for entry in _child_dirs(root):
            unit = _unit(
                "cuda_toolkits",
                budget=budget,
                path=entry.path,
                root=root,
                label=entry.name,
                kind="cuda_toolkit",
                key=entry.name,
                data_root=data_root,
                manage_path=CATEGORY_MANAGE_PATHS["cuda_toolkits"],
            )
            unit_map[unit["id"]] = unit
            units.append(unit)
        unmanaged += _unmanaged_bytes(root, {unit["path"] for unit in units})
    category = _category_payload(
        "cuda_toolkits",
        units=units,
        unmanaged_bytes=unmanaged,
        available=available,
        truncated=budget.truncated,
    )
    return category, unit_map


def _scan_jit_caches(
    data_root: str,
    budget: _ScanBudget,
    *,
    engine_cache_roots: List[Tuple[str, str, str]],
    active_rows: List[dict],
) -> Tuple[dict, Dict[str, dict]]:
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    notes: List[str] = []
    build_active = _has_active_operation(active_rows, kinds=_BUILD_OPERATION_KINDS, keys=())
    cuda_active = _has_active_operation(
        active_rows, kinds=_BUILD_OPERATION_KINDS, keys=("cuda:toolkit",)
    )

    def add_cache(
        path: str,
        *,
        label: str,
        kind: str,
        key: str,
        root: str,
        blocked: Optional[dict] = None,
        detail: Optional[dict] = None,
    ) -> None:
        if not os.path.isdir(path):
            return
        unit = _unit(
            "jit_caches",
            budget=budget,
            path=path,
            root=root,
            label=label,
            kind=kind,
            key=key,
            data_root=data_root,
            manage_path=None,
            blocked=blocked,
            detail=detail,
        )
        if unit["id"] not in unit_map:
            unit_map[unit["id"]] = unit
            units.append(unit)

    for engine, install_dir, caches_dir in engine_cache_roots:
        if not caches_dir or not os.path.isdir(caches_dir):
            continue
        unit_blocked = None
        if _has_active_operation(
            active_rows,
            kinds=_BUILD_OPERATION_KINDS,
            keys=(f"engine:{engine}:{install_dir}",),
        ):
            unit_blocked = _blocked(
                CODE_BUILD_IN_PROGRESS,
                "An engine build or install for this version is in progress.",
            )
        else:
            unit_blocked = _launch_block(caches_dir)
        add_cache(
            caches_dir,
            label=f"{engine}/{os.path.basename(install_dir)}/caches",
            kind="engine_cache",
            key=f"engine:{engine}/{os.path.basename(install_dir)}",
            root=install_dir,
            blocked=unit_blocked,
            detail={"engine": engine},
        )

    workspace_root = _safe_realpath(os.path.join(data_root, "build-workspaces"))
    if workspace_root and os.path.isdir(workspace_root):
        for workspace in _child_dirs(workspace_root):
            add_cache(
                workspace.path,
                label=f"build-workspaces/{workspace.name}",
                kind="build_workspace",
                key=f"workspace/{workspace.name}",
                root=workspace_root,
                blocked=(
                    _blocked(
                        CODE_BUILD_IN_PROGRESS,
                        "A build is in progress and may still be using a compile workspace.",
                    )
                    if build_active
                    else None
                ),
            )

    ccache = _safe_realpath(
        os.getenv("STUDIO_CCACHE_DIR", "").strip() or os.path.join(data_root, "ccache")
    )
    if ccache and os.path.isdir(ccache):
        add_cache(
            ccache,
            label="ccache",
            kind="ccache",
            key="ccache",
            # The unit path is the cache root itself, so the validated root is
            # its parent: a cleanup target is never a storage root.
            root=os.path.dirname(ccache),
            blocked=(
                _blocked(
                    CODE_BUILD_IN_PROGRESS,
                    "A build is in progress and may still be using the compiler cache.",
                )
                if build_active
                else None
            ),
        )

    installer_cache = _safe_realpath(os.path.join(data_root, "temp", "cuda_installers"))
    if installer_cache and os.path.isdir(installer_cache):
        add_cache(
            installer_cache,
            label="CUDA installer cache",
            kind="cuda_installer_cache",
            key="cuda-installers",
            root=os.path.dirname(installer_cache),
            blocked=(
                _blocked(
                    CODE_CUDA_INSTALL_IN_PROGRESS,
                    "A CUDA toolkit install is in progress and may still need the installer cache.",
                )
                if cuda_active
                else None
            ),
        )

    units = _cap_units(units, notes)
    category = _category_payload(
        "jit_caches",
        units=units,
        unmanaged_bytes=0,
        available=True,
        truncated=budget.truncated,
        notes=notes,
    )
    return category, unit_map


def _scan_history_backups(config_dir: str, data_root: str) -> Tuple[dict, Dict[str, dict]]:
    root = _safe_realpath(os.path.join(config_dir, "history"))
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    unmanaged = 0
    available = bool(root) and os.path.isdir(root)
    restore_in_progress = bool(config_dir) and os.path.isfile(
        os.path.join(config_dir, "config_restore.yaml")
    )
    blocked = (
        _blocked(
            CODE_CONFIG_RECOVERY_IN_PROGRESS,
            "A configuration restore is pending. Reconcile it before pruning history.",
        )
        if restore_in_progress
        else None
    )
    entries: List[os.DirEntry] = []
    if available:
        for entry in _direct_children(root):
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file(follow_symlinks=False) and entry.name.endswith(".yaml"):
                    entries.append(entry)
                    continue
                if entry.is_file(follow_symlinks=False):
                    unmanaged += int(entry.stat(follow_symlinks=False).st_size)
                elif entry.is_dir(follow_symlinks=False):
                    unmanaged += _measure_tree(entry.path, _ScanBudget())[0]
            except OSError:
                continue
    entries.sort(key=lambda entry: entry.name, reverse=True)
    budget = _ScanBudget(MAX_HISTORY_ENTRIES)
    for index, entry in enumerate(entries[:MAX_HISTORY_ENTRIES]):
        unit_blocked = blocked
        if unit_blocked is None and index < DEFAULT_HISTORY_KEEP:
            unit_blocked = _blocked(
                CODE_RECENT_HISTORY,
                "Recent revisions are retained. Prune only revisions older than the keep count.",
            )
        unit = _unit(
            "history_backups",
            budget=budget,
            path=entry.path,
            root=root,
            label=entry.name,
            kind="history_revision",
            key=entry.name,
            data_root=data_root,
            manage_path=None,
            blocked=unit_blocked,
            detail={"file": entry.name},
        )
        unit_map[unit["id"]] = unit
        units.append(unit)
    category = _category_payload(
        "history_backups",
        units=units,
        unmanaged_bytes=unmanaged,
        available=available,
        truncated=False,
        blocked=blocked,
    )
    return category, unit_map


def _scan_accounting_db(config_dir: str, data_root: str) -> Tuple[dict, Dict[str, dict]]:
    root = _safe_realpath(config_dir)
    unit_map: Dict[str, dict] = {}
    units: List[dict] = []
    base = os.path.join(config_dir, "proxy-accounting.sqlite3") if config_dir else ""
    files = [base, base + "-wal", base + "-shm"] if base else []
    existing = [path for path in files if os.path.isfile(path)]
    if existing:
        total = sum(_file_bytes(path) for path in existing)
        unit = {
            "id": _unit_id("accounting_db", "proxy-accounting.sqlite3"),
            "label": "proxy-accounting.sqlite3",
            "kind": "accounting_db",
            "bytes": int(total),
            "mtime_ns": max(_mtime_ns(path) for path in existing),
            "path": _safe_realpath(existing[0]),
            "root": root,
            "cleanable": False,
            "blocked": _blocked(
                CODE_REPORT_ONLY,
                "The request accounting database is compacted and reset from Proxy.",
            ),
            "manage_path": CATEGORY_MANAGE_PATHS["accounting_db"],
            "outside_data_root": bool(root and not _is_within(data_root, root)),
            "detail": {"files": len(existing)},
        }
        unit_map[unit["id"]] = unit
        units.append(unit)
    category = _category_payload(
        "accounting_db",
        units=units,
        unmanaged_bytes=0,
        available=bool(existing),
        truncated=False,
    )
    return category, unit_map


def _filesystem(data_root: str) -> dict:
    payload: Dict[str, Any] = {
        "data_root": data_root,
        "total_bytes": None,
        "used_bytes": None,
        "free_bytes": None,
    }
    target = data_root
    while target and not os.path.isdir(target):
        parent = os.path.dirname(target)
        if parent == target:
            break
        target = parent
    try:
        usage = shutil.disk_usage(target or ".")
    except OSError:
        return payload
    payload.update(
        {
            "total_bytes": int(usage.total),
            "used_bytes": int(usage.used),
            "free_bytes": int(usage.free),
        }
    )
    return payload


def _cache_key(data_root: str) -> str:
    ccache = _safe_realpath(
        os.getenv("STUDIO_CCACHE_DIR", "").strip() or os.path.join(data_root, "ccache")
    )
    return "\x00".join([data_root, _hf_cache_root(), ccache])


def clear_storage_inventory_cache() -> None:
    """Drop the TTL cache (tests, and after a cleanup mutates the tree)."""
    with _INVENTORY_LOCK:
        _INVENTORY_CACHE.clear()


def _scan_storage(*, refresh: bool = False) -> Tuple[dict, dict]:
    data_root = _safe_realpath(studio_data_dir())
    key = _cache_key(data_root)
    now = time.monotonic()
    if not refresh:
        with _INVENTORY_LOCK:
            cached = _INVENTORY_CACHE.get(key)
        if cached and cached[0] > now:
            payload = dict(cached[1])
            payload["cached"] = True
            payload["scan"] = {**payload["scan"], "cached": True}
            return payload, cached[2]

    budget = _ScanBudget()
    store = _store_or_none()
    config_dir = _config_dir(store)
    active_rows = _active_operation_rows(store)
    engine_roots = _engine_roots()
    notes: List[str] = []

    models_category, models_index = _scan_models(data_root, store, budget, active_rows=active_rows)
    hf_category, hf_index = _scan_hf_cache(data_root, store, budget, active_rows=active_rows)
    engine_category, engine_index, engine_cache_roots = _scan_engine_installs(
        data_root, budget, roots=engine_roots
    )
    cuda_category, cuda_index = _scan_cuda_toolkits(data_root, budget)
    jit_category, jit_index = _scan_jit_caches(
        data_root,
        budget,
        engine_cache_roots=engine_cache_roots,
        active_rows=active_rows,
    )
    history_category, history_index = _scan_history_backups(config_dir, data_root)
    accounting_category, accounting_index = _scan_accounting_db(config_dir, data_root)

    categories = [
        models_category,
        hf_category,
        engine_category,
        cuda_category,
        jit_category,
        history_category,
        accounting_category,
    ]
    if budget.truncated:
        notes.append(
            f"The scan stopped after {MAX_SCAN_ENTRIES} filesystem entries; sizes may be lower than reality."
        )
    degraded = budget.truncated or any(
        item["blocked"] and item["blocked"]["code"] == CODE_STORE_UNAVAILABLE for item in categories
    )
    payload = {
        "schema_version": INVENTORY_SCHEMA_VERSION,
        "generated_at": _utcnow(),
        "cached": False,
        "filesystem": _filesystem(data_root),
        "scan": {
            "quality": "partial" if degraded else "complete",
            "cached": False,
            "entries_scanned": budget.entries,
            "max_scan_entries": MAX_SCAN_ENTRIES,
            "max_units_per_category": MAX_UNITS_PER_CATEGORY,
            "truncated": budget.truncated,
            "category_count": len(categories),
            "notes": notes,
        },
        "categories": categories,
    }
    index = {
        "data_root": data_root,
        "config_dir": config_dir,
        "categories": {
            "models": {
                "root": _safe_realpath(os.path.join(data_root, "models")),
                "units": models_index,
            },
            "hf_cache": {"root": _hf_cache_root(), "units": hf_index},
            "engine_installs": {
                "root": _safe_realpath(os.path.join(data_root, "llama-cpp")),
                "units": engine_index,
            },
            "cuda_toolkits": {
                "root": _safe_realpath(os.path.join(data_root, "cuda")),
                "units": cuda_index,
            },
            "jit_caches": {"root": data_root, "units": jit_index},
            "history_backups": {
                "root": _safe_realpath(os.path.join(config_dir, "history")),
                "units": history_index,
            },
            "accounting_db": {"root": config_dir, "units": accounting_index},
        },
    }
    with _INVENTORY_LOCK:
        _INVENTORY_CACHE[key] = (now + INVENTORY_TTL_SECONDS, payload, index)
    return payload, index


def collect_storage_inventory(*, refresh: bool = False) -> dict:
    """Public inventory payload for ``GET /api/storage/inventory``."""
    payload, _index = _scan_storage(refresh=refresh)
    return payload


def storage_inventory_index(*, refresh: bool = False) -> dict:
    """Inventory payload plus the private unit index used by the cleanup service."""
    return _scan_storage(refresh=refresh)
