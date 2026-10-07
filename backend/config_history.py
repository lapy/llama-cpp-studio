"""Durable, local revision history for user-managed configuration documents."""

from __future__ import annotations

import copy
import os
import re
import tempfile
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Mapping

import yaml

from backend.data_store import DataStore
from backend.store_io import StoreDurabilityError


HISTORY_SCHEMA_VERSION = 1
MAX_HISTORY_ENTRIES = 100
HISTORY_DOCUMENTS = {
    "settings.yaml",
    "models.yaml",
    "model_config_templates.yaml",
    "llama_swap_routing.yaml",
}
_SAFE_ID = re.compile(r"^[a-f0-9-]{36}$")
_SENSITIVE = re.compile(
    r"token|password|passwd|secret|api[_-]?key|authorization|credential|private_key",
    re.IGNORECASE,
)
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class ConfigHistoryError(ValueError):
    def __init__(self, code: str, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


def record_document_snapshot(
    store: DataStore,
    filename: str,
    snapshot: Mapping[str, Any],
    *,
    reason: str = "configuration change",
) -> str | None:
    """Persist a pre-change document before its replacement is attempted."""
    if filename not in HISTORY_DOCUMENTS:
        return None
    history_dir = _history_dir(store)
    os.makedirs(history_dir, mode=0o700, exist_ok=True)
    try:
        os.chmod(history_dir, 0o700)
    except OSError:
        pass
    entry_id = str(uuid.uuid4())
    entry = {
        "schema_version": HISTORY_SCHEMA_VERSION,
        "id": entry_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_at_ns": time.time_ns(),
        "document": filename,
        "reason": str(reason or "configuration change")[:160],
        "snapshot": copy.deepcopy(dict(snapshot)),
    }
    _atomic_write(os.path.join(history_dir, f"{entry['created_at_ns']}-{entry_id}.yaml"), entry)
    _prune(history_dir)
    return entry_id


def list_config_history(store: DataStore, *, limit: int = 30) -> list[dict]:
    entries = []
    for path in _entry_paths(store):
        entry = _read_entry(path)
        if entry is None:
            continue
        entries.append({key: entry[key] for key in ("id", "created_at", "document", "reason")})
        if len(entries) >= max(1, min(int(limit), MAX_HISTORY_ENTRIES)):
            break
    return entries


def config_history_diff(store: DataStore, entry_id: str) -> dict:
    entry = _entry(store, entry_id)
    filename = entry["document"]
    current = store._read_yaml(filename)
    changes: list[dict] = []
    _diff(entry["snapshot"], current, (), changes)
    return {
        "id": entry["id"],
        "created_at": entry["created_at"],
        "document": filename,
        "reason": entry["reason"],
        "current_revision": store.document_revision(filename),
        "changes": changes,
    }


def restore_config_history_item(
    store: DataStore,
    entry_id: str,
    *,
    kind: str,
    item_id: str,
    expected_revision: str,
) -> dict:
    """Restore one preference, model config, template, profile, or selector."""
    entry = _entry(store, entry_id)
    filename = entry["document"]
    snapshot = entry["snapshot"]
    key = str(item_id or "").strip()
    if not key:
        raise ConfigHistoryError("HISTORY_MALFORMED", "A history item is required.")

    if kind == "preference" and filename == "settings.yaml":
        if _SENSITIVE.search(key):
            raise ConfigHistoryError(
                "HISTORY_FORBIDDEN", "Credentials cannot be restored from this screen.", 403
            )

        def mutate(document):
            if key in snapshot:
                document[key] = copy.deepcopy(snapshot[key])
            else:
                document.pop(key, None)

    elif kind == "model" and filename == "models.yaml":
        old = _find(snapshot.get("models"), key)
        if old is None:
            raise ConfigHistoryError(
                "HISTORY_NOT_FOUND", "The model is not present in this revision.", 404
            )

        def mutate(document):
            current = _find(document.get("models"), key)
            if current is None:
                raise ConfigHistoryError(
                    "HISTORY_NOT_FOUND", "The current model no longer exists.", 404
                )
            current["config"] = copy.deepcopy(old.get("config") or {})

    elif kind == "template" and filename == "model_config_templates.yaml":
        old = _find(snapshot.get("templates"), key)

        def mutate(document):
            rows = document.setdefault("templates", [])
            current = _find_index(rows, key)
            if old is None and current is not None:
                rows.pop(current)
            elif old is not None and current is None:
                rows.append(copy.deepcopy(old))
            elif old is not None and current is not None:
                rows[current] = copy.deepcopy(old)

    elif kind in {"profile", "selector"} and filename == "llama_swap_routing.yaml":
        bucket = "profiles" if kind == "profile" else "selectors"
        old_bucket = snapshot.get(bucket) if isinstance(snapshot.get(bucket), dict) else {}

        def mutate(document):
            current = document.setdefault(bucket, {})
            if key in old_bucket:
                current[key] = copy.deepcopy(old_bucket[key])
            else:
                current.pop(key, None)

    else:
        raise ConfigHistoryError(
            "HISTORY_SCOPE_MISMATCH",
            "That item does not belong to this history revision.",
            409,
        )

    # Keep the optimistic revision check and replacement in one process and
    # interprocess critical section. Otherwise a save could land between the
    # check and the mutation and have one of its items silently overwritten.
    with store.exclusive_documents():
        if store.document_revision(filename) != expected_revision:
            raise ConfigHistoryError(
                "HISTORY_STALE",
                "Configuration changed after the history preview. Refresh the comparison.",
                409,
            )
        store._mutate(filename, mutate)
        revision = store.document_revision(filename)
    return {
        "outcome": "completed",
        "document": filename,
        "kind": kind,
        "item_id": key,
        "revision": revision,
        "notice": "Saved configuration changed. Running models were not restarted or published.",
    }


def _history_dir(store: DataStore) -> str:
    return os.path.join(store._config_dir, "history")


def _entry_paths(store: DataStore) -> list[str]:
    directory = _history_dir(store)
    if not os.path.isdir(directory):
        return []
    return sorted(
        (os.path.join(directory, name) for name in os.listdir(directory) if name.endswith(".yaml")),
        reverse=True,
    )


def _prune(history_dir: str) -> None:
    paths = sorted(
        (
            os.path.join(history_dir, name)
            for name in os.listdir(history_dir)
            if name.endswith(".yaml")
        ),
        reverse=True,
    )
    removed = False
    for path in paths[MAX_HISTORY_ENTRIES:]:
        os.remove(path)
        removed = True
    if removed:
        directory_fd = os.open(history_dir, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def _atomic_write(path: str, entry: dict) -> None:
    directory = os.path.dirname(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".history-", suffix=".tmp", dir=directory)
    committed = False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            yaml.safe_dump(entry, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
        committed = True
        directory_fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as exc:
        if not committed:
            try:
                os.remove(temporary)
            except OSError:
                pass
        raise StoreDurabilityError(
            f"Could not preserve configuration history: {exc}",
            phase="history",
            committed=False if not committed else True,
        ) from exc


def _entry(store: DataStore, entry_id: str) -> dict:
    if not _SAFE_ID.fullmatch(str(entry_id or "")):
        raise ConfigHistoryError("HISTORY_NOT_FOUND", "History revision not found.", 404)
    for path in _entry_paths(store):
        if path.endswith(f"-{entry_id}.yaml"):
            entry = _read_entry(path)
            if entry is not None:
                return entry
    raise ConfigHistoryError("HISTORY_NOT_FOUND", "History revision not found.", 404)


def _read_entry(path: str) -> dict | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            entry = yaml.load(handle, Loader=_LOADER)
    except (OSError, yaml.YAMLError):
        return None
    if not isinstance(entry, dict) or entry.get("schema_version") != HISTORY_SCHEMA_VERSION:
        return None
    if entry.get("document") not in HISTORY_DOCUMENTS or not isinstance(
        entry.get("snapshot"), dict
    ):
        return None
    if not _SAFE_ID.fullmatch(str(entry.get("id") or "")):
        return None
    return entry


def _find(rows: Any, item_id: str) -> dict | None:
    if not isinstance(rows, list):
        return None
    return next(
        (row for row in rows if isinstance(row, dict) and str(row.get("id") or "") == item_id),
        None,
    )


def _find_index(rows: list, item_id: str) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(rows)
            if isinstance(row, dict) and str(row.get("id") or "") == item_id
        ),
        None,
    )


def _diff(before: Any, after: Any, path: tuple[str, ...], changes: list[dict]) -> None:
    if isinstance(before, dict) and isinstance(after, dict):
        for key in sorted(set(before) | set(after), key=str):
            _diff(before.get(key, _MISSING), after.get(key, _MISSING), (*path, str(key)), changes)
        return
    if isinstance(before, list) and isinstance(after, list):
        before_by_id = _by_id(before)
        after_by_id = _by_id(after)
        if before_by_id is not None and after_by_id is not None:
            for key in sorted(set(before_by_id) | set(after_by_id)):
                _diff(
                    before_by_id.get(key, _MISSING),
                    after_by_id.get(key, _MISSING),
                    (*path, key),
                    changes,
                )
            return
    if before == after:
        return
    sensitive = any(_SENSITIVE.search(part) for part in path)
    changes.append(
        {
            "path": ".".join(path) or "$",
            "before": _display(before, sensitive),
            "after": _display(after, sensitive),
        }
    )


def _by_id(rows: list) -> dict | None:
    if not all(isinstance(row, dict) and row.get("id") for row in rows):
        return None
    return {str(row["id"]): row for row in rows}


def _display(value: Any, sensitive: bool) -> Any:
    if value is _MISSING:
        return "[absent]"
    if sensitive:
        return "[redacted]"
    if isinstance(value, (str, int, float, bool)) or value is None:
        text = str(value)
        return text if len(text) <= 160 else text[:157] + "..."
    if isinstance(value, list):
        return f"[{len(value)} items]"
    if isinstance(value, dict):
        return f"{{{len(value)} fields}}"
    return f"[{type(value).__name__}]"


_MISSING = object()
