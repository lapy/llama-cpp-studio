"""Append-only operation history for the development control plane."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from .locks import FileLock, atomic_write_json

MAX_JOURNAL_BYTES = 10 * 1024 * 1024
MAX_READ_RECORDS = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def new_operation_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"op-{stamp}-{uuid.uuid4().hex[:10]}"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"), sort_keys=True)


class Journal:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = _safe_json(record) + "\n"
        with FileLock(self.lock_path, blocking=True):
            self._rotate_if_needed()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)

    def _rotate_if_needed(self) -> None:
        try:
            if self.path.stat().st_size < MAX_JOURNAL_BYTES:
                return
        except OSError:
            return
        for index in (2, 1):
            old = self.path.with_name(f"{self.path.name}.{index}")
            newer = self.path.with_name(f"{self.path.name}.{index + 1}")
            if old.exists():
                try:
                    os.replace(old, newer)
                except OSError:
                    pass
        try:
            os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        except OSError:
            pass

    def read(self, limit: int = 100) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), MAX_READ_RECORDS))
        if not self.path.exists():
            return []
        try:
            size = self.path.stat().st_size
            with open(self.path, "rb") as handle:
                # Reading the tail is enough for an operation history view.
                handle.seek(max(0, size - 4 * 1024 * 1024))
                raw = handle.read().decode("utf-8", errors="replace")
        except OSError:
            return []
        records: list[dict[str, Any]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                records.append(value)
        return records[-limit:]

    def latest_for(self, worktree: Optional[str] = None) -> Optional[dict[str, Any]]:
        for record in reversed(self.read(MAX_READ_RECORDS)):
            if worktree is None or record.get("worktree") == worktree:
                return record
        return None


class CheckpointStore:
    """Metadata for git checkpoint refs, kept separately from the checkout."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read(self) -> dict[str, Any]:
        from .locks import read_json

        value = read_json(self.path, {})
        return value if isinstance(value, dict) else {}

    def list(self, worktree: Optional[str] = None) -> list[dict[str, Any]]:
        items = [item for item in self._read().get("checkpoints", []) if isinstance(item, dict)]
        if worktree is not None:
            items = [item for item in items if item.get("worktree") == worktree]
        return sorted(items, key=lambda item: str(item.get("created_at") or ""), reverse=True)

    def latest(self, worktree: str) -> Optional[dict[str, Any]]:
        items = self.list(worktree)
        return items[0] if items else None

    def get(
        self, checkpoint_id: str, *, worktree: Optional[str] = None
    ) -> Optional[dict[str, Any]]:
        for item in self.list(worktree):
            if item.get("id") == checkpoint_id:
                return item
        return None

    def add(self, item: dict[str, Any], *, keep: int = 25) -> None:
        with FileLock(self.lock_path, blocking=True):
            data = self._read()
            items = [entry for entry in data.get("checkpoints", []) if isinstance(entry, dict)]
            items = [entry for entry in items if entry.get("id") != item.get("id")]
            items.append(item)
            items.sort(key=lambda entry: str(entry.get("created_at") or ""), reverse=True)
            worktree = item.get("worktree")
            if worktree:
                same = [entry for entry in items if entry.get("worktree") == worktree]
                drop = {entry.get("id") for entry in same[keep:]}
                items = [entry for entry in items if entry.get("id") not in drop]
            atomic_write_json(self.path, {"schema": 1, "checkpoints": items})

    def remove(self, checkpoint_id: str) -> None:
        with FileLock(self.lock_path, blocking=True):
            data = self._read()
            items = [
                entry
                for entry in data.get("checkpoints", [])
                if isinstance(entry, dict) and entry.get("id") != checkpoint_id
            ]
            atomic_write_json(self.path, {"schema": 1, "checkpoints": items})


def operation_record(
    *,
    operation_id: str,
    action: str,
    actor: str,
    worktree: str,
    revision_before: Any,
    revision_after: Any,
    status: str,
    request: Optional[dict[str, Any]] = None,
    result: Optional[dict[str, Any]] = None,
    error: Optional[dict[str, Any]] = None,
    started_at: str,
) -> dict[str, Any]:
    return {
        "schema": 1,
        "operation_id": operation_id,
        "action": action,
        "actor": actor,
        "worktree": worktree,
        "status": status,
        "revision_before": revision_before,
        "revision_after": revision_after,
        "started_at": started_at,
        "finished_at": utc_now(),
        "duration_seconds": round((time.time() - _timestamp(started_at)), 3),
        "request": request or {},
        "result": result or {},
        "error": error,
    }


def _timestamp(value: str) -> float:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except Exception:
        return time.time()


def compact_records(records: Iterable[dict[str, Any]], limit: int = 50) -> list[dict[str, Any]]:
    compact = []
    for record in list(records)[-limit:]:
        compact.append(
            {
                "operation_id": record.get("operation_id"),
                "action": record.get("action"),
                "actor": record.get("actor"),
                "worktree": record.get("worktree"),
                "status": record.get("status"),
                "revision_before": record.get("revision_before"),
                "revision_after": record.get("revision_after"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
            }
        )
    return compact
