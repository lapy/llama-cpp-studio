"""In-process supervisor for builds, downloads, and installs.

Restart marks in-flight work interrupted. Only operations explicitly marked
resumable are eligible to continue; everything else keeps its partial files and
a retry explanation.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted"}
ACTIVE_STATES = {"queued", "running", "cancelling"}


class ResourceBusyError(RuntimeError):
    def __init__(self, resource_key: str, owner: str):
        super().__init__(
            f"{resource_key} is already in use by operation {owner}"
        )
        self.resource_key = resource_key
        self.owner = owner


class OperationSupervisor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources: Dict[str, str] = {}

    def start_operation(
        self,
        operation_id: str,
        kind: str,
        resource_key: Optional[str] = None,
        *,
        resumable: bool = False,
        detail: Optional[dict] = None,
    ) -> dict:
        record = {
            "operation_id": operation_id,
            "kind": kind,
            "status": "running",
            "resource_key": resource_key or "",
            "resumable": bool(resumable),
            "detail": detail or {},
            "message": "",
            "updated_at": time.time(),
        }
        with self._lock:
            if resource_key:
                owner = self._resources.get(resource_key)
                if owner and owner != operation_id:
                    raise ResourceBusyError(resource_key, owner)
                self._resources[resource_key] = operation_id
        self._persist(record)
        from backend.ops_metrics import record_operation

        record_operation("running")
        logger.info(
            "operation started",
            extra={
                "operation_id": operation_id,
                "task_id": operation_id,
                "engine": (detail or {}).get("engine"),
                "model_id": (detail or {}).get("model_id"),
            },
        )
        return record

    def finish_operation(
        self,
        operation_id: str,
        status: str,
        message: str = "",
    ) -> None:
        if status not in TERMINAL_STATES:
            status = "failed"
        with self._lock:
            release = [
                key for key, owner in self._resources.items() if owner == operation_id
            ]
            for key in release:
                self._resources.pop(key, None)
        current = self._get(operation_id) or {
            "operation_id": operation_id,
            "kind": "operation",
            "resource_key": "",
            "resumable": False,
            "detail": {},
        }
        current["status"] = status
        current["message"] = message
        current["updated_at"] = time.time()
        self._persist(current)
        from backend.ops_metrics import record_operation

        record_operation(status)

    def note_known(self, operation_id: str) -> bool:
        with self._lock:
            return operation_id in set(self._resources.values()) or bool(self._get(operation_id))

    def reconcile_startup(self) -> int:
        """Mark operations that died with the process as interrupted, then repair engine rows."""
        from backend.data_store import get_store
        from backend.engines.lifecycle import repair_stale_building_versions

        changed = 0
        store = get_store()
        try:
            rows = store.list_operations()
        except Exception as exc:
            logger.warning("Could not read durable operations: %s", exc)
            rows = []
        for row in rows:
            status = str(row.get("status") or "")
            if status not in ACTIVE_STATES:
                continue
            if row.get("resumable"):
                row["status"] = "queued"
                row["message"] = "Operation is resumable and was requeued after restart"
            else:
                row["status"] = "interrupted"
                row["message"] = (
                    "Operation was interrupted by a restart. Retry it, or remove partial files if you want a clean install."
                )
            row["updated_at"] = time.time()
            try:
                store.upsert_operation(row)
                changed += 1
            except Exception as exc:
                logger.warning(
                    "Could not reconcile operation %s: %s",
                    row.get("operation_id"),
                    exc,
                )
        with self._lock:
            self._resources.clear()
        try:
            changed += repair_stale_building_versions(store, get_task=lambda _task_id: None)
        except Exception as exc:
            logger.warning("Could not repair stale engine builds: %s", exc)
        return changed

    def _persist(self, record: dict) -> None:
        from backend.data_store import get_store

        get_store().upsert_operation(record)

    def _get(self, operation_id: str) -> Optional[dict]:
        from backend.data_store import get_store

        try:
            rows = get_store().list_operations()
        except Exception:
            return None
        for row in rows:
            if str(row.get("operation_id") or "") == operation_id:
                return dict(row)
        return None


_supervisor: Optional[OperationSupervisor] = None


def get_supervisor() -> OperationSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = OperationSupervisor()
    return _supervisor
