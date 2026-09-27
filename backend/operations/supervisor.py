"""In-process supervisor for builds, downloads, and installs.

Restart marks in-flight work interrupted. Only operations explicitly marked
resumable are eligible to continue; everything else keeps its partial files and
a retry explanation.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional

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
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancellers: Dict[str, Callable[[], Awaitable[None]]] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()

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

    def spawn(
        self,
        operation_id: str,
        awaitable: Awaitable[Any],
        *,
        cancel: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> asyncio.Task:
        """Own an operation coroutine until it finishes or shutdown drains it."""
        with self._lock:
            existing = self._tasks.get(operation_id)
            if existing is not None and not existing.done():
                if hasattr(awaitable, "close"):
                    awaitable.close()  # type: ignore[attr-defined]
                raise RuntimeError(f"Operation {operation_id} is already running")
            task = asyncio.get_running_loop().create_task(awaitable)
            self._tasks[operation_id] = task
            if cancel is not None:
                self._cancellers[operation_id] = cancel

        def _done(finished: asyncio.Task) -> None:
            with self._lock:
                if self._tasks.get(operation_id) is finished:
                    self._tasks.pop(operation_id, None)
                    self._cancellers.pop(operation_id, None)
            if finished.cancelled():
                return
            try:
                exc = finished.exception()
            except asyncio.CancelledError:
                return
            if exc is None:
                return
            logger.exception(
                "supervised operation failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={"operation_id": operation_id, "task_id": operation_id},
            )
            try:
                from backend.operations.progress import get_progress_manager

                progress = get_progress_manager()
                tracked = progress.get_task(operation_id)
                if tracked and tracked.get("status") not in {
                    "completed",
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    progress.fail_task(operation_id, str(exc))
            except Exception:
                logger.exception(
                    "failed to publish supervised operation failure",
                    extra={"operation_id": operation_id},
                )

        task.add_done_callback(_done)
        return task

    def track_cleanup(self, operation_id: str, task: asyncio.Task) -> None:
        """Own cancellation/process cleanup independently of the worker task."""
        with self._lock:
            self._cleanup_tasks.add(task)

        def _done(finished: asyncio.Task) -> None:
            with self._lock:
                self._cleanup_tasks.discard(finished)
            if finished.cancelled():
                return
            try:
                exc = finished.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.error(
                    "operation cleanup failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                    extra={"operation_id": operation_id, "task_id": operation_id},
                )

        task.add_done_callback(_done)

    async def drain(self, *, cancel: bool = True) -> None:
        """Wait for every owned operation; cancel them first during shutdown."""
        with self._lock:
            items = [
                (operation_id, task, self._cancellers.get(operation_id))
                for operation_id, task in self._tasks.items()
                if not task.done()
            ]
        if cancel:
            cancellations = [callback() for _, _, callback in items if callback]
            if cancellations:
                await asyncio.gather(*cancellations, return_exceptions=True)
            for _, task, callback in items:
                if callback is None and not task.done():
                    task.cancel()
        tasks = [task for _, task, _ in items]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        while True:
            with self._lock:
                cleanups = [
                    task for task in self._cleanup_tasks if not task.done()
                ]
            if not cleanups:
                break
            await asyncio.gather(*cleanups, return_exceptions=True)

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
        try:
            from backend.operations.progress import get_progress_manager

            get_progress_manager().restore_operations(store.list_operations())
        except Exception as exc:
            logger.warning("Could not restore reconciled progress outcomes: %s", exc)
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
