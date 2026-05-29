"""Base class for long-running operations tracked in ProgressManager."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Optional

from backend.operation_cancel import cancel_running_operation
from backend.progress_manager import get_progress_manager
from backend.task_cancel_registry import register_task_cancel, unregister_task_cancel


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CancelResult(Dict[str, Any]):
    """Typed-ish dict returned by cancel_task helpers."""


class CancellableOperationManager:
    """Owns one in-flight operation with a real ProgressManager task_id."""

    MANAGER_NAME: str = ""

    LEGACY_STATUS_EVENT: str = ""
    LEGACY_LOG_EVENT: str = ""
    LEGACY_PROGRESS_EVENT: str = ""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._operation: Optional[str] = None
        self._operation_started_at: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._active_process: Optional[asyncio.subprocess.Process] = None
        self._progress_task_id: Optional[str] = None
        self._last_error: Optional[str] = None

    @property
    def progress_task_id(self) -> Optional[str]:
        return self._progress_task_id

    def is_operation_running(self) -> bool:
        return self._operation is not None

    def _make_task_id(self, operation: str) -> str:
        return f"install_{self.MANAGER_NAME}_{operation}_{int(time.time() * 1000)}"

    async def _begin_operation(
        self,
        operation: str,
        description: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._operation = operation
        self._operation_started_at = _utcnow()
        self._last_error = None
        task_id = self._make_task_id(operation)
        meta = {
            "manager": self.MANAGER_NAME,
            "operation": operation,
            **(metadata or {}),
        }
        get_progress_manager().create_task(
            "install",
            description,
            meta,
            task_id=task_id,
        )
        register_task_cancel(task_id)
        self._progress_task_id = task_id
        await self._emit_legacy_status(
            {
                "status": operation,
                "operation": operation,
                "started_at": self._operation_started_at,
            }
        )
        return task_id

    async def _finish_operation(self, success: bool, message: str = "") -> None:
        task_id = self._progress_task_id
        operation = self._operation
        pm = get_progress_manager()
        if task_id:
            if success:
                pm.complete_task(task_id, message or "Done")
            else:
                pm.fail_task(task_id, message or "Failed")
            unregister_task_cancel(task_id)
        await self._emit_legacy_status(
            {
                "status": "completed" if success else "failed",
                "operation": operation,
                "message": message,
                "ended_at": _utcnow(),
            }
        )
        self._operation = None
        self._operation_started_at = None
        self._progress_task_id = None

    async def _update_progress_task(
        self,
        progress: float,
        message: str = "",
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._progress_task_id:
            return
        get_progress_manager().update_task(
            self._progress_task_id,
            progress=progress,
            message=message,
            metadata_update=metadata_update,
        )

    async def _append_task_log(self, line: str) -> None:
        if not line or not self._progress_task_id:
            return
        get_progress_manager().emit(
            "task_log",
            {
                "task_id": self._progress_task_id,
                "line": line,
                "timestamp": _utcnow(),
            },
        )

    async def _emit_legacy_status(self, payload: Dict[str, Any]) -> None:
        if not self.LEGACY_STATUS_EVENT:
            return
        body = {
            "type": self.LEGACY_STATUS_EVENT,
            **payload,
        }
        if self._progress_task_id:
            body["task_id"] = self._progress_task_id
        await get_progress_manager().broadcast(body)

    async def _emit_legacy_log(self, line: str) -> None:
        if not self.LEGACY_LOG_EVENT:
            return
        body = {
            "type": self.LEGACY_LOG_EVENT,
            "line": line,
            "timestamp": _utcnow(),
        }
        if self._progress_task_id:
            body["task_id"] = self._progress_task_id
        await get_progress_manager().broadcast(body)

    async def _emit_legacy_progress(self, payload: Dict[str, Any]) -> None:
        if not self.LEGACY_PROGRESS_EVENT:
            return
        body = {
            "type": self.LEGACY_PROGRESS_EVENT,
            **payload,
            "timestamp": _utcnow(),
        }
        if self._progress_task_id:
            body["task_id"] = self._progress_task_id
        await get_progress_manager().broadcast(body)

    def _create_task(self, coro: Awaitable[Any]) -> None:
        async def _wrapped() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                self._last_error = "Operation cancelled by user"
                try:
                    await self._finish_operation(False, "Operation cancelled by user")
                except Exception:
                    pass
                raise
            finally:
                self._clear_active_process()

        loop = asyncio.get_running_loop()
        task = loop.create_task(_wrapped())
        self._current_task = task

        def _cleanup(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._on_task_error(exc)
            finally:
                self._current_task = None
                self._clear_active_process()

        task.add_done_callback(_cleanup)

    def _on_task_error(self, exc: Exception) -> None:
        """Override in subclasses for custom error logging."""

    def _clear_active_process(self) -> None:
        self._active_process = None

    def _track_process(self, process: asyncio.subprocess.Process) -> None:
        self._active_process = process

    def cancel_task(self, task_id: str) -> CancelResult:
        task_id = str(task_id or "").strip()
        if not task_id:
            return {"ok": False, "message": "task_id is required"}
        if not self._progress_task_id or self._progress_task_id != task_id:
            return {"ok": False, "message": "No active operation for that task_id."}
        if not self._operation:
            return {"ok": False, "message": "Operation is not running."}
        pm = get_progress_manager()
        tracked = pm.get_task(task_id)
        if tracked and tracked.get("status") != "running":
            return {"ok": False, "message": "Task is not running."}
        cancelled = cancel_running_operation(
            operation=self._operation,
            current_task=self._current_task,
            active_process=self._active_process,
        )
        if not cancelled:
            return {"ok": False, "message": "Could not cancel the operation."}
        return {
            "ok": True,
            "message": "Cancellation requested; the operation will stop shortly.",
            "task_id": task_id,
        }

    def _started_response(self, message: str, **extra: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = {"message": message, "task_id": self._progress_task_id}
        body.update(extra)
        return body
