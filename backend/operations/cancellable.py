"""Base class for long-running operations tracked in ProgressManager."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Optional

from backend.operations.cancel import cancel_running_operation
from backend.operations.progress import get_progress_manager
from backend.task_cancel_registry import register_task_cancel, unregister_task_cancel


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class CancelResult(Dict[str, Any]):
    """Typed-ish dict returned by cancel_task helpers."""


class CancellableOperationManager:
    """Owns one in-flight operation with a real ProgressManager task_id."""

    MANAGER_NAME: str = ""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._operation: Optional[str] = None
        self._operation_started_at: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._active_process: Optional[asyncio.subprocess.Process] = None
        self._progress_task_id: Optional[str] = None
        self._last_error: Optional[str] = None
        self._cancelling = False
        self._cancellation_task: Optional[asyncio.Task] = None

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
        task_id = self._make_task_id(operation)
        meta = {
            "manager": self.MANAGER_NAME,
            "operation": operation,
            **(metadata or {}),
        }
        if not meta.get("resource_key"):
            for attribute in ("_root_dir", "_base_dir", "_cuda_install_dir"):
                resource = getattr(self, attribute, None)
                if resource:
                    meta["resource_key"] = str(resource)
                    break
        get_progress_manager().create_task("install", description, meta, task_id=task_id)
        self._operation = operation
        self._operation_started_at = _utcnow()
        self._last_error = None
        register_task_cancel(task_id)
        self._progress_task_id = task_id
        return task_id

    async def _finish_operation(
        self, success: bool, message: str = "", *, cancelled: bool = False
    ) -> None:
        if self._cancelling and not cancelled:
            return
        task_id = self._progress_task_id
        pm = get_progress_manager()
        if task_id:
            if cancelled:
                pm.update_task(
                    task_id,
                    status="cancelled",
                    message=message or "Operation cancelled by user",
                )
            elif success:
                pm.complete_task(task_id, message or "Done")
            else:
                pm.fail_task(task_id, message or "Failed")
            unregister_task_cancel(task_id)
        self._operation = None
        self._operation_started_at = None
        self._progress_task_id = None
        self._cancelling = False

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

    def _create_task(self, coro: Awaitable[Any]) -> None:
        async def _wrapped() -> None:
            try:
                await coro
            except asyncio.CancelledError:
                self._last_error = "Operation cancelled by user"
                if not self._cancelling:
                    try:
                        await self._finish_operation(
                            False, "Operation cancelled by user", cancelled=True
                        )
                    except Exception:
                        pass
                raise
            finally:
                if not self._cancelling:
                    self._clear_active_process()

        from backend.operations.supervisor import get_supervisor

        operation_id = self._progress_task_id or self._make_task_id(
            self._operation or "operation"
        )
        task = get_supervisor().spawn(
            operation_id, _wrapped(), cancel=self._cancel_for_shutdown
        )
        self._current_task = task

        def _cleanup(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except asyncio.CancelledError:
                pass
            except Exception as exc:
                self._on_task_error(exc)
            finally:
                if not self._cancelling:
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
        if tracked and tracked.get("status") not in {"running", "cancelling"}:
            return {"ok": False, "message": "Task is not running."}
        if self._cancelling:
            return {
                "ok": True,
                "message": "Cancellation is already in progress.",
                "task_id": task_id,
            }

        self._cancelling = True
        if tracked:
            pm.update_task(
                task_id,
                status="cancelling",
                message="Stopping the operation and its subprocesses…",
            )
        operation = self._operation
        current_task = self._current_task
        active_process = self._active_process

        async def _cancel_and_finish() -> None:
            try:
                cancelled = await cancel_running_operation(
                    operation=operation,
                    current_task=current_task,
                    active_process=active_process,
                )
                message = (
                    "Operation cancelled by user"
                    if cancelled
                    else "Operation ended before cancellation completed"
                )
                self._last_error = message
                await self._finish_operation(False, message, cancelled=True)
            finally:
                self._current_task = None
                self._clear_active_process()
                self._cancellation_task = None

        self._cancellation_task = asyncio.get_running_loop().create_task(
            _cancel_and_finish()
        )
        from backend.operations.supervisor import get_supervisor

        get_supervisor().track_cleanup(task_id, self._cancellation_task)
        return {
            "ok": True,
            "message": "Cancellation requested; the operation will stop shortly.",
            "task_id": task_id,
        }

    async def wait_for_cancellation(self) -> None:
        task = self._cancellation_task
        if task is not None:
            await asyncio.shield(task)

    async def _cancel_for_shutdown(self) -> None:
        task_id = self._progress_task_id
        if task_id and not self._cancelling:
            self.cancel_task(task_id)
        await self.wait_for_cancellation()

    def _started_response(self, message: str, **extra: Any) -> Dict[str, Any]:
        body: Dict[str, Any] = {"message": message, "task_id": self._progress_task_id}
        body.update(extra)
        return body
