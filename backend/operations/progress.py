"""SSE-based progress tracking."""

import asyncio
import copy
import json
import threading
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

# Bound per-subscriber memory. Overflow asks the client to resynchronize.
MAX_SUBSCRIBER_QUEUE = 64
MAX_RETAINED_TASKS = 200
HEARTBEAT_SECONDS = 15.0
TERMINAL_STATUSES = {"completed", "failed", "cancelled", "interrupted"}
REPLACEABLE_EVENTS = {"task_updated", "download_progress", "build_progress"}


class ProgressManager:
    """In-memory task tracker with SSE streaming."""

    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._subscribers: list[asyncio.Queue] = []
        self._seq = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def create_task(
        self,
        task_type: str,
        description: str,
        metadata: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Create a new tracked task. Returns task_id (uses provided task_id if given)."""
        task_id = task_id or str(uuid.uuid4())[:8]
        metadata = dict(metadata or {})
        resource_key = str(metadata.get("resource_key") or "").strip()
        self._remember_operation(
            task_id,
            task_type,
            resource_key,
            metadata,
            status="running",
        )
        self._tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "description": description,
            "progress": 0.0,
            "status": "running",
            "message": "",
            "metadata": metadata,
            "created_at": time.time(),
            "updated_at": time.time(),
        }
        self._evict_terminal_tasks()
        self._broadcast({"event": "task_created", "data": self._tasks[task_id]})
        return task_id

    def update_task(
        self,
        task_id: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
        metadata_update: Optional[dict] = None,
        *,
        broadcast: bool = True,
    ):
        """Update a task's progress/status.

        Set ``broadcast=False`` for high-frequency hot paths (e.g. download
        bytes) that already emit a dedicated SSE event such as
        ``download_progress``.
        """
        task = self._tasks.get(task_id)
        if not task:
            return
        if progress is not None:
            # Keep API/SSE payloads human-readable even when callers derive
            # progress from byte ratios or other repeating decimals.
            clamped = min(100.0, max(0.0, float(progress)))
            task["progress"] = round(clamped, 1)
        if message is not None:
            task["message"] = message
        if status is not None:
            task["status"] = status
        if metadata_update:
            task["metadata"].update(metadata_update)
        task["updated_at"] = time.time()
        if status in TERMINAL_STATUSES:
            self._finish_operation(task_id, status, task.get("message") or "")
        if broadcast:
            self._broadcast({"event": "task_updated", "data": task})

    def complete_task(self, task_id: str, message: str = "Done"):
        self.update_task(task_id, progress=100.0, status="completed", message=message)

    def fail_task(self, task_id: str, error: str):
        self.update_task(task_id, status="failed", message=error)

    def get_task(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    def get_active_tasks(self) -> list:
        return [t for t in self._tasks.values() if t["status"] == "running"]

    def snapshot_tasks(self) -> list:
        """Authoritative task list, including recent terminal outcomes."""
        return [copy.deepcopy(task) for task in self._tasks.values()]

    def _remember_operation(
        self,
        task_id: str,
        kind: str,
        resource_key: str,
        metadata: dict,
        *,
        status: str,
    ) -> None:
        from backend.operations.supervisor import get_supervisor

        supervisor = get_supervisor()
        if supervisor.note_known(task_id) and status == "running":
            return
        if status == "running":
            supervisor.start_operation(
                task_id,
                kind,
                resource_key or None,
                resumable=bool(metadata.get("resumable")),
                detail={
                    "engine": metadata.get("engine"),
                    "model_id": metadata.get("model_id") or metadata.get("huggingface_id"),
                },
            )

    def _finish_operation(self, task_id: str, status: str, message: str) -> None:
        from backend.operations.supervisor import get_supervisor

        mapped = {
            "completed": "succeeded",
            "failed": "failed",
            "cancelled": "cancelled",
            "interrupted": "interrupted",
        }.get(status, "failed")
        get_supervisor().finish_operation(task_id, mapped, message)

    def _evict_terminal_tasks(self) -> None:
        if len(self._tasks) <= MAX_RETAINED_TASKS:
            return
        terminal = sorted(
            (
                task
                for task in self._tasks.values()
                if task.get("status") in TERMINAL_STATUSES
            ),
            key=lambda task: task.get("updated_at") or task.get("created_at") or 0,
        )
        while len(self._tasks) > MAX_RETAINED_TASKS and terminal:
            oldest = terminal.pop(0)
            self._tasks.pop(oldest.get("task_id"), None)

    def _prepare_event(self, event: dict) -> dict:
        self._seq += 1
        return {
            "event": event.get("event"),
            "data": copy.deepcopy(event.get("data")),
            "seq": self._seq,
        }

    def _enqueue(self, payload: dict) -> None:
        from backend.ops_metrics import observe_queue_depth

        for queue in self._subscribers:
            observe_queue_depth(queue.qsize())
            try:
                if (
                    payload["event"] in REPLACEABLE_EVENTS
                    and queue.full()
                ):
                    self._signal_overflow(queue)
                    continue
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                self._signal_overflow(queue)

    def _signal_overflow(self, queue: asyncio.Queue) -> None:
        snapshot = {"tasks": self.snapshot_tasks(), "reason": "overflow"}
        overflow = {"event": "resync", "data": snapshot, "seq": self._seq}
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            pass
        try:
            queue.put_nowait(overflow)
        except asyncio.QueueFull:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    def _broadcast(self, event: dict):
        payload = self._prepare_event(event)
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is not None:
            self._loop = running
        loop = self._loop
        if loop is not None and loop.is_closed():
            self._loop = None
            loop = None
        if running is not None or loop is None:
            self._enqueue(payload)
            return
        loop.call_soon_threadsafe(self._enqueue, payload)

    def emit(self, event_type: str, data: Any):
        """Emit a generic event (e.g. log, notification, model_status) to SSE subscribers."""
        self._broadcast({"event": event_type, "data": data})

    @property
    def active_connections(self) -> List:
        """SSE has no persistent connection list; returns empty."""
        return []

    async def send_download_progress(
        self,
        task_id: str,
        progress: int,
        message: str = "",
        bytes_downloaded: int = 0,
        total_bytes: int = 0,
        speed_mbps: float = 0,
        eta_seconds: int = 0,
        filename: str = "",
        model_format: str = "gguf",
        files_completed: int = None,
        files_total: int = None,
        current_filename: str = None,
        huggingface_id: str = None,
        **kwargs,
    ):
        # Keep in-memory task state current for reconnect, but do not also
        # broadcast task_updated — the dedicated download_progress event is
        # enough and avoids double Pinia writes every tick.
        self.update_task(
            task_id,
            progress=float(progress),
            message=message or filename,
            metadata_update={
                "bytes_downloaded": bytes_downloaded,
                "total_bytes": total_bytes,
                "speed_mbps": speed_mbps,
                "eta_seconds": eta_seconds,
                "filename": filename,
                "model_format": model_format,
                "files_completed": files_completed,
                "files_total": files_total,
                "current_filename": current_filename or filename,
                "huggingface_id": huggingface_id,
                **kwargs,
            },
            broadcast=False,
        )
        self.emit(
            "download_progress",
            {
                "task_id": task_id,
                "progress": progress,
                "message": message,
                "bytes_downloaded": bytes_downloaded,
                "total_bytes": total_bytes,
                "speed_mbps": speed_mbps,
                "eta_seconds": eta_seconds,
                "filename": filename,
                "model_format": model_format,
                "files_completed": files_completed,
                "files_total": files_total,
                "current_filename": current_filename or filename,
                "huggingface_id": huggingface_id,
                "timestamp": datetime.utcnow().isoformat(),
                **kwargs,
            },
        )

    async def broadcast(self, message: dict):
        msg_type = message.get("type", "broadcast")
        self.emit(msg_type, message)

    async def send_model_status_update(
        self, model_id: Any, status: str, details: dict = None
    ):
        self.emit(
            "model_status",
            {
                "model_id": model_id,
                "status": status,
                "details": details or {},
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def send_notification(
        self,
        title: str = "",
        message: str = "",
        type: str = "info",
        actions: List[dict] = None,
        *args,
        **kwargs,
    ):
        # Support (title, message, type) keyword and (type, title, message, task_id) positional
        if args and len(args) >= 3:
            type, title, message = args[0], args[1], args[2]
        else:
            type = kwargs.get("type", type)
            title = kwargs.get("title", title)
            message = kwargs.get("message", message)
        self.emit(
            "notification",
            {
                "title": title,
                "message": message,
                "type": type,
                "notification_type": type,
                "actions": actions or [],
                "timestamp": datetime.utcnow().isoformat(),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k not in ("title", "message", "type", "actions")
                },
            },
        )

    def send_build_progress_now(
        self,
        task_id: str,
        stage: str,
        progress: int,
        message: str = "",
        log_lines: List[str] = None,
    ):
        """Sync counterpart of :meth:`send_build_progress` for worker threads."""
        self.update_task(
            task_id,
            progress=float(progress),
            message=message,
            metadata_update={"stage": stage, "log_lines": log_lines or []},
        )
        self.emit(
            "build_progress",
            {
                "task_id": task_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                "log_lines": log_lines or [],
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    async def send_build_progress(
        self,
        task_id: str,
        stage: str,
        progress: int,
        message: str = "",
        log_lines: List[str] = None,
    ):
        self.send_build_progress_now(
            task_id, stage, progress, message, log_lines
        )

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted strings. Sends an initial comment so the client connection opens."""
        self._loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_SUBSCRIBER_QUEUE)
        self._subscribers.append(queue)
        try:
            yield ": heartbeat\n\n"
            await asyncio.sleep(0)
            snapshot = self.snapshot_tasks()
            yield self._format_sse(
                "task_snapshot", {"tasks": snapshot, "reason": "connect"}
            )
            for task in snapshot:
                yield self._format_sse("task_updated", task)
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except asyncio.TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                yield self._format_sse(event["event"], event["data"])
        except asyncio.CancelledError:
            pass
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    @staticmethod
    def _format_sse(event_name: str, data: Any) -> str:
        return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager
