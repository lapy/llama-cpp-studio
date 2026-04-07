"""SSE-based progress tracking."""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional


class ProgressManager:
    """In-memory task tracker with SSE streaming."""

    def __init__(self):
        self._tasks: Dict[str, dict] = {}
        self._subscribers: list[asyncio.Queue] = []

    def create_task(
        self,
        task_type: str,
        description: str,
        metadata: Optional[dict] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Create a new tracked task. Returns task_id (uses provided task_id if given)."""
        task_id = task_id or str(uuid.uuid4())[:8]
        self._tasks[task_id] = {
            "task_id": task_id,
            "type": task_type,
            "description": description,
            "progress": 0.0,
            "status": "running",
            "message": "",
            "metadata": metadata or {},
            "created_at": time.time(),
        }
        self._broadcast({"event": "task_created", "data": self._tasks[task_id]})
        return task_id

    def update_task(
        self,
        task_id: str,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        status: Optional[str] = None,
        metadata_update: Optional[dict] = None,
    ):
        """Update a task's progress/status."""
        task = self._tasks.get(task_id)
        if not task:
            return
        if progress is not None:
            task["progress"] = min(100.0, max(0.0, progress))
        if message is not None:
            task["message"] = message
        if status is not None:
            task["status"] = status
        if metadata_update:
            task["metadata"].update(metadata_update)
        self._broadcast({"event": "task_updated", "data": task})

    def complete_task(self, task_id: str, message: str = "Done"):
        self.update_task(task_id, progress=100.0, status="completed", message=message)

    def fail_task(self, task_id: str, error: str):
        self.update_task(task_id, status="failed", message=error)

    def get_task(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    def get_active_tasks(self) -> list:
        return [t for t in self._tasks.values() if t["status"] == "running"]

    def _broadcast(self, event: dict):
        dead = []
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                dead.append(q)
        for q in dead:
            self._subscribers.remove(q)

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
        self.update_task(
            task_id,
            progress=float(progress),
            message=message or filename,
            metadata_update=kwargs,
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

    async def send_build_progress(
        self,
        task_id: str,
        stage: str,
        progress: int,
        message: str = "",
        log_lines: List[str] = None,
    ):
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

    async def subscribe(self) -> AsyncGenerator[str, None]:
        """Yields SSE-formatted strings. Sends an initial comment so the client connection opens."""
        # Unbounded queue: fixed small maxsize previously dropped subscribers on overflow (silent event loss).
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        try:
            # Send immediate heartbeat so EventSource receives data and fires onopen
            yield ": heartbeat\n\n"
            await asyncio.sleep(0)  # Allow first chunk to be flushed to the client
            for task in self.get_active_tasks():
                yield f"event: task_updated\ndata: {json.dumps(task)}\n\n"
            while True:
                event = await queue.get()
                yield f"event: {event['event']}\ndata: {json.dumps(event['data'])}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


_progress_manager: Optional[ProgressManager] = None


def get_progress_manager() -> ProgressManager:
    global _progress_manager
    if _progress_manager is None:
        _progress_manager = ProgressManager()
    return _progress_manager
