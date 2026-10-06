"""Model download tasks."""

from __future__ import annotations

from typing import Any, Dict

from backend.operations.progress import get_progress_manager
from backend.services.model_downloads import active_downloads
from backend.task_cancel_registry import request_task_cancel


class DownloadTaskManager:
    """Cancellation for HuggingFace download tasks."""

    @classmethod
    def cancel(cls, task_id: str) -> Dict[str, Any]:
        task_id = str(task_id or "").strip()
        if not task_id:
            return {"ok": False, "message": "task_id is required"}

        pm = get_progress_manager()
        task = pm.get_task(task_id)
        if not task or task.get("type") != "download":
            return {"ok": False, "message": "Unknown download task."}
        if task.get("status") == "cancelling":
            return {
                "ok": True,
                "terminated": False,
                "message": (
                    "Cancellation was already requested. "
                    "The download has not been verified as stopped."
                ),
                "task_id": task_id,
            }
        if task.get("status") != "running":
            return {"ok": False, "terminated": False, "message": "Download is not running."}
        if task_id not in active_downloads:
            return {"ok": False, "terminated": False, "message": "Download is not active."}
        if not request_task_cancel(task_id):
            return {"ok": False, "terminated": False, "message": "Download could not be cancelled."}

        pm.update_task(
            task_id,
            status="cancelling",
            message="Cancellation was requested. The download has not been verified as stopped.",
        )
        from backend.operations.supervisor import get_supervisor

        get_supervisor().note_cancellation(task_id)
        return {
            "ok": True,
            "terminated": False,
            "message": "Cancellation was requested. The download has not been verified as stopped.",
            "task_id": task_id,
        }
