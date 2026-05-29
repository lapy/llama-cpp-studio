"""Model download tasks."""

from __future__ import annotations

from typing import Any, Dict

from backend.progress_manager import get_progress_manager
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
        if task.get("status") != "running":
            return {"ok": False, "message": "Download is not running."}
        if task_id not in active_downloads:
            return {"ok": False, "message": "Download is not active."}
        if not request_task_cancel(task_id):
            return {"ok": False, "message": "Download could not be cancelled."}

        pm.fail_task(task_id, "Download cancelled by user")
        return {
            "ok": True,
            "message": "Download cancellation requested.",
            "task_id": task_id,
        }
