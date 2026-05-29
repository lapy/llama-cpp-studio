"""Build tasks (llama.cpp / ik_llama source compiles)."""

from __future__ import annotations

from typing import Any, Dict

from backend.progress_manager import get_progress_manager
from backend.task_cancel_registry import request_task_cancel


class BuildTaskManager:
    """Cancellation for source build tasks registered in the cancel registry."""

    @classmethod
    def cancel(cls, task_id: str) -> Dict[str, Any]:
        task_id = str(task_id or "").strip()
        if not task_id:
            return {"ok": False, "message": "task_id is required"}

        pm = get_progress_manager()
        task = pm.get_task(task_id)
        if not task or task.get("type") != "build":
            return {"ok": False, "message": "Unknown build task."}
        if task.get("status") != "running":
            return {"ok": False, "message": "Build is not running."}
        if request_task_cancel(task_id):
            return {
                "ok": True,
                "message": "Build cancellation requested; it will stop shortly.",
                "task_id": task_id,
            }
        return {"ok": False, "message": "No active cancellable build for that task_id."}
