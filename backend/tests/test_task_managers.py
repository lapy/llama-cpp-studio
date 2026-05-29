"""Tests for per-domain task cancellation managers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from backend.build_task_manager import BuildTaskManager
from backend.download_task_manager import DownloadTaskManager
from backend.task_cancel_registry import register_task_cancel, unregister_task_cancel


@pytest.fixture(autouse=True)
def _clear_cancel_registry():
    from backend import task_cancel_registry as reg

    reg._events.clear()
    yield
    reg._events.clear()


def test_build_task_manager_cancel():
    task_id = "build_test_1"
    register_task_cancel(task_id)

    with patch("backend.build_task_manager.get_progress_manager") as mock_pm_factory:
        pm = MagicMock()
        pm.get_task.return_value = {
            "task_id": task_id,
            "type": "build",
            "status": "running",
        }
        mock_pm_factory.return_value = pm

        result = BuildTaskManager.cancel(task_id)

    assert result["ok"] is True
    assert result["task_id"] == task_id


def test_download_task_manager_cancel():
    from backend.progress_manager import get_progress_manager
    from backend.services import model_downloads

    pm = get_progress_manager()
    pm._tasks.clear()
    model_downloads.active_downloads.clear()

    task_id = "download_test_1"
    register_task_cancel(task_id)
    pm.create_task("download", "Download file.bin", {}, task_id=task_id)
    model_downloads.active_downloads[task_id] = {"huggingface_id": "org/model", "filename": "a.gguf"}

    result = DownloadTaskManager.cancel(task_id)

    assert result["ok"] is True
    assert pm.get_task(task_id)["status"] == "failed"
    unregister_task_cancel(task_id)
    model_downloads.active_downloads.clear()
