"""Bundle/projector coverage for backend.routes.models."""

import asyncio
from pathlib import Path

import pytest
from fastapi import BackgroundTasks, HTTPException

import backend.routes.models as models_routes


class MemoryStore:
    def __init__(self, rows=None):
        self.rows = {row["id"]: dict(row) for row in (rows or [])}
        self.updated = []

    def get_model(self, model_id):
        row = self.rows.get(model_id)
        return dict(row) if row else None

    def update_model(self, model_id, fields):
        self.rows[model_id] = {**self.rows[model_id], **fields}
        self.updated.append((model_id, dict(fields)))


class FakeProgressManager:
    def __init__(self):
        self.created = []
        self.completed = []
        self.failed = []
        self.notifications = []
        self.broadcasts = []
        self.download_updates = []

    def create_task(self, task_type, description, metadata, task_id=None):
        self.created.append((task_type, description, metadata, task_id))

    def complete_task(self, task_id, message):
        self.completed.append((task_id, message))

    def fail_task(self, task_id, message):
        self.failed.append((task_id, message))

    async def send_notification(self, *args, **kwargs):
        self.notifications.append((args, kwargs))

    async def broadcast(self, message):
        self.broadcasts.append(message)

    async def send_download_progress(self, **kwargs):
        self.download_updates.append(kwargs)


def test_download_safetensors_bundle_route_registers_hyphenated_active_download(monkeypatch):
    pm = FakeProgressManager()
    background_tasks = BackgroundTasks()
    request = models_routes.SafetensorsBundleRequest(
        huggingface_id="org/repo",
        files=[{"filename": "model.safetensors", "size": 12}],
    )

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    monkeypatch.setattr(models_routes, "get_progress_manager", lambda: pm)
    monkeypatch.setattr(models_routes.time, "time", lambda: 1234.5)

    try:
        result = asyncio.run(
            models_routes.download_safetensors_bundle(request, background_tasks)
        )

        assert result["message"] == "Safetensors bundle download started"
        assert result["task_id"] in models_routes.active_downloads
        assert (
            models_routes.active_downloads[result["task_id"]]["model_format"]
            == "safetensors-bundle"
        )
        assert len(background_tasks.tasks) == 1
        assert pm.created[0][0] == "download"
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)


def test_download_safetensors_bundle_route_rejects_duplicates(monkeypatch):
    request = models_routes.SafetensorsBundleRequest(
        huggingface_id="org/repo",
        files=[{"filename": "model.safetensors", "size": 12}],
    )
    background_tasks = BackgroundTasks()
    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["existing"] = {
        "huggingface_id": "org/repo",
        "model_format": "safetensors-bundle",
    }

    try:
        with pytest.raises(HTTPException, match="already being downloaded"):
            asyncio.run(models_routes.download_safetensors_bundle(request, background_tasks))
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)


def test_download_gguf_bundle_route_validates_and_schedules(monkeypatch):
    pm = FakeProgressManager()
    background_tasks = BackgroundTasks()
    request = {
        "huggingface_id": "org/repo",
        "quantization": "Q4_K_M",
        "files": [{"filename": "model-Q4_K_M.gguf", "size": 20}],
        "mmproj_filename": "mmproj-F16.gguf",
        "mmproj_size": 5,
    }

    with pytest.raises(HTTPException, match="Invalid projector filename"):
        asyncio.run(
            models_routes.download_gguf_bundle(
                {**request, "mmproj_filename": "not-a-projector.bin"},
                background_tasks,
            )
        )

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    monkeypatch.setattr(models_routes, "get_progress_manager", lambda: pm)
    monkeypatch.setattr(models_routes.time, "time", lambda: 4567.8)

    try:
        result = asyncio.run(models_routes.download_gguf_bundle(request, background_tasks))

        assert result["message"] == "GGUF bundle download started"
        assert result["task_id"] in models_routes.active_downloads
        assert (
            models_routes.active_downloads[result["task_id"]]["model_format"]
            == "gguf-bundle"
        )
        assert len(background_tasks.tasks) == 1
        assert pm.created[0][0] == "download"
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)


def test_download_gguf_bundle_route_rejects_duplicates_and_empty_file_lists():
    background_tasks = BackgroundTasks()

    with pytest.raises(HTTPException, match="No valid files"):
        asyncio.run(
            models_routes.download_gguf_bundle(
                {
                    "huggingface_id": "org/repo",
                    "quantization": "Q4_K_M",
                    "files": [{}],
                },
                background_tasks,
            )
        )

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["existing"] = {
        "huggingface_id": "org/repo",
        "model_format": "gguf-bundle",
        "quantization": "Q4_K_M",
    }

    try:
        with pytest.raises(HTTPException, match="already being downloaded"):
            asyncio.run(
                models_routes.download_gguf_bundle(
                    {
                        "huggingface_id": "org/repo",
                        "quantization": "Q4_K_M",
                        "files": [{"filename": "model-Q4_K_M.gguf", "size": 20}],
                    },
                    background_tasks,
                )
            )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)


def test_update_model_projector_route_handles_clear_cached_duplicate_and_schedule(
    monkeypatch, tmp_path
):
    gguf_store = MemoryStore(
        [
            {
                "id": "org/repo",
                "huggingface_id": "org/repo",
                "format": "gguf",
                "mmproj_filename": "old-mmproj.gguf",
            }
        ]
    )
    background_tasks = BackgroundTasks()
    marked = {"stale": 0}
    pm = FakeProgressManager()

    def mark_stale():
        marked["stale"] += 1

    monkeypatch.setattr(models_routes, "get_store", lambda: gguf_store)
    monkeypatch.setattr(models_routes, "_mark_llama_swap_stale", mark_stale)
    monkeypatch.setattr(models_routes, "get_progress_manager", lambda: pm)
    monkeypatch.setattr(models_routes.time, "time", lambda: 7890.1)

    non_gguf_store = MemoryStore([{"id": "safetensors", "format": "safetensors"}])
    monkeypatch.setattr(models_routes, "get_store", lambda: non_gguf_store)
    with pytest.raises(HTTPException, match="only supported for GGUF"):
        asyncio.run(
            models_routes.update_model_projector(
                "safetensors",
                {"mmproj_filename": "mmproj-F16.gguf"},
                background_tasks,
            )
        )

    monkeypatch.setattr(models_routes, "get_store", lambda: gguf_store)
    same = asyncio.run(
        models_routes.update_model_projector(
            "org/repo",
            {"mmproj_filename": "old-mmproj.gguf"},
            background_tasks,
        )
    )
    assert same["applied"] is True

    cleared = asyncio.run(
        models_routes.update_model_projector(
            "org/repo",
            {"mmproj_filename": ""},
            background_tasks,
        )
    )
    assert cleared["message"] == "Projector cleared"
    assert gguf_store.rows["org/repo"]["mmproj_filename"] is None

    cached_projector = tmp_path / "mmproj-F16.gguf"
    cached_projector.write_text("proj", encoding="utf-8")
    monkeypatch.setattr(
        models_routes, "resolve_cached_model_path", lambda hf_id, filename: str(cached_projector)
    )
    applied = asyncio.run(
        models_routes.update_model_projector(
            "org/repo",
            {"mmproj_filename": "mmproj-F16.gguf"},
            background_tasks,
        )
    )
    assert applied["message"] == "Projector applied"

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["existing"] = {
        "model_id": "org/repo",
        "filename": "mmproj-v2.gguf",
        "model_format": "gguf-projector",
    }
    monkeypatch.setattr(models_routes, "resolve_cached_model_path", lambda hf_id, filename: None)
    try:
        with pytest.raises(HTTPException, match="already being applied"):
            asyncio.run(
                models_routes.update_model_projector(
                    "org/repo",
                    {"mmproj_filename": "mmproj-v2.gguf"},
                    background_tasks,
                )
            )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    try:
        scheduled = asyncio.run(
            models_routes.update_model_projector(
                "org/repo",
                {"mmproj_filename": "mmproj-v2.gguf", "total_bytes": 55},
                background_tasks,
            )
        )

        assert scheduled["message"] == "Projector download started"
        assert len(background_tasks.tasks) >= 1
        assert pm.created[-1][0] == "download"
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)


def test_download_safetensors_bundle_task_success_and_failure(monkeypatch):
    pm = FakeProgressManager()
    store = MemoryStore()
    observed = {"saved": [], "stale": 0}

    async def fake_download(hf_id, filename, proxy, task_id, size_hint, model_format, event_hf_id):
        return (f"/tmp/{filename}", size_hint or 5)

    async def fake_save(store_obj, hf_id, filename, file_path, file_size):
        observed["saved"].append((hf_id, filename, file_size))

    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(models_routes, "download_model_with_progress", fake_download)
    monkeypatch.setattr(models_routes, "_save_safetensors_download", fake_save)
    monkeypatch.setattr(
        models_routes, "_mark_llama_swap_stale", lambda: observed.__setitem__("stale", observed["stale"] + 1)
    )

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["task-ok"] = {"filename": "bundle"}
    try:
        asyncio.run(
            models_routes.download_safetensors_bundle_task(
                "org/repo",
                [{"filename": "a.safetensors", "size": 10}],
                pm,
                "task-ok",
                10,
            )
        )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)

    assert observed["saved"] == [("org/repo", "a.safetensors", 10)]
    assert observed["stale"] == 1
    assert pm.completed == [("task-ok", "Safetensors bundle downloaded")]
    assert pm.broadcasts[-1]["model_format"] == "safetensors-bundle"
    assert pm.broadcasts[-1]["status"] == "completed"

    pm_fail = FakeProgressManager()
    async def broken_download(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(models_routes, "download_model_with_progress", broken_download)
    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["task-fail"] = {"filename": "bundle"}
    try:
        asyncio.run(
            models_routes.download_safetensors_bundle_task(
                "org/repo",
                [{"filename": "b.safetensors", "size": 10}],
                pm_fail,
                "task-fail",
                10,
            )
        )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)

    assert pm_fail.failed == [("task-fail", "network down")]
    assert pm_fail.broadcasts[-1]["model_format"] == "safetensors-bundle"
    assert pm_fail.broadcasts[-1]["status"] == "failed"
    assert pm_fail.broadcasts[-1]["error"] == "network down"


def test_download_gguf_bundle_task_and_projector_task_update_store(monkeypatch, tmp_path):
    model_id = "org--repo--Q4_K_M"
    store = MemoryStore(
        [
            {
                "id": model_id,
                "huggingface_id": "org/repo",
                "format": "gguf",
                "file_size": 0,
                "mmproj_filename": None,
            },
            {
                "id": "org/repo",
                "huggingface_id": "org/repo",
                "format": "gguf",
                "mmproj_filename": None,
            },
        ]
    )
    pm = FakeProgressManager()
    observed = {"recorded": [], "stale": 0}
    cached_projector = tmp_path / "mmproj-F16.gguf"
    cached_projector.write_text("proj", encoding="utf-8")

    async def fake_download(hf_id, filename, proxy, task_id, size_hint, model_format, event_hf_id):
        return (f"/tmp/{filename}", size_hint or 5)

    async def fake_record(store_obj, hf_id, filename, file_path, file_size, pipeline_tag=None, aggregate_size=False):
        observed["recorded"].append((hf_id, filename, file_size, aggregate_size))
        return store_obj.get_model(model_id), {"metadata": {}}

    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(models_routes, "download_model_with_progress", fake_download)
    monkeypatch.setattr(models_routes, "_record_gguf_download_post_fetch", fake_record)
    monkeypatch.setattr(
        models_routes,
        "resolve_cached_model_path",
        lambda hf_id, filename: str(cached_projector) if "mmproj" in filename else None,
    )
    monkeypatch.setattr(
        models_routes, "_mark_llama_swap_stale", lambda: observed.__setitem__("stale", observed["stale"] + 1)
    )

    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["gguf-task"] = {"filename": "bundle"}
    try:
        asyncio.run(
            models_routes.download_gguf_bundle_task(
                "org/repo",
                "Q4_K_M",
                [{"filename": "model-Q4_K_M.gguf", "size": 20}],
                pm,
                "gguf-task",
                25,
                pipeline_tag="text-generation",
                projector={"filename": "mmproj-F16.gguf", "size": 5},
            )
        )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)

    assert observed["recorded"] == [("org/repo", "model-Q4_K_M.gguf", 20, False)]
    assert store.rows[model_id]["file_size"] == 20
    assert store.rows[model_id]["mmproj_filename"] == "mmproj-F16.gguf"
    assert pm.completed == [("gguf-task", "GGUF bundle downloaded")]
    assert pm.broadcasts[-1]["model_format"] == "gguf-bundle"
    assert pm.broadcasts[-1]["status"] == "completed"
    assert observed["stale"] == 1

    projector_pm = FakeProgressManager()
    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["proj-task"] = {"filename": "mmproj-F16.gguf"}
    try:
        asyncio.run(
            models_routes.download_model_projector_task(
                "org/repo",
                "mmproj-F16.gguf",
                projector_pm,
                "proj-task",
                12,
            )
        )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)

    assert store.rows["org/repo"]["mmproj_filename"] == "mmproj-F16.gguf"
    assert projector_pm.completed == [("proj-task", "Applied projector mmproj-F16.gguf")]
    assert projector_pm.broadcasts[-1]["model_format"] == "gguf-projector"
    assert projector_pm.broadcasts[-1]["status"] == "completed"

    missing_pm = FakeProgressManager()
    empty_store = MemoryStore()
    monkeypatch.setattr(models_routes, "get_store", lambda: empty_store)
    original_downloads = dict(models_routes.active_downloads)
    models_routes.active_downloads.clear()
    models_routes.active_downloads["proj-fail"] = {"filename": "mmproj-F16.gguf"}
    try:
        asyncio.run(
            models_routes.download_model_projector_task(
                "missing",
                "mmproj-F16.gguf",
                missing_pm,
                "proj-fail",
                12,
            )
        )
    finally:
        models_routes.active_downloads.clear()
        models_routes.active_downloads.update(original_downloads)

    assert missing_pm.failed == [("proj-fail", "Model no longer exists")]
