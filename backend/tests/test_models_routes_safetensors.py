"""Additional safetensors/helper coverage for backend.routes.models."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import backend.routes.models as models_routes


class MemoryStore:
    def __init__(self, rows=None):
        self.rows = {row["id"]: dict(row) for row in (rows or [])}
        self.added = []
        self.updated = []
        self.deleted = []

    def get_model(self, model_id):
        row = self.rows.get(model_id)
        return dict(row) if row else None

    def add_model(self, row):
        self.rows[row["id"]] = dict(row)
        self.added.append(dict(row))

    def update_model(self, model_id, fields):
        self.rows[model_id] = {**self.rows[model_id], **fields}
        self.updated.append((model_id, dict(fields)))

    def delete_model(self, model_id):
        self.deleted.append(model_id)
        self.rows.pop(model_id, None)

    def list_models(self):
        return [dict(row) for row in self.rows.values()]


def test_passthrough_response_and_mark_stale(monkeypatch):
    response = models_routes._passthrough_llama_swap_response(
        SimpleNamespace(
            content=b"ok",
            status_code=202,
            headers={"content-type": "application/json"},
        )
    )
    assert response.status_code == 202
    assert response.body == b"ok"
    assert response.headers["content-type"] == "application/json"

    called = {}
    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: called.setdefault("marked", True),
    )
    models_routes._mark_llama_swap_stale()
    assert called["marked"] is True

    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    models_routes._mark_llama_swap_stale()


def test_get_safetensors_model_and_manifest_loader(monkeypatch):
    store = MemoryStore(
        [
            {
                "id": "org--repo",
                "huggingface_id": "org/repo",
                "format": "safetensors",
            }
        ]
    )
    model = models_routes._get_safetensors_model(store, "org--repo")
    assert model["huggingface_id"] == "org/repo"

    with pytest.raises(HTTPException, match="not a safetensors"):
        models_routes._get_safetensors_model(
            MemoryStore([{"id": "gguf", "format": "gguf"}]), "gguf"
        )

    monkeypatch.setattr(
        models_routes,
        "get_safetensors_manifest_entries",
        lambda hf_id: {"files": [{"filename": "model.safetensors"}]},
    )
    assert models_routes._load_manifest_entry_for_model(model)["files"]

    monkeypatch.setattr(
        models_routes, "get_safetensors_manifest_entries", lambda hf_id: None
    )
    with pytest.raises(HTTPException, match="manifest not found"):
        models_routes._load_manifest_entry_for_model(model)


def test_bundle_progress_proxy_aggregates_progress_and_forwards_notifications():
    observed = {}

    class BaseManager:
        active_connections = ["client"]

        async def send_download_progress(self, **kwargs):
            observed["progress"] = kwargs

        async def send_notification(self, *args, **kwargs):
            observed["notification"] = (args, kwargs)

        async def broadcast(self, message):
            observed["broadcast"] = message

    proxy = models_routes.BundleProgressProxy(
        BaseManager(),
        "bundle-task",
        bytes_completed=50,
        total_bytes=200,
        file_index=1,
        total_files=3,
        current_filename="part-2.safetensors",
        huggingface_id="org/repo",
    )

    asyncio.run(
        proxy.send_download_progress(
            task_id="ignored",
            progress=20,
            bytes_downloaded=50,
            speed_mbps=12.5,
            eta_seconds=3,
        )
    )
    asyncio.run(proxy.send_notification("hello", severity="info"))
    asyncio.run(proxy.broadcast({"type": "done"}))

    assert proxy.active_connections == ["client"]
    assert observed["progress"]["task_id"] == "bundle-task"
    assert observed["progress"]["progress"] == 50
    assert observed["progress"]["files_completed"] == 2
    assert observed["progress"]["filename"] == "part-2.safetensors"
    assert observed["notification"][0] == ("hello",)
    assert observed["broadcast"] == {"type": "done"}


def test_bundle_progress_proxy_uses_file_progress_when_total_unknown():
    observed = {}

    class BaseManager:
        async def send_download_progress(self, **kwargs):
            observed.update(kwargs)

    proxy = models_routes.BundleProgressProxy(
        BaseManager(),
        "bundle-task",
        bytes_completed=0,
        total_bytes=0,
        file_index=0,
        total_files=2,
        current_filename="file-a.safetensors",
    )

    asyncio.run(
        proxy.send_download_progress(
            task_id="ignored",
            progress=33,
            bytes_downloaded=123,
            total_bytes=0,
        )
    )

    assert observed["progress"] == 33
    assert observed["total_bytes"] == 123


def test_get_cached_gpu_info_reuses_recent_cache(monkeypatch):
    calls = {"count": 0}

    async def fake_gpu_info():
        calls["count"] += 1
        return {"gpus": calls["count"]}

    monkeypatch.setattr(models_routes, "get_gpu_info", fake_gpu_info)
    models_routes._gpu_info_cache["data"] = None
    models_routes._gpu_info_cache["timestamp"] = 0.0

    first = asyncio.run(models_routes.get_cached_gpu_info())
    second = asyncio.run(models_routes.get_cached_gpu_info())
    models_routes._gpu_info_cache["timestamp"] = 0.0
    third = asyncio.run(models_routes.get_cached_gpu_info())

    assert first == {"gpus": 1}
    assert second == {"gpus": 1}
    assert third == {"gpus": 2}


def test_save_safetensors_download_creates_model_and_aggregates_repo_size(monkeypatch):
    store = MemoryStore()
    recorded = {}

    async def fake_collect(hf_id, filename):
        return (
            {"pipeline_tag": "feature-extraction", "max_context_length": 8192},
            {"tensor_count": 42},
            8192,
        )

    monkeypatch.setattr(models_routes, "_collect_safetensors_runtime_metadata", fake_collect)
    monkeypatch.setattr(
        models_routes,
        "record_safetensors_download",
        lambda **kwargs: recorded.setdefault("download", kwargs),
    )
    monkeypatch.setattr(
        "backend.huggingface.list_safetensors_downloads",
        lambda: [
            {
                "huggingface_id": "org/repo",
                "files": [{"file_size": 10}, {"file_size": 15}],
            }
        ],
    )

    model = asyncio.run(
        models_routes._save_safetensors_download(
            store,
            "org/repo",
            "model-00001-of-00002.safetensors",
            "/tmp/model.safetensors",
            10,
        )
    )

    assert model["id"] == "org--repo"
    assert store.rows["org--repo"]["file_size"] == 25
    assert store.rows["org--repo"]["pipeline_tag"] == "feature-extraction"
    assert models_routes._model_is_embedding(store.rows["org--repo"]) is True
    assert recorded["download"]["model_id"] == "org--repo"


def test_save_safetensors_download_updates_existing_model_and_tolerates_aggregation_failures(
    monkeypatch,
):
    store = MemoryStore(
        [
            {
                "id": "org--repo",
                "huggingface_id": "org/repo",
                "format": "safetensors",
                "file_size": 12,
                "config": {},
            }
        ]
    )

    async def fake_collect(hf_id, filename):
        return ({"pipeline_tag": "text-embedding"}, {"tensor_count": 8}, 4096)

    monkeypatch.setattr(models_routes, "_collect_safetensors_runtime_metadata", fake_collect)
    monkeypatch.setattr(models_routes, "record_safetensors_download", lambda **kwargs: None)
    monkeypatch.setattr(
        "backend.huggingface.list_safetensors_downloads",
        lambda: (_ for _ in ()).throw(RuntimeError("no manifest")),
    )

    model = asyncio.run(
        models_routes._save_safetensors_download(
            store,
            "org/repo",
            "model.safetensors",
            "/tmp/model.safetensors",
            12,
        )
    )

    assert model["id"] == "org--repo"
    assert store.updated
    assert models_routes._model_is_embedding(store.rows["org--repo"]) is True


def test_delete_safetensors_model_unregisters_running_model_and_marks_stale(
    monkeypatch,
):
    store = MemoryStore(
        [
            {
                "id": "org--repo",
                "huggingface_id": "org/repo",
                "format": "safetensors",
                "proxy_name": "proxy-a",
            }
        ]
    )
    observed = {}

    class FakeClient:
        async def get_running_models(self):
            return {"running": [{"model": "proxy-a", "state": "ready"}]}

    class FakeManager:
        async def unregister_model(self, proxy_name):
            observed["unregistered"] = proxy_name

    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    monkeypatch.setattr(
        models_routes,
        "get_safetensors_manifest_entries",
        lambda hf_id: {"files": [{"filename": "model.safetensors"}]},
    )
    monkeypatch.setattr(models_routes, "resolve_proxy_name", lambda model: "proxy-a")
    monkeypatch.setattr("backend.llama_swap_client.LlamaSwapClient", FakeClient)
    monkeypatch.setattr(
        "backend.llama_swap_manager.get_llama_swap_manager", lambda: FakeManager()
    )
    monkeypatch.setattr(
        models_routes,
        "purge_safetensors_repo_completely",
        lambda hf_id: observed.setdefault("purged", hf_id),
    )
    monkeypatch.setattr(
        models_routes, "_mark_llama_swap_stale", lambda: observed.setdefault("stale", True)
    )

    result = asyncio.run(
        models_routes.delete_safetensors_model({"huggingface_id": "org/repo"})
    )

    assert result["message"] == "Safetensors model org/repo deleted"
    assert observed == {
        "unregistered": "proxy-a",
        "purged": "org/repo",
        "stale": True,
    }
    assert "org--repo" in store.deleted
