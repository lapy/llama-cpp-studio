"""Concurrent saves and background operation writes.

Request-scoped flushes do not show whether two writers lose fields, or whether
a finish that runs after the response blocks the loop and overwrites the row.
"""

import asyncio
import threading
import time

import httpx
import pytest
from fastapi import FastAPI

from backend.data_store import DataStore


def test_concurrent_model_and_settings_updates_keep_every_field(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_model(
        {
            "id": "org/model",
            "name": "Test Model",
            "display_name": "Original",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {},
        }
    )
    barrier = threading.Barrier(4)
    errors = []

    def run(func):
        try:
            barrier.wait(timeout=2)
            func()
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [
        threading.Thread(
            target=run,
            args=(lambda: store.update_model("org/model", {"display_name": "Renamed"}),),
        ),
        threading.Thread(
            target=run,
            args=(lambda: store.update_model("org/model", {"quantization": "Q8_0"}),),
        ),
        threading.Thread(
            target=run,
            args=(lambda: store.update_settings({"alpha": "a"}),),
        ),
        threading.Thread(
            target=run,
            args=(lambda: store.update_settings({"beta": "b"}),),
        ),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()

    assert errors == []
    model = store.get_model("org/model")
    assert model["display_name"] == "Renamed"
    assert model["quantization"] == "Q8_0"
    settings = store.get_settings()
    assert settings["alpha"] == "a"
    assert settings["beta"] == "b"

    reloaded = DataStore(config_dir=str(tmp_path / "config"))
    reloaded_model = reloaded.get_model("org/model")
    assert reloaded_model["display_name"] == "Renamed"
    assert reloaded_model["quantization"] == "Q8_0"
    assert reloaded.get_settings()["alpha"] == "a"
    assert reloaded.get_settings()["beta"] == "b"


def test_concurrent_operation_inserts_all_survive(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.supervisor import get_supervisor

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    barrier = threading.Barrier(8)
    errors = []

    def start(index):
        try:
            barrier.wait(timeout=2)
            get_supervisor().start_operation(f"op-{index}", "build", f"dir-{index}")
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=start, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert errors == []
    ids = {row["operation_id"]: row for row in store.list_operations()}
    assert set(ids) == {f"op-{index}" for index in range(8)}
    assert ids["op-3"]["resource_key"] == "dir-3"
    assert ids["op-3"]["status"] == "running"


@pytest.mark.asyncio
async def test_background_finish_keeps_the_row_and_leaves_the_loop_free(
    tmp_path, monkeypatch
):
    from backend import data_store
    from backend.operations.supervisor import get_supervisor
    from backend.store_io import StoreIoMiddleware, drain_store_io

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    original = DataStore.upsert_operation

    def slow_upsert(self, operation):
        time.sleep(0.4)
        return original(self, operation)

    monkeypatch.setattr(DataStore, "upsert_operation", slow_upsert)

    mini = FastAPI()
    mini.add_middleware(StoreIoMiddleware)

    @mini.post("/save")
    async def save():
        get_supervisor().start_operation(
            "op-bg",
            "build",
            "install-dir",
            detail={"description": "keep-me", "engine": "llama_cpp"},
        )

        async def finish():
            get_supervisor().finish_operation("op-bg", "failed", "disk full")

        asyncio.create_task(finish())
        return {"ok": True}

    transport = httpx.ASGITransport(app=mini)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        started = time.perf_counter()
        saving = asyncio.create_task(client.post("/save"))
        await asyncio.sleep(0.05)
        elapsed = time.perf_counter() - started
        response = await saving

    assert elapsed < 0.25
    assert response.status_code == 200
    await drain_store_io()
    rows = store.list_operations()
    assert len(rows) == 1
    row = rows[0]
    assert row["operation_id"] == "op-bg"
    assert row["status"] == "failed"
    assert row["message"] == "disk full"
    assert row["kind"] == "build"
    assert row["resource_key"] == "install-dir"
    assert row["detail"]["description"] == "keep-me"
    assert row["detail"]["engine"] == "llama_cpp"


@pytest.mark.asyncio
async def test_dismiss_waits_behind_the_queued_insert(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.progress import get_progress_manager
    from backend.operations.supervisor import get_supervisor
    from backend.store_io import StoreIoMiddleware, drain_store_io

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    original = DataStore.upsert_operation

    def slow_upsert(self, operation):
        time.sleep(0.2)
        return original(self, operation)

    monkeypatch.setattr(DataStore, "upsert_operation", slow_upsert)

    mini = FastAPI()
    mini.add_middleware(StoreIoMiddleware)

    @mini.post("/save")
    async def save():
        get_supervisor().start_operation("op-dismiss", "build", "dir-a")
        assert get_progress_manager().dismiss_task("op-dismiss") is True
        return {"ok": True}

    transport = httpx.ASGITransport(app=mini)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        response = await client.post("/save")

    assert response.status_code == 200
    await drain_store_io()
    assert store.list_operations() == []
    assert get_supervisor()._get("op-dismiss") is None


@pytest.mark.asyncio
async def test_parallel_requests_each_keep_their_operation(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.supervisor import get_supervisor
    from backend.store_io import StoreIoMiddleware

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    mini = FastAPI()
    mini.add_middleware(StoreIoMiddleware)

    @mini.post("/save/{operation_id}")
    async def save(operation_id: str):
        get_supervisor().start_operation(operation_id, "download", f"res-{operation_id}")
        return {"ok": True}

    transport = httpx.ASGITransport(app=mini)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        responses = await asyncio.gather(
            client.post("/save/alpha"),
            client.post("/save/beta"),
        )

    assert [response.status_code for response in responses] == [200, 200]
    ids = {row["operation_id"]: row for row in store.list_operations()}
    assert set(ids) == {"alpha", "beta"}
    assert ids["alpha"]["resource_key"] == "res-alpha"
    assert ids["beta"]["kind"] == "download"
