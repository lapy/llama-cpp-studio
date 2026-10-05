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


def test_store_writes_stop_when_the_pending_queue_is_full():
    from backend import store_io

    release = threading.Event()
    entered = threading.Event()
    threads = []

    def block():
        entered.set()
        assert release.wait(timeout=5)

    def occupy():
        store_io.run_store(lambda: release.wait(timeout=5))

    worker = threading.Thread(target=lambda: store_io.run_store(block))
    threads.append(worker)
    worker.start()
    assert entered.wait(timeout=2)
    try:
        for _ in range(store_io.MAX_PENDING_STORE_WRITES - 1):
            thread = threading.Thread(target=occupy)
            threads.append(thread)
            thread.start()
        deadline = time.time() + 2
        while time.time() < deadline:
            if store_io.pending_store_writes() >= store_io.MAX_PENDING_STORE_WRITES:
                break
            time.sleep(0.01)
        assert store_io.pending_store_writes() == store_io.MAX_PENDING_STORE_WRITES

        async def overflow():
            with pytest.raises(store_io.StoreIoBusy):
                store_io.run_store(lambda: None)

        asyncio.run(overflow())
        assert store_io.pending_store_writes() == store_io.MAX_PENDING_STORE_WRITES

        finished = threading.Event()

        def wait_for_a_slot():
            store_io.run_store(lambda: None)
            finished.set()

        waiter = threading.Thread(target=wait_for_a_slot)
        threads.append(waiter)
        waiter.start()
        time.sleep(0.05)
        assert finished.is_set() is False
    finally:
        release.set()
        for thread in threads:
            thread.join(timeout=5)
            assert not thread.is_alive()
    deadline = time.time() + 2
    while time.time() < deadline and store_io.pending_store_writes():
        time.sleep(0.01)
    assert store_io.pending_store_writes() == 0


def test_supervisor_bounds_terminal_records_in_memory(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.retention import MAX_RETAINED_TERMINAL_OPERATIONS
    from backend.operations.supervisor import OperationSupervisor

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    now = time.time()
    with supervisor._lock:
        supervisor._records["running"] = {
            "operation_id": "running",
            "status": "running",
            "updated_at": now,
        }
        for index in range(MAX_RETAINED_TERMINAL_OPERATIONS + 40):
            supervisor._records[f"failed-{index}"] = {
                "operation_id": f"failed-{index}",
                "status": "failed",
                "updated_at": now + index,
            }
        supervisor._records["succeeded"] = {
            "operation_id": "succeeded",
            "status": "succeeded",
            "updated_at": now + 10_000,
        }
    supervisor._evict_terminal_records()
    assert "running" in supervisor._records
    assert "succeeded" not in supervisor._records
    failed = [key for key in supervisor._records if key.startswith("failed-")]
    assert len(failed) == MAX_RETAINED_TERMINAL_OPERATIONS
    assert "failed-0" not in supervisor._records
    assert f"failed-{MAX_RETAINED_TERMINAL_OPERATIONS + 39}" in supervisor._records


def test_in_flight_success_stays_in_memory_until_the_write_finishes(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.supervisor import OperationSupervisor

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    original = DataStore.upsert_operation
    release = threading.Event()
    writing = threading.Event()

    def slow_upsert(self, operation):
        if operation.get("status") == "succeeded":
            writing.set()
            assert release.wait(timeout=3)
        return original(self, operation)

    monkeypatch.setattr(DataStore, "upsert_operation", slow_upsert)
    supervisor = OperationSupervisor()

    def finish():
        supervisor.start_operation("op-success", "build", "install-dir")
        supervisor.finish_operation("op-success", "succeeded", "done")

    thread = threading.Thread(target=finish)
    thread.start()
    assert writing.wait(timeout=3)
    try:
        cached = supervisor._records["op-success"]
        assert cached["status"] == "succeeded"
        assert cached["resource_key"] == "install-dir"
    finally:
        release.set()
        thread.join(timeout=3)
    assert not thread.is_alive()
    assert "op-success" not in supervisor._records
    assert store.list_operations() == []


def _fill_store_queue():
    """Block every persistence slot. Caller must set the returned event."""
    from backend import store_io

    release = threading.Event()
    entered = threading.Event()
    threads = []

    def block():
        entered.set()
        assert release.wait(timeout=5)

    worker = threading.Thread(target=lambda: store_io.run_store(block))
    threads.append(worker)
    worker.start()
    assert entered.wait(timeout=2)
    for _ in range(store_io.MAX_PENDING_STORE_WRITES - 1):
        thread = threading.Thread(target=lambda: store_io.run_store(lambda: release.wait(timeout=5)))
        threads.append(thread)
        thread.start()
    deadline = time.time() + 2
    while time.time() < deadline and store_io.pending_store_writes() < store_io.MAX_PENDING_STORE_WRITES:
        time.sleep(0.01)
    assert store_io.pending_store_writes() == store_io.MAX_PENDING_STORE_WRITES
    return release, threads


def _drain_store_queue(release, threads):
    from backend import store_io

    release.set()
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    deadline = time.time() + 2
    while time.time() < deadline and store_io.pending_store_writes():
        time.sleep(0.01)
    assert store_io.pending_store_writes() == 0


def test_rejected_supervisor_write_can_be_retried_after_the_queue_drains(tmp_path, monkeypatch):
    from backend import data_store, store_io
    from backend.operations.supervisor import OperationSupervisor, ResourceBusyError

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    release, threads = _fill_store_queue()
    try:
        async def rejected_start():
            with pytest.raises(store_io.StoreIoBusy):
                supervisor.start_operation("rejected", "build", "engine:demo")

        asyncio.run(rejected_start())
        assert supervisor._pending_writes == {}
        assert supervisor._resources == {}
        assert "rejected" not in supervisor._records
        assert store.list_operations() == []
    finally:
        _drain_store_queue(release, threads)

    supervisor.start_operation("accepted", "build", "engine:demo")
    assert supervisor._resources["engine:demo"] == "accepted"
    assert store.list_operations()[0]["status"] == "running"

    release, threads = _fill_store_queue()
    try:
        async def rejected_finish():
            with pytest.raises(store_io.StoreIoBusy):
                supervisor.finish_operation("accepted", "failed", "not stored")

        asyncio.run(rejected_finish())
        assert supervisor._pending_writes == {}
        assert supervisor._resources["engine:demo"] == "accepted"
        assert supervisor._records["accepted"]["status"] == "running"
        assert store.list_operations()[0]["status"] == "running"
        with pytest.raises(ResourceBusyError):
            supervisor.start_operation("other", "build", "engine:demo")
    finally:
        _drain_store_queue(release, threads)

    supervisor.finish_operation("accepted", "failed", "stored")
    rows = {row["operation_id"]: row for row in store.list_operations()}
    assert rows["accepted"]["status"] == "failed"
    assert supervisor._resources == {}
    assert supervisor._pending_writes == {}
