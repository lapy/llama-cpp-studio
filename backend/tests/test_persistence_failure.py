"""Persistence failures over HTTP, and process death between durability points.

These tests use a temporary config directory. They do not read or write the
repository ``data/`` tree and they do not download models or use a GPU.
"""

import errno
import multiprocessing
import os
import signal
from typing import Optional

import pytest
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from backend.data_store import DataStore, set_write_checkpoint
from backend.operations.supervisor import OperationSupervisor, ResourceBusyError
from backend.store_io import (
    StoreDurabilityError,
    StoreIoBusy,
    StoreIoMiddleware,
    persistence_http_error,
)


@pytest.fixture(autouse=True)
def _reset_write_checkpoint():
    set_write_checkpoint(None)
    yield
    set_write_checkpoint(None)


def _complete_mapping(path) -> dict:
    text = path.read_text(encoding="utf-8")
    loaded = yaml.safe_load(text)
    assert isinstance(loaded, dict), text
    return loaded


def _read_settings_in_child(directory: str, connection) -> None:
    from backend.data_store import DataStore as FreshStore

    connection.send(FreshStore(config_dir=directory).get_settings())


def _read_settings_and_models(directory: str, connection) -> None:
    from backend.data_store import DataStore as FreshStore

    fresh = FreshStore(config_dir=directory)
    connection.send(
        {
            "url": fresh.get_settings().get("public_inference_url"),
            "models": [row["id"] for row in fresh.list_models()],
        }
    )


def _fresh_settings(config_dir: str) -> dict:
    """Read settings in a new process so a warm cache cannot answer."""
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_read_settings_in_child, args=(config_dir, writer))
    proc.start()
    writer.close()
    assert reader.poll(10), "fresh process did not read settings"
    payload = reader.recv()
    proc.join(5)
    assert proc.exitcode == 0
    assert isinstance(payload, dict)
    return payload


def _pause_settings_write(
    config_dir: str,
    checkpoint: str,
    connection,
    marker: Optional[str],
) -> None:
    """Reach one settings checkpoint, signal the parent, then wait to be killed."""
    import os as child_os
    import signal as child_signal

    from backend.data_store import DataStore as ChildStore
    from backend.data_store import set_write_checkpoint as child_checkpoint

    def hook(name: str, path: str) -> None:
        if name == checkpoint and child_os.path.basename(path) == "settings.yaml":
            connection.send("ready")
            connection.close()
            while True:
                child_signal.pause()

    child_checkpoint(hook)
    store = ChildStore(config_dir=config_dir)
    if marker is not None:
        store.update_settings({"marker": marker})


def _kill_at_checkpoint(config_dir: str, checkpoint: str, marker: Optional[str]) -> None:
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_pause_settings_write,
        args=(config_dir, checkpoint, writer, marker),
    )
    proc.start()
    writer.close()
    reached = reader.poll(5)
    if not reached:
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(5)
        pytest.fail(f"writer did not reach {checkpoint}")
    assert reader.recv() == "ready"
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(5)
    assert not proc.is_alive()


def _operations_app(supervisor: OperationSupervisor) -> FastAPI:
    """Same queue, middleware, and error payload the production app uses."""
    app = FastAPI()

    @app.exception_handler(StoreIoBusy)
    async def queue_full(_request: Request, exc: StoreIoBusy):
        status, body = persistence_http_error(exc)
        return JSONResponse(status_code=status, content=body)

    @app.exception_handler(StoreDurabilityError)
    async def write_failed(_request: Request, exc: StoreDurabilityError):
        status, body = persistence_http_error(exc)
        return JSONResponse(status_code=status, content=body)

    @app.post("/api/runtime/operations")
    async def start(request: Request):
        body = await request.json()
        try:
            return supervisor.start_operation(
                body["operation_id"],
                body["kind"],
                body.get("resource_key") or None,
            )
        except ResourceBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/runtime/operations/{operation_id}/finish")
    async def finish(operation_id: str, request: Request):
        body = await request.json()
        supervisor.finish_operation(
            operation_id,
            body["status"],
            body.get("message") or "",
        )
        return {"operation_id": operation_id, "status": body["status"]}

    app.add_middleware(StoreIoMiddleware)
    return app


def test_queue_rejection_is_visible_on_the_operation_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend import data_store
    from backend.tests.test_store_concurrency import _drain_store_queue, _fill_store_queue

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    client = TestClient(_operations_app(supervisor))

    release, threads = _fill_store_queue()
    try:
        rejected = client.post(
            "/api/runtime/operations",
            json={
                "operation_id": "rejected",
                "kind": "build",
                "resource_key": "engine:demo",
            },
        )
        assert rejected.status_code == 503
        body = rejected.json()
        assert body["code"] == "STORE_QUEUE_FULL"
        assert body["committed"] is False
        assert "retry" in body["detail"].lower()
        assert supervisor._resources == {}
        assert supervisor._pending_writes == {}
        assert store.list_operations() == []
    finally:
        _drain_store_queue(release, threads)

    accepted = client.post(
        "/api/runtime/operations",
        json={
            "operation_id": "accepted",
            "kind": "build",
            "resource_key": "engine:demo",
        },
    )
    assert accepted.status_code == 200
    assert accepted.json()["status"] == "running"
    assert supervisor._resources["engine:demo"] == "accepted"
    assert store.list_operations()[0]["status"] == "running"

    release, threads = _fill_store_queue()
    try:
        rejected_finish = client.post(
            "/api/runtime/operations/accepted/finish",
            json={"status": "failed", "message": "not stored"},
        )
        assert rejected_finish.status_code == 503
        finish_body = rejected_finish.json()
        assert finish_body["code"] == "STORE_QUEUE_FULL"
        assert finish_body["committed"] is False
        assert supervisor._resources["engine:demo"] == "accepted"
        assert store.list_operations()[0]["status"] == "running"
        conflict = client.post(
            "/api/runtime/operations",
            json={
                "operation_id": "other",
                "kind": "build",
                "resource_key": "engine:demo",
            },
        )
        assert conflict.status_code == 409
    finally:
        _drain_store_queue(release, threads)

    stored = client.post(
        "/api/runtime/operations/accepted/finish",
        json={"status": "failed", "message": "stored"},
    )
    assert stored.status_code == 200
    rows = {row["operation_id"]: row for row in store.list_operations()}
    assert rows["accepted"]["status"] == "failed"
    assert supervisor._resources == {}


def test_a_queued_operation_write_failure_is_not_a_successful_response(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from backend import data_store

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    client = TestClient(_operations_app(supervisor))

    def fail_operations(name: str, path: str) -> None:
        if name == "before_fsync" and os.path.basename(path) == "operations.yaml":
            raise OSError(errno.ENOSPC, "No space left on device")

    set_write_checkpoint(fail_operations)
    rejected = client.post(
        "/api/runtime/operations",
        json={
            "operation_id": "disk-rejected",
            "kind": "build",
            "resource_key": "engine:demo",
        },
    )
    assert rejected.status_code == 500
    body = rejected.json()
    assert body["code"] == "STORE_WRITE_FAILED"
    assert body["committed"] is False
    assert supervisor._resources == {}
    assert store.list_operations() == []

    set_write_checkpoint(None)
    accepted = client.post(
        "/api/runtime/operations",
        json={
            "operation_id": "disk-accepted",
            "kind": "build",
            "resource_key": "engine:demo",
        },
    )
    assert accepted.status_code == 200
    set_write_checkpoint(fail_operations)
    rejected_finish = client.post(
        "/api/runtime/operations/disk-accepted/finish",
        json={"status": "failed", "message": "disk full"},
    )
    assert rejected_finish.status_code == 500
    finish_body = rejected_finish.json()
    assert finish_body["code"] == "STORE_WRITE_FAILED"
    assert finish_body["committed"] is False
    assert supervisor._resources["engine:demo"] == "disk-accepted"
    assert store.list_operations()[0]["status"] == "running"


@pytest.mark.parametrize(
    "checkpoint,phase",
    [
        ("before_temp_write", "temp_write"),
        ("before_fsync", "fsync"),
        ("before_replace", "replace"),
    ],
)
def test_settings_api_reports_an_uncommitted_disk_failure(client, checkpoint, phase):
    from backend.data_store import get_store

    store = get_store()
    store.update_settings({"public_inference_url": "http://old.example"})
    config_dir = store._config_dir

    def fail(name: str, path: str) -> None:
        if name == checkpoint and os.path.basename(path) == "settings.yaml":
            raise OSError(errno.ENOSPC, "No space left on device")

    set_write_checkpoint(fail)
    response = client.put(
        "/api/settings/inference",
        json={"public_inference_url": "http://new.example"},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "STORE_WRITE_FAILED"
    assert body["phase"] == phase
    assert body["committed"] is False
    assert "unchanged" in body["detail"]
    assert "no document was written" in body["detail"]
    assert "replaced" not in body["detail"]
    assert config_dir not in response.text
    assert "No space left" not in response.text
    assert _fresh_settings(config_dir)["public_inference_url"] == "http://old.example"


def test_settings_api_does_not_deny_a_save_that_already_replaced_the_file(client):
    from backend.data_store import get_store

    store = get_store()
    store.update_settings({"public_inference_url": "http://old.example"})
    config_dir = store._config_dir

    def fail_acknowledgement(name: str, path: str) -> None:
        if name == "after_replace" and os.path.basename(path) == "settings.yaml":
            raise OSError(errno.EIO, "acknowledgement failed")

    set_write_checkpoint(fail_acknowledgement)
    response = client.put(
        "/api/settings/inference",
        json={"public_inference_url": "http://new.example"},
    )
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "STORE_WRITE_FAILED"
    assert body["phase"] == "acknowledge"
    assert body["committed"] is True
    assert "was replaced" in body["detail"]
    assert "acknowledgement failed" in body["detail"]
    assert "unchanged" not in body["detail"]
    assert "nothing was saved" not in body["detail"].lower()
    assert config_dir not in response.text
    assert _fresh_settings(config_dir)["public_inference_url"] == "http://new.example"


def test_acknowledged_settings_save_survives_a_new_process(client):
    from backend.data_store import get_store

    response = client.put(
        "/api/settings/inference",
        json={"public_inference_url": "http://kept.example"},
    )
    assert response.status_code == 200
    assert response.json()["public_inference_url"] == "http://kept.example"
    assert (
        _fresh_settings(get_store()._config_dir)["public_inference_url"]
        == "http://kept.example"
    )


def test_killing_the_first_settings_write_does_not_leave_a_truncated_document(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _kill_at_checkpoint(str(config_dir), "before_replace", None)
    assert not (config_dir / "settings.yaml").exists()
    for name in ("models.yaml", "engines.yaml"):
        assert _complete_mapping(config_dir / name)


@pytest.mark.parametrize(
    "checkpoint,expected",
    [
        ("before_fsync", "old"),
        ("before_replace", "old"),
        ("after_replace", "new"),
    ],
)
def test_killing_between_write_fsync_and_replace_leaves_a_complete_document(
    tmp_path, checkpoint, expected
):
    config_dir = tmp_path / "config"
    DataStore(config_dir=str(config_dir)).update_settings({"marker": "old"})
    _kill_at_checkpoint(str(config_dir), checkpoint, "new")
    document = _complete_mapping(config_dir / "settings.yaml")
    assert document["marker"] == expected
    assert _fresh_settings(str(config_dir))["marker"] == expected


def test_a_failed_settings_write_does_not_roll_back_another_document(tmp_path):
    config_dir = str(tmp_path / "config")
    store = DataStore(config_dir=config_dir)
    store.update_settings({"public_inference_url": "http://old.example"})

    def fail_settings(name: str, path: str) -> None:
        if name == "before_fsync" and os.path.basename(path) == "settings.yaml":
            raise OSError(errno.ENOSPC, "No space left on device")

    set_write_checkpoint(fail_settings)
    with pytest.raises(StoreDurabilityError) as caught:
        store.update_settings({"public_inference_url": "http://new.example"})
    assert caught.value.committed is False
    set_write_checkpoint(None)
    store.add_model({"id": "kept-model", "huggingface_id": "org/kept", "config": {}})

    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_read_settings_and_models, args=(config_dir, writer))
    proc.start()
    writer.close()
    assert reader.poll(10)
    payload = reader.recv()
    proc.join(5)
    assert proc.exitcode == 0
    assert payload["url"] == "http://old.example"
    assert payload["models"] == ["kept-model"]


def test_liveness_stays_available_while_a_settings_write_is_blocked(client):
    import threading

    from backend.data_store import get_store

    started = threading.Event()
    release = threading.Event()

    def block(name: str, path: str) -> None:
        if name == "before_fsync" and os.path.basename(path) == "settings.yaml":
            started.set()
            assert release.wait(timeout=5)

    set_write_checkpoint(block)
    writer = threading.Thread(
        target=lambda: get_store().update_settings({"marker": "blocked"})
    )
    writer.start()
    assert started.wait(timeout=2)
    try:
        result = {}

        def probe():
            result["response"] = client.get("/api/live")

        getter = threading.Thread(target=probe)
        getter.start()
        getter.join(2)
        assert not getter.is_alive()
        response = result["response"]
        assert response.status_code == 200
        assert response.json() == {"live": True}
    finally:
        release.set()
        writer.join(5)
    assert not writer.is_alive()


def test_unknown_replacement_is_not_reported_as_committed():
    from backend.store_io import StoreDurabilityError, persistence_http_error

    status, body = persistence_http_error(
        StoreDurabilityError("lost track", phase="replace", committed="unknown")
    )
    assert status == 500
    assert body["committed"] == "unknown"
    assert "could not be established" in body["detail"]
    assert body["committed"] is not True


def _operation_until_killed(data_dir: str, mode: str, connection) -> None:
    """Child process: persist an operation, then wait to be killed."""
    os.environ["STUDIO_DATA_DIR"] = data_dir
    from backend.data_store import set_write_checkpoint as child_checkpoint
    from backend.operations.supervisor import OperationSupervisor

    hits = {"before_replace": 0, "after_replace": 0}

    def hook(name: str, path: str) -> None:
        if os.path.basename(path) != "operations.yaml" or name not in hits:
            return
        hits[name] += 1
        pause = (
            (mode == "after_running" and name == "after_replace" and hits[name] == 1)
            or (
                mode == "before_terminal_replace"
                and name == "before_replace"
                and hits[name] == 2
            )
            or (
                mode == "after_terminal_replace"
                and name == "after_replace"
                and hits[name] == 2
            )
        )
        if not pause:
            return
        connection.send("ready")
        connection.close()
        while True:
            signal.pause()

    child_checkpoint(hook)
    config_dir = os.path.join(data_dir, "config")
    result_path = os.path.join(config_dir, "result-marker")
    detail = {}
    if mode in {"before_terminal_replace", "after_terminal_replace"}:
        detail = {"result_path": result_path}
    supervisor = OperationSupervisor()
    supervisor.start_operation("op-kill", "build", "engine:demo", detail=detail)
    if mode == "during_execution":
        os.makedirs(config_dir, exist_ok=True)
        with open(os.path.join(config_dir, "work-started"), "w", encoding="utf-8") as handle:
            handle.write("started\n")
        connection.send("ready")
        connection.close()
        while True:
            signal.pause()
    if mode in {"before_terminal_replace", "after_terminal_replace"}:
        os.makedirs(config_dir, exist_ok=True)
        with open(result_path, "w", encoding="utf-8") as handle:
            handle.write("finished\n")
        supervisor.finish_operation("op-kill", "failed", "finished")


def _kill_operation(data_dir: str, mode: str) -> None:
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_operation_until_killed, args=(data_dir, mode, writer))
    proc.start()
    writer.close()
    reached = reader.poll(5)
    if not reached:
        if proc.pid:
            os.kill(proc.pid, signal.SIGKILL)
        proc.join(5)
        pytest.fail(f"operation writer did not reach {mode}")
    assert reader.recv() == "ready"
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(5)
    assert not proc.is_alive()


def _reconcile_killed_store(data_dir: str, monkeypatch):
    from backend import data_store
    from backend.operations.supervisor import OperationSupervisor

    store = DataStore(config_dir=os.path.join(data_dir, "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()

    def replay(*_args, **_kwargs):
        raise AssertionError("restart replayed an operation")

    monkeypatch.setattr(supervisor, "start_operation", replay)
    result = supervisor.reconcile_startup()
    assert result["replayed"] == 0
    assert supervisor._resources == {}
    assert not os.path.exists(os.path.join(data_dir, "config", "replayed"))
    return result, store


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("after_running", "unknown"),
        ("during_execution", "unknown"),
        ("before_terminal_replace", "unknown"),
        ("after_terminal_replace", "failed"),
    ],
)
def test_restart_reconciles_a_killed_operation_without_replaying_it(
    tmp_path, monkeypatch, mode, expected
):
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    _kill_operation(data_dir, mode)
    document = _complete_mapping(tmp_path / "data" / "config" / "operations.yaml")
    assert isinstance(document.get("operations"), list)
    result, store = _reconcile_killed_store(data_dir, monkeypatch)
    assert result["outcome"] == "reconciled"
    rows = store.list_operations()
    assert [row["operation_id"] for row in rows] == ["op-kill"]
    assert rows[0]["status"] == expected
    if expected == "unknown":
        assert "could not be established" in rows[0]["message"]
        assert "not run again" in rows[0]["message"]
    if expected == "unknown":
        assert "not run again" in rows[0]["message"]
        assert result["unknown"] == 1
    if mode == "during_execution":
        assert (tmp_path / "data" / "config" / "work-started").read_text(encoding="utf-8") == "started\n"
    if mode == "after_running":
        assert not (tmp_path / "data" / "config" / "work-started").exists()


def test_failed_reconciliation_leaves_the_durable_row_and_reports_unknown(
    tmp_path, monkeypatch
):
    from backend import data_store
    from backend.operations.supervisor import OperationSupervisor

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    supervisor.start_operation("op-open", "build", "engine:demo")

    def fail(name: str, path: str) -> None:
        if name == "before_fsync" and os.path.basename(path) == "operations.yaml":
            raise OSError(errno.ENOSPC, "No space left on device")

    set_write_checkpoint(fail)
    result = supervisor.reconcile_startup()
    assert result["outcome"] == "unknown"
    assert result["replayed"] == 0
    assert "could not be stored" in result["detail"]
    assert supervisor._resources == {}
    assert supervisor._records["op-open"]["status"] == "running"
    assert store.list_operations()[0]["status"] == "running"


def test_reconcile_api_reports_an_uncertain_outcome(client):
    from backend.data_store import get_store
    from backend.operations.supervisor import get_supervisor

    get_supervisor().start_operation("op-open", "build", "engine:demo")

    def fail(name: str, path: str) -> None:
        if name == "before_fsync" and os.path.basename(path) == "operations.yaml":
            raise OSError(errno.ENOSPC, "No space left on device")

    set_write_checkpoint(fail)
    response = client.post("/api/operations/reconcile")
    assert response.status_code == 500
    body = response.json()
    assert body["code"] == "RECONCILE_UNKNOWN"
    assert body["committed"] == "unknown"
    assert body["outcome"] == "unknown"
    assert "Refresh" in body["detail"]
    assert next(
        row for row in get_store().list_operations()
        if row.get("operation_id") == "op-open"
    )["status"] == "running"
    recovery = client.get("/api/operations/recovery")
    assert recovery.status_code == 200
    assert recovery.json()["outcome"] == "unknown"
