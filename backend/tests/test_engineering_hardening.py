"""Regression coverage for the engineering improvement plan."""

import asyncio
import json
import os
import subprocess
import threading
import time
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from backend.data_store import (
    DataStore,
    DuplicateIdentifierError,
    StorageCorruptionError,
)
from backend.operations.cancel import terminate_process_tree
from backend.operations.progress import get_progress_manager
from backend.operations.supervisor import (
    OperationSupervisor,
    ResourceBusyError,
    get_supervisor,
)


def test_concurrent_setting_updates_all_survive(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    barrier = threading.Barrier(2)

    def update(key, value):
        barrier.wait(timeout=2)
        store.update_settings({key: value})

    threads = [
        threading.Thread(target=update, args=("alpha", 1)),
        threading.Thread(target=update, args=("beta", 2)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    settings = store.get_settings()
    assert settings["alpha"] == 1
    assert settings["beta"] == 2
    assert settings["proxy_port"] == 2000


def test_corrupt_settings_are_not_overwritten(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    path = tmp_path / "config" / "settings.yaml"
    path.write_text("settings: [\n", encoding="utf-8")
    with pytest.raises(StorageCorruptionError):
        store.update_settings({"proxy_port": 2001})
    assert path.read_text(encoding="utf-8") == "settings: [\n"
    assert list(path.parent.glob("settings.yaml.corrupt-*"))


def test_failed_replace_preserves_the_last_valid_document(tmp_path, monkeypatch):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.update_settings({"proxy_port": 2000, "kept": "yes"})
    path = tmp_path / "config" / "settings.yaml"
    before = path.read_text(encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("injected replace failure")

    monkeypatch.setattr("backend.data_store.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        store.update_settings({"proxy_port": 2001})
    assert path.read_text(encoding="utf-8") == before
    assert "kept:" in before


def test_duplicate_model_and_version_ids_are_rejected(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_model({"id": "same", "huggingface_id": "org/same", "config": {}})
    with pytest.raises(DuplicateIdentifierError):
        store.add_model({"id": "same", "huggingface_id": "org/other", "config": {}})
    assert [model["id"] for model in store.list_models()] == ["same"]

    store.add_engine_version("llama_cpp", {"version": "v1"})
    with pytest.raises(DuplicateIdentifierError):
        store.add_engine_version("llama_cpp", {"version": "v1"})
    assert len(store.get_engine_versions("llama_cpp")) == 1












@pytest.mark.asyncio
async def test_cancellation_keeps_manager_busy_until_termination(monkeypatch):
    from backend.operations.cancellable import CancellableOperationManager

    manager = CancellableOperationManager()
    manager.MANAGER_NAME = "test"
    task_id = await manager._begin_operation("install", "Install test")
    worker_started = asyncio.Event()

    async def worker():
        worker_started.set()
        await asyncio.Event().wait()

    manager._create_task(worker())
    await worker_started.wait()
    termination_started = asyncio.Event()
    release_termination = asyncio.Event()

    async def gated_cancel(**_kwargs):
        termination_started.set()
        await release_termination.wait()
        current = _kwargs["current_task"]
        current.cancel()
        with pytest.raises(asyncio.CancelledError):
            await current
        return True

    monkeypatch.setattr(
        "backend.operations.cancellable.cancel_running_operation", gated_cancel
    )
    assert manager.cancel_task(task_id)["ok"] is True
    await termination_started.wait()
    assert manager.is_operation_running()
    assert manager.progress_task_id == task_id
    assert get_progress_manager().get_task(task_id)["status"] == "cancelling"
    release_termination.set()
    await manager.wait_for_cancellation()
    assert not manager.is_operation_running()
    assert get_progress_manager().get_task(task_id)["status"] == "cancelled"


@pytest.mark.asyncio
async def test_supervisor_shutdown_drain_waits_for_operation_cleanup():
    supervisor = OperationSupervisor()
    worker_started = asyncio.Event()
    cleanup_finished = asyncio.Event()

    async def worker():
        worker_started.set()
        await asyncio.Event().wait()

    task = supervisor.spawn("owned", worker())

    async def cancel():
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        cleanup_finished.set()

    supervisor._cancellers["owned"] = cancel
    await worker_started.wait()
    await supervisor.drain()
    assert cleanup_finished.is_set()
    assert task.done()


@pytest.mark.asyncio
async def test_supervisor_drain_terminates_native_command_process(tmp_path):
    from backend.engines.llama_cpp.manager import LlamaManager

    pid_path = tmp_path / "native.pid"
    script = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"target = {str(pid_path)!r}\n"
        "temporary = target + '.tmp'\n"
        "with open(temporary, 'w', encoding='utf-8') as handle:\n"
        "    handle.write(str(os.getpid()))\n"
        "    handle.flush()\n"
        "    os.fsync(handle.fileno())\n"
        "os.replace(temporary, target)\n"
        "time.sleep(60)\n"
    )
    supervisor = OperationSupervisor()
    manager = LlamaManager()
    supervisor.spawn(
        "native-build",
        manager._run_command_streaming(["python3", "-c", script]),
    )
    pid = None
    for _ in range(100):
        text = pid_path.read_text(encoding="utf-8").strip() if pid_path.exists() else ""
        if text.isdigit():
            pid = int(text)
            break
        await asyncio.sleep(0.01)
    assert pid is not None

    await supervisor.drain()

    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_supervisor_drain_waits_for_cleanup_after_worker_finishes(monkeypatch):
    from backend.operations.cancellable import CancellableOperationManager

    manager = CancellableOperationManager()
    manager.MANAGER_NAME = "cleanup-race"
    task_id = await manager._begin_operation("install", "Install test")
    worker_release = asyncio.Event()
    cleanup_started = asyncio.Event()
    cleanup_release = asyncio.Event()

    async def worker():
        await worker_release.wait()

    async def delayed_cleanup(**_kwargs):
        cleanup_started.set()
        await cleanup_release.wait()
        return True

    manager._create_task(worker())
    monkeypatch.setattr(
        "backend.operations.cancellable.cancel_running_operation", delayed_cleanup
    )
    manager.cancel_task(task_id)
    await cleanup_started.wait()
    worker_release.set()
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    drain = asyncio.create_task(get_supervisor().drain())
    await asyncio.sleep(0.01)
    assert not drain.done()
    cleanup_release.set()
    await drain
    assert get_progress_manager().get_task(task_id)["status"] == "cancelled"




def test_studio_data_dir_environment_is_authoritative(tmp_path, monkeypatch):
    from backend.paths import studio_data_dir
    from backend.engines.audio_cpp.manager import _data_root
    from backend.engines.llama_cpp.manager import LlamaManager
    from backend.tests.venv_fixtures import LMDeployInstaller

    isolated = tmp_path / "isolated"
    monkeypatch.setenv("STUDIO_DATA_DIR", str(isolated))
    assert studio_data_dir() == str(isolated.resolve())
    assert _data_root() == str(isolated.resolve())
    assert LlamaManager().llama_dir == str(isolated / "llama-cpp")
    assert LMDeployInstaller()._root_dir == str(isolated / "lmdeploy")


@pytest.mark.asyncio
async def test_upstream_check_yields_while_the_request_is_in_flight(monkeypatch):
    from backend.services.upstream_versions import check_engine_updates

    started = asyncio.Event()
    release = asyncio.Event()

    async def slow(_url):
        started.set()
        await release.wait()
        response = httpx.Response(200, json=[])
        return response

    monkeypatch.setattr("backend.services.upstream_versions.get_text", slow)
    task = asyncio.create_task(check_engine_updates("ik_llama"))
    await asyncio.wait_for(started.wait(), timeout=0.2)
    await asyncio.wait_for(asyncio.sleep(0.01), timeout=0.1)
    release.set()
    result = await task
    assert result["latest_release"] is None


def test_github_requests_have_a_deadline():
    from backend.engines.llama_cpp import github_refs as llama_github_refs

    with patch.object(llama_github_refs.requests, "get", return_value=type("R", (), {
        "status_code": 404,
        "json": lambda self: [],
        "raise_for_status": lambda self: None,
    })()) as get:
        llama_github_refs.fetch_latest_release_for_repository_source("llama.cpp")
    assert get.call_args.kwargs["timeout"] == llama_github_refs.GITHUB_TIMEOUT


def test_process_tree_cancellation_stops_the_child():
    process = subprocess.Popen(
        ["bash", "-c", "sleep 30"],
        start_new_session=True,
    )
    try:
        terminate_process_tree(process.pid, term_timeout=0.3, kill_timeout=1.0)
        process.wait(timeout=3)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.poll() is not None


def test_cancellation_stops_a_child_that_left_the_process_group(tmp_path):
    pid_file = tmp_path / "child.pid"
    script = (
        "import os, time\n"
        f"pid_file = {str(pid_file)!r}\n"
        "child = os.fork()\n"
        "if child == 0:\n"
        "    os.setsid()\n"
        "    open(pid_file, 'w', encoding='utf-8').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    raise SystemExit(0)\n"
        "os.wait()\n"
    )
    process = subprocess.Popen(
        ["python3", "-c", script],
        start_new_session=True,
    )
    child_pid = 0
    try:
        for _ in range(50):
            if pid_file.exists() and pid_file.read_text(encoding="utf-8").strip():
                child_pid = int(pid_file.read_text(encoding="utf-8"))
                break
            time.sleep(0.05)
        assert child_pid
        terminate_process_tree(process.pid, term_timeout=0.4, kill_timeout=1.0)
        process.wait(timeout=3)
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if process.poll() is None:
            process.kill()
        if child_pid:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def test_supervisor_rejects_the_same_install_directory_and_reconciles_restart(tmp_path, monkeypatch):
    from backend import data_store

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    supervisor = OperationSupervisor()
    supervisor.start_operation("build-1", "build", str(tmp_path / "install"))
    with pytest.raises(ResourceBusyError):
        supervisor.start_operation("build-2", "build", str(tmp_path / "install"))
    changed = supervisor.reconcile_startup()
    assert changed["outcome"] == "reconciled"
    assert changed["unknown"] >= 1
    assert changed["interrupted"] == 0
    assert changed["replayed"] == 0
    rows = store.list_operations()
    unknown = next(row for row in rows if row["operation_id"] == "build-1")
    assert unknown["status"] == "unknown"
    assert "not run again" in unknown["message"]
    restored = get_progress_manager().get_task("build-1")
    assert restored["status"] == "unknown"
    assert restored["metadata"]["recovered"] is True


def test_unauthenticated_remote_client_cannot_mutate(monkeypatch):
    from backend.main import app

    monkeypatch.setenv("STUDIO_ACCESS_MODE", "local")
    monkeypatch.setattr("backend.access_policy.in_container", lambda: False)
    transport = httpx.ASGITransport(app=app, client=("10.9.8.7", 4000))

    async def request():
        async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
            live = await client.get("/api/live")
            denied = await client.post("/api/models", json={"id": "x"})
            return live.status_code, denied.status_code

    live, denied = asyncio.run(request())
    assert live == 200
    assert denied == 403


def test_remote_mode_requires_a_token_and_csrf(monkeypatch):
    from backend.main import app

    monkeypatch.setenv("STUDIO_ACCESS_MODE", "remote")
    monkeypatch.setenv("STUDIO_API_TOKEN", "studio-test-token")

    with TestClient(app) as client:
        assert client.get("/api/models").status_code == 401
        assert client.get(
            "/api/models", headers={"Authorization": "Bearer studio-test-token"}
        ).status_code != 401
        login = client.post("/api/session", json={"token": "studio-test-token"})
        assert login.status_code == 200
        assert client.post("/api/models", json={"id": "x"}).status_code == 401
        csrf = login.cookies.get("studio_csrf")
        accepted = client.post(
            "/api/models",
            json={"id": "x"},
            headers={"X-CSRF-Token": csrf},
        )
        assert accepted.status_code != 401


def test_openapi_operation_ids_are_unique():
    from backend.main import app

    schema = app.openapi()
    ids = []
    for methods in schema["paths"].values():
        for method, operation in methods.items():
            if method.startswith("x-") or "operationId" not in operation:
                continue
            ids.append(operation["operationId"])
    assert ids
    assert len(ids) == len(set(ids))


def test_lifespan_starts_and_stops():
    from backend.main import app

    with TestClient(app) as client:
        live = client.get("/api/live")
        ready = client.get("/api/ready")
    assert live.status_code == 200
    assert live.json()["live"] is True
    assert ready.status_code == 200
    assert ready.json()["ready"] is True








def _local_status(monkeypatch, host: str) -> int:
    from backend.main import app

    monkeypatch.setenv("STUDIO_ACCESS_MODE", "local")
    transport = httpx.ASGITransport(app=app, client=(host, 4000))

    async def request():
        async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
            return (await client.get("/api/status")).status_code

    return asyncio.run(request())


def test_compose_bridge_is_allowed_only_inside_the_container(monkeypatch):
    monkeypatch.setattr("backend.access_policy.in_container", lambda: True)
    assert _local_status(monkeypatch, "172.18.0.1") == 200
    assert _local_status(monkeypatch, "8.8.8.8") == 403
    monkeypatch.setattr("backend.access_policy.in_container", lambda: False)
    assert _local_status(monkeypatch, "172.18.0.1") == 403


def test_sync_requests_include_timeout_on_ik_tip():
    from backend.engines.llama_cpp import github_refs as llama_github_refs

    response = type("R", (), {
        "status_code": 200,
        "raise_for_status": lambda self: None,
        "json": lambda self: [],
    })()
    with patch.object(llama_github_refs.requests, "get", return_value=response) as get:
        assert llama_github_refs.fetch_ik_llama_main_tip_commit() is None
    assert get.call_args.kwargs["timeout"] == (5, 20)
