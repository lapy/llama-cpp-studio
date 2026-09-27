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
from backend.operations.supervisor import OperationSupervisor, ResourceBusyError


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


def test_proxy_replace_failure_keeps_the_previous_file(tmp_path, monkeypatch):
    import backend.proxy.llama_swap.manager as llama_swap_manager

    path = tmp_path / "swap.yaml"
    path.write_text("models: {keep: {}}\n", encoding="utf-8")
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(path))
    removed = []
    real_remove = os.remove

    def track_remove(target):
        removed.append(target)
        return real_remove(target)

    def fail_replace(src, dst):
        if os.path.abspath(dst) == os.path.abspath(manager.config_path):
            raise OSError("injected rename failure")
        return os.replace(src, dst)

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: type("S", (), {"list_models": lambda self: []})())
    monkeypatch.setattr(
        "backend.proxy.llama_swap.config.any_active_runtime_in_db", lambda: True
    )
    monkeypatch.setattr(
        llama_swap_manager,
        "generate_llama_swap_config",
        lambda running_models, all_models=None, sidecar_payloads=None: "models: {}\n",
    )
    monkeypatch.setattr(llama_swap_manager.os, "remove", track_remove)
    monkeypatch.setattr(llama_swap_manager.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected"):
        asyncio.run(manager._write_config())
    assert path.read_text(encoding="utf-8") == "models: {keep: {}}\n"
    assert str(path) not in removed


def test_apply_keeps_a_concurrent_edit_pending(tmp_path, monkeypatch):
    import backend.proxy.llama_swap.manager as llama_swap_manager

    path = tmp_path / "swap.yaml"
    path.write_text("models: {}\n", encoding="utf-8")
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(path))

    async def compose():
        return "models: {next: {}}\n", {}

    async def allow(*_args, **_kwargs):
        return None

    def publish(content, sidecars):
        manager.mark_swap_config_stale()
        path.write_text(content, encoding="utf-8")

    monkeypatch.setattr(manager, "_compose_config", compose)
    monkeypatch.setattr(manager, "_unload_before_apply", allow)
    monkeypatch.setattr(manager, "_publish_config_files", publish)
    monkeypatch.setattr(manager, "_regenerate_start_only", allow)
    monkeypatch.setattr(manager, "_confirm_proxy_accepted", allow)

    asyncio.run(manager.user_apply_regenerate_config())
    assert manager._swap_config_stale is True
    assert "next" in path.read_text(encoding="utf-8")


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
    assert changed >= 1
    rows = store.list_operations()
    interrupted = next(row for row in rows if row["operation_id"] == "build-1")
    assert interrupted["status"] == "interrupted"
    assert "interrupted" in interrupted["message"]


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


def test_failed_publish_after_restart_keeps_the_referenced_sidecar(tmp_path, monkeypatch):
    import backend.proxy.llama_swap.manager as llama_swap_manager

    root = tmp_path / "sidecars"
    root.mkdir()
    live = root / "model.r1.json"
    live.write_text('{"generation": 1}\n', encoding="utf-8")
    raw = root / "model.json"
    config = tmp_path / "swap.yaml"
    original = f"cmd: {live}\n"
    config.write_text(original, encoding="utf-8")
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(config))
    monkeypatch.setattr(
        "backend.engines.audio_cpp.manager.get_audio_cpp_manager",
        lambda: type("Mgr", (), {"server_configs_dir": str(root)})(),
    )
    real_replace = os.replace

    def fail_config_replace(src, dst):
        if os.path.abspath(dst) == os.path.abspath(config):
            raise OSError("injected")
        return real_replace(src, dst)

    monkeypatch.setattr(llama_swap_manager.os, "replace", fail_config_replace)
    with pytest.raises(OSError, match="injected"):
        manager._publish_config_files(
            f"cmd: {raw}\n",
            {str(raw): {"generation": 2}},
        )
    assert json.loads(live.read_text(encoding="utf-8"))["generation"] == 1
    assert config.read_text(encoding="utf-8") == original


def test_rejected_proxy_reload_is_not_acceptance(tmp_path):
    import backend.proxy.llama_swap.manager as llama_swap_manager

    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = type("Proc", (), {"poll": lambda self: None})()
    manager._arm_reload_watch()
    manager._observe_proxy_log("failed to reload config: models.demo.cmd is required")

    def health_must_not_decide():
        raise AssertionError("health was treated as config acceptance")

    manager._client = health_must_not_decide
    with pytest.raises(RuntimeError, match="rejected the candidate"):
        asyncio.run(manager._confirm_proxy_accepted())


def test_healthy_proxy_without_reload_is_not_acceptance(tmp_path):
    import backend.proxy.llama_swap.manager as llama_swap_manager

    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = type("Proc", (), {"poll": lambda self: None})()
    manager._reload_confirm_timeout = 0.05
    manager._arm_reload_watch()
    manager._client = lambda: (_ for _ in ()).throw(
        AssertionError("health was treated as config acceptance")
    )
    with pytest.raises(RuntimeError, match="did not confirm"):
        asyncio.run(manager._confirm_proxy_accepted())


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
