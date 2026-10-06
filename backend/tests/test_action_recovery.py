"""Action recovery: evidence, lookup, and whether a retry is safe.

These tests use temporary directories only. They do not start a proxy, a
build, or a download.
"""

import json
import os
import time
from urllib.parse import quote

import pytest

import httpx

from backend.operations.action_recovery import (
    classify_action,
    classify_apply_journal,
)
from backend.operations.supervisor import OperationSupervisor
from backend.services.model_runtime_apply import reconcile_journals


def test_a_start_that_never_dispatched_can_be_retried():
    decision = classify_action(
        "model_start",
        {"effect_started": False, "model_id": "org/model"},
    )
    assert decision["evidence"] == "negative"
    assert decision["status"] == "interrupted"
    assert decision["retry"] == "safe"
    assert decision["replayed"] == 0


def test_a_dispatched_start_stays_unknown_without_a_verified_observation():
    decision = classify_action(
        "model_start",
        {"effect_started": True},
        {"quality": "unreachable", "state": "stopped"},
    )
    assert decision["evidence"] == "unproven"
    assert decision["status"] == "unknown"
    assert decision["retry"] == "withheld"
    assert decision["replayed"] == 0


def test_a_verified_stopped_model_does_not_make_a_start_retry_safe():
    decision = classify_action(
        "model_start",
        {"effect_started": True},
        {"quality": "verified", "state": "stopped"},
    )
    assert decision["evidence"] == "unproven"
    assert decision["retry"] == "withheld"
    assert decision["replayed"] == 0


def test_a_verified_running_model_does_not_start_again():
    decision = classify_action(
        "model_start",
        {"effect_started": True},
        {"quality": "verified", "state": "running"},
    )
    assert decision["evidence"] == "completed"
    assert decision["retry"] == "unnecessary"


def test_a_stop_retries_only_while_the_model_is_still_active():
    running = classify_action(
        "model_stop",
        {"effect_started": True},
        {"quality": "verified", "state": "running"},
    )
    stopped = classify_action(
        "model_stop",
        {"effect_started": True},
        {"quality": "verified", "state": "stopped"},
    )
    assert running["evidence"] == "unproven"
    assert running["retry"] == "withheld"
    assert stopped["retry"] == "unnecessary"
    assert stopped["evidence"] == "completed"


def test_saved_settings_do_not_complete_a_build_or_install():
    for kind in ("build", "install", "install_source"):
        decision = classify_action(kind, {"settings_saved": True, "engine": "llama_cpp"})
        assert decision["evidence"] == "unproven"
        assert decision["status"] == "unknown"
        assert decision["retry"] == "withheld"
        assert decision["replayed"] == 0


def test_a_build_that_recorded_it_had_not_started_can_be_retried():
    decision = classify_action("build", {"effect_started": False, "settings_saved": True})
    assert decision["evidence"] == "negative"
    assert decision["retry"] == "safe"


def test_an_unknown_kind_stays_unproven():
    decision = classify_action("future_engine_task", {"settings_saved": True})
    assert decision["evidence"] == "unproven"
    assert decision["retry"] == "withheld"
    assert decision["replayed"] == 0


def test_apply_lookup_uses_the_pointer_and_receipt():
    untouched = classify_apply_journal(
        {
            "phase": "validated",
            "mode": "restart_now",
            "published_revision": "a" * 64,
            "desired_revision": "b" * 64,
            "was_running": True,
        },
        published_revision="a" * 64,
        running_revision="a" * 64,
    )
    assert untouched["evidence"] == "negative"
    assert untouched["retry"] == "safe"
    assert untouched["replayed"] == 0

    stopped_without_proof = classify_apply_journal(
        {
            "phase": "stopping",
            "mode": "restart_now",
            "published_revision": "a" * 64,
            "running_revision": "a" * 64,
            "was_running": True,
        },
        published_revision="a" * 64,
        running_revision=None,
    )
    assert stopped_without_proof["evidence"] == "unproven"
    assert stopped_without_proof["retry"] == "withheld"

    other_generation = classify_apply_journal(
        {
            "operation_id": "op-1",
            "phase": "starting",
            "mode": "restart_now",
            "desired_revision": "b" * 64,
            "was_running": True,
        },
        published_revision="b" * 64,
        running_revision="b" * 64,
        observed_launch_id="someone-else",
    )
    assert other_generation["evidence"] == "unproven"
    assert other_generation["retry"] == "withheld"

    observed_start = classify_apply_journal(
        {
            "operation_id": "op-1",
            "phase": "starting",
            "mode": "restart_now",
            "desired_revision": "b" * 64,
            "was_running": True,
        },
        published_revision="b" * 64,
        running_revision="b" * 64,
        observed_launch_id="op-1",
    )
    assert observed_start["evidence"] == "completed"
    assert observed_start["retry"] == "unnecessary"
    assert observed_start["replayed"] == 0


def test_reconcile_does_not_replay_a_start_that_was_dispatched(tmp_path, monkeypatch):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    supervisor.start_operation(
        "start-1",
        "model_start",
        "model_start:org/model",
        detail={"effect_started": True, "settings_saved": True},
    )

    def replay(*_args, **_kwargs):
        raise AssertionError("restart replayed an action")

    monkeypatch.setattr(supervisor, "start_operation", replay)
    result = supervisor.reconcile_startup()
    assert result["replayed"] == 0
    assert result["unknown"] == 1
    assert result["interrupted"] == 0
    row = store.list_operations()[0]
    assert row["status"] == "unknown"
    assert row["detail"]["recovery"]["retry"] == "withheld"


def test_reconcile_records_a_build_that_had_not_started(tmp_path, monkeypatch):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    supervisor.start_operation(
        "build-not-started",
        "build",
        "install-dir",
        detail={"effect_started": False, "settings_saved": True},
    )
    result = supervisor.reconcile_startup()
    assert result["replayed"] == 0
    assert result["interrupted"] == 1
    row = store.list_operations()[0]
    assert row["status"] == "interrupted"
    assert row["detail"]["recovery"]["retry"] == "safe"
    assert row["detail"]["settings_saved"] is True


def test_same_process_means_the_apply_stop_did_not_happen(tmp_path):
    from backend.proxy.manifests import LaunchManifestStore

    store = LaunchManifestStore(str(tmp_path / "models"))
    revision = "a" * 64
    store.write_receipt(
        "model-a",
        revision=revision,
        launch_id="launch-1",
        pid=os.getpid(),
        start_ticks=_ticks(os.getpid()),
    )
    ops = tmp_path / "operations"
    ops.mkdir()
    (ops / f"{'c' * 32}.json").write_text(
        json.dumps(
            {
                "operation_id": "c" * 32,
                "model_id": "model-a",
                "phase": "stopping",
                "mode": "restart_now",
                "was_running": True,
                "published_revision": None,
                "running_revision": revision,
                "prior_launch_id": "launch-1",
                "desired_revision": "b" * 64,
            }
        ),
        encoding="utf-8",
    )
    assert reconcile_journals(store) == 1
    journal = json.loads((ops / f"{'c' * 32}.json").read_text(encoding="utf-8"))
    assert journal["phase"] == "interrupted"
    assert journal["recovery"]["retry"] == "safe"
    assert journal["recovery"]["replayed"] == 0


def test_unknown_start_is_not_sent_again_until_it_is_observed(client, monkeypatch, tmp_path):
    from backend.tests.test_api_routes import _install_temp_store, _seed_model

    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.upsert_operation(
        {
            "operation_id": "start-unknown",
            "kind": "model_start",
            "status": "unknown",
            "resource_key": "model_start:org/model",
            "detail": {"effect_started": True},
            "message": "unknown",
            "updated_at": time.time(),
        }
    )
    called = {"start": 0}

    async def fake_running(self, method, path, **kwargs):
        raise ConnectionError("proxy down")

    async def fake_start(self, model_name):
        called["start"] += 1
        return httpx.Response(202, json={"model": model_name, "state": "loading"})

    from backend.proxy.llama_swap.client import LlamaSwapClient

    monkeypatch.setattr(LlamaSwapClient, "request", fake_running)
    monkeypatch.setattr(LlamaSwapClient, "start_model_passthrough", fake_start)
    response = client.post(f"/api/models/{quote('org/model', safe='')}/start")
    assert response.status_code == 409
    body = response.json()["detail"]
    assert body["code"] == "ACTION_RETRY_WITHHELD"
    assert body["operation_id"] == "start-unknown"
    assert body["state_token"]
    assert called["start"] == 0


def test_a_verified_stopped_model_does_not_allow_a_new_start(client, monkeypatch, tmp_path):
    from backend.tests.test_api_routes import _install_temp_store, _seed_model

    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.upsert_operation(
        {
            "operation_id": "start-unknown",
            "kind": "model_start",
            "status": "unknown",
            "resource_key": "model_start:org/model",
            "detail": {"effect_started": True},
            "message": "unknown",
            "updated_at": time.time(),
        }
    )

    async def fake_running(self, method, path, **kwargs):
        assert method == "GET"
        assert path == "/running"
        return httpx.Response(200, json={"running": []})

    from backend.proxy.llama_swap.client import LlamaSwapClient

    called = {"start": 0}

    async def fake_start(self, model_name):
        called["start"] += 1
        row = next(
            item for item in store.list_operations()
            if item.get("status") == "running"
        )
        assert row["detail"]["effect_started"] is True
        return httpx.Response(202, json={"model": model_name, "state": "loading"})

    monkeypatch.setattr(LlamaSwapClient, "request", fake_running)
    monkeypatch.setattr(LlamaSwapClient, "start_model_passthrough", fake_start)
    path = f"/api/models/{quote('org/model', safe='')}/start"
    withheld = client.post(path)
    assert withheld.status_code == 409
    detail = withheld.json()["detail"]
    assert detail["code"] == "ACTION_RETRY_WITHHELD"
    assert detail["operation_id"] == "start-unknown"
    assert "may already have happened" in detail["message"]
    assert called["start"] == 0

    stale = client.post(path, json={
        "confirm_operation_id": "start-unknown",
        "confirm_state": "not-the-token",
    })
    assert stale.status_code == 409
    assert called["start"] == 0

    allowed = client.post(path, json={
        "confirm_operation_id": detail["operation_id"],
        "confirm_state": detail["state_token"],
    })
    assert allowed.status_code == 202
    assert called["start"] == 1

    reused = client.post(path, json={
        "confirm_operation_id": detail["operation_id"],
        "confirm_state": detail["state_token"],
    })
    assert reused.status_code == 409
    assert called["start"] == 1


def test_the_effect_fence_is_durable_before_a_start_is_sent(client, monkeypatch, tmp_path):
    from backend.operations.supervisor import OperationSupervisor
    from backend.store_io import StoreDurabilityError
    from backend.tests.test_api_routes import _install_temp_store, _seed_model

    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    called = {"start": 0}

    async def fake_running(self, method, path, **kwargs):
        return httpx.Response(200, json={"running": []})

    async def fake_start(self, model_name):
        called["start"] += 1
        return httpx.Response(202, json={"model": model_name})

    async def note_failed(self, operation_id):
        raise StoreDurabilityError("replace did not happen", phase="replace", committed=False)

    from backend.proxy.llama_swap.client import LlamaSwapClient

    monkeypatch.setattr(LlamaSwapClient, "request", fake_running)
    monkeypatch.setattr(LlamaSwapClient, "start_model_passthrough", fake_start)
    monkeypatch.setattr(OperationSupervisor, "note_effect_started", note_failed)
    response = client.post(f"/api/models/{quote('org/model', safe='')}/start")
    assert response.status_code == 503
    assert response.json()["detail"]["committed"] is False
    assert called["start"] == 0
    row = store.list_operations()[-1]
    assert row["status"] == "interrupted"
    assert row["detail"]["effect_started"] is False


def test_an_unknown_fence_does_not_authorize_a_start(client, monkeypatch, tmp_path):
    from backend.operations.supervisor import OperationSupervisor
    from backend.store_io import StoreDurabilityError
    from backend.tests.test_api_routes import _install_temp_store, _seed_model

    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    called = {"start": 0}

    async def fake_running(self, method, path, **kwargs):
        return httpx.Response(200, json={"running": []})

    async def fake_start(self, model_name):
        called["start"] += 1
        return httpx.Response(202, json={"model": model_name})

    async def note_unknown(self, operation_id):
        raise StoreDurabilityError("outcome unknown", phase="replace", committed="unknown")

    from backend.proxy.llama_swap.client import LlamaSwapClient

    monkeypatch.setattr(LlamaSwapClient, "request", fake_running)
    monkeypatch.setattr(LlamaSwapClient, "start_model_passthrough", fake_start)
    monkeypatch.setattr(OperationSupervisor, "note_effect_started", note_unknown)
    response = client.post(f"/api/models/{quote('org/model', safe='')}/start")
    assert response.status_code == 500
    assert response.json()["detail"]["committed"] == "unknown"
    assert called["start"] == 0
    row = store.list_operations()[-1]
    assert row["status"] == "unknown"
    assert row["detail"]["effect_started"] is True


def test_build_admission_rejects_an_active_attempt_and_a_restarted_unknown(tmp_path, monkeypatch):
    import threading

    from backend import data_store
    from backend.operations.action_recovery import (
        ActionAdmissionError,
        bind_action_confirmation,
        state_token,
    )
    from backend.operations.supervisor import OperationSupervisor, ResourceBusyError

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    bind_action_confirmation({})
    supervisor = OperationSupervisor()
    results = []
    errors = []

    def attempt(index):
        try:
            supervisor.start_operation(
                f"build-sim-{index}",
                "build",
                "install/same",
                detail={"effect_started": False},
            )
            results.append(index)
        except (ResourceBusyError, ActionAdmissionError) as exc:
            errors.append(exc)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 1
    assert len(errors) == 1

    restarted = OperationSupervisor()
    with pytest.raises((ResourceBusyError, ActionAdmissionError)):
        restarted.start_operation(
            "build-after-restart",
            "build",
            "install/same",
            detail={"effect_started": False},
        )


def test_build_retry_follows_negative_evidence_and_a_bound_confirmation(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.action_recovery import (
        ActionAdmissionError,
        bind_action_confirmation,
        state_token,
    )
    from backend.operations.supervisor import OperationSupervisor

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    now = time.time()
    store.upsert_operation({
        "operation_id": "build-never",
        "kind": "build",
        "status": "interrupted",
        "resource_key": "install/never",
        "detail": {"effect_started": False, "settings_saved": True},
        "message": "not started",
        "updated_at": now,
    })
    store.upsert_operation({
        "operation_id": "build-unknown",
        "kind": "build",
        "status": "unknown",
        "resource_key": "install/unknown",
        "detail": {"effect_started": True, "settings_saved": True},
        "message": "unknown",
        "updated_at": now,
    })
    bind_action_confirmation({})
    supervisor = OperationSupervisor()
    admitted = supervisor.start_operation(
        "build-retry-never",
        "build",
        "install/never",
        detail={"effect_started": False},
    )
    assert admitted["operation_id"] == "build-retry-never"

    with pytest.raises(ActionAdmissionError) as withheld:
        supervisor.start_operation(
            "build-retry-unknown",
            "build",
            "install/unknown",
            detail={"effect_started": False},
        )
    detail = withheld.value.detail
    assert detail["code"] == "ACTION_RETRY_WITHHELD"
    assert detail["operation_id"] == "build-unknown"
    assert "may already have happened" in detail["message"]

    row = next(item for item in store.list_operations() if item["operation_id"] == "build-unknown")
    token = state_token(row)
    bind_action_confirmation({"confirm_operation_id": "build-unknown", "confirm_state": "stale"})
    with pytest.raises(ActionAdmissionError):
        supervisor.start_operation(
            "build-stale",
            "build",
            "install/unknown",
            detail={"effect_started": False},
        )

    bind_action_confirmation({"confirm_operation_id": "build-unknown", "confirm_state": token})
    confirmed = supervisor.start_operation(
        "build-confirmed",
        "build",
        "install/unknown",
        detail={"effect_started": False},
    )
    assert confirmed["operation_id"] == "build-confirmed"
    supervisor.finish_operation("build-confirmed", "succeeded", "")
    bind_action_confirmation({"confirm_operation_id": "build-unknown", "confirm_state": token})
    with pytest.raises(ActionAdmissionError):
        supervisor.start_operation(
            "build-reused",
            "build",
            "install/unknown",
            detail={"effect_started": False},
        )
    bind_action_confirmation({})


def test_a_repo_download_and_a_catalog_install_share_one_resource():
    from backend.operations.action_recovery import admit_action, huggingface_resource_key

    repo = huggingface_resource_key("org/audio")
    one_file = huggingface_resource_key("org/audio", "model.gguf")
    assert repo == "hf:org/audio"
    assert one_file == "hf:org/audio:model.gguf"
    decision = admit_action(
        [{
            "operation_id": "catalog-1",
            "resource_key": repo,
            "status": "running",
            "detail": {"effect_started": True},
            "updated_at": time.time(),
        }],
        one_file,
    )
    assert decision["admit"] is False
    assert decision["code"] == "ACTION_IN_FLIGHT"
    activate = admit_action(
        [{
            "operation_id": "activate-1",
            "resource_key": "engine:llama_cpp",
            "status": "running",
            "detail": {"effect_started": True},
            "updated_at": time.time(),
        }],
        "engine:llama_cpp",
    )
    assert activate["code"] == "ACTION_IN_FLIGHT"
    proxy = admit_action(
        [{
            "operation_id": "apply-global",
            "resource_key": "proxy:runtime",
            "status": "running",
            "detail": {"effect_started": True},
            "updated_at": time.time(),
        }],
        "runtime-apply:org--model",
    )
    assert proxy["code"] == "ACTION_IN_FLIGHT"
    reverse_file = admit_action(
        [{
            "operation_id": "file-1",
            "resource_key": one_file,
            "status": "running",
            "detail": {"effect_started": True},
            "updated_at": time.time(),
        }],
        repo,
    )
    assert reverse_file["code"] == "ACTION_IN_FLIGHT"
    reverse_apply = admit_action(
        [{
            "operation_id": "apply-model",
            "resource_key": "runtime-apply:org--model",
            "status": "running",
            "detail": {"effect_started": True},
            "updated_at": time.time(),
        }],
        "proxy:runtime",
    )
    assert reverse_apply["code"] == "ACTION_IN_FLIGHT"


def test_source_sync_overlaps_the_same_installation_in_both_directions():
    from backend.operations.action_recovery import admit_action, engine_installation_key

    installation = engine_installation_key("llama_cpp", "/opt/llama/source-main")
    other = engine_installation_key("llama_cpp", "/opt/llama/other")
    now = time.time()

    def row(operation_id, key, kind):
        return {
            "operation_id": operation_id,
            "kind": kind,
            "resource_key": key,
            "status": "running",
            "detail": {"effect_started": True, "engine": "llama_cpp"},
            "updated_at": now,
        }

    pairs = [
        (row("build-1", installation, "build"), installation, "sync_source"),
        (row("sync-1", installation, "sync_source"), installation, "build"),
        (row("update-1", installation, "update"), installation, "sync_source"),
        (row("sync-2", installation, "sync_source"), installation, "update"),
        (row("activate-1", "engine:llama_cpp", "activate"), installation, "sync_source"),
        (row("sync-3", installation, "sync_source"), "engine:llama_cpp", "activate"),
        (row("delete-1", installation, "remove"), installation, "sync_source"),
        (row("sync-4", installation, "sync_source"), installation, "remove"),
    ]
    for existing, requested, _kind in pairs:
        decision = admit_action([existing], requested, depends_on="cuda:toolkit")
        assert decision["code"] == "ACTION_IN_FLIGHT", existing["operation_id"]
    assert admit_action([row("sync-other", other, "sync_source")], installation)["admit"] is True


def test_cuda_uninstall_overlaps_install_and_dependent_work_in_both_directions():
    from backend.operations.action_recovery import CUDA_TOOLKIT_KEY, admit_action

    now = time.time()
    uninstall = {
        "operation_id": "cuda-uninstall",
        "kind": "uninstall",
        "resource_key": CUDA_TOOLKIT_KEY,
        "status": "running",
        "detail": {"effect_started": True},
        "updated_at": now,
    }
    install = {
        "operation_id": "cuda-install",
        "kind": "install",
        "resource_key": CUDA_TOOLKIT_KEY,
        "status": "running",
        "detail": {"effect_started": True},
        "updated_at": now,
    }
    build = {
        "operation_id": "llama-build",
        "kind": "build",
        "resource_key": "engine:llama_cpp:/opt/llama/source-main",
        "status": "running",
        "detail": {"effect_started": True, "depends_on": CUDA_TOOLKIT_KEY},
        "updated_at": now,
    }
    assert admit_action([install], CUDA_TOOLKIT_KEY)["code"] == "ACTION_IN_FLIGHT"
    assert admit_action([uninstall], CUDA_TOOLKIT_KEY)["code"] == "ACTION_IN_FLIGHT"
    assert admit_action(
        [build], CUDA_TOOLKIT_KEY
    )["code"] == "ACTION_IN_FLIGHT"
    assert admit_action(
        [uninstall],
        build["resource_key"],
        depends_on=CUDA_TOOLKIT_KEY,
    )["code"] == "ACTION_IN_FLIGHT"


def test_cancellation_keeps_ownership_when_restart_cannot_verify_termination(
    tmp_path, monkeypatch
):
    from backend import data_store
    from backend.operations.action_recovery import admit_action

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    supervisor = OperationSupervisor()
    supervisor.start_operation(
        "download-cancel",
        "download",
        "hf:org/model:file.gguf",
        detail={"effect_started": True},
    )
    supervisor.note_cancellation("download-cancel")
    result = supervisor.reconcile_startup()
    assert result["replayed"] == 0
    assert result["unknown"] == 0
    row = store.list_operations()[0]
    assert row["status"] == "cancelling"
    assert supervisor._resources.get("hf:org/model:file.gguf") == "download-cancel"
    decision = admit_action(store.list_operations(), "hf:org/model:file.gguf")
    assert decision["code"] == "ACTION_IN_FLIGHT"


@pytest.mark.asyncio
async def test_only_a_pre_effect_rejection_releases_without_confirmation(
    tmp_path, monkeypatch
):
    from backend import data_store
    from backend.operations.action_recovery import admit_action
    from backend.operations.exclusive import exclusive_action

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)

    class Early(Exception):
        before_side_effect = True

    class AfterDispatch(Exception):
        pass

    from backend.operations.supervisor import get_supervisor
    from backend.store_io import drain_store_io

    with pytest.raises(Early):
        async with exclusive_action("runtime_apply", "apply:early"):
            raise Early("preflight")
    await drain_store_io()
    early = store.list_operations()[-1]
    assert early["status"] == "failed"
    assert get_supervisor()._resources.get("apply:early") is None
    assert admit_action(store.list_operations(), "apply:early")["admit"] is True

    with pytest.raises(AfterDispatch):
        async with exclusive_action("runtime_apply", "apply:late"):
            raise AfterDispatch("proxy returned 404")
    await drain_store_io()
    late = [row for row in store.list_operations() if row["resource_key"] == "apply:late"][-1]
    assert late["status"] == "unknown"
    withheld = admit_action(
        store.list_operations(),
        "apply:late",
        confirm_operation_id="not-this",
        confirm_state="stale",
    )
    assert withheld["code"] == "ACTION_RETRY_WITHHELD"


def test_consuming_a_confirmation_and_reserving_are_one_write(tmp_path, monkeypatch):
    import threading

    from backend import data_store
    from backend.operations.action_recovery import (
        ActionAdmissionError,
        admit_action,
        bind_action_confirmation,
        state_token,
    )
    from backend.operations.supervisor import OperationSupervisor, ResourceBusyError
    from backend.store_io import StoreDurabilityError

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(data_store, "get_store", lambda: store)
    now = time.time()
    store.upsert_operation({
        "operation_id": "build-open",
        "kind": "build",
        "status": "unknown",
        "resource_key": "install/shared",
        "detail": {"effect_started": True},
        "message": "unknown",
        "updated_at": now,
    })
    token = state_token(store.list_operations()[0])
    confirmation = {"confirm_operation_id": "build-open", "confirm_state": token}

    def refuse_replace(self, operations):
        raise StoreDurabilityError("replace did not happen", phase="replace", committed=False)

    original_replace = data_store.DataStore.replace_operations
    monkeypatch.setattr(data_store.DataStore, "replace_operations", refuse_replace)
    bind_action_confirmation(confirmation)
    supervisor = OperationSupervisor()
    with pytest.raises(StoreDurabilityError):
        supervisor.start_operation(
            "build-replacement",
            "build",
            "install/shared",
            detail={"effect_started": False},
        )
    assert store.list_operations()[0]["operation_id"] == "build-open"
    assert admit_action(
        store.list_operations(),
        "install/shared",
        confirm_operation_id="build-open",
        confirm_state=token,
    )["admit"] is True

    monkeypatch.setattr(data_store.DataStore, "replace_operations", original_replace)
    bind_action_confirmation(confirmation)
    results = []
    errors = []

    def attempt(index):
        try:
            bind_action_confirmation(confirmation)
            supervisor.start_operation(
                f"build-replacement-{index}",
                "build",
                "install/shared",
                detail={"effect_started": False},
            )
            results.append(index)
        except (ResourceBusyError, ActionAdmissionError, StoreDurabilityError) as exc:
            errors.append(exc)

    threads = [threading.Thread(target=attempt, args=(index,)) for index in (1, 2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 1
    assert len(errors) == 1
    running = [
        row for row in store.list_operations()
        if row.get("status") == "running"
    ]
    assert len(running) == 1
    withheld = admit_action(
        store.list_operations(),
        "install/shared",
        confirm_operation_id="build-open",
        confirm_state=token,
    )
    assert withheld["admit"] is False


def _ticks(pid: int) -> str:
    with open(f"/proc/{pid}/stat", "r", encoding="utf-8") as handle:
        stat = handle.read()
    fields = stat[stat.rfind(")") + 2 :].split()
    return fields[19]
