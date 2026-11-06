"""One apply confirmation must clear both records of the earlier attempt.

A launch apply is durable twice: a request row in the operation supervisor and
a launch journal in the runtime store. Recovering from an interrupted apply
used to ask for one token per record, so the two gates handed the client their
own tokens in turn and no retry could ever pass both.
"""

from __future__ import annotations

import json

import pytest

from backend.proxy.launch_spec import (
    PORT_PLACEHOLDER,
    CompiledModel,
    EnvSpec,
    LaunchSpec,
    ProxyModelSpec,
    launch_revision,
)

MODEL_ID = "org/model"


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


def _seed_model(store):
    store.add_model(
        {
            "id": MODEL_ID,
            "name": "Test Model",
            "display_name": "Test Model",
            "huggingface_id": MODEL_ID,
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {"engine": "llama_cpp"},
        }
    )


def _compiled(model_id: str = MODEL_ID) -> CompiledModel:
    launch = LaunchSpec(
        model_id=model_id,
        engine_id="llama_cpp",
        engine_install_id="install",
        executable="/bin/llama-server",
        argv=["--model", "/models/model.gguf", "--port", dict(PORT_PLACEHOLDER)],
        cwd=None,
        env=EnvSpec(set={}, unset=[]),
    )
    proxy = ProxyModelSpec(
        model_id=model_id,
        catalog_id=model_id,
        engine_id="llama_cpp",
        aliases=[],
        filters=None,
        use_model_name=None,
    )
    return CompiledModel(
        launch=launch,
        proxy=proxy,
        revision=launch_revision(launch),
        engine_command="engine",
        launcher_command="native-launcher",
    )


def _write_withheld_journal(operations, operation_id: str, model_id: str = MODEL_ID) -> dict:
    operations.mkdir(parents=True, exist_ok=True)
    journal = {
        "operation_id": operation_id,
        "catalog_id": model_id,
        "model_id": model_id,
        "mode": "restart_now",
        "phase": "unknown",
        "state": "loading",
        "was_running": True,
        "desired_revision": "d" * 64,
        "published_revision": "p" * 64,
        "observed_published_revision": "p" * 64,
        "observed_running_revision": None,
        "prior_launch_id": "launch-1",
        "message": "The restart was not replayed.",
        "recovery": {
            "kind": "runtime_apply",
            "evidence": "unproven",
            "status": "unknown",
            "retry": "withheld",
            "replayed": 0,
            "message": "The restart was not replayed.",
        },
    }
    (operations / f"{operation_id}.json").write_text(json.dumps(journal), encoding="utf-8")
    return journal


def test_a_confirmed_journal_settles_instead_of_withholding_again(tmp_path):
    from backend.proxy.manifests import LaunchManifestStore
    from backend.services.model_runtime_apply import withheld_apply

    store = LaunchManifestStore(str(tmp_path / "models"))
    operations = tmp_path / "operations"
    _write_withheld_journal(operations, "a" * 32)

    held = withheld_apply(store, [MODEL_ID])
    assert held["code"] == "ACTION_RETRY_WITHHELD"

    stale = withheld_apply(
        store,
        [MODEL_ID],
        confirm_operation_id=held["operation_id"],
        confirm_state="0" * 64,
    )
    assert stale["code"] == "ACTION_RETRY_WITHHELD"
    assert withheld_apply(store, [MODEL_ID]) is not None

    confirmed = withheld_apply(
        store,
        [MODEL_ID],
        confirm_operation_id=held["operation_id"],
        confirm_state=held["state_token"],
    )
    assert confirmed["code"] == "ACTION_CONFIRMED"

    saved = json.loads((operations / f"{'a' * 32}.json").read_text(encoding="utf-8"))
    assert saved["phase"] == "interrupted"
    assert saved["recovery"]["retry"] == "safe"
    assert saved["confirmation_consumed"] == held["state_token"]
    assert "#confirmed" not in saved["observed_published_revision"]

    # The retry is released, and the same pair cannot be replayed.
    assert withheld_apply(store, [MODEL_ID]) is None
    assert (
        withheld_apply(
            store,
            [MODEL_ID],
            confirm_operation_id=held["operation_id"],
            confirm_state=held["state_token"],
        )
        is None
    )


@pytest.mark.asyncio
async def test_apply_retry_is_not_withheld_by_the_request_row_after_journal_confirmation(
    monkeypatch, tmp_path
):
    import backend.services.model_runtime_apply as apply_module
    from fastapi import HTTPException

    from backend.proxy.manifests import LaunchManifestStore
    from backend.routes.models import RuntimeApplyBody, apply_model_runtime

    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    compiled = _compiled()
    monkeypatch.setattr(apply_module, "compile_model_runtime", lambda model: compiled)
    runtime_store = LaunchManifestStore(str(tmp_path / "models"))
    monkeypatch.setattr(apply_module, "LaunchManifestStore", lambda *args, **kwargs: runtime_store)

    operations = tmp_path / "operations"
    _write_withheld_journal(operations, "b" * 32)

    stale_row = {
        "operation_id": "older-plan:org/model:restart_now",
        "kind": "runtime_apply",
        "status": "unknown",
        "resource_key": f"runtime-apply:{MODEL_ID}",
        "resumable": False,
        "detail": {"model_id": MODEL_ID, "mode": "restart_now", "effect_started": True},
        "message": "Whether this action finished could not be established.",
        "updated_at": 1.0,
    }
    store.upsert_operation(stale_row)

    body = RuntimeApplyBody(mode="next_start", idempotency_key="plan-2:org/model:next_start")

    with pytest.raises(HTTPException) as first:
        await apply_model_runtime(MODEL_ID, body)
    assert first.value.status_code == 409
    detail = first.value.detail
    assert detail["code"] == "ACTION_RETRY_WITHHELD"
    assert detail["operation_id"] == "b" * 32

    retry = await apply_model_runtime(
        MODEL_ID,
        RuntimeApplyBody(
            mode="next_start",
            idempotency_key=body.idempotency_key,
            confirm_operation_id=detail["operation_id"],
            confirm_state=detail["state_token"],
        ),
    )
    assert retry["status"] == "succeeded"

    rows = {row["operation_id"]: row for row in store.list_operations()}
    assert rows[stale_row["operation_id"]]["detail"]["confirmation_consumed"] is True
    saved = json.loads((operations / f"{'b' * 32}.json").read_text(encoding="utf-8"))
    assert saved["phase"] == "interrupted"

    # The next apply is not withheld again by either record.
    again = await apply_model_runtime(
        MODEL_ID,
        RuntimeApplyBody(mode="next_start", idempotency_key="plan-3:org/model:next_start"),
    )
    assert again["status"] == "succeeded"
