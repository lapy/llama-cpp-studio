"""Configuration backup, preview, and crash recovery.

These tests use a temporary config directory. They do not read or write the
repository ``data/`` tree and they do not download models or use a GPU.
"""

import multiprocessing
import os
import signal
import time

import pytest

from backend.config_backup import (
    apply_backup,
    export_backup,
    preview_backup,
    reconcile_config_restore,
)
from backend.data_store import DataStore, set_write_checkpoint
from backend.tests.test_api_routes import _install_temp_store


@pytest.fixture(autouse=True)
def _reset_write_checkpoint():
    set_write_checkpoint(None)
    yield
    set_write_checkpoint(None)


def _seed(store: DataStore) -> None:
    store.update_settings({
        "huggingface_token": "hf_KEEPTOKEN",
        "public_inference_url": "http://old.example",
        "proxy_port": 2000,
    })
    store.add_model({
        "id": "org--model",
        "huggingface_id": "org/model",
        "format": "gguf",
        "local_path": "/models/org/model.gguf",
        "config": {
            "engine": "llama_cpp",
            "engines": {
                "llama_cpp": {
                    "ctx_size": 4096,
                    "executable_path": "/usr/bin/llama-server",
                }
            },
        },
    })
    store.add_config_template({
        "id": "template-1",
        "name": "Local",
        "description": "",
        "include_routing": False,
        "engines_scope": "all",
        "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {"ctx_size": 1024}}},
    })
    store.set_llama_swap_routing({"profiles": {"day": {"note": "local"}}, "selectors": {}})
    store.upsert_operation({
        "operation_id": "op-keep",
        "kind": "build",
        "status": "unknown",
        "resource_key": "engine:llama_cpp",
        "updated_at": time.time(),
        "detail": {},
    })


def _backup_with_changes(store: DataStore) -> dict:
    document = export_backup(store)
    document["preferences"]["public_inference_url"] = "http://new.example"
    document["models"][0]["config"]["engines"]["llama_cpp"]["ctx_size"] = 8192
    document["templates"].append({
        "id": "template-2",
        "name": "Imported",
        "description": "",
        "include_routing": False,
        "engines_scope": "all",
        "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {"ctx_size": 2048}}},
    })
    document["routing"]["profiles"]["night"] = {"note": "imported"}
    return document


def _replace_decisions() -> dict:
    return {
        "preferences": {"public_inference_url": "replace"},
        "models": {"huggingface:org/model": "replace"},
        "templates": {"template-2": "add"},
        "routing": {"profile:night": "add"},
    }


def test_export_omits_credentials_paths_and_runtime_state(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    _seed(store)
    document = export_backup(store)
    text = str(document)
    assert document["schema_version"] == 1
    assert document["kind"] == "llama-cpp-studio-config-backup"
    assert document["application_version"] == "1.0.0"
    assert "credentials" in document["limits"]["excludes"]
    assert "hf_KEEPTOKEN" not in text
    assert "/models/org/model.gguf" not in text
    assert "/usr/bin/llama-server" not in text
    assert "op-keep" not in text
    assert document["models"][0]["ref"] == "huggingface:org/model"
    assert document["models"][0]["config"]["engines"]["llama_cpp"]["ctx_size"] == 4096
    assert "executable_path" not in document["models"][0]["config"]["engines"]["llama_cpp"]


def test_round_trip_keeps_credentials_and_does_not_launch(tmp_path, monkeypatch):
    store = DataStore(config_dir=str(tmp_path / "config"))
    _seed(store)
    calls = []
    monkeypatch.setattr(
        "backend.proxy.llama_swap.manager.get_llama_swap_manager",
        lambda: calls.append("launch") or None,
    )
    document = _backup_with_changes(store)
    decisions = _replace_decisions()
    preview = preview_backup(store, document, decisions=decisions)
    assert preview["applicable"] is True
    assert "does not start, stop, or publish" in preview["notice"]
    before = store.document_revision("settings.yaml")
    preview_backup(store, document, decisions=decisions)
    assert store.document_revision("settings.yaml") == before
    result = apply_backup(store, document, plan_id=preview["plan_id"], decisions=decisions)
    assert result["outcome"] == "completed"
    assert calls == []
    fresh = DataStore(config_dir=str(tmp_path / "config"))
    settings = fresh.get_settings()
    assert settings["huggingface_token"] == "hf_KEEPTOKEN"
    assert settings["public_inference_url"] == "http://new.example"
    model = fresh.get_model("org--model")
    assert model["local_path"] == "/models/org/model.gguf"
    assert model["config"]["engines"]["llama_cpp"]["ctx_size"] == 8192
    assert {item["id"] for item in fresh.list_config_templates()} == {"template-1", "template-2"}
    assert fresh.get_llama_swap_routing()["profiles"]["day"] == {"note": "local"}
    assert fresh.get_llama_swap_routing()["profiles"]["night"] == {"note": "imported"}
    assert fresh.list_operations()[0]["operation_id"] == "op-keep"
    assert not (tmp_path / "config" / "config_restore.yaml").exists()


def test_default_is_keep_and_a_stale_preview_does_not_overwrite(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    _seed(store)
    document = _backup_with_changes(store)
    kept = preview_backup(store, document)
    apply_backup(store, document, plan_id=kept["plan_id"])
    assert store.get_settings()["public_inference_url"] == "http://old.example"
    decisions = _replace_decisions()
    preview = preview_backup(store, document, decisions=decisions)
    store.update_settings({"public_inference_url": "http://concurrent.example"})
    with pytest.raises(Exception) as caught:
        apply_backup(store, document, plan_id=preview["plan_id"], decisions=decisions)
    assert caught.value.code == "BACKUP_STALE"
    fresh = DataStore(config_dir=str(tmp_path / "config"))
    assert fresh.get_settings()["public_inference_url"] == "http://concurrent.example"
    assert fresh.get_settings()["huggingface_token"] == "hf_KEEPTOKEN"
    assert fresh.get_model("org--model")["config"]["engines"]["llama_cpp"]["ctx_size"] == 4096


def test_unresolved_models_are_not_created_until_mapped(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    _seed(store)
    document = export_backup(store)
    document["models"] = [{
        "ref": "huggingface:org/missing",
        "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {"ctx_size": 111}}},
    }]
    preview = preview_backup(store, document)
    assert preview["applicable"] is False
    assert preview["plan_id"] is None
    missing = next(item for item in preview["items"] if item["id"] == "huggingface:org/missing")
    assert missing["action"] == "unresolved"
    with pytest.raises(Exception) as caught:
        apply_backup(store, document, plan_id="unused")
    assert caught.value.code == "BACKUP_UNRESOLVED"
    assert store.get_model("org/missing") is None
    mapped = preview_backup(
        store,
        document,
        mapping={"huggingface:org/missing": "org--model"},
        decisions={"models": {"huggingface:org/missing": "replace"}},
    )
    apply_backup(
        store,
        document,
        plan_id=mapped["plan_id"],
        mapping={"huggingface:org/missing": "org--model"},
        decisions={"models": {"huggingface:org/missing": "replace"}},
    )
    assert len(store.list_models()) == 1
    assert store.get_model("org--model")["config"]["engines"]["llama_cpp"]["ctx_size"] == 111


def test_rejected_backups_do_not_mutate(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    _seed(store)
    before = store.document_revision("settings.yaml")
    rejected = [
        {},
        {"schema_version": 2, "kind": "llama-cpp-studio-config-backup"},
        {
            "schema_version": 1,
            "kind": "llama-cpp-studio-config-backup",
            "preferences": {"huggingface_token": "hf_PLANTED"},
            "models": [],
            "templates": [],
            "routing": {"profiles": {}, "selectors": {}},
        },
        {
            "schema_version": 1,
            "kind": "llama-cpp-studio-config-backup",
            "preferences": {},
            "models": [{
                "ref": "huggingface:org/model",
                "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {"cmd": "/bin/sh"}}},
            }],
            "templates": [],
            "routing": {"profiles": {}, "selectors": {}},
        },
        {
            "schema_version": 1,
            "kind": "llama-cpp-studio-config-backup",
            "preferences": {},
            "models": [{"ref": "../secret", "config": {}}],
            "templates": [],
            "routing": {"profiles": {}, "selectors": {}},
        },
    ]
    oversized = export_backup(store)
    oversized["templates"].append({
        "id": "template-big",
        "name": "Big",
        "description": "x" * 1_100_000,
        "config": {"engine": "llama_cpp", "engines": {}},
    })
    rejected.append(oversized)
    for document in rejected:
        with pytest.raises(Exception):
            preview_backup(store, document)
    assert store.document_revision("settings.yaml") == before
    assert store.get_settings()["huggingface_token"] == "hf_KEEPTOKEN"


def _pause_apply(config_dir, backup, plan_id, decisions, checkpoint, connection):
    import signal as child_signal

    from backend.config_backup import apply_backup as child_apply
    from backend.data_store import DataStore as ChildStore
    from backend.data_store import set_write_checkpoint as child_checkpoint

    def hook(name, _path):
        if name == checkpoint:
            connection.send("ready")
            connection.close()
            while True:
                child_signal.pause()

    child_checkpoint(hook)
    child_apply(
        ChildStore(config_dir=config_dir),
        backup,
        plan_id=plan_id,
        decisions=decisions,
    )


def _kill_apply(config_dir, backup, plan_id, decisions, checkpoint):
    ctx = multiprocessing.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_pause_apply,
        args=(config_dir, backup, plan_id, decisions, checkpoint, writer),
    )
    proc.start()
    writer.close()
    if not reader.poll(8):
        os.kill(proc.pid, signal.SIGKILL)
        proc.join(5)
        pytest.fail(f"restore did not reach {checkpoint}")
    assert reader.recv() == "ready"
    os.kill(proc.pid, signal.SIGKILL)
    proc.join(5)
    assert not proc.is_alive()


@pytest.mark.parametrize(
    "checkpoint,expect",
    [
        ("restore_prepared", "old"),
        ("restore_replaced:settings.yaml", "old"),
        ("restore_replaced:models.yaml", "new"),
    ],
)
def test_interrupted_restore_is_pre_import_or_complete(tmp_path, checkpoint, expect):
    config_dir = str(tmp_path / "config")
    store = DataStore(config_dir=config_dir)
    _seed(store)
    document = _backup_with_changes(store)
    decisions = _replace_decisions()
    preview = preview_backup(store, document, decisions=decisions)
    _kill_apply(config_dir, document, preview["plan_id"], decisions, checkpoint)
    outcome = reconcile_config_restore(DataStore(config_dir=config_dir))
    fresh = DataStore(config_dir=config_dir)
    settings = fresh.get_settings()
    ctx = fresh.get_model("org--model")["config"]["engines"]["llama_cpp"]["ctx_size"]
    assert settings["huggingface_token"] == "hf_KEEPTOKEN"
    if expect == "old":
        assert outcome["outcome"] == "pre_import"
        assert settings["public_inference_url"] == "http://old.example"
        assert ctx == 4096
        assert "template-2" not in {item["id"] for item in fresh.list_config_templates()}
    else:
        assert outcome["outcome"] == "completed"
        assert settings["public_inference_url"] == "http://new.example"
        assert ctx == 8192
        assert "template-2" in {item["id"] for item in fresh.list_config_templates()}
    assert not (tmp_path / "config" / "config_restore.yaml").exists()
    again = reconcile_config_restore(fresh)
    assert again["outcome"] == "idle"


def test_journal_directory_sync_failure_does_not_replace_configuration(tmp_path):
    config_dir = str(tmp_path / "config")
    store = DataStore(config_dir=config_dir)
    _seed(store)
    document = _backup_with_changes(store)
    decisions = _replace_decisions()
    preview = preview_backup(store, document, decisions=decisions)
    seen = {"n": 0}

    def hook(name, path):
        if name == "before_directory_sync" and str(path).endswith("config_restore.yaml"):
            seen["n"] += 1
            if seen["n"] == 1:
                raise OSError("directory sync failed")

    set_write_checkpoint(hook)
    with pytest.raises(Exception) as caught:
        apply_backup(store, document, plan_id=preview["plan_id"], decisions=decisions)
    assert caught.value.code == "BACKUP_INCOMPLETE"
    fresh = DataStore(config_dir=config_dir)
    assert fresh.get_settings()["public_inference_url"] == "http://old.example"
    assert fresh.get_settings()["huggingface_token"] == "hf_KEEPTOKEN"
    assert fresh.get_model("org--model")["config"]["engines"]["llama_cpp"]["ctx_size"] == 4096


def test_a_corrupt_journal_stops_recovery_without_resetting(tmp_path):
    config_dir = tmp_path / "config"
    store = DataStore(config_dir=str(config_dir))
    _seed(store)
    journal = config_dir / "config_restore.yaml"
    journal.write_text("42\n", encoding="utf-8")
    outcome = reconcile_config_restore(store)
    assert outcome["outcome"] == "unknown"
    assert outcome["code"] == "RESTORE_JOURNAL_CORRUPT"
    assert journal.exists()
    assert store.get_settings()["public_inference_url"] == "http://old.example"
    assert store.get_settings()["huggingface_token"] == "hf_KEEPTOKEN"


def test_a_second_interruption_during_recovery_is_safe(tmp_path):
    config_dir = str(tmp_path / "config")
    store = DataStore(config_dir=config_dir)
    _seed(store)
    snapshot = {
        "settings.yaml": store._read_yaml("settings.yaml"),
        "model_config_templates.yaml": store._read_yaml("model_config_templates.yaml"),
        "llama_swap_routing.yaml": store._read_yaml("llama_swap_routing.yaml"),
        "models.yaml": store._read_yaml("models.yaml"),
    }
    store.update_settings({"public_inference_url": "http://partial.example"})
    intended = {
        name: dict(document) if isinstance(document, dict) else document
        for name, document in snapshot.items()
    }
    intended["settings.yaml"] = dict(snapshot["settings.yaml"])
    intended["settings.yaml"]["public_inference_url"] = "http://finished.example"
    store.write_document(
        "config_restore.yaml",
        {
            "schema_version": 1,
            "phase": "applying",
            "snapshot": snapshot,
            "intended": intended,
            "applied": [],
            "pending": "settings.yaml",
        },
        require_directory_sync=True,
    )
    seen = {"n": 0}

    def hook(name, _path):
        if name == "restore_rollback:settings.yaml":
            seen["n"] += 1
            if seen["n"] == 1:
                raise RuntimeError("recovery interrupted")

    set_write_checkpoint(hook)
    with pytest.raises(RuntimeError):
        reconcile_config_restore(store)
    assert store.get_settings()["public_inference_url"] == "http://partial.example"
    set_write_checkpoint(None)
    outcome = reconcile_config_restore(DataStore(config_dir=config_dir))
    assert outcome["outcome"] == "pre_import"
    fresh = DataStore(config_dir=config_dir)
    assert fresh.get_settings()["public_inference_url"] == "http://old.example"
    assert fresh.get_settings()["huggingface_token"] == "hf_KEEPTOKEN"
    assert not (tmp_path / "config" / "config_restore.yaml").exists()


def test_configuration_writes_wait_until_recovery_releases(tmp_path):
    import threading

    from backend.data_store import hold_configuration_writes, release_configuration_writes

    store = DataStore(config_dir=str(tmp_path / "config"))
    store.update_settings({"public_inference_url": "http://old.example"})
    hold_configuration_writes()
    started = threading.Event()
    finished = threading.Event()

    def writer():
        started.set()
        store.update_settings({"public_inference_url": "http://after.example"})
        finished.set()

    thread = threading.Thread(target=writer)
    thread.start()
    try:
        assert started.wait(2)
        assert finished.wait(0.3) is False
        assert store.get_settings()["public_inference_url"] == "http://old.example"
        release_configuration_writes()
        assert finished.wait(2)
    finally:
        release_configuration_writes()
        thread.join(2)
    assert store.get_settings()["public_inference_url"] == "http://after.example"


def test_backup_routes_preview_and_apply(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed(store)
    exported = client.get("/api/config-backup")
    assert exported.status_code == 200
    assert "studio-config-backup.json" in exported.headers["content-disposition"]
    assert "hf_KEEPTOKEN" not in exported.text
    document = exported.json()
    document["preferences"]["public_inference_url"] = "http://route.example"
    decisions = {"preferences": {"public_inference_url": "replace"}}
    preview = client.post("/api/config-backup/preview", json={"backup": document, "decisions": decisions})
    assert preview.status_code == 200
    assert preview.json()["applicable"] is True
    applied = client.post(
        "/api/config-backup/apply",
        json={"backup": document, "decisions": decisions, "plan_id": preview.json()["plan_id"]},
    )
    assert applied.status_code == 200
    assert store.get_settings()["public_inference_url"] == "http://route.example"
    assert store.get_settings()["huggingface_token"] == "hf_KEEPTOKEN"
