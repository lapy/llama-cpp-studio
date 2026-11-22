"""Guarded cleanup preview/apply plans for reclaimable storage."""

import os
import time

import pytest

from backend.services import storage_cleanup as cleanup
from backend.services import storage_inventory as inventory
from backend.tests.test_storage_inventory import (
    _add_models,
    _build_tree,
    _seed_storage,
    _write,
)
from backend.utils.fs_ops import FilesystemRefusal


def _preview(category, unit_ids=None, keep_history=None):
    return cleanup.preview_cleanup(category, unit_ids, keep_history)


def _codes(refused):
    return {item["code"] for item in refused}


def test_preview_and_apply_reclaim_only_the_orphan_model_directory(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    orphan = data / "models" / "audio-cpp" / "orphan-a"
    keep = data / "models" / "audio-cpp" / "keep"

    preview = _preview("models")
    assert [unit["label"] for unit in preview["units"]] == ["audio-cpp/orphan-a"]
    assert preview["reclaimable_bytes"] == 50
    assert preview["confirmation_phrase"] == "DELETE 1 MODEL DIRECTORIES"
    assert {"CATALOG_MODEL", "SHARED_DATA"} <= _codes(preview["refused"])
    assert preview["keep_history"] is None
    assert len(preview["plan_id"]) == 64

    result = cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert result["outcome"] == "completed"
    assert result["freed_bytes"] == 50
    assert result["removed"][0]["label"] == "audio-cpp/orphan-a"
    assert result["failed"] == []
    assert not orphan.exists()
    assert keep.exists()
    # A plan is one-shot: applying it twice is a stale plan.
    with pytest.raises(cleanup.StorageCleanupError) as again:
        cleanup.apply_cleanup(preview["plan_id"], confirm=True)
    assert again.value.code == "PLAN_STALE"


def test_apply_requires_confirmation_before_doing_anything(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    orphan = data / "models" / "audio-cpp" / "orphan-a"
    preview = _preview("models")

    with pytest.raises(cleanup.StorageCleanupError) as refusal:
        cleanup.apply_cleanup(preview["plan_id"], confirm=False)

    assert refusal.value.code == "CONFIRMATION_REQUIRED"
    assert refusal.value.status_code == 400
    assert orphan.exists()


def test_inventory_change_between_preview_and_apply_is_stale(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    preview = _preview("models")
    _write(data / "models" / "audio-cpp" / "orphan-a" / "extra.gguf", 3)

    with pytest.raises(cleanup.StorageCleanupError) as stale:
        cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert stale.value.code == "PLAN_STALE"


def test_expired_plan_is_stale(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    preview = _preview("models")
    cleanup._PLANS[preview["plan_id"]]["expires_monotonic"] = time.monotonic() - 1

    with pytest.raises(cleanup.StorageCleanupError) as stale:
        cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert stale.value.code == "PLAN_STALE"


def test_scope_cap_refuses_too_many_units(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    _write(data / "models" / "audio-cpp" / "orphan-b" / "weights.gguf", 5)
    inventory.clear_storage_inventory_cache()
    inventory_payload, _ = inventory.storage_inventory_index(refresh=True)
    ids = [
        unit["id"]
        for unit in next(
            item for item in inventory_payload["categories"] if item["id"] == "models"
        )["units"]
        if unit["cleanable"]
    ]
    assert len(ids) == 2
    monkeypatch.setattr(cleanup, "MAX_CLEANUP_UNITS", 1)

    with pytest.raises(cleanup.StorageCleanupError) as capped:
        _preview("models", ids)

    assert capped.value.code == "CLEANUP_SCOPE_TOO_LARGE"
    assert capped.value.status_code == 409


def test_apply_rechecks_the_scope_cap(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    _write(data / "models" / "audio-cpp" / "orphan-b" / "weights.gguf", 5)
    inventory.clear_storage_inventory_cache()
    preview = _preview("models")
    assert len(preview["units"]) == 2
    monkeypatch.setattr(cleanup, "MAX_CLEANUP_UNITS", 1)

    with pytest.raises(cleanup.StorageCleanupError) as capped:
        cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert capped.value.code == "CLEANUP_SCOPE_TOO_LARGE"


def test_concurrent_apply_is_refused(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    preview = _preview("models")
    assert cleanup._APPLY_LOCK.acquire(blocking=False)
    try:
        with pytest.raises(cleanup.StorageCleanupError) as busy:
            cleanup.apply_cleanup(preview["plan_id"], confirm=True)
    finally:
        cleanup._APPLY_LOCK.release()

    assert busy.value.code == "CLEANUP_IN_FLIGHT"
    assert (data / "models" / "audio-cpp" / "orphan-a").exists()


def test_unknown_unit_ids_are_refused_and_never_treated_as_paths(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    outside = tmp_path / "outside"
    _write(outside / "keep.bin", 10)

    preview = _preview("models", ["../../etc/passwd", str(outside)])

    assert preview["units"] == []
    assert _codes(preview["refused"]) == {"UNKNOWN_UNIT"}
    with pytest.raises(cleanup.StorageCleanupError) as nothing:
        cleanup.apply_cleanup(preview["plan_id"], confirm=True)
    assert nothing.value.code == "NOTHING_TO_CLEAN"
    assert outside.exists()


def test_report_only_category_cannot_be_cleaned(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    with pytest.raises(cleanup.StorageCleanupError) as refusal:
        _preview("engine_installs")
    assert refusal.value.code == "CATEGORY_NOT_CLEANABLE"
    assert refusal.value.status_code == 400


def test_history_pruning_keeps_the_requested_number_of_revisions(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _add_models(store, data)
    history_dir = data / "config" / "history"
    for index in range(20):
        _write(history_dir / f"9{index:019d}-synthetic.yaml", 4)
    inventory.clear_storage_inventory_cache()
    total = len(list(history_dir.glob("*.yaml")))
    assert total >= 20

    preview = _preview("history_backups", keep_history=5)
    assert preview["keep_history"] == 5
    assert len(preview["units"]) == total - 5
    assert preview["confirmation_phrase"] == f"PRUNE {total - 5} CONFIGURATION REVISIONS"
    expected_freed = preview["reclaimable_bytes"]

    result = cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert result["outcome"] == "completed"
    assert len(list(history_dir.glob("*.yaml"))) == 5
    assert result["freed_bytes"] == expected_freed


def test_partial_apply_reports_removed_and_failed_units(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    _write(data / "models" / "audio-cpp" / "orphan-b" / "weights.gguf", 5)
    inventory.clear_storage_inventory_cache()
    preview = _preview("models")
    assert len(preview["units"]) == 2

    original = cleanup._remove_unit

    def fail_one(unit, category):
        if unit["label"] == "audio-cpp/orphan-b":
            raise OSError("device busy")
        return original(unit, category)

    monkeypatch.setattr(cleanup, "_remove_unit", fail_one)
    result = cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert result["outcome"] == "partial"
    assert result["freed_bytes"] == 50
    assert [item["label"] for item in result["removed"]] == ["audio-cpp/orphan-a"]
    assert [item["label"] for item in result["failed"]] == ["audio-cpp/orphan-b"]
    assert result["failed"][0]["code"] == "CLEANUP_FAILED"
    assert (data / "models" / "audio-cpp" / "orphan-b").exists()


def test_retained_launch_reference_refuses_the_delete(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    orphan = data / "models" / "audio-cpp" / "orphan-a"
    preview = _preview("models")

    def refuse(path, *, retire_references):
        raise FilesystemRefusal(
            "A published launch generation still references this path.",
            code="RETAINED_LAUNCH_REFERENCE",
        )

    monkeypatch.setattr(cleanup, "release_launch_hold", refuse)
    result = cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert result["outcome"] == "refused"
    assert result["freed_bytes"] == 0
    assert result["failed"][0]["code"] == "RETAINED_LAUNCH_REFERENCE"
    assert orphan.exists()


def test_jit_cache_roots_are_deletable(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)

    preview = _preview("jit_caches")
    labels = {unit["label"] for unit in preview["units"]}
    engine_caches = {label for label in labels if label.endswith("/demo/caches")}
    assert len(engine_caches) == 1
    assert labels - engine_caches == {
        "build-workspaces/workspace-1",
        "ccache",
        "CUDA installer cache",
    }

    result = cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert result["outcome"] == "completed"
    assert result["freed_bytes"] == 70 + 5 + 7 + 9
    assert not (data / "ccache").exists()
    assert not (data / "temp" / "cuda_installers").exists()
    assert not (data / "build-workspaces" / "workspace-1").exists()
    assert not (data / "llama-cpp" / "demo" / "caches").exists()
    # Installed engine files stay; only the regenerable cache subtree is gone.
    assert (data / "llama-cpp" / "demo" / "server").exists()


def test_apply_refuses_when_the_rescan_fails(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    orphan = data / "models" / "audio-cpp" / "orphan-a"
    preview = _preview("models")

    monkeypatch.setattr(
        inventory,
        "storage_inventory_index",
        lambda refresh=False: (_ for _ in ()).throw(RuntimeError("scan failed")),
    )
    with pytest.raises(cleanup.StorageCleanupError) as refusal:
        cleanup.apply_cleanup(preview["plan_id"], confirm=True)

    assert refusal.value.code == "STORAGE_SCAN_FAILED"
    assert refusal.value.status_code == 503
    assert orphan.exists()


def test_target_validation_refuses_symlinks_and_escapes(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    models_root = data / "models"
    victim = tmp_path / "victim"
    _write(victim / "file", 1)
    link = models_root / "audio-cpp" / "linked"
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(victim, target_is_directory=True)
    except OSError:
        pytest.skip("This platform cannot create directory symlinks")

    root = os.path.realpath(models_root)
    escaping = cleanup._validate_target({"path": str(link), "root": root})
    assert escaping and "outside" in escaping.lower()
    outside = cleanup._validate_target({"path": str(victim), "root": root})
    assert outside and "outside" in outside.lower()
    root_refusal = cleanup._validate_target({"path": root, "root": root})
    assert root_refusal and "root" in root_refusal.lower()
