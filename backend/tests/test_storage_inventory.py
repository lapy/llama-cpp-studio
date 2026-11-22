"""Storage inventory classification, bounds, caching and refusal guards."""

import os

import pytest

from backend.services import storage_cleanup, storage_inventory


def _write(path, size, content=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content if content is not None else b"x" * size)
    return path


def _seed_storage(tmp_path, monkeypatch):
    """Isolated data root, HF cache, ccache and store for a storage scan."""
    from backend import data_store

    data = tmp_path / "data"
    (data / "config").mkdir(parents=True)
    monkeypatch.setenv("STUDIO_DATA_DIR", str(data))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(data / "hf-cache" / "hub"))
    monkeypatch.setenv("STUDIO_CCACHE_DIR", str(data / "ccache"))
    store = data_store.DataStore(config_dir=str(data / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    storage_inventory.clear_storage_inventory_cache()
    storage_cleanup.clear_storage_cleanup_plans()
    return data, store


def _build_tree(tmp_path, data):
    root = tmp_path
    models = data / "models" / "audio-cpp"
    _write(models / "keep" / "model.gguf", 100)
    _write(models / "orphan-a" / "weights.gguf", 50)
    _write(models / "reference-audio" / "clip.wav", 20)
    _write(models / "community-voices" / "voice.wav", 30)

    hub = data / "hf-cache" / "hub"
    _write(hub / "models--org--keep" / "blobs" / "a", 60)
    _write(hub / "models--org--orphan" / "blobs" / "b", 40)

    install = data / "llama-cpp" / "demo"
    _write(install / "server", 30)
    _write(install / "caches" / "triton-cache-dir" / "kernel.bin", 70)

    _write(data / "cuda" / "cuda-12.8" / "bin" / "nvcc", 11)
    _write(data / "build-workspaces" / "workspace-1" / "CMakeCache.txt", 5)
    _write(data / "ccache" / "0" / "entry", 7)
    _write(data / "temp" / "cuda_installers" / "cuda.run", 9)
    _write(data / "config" / "proxy-accounting.sqlite3", 13)


def _add_models(store, data):
    store.add_model(
        {
            "id": "demo-keep",
            "name": "Keep",
            "format": "safetensors",
            "local_path": str(data / "models" / "audio-cpp" / "keep"),
            "config": {},
        }
    )
    store.add_model(
        {
            "id": "hf-keep",
            "name": "HF Keep",
            "huggingface_id": "org/keep",
            "format": "gguf",
            "config": {},
        }
    )


def _category(payload, category_id):
    return next(item for item in payload["categories"] if item["id"] == category_id)


def _unit(category, label):
    return next(unit for unit in category["units"] if unit["label"] == label)


def test_inventory_is_seven_disjoint_categories_with_expected_bytes(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)

    payload, index = storage_inventory.storage_inventory_index(refresh=True)

    assert payload["schema_version"] == storage_inventory.INVENTORY_SCHEMA_VERSION
    assert payload["filesystem"]["data_root"] == os.path.realpath(data)
    assert payload["scan"]["quality"] == "complete"
    assert [item["id"] for item in payload["categories"]] == list(storage_inventory.CATEGORY_ORDER)
    assert all("path" not in unit for item in payload["categories"] for unit in item["units"])

    models = _category(payload, "models")
    assert _unit(models, "audio-cpp/keep")["blocked"]["code"] == "CATALOG_MODEL"
    assert _unit(models, "audio-cpp/orphan-a")["cleanable"] is True
    assert _unit(models, "audio-cpp/reference-audio")["blocked"]["code"] == "SHARED_DATA"
    assert _unit(models, "audio-cpp/community-voices")["blocked"]["code"] == "SHARED_DATA"
    assert models["bytes"] == 200
    assert models["cleanable_bytes"] == 50
    assert models["manage_path"] == "/models"

    hf = _category(payload, "hf_cache")
    assert _unit(hf, "org/keep")["blocked"]["code"] == "CATALOG_MODEL"
    assert _unit(hf, "org/orphan")["cleanable"] is True
    assert hf["bytes"] == 100
    assert hf["outside_data_root"] is False

    engines = _category(payload, "engine_installs")
    assert engines["bytes"] == 30  # cache bytes are reported by jit_caches
    assert engines["cleanable"] is False
    assert engines["blocked"]["code"] == "REPORT_ONLY"
    assert engines["manage_path"] == "/engines"
    assert _unit(engines, "llama_cpp/demo")["cleanable"] is False
    assert _unit(engines, "llama_cpp/demo")["manage_path"] == "/engines"

    cuda = _category(payload, "cuda_toolkits")
    assert cuda["bytes"] == 11
    assert cuda["blocked"]["code"] == "REPORT_ONLY"
    assert _unit(cuda, "cuda-12.8")["manage_path"] == "/system#hardware"

    jit = _category(payload, "jit_caches")
    jit_labels = {unit["label"] for unit in jit["units"]}
    assert jit_labels == {
        "llama_cpp/demo/caches",
        "build-workspaces/workspace-1",
        "ccache",
        "CUDA installer cache",
    }
    assert jit["bytes"] == 70 + 5 + 7 + 9
    assert jit["cleanable"] is True

    accounting = _category(payload, "accounting_db")
    assert accounting["bytes"] == 13
    assert accounting["blocked"]["code"] == "REPORT_ONLY"
    assert accounting["manage_path"] == "/proxy"

    # No byte belongs to two categories: every unit is measured inside exactly
    # one category root, and engine caches are excluded from engine installs.
    all_units = {
        category_id: list(section["units"].values())
        for category_id, section in index["categories"].items()
    }
    assert all_units["engine_installs"][0]["bytes"] == 30
    assert all_units["jit_caches"][0]["path"].startswith(
        all_units["engine_installs"][0]["path"] + os.sep
    )
    expected = 200 + 100 + 30 + 11 + 91 + 13 + _category(payload, "history_backups")["bytes"]
    assert sum(item["bytes"] for item in payload["categories"]) == expected


def test_history_units_keep_the_newest_default_window(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _add_models(store, data)
    history_dir = data / "config" / "history"
    before = len(list(history_dir.glob("*.yaml"))) if history_dir.is_dir() else 0
    for index in range(55):
        _write(history_dir / f"9{index:019d}-synthetic.yaml", 2)
    total = before + 55

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    category = _category(payload, "history_backups")
    assert category["unit_count"] == total
    cleanable = [unit for unit in category["units"] if unit["cleanable"]]
    retained = [unit for unit in category["units"] if not unit["cleanable"]]
    assert len(cleanable) == max(0, total - storage_inventory.DEFAULT_HISTORY_KEEP)
    assert len(retained) == min(total, storage_inventory.DEFAULT_HISTORY_KEEP)
    assert all(unit["blocked"]["code"] == "RECENT_HISTORY" for unit in retained)


def test_inventory_cache_serves_for_ten_seconds_and_refresh_bypasses_it(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)

    first = storage_inventory.collect_storage_inventory()
    assert first["cached"] is False
    _write(data / "models" / "audio-cpp" / "late" / "weights.gguf", 10)
    cached = storage_inventory.collect_storage_inventory()
    assert cached["cached"] is True
    assert cached["scan"]["cached"] is True
    assert cached["generated_at"] == first["generated_at"]

    fresh = storage_inventory.collect_storage_inventory(refresh=True)
    assert fresh["cached"] is False
    assert fresh["generated_at"] != first["generated_at"]
    assert "audio-cpp/late" in {unit["label"] for unit in _category(fresh, "models")["units"]}


def test_hf_repository_referenced_by_a_local_model_path_is_report_only(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    repo = data / "hf-cache" / "hub" / "models--org--imported"
    blob = _write(repo / "blobs" / "a", 5)
    store.add_model(
        {
            "id": "local-import",
            "name": "Local import",
            "format": "gguf",
            "local_path": str(blob),
            "config": {},
        }
    )

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    unit = _unit(_category(payload, "hf_cache"), "org/imported")
    assert unit["blocked"]["code"] == "CATALOG_MODEL"
    assert unit["cleanable"] is False


def test_hf_cache_outside_the_data_root_is_flagged(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    outside = tmp_path / "outside-hub"
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(outside))
    storage_inventory.clear_storage_inventory_cache()
    _write(outside / "models--org--demo" / "blobs" / "a", 12)

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    category = _category(payload, "hf_cache")
    assert category["outside_data_root"] is True
    assert category["units"][0]["outside_data_root"] is True
    assert category["units"][0]["cleanable"] is True


def test_symlinked_directories_are_not_followed(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    victim = tmp_path / "victim"
    _write(victim / "secret.bin", 4096)
    hub = data / "hf-cache" / "hub"
    hub.mkdir(parents=True)
    link = hub / "models--org--evil"
    try:
        link.symlink_to(victim, target_is_directory=True)
    except OSError:
        pytest.skip("This platform cannot create directory symlinks")

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    category = _category(payload, "hf_cache")
    assert category["units"] == []
    assert category["bytes"] == 0


def test_scan_caps_mark_the_result_partial(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    for index in range(30):
        _write(data / "models" / "audio-cpp" / f"orphan-{index}" / "weights.gguf", 1)
    monkeypatch.setattr(storage_inventory, "MAX_SCAN_ENTRIES", 10)
    monkeypatch.setattr(storage_inventory, "MAX_UNITS_PER_CATEGORY", 3)
    storage_inventory.clear_storage_inventory_cache()

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    assert payload["scan"]["truncated"] is True
    assert payload["scan"]["quality"] == "partial"
    assert payload["scan"]["notes"]
    models = _category(payload, "models")
    assert models["truncated"] is True
    assert models["unit_count"] <= 3


def test_unreadable_catalog_blocks_model_cleanup(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    monkeypatch.setattr(storage_inventory, "_store_or_none", lambda: None)
    storage_inventory.clear_storage_inventory_cache()

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    models = _category(payload, "models")
    assert models["blocked"]["code"] == "STORE_UNAVAILABLE"
    assert models["cleanable"] is False
    assert all(unit["cleanable"] is False for unit in models["units"])
    assert payload["scan"]["quality"] == "partial"


def test_running_model_and_retained_launch_are_refused(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    monkeypatch.setattr("backend.proxy.manifests.deletion_block_reason", lambda path: "held")
    monkeypatch.setattr(storage_inventory, "running_model_ids", lambda path: ["demo-model"])
    storage_inventory.clear_storage_inventory_cache()

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    orphan = _unit(_category(payload, "models"), "audio-cpp/orphan-a")
    assert orphan["blocked"]["code"] == "RUNNING_MODEL"
    assert orphan["cleanable"] is False
    assert "demo-model" in orphan["blocked"]["message"]

    monkeypatch.setattr(storage_inventory, "running_model_ids", lambda path: [])
    storage_inventory.clear_storage_inventory_cache()
    payload = storage_inventory.collect_storage_inventory(refresh=True)
    orphan = _unit(_category(payload, "models"), "audio-cpp/orphan-a")
    assert orphan["blocked"]["code"] == "RETAINED_LAUNCH_REFERENCE"


def test_active_download_and_build_refuse_their_units(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    install = os.path.realpath(data / "llama-cpp" / "demo")
    store.upsert_operation(
        {
            "operation_id": "download-1",
            "kind": "download",
            "status": "running",
            "resource_key": "hf:org/orphan",
            "detail": {"huggingface_id": "org/orphan"},
        }
    )
    store.upsert_operation(
        {
            "operation_id": "build-1",
            "kind": "build",
            "status": "running",
            "resource_key": f"engine:llama_cpp:{install}",
            "detail": {"engine": "llama_cpp"},
        }
    )
    storage_inventory.clear_storage_inventory_cache()

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    assert (
        _unit(_category(payload, "hf_cache"), "org/orphan")["blocked"]["code"]
        == "DOWNLOAD_IN_PROGRESS"
    )
    assert (
        _unit(_category(payload, "jit_caches"), "llama_cpp/demo/caches")["blocked"]["code"]
        == "BUILD_IN_PROGRESS"
    )
    # Report-only categories keep their own report-only refusal.
    assert (
        _unit(_category(payload, "engine_installs"), "llama_cpp/demo")["blocked"]["code"]
        == "REPORT_ONLY"
    )


def test_pending_configuration_restore_blocks_history_pruning(tmp_path, monkeypatch):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _write(data / "config" / "history" / "100-keep.yaml", 1)
    _write(data / "config" / "config_restore.yaml", 10)
    storage_inventory.clear_storage_inventory_cache()

    payload = storage_inventory.collect_storage_inventory(refresh=True)
    history = _category(payload, "history_backups")
    assert history["blocked"]["code"] == "CONFIG_RECOVERY_IN_PROGRESS"
    assert history["cleanable"] is False
    assert all(unit["cleanable"] is False for unit in history["units"])
