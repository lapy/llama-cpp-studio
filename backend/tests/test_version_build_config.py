"""Per-build frozen CMake config: listing, PUT, and sync/retry consumption."""

from backend.llama_build_options import default_build_settings, stored_config_to_settings
from backend.routes.llama_versions import _build_config_for_source_sync


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


def _llama_source_row(**overrides):
    row = {
        "version": "source-main",
        "type": "source",
        "install_type": "source",
        "binary_path": "/bin/llama-server",
        "source_ref": "main",
        "source_ref_type": "branch",
        "source_branch": "main",
        "source_repo": "https://example.test/llama.cpp.git",
        "build_config": {"enable_cuda": False, "build_type": "Release"},
        "repository_source": "llama.cpp",
        "build_status": "ready",
    }
    row.update(overrides)
    return row


def test_stored_config_to_settings_defaults_and_passthrough():
    defaults = default_build_settings()
    assert stored_config_to_settings(None) == defaults
    assert stored_config_to_settings("nope") == defaults

    ui_shaped = stored_config_to_settings({"cuda": True, "vulkan": True})
    assert ui_shaped["cuda"] is True
    assert ui_shaped["vulkan"] is True
    assert ui_shaped["build_type"] == defaults["build_type"]

    mixed = stored_config_to_settings(
        {
            "enable_cuda": True,
            "custom_cmake_args": "-DFOO=ON",
            "cflags": "-O2",
        }
    )
    assert mixed["cuda"] is True
    assert mixed["custom_cmake_args"] == "-DFOO=ON"
    assert mixed["cflags"] == "-O2"


def test_source_sync_prefers_frozen_version_config(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.replace_engine_build_settings(
        "llama_cpp", {"cuda": False, "vulkan": False, "build_type": "Release"}
    )
    cfg = _build_config_for_source_sync(
        "llama_cpp",
        {"build_config": {"enable_cuda": True, "enable_vulkan": True, "build_type": "Debug"}},
        store,
    )
    assert cfg.enable_cuda is True
    assert cfg.enable_vulkan is True
    assert cfg.build_type == "Debug"


def test_source_sync_falls_back_to_global_when_unfrozen(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.replace_engine_build_settings(
        "llama_cpp", {"cuda": True, "build_type": "RelWithDebInfo"}
    )
    cfg = _build_config_for_source_sync("llama_cpp", {}, store)
    assert cfg.enable_cuda is True
    assert cfg.build_type == "RelWithDebInfo"


def test_source_sync_ik_forces_examples(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    cfg = _build_config_for_source_sync(
        "ik_llama",
        {"build_config": {"cuda": True, "build_examples": False}},
        store,
    )
    assert cfg.enable_cuda is True
    assert cfg.build_examples is True


def test_list_exposes_cmake_editable_and_ui_shaped_config(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version("llama_cpp", _llama_source_row())
    store.add_engine_version(
        "ik_llama",
        _llama_source_row(
            version="ik-main",
            repository_source="ik_llama.cpp",
            build_config={"enable_cuda": True, "build_type": "Release"},
        ),
    )
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "audio-main",
            "type": "source",
            "install_type": "source",
            "source_ref": "main",
            "build_config": {"backend": "cpu", "cuda": False},
            "repository_source": "audio.cpp",
        },
    )
    store.add_engine_version(
        "lmdeploy",
        {
            "version": "pip-latest",
            "type": "pip",
            "install_type": "pip",
            "venv_path": str(tmp_path / "venv"),
        },
    )
    store.add_engine_version(
        "1cat_vllm",
        {
            "version": "v1",
            "type": "release",
            "install_type": "release",
            "venv_path": str(tmp_path / "oc"),
        },
    )

    rows = {item["id"]: item for item in client.get("/api/llama-versions").json()}
    llama = rows["llama_cpp:source-main"]
    assert llama["cmake_editable"] is True
    assert llama["build_config"]["cuda"] is False
    assert "enable_cuda" not in llama["build_config"]

    ik = rows["ik_llama:ik-main"]
    assert ik["cmake_editable"] is True
    assert ik["build_config"]["cuda"] is True
    assert ik["build_config"]["build_examples"] is True

    assert rows["audio_cpp:audio-main"]["cmake_editable"] is True
    assert rows["lmdeploy:pip-latest"]["cmake_editable"] is False
    assert rows["1cat_vllm:v1"]["cmake_editable"] is False


def test_list_marks_orphans_not_cmake_editable(client, monkeypatch, tmp_path):
    from backend.routes import llama_versions as llama_routes

    store = _install_temp_store(monkeypatch, tmp_path)
    llama_root = tmp_path / "llama-cpp"
    (llama_root / "disk-only").mkdir(parents=True)
    llama_routes.llama_manager.llama_dir = str(llama_root)
    monkeypatch.setattr(
        "backend.engine_version_lifecycle.discover_engine_install_roots",
        lambda: {
            "llama_cpp": str(llama_root),
            "ik_llama": str(llama_root),
            "audio_cpp": str(tmp_path / "audio-builds"),
            "lmdeploy": str(tmp_path / "lm"),
            "1cat_vllm": str(tmp_path / "oc"),
        },
    )
    rows = client.get("/api/llama-versions").json()
    orphan = next(item for item in rows if item["version"] == "disk-only")
    assert orphan["orphan"] is True
    assert orphan["cmake_editable"] is False


def test_put_build_config_validation(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version("llama_cpp", _llama_source_row())

    missing_id = client.put(
        "/api/llama-versions/versions/build-config",
        json={"build_config": {"cuda": True}},
    )
    assert missing_id.status_code == 400

    missing_config = client.put(
        "/api/llama-versions/versions/build-config",
        json={"version_id": "llama_cpp:source-main"},
    )
    assert missing_config.status_code == 400

    not_object = client.put(
        "/api/llama-versions/versions/build-config",
        json={"version_id": "llama_cpp:source-main", "build_config": "cuda"},
    )
    assert not_object.status_code == 400

    missing_version = client.put(
        "/api/llama-versions/versions/build-config",
        json={"version_id": "llama_cpp:nope", "build_config": {"cuda": True}},
    )
    assert missing_version.status_code == 404


def test_put_ik_llama_build_config_preserves_repo_and_examples(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version(
        "ik_llama",
        _llama_source_row(
            version="ik-main",
            repository_source="ik_llama.cpp",
            build_config={
                "enable_cuda": False,
                "build_examples": True,
                "repository_source": "ik_llama.cpp",
            },
        ),
    )
    r = client.put(
        "/api/llama-versions/versions/build-config",
        json={
            "version_id": "ik_llama:ik-main",
            "build_config": {"cuda": True, "build_examples": False, "build_type": "Debug"},
        },
    )
    assert r.status_code == 200
    stored = store.get_engine_versions("ik_llama")[0]["build_config"]
    assert stored["enable_cuda"] is True
    assert stored["build_examples"] is True
    assert stored["repository_source"] == "ik_llama.cpp"
    assert stored["build_type"] == "Debug"


def test_put_does_not_mutate_global_build_settings(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version("llama_cpp", _llama_source_row())
    store.replace_engine_build_settings(
        "llama_cpp", {"cuda": False, "vulkan": False, "build_type": "Release"}
    )
    r = client.put(
        "/api/llama-versions/versions/build-config",
        json={
            "version_id": "llama_cpp:source-main",
            "build_config": {"cuda": True, "vulkan": True, "build_type": "Debug"},
        },
    )
    assert r.status_code == 200
    global_settings = store.get_engine_build_settings("llama_cpp")
    assert global_settings.get("cuda") is False
    assert global_settings.get("vulkan") is False
    assert global_settings.get("build_type") == "Release"


def test_sync_uses_put_frozen_config_not_global(client, monkeypatch, tmp_path):
    from backend.routes import llama_versions as llama_routes

    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version("llama_cpp", _llama_source_row())
    store.replace_engine_build_settings(
        "llama_cpp", {"cuda": False, "vulkan": False, "build_type": "Release"}
    )
    put = client.put(
        "/api/llama-versions/versions/build-config",
        json={
            "version_id": "llama_cpp:source-main",
            "build_config": {"cuda": True, "vulkan": True, "build_type": "Debug"},
        },
    )
    assert put.status_code == 200

    called = {}

    def fake_schedule(**kwargs):
        called.update(kwargs)
        return {"status": "started", "task_id": "build_sync_frozen"}

    monkeypatch.setattr(llama_routes, "_schedule_source_sync", fake_schedule)
    r = client.post(
        "/api/llama-versions/versions/sync",
        json={"version_id": "llama_cpp:source-main"},
    )
    assert r.status_code == 200
    assert called["build_config"].enable_cuda is True
    assert called["build_config"].enable_vulkan is True
    assert called["build_config"].build_type == "Debug"


def test_put_1cat_vllm_rejected(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version(
        "1cat_vllm",
        {
            "version": "v1",
            "type": "release",
            "install_type": "release",
            "venv_path": str(tmp_path / "oc"),
        },
    )
    r = client.put(
        "/api/llama-versions/versions/build-config",
        json={"version_id": "1cat_vllm:v1", "build_config": {"cuda": True}},
    )
    assert r.status_code == 400


def test_put_audio_rejects_unsupported_backend(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "audio-main",
            "type": "source",
            "install_type": "source",
            "source_ref": "main",
            "build_config": {"backend": "cpu"},
            "repository_source": "audio.cpp",
        },
    )
    from backend.audio_cpp_manager import AudioCppManager

    monkeypatch.setattr(
        AudioCppManager,
        "supported_build_backends",
        staticmethod(lambda: ["cpu", "cuda", "hip", "vulkan"]),
    )
    r = client.put(
        "/api/llama-versions/versions/build-config",
        json={
            "version_id": "audio_cpp:audio-main",
            "build_config": {"metal": True},
        },
    )
    assert r.status_code == 422
    assert "metal" in str(r.json()["detail"]).lower()
