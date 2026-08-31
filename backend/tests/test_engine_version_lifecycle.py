"""Engine version registration, failure status, orphans, and retryability."""

from __future__ import annotations

import pytest

from backend.data_store import DataStore
from backend.engine_version_lifecycle import (
    BUILD_STATUS_BROKEN,
    BUILD_STATUS_FAILED,
    BUILD_STATUS_READY,
    annotate_version_row,
    collect_orphan_engine_rows,
    engine_version_is_retryable,
    mark_engine_version_building,
    mark_engine_version_failed,
    mark_engine_version_ready,
    normalize_engine_version_status,
    repair_stale_building_versions,
    upsert_engine_version,
)


def _store(tmp_path):
    return DataStore(config_dir=str(tmp_path / "config"))


def test_upsert_inserts_then_merges(tmp_path):
    store = _store(tmp_path)
    upsert_engine_version(
        store, "llama_cpp", {"version": "source-main", "binary_path": None}
    )
    upsert_engine_version(
        store, "llama_cpp", {"version": "source-main", "binary_path": "/bin/llama"}
    )
    rows = store.get_engine_versions("llama_cpp")
    assert len(rows) == 1
    assert rows[0]["binary_path"] == "/bin/llama"


def test_failed_build_is_registered_and_retryable(tmp_path):
    store = _store(tmp_path)
    mark_engine_version_building(
        store,
        "llama_cpp",
        {
            "version": "source-main-1",
            "source_ref": "main",
            "source_repo": "https://github.com/ggerganov/llama.cpp.git",
            "install_type": "source",
        },
        task_id="build_1",
    )
    mark_engine_version_failed(
        store,
        "llama_cpp",
        "source-main-1",
        error="cmake failed",
        extra={"source_ref": "main"},
    )
    row = store.get_engine_versions("llama_cpp")[0]
    assert row["build_status"] == BUILD_STATUS_FAILED
    assert row["build_error"] == "cmake failed"
    annotated = annotate_version_row("llama_cpp", row)
    assert annotated["retryable"] is True
    assert engine_version_is_retryable("llama_cpp", row)


def test_legacy_row_without_paths_is_broken(tmp_path):
    store = _store(tmp_path)
    store.add_engine_version("llama_cpp", {"version": "ghost", "type": "source"})
    row = store.get_engine_versions("llama_cpp")[0]
    assert normalize_engine_version_status("llama_cpp", row) == BUILD_STATUS_BROKEN


def test_ready_row_with_binary_stays_ready():
    row = {"version": "v1", "binary_path": "/bin/llama-server", "type": "source"}
    assert normalize_engine_version_status("llama_cpp", row) == BUILD_STATUS_READY


def test_stale_building_is_repaired(tmp_path):
    store = _store(tmp_path)
    mark_engine_version_building(
        store,
        "llama_cpp",
        {"version": "source-stale", "source_ref": "main"},
        task_id="gone",
    )
    repaired = repair_stale_building_versions(store, get_task=lambda _tid: None)
    assert repaired == 1
    assert store.get_engine_versions("llama_cpp")[0]["build_status"] == BUILD_STATUS_BROKEN


def test_orphan_dirs_are_listed_as_broken(tmp_path):
    store = _store(tmp_path)
    llama_root = tmp_path / "llama-cpp"
    orphan = llama_root / "source-orphan-1"
    (orphan / "llama.cpp" / ".git").mkdir(parents=True)
    (orphan / "llama.cpp" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (orphan / "llama.cpp" / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/ggerganov/llama.cpp.git\n'
    )
    store.add_engine_version(
        "llama_cpp",
        {
            "version": "kept",
            "binary_path": str(llama_root / "kept" / "llama-server"),
            "install_dir": str(llama_root / "kept"),
        },
    )
    (llama_root / "kept").mkdir()

    orphans = collect_orphan_engine_rows(
        store,
        roots={
            "llama_cpp": str(llama_root),
            "ik_llama": str(llama_root),
            "audio_cpp": str(tmp_path / "audio"),
            "lmdeploy": str(tmp_path / "lm"),
            "1cat_vllm": str(tmp_path / "oc"),
        },
    )
    names = {row["version"] for row in orphans}
    assert "source-orphan-1" in names
    assert "kept" not in names
    orphan_row = next(row for row in orphans if row["version"] == "source-orphan-1")
    assert orphan_row["build_status"] == BUILD_STATUS_BROKEN
    assert orphan_row["engine"] == "llama_cpp"
    assert orphan_row["retryable"] is True
    assert orphan_row["source_ref"] == "main"


@pytest.mark.asyncio
async def test_build_source_task_registers_failure(monkeypatch, tmp_path):
    from backend import data_store
    from backend.progress_manager import get_progress_manager
    from backend.routes import llama_versions as llama_routes

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)

    async def boom(*_args, **_kwargs):
        raise RuntimeError("compile exploded")

    monkeypatch.setattr(llama_routes.llama_manager, "build_source", boom)
    monkeypatch.setattr(llama_routes, "mark_swap_config_stale", lambda: None)
    llama_routes.llama_manager.llama_dir = str(tmp_path / "llama-cpp")

    pm = get_progress_manager()
    task_id = "build_fail_reg_1"
    pm.create_task("build", "fail", {"engine": "llama_cpp"}, task_id=task_id)

    await llama_routes.build_source_task(
        "main",
        [],
        llama_routes.BuildConfig(),
        "source-main-fail",
        "llama.cpp",
        "https://github.com/ggerganov/llama.cpp.git",
        pm,
        task_id,
        auto_activate=False,
        source_ref_type="branch",
    )

    saved = store.get_engine_versions("llama_cpp")
    assert len(saved) == 1
    assert saved[0]["version"] == "source-main-fail"
    assert saved[0]["build_status"] == BUILD_STATUS_FAILED
    assert "compile exploded" in saved[0]["build_error"]
    assert engine_version_is_retryable("llama_cpp", saved[0])


def test_list_api_includes_failed_and_orphan(client, monkeypatch, tmp_path):
    from backend import data_store
    from backend.routes import llama_versions as llama_routes

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
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

    store.add_engine_version(
        "llama_cpp",
        {
            "version": "source-failed",
            "type": "source",
            "install_type": "source",
            "source_ref": "main",
            "source_repo": "https://github.com/ggerganov/llama.cpp.git",
            "build_status": "failed",
            "build_error": "ninja died",
            "repository_source": "llama.cpp",
        },
    )

    r = client.get("/api/llama-versions")
    assert r.status_code == 200
    rows = r.json()
    failed = next(item for item in rows if item["version"] == "source-failed")
    assert failed["build_status"] == "failed"
    assert failed["retryable"] is True
    assert failed["cmake_editable"] is True
    orphan = next(item for item in rows if item["version"] == "disk-only")
    assert orphan["build_status"] == "broken"
    assert orphan["orphan"] is True
    assert orphan["cmake_editable"] is False


def test_retry_endpoint_reschedules_failed_build(client, monkeypatch, tmp_path):
    from backend import data_store
    from backend.routes import llama_versions as llama_routes

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(llama_routes.asyncio, "create_task", lambda coro: coro.close())
    store.add_engine_version(
        "llama_cpp",
        {
            "version": "source-main-retry",
            "type": "source",
            "install_type": "source",
            "source_ref": "main",
            "source_ref_type": "branch",
            "source_branch": "main",
            "source_repo": "https://github.com/ggerganov/llama.cpp.git",
            "build_status": "failed",
            "build_error": "boom",
            "repository_source": "llama.cpp",
        },
    )

    r = client.post(
        "/api/llama-versions/versions/retry",
        json={"version_id": "llama_cpp:source-main-retry"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["retry"] is True
    assert body["version_name"] == "source-main-retry"
    assert store.get_engine_versions("llama_cpp")[0]["build_status"] == "building"


def test_activate_rejects_failed_version(client, monkeypatch, tmp_path):
    from backend import data_store

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    store.add_engine_version(
        "llama_cpp",
        {
            "version": "source-bad",
            "binary_path": "/nope",
            "build_status": "failed",
            "source_ref": "main",
            "repository_source": "llama.cpp",
        },
    )
    r = client.post(
        "/api/llama-versions/versions/activate",
        json={"version_id": "llama_cpp:source-bad"},
    )
    assert r.status_code == 400
    assert "failed" in r.json()["detail"]
