"""Persist + list API coverage: non-default GitHub owners are labeled fork."""

from __future__ import annotations

import asyncio

import pytest

from backend.repo_identity import CANONICAL_REPOSITORY_URLS


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


@pytest.fixture
def store(client, monkeypatch, tmp_path):
    return _install_temp_store(monkeypatch, tmp_path)


@pytest.mark.parametrize(
    "engine,version,repo_url,extra",
    [
        (
            "llama_cpp",
            "source-main-fork",
            "https://github.com/alice/llama.cpp.git",
            {"repository_source": "llama.cpp", "binary_path": "/tmp/llama-server"},
        ),
        (
            "ik_llama",
            "source-main-fork",
            "https://github.com/bob/ik_llama.cpp.git",
            {"repository_source": "ik_llama.cpp", "binary_path": "/tmp/ik-server"},
        ),
        (
            "audio_cpp",
            "source-main-fork",
            "https://github.com/carol/audio.cpp.git",
            {"repository_source": "audio.cpp"},
        ),
        (
            "lmdeploy",
            "source-main-fork",
            "https://github.com/dave/lmdeploy.git",
            {"venv_path": "/tmp/lmdeploy/venv", "source_branch": "main"},
        ),
        (
            "1cat_vllm",
            "source-main-fork",
            "https://github.com/erin/1Cat-vLLM.git",
            {"venv_path": "/tmp/1cat/venv", "source_branch": "main"},
        ),
    ],
)
def test_list_versions_exposes_fork_label(client, store, engine, version, repo_url, extra):
    store.add_engine_version(
        engine,
        {
            "version": version,
            "type": "fork",
            "install_type": "source",
            "is_fork": True,
            "source_repo": repo_url,
            "source_ref": "main",
            "source_ref_type": "branch",
            "source_branch": "main",
            **extra,
        },
    )

    r = client.get("/api/llama-versions")
    assert r.status_code == 200
    row = next(item for item in r.json() if item.get("id") == f"{engine}:{version}")
    assert row["type"] == "fork"
    assert row["install_type"] == "source"
    assert row["is_fork"] is True
    assert row["source_repo"] == repo_url


def test_list_upstream_source_is_not_fork(client, store):
    store.add_engine_version(
        "llama_cpp",
        {
            "version": "source-main-upstream",
            "type": "source",
            "install_type": "source",
            "is_fork": False,
            "source_repo": CANONICAL_REPOSITORY_URLS["llama_cpp"],
            "source_ref": "master",
            "source_ref_type": "branch",
            "source_branch": "master",
            "repository_source": "llama.cpp",
            "binary_path": "/tmp/llama-server",
        },
    )
    r = client.get("/api/llama-versions")
    row = next(item for item in r.json() if item["version"] == "source-main-upstream")
    assert row["type"] == "source"
    assert row["is_fork"] is False


def test_branch_for_source_entry_allows_fork():
    from backend.routes.llama_versions import _branch_for_source_entry

    assert (
        _branch_for_source_entry(
            {
                "type": "fork",
                "install_type": "source",
                "source_branch": "dev",
            }
        )
        == "dev"
    )
    assert (
        _branch_for_source_entry(
            {
                "type": "fork",
                "install_type": "source",
                "source_ref": "feature",
                "source_ref_type": "branch",
            }
        )
        == "feature"
    )
    assert (
        _branch_for_source_entry(
            {
                "type": "local",
                "install_type": "local",
                "source_ref": "main",
            }
        )
        == "main"
    )


def test_sync_accepts_fork_install(client, store, monkeypatch, tmp_path):
    store.add_engine_version(
        "llama_cpp",
        {
            "version": "source-main",
            "type": "fork",
            "install_type": "source",
            "is_fork": True,
            "binary_path": str(tmp_path / "llama-server"),
            "source_ref": "main",
            "source_ref_type": "branch",
            "source_branch": "main",
            "source_repo": "https://github.com/alice/llama.cpp.git",
            "build_config": {"enable_cuda": False, "build_type": "Release"},
            "repository_source": "llama.cpp",
        },
    )

    from backend.routes import llama_versions as llama_routes

    monkeypatch.setattr(llama_routes.asyncio, "create_task", lambda coro: coro.close())
    monkeypatch.setattr(llama_routes.time, "time", lambda: 1_700_000_000)

    r = client.post(
        "/api/llama-versions/versions/sync",
        json={"version_id": "llama_cpp:source-main"},
    )
    assert r.status_code == 200
    assert r.json().get("task_id")


@pytest.mark.asyncio
async def test_llama_build_source_task_persists_fork(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)

    from backend.routes import llama_versions as llama_routes
    from backend.progress_manager import get_progress_manager

    async def fake_build(*_args, **_kwargs):
        return str(tmp_path / "bin" / "llama-server")

    async def fake_checkout(_version_name):
        return "abcdef1234567890"

    monkeypatch.setattr(llama_routes.llama_manager, "build_source", fake_build)
    monkeypatch.setattr(llama_routes, "_llama_checkout_commit", fake_checkout)
    monkeypatch.setattr(llama_routes, "mark_swap_config_stale", lambda: None)

    pm = get_progress_manager()
    task_id = "build_fork_test_1"
    pm.create_task("build", "fork test", {"engine": "llama_cpp"}, task_id=task_id)

    await llama_routes.build_source_task(
        "main",
        [],
        llama_routes.BuildConfig(),
        "source-main-forktest",
        "llama.cpp",
        "https://github.com/alice/llama.cpp.git",
        pm,
        task_id,
        auto_activate=False,
        source_ref_type="branch",
    )

    saved = store.get_engine_versions("llama_cpp")
    assert len(saved) == 1
    assert saved[0]["type"] == "fork"
    assert saved[0]["install_type"] == "source"
    assert saved[0]["is_fork"] is True
    assert saved[0]["source_repo"] == "https://github.com/alice/llama.cpp.git"


@pytest.mark.asyncio
async def test_llama_build_source_task_upstream_stays_source(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)

    from backend.routes import llama_versions as llama_routes
    from backend.progress_manager import get_progress_manager

    async def fake_build(*_args, **_kwargs):
        return str(tmp_path / "bin" / "llama-server")

    async def fake_checkout(_version_name):
        return "abcdef1234567890"

    monkeypatch.setattr(llama_routes.llama_manager, "build_source", fake_build)
    monkeypatch.setattr(llama_routes, "_llama_checkout_commit", fake_checkout)
    monkeypatch.setattr(llama_routes, "mark_swap_config_stale", lambda: None)

    pm = get_progress_manager()
    task_id = "build_upstream_test_1"
    pm.create_task("build", "upstream test", {"engine": "llama_cpp"}, task_id=task_id)

    await llama_routes.build_source_task(
        "master",
        [],
        llama_routes.BuildConfig(),
        "source-master-upstream",
        "llama.cpp",
        CANONICAL_REPOSITORY_URLS["llama_cpp"],
        pm,
        task_id,
        auto_activate=False,
        source_ref_type="branch",
    )

    saved = store.get_engine_versions("llama_cpp")[0]
    assert saved["type"] == "source"
    assert saved["is_fork"] is False


@pytest.mark.asyncio
async def test_audio_build_task_persists_fork(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)

    from backend.routes import audio_cpp_versions as audio_routes
    from backend.audio_cpp_manager import AudioCppBuildConfig
    from backend.progress_manager import get_progress_manager

    class FakeManager:
        async def build_source(self, **kwargs):
            return {
                "version": kwargs["version_name"],
                "source_ref": kwargs["source_ref"],
                "source_repo": kwargs["repository_url"],
                "build_config": {"backend": "cpu"},
            }

    monkeypatch.setattr(audio_routes, "get_audio_cpp_manager", lambda: FakeManager())
    monkeypatch.setattr(audio_routes, "_activate", lambda *_a, **_k: asyncio.sleep(0))

    pm = get_progress_manager()
    task_id = "build_audio_fork_1"
    pm.create_task("build", "audio fork", {"engine": "audio_cpp"}, task_id=task_id)

    await audio_routes._build_task(
        task_id=task_id,
        version_name="source-main-fork",
        source_ref="main",
        source_ref_type="branch",
        repository_url="https://github.com/other/audio.cpp.git",
        build_config=AudioCppBuildConfig(),
        auto_activate=False,
    )

    saved = store.get_engine_versions("audio_cpp")[0]
    assert saved["type"] == "fork"
    assert saved["install_type"] == "source"
    assert saved["is_fork"] is True
