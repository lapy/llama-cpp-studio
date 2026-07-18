"""API coverage for audio.cpp tracking, update/sync, and status."""

from __future__ import annotations

import pytest

from backend.audio_cpp_manager import AudioCppBuildConfig


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


@pytest.fixture
def store(client, monkeypatch, tmp_path):
    return _install_temp_store(monkeypatch, tmp_path)


def test_build_settings_round_trip_preserves_tracking_ref(client, store, monkeypatch):
    monkeypatch.setattr(
        "backend.audio_cpp_tracking.resolve_bootstrap_tracking_ref",
        lambda: "release-bootstrap",
    )
    r = client.get("/api/audio-cpp/build-settings")
    assert r.status_code == 200
    body = r.json()
    assert body["tracking_ref"] == "release-bootstrap"
    assert "backend" in body

    r = client.put(
        "/api/audio-cpp/build-settings",
        json={
            "tracking_ref": "release-0.3",
            "repository_url": "https://github.com/0xShug0/audio.cpp.git",
            "backend": "cuda",
            "jobs": 4,
        },
    )
    assert r.status_code == 200
    saved = r.json()
    assert saved["tracking_ref"] == "release-0.3"
    assert saved["backend"] == "cuda"
    assert saved["jobs"] == 4

    r = client.get("/api/audio-cpp/build-settings")
    assert r.json()["tracking_ref"] == "release-0.3"
    assert r.json()["backend"] == "cuda"


def test_check_updates_uses_stored_tracking_ref(client, store, monkeypatch):
    store.update_engine_build_settings(
        "audio_cpp",
        {
            "tracking_ref": "my-branch",
            "repository_url": "https://github.com/0xShug0/audio.cpp.git",
            "backend": "cpu",
        },
    )
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "source-my-branch-aaaa",
            "source_commit": "aaaa1111bbbb2222cccc3333dddd4444eeee5555",
            "repository_source": "audio.cpp",
            "server_binary_path": "/tmp/audiocpp_server",
            "cli_binary_path": "/tmp/audiocpp_cli",
        },
    )
    store.set_active_engine_version("audio_cpp", "source-my-branch-aaaa")

    from backend.routes import audio_cpp_versions as routes

    async def fake_latest(ref: str, repository_url: str | None = None):
        assert ref == "my-branch"
        assert repository_url == "https://github.com/0xShug0/audio.cpp.git"
        return {
            "sha": "ffff1111bbbb2222cccc3333dddd4444eeee5555",
            "html_url": "https://example.test/commit/ffff",
            "ref": ref,
            "repository": "0xShug0/audio.cpp",
        }

    monkeypatch.setattr(routes, "_latest_upstream", fake_latest)

    r = client.get("/api/audio-cpp/check-updates")
    assert r.status_code == 200
    data = r.json()
    assert data["tracking_ref"] == "my-branch"
    assert data["repository_url"] == "https://github.com/0xShug0/audio.cpp.git"
    assert data["update_available"] is True
    assert data["latest_version"].startswith("ffff")


def test_check_updates_uses_fork_repository_url(client, store, monkeypatch):
    store.update_engine_build_settings(
        "audio_cpp",
        {
            "tracking_ref": "feature/catalog",
            "repository_url": "https://github.com/lapy/audio.cpp.git",
            "backend": "cpu",
        },
    )

    from backend.routes import audio_cpp_versions as routes

    async def fake_latest(ref: str, repository_url: str | None = None):
        assert ref == "feature/catalog"
        assert repository_url == "https://github.com/lapy/audio.cpp.git"
        return {
            "sha": "abcd1111bbbb2222cccc3333dddd4444eeee5555",
            "html_url": "https://github.com/lapy/audio.cpp/commit/abcd",
            "ref": ref,
            "repository": "lapy/audio.cpp",
        }

    monkeypatch.setattr(routes, "_latest_upstream", fake_latest)
    r = client.get("/api/audio-cpp/check-updates")
    assert r.status_code == 200
    assert r.json()["repository_url"].endswith("lapy/audio.cpp.git")


def test_github_api_repo_slug_parses_https_and_ssh():
    from backend.routes.audio_cpp_versions import _github_api_repo_slug

    assert (
        _github_api_repo_slug("https://github.com/lapy/audio.cpp.git")
        == "lapy/audio.cpp"
    )
    assert (
        _github_api_repo_slug("git@github.com:0xShug0/audio.cpp.git")
        == "0xShug0/audio.cpp"
    )
    assert _github_api_repo_slug("https://gitlab.com/org/audio.cpp.git") is None


def test_update_prefers_in_place_sync_when_branch_matches(
    client, store, monkeypatch, tmp_path
):
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    store.update_engine_build_settings(
        "audio_cpp",
        {"tracking_ref": "release-0.3", "backend": "cpu"},
    )
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "source-release-0.3-old",
            "source_path": str(source_dir),
            "source_ref": "release-0.3",
            "source_ref_type": "branch",
            "source_branch": "release-0.3",
            "source_commit": "1111111111111111111111111111111111111111",
            "repository_source": "audio.cpp",
            "server_binary_path": str(tmp_path / "server"),
            "cli_binary_path": str(tmp_path / "cli"),
        },
    )
    store.set_active_engine_version("audio_cpp", "source-release-0.3-old")

    from backend.routes import audio_cpp_versions as routes

    async def fake_latest(ref: str, repository_url: str | None = None):
        return {"sha": "2222222222222222222222222222222222222222", "ref": ref}

    called = {}

    def fake_sync(version_entry, branch, build_config):
        called["mode"] = "sync"
        called["branch"] = branch
        called["version"] = version_entry.get("version")
        assert isinstance(build_config, AudioCppBuildConfig)
        return {
            "message": "syncing",
            "task_id": "sync-1",
            "status": "started",
            "sync": True,
            "source_ref": branch,
            "source_ref_type": "branch",
            "version_name": version_entry.get("version"),
        }

    monkeypatch.setattr(routes, "_latest_upstream", fake_latest)
    monkeypatch.setattr(routes, "schedule_audio_cpp_sync", fake_sync)

    r = client.post("/api/audio-cpp/update", json={})
    assert r.status_code == 200
    assert r.json()["sync"] is True
    assert called["mode"] == "sync"
    assert called["branch"] == "release-0.3"


def test_update_rebuilds_branch_install_when_no_matching_checkout(
    client, store, monkeypatch
):
    store.update_engine_build_settings(
        "audio_cpp",
        {"tracking_ref": "release-0.3", "backend": "cpu"},
    )
    # Active install is a detached commit — cannot sync
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "source-deadbeef",
            "source_ref": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "source_ref_type": "commit",
            "source_branch": None,
            "source_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
            "repository_source": "audio.cpp",
        },
    )
    store.set_active_engine_version("audio_cpp", "source-deadbeef")

    from backend.routes import audio_cpp_versions as routes

    async def fake_latest(ref: str, repository_url: str | None = None):
        return {"sha": "abcdabcdabcdabcdabcdabcdabcdabcdabcdabcd", "ref": ref}

    called = {}

    def fake_schedule(payload):
        called["payload"] = payload
        return {
            "message": "building",
            "task_id": "build-1",
            "status": "started",
            "version_name": "source-release-0.3-abcdabcd",
            "source_ref": payload.get("source_ref"),
            "source_ref_type": payload.get("source_ref_type"),
        }

    monkeypatch.setattr(routes, "_latest_upstream", fake_latest)
    monkeypatch.setattr(routes, "_schedule_build", fake_schedule)

    r = client.post("/api/audio-cpp/update", json={})
    assert r.status_code == 200
    assert r.json().get("sync") is not True
    assert called["payload"]["source_ref"] == "release-0.3"
    assert called["payload"]["source_ref_type"] == "branch"
    assert called["payload"]["auto_activate"] is True


def test_status_exposes_tracking_and_contract_fields(client, store, monkeypatch, tmp_path):
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_text("x")
    cli.write_text("x")
    store.update_engine_build_settings(
        "audio_cpp",
        {"tracking_ref": "main", "repository_url": "https://example.test/audio.cpp.git"},
    )
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "v1",
            "source_commit": "abc",
            "server_binary_path": str(server),
            "cli_binary_path": str(cli),
            "model_manager_path": str(tmp_path / "missing.py"),
            "repository_source": "audio.cpp",
        },
    )
    store.set_active_engine_version("audio_cpp", "v1")

    monkeypatch.setattr(
        "backend.routes.audio_cpp_versions.get_version_entry",
        lambda *a, **k: {
            "contract_fingerprint": "f" * 64,
            "contract_changed": True,
            "previous_contract_fingerprint": "0" * 64,
            "capabilities": {
                "families": ["qwen3_tts"],
                "tasks": ["tts"],
            },
        },
    )
    monkeypatch.setattr(
        "backend.audio_cpp_tracking.resolve_bootstrap_tracking_ref",
        lambda: "main",
    )

    r = client.get("/api/audio-cpp/status")
    assert r.status_code == 200
    data = r.json()
    assert data["tracking_ref"] == "main"
    assert data["contract_fingerprint"] == "f" * 64
    assert data["contract_changed"] is True
    assert "qwen3_tts" in data["families"]
    assert "tts" in data["tasks"]
    assert "capability_delta" in data
    assert "affected_models" in data
    assert "compatibility_commit" not in data
    assert "compatibility_verified" not in data


def test_activate_returns_capability_delta(client, store, monkeypatch, tmp_path):
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_text("x")
    cli.write_text("x")
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "v2",
            "server_binary_path": str(server),
            "cli_binary_path": str(cli),
            "repository_source": "audio.cpp",
        },
    )
    store.add_model(
        {
            "id": "audio-cpp--demo",
            "name": "Demo",
            "compatible_engines": ["audio_cpp"],
            "config": {"engine": "audio_cpp", "engines": {"audio_cpp": {"family": "demo"}}},
        }
    )

    from backend.routes import audio_cpp_versions as routes

    monkeypatch.setattr(
        routes,
        "get_version_entry",
        lambda *a, **k: {"capabilities": {"families": ["old_fam"], "tasks": ["tts"]}},
    )

    def fake_scan(store_obj, engine, row):
        return {
            "contract_fingerprint": "a" * 64,
            "contract_changed": True,
            "capabilities": {"families": ["old_fam", "new_fam"], "tasks": ["tts", "asr"]},
        }

    profile_calls = []

    def fake_profile(store_obj, row, model, force=False):
        profile_calls.append({"id": model.get("id"), "force": force})
        return {"fingerprint": "fp-demo", "scan_error": None, "sections": []}

    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_engine_version",
        fake_scan,
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        fake_profile,
    )
    monkeypatch.setattr(
        "backend.llama_swap_manager.mark_swap_config_stale",
        lambda: None,
    )

    class FakeSwap:
        async def start_proxy(self):
            return None

    monkeypatch.setattr(
        "backend.llama_swap_manager.get_llama_swap_manager",
        lambda: FakeSwap(),
    )

    r = client.post("/api/audio-cpp/versions/activate", json={"version_id": "v2"})
    assert r.status_code == 200
    data = r.json()
    assert data["capability_delta"]["added_families"] == ["new_fam"]
    assert data["capability_delta"]["added_tasks"] == ["asr"]
    assert data["contract_changed"] is True
    assert profile_calls == [{"id": "audio-cpp--demo", "force": True}]
    assert data["profiles_rescanned"] == [
        {
            "id": "audio-cpp--demo",
            "ok": True,
            "scan_error": None,
            "fingerprint": "fp-demo",
        }
    ]


def test_llama_versions_activate_delegates_to_audio_cpp_activate(
    client, store, monkeypatch, tmp_path
):
    """Engines UI uses /llama-versions/activate; it must hit the rich audio path."""
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_text("x")
    cli.write_text("x")
    store.add_engine_version(
        "audio_cpp",
        {
            "version": "v-rich",
            "server_binary_path": str(server),
            "cli_binary_path": str(cli),
            "repository_source": "audio.cpp",
        },
    )

    called = {}

    async def fake_activate(version: str):
        called["version"] = version
        return {
            "message": f"Activated audio.cpp version {version}",
            "capability_delta": {
                "added_families": ["demo"],
                "removed_families": [],
                "added_tasks": [],
                "removed_tasks": [],
            },
            "contract_changed": False,
            "affected_models": [],
        }

    monkeypatch.setattr(
        "backend.routes.audio_cpp_versions._activate",
        fake_activate,
    )

    r = client.post(
        "/api/llama-versions/versions/activate",
        json={"version_id": "audio_cpp:v-rich"},
    )
    assert r.status_code == 200, r.text
    assert called["version"] == "v-rich"
    assert r.json()["capability_delta"]["added_families"] == ["demo"]
