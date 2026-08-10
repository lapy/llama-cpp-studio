"""Tests for python_venv engine install/build settings persistence."""

import pytest

from backend.venv_install_settings import coerce_install_settings, default_install_settings


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


@pytest.fixture
def store(client, monkeypatch, tmp_path):
    return _install_temp_store(monkeypatch, tmp_path)


def test_lmdeploy_defaults():
    defaults = default_install_settings("lmdeploy")
    assert "InternLM/lmdeploy" in defaults["source_repo"]
    assert defaults["source_branch"] == "main"
    assert defaults["pip_version"] == ""


def test_onecat_defaults():
    defaults = default_install_settings("1cat_vllm")
    assert "1Cat-vLLM" in defaults["source_repo"]
    assert defaults["source_branch"] == "main"
    assert defaults["release_version"] == ""


def test_coerce_fills_missing_and_keeps_versions():
    out = coerce_install_settings(
        "lmdeploy",
        {"source_repo": "https://example.com/lm.git", "pip_version": "0.9.0"},
    )
    assert out["source_repo"] == "https://example.com/lm.git"
    assert out["source_branch"] == "main"
    assert out["pip_version"] == "0.9.0"


def test_lmdeploy_build_settings_round_trip(client, store):
    r = client.get("/api/lmdeploy/build-settings")
    assert r.status_code == 200
    body = r.json()
    assert body["source_branch"] == "main"

    r = client.put(
        "/api/lmdeploy/build-settings",
        json={
            "source_repo": "https://example.com/fork.git",
            "source_branch": "dev",
            "pip_version": "0.8.1",
        },
    )
    assert r.status_code == 200
    assert r.json()["source_repo"] == "https://example.com/fork.git"
    assert r.json()["pip_version"] == "0.8.1"

    stored = store.get_engine_build_settings("lmdeploy")
    assert stored["source_branch"] == "dev"

    r = client.get("/api/lmdeploy/build-settings")
    assert r.json()["source_repo"] == "https://example.com/fork.git"


def test_onecat_build_settings_round_trip(client, store):
    r = client.put(
        "/api/1cat-vllm/build-settings",
        json={
            "source_repo": "https://example.com/1cat.git",
            "source_branch": "sm70",
            "release_version": "0.1.0",
        },
    )
    assert r.status_code == 200
    assert r.json()["release_version"] == "0.1.0"
    assert store.get_engine_build_settings("1cat_vllm")["source_branch"] == "sm70"


def test_llama_build_settings_persists_tracking_ref(client, store):
    r = client.put(
        "/api/llama-versions/build-settings",
        params={"engine": "llama_cpp"},
        json={"cuda": True, "tracking_ref": "master"},
    )
    assert r.status_code == 200
    assert r.json().get("tracking_ref") == "master"
    assert r.json().get("cuda") is True

    r = client.get("/api/llama-versions/build-settings", params={"engine": "llama_cpp"})
    assert r.status_code == 200
    assert r.json().get("tracking_ref") == "master"
