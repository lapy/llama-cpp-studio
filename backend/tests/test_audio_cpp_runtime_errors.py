"""Failure paths and internal helpers for audio.cpp runtime building."""

from pathlib import Path

import pytest

import backend.audio_cpp_runtime as audio_runtime
from backend.model_config import normalize_model_config


class _Store:
    def __init__(self, active):
        self.active = active

    def get_active_engine_version(self, engine):
        return self.active if engine == "audio_cpp" else None


def _base_fixture(tmp_path):
    model_root = tmp_path / "models" / "demo"
    model_root.mkdir(parents=True)
    binary = tmp_path / "build" / "bin" / "audiocpp_server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    active = {
        "version": "v1",
        "server_binary_path": str(binary),
        "cli_binary_path": str(binary.parent / "audiocpp_cli"),
        "source_path": str(tmp_path),
        "build_config": {"backend": "cuda"},
    }
    config = {
        "engine": "audio_cpp",
        "family": "demo_tts",
        "task": "tts",
        "mode": "offline",
        "backend": "cuda",
        "device": 0,
        "threads": 4,
    }
    model = {
        "id": "audio-cpp--demo",
        "artifact": {"package_kind": "prepared_bundle", "path": str(model_root)},
        "compatible_engines": ["audio_cpp"],
        "config": normalize_model_config(
            {"engine": "audio_cpp", "engines": {"audio_cpp": config}}
        ),
    }
    return _Store(active), model, config


def test_safe_sidecar_name_is_stable_and_unique():
    first = audio_runtime._safe_sidecar_name("org/model:q4_k_m")
    second = audio_runtime._safe_sidecar_name("org/model:q4_k_m")
    other = audio_runtime._safe_sidecar_name("other-model")
    assert first == second
    assert first != other
    assert first.endswith(".json")
    assert len(first) <= 90


def test_artifact_model_path_prefers_artifact_path(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    model = {
        "artifact": {"path": str(model_root)},
        "local_path": str(tmp_path / "ignored"),
    }
    assert audio_runtime._artifact_model_path(model) == str(model_root.resolve())


def test_artifact_model_path_falls_back_to_local_path(tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    model = {"local_path": str(local)}
    assert audio_runtime._artifact_model_path(model) == str(local.resolve())


def test_artifact_model_path_falls_back_to_model_path(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    model = {"model_path": str(path)}
    assert audio_runtime._artifact_model_path(model) == str(path.resolve())


def test_clean_options_drops_empty_values():
    assert audio_runtime._clean_options(
        {"language": "en", "empty": "", "none": None, "keep": 0}
    ) == {"language": "en", "keep": 0}


def test_custom_args_rejects_studio_owned_flags():
    with pytest.raises(ValueError, match="Studio-owned"):
        audio_runtime._custom_args("--port 9999")


def test_custom_args_rejects_invalid_shell_syntax():
    with pytest.raises(ValueError, match="Invalid custom_args shell syntax"):
        audio_runtime._custom_args('echo "unterminated')


def test_custom_args_accepts_valid_tokens():
    assert audio_runtime._custom_args("--foo bar") == ["--foo", "bar"]


def test_flag_tokens_repeatable_expands_items():
    row = {"primary_flag": "--ctx", "value_kind": "repeatable"}
    assert audio_runtime._flag_tokens("ctx", ["a", "b"], row) == [
        "--ctx",
        "a",
        "--ctx",
        "b",
    ]


def test_flag_tokens_bool_uses_negative_flag():
    row = {
        "primary_flag": "--lazy-load",
        "negative_flag": "--no-lazy-load",
        "value_kind": "flag",
    }
    assert audio_runtime._flag_tokens("lazy_load", True, row) == ["--lazy-load"]
    assert audio_runtime._flag_tokens("lazy_load", False, row) == ["--no-lazy-load"]


def test_build_audio_runtime_raises_when_feature_disabled(tmp_path, monkeypatch):
    store, model, config = _base_fixture(tmp_path)
    monkeypatch.setattr(audio_runtime, "audio_cpp_enabled", lambda: False)
    with pytest.raises(ValueError, match="AUDIO_CPP_ENABLED"):
        audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")


def test_build_audio_runtime_raises_when_no_active_binary(tmp_path, monkeypatch):
    store, model, config = _base_fixture(tmp_path)
    store.active = None
    monkeypatch.setattr(audio_runtime, "audio_cpp_enabled", lambda: True)
    with pytest.raises(ValueError, match="No active audio.cpp server binary"):
        audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")


def test_build_audio_runtime_raises_when_binary_missing(tmp_path, monkeypatch):
    store, model, config = _base_fixture(tmp_path)
    Path(store.active["server_binary_path"]).unlink()
    monkeypatch.setattr(audio_runtime, "audio_cpp_enabled", lambda: True)
    with pytest.raises(ValueError, match="server binary not found"):
        audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")


def test_build_audio_runtime_raises_when_model_directory_missing(tmp_path, monkeypatch):
    store, model, config = _base_fixture(tmp_path)
    model_root = Path(model["artifact"]["path"])
    model_root.rmdir()
    monkeypatch.setattr(audio_runtime, "audio_cpp_enabled", lambda: True)
    with pytest.raises(ValueError, match="model directory does not exist"):
        audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")


def test_build_audio_runtime_propagates_validation_errors(tmp_path, monkeypatch):
    store, model, config = _base_fixture(tmp_path)

    def fail_validation(*_args, **_kwargs):
        raise ValueError("family is required")

    monkeypatch.setattr(audio_runtime, "audio_cpp_enabled", lambda: True)
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", fail_validation)
    with pytest.raises(ValueError, match="family is required"):
        audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")


def test_build_audio_runtime_includes_config_weight_and_model_lazy(
    tmp_path, monkeypatch
):
    store, model, config = _base_fixture(tmp_path)
    config.update({"config": "main", "weight": "default", "model_lazy": True})
    monkeypatch.setattr(audio_runtime, "audio_cpp_enabled", lambda: True)
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    row = runtime["sidecar"]["models"][0]
    assert row["config"] == "main"
    assert row["weight"] == "default"
    assert row["lazy"] is True
