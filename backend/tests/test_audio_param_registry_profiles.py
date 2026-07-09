"""Param registry integration for audio.cpp task profiles."""

from types import SimpleNamespace

import pytest

from backend.model_config import normalize_model_config
from backend.routes.models import _build_param_registry_payload


class _Store:
    def __init__(self, model):
        self.model = model
        self.active = {
            "version": "v1",
            "server_binary_path": "/server",
            "cli_binary_path": "/cli",
            "build_config": {"backend": "cuda"},
        }

    def get_active_engine_version(self, engine):
        assert engine == "audio_cpp"
        return self.active

    def get_model(self, model_id):
        if model_id == self.model.get("id"):
            return self.model
        return None


def _audio_model(model_id, family, task, **audio_config):
    config = {
        "family": family,
        "task": task,
        "mode": "offline",
        "backend": "cuda",
    }
    config.update(audio_config)
    return {
        "id": model_id,
        "family": family,
        "tasks": [task],
        "compatible_engines": ["audio_cpp"],
        "artifact": {"path": "/tmp/model"},
        "config": normalize_model_config(
            {"engine": "audio_cpp", "engines": {"audio_cpp": config}}
        ),
    }


@pytest.mark.parametrize(
    ("family", "task", "defaults_key", "endpoint"),
    [
        ("omnivoice", "tts", "speech_defaults", "/v1/audio/speech"),
        ("nemotron_asr", "asr", "transcription_defaults", "/v1/audio/transcriptions"),
        ("ace_step", "gen", "task_defaults", "/v1/tasks/run"),
        ("seed_vc", "vc", "task_defaults", "/v1/tasks/run"),
        ("silero_vad", "vad", "task_defaults", "/v1/tasks/run"),
        ("qwen3_forced_aligner", "align", "task_defaults", "/v1/tasks/run"),
    ],
)
def test_param_registry_includes_task_profile_metadata(
    monkeypatch, family, task, defaults_key, endpoint
):
    model = _audio_model(f"audio-{family}", family, task)
    store = _Store(model)

    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [],
            "inspection": {"family": family, "tasks": [{"task": task, "modes": ["offline"]}]},
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert payload["task_profile"]["label"]
    assert payload["request_defaults_key"] == defaults_key
    assert payload["api_endpoint"] == endpoint
    assert payload["request_field_groups"]
    assert payload["api_example_hint"]


def test_param_registry_tts_backward_compatible_aliases(monkeypatch):
    model = _audio_model("audio-kokoro", "kokoro_tts", "tts")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {"sections": [], "inspection": {"family": "kokoro_tts"}},
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert payload["tts_profile"] == payload["task_profile"]
    assert payload["speech_field_groups"] == payload["request_field_groups"]


def test_param_registry_asr_backward_compatible_aliases(monkeypatch):
    model = _audio_model("audio-nemotron", "nemotron_asr", "asr")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {"sections": [], "inspection": {"family": "nemotron_asr"}},
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert payload["asr_profile"] == payload["task_profile"]
    assert payload["transcription_field_groups"] == payload["request_field_groups"]


def test_param_registry_omits_profiles_for_unknown_family(monkeypatch):
    model = _audio_model("audio-unknown", "unknown_family", "tts")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {"sections": [], "inspection": {"family": "unknown_family"}},
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert "task_profile" not in payload
    assert "request_field_groups" not in payload
