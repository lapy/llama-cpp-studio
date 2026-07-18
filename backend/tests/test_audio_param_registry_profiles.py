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
    assert "instructions_policy" in payload
    assert payload["supports_voice_presets"] is (defaults_key == "speech_defaults")


def test_param_registry_exposes_qwen3_aligned_asr_sidecar_fields(monkeypatch):
    model = _audio_model("audio-qwen3-asr", "qwen3_asr", "asr")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [],
            "inspection": {
                "family": "qwen3_asr",
                "tasks": [{"task": "asr", "modes": ["offline"]}],
            },
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])
    keys = {field["key"] for field in payload.get("sidecar_session_fields") or []}
    # Empty profile sections → curated path overlays still fill the gap.
    assert "qwen3_asr.forced_aligner_model_path" in keys
    assert "qwen3_asr.vad_model_path" in keys


def test_param_registry_omits_curated_sidecar_when_profile_already_has_keys(
    monkeypatch,
):
    model = _audio_model("audio-qwen3-asr-discovered", "qwen3_asr", "asr")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [
                {
                    "id": "model_session_options",
                    "params": [
                        {
                            "key": "qwen3_asr.forced_aligner_model_path",
                            "scope": "session_option",
                            "type": "path",
                        },
                        {
                            "key": "qwen3_asr.vad_model_path",
                            "scope": "session_option",
                            "type": "path",
                        },
                        {
                            "key": "qwen3_asr.weight_type",
                            "scope": "session_option",
                            "type": "select",
                        },
                    ],
                }
            ],
            "inspection": {
                "family": "qwen3_asr",
                "tasks": [{"task": "asr", "modes": ["offline"]}],
            },
        },
    )
    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])
    assert payload.get("sidecar_session_fields") == []


def test_param_registry_includes_generic_profile_for_unknown_family(monkeypatch):
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

    assert payload["task_profile"].get("generic") is True
    assert payload["api_endpoint"] == "/v1/audio/speech"
    assert payload["request_defaults_key"] == "speech_defaults"
    assert payload["supports_voice_presets"] is True
    assert payload["instructions_policy"]
    assert "tts_profile" not in payload
    assert "asr_profile" not in payload


def test_param_registry_draft_family_task_overrides_saved_config(monkeypatch):
    model = _audio_model("audio-switch", "omnivoice", "tts")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [],
            "inspection": {"family": "omnivoice", "tasks": [{"task": "tts"}]},
        },
    )

    payload = _build_param_registry_payload(
        store,
        "audio_cpp",
        model_id=model["id"],
        draft_family="ace_step",
        draft_task="gen",
    )

    assert payload["policy_family"] == "ace_step"
    assert payload["policy_task"] == "gen"
    assert payload["request_defaults_key"] == "task_defaults"
    assert payload["api_endpoint"] == "/v1/tasks/run"
    assert payload["supports_voice_presets"] is False


def test_param_registry_uses_inspect_help_for_tasks_run_routing(monkeypatch):
    model = _audio_model("audio-qwen-multi", "qwen3_tts", "tts")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [
                {
                    "params": [
                        {"name": "task-route"},
                        {"name": "source-audio"},
                    ]
                }
            ],
            "inspection": {
                "family": "qwen3_tts",
                "tasks": [{"task": "tts"}, {"task": "vc"}],
            },
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert payload["api_endpoint"] == "/v1/tasks/run"
    assert payload["request_defaults_key"] == "task_defaults"
    assert payload["supports_voice_presets"] is False


def test_param_registry_chatterbox_vc_stays_on_speech_with_inspect(monkeypatch):
    model = _audio_model("audio-chatterbox-vc", "chatterbox", "vc")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [],
            "inspection": {
                "family": "chatterbox",
                "tasks": [{"task": "tts"}, {"task": "vc"}],
            },
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert payload["api_endpoint"] == "/v1/audio/speech"
    assert payload["request_defaults_key"] == "speech_defaults"
    assert payload["supports_voice_presets"] is True
