"""Param registry contract across every documented audio.cpp family."""

import pytest

from backend.model_config import normalize_model_config
from backend.routes.models import _build_param_registry_payload
from backend.tests.audio_profile_fixtures import DOC_PROFILED_FAMILIES

# Full inspect task sets for multi-route families (matches loader capabilities).
_FAMILY_INSPECT_TASKS = {
    "chatterbox": ["clon", "vc"],
    "vevo2": ["tts", "vc", "s2s", "svc"],
    "seed_vc": ["vc", "svc"],
    "miocodec": ["vc", "s2s"],
}


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


def _inspection_tasks_for(family: str, task: str) -> list[dict]:
    tasks = list(_FAMILY_INSPECT_TASKS.get(family) or [task])
    if task not in tasks:
        tasks.insert(0, task)
    return [{"task": name, "modes": ["offline"]} for name in tasks]


@pytest.mark.parametrize(
    ("task", "family", "defaults_key", "endpoint"),
    DOC_PROFILED_FAMILIES,
)
def test_param_registry_payload_for_every_documented_family(
    monkeypatch, task, family, defaults_key, endpoint
):
    model = _audio_model(f"audio-{family}-{task}", family, task)
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
                "family": family,
                "tasks": _inspection_tasks_for(family, task),
            },
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])

    assert payload["task_profile"]["label"]
    assert payload["request_defaults_key"] == defaults_key
    assert payload["api_endpoint"] == endpoint
    assert payload["request_field_groups"]
    assert payload["api_example_hint"]
    assert "tts_profile" not in payload
    assert "asr_profile" not in payload
    assert "speech_field_groups" not in payload
    assert "transcription_field_groups" not in payload


def test_param_registry_warns_on_unknown_saved_load_options(monkeypatch):
    model = _audio_model(
        "audio-unknown-load",
        "omnivoice",
        "tts",
        load_options={"unknown_future_flag": True},
    )
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
                        {"key": "language", "scope": "load_option"},
                    ]
                }
            ],
            "inspection": {"family": "omnivoice"},
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])
    warnings = payload.get("compatibility_warnings") or []
    assert any("unknown_future_flag" in warning for warning in warnings)


def test_param_registry_warns_when_model_missing(monkeypatch):
    store = _Store(_audio_model("audio-missing", "omnivoice", "tts"))
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {"sections": [], "inspection": {}},
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id="not-found")
    warnings = payload.get("compatibility_warnings") or []
    assert any("Model context was not found" in warning for warning in warnings)


def test_param_registry_includes_scan_error_warning(monkeypatch):
    model = _audio_model("audio-scan-error", "omnivoice", "tts")
    store = _Store(model)
    monkeypatch.setattr(
        "backend.engine_param_catalog.get_version_entry",
        lambda *_a, **_k: {"sections": []},
    )
    monkeypatch.setattr(
        "backend.engine_param_scanner.scan_audio_cpp_model_profile",
        lambda *_a, **_k: {
            "sections": [],
            "scan_error": "CLI inspection timed out",
            "inspection": {"family": "omnivoice"},
        },
    )

    payload = _build_param_registry_payload(store, "audio_cpp", model_id=model["id"])
    warnings = payload.get("compatibility_warnings") or []
    assert any("CLI inspection timed out" in warning for warning in warnings)
