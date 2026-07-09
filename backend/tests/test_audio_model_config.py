"""Typed, inspected audio.cpp model configuration validation."""

import pytest

from backend.audio_model_config import validate_audio_model_config
from backend.model_config import normalize_model_config


def _profile(model_root):
    return {
        "fingerprint": "profile-1",
        "inspection": {
            "family": "demo_tts",
            "tasks": [
                {"task": "tts", "modes": ["offline", "streaming"]},
                {"task": "asr", "modes": ["offline"]},
            ],
            "configs": [{"id": "main", "path": str(model_root / "config.json")}],
            "weights": [{"id": "default", "path": str(model_root / "model.safetensors")}],
            "capabilities": {"streaming": True},
        },
        "sections": [
            {
                "params": [
                    {
                        "key": "temperature",
                        "scope": "session_option",
                        "type": "float",
                        "minimum": 0.0,
                        "maximum": 2.0,
                    },
                    {
                        "key": "language",
                        "scope": "load_option",
                        "type": "select",
                        "options": [
                            {"value": "en", "label": "English"},
                            {"value": "fr", "label": "French"},
                        ],
                    },
                ]
            }
        ],
    }


class _Store:
    def __init__(self, active):
        self.active = active

    def get_active_engine_version(self, engine):
        assert engine == "audio_cpp"
        return self.active


def _model(model_root):
    return {
        "id": "audio-cpp--demo",
        "family": "demo_tts",
        "tasks": ["tts", "asr"],
        "compatible_engines": ["audio_cpp"],
        "artifact": {
            "package_kind": "prepared_bundle",
            "path": str(model_root),
        },
    }


def _config(**updates):
    audio = {
        "family": "demo_tts",
        "task": "tts",
        "mode": "streaming",
        "backend": "cuda",
        "device": 0,
        "threads": 4,
        "config": "main",
        "weight": "default",
        "load_options": {"language": "en"},
        "session_options": {"temperature": 0.8},
    }
    audio.update(updates)
    return normalize_model_config(
        {"engine": "audio_cpp", "engines": {"audio_cpp": audio}}
    )


def test_validates_audio_identity_assets_backend_and_nested_options(
    tmp_path, monkeypatch
):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    active = {
        "version": "v1",
        "server_binary_path": "/server",
        "cli_binary_path": "/cli",
        "build_config": {"backend": "cuda"},
    }
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: _profile(model_root),
    )

    result = validate_audio_model_config(
        _Store(active), _model(model_root), _config()
    )

    assert result["errors"] == []
    assert result["profile_fingerprint"] == "profile-1"
    assert result["inspection"]["capabilities"]["streaming"] is True


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"family": "wrong"}, "does not match inspected family"),
        ({"task": "vad"}, "is not exposed by this package"),
        ({"task": "asr", "mode": "streaming"}, "is not supported for task"),
        ({"backend": "vulkan"}, "unavailable in the active cuda"),
        (
            {"session_options": {"temperature": 3.0}},
            "temperature must be at most 2.0",
        ),
        (
            {"load_options": {"language": "xx"}},
            "language has unsupported value",
        ),
        (
            {"custom_args": "--port 9999"},
            "--port is Studio-owned",
        ),
        (
            {"request_options": {"seed": 7}},
            "request_options are request-time capabilities",
        ),
    ],
)
def test_rejects_incompatible_audio_configuration(
    tmp_path, monkeypatch, updates, message
):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    active = {
        "version": "v1",
        "server_binary_path": "/server",
        "cli_binary_path": "/cli",
        "build_config": {"backend": "cuda"},
    }
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: _profile(model_root),
    )

    with pytest.raises(ValueError, match=message):
        validate_audio_model_config(
            _Store(active), _model(model_root), _config(**updates)
        )

