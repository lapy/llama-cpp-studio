"""Typed, inspected audio.cpp model configuration validation."""

import pytest

from backend.audio_model_config import sanitize_audio_engine_section, validate_audio_model_config
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


def test_sanitize_audio_engine_section_drops_documentation_metavars():
    cleaned = sanitize_audio_engine_section(
        {
            "family": "omnivoice",
            "task": "tts",
            "log": True,
            "key=value": True,
            "device": 0,
        }
    )
    assert cleaned == {
        "family": "omnivoice",
        "task": "tts",
        "log": True,
        "device": 0,
    }


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


def test_coerces_numeric_string_device_and_threads(tmp_path, monkeypatch):
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
    normalized = _config(device="0", threads="4")

    result = validate_audio_model_config(
        _Store(active), _model(model_root), normalized
    )

    assert result["errors"] == []
    effective = normalized["engines"]["audio_cpp"]
    assert effective["device"] == 0
    assert effective["threads"] == 4


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
            {"custom_args": "--backend cuda"},
            "--backend is Studio-owned",
        ),
        (
            {"custom_args": ["--log"]},
            "custom_args must be string",
        ),
        (
            {"device": "gpu"},
            "device must be int",
        ),
        (
            {"threads": 0},
            "threads must be at least 1",
        ),
        (
            {"lazy_load": "yes"},
            "lazy_load must be bool",
        ),
        (
            {"model_lazy": 1},
            "model_lazy must be bool",
        ),
        (
            {"family": ["demo_tts"]},
            "family must be string",
        ),
        (
            {"task": ["tts"]},
            "task must be string",
        ),
        (
            {"mode": ["offline"]},
            "mode must be string",
        ),
        (
            {"backend": 1},
            "backend must be string",
        ),
        (
            {"backend": False},
            "backend must be string",
        ),
        (
            {"request_options": {"seed": 7}},
            "request_options are request-time capabilities",
        ),
        (
            {"speech_defaults": "not-an-object"},
            "speech_defaults must be an object",
        ),
        (
            {"task": "asr", "transcription_defaults": []},
            "transcription_defaults must be an object",
        ),
        (
            {"task": "gen", "family": "ace_step", "task_defaults": "bad"},
            "task_defaults must be an object",
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


def test_rejects_invalid_voice_presets(tmp_path, monkeypatch):
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

    with pytest.raises(ValueError, match="default_voice_preset 'missing'"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(
                voice_presets={"assistant": {"voice_id": "M1"}},
                default_voice_preset="missing",
            ),
        )


def test_accepts_valid_speech_and_transcription_defaults(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    refs = model_root / "refs"
    refs.mkdir()
    (refs / "voice.wav").write_bytes(b"RIFF")
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
        _Store(active),
        _model(model_root),
        _config(
            speech_defaults={
                "voice": "assistant",
                "voice_ref": "refs/voice.wav",
                "temperature": 0.7,
            },
        ),
    )
    assert result["errors"] == []


def test_rejects_invalid_speech_default_reference_audio(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    (tmp_path / "outside.wav").write_bytes(b"RIFF")
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

    with pytest.raises(ValueError, match="speech_defaults.voice_ref does not exist"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(speech_defaults={"voice_ref": "refs/missing.wav"}),
        )

    with pytest.raises(ValueError, match="speech_defaults.voice_ref escapes"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(speech_defaults={"voice_ref": "../outside.wav"}),
        )


def test_accepts_transcription_defaults_for_asr_task(tmp_path, monkeypatch):
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
    profile = _profile(model_root)
    profile["inspection"]["family"] = "nemotron_asr"
    profile["inspection"]["tasks"] = [{"task": "asr", "modes": ["offline"]}]
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: profile,
    )

    result = validate_audio_model_config(
        _Store(active),
        _model(model_root),
        _config(
            family="nemotron_asr",
            task="asr",
            mode="offline",
            transcription_defaults={"language": "en"},
        ),
    )
    assert result["errors"] == []


def test_rejects_missing_model_directory(tmp_path, monkeypatch):
    model_root = tmp_path / "missing"
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

    with pytest.raises(ValueError, match="model directory does not exist"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(),
        )


def test_rejects_no_runnable_active_engine(tmp_path, monkeypatch):
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: _profile(model_root),
    )

    with pytest.raises(ValueError, match="No runnable audio.cpp version"):
        validate_audio_model_config(
            _Store(None),
            _model(model_root),
            _config(),
        )


def test_rejects_incompatible_engines_list(tmp_path, monkeypatch):
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
    model = _model(model_root)
    model["compatible_engines"] = ["llama_cpp"]

    with pytest.raises(ValueError, match="not verified compatible with audio.cpp"):
        validate_audio_model_config(_Store(active), model, _config())


def test_rejects_allow_scan_false_without_cached_profile(tmp_path, monkeypatch):
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
        "backend.audio_model_config.get_model_profile_entry",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="No cached audio.cpp model profile"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(),
            allow_scan=False,
        )


def test_scan_error_surfaces_in_validation(tmp_path, monkeypatch):
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
        lambda *args, **kwargs: {
            **_profile(model_root),
            "scan_error": "model help failed",
        },
    )

    with pytest.raises(ValueError, match="Model capability inspection failed"):
        validate_audio_model_config(_Store(active), _model(model_root), _config())


def test_rejects_load_options_not_object(tmp_path, monkeypatch):
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

    with pytest.raises(ValueError, match="load_options must be an object"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(load_options="bad"),
        )


def test_skips_voice_preset_validation_for_asr_task(tmp_path, monkeypatch):
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
    profile = _profile(model_root)
    profile["inspection"]["family"] = "nemotron_asr"
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: profile,
    )

    result = validate_audio_model_config(
        _Store(active),
        _model(model_root),
        _config(
            family="nemotron_asr",
            task="asr",
            mode="offline",
            default_voice_preset="missing",
            voice_presets={"assistant": {"voice_id": "M1"}},
        ),
    )
    assert result["errors"] == []


def test_rejects_invalid_omnivoice_instruct_attributes(tmp_path, monkeypatch):
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
    profile = _profile(model_root)
    profile["inspection"]["family"] = "omnivoice"
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: profile,
    )

    with pytest.raises(ValueError, match="Unsupported attribute"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(
                family="omnivoice",
                speech_defaults={
                    "instructions": "female, calm, kind, motherly tone",
                },
            ),
        )


def test_rejects_speech_defaults_on_vevo2_tts_task(tmp_path, monkeypatch):
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
    profile = _profile(model_root)
    profile["inspection"]["family"] = "vevo2"
    profile["inspection"]["tasks"] = [
        {"task": "tts", "modes": ["offline"]},
        {"task": "vc", "modes": ["offline"]},
    ]
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: profile,
    )

    with pytest.raises(ValueError, match="task_defaults"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(
                family="vevo2",
                task="tts",
                speech_defaults={"text": "Hello"},
            ),
        )


def test_rejects_voxcpm2_instructions_in_speech_defaults(tmp_path, monkeypatch):
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
    profile = _profile(model_root)
    profile["inspection"]["family"] = "voxcpm2"
    monkeypatch.setattr(
        "backend.audio_model_config.scan_audio_cpp_model_profile",
        lambda *args, **kwargs: profile,
    )

    with pytest.raises(ValueError, match="does not use instructions"):
        validate_audio_model_config(
            _Store(active),
            _model(model_root),
            _config(
                family="voxcpm2",
                speech_defaults={"instructions": "calm narrator"},
            ),
        )
