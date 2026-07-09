"""Voice preset validation edge cases."""

import pytest

from backend.audio_voice_presets import (
    audio_request_defaults_to_swap_set_params,
    normalize_default_voice_preset,
    normalize_voice_preset,
    normalize_voice_presets,
    validate_voice_presets,
)


def test_audio_request_defaults_to_swap_set_params_maps_tts_fields():
    params = audio_request_defaults_to_swap_set_params(
        {
            "engine": "audio_cpp",
            "task": "vdes",
            "speech_defaults": {
                "instructions": "warm adult narrator",
                "temperature": 0.7,
                "voice": "assistant",
                "options": {"speaker": "Vivian"},
            },
        }
    )
    assert params == {
        "instructions": "warm adult narrator",
        "temperature": 0.7,
        "voice": "assistant",
        "options": {"speaker": "Vivian"},
    }


def test_audio_request_defaults_to_swap_set_params_maps_transcription_fields():
    from backend.audio_voice_presets import audio_request_defaults_to_swap_set_params

    params = audio_request_defaults_to_swap_set_params(
        {
            "engine": "audio_cpp",
            "task": "asr",
            "family": "qwen3_asr",
            "transcription_defaults": {
                "language": "en",
                "stream": True,
                "prompt": "Transcribe clearly.",
                "options": {"num_beams": 4},
            },
        }
    )
    assert params == {
        "language": "en",
        "stream": True,
        "options": {"num_beams": 4, "text": "Transcribe clearly."},
    }


def test_audio_request_defaults_to_swap_set_params_maps_task_defaults():
    from backend.audio_voice_presets import audio_request_defaults_to_swap_set_params

    params = audio_request_defaults_to_swap_set_params(
        {
            "engine": "audio_cpp",
            "task": "gen",
            "family": "ace_step",
            "task_defaults": {
                "duration_seconds": 30.0,
                "guidance_scale": 7.5,
                "options": {"steps": 50},
            },
        }
    )
    assert params == {
        "duration_seconds": 30.0,
        "guidance_scale": 7.5,
        "options": {"steps": 50},
    }


def test_audio_request_defaults_to_swap_set_params_ignores_non_tts_tasks():
    assert audio_request_defaults_to_swap_set_params(
        {
            "engine": "audio_cpp",
            "task": "asr",
            "transcription_defaults": {"language": "en"},
        }
    ) == {
        "language": "en",
    }


def test_audio_request_defaults_to_swap_set_params_ignores_other_engines():
    assert audio_request_defaults_to_swap_set_params(
        {
            "engine": "llama_cpp",
            "task": "tts",
            "speech_defaults": {"instructions": "ignored"},
        }
    ) == {}


def test_normalize_voice_preset_requires_at_least_one_field():
    assert normalize_voice_preset({}, model_root="/tmp") is None
    assert normalize_voice_preset({"language": "en"}, model_root="/tmp") is None


def test_normalize_voice_preset_accepts_voice_id_only():
    out = normalize_voice_preset({"voice_id": "M1"}, model_root="/tmp")
    assert out == {"voice_id": "M1"}


def test_normalize_voice_presets_skips_empty_names_and_invalid_entries(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    presets = normalize_voice_presets(
        {
            "": {"voice_id": "x"},
            "valid": {"voice_id": "M1"},
            "invalid": "not-a-dict",
        },
        model_root=str(model_root),
    )
    assert list(presets.keys()) == ["valid"]


def test_validate_voice_presets_missing_named_default(tmp_path):
    errors: list[str] = []
    validate_voice_presets(
        {
            "voice_presets": {"assistant": {"voice_id": "M1"}},
            "default_voice_preset": "missing",
        },
        model_root=str(tmp_path),
        errors=errors,
    )
    assert any("not defined in voice_presets" in err for err in errors)


def test_validate_voice_presets_missing_voice_ref_file(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    errors: list[str] = []
    validate_voice_presets(
        {
            "voice_presets": {
                "clone": {"voice_ref": "missing.wav"},
            }
        },
        model_root=str(model_root),
        errors=errors,
    )
    assert any("voice_ref does not exist" in err for err in errors)


def test_validate_voice_presets_inline_default_object_missing_fields(tmp_path):
    errors: list[str] = []
    validate_voice_presets(
        {"default_voice_preset": {"language": "en"}},
        model_root=str(tmp_path),
        errors=errors,
    )
    assert any("must include voice_id, voice_ref, or reference_text" in err for err in errors)


def test_normalize_speech_defaults_resolves_voice_ref_paths(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    wav = model_root / "refs" / "voice.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"RIFF")
    from backend.audio_voice_presets import normalize_speech_defaults

    out = normalize_speech_defaults(
        {
            "voice_ref": "refs/voice.wav",
            "temperature": "0.8",
            "max_tokens": "256",
            "options": {"speaker": "Vivian"},
        }
    )
    assert out["voice_ref"] == "refs/voice.wav"
    assert out["temperature"] == 0.8
    assert out["max_tokens"] == 256
    assert out["options"]["speaker"] == "Vivian"


def test_task_swap_params_merges_prompt_into_existing_options():
    from backend.audio_voice_presets import _task_normalized_to_swap_params

    params = _task_normalized_to_swap_params(
        {
            "text": "generate music",
            "prompt": "upbeat chorus",
            "options": {"task_route": "text2music", "text": "old"},
        }
    )
    assert params["text"] == "generate music"
    assert params["options"]["task_route"] == "text2music"
    assert params["options"]["text"] == "upbeat chorus"


def test_audio_request_defaults_to_swap_set_params_for_clon_task():
    params = audio_request_defaults_to_swap_set_params(
        {
            "engine": "audio_cpp",
            "task": "clon",
            "family": "chatterbox",
            "speech_defaults": {"voice": "assistant", "temperature": 0.5},
        }
    )
    assert params == {"voice": "assistant", "temperature": 0.5}


def test_validate_voice_presets_reference_text_only_preset_is_valid(tmp_path):
    errors: list[str] = []
    validate_voice_presets(
        {
            "voice_presets": {
                "clone": {"reference_text": "Hello world."},
            }
        },
        model_root=str(tmp_path),
        errors=errors,
    )
    assert errors == []


def test_normalize_speech_defaults_ignores_invalid_numeric_fields():
    from backend.audio_voice_presets import normalize_speech_defaults

    out = normalize_speech_defaults(
        {"temperature": "hot", "max_tokens": "many", "voice": "assistant"}
    )
    assert out == {"voice": "assistant"}


def test_normalize_default_voice_preset_inline_object(tmp_path):
    wav = tmp_path / "ref.wav"
    wav.write_bytes(b"RIFF")
    out = normalize_default_voice_preset(
        {"voice_ref": str(wav)},
        model_root=str(tmp_path),
    )
    assert out["voice_ref"] == str(wav.resolve())
