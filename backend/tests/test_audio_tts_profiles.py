"""Voice preset and TTS profile helpers."""

import json

import pytest

from backend.audio.families.tts import (
    is_tts_task,
    speech_request_field_groups,
    tts_profile_for_family,
)
from backend.audio.voice_presets import (
    normalize_default_voice_preset,
    normalize_voice_preset,
    normalize_voice_presets,
)
from backend.tests.audio_profile_fixtures import (
    TTS_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)

@pytest.mark.parametrize("family", TTS_FAMILIES)
def test_tts_profile_exists_for_documented_family(family):
    assert tts_profile_for_family(family) is None

@pytest.mark.parametrize("family", TTS_FAMILIES)
def test_speech_field_groups_are_well_formed(family):
    assert speech_request_field_groups(family) == []

def test_installed_model_spec_types_request_fields_for_any_family(tmp_path):
    from backend.audio.task_profiles import request_field_groups_for
    from backend.engines.audio_cpp.spec_fields import merge_spec_contract_into_sections

    spec_dir = tmp_path / "model_specs"
    spec_dir.mkdir()
    (spec_dir / "kokoro_tts.json").write_text(
        json.dumps(
            {
                "family": "kokoro_tts",
                "tasks": ["tts"],
                "options": {
                    "request": [
                        {
                            "name": "speed",
                            "type": "float",
                            "description": "Playback speed from the installed tree.",
                            "default": 1.0,
                        }
                    ],
                    "session": [
                        {
                            "name": "reference_duration_sec",
                            "type": "float",
                            "description": "Trim the speaker reference.",
                            "default": 15,
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    (spec_dir / "canary_asr.json").write_text(
        json.dumps(
            {
                "family": "canary_asr",
                "tasks": ["asr"],
                "options": {
                    "request": [
                        {
                            "name": "audio_chunk_duration_sec",
                            "type": "float",
                            "description": "Chunk length from the installed tree.",
                            "min": 1,
                            "max": 30,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )

    speech = request_field_groups_for("tts", "kokoro_tts", source_path=str(tmp_path))
    speech_fields = {
        field["key"]: field for group in speech for field in group["fields"]
    }
    assert speech_fields["speed"]["type"] == "float"
    assert speech_fields["speed"]["hint"] == "Playback speed from the installed tree."

    asr = request_field_groups_for("asr", "canary_asr", source_path=str(tmp_path))
    asr_fields = {field["key"]: field for group in asr for field in group["fields"]}
    assert asr_fields["audio_chunk_duration_sec"]["type"] == "float"
    assert asr_fields["audio_chunk_duration_sec"]["maximum"] == 30

    sections = merge_spec_contract_into_sections(
        [
            {
                "params": [
                    {
                        "key": "kokoro_tts.reference_duration_sec",
                        "scope": "session_option",
                        "type": "string",
                    }
                ]
            }
        ],
        {
            "family": "kokoro_tts",
            "options": {
                "session": [
                    {"name": "reference_duration_sec", "type": "float", "default": 15}
                ]
            },
        },
    )
    session = next(
        param
        for section in sections
        for param in section["params"]
        if param["key"] == "kokoro_tts.reference_duration_sec"
    )
    assert session["type"] == "float"
    assert session["scope"] == "session_option"

def test_echo_duration_hint_comes_from_model_spec(tmp_path):
    from backend.audio.task_profiles import request_field_groups_for

    spec_dir = tmp_path / "model_specs"
    spec_dir.mkdir()
    (spec_dir / "echo_tts.json").write_text(
        json.dumps(
            {
                "family": "echo_tts",
                "display_name": "Echo-TTS",
                "description": "English zero-shot cloning from the installed spec.",
                "tasks": ["clone"],
                "options": {
                    "request": [
                        {
                            "name": "max_duration_sec",
                            "type": "float",
                            "description": "Cap the generation window, up to 29.7215 s.",
                            "max": 29.7215,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    groups = request_field_groups_for("clon", "echo_tts", source_path=str(tmp_path))
    fields = {field["key"]: field for group in groups for field in group["fields"]}
    assert fields["max_duration_sec"]["type"] == "float"
    assert "29.7215" in fields["max_duration_sec"]["hint"]
    assert fields["max_duration_sec"]["maximum"] == 29.7215

@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("tts", True),
        ("clon", True),
        ("vdes", True),
        ("vc", False),
        ("svc", False),
        ("s2s", False),
        ("asr", False),
        ("gen", False),
    ],
)
def test_is_tts_task(task, expected):
    assert is_tts_task(task) is expected

def test_normalize_voice_presets_resolve_relative_paths(tmp_path):
    model_root = tmp_path / "bundle"
    model_root.mkdir()
    wav = model_root / "refs" / "voice.wav"
    wav.parent.mkdir()
    wav.write_bytes(b"RIFF")
    presets = normalize_voice_presets(
        {
            "assistant": {
                "voice_ref": "refs/voice.wav",
                "reference_text": "Hello there.",
            }
        },
        model_root=str(model_root),
    )
    assert presets["assistant"]["voice_ref"] == str(wav.resolve())
    assert presets["assistant"]["reference_text"] == "Hello there."

def test_normalize_default_voice_preset_accepts_named_preset():
    assert (
        normalize_default_voice_preset("assistant", model_root="/tmp")
        == "assistant"
    )

def test_unknown_tts_family_returns_empty_groups():
    assert tts_profile_for_family("unknown_tts") is None
    assert speech_request_field_groups("unknown_tts") == []

def test_installed_spec_replaces_studio_profile_prose(tmp_path):
    from backend.audio.task_profiles import overlay_task_profile_from_spec

    spec_dir = tmp_path / "model_specs"
    spec_dir.mkdir()
    (spec_dir / "pocket_tts.json").write_text(
        json.dumps(
            {
                "family": "pocket_tts",
                "schema_version": 1,
                "display_name": "PocketTTS",
                "description": "Kyutai package set from the installed spec.",
                "tasks": ["tts", "clone"],
                "ui": {"default_voice": "alba", "builtin_voices": ["alba"]},
            }
        ),
        encoding="utf-8",
    )
    overlaid = overlay_task_profile_from_spec(
        tts_profile_for_family("pocket_tts"),
        str(tmp_path),
        "pocket_tts",
    )
    assert overlaid["summary"] == "Kyutai package set from the installed spec."
    assert "api_hint" not in overlaid
    assert overlaid["default_voice"] == "alba"
    assert overlaid["builtin_voices"] == ["alba"]
    assert overlaid["workflows"] == ["tts", "clone"]
