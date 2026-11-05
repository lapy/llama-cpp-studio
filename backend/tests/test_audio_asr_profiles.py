"""ASR profile and transcription default helpers."""

import json

import pytest

from backend.audio.families.asr import (
    asr_profile_for_family,
    is_asr_task,
    transcription_request_field_groups,
)
from backend.audio.transcription_defaults import normalize_transcription_defaults
from backend.tests.audio_profile_fixtures import (
    ASR_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)

@pytest.mark.parametrize("family", ASR_FAMILIES)
def test_asr_profile_exists_for_documented_family(family):
    assert asr_profile_for_family(family) is None

@pytest.mark.parametrize("family", ASR_FAMILIES)
def test_transcription_field_groups_are_well_formed(family):
    assert transcription_request_field_groups(family) == []

def test_qwen3_asr_sidecar_session_fields_come_from_model_spec(tmp_path):
    from backend.audio.task_profiles import sidecar_session_fields_for

    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "qwen3_asr",
                "category": "asr",
                "tasks": ["asr"],
                "modes": ["offline"],
                "dependencies": [
                    {
                        "kind": "model",
                        "family": "qwen3_forced_aligner",
                        "scope": "session",
                        "option": "forced_aligner_path",
                        "required": False,
                        "required_when": [
                            {
                                "scope": "request",
                                "option_key": "return_timestamps",
                                "equals": True,
                            }
                        ],
                    },
                    {
                        "kind": "bundled_model",
                        "family": "silero_vad",
                        "path": "assets/framework/models/silero_vad",
                        "scope": "session",
                        "option": "vad_path",
                        "required": False,
                        "required_when": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    fields = sidecar_session_fields_for("asr", "qwen3_asr", source_path=str(tmp_path))
    keys = {field["key"] for field in fields}
    assert "qwen3_asr.forced_aligner_path" in keys
    assert "qwen3_asr.vad_path" in keys
    assert all(field.get("scope") == "session_option" for field in fields)

def test_normalize_transcription_defaults_maps_prompt_and_options():
    defaults = normalize_transcription_defaults(
        {
            "language": "en-US",
            "stream": True,
            "prompt": "Transcribe the speech.",
            "options": {
                "lookahead_tokens": "4",
                "keep_language_tags": False,
            },
        }
    )
    assert defaults["language"] == "en-US"
    assert defaults["stream"] is True
    assert defaults["prompt"] == "Transcribe the speech."
    assert defaults["options"]["lookahead_tokens"] == "4"
    assert defaults["options"]["keep_language_tags"] is False

def test_normalize_transcription_defaults_ignores_invalid_ints():
    out = normalize_transcription_defaults({"max_tokens": "many"})
    assert "max_tokens" not in out

@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("asr", True),
        ("tts", False),
        ("align", False),
    ],
)
def test_is_asr_task(task, expected):
    assert is_asr_task(task) is expected
