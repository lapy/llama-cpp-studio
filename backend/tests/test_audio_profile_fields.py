"""Shared audio profile field spec tests."""

import pytest

from backend.audio_profile_fields import FIELD_SPECS, field_spec


@pytest.mark.parametrize(
    "key",
    [
        "text",
        "lyrics",
        "tags",
        "language",
        "prompt",
        "transcript",
        "voice_ref",
        "source_audio",
        "task_route",
        "duration_seconds",
        "temperature",
        "audio_chunk_mode",
        "threshold",
        "use_pitch_shift",
    ],
)
def test_field_spec_returns_known_keys(key):
    spec = field_spec(key)
    assert spec["key"] == key
    assert spec["label"]
    assert spec["type"]


def test_field_spec_nested_marks_options():
    spec = field_spec("track_name", nested=True)
    assert spec["nested"] is True
    assert spec["options_key"] == "track_name"


def test_field_spec_unknown_key_falls_back_to_string():
    spec = field_spec("custom_future_option")
    assert spec["key"] == "custom_future_option"
    assert spec["type"] == "string"


def test_prompt_field_maps_to_nested_text():
    spec = FIELD_SPECS["prompt"]
    assert spec["nested"] is True
    assert spec["options_key"] == "text"


def test_transcript_field_maps_to_request_text():
    spec = FIELD_SPECS["transcript"]
    assert spec["request_field"] == "text"


@pytest.mark.parametrize("key", sorted(FIELD_SPECS.keys()))
def test_all_field_specs_have_required_keys(key):
    spec = FIELD_SPECS[key]
    assert "key" in spec
    assert "label" in spec
    assert "type" in spec


def test_build_field_groups_from_profile_keys():
    from backend.audio_profile_fields import build_field_groups

    groups = build_field_groups(
        {},
        [
            ("voice", "Voice", "Voice fields", ["voice_id", "voice_ref"]),
            ("decode", "Decode", "Decode fields", ["temperature", "num_beams"]),
        ],
    )
    assert len(groups) == 2
    assert groups[0]["id"] == "voice"
    assert {field["key"] for field in groups[0]["fields"]} == {"voice_id", "voice_ref"}
    assert groups[1]["fields"][0]["key"] == "temperature"
