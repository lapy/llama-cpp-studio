"""Family-specific request defaults validation."""

import pytest

from backend.audio_gen_profiles import generation_request_field_groups
from backend.audio_request_defaults_validation import validate_saved_request_defaults
from backend.audio_tts_profiles import speech_request_field_groups


@pytest.mark.parametrize(
    ("family", "task", "config", "message"),
    [
        (
            "vevo2",
            "tts",
            {"speech_defaults": {"voice": "assistant"}},
            "task_defaults",
        ),
        (
            "seed_vc",
            "tts",
            {"speech_defaults": {"temperature": 0.7}},
            "task_defaults",
        ),
        (
            "voxcpm2",
            "tts",
            {"speech_defaults": {"instructions": "calm narrator"}},
            "does not use instructions",
        ),
        (
            "irodori_tts",
            "tts",
            {"speech_defaults": {"instructions": "soft voice"}},
            "options.caption",
        ),
        (
            "omnivoice",
            "tts",
            {"speech_defaults": {"instructions": "female, calm, kind"}},
            "Unsupported attribute",
        ),
        (
            "heartmula",
            "gen",
            {"speech_defaults": {"text": "unused"}},
            "task_defaults",
        ),
    ],
)
def test_validate_saved_request_defaults_rejects_mismatched_families(
    family, task, config, message
):
    errors = validate_saved_request_defaults(task=task, family=family, config=config)
    assert errors
    assert any(message in err for err in errors)


def test_validate_saved_request_defaults_accepts_qwen3_natural_language_instructions():
    errors = validate_saved_request_defaults(
        task="vdes",
        family="qwen3_tts",
        config={
            "speech_defaults": {
                "instructions": "A calm, kind, motherly narrator with gentle pacing",
            }
        },
    )
    assert errors == []


def test_validate_saved_request_defaults_accepts_omnivoice_canonical_attributes():
    errors = validate_saved_request_defaults(
        task="tts",
        family="omnivoice",
        config={
            "speech_defaults": {
                "instructions": "female, young adult, moderate pitch, british accent",
            }
        },
    )
    assert errors == []


def test_validate_saved_request_defaults_accepts_heartmula_task_defaults_tags():
    errors = validate_saved_request_defaults(
        task="gen",
        family="heartmula",
        config={"task_defaults": {"tags": "pop, upbeat, drums, bright"}},
    )
    assert errors == []


def test_validate_rejects_speech_defaults_when_inspect_routes_to_tasks_run():
    """Inspection/help signals override family heuristics for endpoint/defaults key."""
    errors = validate_saved_request_defaults(
        task="tts",
        family="qwen3_tts",
        config={"speech_defaults": {"temperature": 0.7}},
        inspection={"tasks": [{"task": "tts"}, {"task": "vc"}]},
        model_profile={
            "sections": [
                {
                    "params": [
                        {"name": "task-route"},
                        {"name": "source-audio"},
                    ]
                }
            ]
        },
    )
    assert errors
    assert any("task_defaults" in err for err in errors)


def test_validate_accepts_task_defaults_when_inspect_routes_to_tasks_run():
    errors = validate_saved_request_defaults(
        task="tts",
        family="custom_family",
        config={"task_defaults": {"text": "hello", "options": {"task_route": "tts"}}},
        inspection={"tasks": [{"task": "tts"}, {"task": "vc"}]},
        model_profile={
            "params": [
                {"name": "task-route"},
                {"name": "source-audio"},
            ]
        },
    )
    assert errors == []


def test_heartmula_tags_field_includes_freeform_hint():
    groups = generation_request_field_groups("heartmula")
    tags_field = next(
        field
        for group in groups
        if group["id"] == "prompt"
        for field in group["fields"]
        if field["key"] == "tags"
    )
    assert "free-form" in tags_field["hint"].lower()
    assert "omnivoice" in tags_field["hint"].lower()


def test_irodori_caption_field_includes_hint():
    groups = speech_request_field_groups("irodori_tts")
    caption_field = next(
        field
        for group in groups
        if group["id"] == "options"
        for field in group["fields"]
        if field["key"] == "caption"
    )
    assert "caption" in caption_field["hint"].lower()
    assert "instructions" in caption_field["hint"].lower()


def test_qwen3_design_field_uses_natural_language_hint():
    groups = speech_request_field_groups("qwen3_tts")
    design = next(group for group in groups if group["id"] == "design")
    field = design["fields"][0]
    assert "natural-language" in field["hint"].lower()
    assert "omnivoice" in field["hint"].lower()
