"""OmniVoice instruct attribute validation."""

import pytest

from backend.audio_omnivoice_instruct import (
    OMNIVOICE_INSTRUCT_EXAMPLE,
    parse_omnivoice_instruct_items,
    unsupported_omnivoice_instruct_items,
    validate_omnivoice_instruct,
)
from backend.audio_tts_profiles import speech_request_field_groups


def test_parse_omnivoice_instruct_items_splits_on_commas():
    assert parse_omnivoice_instruct_items(
        "female, calm, kind, motherly tone"
    ) == ["female", "calm", "kind", "motherly tone"]


def test_unsupported_omnivoice_instruct_items_flags_unknown_tokens():
    unsupported = unsupported_omnivoice_instruct_items(
        "female, calm, kind, motherly tone"
    )
    assert unsupported == ["calm", "kind", "motherly tone"]


def test_validate_omnivoice_instruct_accepts_canonical_attributes():
    assert validate_omnivoice_instruct(OMNIVOICE_INSTRUCT_EXAMPLE) == []


def test_validate_omnivoice_instruct_rejects_free_form_prose():
    errors = validate_omnivoice_instruct("female, calm, kind, motherly tone")
    assert len(errors) == 1
    assert "Unsupported attribute(s)" in errors[0]
    assert "'calm'" in errors[0]
    assert "not free-form prose" in errors[0]


def test_omnivoice_field_group_uses_attribute_hint_not_natural_language():
    groups = speech_request_field_groups("omnivoice")
    design = next(group for group in groups if group["id"] == "design")
    assert "comma-separated" in design["description"].lower()
    assert "natural-language" not in design["description"].lower()
    field = design["fields"][0]
    assert field["label"] == "Voice design attributes (instruct)"
    assert field["hint"]
    assert "not natural-language prose" in field["hint"].lower()


def test_qwen3_tts_keeps_natural_language_instructions_group():
    groups = speech_request_field_groups("qwen3_tts")
    design = next(group for group in groups if group["id"] == "design")
    assert "Natural-language" in design["description"]
    assert "natural-language" in design["fields"][0]["hint"].lower()
