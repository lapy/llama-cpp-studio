"""Voice preset and TTS profile helpers."""

import pytest

from backend.audio_tts_profiles import (
    is_tts_task,
    speech_request_field_groups,
    tts_profile_for_family,
)
from backend.audio_voice_presets import (
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
    profile = tts_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


@pytest.mark.parametrize("family", TTS_FAMILIES)
def test_speech_field_groups_are_well_formed(family):
    groups = speech_request_field_groups(family)
    assert groups
    assert_field_groups_shape(groups)


def test_omnivoice_includes_voice_and_design_groups():
    profile = tts_profile_for_family("omnivoice")
    assert "clone" in profile["workflows"]
    assert profile["supports_instructions"] is True
    assert profile["instructions_style"] == "omnivoice_attributes"
    groups = speech_request_field_groups("omnivoice")
    ids = [group["id"] for group in groups]
    assert "voice" in ids
    assert "design" in ids
    design = next(group for group in groups if group["id"] == "design")
    assert design["fields"][0]["hint"]


def test_chatterbox_voice_clone_fields():
    groups = speech_request_field_groups("chatterbox")
    voice_fields = {
        field["key"]
        for group in groups
        if group["id"] == "voice"
        for field in group["fields"]
    }
    assert "voice_ref" in voice_fields


def test_vibevoice_multi_speaker_voice_samples():
    groups = speech_request_field_groups("vibevoice")
    voice_fields = {
        field["key"]
        for group in groups
        if group["id"] == "voice"
        for field in group["fields"]
    }
    assert "voice_samples" in voice_fields


def test_qwen3_tts_includes_speaker_and_subtalker_options():
    groups = speech_request_field_groups("qwen3_tts")
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "speaker" in field_keys
    assert "subtalker_temperature" in field_keys


def test_irodori_tts_includes_no_ref_and_caption_options():
    groups = speech_request_field_groups("irodori_tts")
    option_keys = {
        field["key"]
        for group in groups
        if group["id"] == "options"
        for field in group["fields"]
    }
    assert {"no_ref", "caption"}.issubset(option_keys)


def test_supertonic_preset_voice_fields():
    groups = speech_request_field_groups("supertonic")
    voice_fields = {
        field["key"]
        for group in groups
        if group["id"] == "voice"
        for field in group["fields"]
    }
    assert "voice_id" in voice_fields


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("tts", True),
        ("clon", True),
        ("vdes", True),
        ("vc", True),
        ("svc", True),
        ("s2s", True),
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


def test_kokoro_tts_preset_voice_id_field():
    groups = speech_request_field_groups("kokoro_tts")
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "voice_id" in field_keys


def test_higgs_audio_tts_clone_workflow_fields():
    profile = tts_profile_for_family("higgs_audio_tts")
    assert "clone" in profile["workflows"]
    groups = speech_request_field_groups("higgs_audio_tts")
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "voice_ref" in field_keys


def test_pocket_tts_dual_voice_fields():
    groups = speech_request_field_groups("pocket_tts")
    voice_fields = {
        field["key"]
        for group in groups
        if group["id"] == "voice"
        for field in group["fields"]
    }
    assert {"voice_id", "voice_ref"}.issubset(voice_fields)


def test_voxcpm2_generation_fields_include_guidance_and_max_tokens():
    groups = speech_request_field_groups("voxcpm2")
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert {"guidance_scale", "max_tokens"}.issubset(field_keys)


def test_moss_tts_reference_text_optional_field():
    groups = speech_request_field_groups("moss_tts")
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "reference_text" in field_keys
