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
    groups = speech_request_field_groups("omnivoice")
    ids = [group["id"] for group in groups]
    assert "voice" in ids
    assert "design" in ids


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


def test_higgs_audio_tts_alias_matches_higgs_tts():
    primary = tts_profile_for_family("higgs_tts")
    alias = tts_profile_for_family("higgs_audio_tts")
    assert primary["label"] == alias["label"]


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
