"""Voice conversion profile tests."""

import pytest

from backend.audio_vc_profiles import (
    conversion_request_field_groups,
    is_vc_task,
    vc_profile_for_family,
)
from backend.tests.audio_profile_fixtures import (
    VC_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)


@pytest.mark.parametrize("family", VC_FAMILIES)
def test_vc_profile_exists_for_documented_family(family):
    profile = vc_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


@pytest.mark.parametrize("family", VC_FAMILIES)
def test_conversion_field_groups_are_well_formed(family):
    groups = conversion_request_field_groups(family)
    assert groups
    assert_field_groups_shape(groups)


def test_seed_vc_includes_route_and_f0_options():
    groups = conversion_request_field_groups("seed_vc")
    ids = [group["id"] for group in groups]
    assert "route" in ids
    assert "options" in ids
    option_keys = {
        field["key"]
        for group in groups
        if group["id"] == "options"
        for field in group["fields"]
    }
    assert {"f0_condition", "semi_tone_shift", "length_adjust"}.issubset(option_keys)


def test_miocodec_minimal_audio_only_fields():
    groups = conversion_request_field_groups("miocodec")
    ids = [group["id"] for group in groups]
    assert ids == ["audio"]
    field_keys = {field["key"] for field in groups[0]["fields"]}
    assert field_keys == {"audio", "voice_ref"}


def test_vevo2_includes_multi_role_audio_and_text_fields():
    groups = conversion_request_field_groups("vevo2")
    ids = [group["id"] for group in groups]
    assert "route" in ids
    assert "audio" in ids
    assert "text" in ids
    audio_keys = {
        field["key"]
        for group in groups
        if group["id"] == "audio"
        for field in group["fields"]
    }
    assert {"source_audio", "voice_ref", "prosody_ref", "style_ref"}.issubset(audio_keys)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("vc", True),
        ("svc", True),
        ("s2s", True),
        ("tts", False),
        ("asr", False),
    ],
)
def test_is_vc_task(task, expected):
    assert is_vc_task(task) is expected


def test_vevo2_profile_same_regardless_of_task():
    for task in ("tts", "vc", "s2s", "svc"):
        profile = vc_profile_for_family("vevo2")
        assert profile["label"] == "VeVo2"
