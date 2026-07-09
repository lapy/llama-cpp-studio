"""Music/SFX generation profile tests."""

import pytest

from backend.audio_gen_profiles import (
    gen_profile_for_family,
    generation_request_field_groups,
    is_gen_task,
)
from backend.tests.audio_profile_fixtures import (
    GEN_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)


@pytest.mark.parametrize("family", GEN_FAMILIES)
def test_gen_profile_exists_for_documented_family(family):
    profile = gen_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


@pytest.mark.parametrize("family", GEN_FAMILIES)
def test_generation_field_groups_are_well_formed(family):
    groups = generation_request_field_groups(family)
    assert groups
    assert_field_groups_shape(groups)


def test_ace_step_includes_route_and_repaint_fields():
    groups = generation_request_field_groups("ace_step")
    ids = [group["id"] for group in groups]
    assert "route" in ids
    assert "timing" in ids
    field_keys = {
        field["key"]
        for group in groups
        for field in group["fields"]
    }
    assert "task_route" in field_keys
    assert "repaint_start" in field_keys
    assert "repaint_end" in field_keys


def test_stable_audio_includes_conditioning_fields():
    groups = generation_request_field_groups("stable_audio")
    ids = [group["id"] for group in groups]
    assert "conditioning" in ids
    field_keys = {
        field["key"]
        for group in groups
        for field in group["fields"]
    }
    assert "audio_input_kind" in field_keys
    assert "inpaint_mask_start_seconds" in field_keys


def test_heartmula_includes_tags_and_infinite_mode():
    groups = generation_request_field_groups("heartmula")
    field_keys = {
        field["key"]
        for group in groups
        for field in group["fields"]
    }
    assert "tags" in field_keys
    assert "infinite_mode" in field_keys
    assert field_keys.intersection({"text", "lyrics"})


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("gen", True),
        ("GEN", True),
        ("tts", False),
        ("", False),
        (None, False),
    ],
)
def test_is_gen_task(task, expected):
    assert is_gen_task(task) is expected


def test_unknown_gen_family_returns_none():
    assert gen_profile_for_family("musicgen") is None
    assert generation_request_field_groups("musicgen") == []
