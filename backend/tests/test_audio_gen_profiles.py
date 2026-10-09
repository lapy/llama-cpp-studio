"""Music/SFX generation profile tests."""

import pytest

from backend.audio.families.gen import (
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
    assert gen_profile_for_family(family) is None

@pytest.mark.parametrize("family", GEN_FAMILIES)
def test_generation_field_groups_are_well_formed(family):
    assert generation_request_field_groups(family) == []

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
