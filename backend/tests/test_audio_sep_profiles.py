"""Source separation profile tests."""

import pytest

from backend.audio_sep_profiles import (
    is_sep_task,
    sep_profile_for_family,
    separation_request_field_groups,
)
from backend.tests.audio_profile_fixtures import (
    SEP_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)


@pytest.mark.parametrize("family", SEP_FAMILIES)
def test_sep_profile_exists_for_documented_family(family):
    profile = sep_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


@pytest.mark.parametrize("family", SEP_FAMILIES)
def test_separation_field_groups_include_audio_input(family):
    groups = separation_request_field_groups(family)
    assert len(groups) == 1
    assert groups[0]["id"] == "audio"
    assert groups[0]["fields"][0]["key"] == "audio"
    assert_field_groups_shape(groups)


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("sep", True),
        ("gen", False),
        ("vc", False),
    ],
)
def test_is_sep_task(task, expected):
    assert is_sep_task(task) is expected
