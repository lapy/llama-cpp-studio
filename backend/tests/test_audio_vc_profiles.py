"""Voice conversion profile tests."""

import pytest

from backend.audio.families.vc import (
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
    assert vc_profile_for_family(family) is None

@pytest.mark.parametrize("family", VC_FAMILIES)
def test_conversion_field_groups_are_well_formed(family):
    assert conversion_request_field_groups(family) == []

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
