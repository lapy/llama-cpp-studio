"""Forced alignment profile tests."""

import pytest

from backend.audio.families.align import (
    align_profile_for_family,
    alignment_request_field_groups,
    is_align_task,
)
from backend.tests.audio_profile_fixtures import (
    ALIGN_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)

@pytest.mark.parametrize("family", ALIGN_FAMILIES)
def test_align_profile_exists_for_documented_family(family):
    assert align_profile_for_family(family) is None

@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("align", True),
        ("asr", False),
        ("tts", False),
    ],
)
def test_is_align_task(task, expected):
    assert is_align_task(task) is expected
