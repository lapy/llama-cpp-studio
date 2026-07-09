"""Forced alignment profile tests."""

import pytest

from backend.audio_align_profiles import (
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
    profile = align_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


def test_qwen3_forced_aligner_requires_transcript_and_language():
    groups = alignment_request_field_groups("qwen3_forced_aligner")
    ids = [group["id"] for group in groups]
    assert "audio" in ids
    assert "context" in ids
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "transcript" in field_keys
    assert "language" in field_keys
    assert_field_groups_shape(groups)


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
