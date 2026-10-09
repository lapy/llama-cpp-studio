"""VAD and diarization profile tests."""

import pytest

from backend.audio.families.analysis import (
    analysis_profile_for_family,
    analysis_request_field_groups,
    is_analysis_task,
    is_diar_task,
    is_vad_task,
)
from backend.tests.audio_profile_fixtures import (
    ANALYSIS_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)

@pytest.mark.parametrize("family", ANALYSIS_FAMILIES)
def test_analysis_profile_exists_for_documented_family(family):
    assert analysis_profile_for_family(family) is None

@pytest.mark.parametrize("family", ANALYSIS_FAMILIES)
def test_analysis_field_groups_are_well_formed(family):
    assert analysis_request_field_groups(family) == []

@pytest.mark.parametrize(
    ("task", "fn", "expected"),
    [
        ("vad", is_vad_task, True),
        ("diar", is_diar_task, True),
        ("asr", is_vad_task, False),
        ("vad", is_analysis_task, True),
        ("diar", is_analysis_task, True),
        ("gen", is_analysis_task, False),
    ],
)
def test_analysis_task_classifiers(task, fn, expected):
    assert fn(task) is expected
