"""VAD and diarization profile tests."""

import pytest

from backend.audio_analysis_profiles import (
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
    profile = analysis_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


@pytest.mark.parametrize("family", ANALYSIS_FAMILIES)
def test_analysis_field_groups_are_well_formed(family):
    groups = analysis_request_field_groups(family)
    assert groups
    assert_field_groups_shape(groups)


def test_silero_vad_includes_streaming_and_chunk_controls():
    groups = analysis_request_field_groups("silero_vad")
    ids = [group["id"] for group in groups]
    assert "session" in ids
    assert "chunking" in ids
    assert "options" in ids
    option_keys = {
        field["key"]
        for group in groups
        if group["id"] == "options"
        for field in group["fields"]
    }
    assert "threshold" in option_keys
    assert "min_speech_duration_ms" in option_keys


def test_marblenet_vad_minimal_threshold_only():
    groups = analysis_request_field_groups("marblenet_vad")
    ids = [group["id"] for group in groups]
    assert ids == ["audio", "options"]
    option_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "threshold" in option_keys


def test_sortformer_diar_audio_only():
    groups = analysis_request_field_groups("sortformer_diar")
    assert len(groups) == 1
    assert groups[0]["id"] == "audio"


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
