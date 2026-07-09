"""Unified task profile facade tests."""

import pytest

from backend.audio_task_profiles import (
    api_endpoint_for,
    api_example_hint_for,
    is_profiled_task,
    request_defaults_key_for,
    request_field_groups_for,
    task_profile_for,
)
from backend.tests.audio_profile_fixtures import (
    DOC_PROFILED_FAMILIES,
    UNKNOWN_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)


@pytest.mark.parametrize(
    ("task", "family", "defaults_key", "endpoint"),
    DOC_PROFILED_FAMILIES,
)
def test_documented_family_profile_contract(task, family, defaults_key, endpoint):
    assert is_profiled_task(task, family)
    profile = task_profile_for(task, family)
    assert profile is not None
    assert_profile_shape(profile)

    groups = request_field_groups_for(task, family)
    assert groups
    assert_field_groups_shape(groups)

    assert request_defaults_key_for(task, family) == defaults_key
    assert api_endpoint_for(task, family) == endpoint
    assert api_example_hint_for(task, family)


@pytest.mark.parametrize("task,family", [(t, f) for t, f, *_ in UNKNOWN_FAMILIES])
def test_unknown_families_are_not_profiled(task, family):
    assert not is_profiled_task(task, family)
    assert task_profile_for(task, family) is None
    assert request_field_groups_for(task, family) == []


def test_family_case_insensitive():
    profile = task_profile_for("tts", "KOKORO_TTS")
    assert profile is not None
    assert profile["label"] == "Kokoro"


def test_vevo2_uses_vc_profile_even_for_tts_task():
    profile = task_profile_for("tts", "vevo2")
    assert profile["label"] == "VeVo2"
    assert request_defaults_key_for("tts", "vevo2") == "task_defaults"
    assert api_endpoint_for("tts", "vevo2") == "/v1/tasks/run"


def test_citrinet_alias_matches_citrinet_asr():
    primary = task_profile_for("asr", "citrinet_asr")
    alias = task_profile_for("asr", "citrinet")
    assert primary["label"] == alias["label"]


def test_hviske_alias_matches_hviske_asr():
    primary = task_profile_for("asr", "hviske_asr")
    alias = task_profile_for("asr", "hviske")
    assert primary["summary"] == alias["summary"]


def test_api_example_hint_for_speech_endpoint():
    hint = api_example_hint_for("tts", "kokoro_tts")
    assert "speech" in hint.lower()


def test_api_example_hint_for_transcription_endpoint():
    hint = api_example_hint_for("asr", "nemotron_asr")
    assert "multipart" in hint.lower() or "audio path" in hint.lower()


def test_api_example_hint_for_generic_tasks_run():
    hint = api_example_hint_for("gen", "ace_step")
    assert "/v1/tasks/run" in hint


@pytest.mark.parametrize(
    ("task", "family"),
    [
        ("tts", "omnivoice"),
        ("asr", "nemotron_asr"),
        ("gen", "heartmula"),
        ("vc", "seed_vc"),
        ("vad", "silero_vad"),
        ("sep", "htdemucs"),
        ("align", "qwen3_forced_aligner"),
    ],
)
def test_backward_compatible_field_groups_non_empty(task, family):
    groups = request_field_groups_for(task, family)
    assert len(groups) >= 1


def test_empty_task_and_family_not_profiled():
    assert not is_profiled_task(None, None)
    assert not is_profiled_task("", "")
