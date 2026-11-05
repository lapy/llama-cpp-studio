"""Unified task profile facade tests."""

import pytest

from backend.audio.families.align import is_align_task
from backend.audio.families.analysis import is_analysis_task, is_diar_task, is_vad_task
from backend.audio.families.gen import is_gen_task
from backend.audio.families.sep import is_sep_task
from backend.audio.families.vc import is_vc_task
from backend.audio.families.asr import is_asr_task
from backend.audio.families.tts import is_tts_task

from backend.audio.task_profiles import (
    api_endpoint_for,
    api_example_hint_for,
    is_profiled_task,
    request_defaults_key_for,
    request_field_groups_for,
    task_profile_for,
)
from backend.tests.audio_profile_fixtures import (
    AUDIO_TASK_CONTRACTS,
    assert_profile_shape,
)


@pytest.mark.parametrize(
    ("task", "family", "defaults_key", "endpoint"),
    AUDIO_TASK_CONTRACTS,
)
def test_task_profile_contract_without_a_family_catalog(task, family, defaults_key, endpoint):
    assert is_profiled_task(task, family)
    profile = task_profile_for(task, family)
    assert profile is not None
    assert_profile_shape(profile)

    assert request_field_groups_for(task, family) == []
    assert profile.get("generic") is True

    assert request_defaults_key_for(task, family) == defaults_key
    assert api_endpoint_for(task, family) == endpoint
    assert api_example_hint_for(task, family)


def test_family_case_insensitive():
    profile = task_profile_for("tts", "POCKET_TTS")
    assert profile is not None
    assert profile["generic"] is True
    assert profile["label"] == task_profile_for("tts", "pocket_tts")["label"]


def test_task_selects_the_route_without_a_family_catalog():
    assert request_defaults_key_for("tts", "vevo2") == "speech_defaults"
    assert api_endpoint_for("tts", "vevo2") == "/v1/audio/speech"
    assert request_defaults_key_for("vc", "vevo2") == "task_defaults"
    assert api_endpoint_for("vc", "vevo2") == "/audioapi/v1/tasks/run"


def test_api_example_hint_for_speech_endpoint():
    hint = api_example_hint_for("tts", "pocket_tts")
    assert "speech" in hint.lower()


def test_api_example_hint_for_transcription_endpoint():
    hint = api_example_hint_for("asr", "nemotron_asr")
    assert "multipart" in hint.lower() or "audio path" in hint.lower()


def test_merge_request_field_groups_prefers_scanned():
    from backend.audio.task_profiles import merge_request_field_groups

    merged = merge_request_field_groups(
        [
            {
                "id": "curated",
                "label": "Curated",
                "fields": [
                    {"key": "text", "label": "Text"},
                    {"key": "engine_only", "label": "Should keep"},
                ],
            }
        ],
        [
            {
                "id": "scanned",
                "label": "Scanned",
                "fields": [
                    {"key": "temperature", "label": "Temperature"},
                    {"key": "text", "label": "Engine Text"},
                ],
            }
        ],
    )
    assert merged[0]["id"] == "scanned"
    keys = [field["key"] for group in merged for field in group["fields"]]
    assert keys[0] == "temperature"
    assert "text" in keys
    assert keys.count("text") == 1
    assert "engine_only" in keys


def test_unknown_family_with_scanned_sections_gets_fields():
    groups = request_field_groups_for(
        "tts",
        "brand_new_tts",
        profile_sections=[
            {
                "id": "request",
                "params": [
                    {
                        "key": "temperature",
                        "label": "Temperature",
                        "scope": "request_option",
                        "type": "float",
                    }
                ],
            }
        ],
    )
    assert groups
    assert any(
        field.get("key") == "temperature" for group in groups for field in group.get("fields") or []
    )


def test_api_example_hint_for_generic_tasks_run():
    hint = api_example_hint_for("gen", "ace_step")
    assert "/audioapi/v1/tasks/run" in hint


def test_clon_and_vdes_use_speech_endpoint_and_defaults():
    assert request_defaults_key_for("clon", "chatterbox") == "speech_defaults"
    assert request_defaults_key_for("vdes", "qwen3_tts") == "speech_defaults"
    assert api_endpoint_for("clon", "chatterbox") == "/v1/audio/speech"
    assert api_endpoint_for("vdes", "qwen3_tts") == "/v1/audio/speech"


def test_svc_and_s2s_use_task_defaults_and_tasks_run():
    assert request_defaults_key_for("svc", "seed_vc") == "task_defaults"
    assert request_defaults_key_for("s2s", "vevo2") == "task_defaults"
    assert api_endpoint_for("svc", "seed_vc") == "/audioapi/v1/tasks/run"
    assert api_endpoint_for("s2s", "vevo2") == "/audioapi/v1/tasks/run"


def test_diar_uses_the_task_route():
    assert api_endpoint_for("diar", "sortformer") == "/audioapi/v1/tasks/run"
    assert request_field_groups_for("diar", "sortformer") == []


def test_configured_task_selects_the_endpoint():
    for family in ("vevo2", "seed_vc", "miocodec", "personaplex"):
        assert api_endpoint_for("tts", family) == "/v1/audio/speech"
        assert api_endpoint_for("vc", family) == "/audioapi/v1/tasks/run"
    assert api_endpoint_for("vc", "chatterbox_turbo") == "/audioapi/v1/tasks/run"
    assert api_endpoint_for("asr", "firered_audio") == "/v1/audio/transcriptions"
    assert not is_profiled_task(None, None)
    assert not is_profiled_task("", "")


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


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("asr", True),
        ("tts", False),
        ("align", False),
    ],
)
def test_is_asr_task(task, expected):
    assert is_asr_task(task) is expected


@pytest.mark.parametrize(
    ("task", "expected"),
    [
        ("tts", True),
        ("clon", True),
        ("vdes", True),
        ("vc", False),
        ("svc", False),
        ("s2s", False),
        ("asr", False),
        ("gen", False),
    ],
)
def test_is_tts_task(task, expected):
    assert is_tts_task(task) is expected
