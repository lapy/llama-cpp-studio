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
def test_unknown_families_get_generic_profiles(task, family):
    assert is_profiled_task(task, family)
    profile = task_profile_for(task, family)
    assert profile is not None
    assert profile.get("generic") is True
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


def test_vibevoice_task_specific_profiles():
    asr_profile = task_profile_for("asr", "vibevoice")
    tts_profile = task_profile_for("tts", "vibevoice")
    assert asr_profile["label"] == "VibeVoice ASR"
    assert tts_profile["label"] == "VibeVoice"


def test_api_example_hint_for_speech_endpoint():
    hint = api_example_hint_for("tts", "kokoro_tts")
    assert "speech" in hint.lower()


def test_api_example_hint_for_transcription_endpoint():
    hint = api_example_hint_for("asr", "nemotron_asr")
    assert "multipart" in hint.lower() or "audio path" in hint.lower()


def test_merge_request_field_groups_prefers_scanned():
    from backend.audio_task_profiles import merge_request_field_groups

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
        field.get("key") == "temperature"
        for group in groups
        for field in group.get("fields") or []
    )


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
def test_profiled_field_groups_non_empty(task, family):
    groups = request_field_groups_for(task, family)
    assert len(groups) >= 1


def test_clon_and_vdes_use_speech_endpoint_and_defaults():
    assert request_defaults_key_for("clon", "chatterbox") == "speech_defaults"
    assert request_defaults_key_for("vdes", "qwen3_tts") == "speech_defaults"
    assert api_endpoint_for("clon", "chatterbox") == "/v1/audio/speech"
    assert api_endpoint_for("vdes", "qwen3_tts") == "/v1/audio/speech"


def test_svc_and_s2s_use_task_defaults_and_tasks_run():
    assert request_defaults_key_for("svc", "seed_vc") == "task_defaults"
    assert request_defaults_key_for("s2s", "vevo2") == "task_defaults"
    assert api_endpoint_for("svc", "seed_vc") == "/v1/tasks/run"
    assert api_endpoint_for("s2s", "vevo2") == "/v1/tasks/run"


def test_diar_routes_to_analysis_profile():
    profile = task_profile_for("diar", "sortformer")
    assert profile is not None
    groups = request_field_groups_for("diar", "sortformer")
    assert groups


def test_vevo2_seed_vc_miocodec_always_use_tasks_run_for_tts():
    for family in ("vevo2", "seed_vc", "miocodec"):
        assert api_endpoint_for("tts", family) == "/v1/tasks/run"
        assert request_defaults_key_for("tts", family) == "task_defaults"

    assert not is_profiled_task(None, None)
    assert not is_profiled_task("", "")
