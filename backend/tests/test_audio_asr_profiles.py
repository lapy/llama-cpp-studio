"""ASR profile and transcription default helpers."""

import pytest

from backend.audio_asr_profiles import (
    asr_profile_for_family,
    is_asr_task,
    transcription_request_field_groups,
)
from backend.audio_transcription_defaults import normalize_transcription_defaults
from backend.tests.audio_profile_fixtures import (
    ASR_FAMILIES,
    assert_field_groups_shape,
    assert_profile_shape,
)


@pytest.mark.parametrize("family", ASR_FAMILIES)
def test_asr_profile_exists_for_documented_family(family):
    profile = asr_profile_for_family(family)
    assert profile is not None
    assert_profile_shape(profile)


@pytest.mark.parametrize("family", ASR_FAMILIES)
def test_transcription_field_groups_are_well_formed(family):
    groups = transcription_request_field_groups(family)
    assert groups
    assert_field_groups_shape(groups)


def test_nemotron_asr_streaming_workflow_and_options():
    profile = asr_profile_for_family("nemotron_asr")
    assert "streaming" in profile["workflows"]
    groups = transcription_request_field_groups("nemotron_asr")
    ids = [group["id"] for group in groups]
    assert "context" in ids
    assert "session" in ids
    # Nested options are no longer hardcoded — they come from model scan merge.
    assert "options" not in ids
    assert not profile.get("request_option_fields")


def test_nemotron_scanned_request_options_merge_into_field_groups():
    from backend.audio_task_profiles import request_field_groups_for

    groups = request_field_groups_for(
        "asr",
        "nemotron_asr",
        profile_sections=[
            {
                "id": "model_request_options",
                "params": [
                    {
                        "key": "lookahead_tokens",
                        "scope": "request_option",
                        "type": "string",
                    },
                    {
                        "key": "keep_language_tags",
                        "scope": "request_option",
                        "type": "bool",
                    },
                ],
            }
        ],
    )
    keys = {field["key"] for group in groups for field in group.get("fields") or []}
    assert {"lookahead_tokens", "keep_language_tags", "language", "stream"}.issubset(keys)


def test_higgs_audio_stt_includes_prompt_and_chunking():
    groups = transcription_request_field_groups("higgs_audio_stt")
    ids = [group["id"] for group in groups]
    assert "context" in ids
    assert "chunking" in ids
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "prompt" in field_keys
    assert "enable_thinking" not in field_keys
    assert not (asr_profile_for_family("higgs_audio_stt") or {}).get(
        "request_option_fields"
    )


def test_hviske_includes_beam_search_decode_fields():
    groups = transcription_request_field_groups("hviske")
    ids = [group["id"] for group in groups]
    assert "decode" in ids
    decode_keys = {
        field["key"]
        for group in groups
        if group["id"] == "decode"
        for field in group["fields"]
    }
    assert {"num_beams", "do_sample", "temperature"}.issubset(decode_keys)


def test_vibevoice_asr_includes_prompt_and_decode():
    groups = transcription_request_field_groups("vibevoice")
    field_keys = {field["key"] for group in groups for field in group["fields"]}
    assert "prompt" in field_keys
    assert "num_beams" in field_keys


def test_qwen3_asr_chunking_fields():
    groups = transcription_request_field_groups("qwen3_asr")
    chunk_keys = {
        field["key"]
        for group in groups
        if group["id"] == "chunking"
        for field in group["fields"]
    }
    assert {"audio_chunk_mode", "audio_chunk_seconds"}.issubset(chunk_keys)


def test_qwen3_asr_sidecar_session_fields_for_aligned_asr():
    from backend.audio_asr_profiles import sidecar_session_fields_for_family

    fields = sidecar_session_fields_for_family("qwen3_asr")
    keys = {field["key"] for field in fields}
    assert "qwen3_asr.forced_aligner_model_path" in keys
    assert "qwen3_asr.vad_model_path" in keys
    assert all(field.get("scope") == "session_option" for field in fields)


def test_upstream_asr_family_aliases_resolve_profiles():
    assert asr_profile_for_family("citrinet_asr")["label"] == asr_profile_for_family("citrinet")["label"]
    assert asr_profile_for_family("hviske_asr")["label"] == asr_profile_for_family("hviske")["label"]
    assert asr_profile_for_family("vibevoice_asr")["label"] == asr_profile_for_family("vibevoice")["label"]
    assert transcription_request_field_groups("citrinet_asr")
    assert transcription_request_field_groups("hviske_asr")
    assert transcription_request_field_groups("vibevoice_asr")


def test_citrinet_minimal_profile():
    groups = transcription_request_field_groups("citrinet")
    ids = [group["id"] for group in groups]
    assert ids == ["context"]


def test_normalize_transcription_defaults_maps_prompt_and_options():
    defaults = normalize_transcription_defaults(
        {
            "language": "en-US",
            "stream": True,
            "prompt": "Transcribe the speech.",
            "options": {
                "lookahead_tokens": "4",
                "keep_language_tags": False,
            },
        }
    )
    assert defaults["language"] == "en-US"
    assert defaults["stream"] is True
    assert defaults["prompt"] == "Transcribe the speech."
    assert defaults["options"]["lookahead_tokens"] == "4"
    assert defaults["options"]["keep_language_tags"] is False


def test_normalize_transcription_defaults_ignores_invalid_ints():
    out = normalize_transcription_defaults({"max_tokens": "many"})
    assert "max_tokens" not in out


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


def test_parakeet_tdt_streaming_workflow_and_session_fields():
    profile = asr_profile_for_family("parakeet_tdt")
    assert "streaming" in profile["workflows"]
    groups = transcription_request_field_groups("parakeet_tdt")
    ids = [group["id"] for group in groups]
    assert "session" in ids
