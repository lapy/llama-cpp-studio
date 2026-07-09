"""Unified facade for audio.cpp task family profiles and request defaults."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_align_profiles import (
    align_profile_for_family,
    alignment_request_field_groups,
    is_align_task,
)
from backend.audio_analysis_profiles import (
    analysis_profile_for_family,
    analysis_request_field_groups,
    is_analysis_task,
)
from backend.audio_asr_profiles import (
    asr_profile_for_family,
    is_asr_task,
    transcription_request_field_groups,
)
from backend.audio_gen_profiles import (
    gen_profile_for_family,
    generation_request_field_groups,
    is_gen_task,
)
from backend.audio_sep_profiles import (
    is_sep_task,
    sep_profile_for_family,
    separation_request_field_groups,
)
from backend.audio_tts_profiles import (
    is_tts_task,
    speech_request_field_groups,
    tts_profile_for_family,
)
from backend.audio_vc_profiles import (
    conversion_request_field_groups,
    is_vc_task,
    vc_profile_for_family,
)

_FAMILY_GETTERS = (
    gen_profile_for_family,
    vc_profile_for_family,
    analysis_profile_for_family,
    sep_profile_for_family,
    align_profile_for_family,
    asr_profile_for_family,
    tts_profile_for_family,
)

_FAMILY_FIELD_GROUP_GETTERS = {
    "ace_step": generation_request_field_groups,
    "stable_audio": generation_request_field_groups,
    "heartmula": generation_request_field_groups,
    "seed_vc": conversion_request_field_groups,
    "miocodec": conversion_request_field_groups,
    "vevo2": conversion_request_field_groups,
    "silero_vad": analysis_request_field_groups,
    "marblenet_vad": analysis_request_field_groups,
    "marblenet": analysis_request_field_groups,
    "sortformer_diar": analysis_request_field_groups,
    "sortformer": analysis_request_field_groups,
    "htdemucs": separation_request_field_groups,
    "mel_band_roformer": separation_request_field_groups,
    "qwen3_forced_aligner": alignment_request_field_groups,
}

_GENERIC_TASK_RUN_FAMILIES = frozenset(_FAMILY_FIELD_GROUP_GETTERS.keys())


def _family_key(family: Optional[str]) -> str:
    return str(family or "").strip().lower()


def task_profile_for(task: Optional[str], family: Optional[str]) -> Optional[Dict[str, Any]]:
    family_key = _family_key(family)
    for getter in _FAMILY_GETTERS:
        profile = getter(family_key)
        if profile:
            return profile
    return None


def request_field_groups_for(task: Optional[str], family: Optional[str]) -> List[Dict[str, Any]]:
    family_key = _family_key(family)
    getter = _FAMILY_FIELD_GROUP_GETTERS.get(family_key)
    if getter:
        return getter(family_key)
    task_key = str(task or "").strip().lower()
    if is_asr_task(task_key):
        return transcription_request_field_groups(family_key)
    if is_tts_task(task_key):
        return speech_request_field_groups(family_key)
    if is_gen_task(task_key):
        return generation_request_field_groups(family_key)
    if is_vc_task(task_key):
        return conversion_request_field_groups(family_key)
    if is_analysis_task(task_key):
        return analysis_request_field_groups(family_key)
    if is_sep_task(task_key):
        return separation_request_field_groups(family_key)
    if is_align_task(task_key):
        return alignment_request_field_groups(family_key)
    return []


def is_profiled_task(task: Optional[str], family: Optional[str] = None) -> bool:
    return task_profile_for(task, family) is not None


def request_defaults_key_for(task: Optional[str], family: Optional[str] = None) -> str:
    family_key = _family_key(family)
    if family_key in _GENERIC_TASK_RUN_FAMILIES:
        return "task_defaults"
    task_key = str(task or "").strip().lower()
    if is_asr_task(task_key):
        return "transcription_defaults"
    if is_tts_task(task_key) and family_key not in {"vevo2", "seed_vc", "miocodec"}:
        return "speech_defaults"
    return "task_defaults"


def api_endpoint_for(task: Optional[str], family: Optional[str] = None) -> str:
    family_key = _family_key(family)
    if family_key in _GENERIC_TASK_RUN_FAMILIES:
        return "/v1/tasks/run"
    task_key = str(task or "").strip().lower()
    if is_asr_task(task_key):
        return "/v1/audio/transcriptions"
    if is_tts_task(task_key) and family_key not in {"vevo2", "seed_vc", "miocodec"}:
        return "/v1/audio/speech"
    return "/v1/tasks/run"


def api_example_hint_for(task: Optional[str], family: Optional[str] = None) -> str:
    endpoint = api_endpoint_for(task, family)
    if endpoint == "/v1/audio/transcriptions":
        return (
            "JSON uses a server-local audio path. Multipart upload with a file field is also supported."
        )
    if endpoint == "/v1/audio/speech":
        return "OpenAI-compatible speech synthesis request."
    return (
        "Generic task request via /v1/tasks/run. Use the direct upstream path "
        "/upstream/{model}/v1/tasks/run until llama-swap routes this endpoint."
    )
