"""Unified facade for audio.cpp task family profiles and request defaults."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

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
    sidecar_session_fields_for_family,
    transcription_request_field_groups,
)
from backend.audio_cpp_discovery import (
    resolve_api_endpoint,
    resolve_defaults_key_for_endpoint,
)
from backend.audio_gen_profiles import (
    gen_profile_for_family,
    generation_request_field_groups,
    is_gen_task,
)
from backend.audio_request_policy import build_request_policy
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


def _family_key(family: Optional[str]) -> str:
    return str(family or "").strip().lower()


def _generic_profile_for_task(task: Optional[str], family: Optional[str]) -> Dict[str, Any]:
    task_key = str(task or "").strip().lower()
    family_key = _family_key(family) or "unknown"
    endpoint = api_endpoint_for(task_key, family_key)
    workflows = [task_key] if task_key else ["run"]
    return {
        "label": family_key.replace("_", " ").title(),
        "workflows": workflows,
        "summary": (
            f"Auto-discovered audio.cpp profile for {family_key}"
            f" ({task_key or 'task'}). Options come from model --help / inspect."
        ),
        "api_hint": api_example_hint_for(task_key, family_key),
        "generic": True,
        "api_endpoint": endpoint,
    }


def task_profile_for(task: Optional[str], family: Optional[str] = None) -> Optional[Dict[str, Any]]:
    family_key = _family_key(family)
    task_key = str(task or "").strip().lower()

    # Prefer conversion/gen/analysis profiles when the family is multi-route
    if vc_profile_for_family(family_key):
        profile = vc_profile_for_family(family_key)
        if profile and (
            is_vc_task(task_key)
            or task_key in {"tts", "clon", "vdes", ""}
            or family_key in _FAMILY_FIELD_GROUP_GETTERS
        ):
            # VeVo2-style: conversion profile owns routing even for tts task labels
            if family_key in _FAMILY_FIELD_GROUP_GETTERS or is_vc_task(task_key):
                return profile

    if family_key in _FAMILY_FIELD_GROUP_GETTERS:
        for getter in (
            gen_profile_for_family,
            vc_profile_for_family,
            analysis_profile_for_family,
            sep_profile_for_family,
            align_profile_for_family,
        ):
            profile = getter(family_key)
            if profile:
                return profile

    if is_asr_task(task_key):
        profile = asr_profile_for_family(family_key)
        if profile:
            return profile
    if is_tts_task(task_key):
        profile = tts_profile_for_family(family_key)
        if profile:
            return profile
    if is_gen_task(task_key):
        profile = gen_profile_for_family(family_key)
        if profile:
            return profile
    if is_vc_task(task_key):
        profile = vc_profile_for_family(family_key)
        if profile:
            return profile
    if is_analysis_task(task_key):
        profile = analysis_profile_for_family(family_key)
        if profile:
            return profile
    if is_sep_task(task_key):
        profile = sep_profile_for_family(family_key)
        if profile:
            return profile
    if is_align_task(task_key):
        profile = align_profile_for_family(family_key)
        if profile:
            return profile

    for getter in _FAMILY_GETTERS:
        profile = getter(family_key)
        if profile:
            return profile

    if task_key or family_key:
        return _generic_profile_for_task(task_key, family_key)
    return None


def _curated_request_field_groups(
    task: Optional[str], family: Optional[str] = None
) -> List[Dict[str, Any]]:
    family_key = _family_key(family)
    getter = _FAMILY_FIELD_GROUP_GETTERS.get(family_key)
    if getter:
        return getter(family_key)
    task_key = str(task or "").strip().lower()
    if is_asr_task(task_key):
        return transcription_request_field_groups(family_key)
    if is_tts_task(task_key):
        groups = speech_request_field_groups(family_key)
        if groups:
            return groups
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


def merge_request_field_groups(
    curated: Sequence[Dict[str, Any]],
    scanned: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Prefer scanned engine options; overlay curated OpenAI/workflow fields."""
    seen = set()
    merged: List[Dict[str, Any]] = []
    for group in scanned or []:
        fields = []
        for field in group.get("fields") or []:
            key = str(field.get("key") or "")
            if key:
                seen.add(key)
            fields.append(field)
        if fields:
            merged.append({**group, "fields": fields})
    for group in curated or []:
        fields = []
        for field in group.get("fields") or []:
            key = str(field.get("key") or "")
            if not key or key in seen:
                continue
            seen.add(key)
            fields.append(field)
        if fields:
            merged.append({**group, "fields": fields})
    return merged


def request_field_groups_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    profile_sections: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    curated = _curated_request_field_groups(task, family)
    scanned: List[Dict[str, Any]] = []
    if profile_sections:
        from backend.audio_cpp_option_discovery import scanned_request_field_groups

        scanned = scanned_request_field_groups(profile_sections)
    if scanned:
        return merge_request_field_groups(curated, scanned)
    return curated


def sidecar_session_fields_for(
    task: Optional[str] = None,
    family: Optional[str] = None,
    *,
    profile_sections: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Thin curated overlays; omit keys already discovered in the model profile."""
    curated = sidecar_session_fields_for_family(_family_key(family))
    if not profile_sections:
        return curated
    known = {
        str(param.get("key") or "")
        for section in profile_sections
        for param in section.get("params") or []
        if param.get("scope") == "session_option" and param.get("key")
    }
    return [field for field in curated if str(field.get("key") or "") not in known]


def sidecar_load_fields_for(
    task: Optional[str] = None, family: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Load options come from the model --help scan; no hardcoded catalog."""
    return []


def is_profiled_task(task: Optional[str], family: Optional[str] = None) -> bool:
    return task_profile_for(task, family) is not None


def _synthetic_inspection_tasks(task: Optional[str], family: Optional[str]) -> List[str]:
    """Build inspect-like task signals when only family/task labels are available."""
    family_key = _family_key(family)
    task_key = str(task or "").strip().lower()
    tasks: List[str] = []
    if task_key:
        tasks.append(task_key)

    has_vc_profile = bool(vc_profile_for_family(family_key))
    has_tts_profile = bool(tts_profile_for_family(family_key))
    # Multi-route conversion families force tasks/run even for tts labels.
    if has_vc_profile or family_key in {"vevo2", "seed_vc", "miocodec"}:
        tasks.extend(["vc", "tts"])
    elif is_vc_task(task_key) and has_tts_profile:
        # TTS families that also expose a vc workflow (e.g. chatterbox) still
        # use the OpenAI speech endpoint.
        tasks.append("tts")
        tasks = [t for t in tasks if t != "vc"]

    if gen_profile_for_family(family_key) or is_gen_task(task_key):
        tasks.append("gen")
    if analysis_profile_for_family(family_key) or is_analysis_task(task_key):
        tasks.append(task_key or "vad")
    if sep_profile_for_family(family_key) or is_sep_task(task_key):
        tasks.append("sep")
    if align_profile_for_family(family_key) or is_align_task(task_key):
        tasks.append("align")
    return list(dict.fromkeys(tasks))


def request_defaults_key_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    inspection: Optional[dict] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    model_profile: Optional[dict] = None,
) -> str:
    endpoint = api_endpoint_for(
        task,
        family,
        inspection=inspection,
        help_option_keys=help_option_keys,
        model_profile=model_profile,
    )
    return resolve_defaults_key_for_endpoint(endpoint)


def api_endpoint_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    inspection: Optional[dict] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    model_profile: Optional[dict] = None,
) -> str:
    if inspection is not None or help_option_keys is not None or model_profile is not None:
        policy = build_request_policy(
            task=task,
            family=family,
            inspection=inspection,
            model_profile=model_profile,
            help_option_keys=help_option_keys,
        )
        return str(policy.get("api_endpoint") or "/v1/tasks/run")

    synthetic = _synthetic_inspection_tasks(task, family)
    # When a TTS family exposes vc as a speech workflow, route using speech task.
    route_task = task
    if (
        is_vc_task(task)
        and tts_profile_for_family(_family_key(family))
        and not vc_profile_for_family(_family_key(family))
        and "tts" in synthetic
        and "vc" not in synthetic
    ):
        route_task = "tts"
    return resolve_api_endpoint(
        task=route_task,
        inspection_tasks=synthetic,
        help_option_keys=help_option_keys,
    )


def api_example_hint_for(
    task: Optional[str],
    family: Optional[str] = None,
    *,
    inspection: Optional[dict] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    model_profile: Optional[dict] = None,
) -> str:
    endpoint = api_endpoint_for(
        task,
        family,
        inspection=inspection,
        help_option_keys=help_option_keys,
        model_profile=model_profile,
    )
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


def supports_voice_presets_for(
    *,
    request_defaults_key: Optional[str],
) -> bool:
    """Voice presets apply only when the model routes through speech_defaults."""
    return str(request_defaults_key or "").strip() == "speech_defaults"
