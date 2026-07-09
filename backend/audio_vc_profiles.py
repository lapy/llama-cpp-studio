"""Curated voice conversion family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_profile_fields import field_spec

_VC_TASKS = frozenset({"vc", "svc", "s2s"})


def is_vc_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _VC_TASKS


def vc_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    key = str(family or "").strip().lower()
    return _FAMILY_PROFILES.get(key) if key else None


def conversion_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    profile = vc_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []
    for group_id, label, description in _GROUP_META:
        keys = profile.get(f"{group_id}_fields") or []
        fields = [field_spec(key) for key in keys]
        if fields:
            groups.append({"id": group_id, "label": label, "description": description, "fields": fields})
    return groups


_GROUP_META = [
    ("route", "Route", "Conversion route for multi-route families."),
    ("audio", "Audio inputs", "Source and reference audio roles."),
    ("text", "Text inputs", "Target text or lyrics when required by the route."),
    ("generation", "Generation defaults", "Sampling and inference controls."),
    ("options", "Model request options", "Family-specific request options."),
]

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "seed_vc": {
        "label": "Seed-VC",
        "workflows": ["v2_vc", "v1_whisper_bigvgan_vc", "v1_xlsr_hift_vc", "v1_svc"],
        "summary": "Voice and singing conversion from source audio plus a target voice reference.",
        "route_fields": ["task_route"],
        "audio_fields": ["audio", "voice_ref"],
        "generation_fields": [
            "num_inference_steps",
            "temperature",
            "top_p",
            "repetition_penalty",
            "seed",
        ],
        "options_fields": [
            "length_adjust",
            "intelligibility_cfg_rate",
            "similarity_cfg_rate",
            "inference_cfg_rate",
            "f0_condition",
            "auto_f0_adjust",
            "semi_tone_shift",
        ],
        "api_hint": "Default vc route is v2_vc; svc default is v1_svc.",
    },
    "miocodec": {
        "label": "MioCodec",
        "workflows": ["vc", "s2s"],
        "summary": "Speech codec conversion from source speech plus a target voice reference.",
        "audio_fields": ["audio", "voice_ref"],
        "api_hint": "vc and s2s share the same audio + voice_ref inputs.",
    },
    "vevo2": {
        "label": "VeVo2",
        "workflows": [
            "zero_shot_tts",
            "text_to_singing",
            "style_preserved_vc",
            "editing",
            "style_preserved_svc",
        ],
        "summary": "Speech, singing, conversion, and editing through explicit task routes.",
        "route_fields": ["task_route"],
        "audio_fields": [
            "source_audio",
            "voice_ref",
            "target_voice",
            "prosody_ref",
            "style_ref",
        ],
        "text_fields": ["text", "target_text", "reference_text", "style_ref_text"],
        "generation_fields": [
            "num_inference_steps",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "max_tokens",
            "seed",
            "target_duration_seconds",
        ],
        "options_fields": ["use_prosody_code", "use_pitch_shift"],
        "api_hint": "Route controls which audio roles are required. Align reference transcript with voice_ref for zero-shot TTS.",
    },
}
