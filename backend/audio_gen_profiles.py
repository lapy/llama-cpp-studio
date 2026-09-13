"""Curated music/SFX generation family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_profile_fields import field_spec

_GEN_TASKS = frozenset({"gen"})


def is_gen_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _GEN_TASKS


def gen_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    key = str(family or "").strip().lower()
    return _FAMILY_PROFILES.get(key) if key else None


def generation_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    profile = gen_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []
    hints = profile.get("field_hints") or {}
    for group_id, label, description in _GROUP_META:
        keys = profile.get(f"{group_id}_fields") or []
        fields = []
        for key in keys:
            spec = field_spec(key)
            hint = hints.get(key)
            if hint:
                spec = dict(spec)
                spec["hint"] = hint
            fields.append(spec)
        if fields:
            groups.append({"id": group_id, "label": label, "description": description, "fields": fields})
    return groups


_GROUP_META = [
    ("route", "Route", "Select the generation or edit route."),
    ("prompt", "Prompt & lyrics", "Text inputs for music or SFX generation."),
    ("audio", "Source audio", "Optional or required source audio depending on route."),
    ("timing", "Duration & repaint", "Length controls and repaint windows."),
    ("generation", "Generation defaults", "Diffusion and sampling controls."),
    ("conditioning", "Audio conditioning", "Init-audio and inpainting controls."),
    ("options", "Model request options", "Family-specific options in the request options object."),
]

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "ace_step": {
        "label": "ACE-Step",
        "workflows": ["text2music", "complete", "lego", "extract", "cover", "repaint"],
        "summary": "Generate and edit music from prompts, lyrics, and optional source audio.",
        "route_fields": ["task_route"],
        "prompt_fields": ["text", "lyrics", "language"],
        "audio_fields": ["audio"],
        "timing_fields": ["duration_seconds", "repaint_start", "repaint_end"],
        "generation_fields": ["num_inference_steps", "guidance_scale", "seed"],
        "options_fields": [
            "track_name",
            "negative_prompt",
            "repaint_mode",
            "repaint_strength",
            "audio_input_kind",
        ],
        "api_hint": "Routes control whether source audio is ignored, optional, or required. See ace_step.md for route details.",
    },
    "stable_audio": {
        "label": "Stable Audio",
        "workflows": ["text2music", "text2sfx", "init_audio", "inpaint"],
        "summary": "Generate music or SFX from text; music models support init-audio and inpainting.",
        "prompt_fields": ["text"],
        "audio_fields": ["audio"],
        "timing_fields": ["duration_seconds"],
        "generation_fields": ["num_inference_steps", "guidance_scale", "seed"],
        "conditioning_fields": [
            "audio_input_kind",
            "init_noise_level",
            "inpaint_mask_start_seconds",
            "inpaint_mask_end_seconds",
        ],
        "options_fields": ["negative_prompt", "sampler"],
        "api_hint": "Use audio_input_kind=init_audio or inpaint_audio when conditioning on source audio.",
    },
    "heartmula": {
        "label": "HeartMuLa",
        "workflows": ["lyrics2music", "infinite"],
        "summary": "Generate music from lyrics and comma-separated style tags.",
        "prompt_fields": ["text", "lyrics", "tags"],
        "timing_fields": ["duration_seconds"],
        "generation_fields": [
            "temperature",
            "top_k",
            "guidance_scale",
            "num_inference_steps",
            "seed",
            "text_chunk_size",
        ],
        "options_fields": ["infinite_mode", "codec_duration", "codec_guidance_scale"],
        "field_hints": {
            "tags": (
                "Comma-separated free-form style descriptors (genre, mood, instruments). "
                "This is not the OmniVoice voice-attribute vocabulary."
            ),
        },
        "api_hint": "Tags are comma-separated descriptors (genre, mood, instruments). Enable infinite_mode for long outputs.",
    },
    "midashenglm_gen": {
        "label": "MiDashengLM-Gen",
        "workflows": ["text2music", "text2sfx"],
        "summary": "Structured-prompt generation for speech, music, sound effects, and ambience.",
        "prompt_fields": ["text"],
        "timing_fields": ["duration_seconds"],
        "generation_fields": ["temperature", "guidance_scale", "seed"],
        "api_hint": "Describe the desired audio in text. Extra structure options come from model --help.",
    },
    "minimax_music3": {
        "label": "MiniMax Music 3",
        "workflows": ["lyrics2music"],
        "summary": "Text-to-music generation with optional lyrics conditioning.",
        "prompt_fields": ["text", "lyrics"],
        "timing_fields": ["duration_seconds"],
        "generation_fields": ["temperature", "guidance_scale", "seed"],
        "api_hint": "Use text for style and lyrics for vocals. Routes through tasks/run.",
    },
    "minimax_h3": {
        "label": "MiniMax-H3",
        "workflows": ["text2audio"],
        "summary": "MiniMax-H3 text-to-audio generation.",
        "prompt_fields": ["text"],
        "timing_fields": ["duration_seconds"],
        "generation_fields": ["temperature", "guidance_scale", "seed"],
        "api_hint": "Describe the clip in text. Video-oriented options come from model --help after install.",
    },
    "controlfoley": {
        "label": "ControlFoley",
        "workflows": ["text2sfx"],
        "summary": "44 kHz Foley generation from text and optional reference audio.",
        "prompt_fields": ["text"],
        "audio_fields": ["audio"],
        "timing_fields": ["duration_seconds"],
        "generation_fields": ["guidance_scale", "seed"],
        "api_hint": "Text describes the Foley event. Reference audio is optional conditioning.",
    },
}
