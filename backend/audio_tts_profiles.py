"""Curated TTS family guidance for voice preset and speech request configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_TTS_TASKS = frozenset({"tts", "clon", "vdes", "vc", "svc", "s2s"})


def is_tts_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _TTS_TASKS


def tts_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return workflow guidance for a TTS family, or None if unknown."""
    key = str(family or "").strip().lower()
    if not key:
        return None
    return _FAMILY_PROFILES.get(key)


def _apply_field_hints(fields: List[Dict[str, Any]], profile: Dict[str, Any]) -> List[Dict[str, Any]]:
    hints = profile.get("field_hints") or {}
    if not hints:
        return fields
    out: List[Dict[str, Any]] = []
    for field in fields:
        spec = dict(field)
        hint = hints.get(spec.get("key"))
        if hint:
            spec["hint"] = hint
        out.append(spec)
    return out


def speech_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    """Grouped speech API / request-default fields for the model editor."""
    profile = tts_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []

    voice_fields = []
    for field in profile.get("voice_fields") or []:
        voice_fields.append(_field_spec(field))
    for field in profile.get("optional_voice_fields") or []:
        spec = _field_spec(field)
        spec["optional"] = True
        voice_fields.append(spec)
    if voice_fields:
        groups.append(
            {
                "id": "voice",
                "label": "Voice & reference",
                "description": "Saved as server voice presets or per-request speech defaults.",
                "fields": voice_fields,
            }
        )

    if profile.get("supports_instructions"):
        instr_style = str(profile.get("instructions_style") or "natural_language")
        if instr_style == "omnivoice_attributes":
            from backend.audio_omnivoice_instruct import (
                OMNIVOICE_INSTRUCT_EXAMPLE,
                format_omnivoice_instruct_hint,
            )

            field = _field_spec("instructions")
            field["label"] = "Voice design attributes (instruct)"
            field["placeholder"] = OMNIVOICE_INSTRUCT_EXAMPLE
            field["hint"] = format_omnivoice_instruct_hint()
            groups.append(
                {
                    "id": "design",
                    "label": "Voice design",
                    "description": (
                        "Comma-separated speaker attributes for OmniVoice voice design. "
                        "Mapped to OpenAI instructions / options.instruct — each comma-separated "
                        "token must be from the OmniVoice vocabulary."
                    ),
                    "fields": [field],
                }
            )
        else:
            field = _field_spec("instructions")
            if instr_style == "natural_language":
                field["placeholder"] = "A warm, calm motherly narrator with gentle pacing"
                field["hint"] = (
                    "Free-form natural-language voice description. "
                    "Used by Qwen3 VoiceDesign (vdes) — not the OmniVoice attribute vocabulary."
                )
            groups.append(
                {
                    "id": "design",
                    "label": "Voice design",
                    "description": "Natural-language style instructions (OpenAI field: instructions).",
                    "fields": [field],
                }
            )

    generation_fields = [_field_spec(key) for key in profile.get("generation_fields") or []]
    if generation_fields:
        groups.append(
            {
                "id": "generation",
                "label": "Generation defaults",
                "description": "Optional per-request sampling and length controls.",
                "fields": generation_fields,
            }
        )

    option_fields = _apply_field_hints(
        [_field_spec(key, nested=True) for key in profile.get("request_option_fields") or []],
        profile,
    )
    if option_fields:
        groups.append(
            {
                "id": "options",
                "label": "Model request options",
                "description": "Family-specific options passed in the speech request options object.",
                "fields": option_fields,
            }
        )
    return groups


def _field_spec(key: str, *, nested: bool = False) -> Dict[str, Any]:
    spec = dict(_FIELD_SPECS.get(key) or {"key": key, "label": key, "type": "string"})
    if nested:
        spec["nested"] = True
        spec["options_key"] = key
    return spec


_FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "voice_id": {
        "key": "voice_id",
        "label": "Built-in voice id",
        "type": "string",
        "placeholder": "af_heart",
        "preset_field": True,
    },
    "voice_ref": {
        "key": "voice_ref",
        "label": "Reference audio (WAV)",
        "type": "path",
        "placeholder": "samples/reference.wav",
        "preset_field": True,
        "speech_field": "voice_ref",
    },
    "reference_text": {
        "key": "reference_text",
        "label": "Reference transcript",
        "type": "textarea",
        "placeholder": "Transcript matching the reference audio…",
        "preset_field": True,
        "speech_field": "reference_text",
    },
    "instructions": {
        "key": "instructions",
        "label": "Voice design instructions",
        "type": "textarea",
        "placeholder": "female, young adult, moderate pitch",
        "speech_field": "instructions",
    },
    "language": {
        "key": "language",
        "label": "Language",
        "type": "string",
        "placeholder": "en",
        "speech_field": "language",
    },
    "seed": {"key": "seed", "label": "Seed", "type": "int", "speech_field": "seed"},
    "temperature": {
        "key": "temperature",
        "label": "Temperature",
        "type": "float",
        "speech_field": "temperature",
    },
    "top_p": {"key": "top_p", "label": "Top P", "type": "float", "speech_field": "top_p"},
    "top_k": {"key": "top_k", "label": "Top K", "type": "int", "speech_field": "top_k"},
    "max_tokens": {
        "key": "max_tokens",
        "label": "Max tokens",
        "type": "int",
        "speech_field": "max_tokens",
    },
    "guidance_scale": {
        "key": "guidance_scale",
        "label": "Guidance scale",
        "type": "float",
        "speech_field": "guidance_scale",
    },
    "num_inference_steps": {
        "key": "num_inference_steps",
        "label": "Inference steps",
        "type": "int",
        "speech_field": "num_inference_steps",
    },
    "repetition_penalty": {
        "key": "repetition_penalty",
        "label": "Repetition penalty",
        "type": "float",
        "speech_field": "repetition_penalty",
    },
    "text_chunk_size": {
        "key": "text_chunk_size",
        "label": "Text chunk size",
        "type": "int",
        "options_key": "text_chunk_size",
        "nested": True,
    },
    "do_sample": {
        "key": "do_sample",
        "label": "Stochastic sampling",
        "type": "bool",
        "options_key": "do_sample",
        "nested": True,
    },
    "speaking_rate": {
        "key": "speaking_rate",
        "label": "Speaking rate",
        "type": "float",
        "options_key": "speaking_rate",
        "nested": True,
    },
    "speed": {
        "key": "speed",
        "label": "Speech speed",
        "type": "float",
        "options_key": "speed",
        "nested": True,
    },
    "no_ref": {
        "key": "no_ref",
        "label": "No-reference mode",
        "type": "bool",
        "options_key": "no_ref",
        "nested": True,
    },
    "caption": {
        "key": "caption",
        "label": "Voice-design caption",
        "type": "textarea",
        "options_key": "caption",
        "nested": True,
    },
    "voice_samples": {
        "key": "voice_samples",
        "label": "Speaker reference WAVs",
        "type": "string",
        "placeholder": "speaker1.wav,speaker2.wav",
        "options_key": "voice_samples",
        "nested": True,
    },
    "voice": {
        "key": "voice",
        "label": "Preset or built-in voice",
        "type": "string",
        "placeholder": "assistant",
        "speech_field": "voice",
    },
    "speaker": {
        "key": "speaker",
        "label": "Built-in speaker",
        "type": "string",
        "placeholder": "Vivian",
        "speech_field": "speaker",
    },
}

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "chatterbox": {
        "label": "Chatterbox",
        "workflows": ["clone"],
        "summary": "Voice-clone TTS with reference WAV. Task is usually clon.",
        "voice_fields": ["voice_ref"],
        "optional_voice_fields": ["reference_text", "language"],
        "generation_fields": [
            "temperature",
            "top_p",
            "guidance_scale",
            "repetition_penalty",
            "max_tokens",
            "text_chunk_size",
            "do_sample",
        ],
        "api_hint": "Clone with a reference WAV. Set a default voice preset so clients can omit voice_ref.",
    },
    "kokoro_tts": {
        "label": "Kokoro",
        "workflows": ["preset"],
        "summary": "Compact preset-voice TTS using packaged voice tensors.",
        "voice_fields": ["voice_id"],
        "optional_voice_fields": ["language"],
        "generation_fields": ["text_chunk_size"],
        "api_hint": "Use voice presets with voice_id, or pass voice in each /v1/audio/speech request.",
    },
    "miotts": {
        "label": "MioTTS",
        "workflows": ["clone"],
        "summary": "Voice-clone TTS requiring a reference WAV.",
        "voice_fields": ["voice_ref"],
        "generation_fields": [
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "max_tokens",
            "text_chunk_size",
            "do_sample",
        ],
        "request_option_fields": ["best_of_n_enabled"],
        "api_hint": "Reference WAV is required unless a default voice preset is configured.",
    },
    "moss_tts": {
        "label": "MOSS-TTS",
        "workflows": ["clone"],
        "summary": "Compact voice-clone TTS; reference transcript improves alignment.",
        "voice_fields": ["voice_ref"],
        "optional_voice_fields": ["reference_text"],
        "generation_fields": ["text_chunk_size"],
        "request_option_fields": [
            "text_temperature",
            "text_top_p",
            "text_top_k",
            "audio_temperature",
            "audio_top_p",
            "audio_top_k",
        ],
        "api_hint": "Include reference_text in presets when you have a transcript for the reference clip.",
    },
    "omnivoice": {
        "label": "OmniVoice",
        "workflows": ["clone", "design", "instruct"],
        "summary": "Clone from reference audio or design a voice with instruct attributes.",
        "supports_instructions": True,
        "instructions_style": "omnivoice_attributes",
        "voice_fields": ["voice_ref"],
        "optional_voice_fields": ["reference_text"],
        "generation_fields": ["guidance_scale", "num_inference_steps", "text_chunk_size"],
        "request_option_fields": [
            "speed",
            "audio_chunk_duration_seconds",
            "audio_chunk_threshold_seconds",
        ],
        "api_hint": (
            "Voice design uses comma-separated instruct attributes such as "
            "'female, young adult, moderate pitch, british accent' — not free-form prose "
            "like 'calm, kind, motherly'. Use voice presets for cloning."
        ),
    },
    "pocket_tts": {
        "label": "PocketTTS",
        "workflows": ["preset", "clone"],
        "summary": "Built-in voices or reference WAV cloning.",
        "voice_fields": ["voice_id", "voice_ref"],
        "generation_fields": ["text_chunk_size"],
        "request_option_fields": ["language"],
        "api_hint": "Configure preset voice_id defaults, or clone with voice_ref in a named preset.",
    },
    "voxcpm2": {
        "label": "VoxCPM2",
        "workflows": ["design", "clone", "ultimate_clone"],
        "summary": "Style prefix in text, optional clone, or transcript-assisted ultimate clone.",
        "supports_instructions": False,
        "voice_fields": ["voice_ref"],
        "optional_voice_fields": ["reference_text"],
        "generation_fields": [
            "guidance_scale",
            "num_inference_steps",
            "max_tokens",
            "text_chunk_size",
        ],
        "api_hint": (
            "Voice design uses parentheses at the start of input text, e.g. '(calm) Hello'. "
            "Do not use speech_defaults.instructions."
        ),
    },
    "higgs_audio_tts": {
        "label": "Higgs Audio v3 TTS",
        "workflows": ["clone"],
        "summary": "Voice-clone TTS with optional reference transcript.",
        "voice_fields": ["voice_ref"],
        "optional_voice_fields": ["reference_text"],
        "generation_fields": [
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "max_tokens",
            "text_chunk_size",
        ],
        "api_hint": "Save clone context in a default voice preset for simpler API calls.",
    },
    "irodori_tts": {
        "label": "Irodori-TTS",
        "workflows": ["no_ref", "clone", "design"],
        "summary": "Japanese TTS with no-reference, reference-conditioned, or caption-based design.",
        "voice_fields": ["voice_ref"],
        "optional_voice_fields": ["language"],
        "generation_fields": ["num_inference_steps"],
        "request_option_fields": [
            "no_ref",
            "caption",
            "duration_scale",
            "min_seconds",
            "max_seconds",
            "text_guidance_scale",
            "speaker_guidance_scale",
            "caption_guidance_scale",
        ],
        "field_hints": {
            "caption": (
                "Japanese voice-design caption for Irodori. Use options.caption instead of "
                "the OpenAI instructions field."
            ),
        },
        "api_hint": (
            "Voice design uses options.caption, not instructions. "
            "Use request option no_ref=false with a voice preset for reference-conditioned speech."
        ),
    },
    "supertonic": {
        "label": "Supertonic",
        "workflows": ["preset"],
        "summary": "Preset-voice multilingual TTS (M1, F1).",
        "voice_fields": ["voice_id"],
        "optional_voice_fields": ["language"],
        "generation_fields": ["num_inference_steps", "seed"],
        "request_option_fields": ["speaking_rate", "voice"],
        "api_hint": "Package voices M1/F1 work well as named voice presets.",
    },
    "vibevoice": {
        "label": "VibeVoice",
        "workflows": ["multi_speaker"],
        "summary": "Long-form multi-speaker TTS with ordered speaker reference WAVs.",
        "voice_fields": ["voice_samples"],
        "generation_fields": [
            "guidance_scale",
            "num_inference_steps",
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "do_sample",
        ],
        "request_option_fields": ["max_length_times", "vibevoice.lora"],
        "api_hint": "Use Speaker 1:/Speaker 2: script lines and comma-separated voice_samples presets.",
    },
    "qwen3_tts": {
        "label": "Qwen3 TTS",
        "workflows": ["clone", "design", "custom_voice"],
        "summary": "Base clones from reference audio; VoiceDesign uses instruct; CustomVoice uses packaged speakers.",
        "voice_fields": ["voice_ref", "speaker"],
        "optional_voice_fields": ["reference_text", "language"],
        "supports_instructions": True,
        "instructions_style": "natural_language",
        "generation_fields": [
            "max_tokens",
            "text_chunk_size",
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "do_sample",
            "seed",
        ],
        "request_option_fields": [
            "subtalker_do_sample",
            "subtalker_temperature",
            "subtalker_top_k",
            "subtalker_top_p",
        ],
        "api_hint": "Base uses voice_ref; VoiceDesign (vdes) uses free-form instruct; CustomVoice uses speaker names like Vivian or Ryan.",
    },
}
