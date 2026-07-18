"""Curated ASR family guidance for transcription request configuration."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

_ASR_TASKS = frozenset({"asr"})

# Upstream loader names → curated Studio profile keys.
_FAMILY_ALIASES = {
    "citrinet_asr": "citrinet",
    "hviske_asr": "hviske",
    "vibevoice_asr": "vibevoice",
    "parakeet_tdt_0_6b_v3": "parakeet_tdt",
}


def is_asr_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _ASR_TASKS


def _canonical_family(family: Optional[str]) -> str:
    key = str(family or "").strip().lower()
    return _FAMILY_ALIASES.get(key, key)


def asr_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return workflow guidance for an ASR family, or None if unknown."""
    key = _canonical_family(family)
    if not key:
        return None
    return _FAMILY_PROFILES.get(key)


def transcription_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    """Grouped transcription API / request-default fields for the model editor."""
    profile = asr_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []

    context_fields = [_field_spec(key) for key in profile.get("context_fields") or []]
    if context_fields:
        groups.append(
            {
                "id": "context",
                "label": "Prompt & language",
                "description": "Optional ASR prompt text and language hints for each request.",
                "fields": context_fields,
            }
        )

    session_fields = [_field_spec(key) for key in profile.get("session_fields") or []]
    if session_fields:
        groups.append(
            {
                "id": "session",
                "label": "Session mode",
                "description": "Streaming requires server mode=streaming and stream=true in the request.",
                "fields": session_fields,
            }
        )

    chunk_fields = [_field_spec(key) for key in profile.get("chunk_fields") or []]
    if chunk_fields:
        groups.append(
            {
                "id": "chunking",
                "label": "Long-audio chunking",
                "description": "Chunking controls passed in the request options object.",
                "fields": chunk_fields,
            }
        )

    decode_fields = [_field_spec(key) for key in profile.get("decode_fields") or []]
    if decode_fields:
        groups.append(
            {
                "id": "decode",
                "label": "Decode defaults",
                "description": "Greedy, sampling, or beam-search controls when supported by the family.",
                "fields": decode_fields,
            }
        )

    generation_fields = [_field_spec(key) for key in profile.get("generation_fields") or []]
    if generation_fields:
        groups.append(
            {
                "id": "generation",
                "label": "Generation limits",
                "description": "Token and length limits for transcript generation.",
                "fields": generation_fields,
            }
        )

    option_fields = [
        _field_spec(key, nested=True)
        for key in profile.get("request_option_fields") or []
    ]
    if option_fields:
        groups.append(
            {
                "id": "options",
                "label": "Model request options",
                "description": "Family-specific options passed in the transcription request options object.",
                "fields": option_fields,
            }
        )
    return groups


def sidecar_session_fields_for_family(family: Optional[str]) -> List[Dict[str, Any]]:
    """Thin curated overlays for companion paths the --help scan cannot label well.

    Most session/load options come from ``audiocpp_cli --model … --help`` at install
    time. Keep this list small (path pickers / product hints only).
    """
    profile = asr_profile_for_family(family) or {}
    fields: List[Dict[str, Any]] = []
    for key in profile.get("sidecar_session_fields") or []:
        spec = _field_spec(key)
        spec["scope"] = "session_option"
        fields.append(spec)
    return fields


def _field_spec(key: str, *, nested: bool = False) -> Dict[str, Any]:
    spec = dict(_FIELD_SPECS.get(key) or {"key": key, "label": key, "type": "string"})
    if nested:
        spec["nested"] = True
        spec["options_key"] = key
    return spec


_FIELD_SPECS: Dict[str, Dict[str, Any]] = {
    "language": {
        "key": "language",
        "label": "Language",
        "type": "string",
        "placeholder": "en-US",
        "transcription_field": "language",
    },
    "prompt": {
        "key": "prompt",
        "label": "ASR prompt / context",
        "type": "textarea",
        "placeholder": "Transcribe the speech.",
        "transcription_field": "prompt",
        "options_key": "text",
        "nested": True,
    },
    "stream": {
        "key": "stream",
        "label": "Stream partial transcripts",
        "type": "bool",
        "transcription_field": "stream",
    },
    "max_tokens": {
        "key": "max_tokens",
        "label": "Max tokens",
        "type": "int",
        "transcription_field": "max_tokens",
        "options_key": "max_tokens",
        "nested": True,
    },
    "temperature": {
        "key": "temperature",
        "label": "Temperature",
        "type": "float",
        "transcription_field": "temperature",
        "options_key": "temperature",
        "nested": True,
    },
    "top_p": {
        "key": "top_p",
        "label": "Top P",
        "type": "float",
        "transcription_field": "top_p",
        "options_key": "top_p",
        "nested": True,
    },
    "top_k": {
        "key": "top_k",
        "label": "Top K",
        "type": "int",
        "transcription_field": "top_k",
        "options_key": "top_k",
        "nested": True,
    },
    "num_beams": {
        "key": "num_beams",
        "label": "Beam count",
        "type": "int",
        "options_key": "num_beams",
        "nested": True,
    },
    "do_sample": {
        "key": "do_sample",
        "label": "Stochastic sampling",
        "type": "bool",
        "options_key": "do_sample",
        "nested": True,
    },
    "seed": {
        "key": "seed",
        "label": "Seed",
        "type": "int",
        "options_key": "seed",
        "nested": True,
    },
    "repetition_penalty": {
        "key": "repetition_penalty",
        "label": "Repetition penalty",
        "type": "float",
        "options_key": "repetition_penalty",
        "nested": True,
    },
    "audio_chunk_mode": {
        "key": "audio_chunk_mode",
        "label": "Audio chunk mode",
        "type": "string",
        "placeholder": "auto",
        "options_key": "audio_chunk_mode",
        "nested": True,
    },
    "audio_chunk_seconds": {
        "key": "audio_chunk_seconds",
        "label": "Audio chunk seconds",
        "type": "float",
        "options_key": "audio_chunk_seconds",
        "nested": True,
    },
    "enable_thinking": {
        "key": "enable_thinking",
        "label": "Enable thinking prompt",
        "type": "bool",
        "options_key": "enable_thinking",
        "nested": True,
    },
    "punctuation": {
        "key": "punctuation",
        "label": "Enable punctuation",
        "type": "bool",
        "options_key": "punctuation",
        "nested": True,
    },
    "length_penalty": {
        "key": "length_penalty",
        "label": "Beam length penalty",
        "type": "float",
        "options_key": "length_penalty",
        "nested": True,
    },
    "lookahead_tokens": {
        "key": "lookahead_tokens",
        "label": "Lookahead tokens",
        "type": "int",
        "options_key": "lookahead_tokens",
        "nested": True,
    },
    "keep_language_tags": {
        "key": "keep_language_tags",
        "label": "Keep language tags",
        "type": "bool",
        "options_key": "keep_language_tags",
        "nested": True,
    },
    "qwen3_asr.forced_aligner_model_path": {
        "key": "qwen3_asr.forced_aligner_model_path",
        "label": "Forced aligner model path",
        "type": "path",
        "placeholder": "/data/models/audio-cpp/qwen3_forced_aligner_0_6b/Qwen3-ForcedAligner-0.6B",
        "description": (
            "Path to an installed Qwen3 Forced Aligner bundle. Required for word "
            "timestamps (aligned ASR). Install package qwen3_forced_aligner_0_6b first."
        ),
        "scope": "session_option",
    },
    "qwen3_asr.vad_model_path": {
        "key": "qwen3_asr.vad_model_path",
        "label": "VAD model path (timestamp chunking)",
        "type": "path",
        "placeholder": "assets/framework/models/silero_vad",
        "description": (
            "Optional VAD used when word timestamps are enabled and audio is long "
            "enough to need chunking."
        ),
        "scope": "session_option",
    },
}

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "citrinet": {
        "label": "Citrinet ASR",
        "workflows": ["offline"],
        "summary": "Offline CTC ASR. Provide 16 kHz WAV input.",
        "context_fields": ["language"],
        "api_hint": "Simple offline transcription. Language hint is optional.",
    },
    "higgs_audio_stt": {
        "label": "Higgs Audio STT",
        "workflows": ["offline", "streaming"],
        "summary": "Offline or streaming ASR with optional prompt text and long-audio chunking.",
        "context_fields": ["language", "prompt"],
        "session_fields": ["stream"],
        "generation_fields": ["max_tokens"],
        "chunk_fields": ["audio_chunk_mode", "audio_chunk_seconds"],
        # Nested options (enable_thinking, …) come from model --help / source scan.
        "api_hint": "Use stream=true for partial transcripts when the server mode is streaming.",
    },
    "hviske": {
        "label": "Hviske ASR",
        "workflows": ["offline"],
        "summary": "Danish offline ASR with punctuation and beam-search controls.",
        "context_fields": ["language"],
        "generation_fields": ["max_tokens"],
        "decode_fields": ["num_beams", "do_sample", "temperature", "top_k", "top_p", "seed"],
        "chunk_fields": ["audio_chunk_mode", "audio_chunk_seconds"],
        # Nested options (punctuation, length_penalty, …) come from model scan.
        "api_hint": "Default language is Danish (da). num_beams=1 uses greedy or sampling decode.",
    },
    "nemotron_asr": {
        "label": "Nemotron ASR",
        "workflows": ["offline", "streaming"],
        "summary": "RNNT ASR with language prompts and optional token timestamps.",
        "context_fields": ["language"],
        "session_fields": ["stream"],
        "generation_fields": ["max_tokens"],
        # Nested options (lookahead_tokens, keep_language_tags, …) come from model scan.
        "api_hint": "Supports en-US, da-DK, auto language hints. Use stream=true for streaming mode.",
    },
    "vibevoice": {
        "label": "VibeVoice ASR",
        "workflows": ["offline"],
        "summary": "Offline ASR with optional segment and speaker-turn structured output.",
        "context_fields": ["language", "prompt"],
        "generation_fields": ["max_tokens"],
        "decode_fields": [
            "temperature",
            "top_p",
            "top_k",
            "num_beams",
            "seed",
            "repetition_penalty",
        ],
        "chunk_fields": ["audio_chunk_mode", "audio_chunk_seconds"],
        "api_hint": "Use a meeting-context prompt for multi-speaker recordings.",
    },
    "qwen3_asr": {
        "label": "Qwen3 ASR",
        "workflows": ["offline"],
        "summary": "Qwen3 ASR with chunking and optional word timestamps via forced aligner.",
        "context_fields": ["language", "prompt"],
        "generation_fields": ["max_tokens"],
        "chunk_fields": ["audio_chunk_mode", "audio_chunk_seconds"],
        "sidecar_session_fields": [
            "qwen3_asr.forced_aligner_model_path",
            "qwen3_asr.vad_model_path",
        ],
        "api_hint": (
            "Aligned ASR (word timestamps) requires session option "
            "qwen3_asr.forced_aligner_model_path pointing at an installed "
            "qwen3_forced_aligner_0_6b bundle. For long audio, prefer this ASR+aligner "
            "path over the standalone align task."
        ),
    },
    "parakeet_tdt": {
        "label": "Parakeet TDT",
        "workflows": ["offline", "streaming"],
        "summary": "NVIDIA Parakeet TDT ASR with offline and streaming modes.",
        "context_fields": ["language"],
        "session_fields": ["stream"],
        "api_hint": "Use stream=true when the server is configured with mode=streaming.",
    },
}
