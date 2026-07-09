"""Normalize Studio-only request defaults for all audio.cpp task types."""

from __future__ import annotations

from typing import Any, Dict

from backend.audio_transcription_defaults import normalize_transcription_defaults
from backend.audio_voice_presets import normalize_speech_defaults


def normalize_task_defaults(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, raw in value.items():
        if key == "options":
            continue
        if raw is None or raw == "":
            continue
        if key == "stream":
            out[key] = bool(raw)
        elif key in {"seed", "max_tokens", "top_k", "num_inference_steps", "text_chunk_size"}:
            try:
                out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        elif key in {
            "temperature",
            "top_p",
            "guidance_scale",
            "repetition_penalty",
            "duration_seconds",
            "repaint_start",
            "repaint_end",
            "init_noise_level",
            "codec_duration",
            "codec_guidance_scale",
            "length_adjust",
            "intelligibility_cfg_rate",
            "similarity_cfg_rate",
            "inference_cfg_rate",
            "target_duration_seconds",
            "audio_chunk_seconds",
            "vad_chunk_max_seconds",
            "vad_chunk_merge_gap_seconds",
            "vad_chunk_padding_seconds",
            "threshold",
            "neg_threshold",
        }:
            try:
                out[key] = float(raw)
            except (TypeError, ValueError):
                continue
        elif isinstance(raw, bool):
            out[key] = raw
        else:
            text = str(raw).strip()
            if text:
                out[key] = text
    options = value.get("options")
    if isinstance(options, dict):
        cleaned: Dict[str, Any] = {}
        for raw_key, raw_value in options.items():
            key = str(raw_key).strip()
            if not key or raw_value is None:
                continue
            if isinstance(raw_value, bool):
                cleaned[key] = raw_value
            elif isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool):
                cleaned[key] = raw_value
            else:
                text = str(raw_value).strip()
                if text:
                    cleaned[key] = text
        if cleaned:
            out["options"] = cleaned
    return out


def normalize_request_defaults(key: str, value: Any) -> Dict[str, Any]:
    if key == "speech_defaults":
        return normalize_speech_defaults(value)
    if key == "transcription_defaults":
        return normalize_transcription_defaults(value)
    return normalize_task_defaults(value)
