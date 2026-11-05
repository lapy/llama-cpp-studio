"""Normalize Studio-only request defaults for all audio.cpp task types."""

from __future__ import annotations

from typing import Any, Dict

from backend.audio.transcription_defaults import normalize_transcription_defaults
from backend.audio.voice_presets import normalize_speech_defaults
from backend.engines.audio_cpp.request_vocab import request_defaults_vocab


def normalize_task_defaults(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    vocab = request_defaults_vocab("task")
    bool_keys = set(vocab.get("bool_keys") or ())
    int_keys = set(vocab.get("int_keys") or ())
    float_keys = set(vocab.get("float_keys") or ())
    out: Dict[str, Any] = {}
    for key, raw in value.items():
        if key == "options":
            continue
        if raw is None or raw == "":
            continue
        if key in bool_keys:
            out[key] = bool(raw)
        elif key in int_keys:
            try:
                out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        elif key in float_keys:
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
