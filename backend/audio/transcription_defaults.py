"""Normalize Studio-only transcription request defaults for ASR models."""

from __future__ import annotations

from typing import Any, Dict

from backend.engines.audio_cpp.request_vocab import request_defaults_vocab


def normalize_transcription_defaults(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    vocab = request_defaults_vocab("transcription")
    bool_keys = list(vocab.get("bool_keys") or ())
    int_keys = set(vocab.get("int_keys") or ())
    float_keys = set(vocab.get("float_keys") or ())
    out: Dict[str, Any] = {}
    if "language" in value:
        text = str(value.get("language") or "").strip()
        if text:
            out["language"] = text
    # Boolean defaults: exactly the DECLARED keys, always coerced when present.
    for key in bool_keys:
        if key in value:
            out[key] = bool(value.get(key))
    if "prompt" in value:
        text = str(value.get("prompt") or "").strip()
        if text:
            out["prompt"] = text
    for key, raw in value.items():
        if key in ("language", "prompt", "options") or key in bool_keys:
            continue
        if key not in int_keys and key not in float_keys:
            continue
        if raw is None or raw == "":
            continue
        if key in int_keys:
            try:
                out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        else:
            try:
                out[key] = float(raw)
            except (TypeError, ValueError):
                continue
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
