"""Normalize Studio-only transcription request defaults for ASR models."""

from __future__ import annotations

from typing import Any, Dict


def normalize_transcription_defaults(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    if "language" in value:
        text = str(value.get("language") or "").strip()
        if text:
            out["language"] = text
    if "stream" in value:
        out["stream"] = bool(value.get("stream"))
    if "prompt" in value:
        text = str(value.get("prompt") or "").strip()
        if text:
            out["prompt"] = text
    for key in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
        if key not in value:
            continue
        raw = value.get(key)
        if raw is None or raw == "":
            continue
        if key in {"max_tokens", "top_k", "seed"}:
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
