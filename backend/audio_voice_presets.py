"""Normalize voice presets and speech defaults for audio.cpp sidecars."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


_PRESET_KEYS = frozenset({"voice_id", "voice_ref", "reference_text"})


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_voice_ref(model_root: str, value: Any) -> Optional[str]:
    raw = _clean_text(value)
    if not raw:
        return None
    if os.path.isabs(raw):
        return raw
    root = os.path.abspath(model_root)
    return os.path.abspath(os.path.join(root, raw))


def normalize_voice_preset(preset: Any, *, model_root: str) -> Optional[Dict[str, str]]:
    if not isinstance(preset, dict):
        return None
    out: Dict[str, str] = {}
    voice_id = _clean_text(preset.get("voice_id"))
    if voice_id:
        out["voice_id"] = voice_id
    voice_ref = _resolve_voice_ref(model_root, preset.get("voice_ref"))
    if voice_ref:
        out["voice_ref"] = voice_ref
    reference_text = _clean_text(preset.get("reference_text"))
    if reference_text:
        out["reference_text"] = reference_text
    return out or None


def normalize_voice_presets(
    presets: Any,
    *,
    model_root: str,
) -> Dict[str, Dict[str, str]]:
    if not isinstance(presets, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for raw_name, raw_preset in presets.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        normalized = normalize_voice_preset(raw_preset, model_root=model_root)
        if normalized:
            out[name] = normalized
    return out


def normalize_default_voice_preset(
    value: Any,
    *,
    model_root: str,
    voice_presets: Optional[Dict[str, Dict[str, str]]] = None,
) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        name = value.strip()
        return name or None
    if isinstance(value, dict):
        normalized = normalize_voice_preset(value, model_root=model_root)
        return normalized
    return None


def normalize_speech_defaults(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for key in (
        "voice",
        "voice_ref",
        "reference_text",
        "instructions",
        "language",
        "seed",
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "guidance_scale",
        "num_inference_steps",
        "repetition_penalty",
    ):
        if key not in value:
            continue
        raw = value.get(key)
        if raw is None or raw == "":
            continue
        if key in {"seed", "top_k", "max_tokens", "num_inference_steps"}:
            try:
                out[key] = int(raw)
            except (TypeError, ValueError):
                continue
        elif key in {
            "temperature",
            "top_p",
            "guidance_scale",
            "repetition_penalty",
        }:
            try:
                out[key] = float(raw)
            except (TypeError, ValueError):
                continue
        else:
            text = str(raw).strip()
            if text:
                out[key] = text
    options = value.get("options")
    if isinstance(options, dict):
        cleaned = {
            str(k): str(v).strip() if v is not None else ""
            for k, v in options.items()
            if str(k).strip() and v is not None and str(v).strip() != ""
        }
        if cleaned:
            out["options"] = cleaned
    return out


def validate_voice_presets(
    config: dict,
    *,
    model_root: str,
    errors: list[str],
) -> Dict[str, Dict[str, str]]:
    presets = normalize_voice_presets(config.get("voice_presets"), model_root=model_root)
    default_value = config.get("default_voice_preset")
    if isinstance(default_value, str):
        name = default_value.strip()
        if name and name not in presets:
            errors.append(f"default_voice_preset '{name}' is not defined in voice_presets")
    elif isinstance(default_value, dict):
        if not normalize_voice_preset(default_value, model_root=model_root):
            errors.append(
                "default_voice_preset must include voice_id, voice_ref, or reference_text"
            )
        for path_key in ("voice_ref",):
            raw = default_value.get(path_key)
            if raw and not os.path.exists(_resolve_voice_ref(model_root, raw) or ""):
                errors.append(f"default_voice_preset {path_key} does not exist: {raw}")
    for name, preset in presets.items():
        voice_ref = preset.get("voice_ref")
        if voice_ref and not os.path.exists(voice_ref):
            errors.append(f"voice_presets.{name}.voice_ref does not exist: {voice_ref}")
    return presets
