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


def _reference_roots(model_root: str, reference_root: Optional[str] = None) -> list[str]:
    roots: list[str] = []
    for root in (reference_root, model_root):
        text = _clean_text(root)
        if text:
            resolved = os.path.abspath(text)
            if resolved not in roots:
                roots.append(resolved)
    return roots


def _resolve_voice_ref(
    model_root: str,
    value: Any,
    *,
    reference_root: Optional[str] = None,
) -> Optional[str]:
    raw = _clean_text(value)
    if not raw:
        return None
    if os.path.isabs(raw):
        return raw
    roots = _reference_roots(model_root, reference_root)
    for root in roots:
        candidate = os.path.abspath(os.path.join(root, raw))
        if os.path.exists(candidate):
            return candidate
    root = roots[0] if roots else os.path.abspath(model_root)
    return os.path.abspath(os.path.join(root, raw))


def _relative_voice_ref_escapes(
    model_root: str,
    value: Any,
    *,
    reference_root: Optional[str] = None,
) -> bool:
    raw = _clean_text(value)
    if not raw or os.path.isabs(raw):
        return False
    escaped_all_roots = True
    for root in _reference_roots(model_root, reference_root):
        root_real = os.path.realpath(root)
        target = os.path.realpath(os.path.join(root_real, raw))
        try:
            if os.path.commonpath([root_real, target]) == root_real:
                escaped_all_roots = False
        except ValueError:
            continue
    return escaped_all_roots


def _validate_voice_ref_path(
    *,
    label: str,
    value: Any,
    model_root: str,
    reference_root: Optional[str] = None,
    errors: list[str],
) -> None:
    raw = _clean_text(value)
    if not raw:
        return
    if _relative_voice_ref_escapes(model_root, raw, reference_root=reference_root):
        errors.append(f"{label} escapes model bundle: {raw}")
        return
    resolved = _resolve_voice_ref(model_root, raw, reference_root=reference_root)
    if not resolved or not os.path.exists(resolved):
        errors.append(f"{label} does not exist: {resolved or raw}")


def normalize_voice_preset(
    preset: Any,
    *,
    model_root: str,
    reference_root: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    if not isinstance(preset, dict):
        return None
    out: Dict[str, str] = {}
    voice_id = _clean_text(preset.get("voice_id"))
    if voice_id:
        out["voice_id"] = voice_id
    voice_ref = _resolve_voice_ref(
        model_root,
        preset.get("voice_ref"),
        reference_root=reference_root,
    )
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
    reference_root: Optional[str] = None,
) -> Dict[str, Dict[str, str]]:
    if not isinstance(presets, dict):
        return {}
    out: Dict[str, Dict[str, str]] = {}
    for raw_name, raw_preset in presets.items():
        name = str(raw_name or "").strip()
        if not name:
            continue
        normalized = normalize_voice_preset(
            raw_preset,
            model_root=model_root,
            reference_root=reference_root,
        )
        if normalized:
            out[name] = normalized
    return out


def normalize_default_voice_preset(
    value: Any,
    *,
    model_root: str,
    reference_root: Optional[str] = None,
    voice_presets: Optional[Dict[str, Dict[str, str]]] = None,
) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        name = value.strip()
        return name or None
    if isinstance(value, dict):
        normalized = normalize_voice_preset(
            value,
            model_root=model_root,
            reference_root=reference_root,
        )
        return normalized
    return None


_SPEECH_SWAP_PARAM_KEYS = frozenset(
    {
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
    }
)


def _speech_normalized_to_swap_params(normalized: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key in _SPEECH_SWAP_PARAM_KEYS:
        if key in normalized:
            params[key] = normalized[key]
    options = normalized.get("options")
    if isinstance(options, dict) and options:
        params["options"] = dict(options)
    return params


def _transcription_normalized_to_swap_params(normalized: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    options: Dict[str, Any] = dict(normalized.get("options") or {})
    for key in ("language", "stream"):
        if key in normalized:
            params[key] = normalized[key]
    for key in ("max_tokens", "temperature", "top_p", "top_k", "seed"):
        if key in normalized:
            options[key] = normalized[key]
    prompt = normalized.get("prompt")
    if prompt:
        options["text"] = prompt
    if options:
        params["options"] = options
    return params


def _task_normalized_to_swap_params(normalized: Dict[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    options: Dict[str, Any] = dict(normalized.get("options") or {})
    for key, value in normalized.items():
        if key in {"options", "prompt"}:
            continue
        params[key] = value
    prompt = normalized.get("prompt")
    if prompt:
        options["text"] = prompt
    if options:
        existing = params.get("options")
        if isinstance(existing, dict):
            params["options"] = {**options, **existing}
        else:
            params["options"] = options
    return params


def _resolve_normalized_reference_fields(
    normalized: Dict[str, Any],
    *,
    model_root: Optional[str] = None,
    reference_root: Optional[str] = None,
) -> Dict[str, Any]:
    if not model_root or "voice_ref" not in normalized:
        return normalized
    resolved = _resolve_voice_ref(
        model_root,
        normalized.get("voice_ref"),
        reference_root=reference_root,
    )
    if not resolved:
        return normalized
    return {**normalized, "voice_ref": resolved}


def audio_request_defaults_to_swap_set_params(
    config: dict,
    *,
    model_root: Optional[str] = None,
    reference_root: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Map Studio request defaults to llama-swap ``filters.setParams`` for audio.cpp.

    llama-swap injects these into JSON request bodies before they reach audio.cpp.
    """
    if str(config.get("engine") or "") != "audio_cpp":
        return {}
    from backend.audio_request_defaults import normalize_request_defaults
    from backend.audio_task_profiles import request_defaults_key_for

    defaults_key = request_defaults_key_for(config.get("task"), config.get("family"))
    normalized = normalize_request_defaults(defaults_key, config.get(defaults_key))
    normalized = _resolve_normalized_reference_fields(
        normalized,
        model_root=model_root,
        reference_root=reference_root,
    )
    if not normalized:
        return {}
    if defaults_key == "speech_defaults":
        return _speech_normalized_to_swap_params(normalized)
    if defaults_key == "transcription_defaults":
        return _transcription_normalized_to_swap_params(normalized)
    return _task_normalized_to_swap_params(normalized)


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
    reference_root: Optional[str] = None,
    errors: list[str],
) -> Dict[str, Dict[str, str]]:
    presets = normalize_voice_presets(
        config.get("voice_presets"),
        model_root=model_root,
        reference_root=reference_root,
    )
    default_value = config.get("default_voice_preset")
    if isinstance(default_value, str):
        name = default_value.strip()
        if name and name not in presets:
            errors.append(f"default_voice_preset '{name}' is not defined in voice_presets")
    elif isinstance(default_value, dict):
        if not normalize_voice_preset(
            default_value,
            model_root=model_root,
            reference_root=reference_root,
        ):
            errors.append(
                "default_voice_preset must include voice_id, voice_ref, or reference_text"
            )
        for path_key in ("voice_ref",):
            raw = default_value.get(path_key)
            _validate_voice_ref_path(
                label=f"default_voice_preset {path_key}",
                value=raw,
                model_root=model_root,
                reference_root=reference_root,
                errors=errors,
            )
    raw_presets = config.get("voice_presets")
    raw_presets = raw_presets if isinstance(raw_presets, dict) else {}
    for name, preset in presets.items():
        raw_preset = raw_presets.get(name)
        raw_voice_ref = (
            raw_preset.get("voice_ref")
            if isinstance(raw_preset, dict)
            else preset.get("voice_ref")
        )
        _validate_voice_ref_path(
            label=f"voice_presets.{name}.voice_ref",
            value=raw_voice_ref,
            model_root=model_root,
            reference_root=reference_root,
            errors=errors,
        )
    return presets


def validate_speech_default_references(
    config: dict,
    *,
    model_root: str,
    reference_root: Optional[str] = None,
    errors: list[str],
) -> None:
    defaults = config.get("speech_defaults")
    if not isinstance(defaults, dict):
        return
    _validate_voice_ref_path(
        label="speech_defaults.voice_ref",
        value=defaults.get("voice_ref"),
        model_root=model_root,
        reference_root=reference_root,
        errors=errors,
    )
