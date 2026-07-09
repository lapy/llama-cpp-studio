"""Validation for model-aware audio.cpp configuration."""

from __future__ import annotations

import os
import shlex
from typing import Any, Dict, Iterable, List, Optional

from backend.engine_param_catalog import get_model_profile_entry
from backend.engine_param_scanner import (
    audio_cpp_model_profile_fingerprint,
    scan_audio_cpp_model_profile,
)
from backend.engine_registry import active_engine_row_is_runnable
from backend.feature_flags import audio_cpp_enabled
from backend.audio_asr_profiles import is_asr_task
from backend.audio_tts_profiles import is_tts_task
from backend.audio_voice_presets import validate_voice_presets
from backend.model_config import effective_model_config


_RESERVED_AUDIO_FLAGS = {
    "--config",
    "--host",
    "--port",
    "--model",
}
_NESTED_SCOPE_KEYS = {
    "load_option": "load_options",
    "session_option": "session_options",
    "request_option": "request_options",
}


def _present(value: Any) -> bool:
    return value is not None and value != "" and value != []


def _type_matches(value: Any, row: dict) -> bool:
    expected = str(row.get("type") or row.get("scalar_type") or "string")
    if expected == "bool":
        return isinstance(value, bool)
    if expected == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected in {"list", "multiselect"}:
        return isinstance(value, list)
    if expected == "json":
        return isinstance(value, dict)
    return isinstance(value, str)


def _row_value(config: dict, row: dict) -> Any:
    scope = str(row.get("scope") or "process")
    key = str(row.get("key") or "")
    nested_key = _NESTED_SCOPE_KEYS.get(scope)
    if nested_key:
        nested = config.get(nested_key)
        return nested.get(key) if isinstance(nested, dict) else None
    return config.get(key)


def _validate_param_value(row: dict, value: Any, errors: List[str]) -> None:
    key = str(row.get("key") or "parameter")
    if not _present(value):
        if row.get("required"):
            errors.append(f"{key} is required")
        return
    if not _type_matches(value, row):
        expected = str(row.get("type") or row.get("scalar_type") or "string")
        errors.append(f"{key} must be {expected}")
        return
    options = [
        option.get("value")
        for option in row.get("options") or []
        if isinstance(option, dict) and "value" in option
    ]
    if options:
        selected = value if isinstance(value, list) else [value]
        invalid = [item for item in selected if item not in options]
        if invalid:
            errors.append(f"{key} has unsupported value(s): {invalid}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = row.get("minimum")
        maximum = row.get("maximum")
        if minimum is not None and value < minimum:
            errors.append(f"{key} must be at least {minimum}")
        if maximum is not None and value > maximum:
            errors.append(f"{key} must be at most {maximum}")


def _asset_path_exists(model: dict, asset: dict) -> bool:
    raw = str(asset.get("path") or "")
    if not raw:
        return False
    if os.path.isabs(raw):
        return os.path.exists(raw)
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    model_path = str(
        artifact.get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    )
    candidates = [
        os.path.join(model_path, raw),
        os.path.join(os.path.dirname(model_path), raw),
    ]
    return any(os.path.exists(path) for path in candidates)


def _selected_asset(
    model: dict,
    config: dict,
    inspection: dict,
    config_key: str,
    inspection_key: str,
    errors: List[str],
) -> None:
    selected = config.get(config_key)
    if not _present(selected):
        return
    assets = [
        item
        for item in inspection.get(inspection_key) or []
        if isinstance(item, dict)
    ]
    asset = next((item for item in assets if item.get("id") == selected), None)
    if not asset:
        errors.append(f"{config_key} is not exposed by the inspected package")
        return
    if not _asset_path_exists(model, asset):
        errors.append(f"Selected {config_key} asset does not exist: {selected}")


def _validate_custom_args(value: Any, errors: List[str]) -> None:
    if not value:
        return
    try:
        tokens = shlex.split(str(value))
    except ValueError as exc:
        errors.append(f"custom_args could not be parsed: {exc}")
        return
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag in _RESERVED_AUDIO_FLAGS:
            errors.append(f"{flag} is Studio-owned and cannot be set in custom_args")


def validate_audio_model_config(
    store: Any,
    model: dict,
    normalized_config: dict,
    *,
    allow_scan: bool = True,
) -> Dict[str, Any]:
    """Validate the active audio.cpp section, returning profile metadata.

    Raises ``ValueError`` with all user-actionable validation failures.
    """

    effective = effective_model_config(normalized_config)
    if effective.get("engine") != "audio_cpp":
        return {"errors": [], "warnings": []}
    if not audio_cpp_enabled():
        raise ValueError(
            "The experimental audio.cpp integration is disabled by AUDIO_CPP_ENABLED"
        )

    errors: List[str] = []
    warnings: List[str] = []
    compatible = set(model.get("compatible_engines") or [])
    if compatible and "audio_cpp" not in compatible:
        errors.append("This model is not verified compatible with audio.cpp")

    active = store.get_active_engine_version("audio_cpp")
    if not active_engine_row_is_runnable("audio_cpp", active):
        errors.append("No runnable audio.cpp version is active")
        active = None

    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    model_path = str(
        artifact.get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    )
    if not model_path or not os.path.isdir(model_path):
        errors.append("The prepared audio.cpp model directory does not exist")

    profile: Dict[str, Any] = {}
    if active and model_path and os.path.isdir(model_path):
        if allow_scan:
            profile = scan_audio_cpp_model_profile(store, active, model, force=False)
        else:
            fingerprint = audio_cpp_model_profile_fingerprint(active, model)
            profile = (
                get_model_profile_entry(
                    store,
                    "audio_cpp",
                    str(active.get("version") or ""),
                    fingerprint,
                )
                or {}
            )
            if not profile:
                errors.append(
                    "No cached audio.cpp model profile is available; inspect the model before previewing runtime configuration"
                )
        if profile.get("scan_error"):
            errors.append(f"Model capability inspection failed: {profile['scan_error']}")
    inspection = profile.get("inspection") if isinstance(profile, dict) else {}
    if not isinstance(inspection, dict):
        inspection = {}

    family = effective.get("family")
    inspected_family = inspection.get("family") or model.get("family")
    if not family:
        errors.append("family is required")
    elif inspected_family and family != inspected_family:
        errors.append(
            f"family '{family}' does not match inspected family '{inspected_family}'"
        )

    task = effective.get("task")
    task_rows = [
        row for row in inspection.get("tasks") or [] if isinstance(row, dict)
    ]
    task_names = {
        str(row.get("task")) for row in task_rows if row.get("task")
    } or set(model.get("tasks") or [])
    if not task:
        errors.append("task is required")
    elif task_names and task not in task_names:
        errors.append(f"task '{task}' is not exposed by this package")

    mode = effective.get("mode")
    selected_task = next(
        (row for row in task_rows if str(row.get("task")) == str(task)),
        None,
    )
    allowed_modes = set((selected_task or {}).get("modes") or [])
    if not mode:
        errors.append("mode is required")
    elif allowed_modes and mode not in allowed_modes:
        errors.append(
            f"mode '{mode}' is not supported for task '{task}'"
        )

    selected_backend = str(effective.get("backend") or "cpu")
    build_config = active.get("build_config") if isinstance(active, dict) else {}
    built_backend = str((build_config or {}).get("backend") or "cpu")
    available_backends = {"cpu", built_backend}
    if selected_backend not in available_backends:
        errors.append(
            f"backend '{selected_backend}' is unavailable in the active "
            f"{built_backend} audio.cpp build"
        )

    request_options = effective.get("request_options")
    if isinstance(request_options, dict) and request_options:
        errors.append(
            "request_options are request-time capabilities and cannot be saved as server configuration"
        )

    model_root = str(
        (model.get("artifact") or {}).get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    )
    if is_tts_task(task):
        validate_voice_presets(
            effective,
            model_root=model_root,
            errors=errors,
        )
        if effective.get("speech_defaults") is not None and not isinstance(
            effective.get("speech_defaults"), dict
        ):
            errors.append("speech_defaults must be an object")
    if is_asr_task(task):
        if effective.get("transcription_defaults") is not None and not isinstance(
            effective.get("transcription_defaults"), dict
        ):
            errors.append("transcription_defaults must be an object")
    for defaults_key in ("task_defaults",):
        if effective.get(defaults_key) is not None and not isinstance(
            effective.get(defaults_key), dict
        ):
            errors.append(f"{defaults_key} must be an object")

    for section in profile.get("sections") or []:
        for row in section.get("params") or []:
            if not isinstance(row, dict) or row.get("reserved"):
                continue
            if str(row.get("scope") or "") == "request_option":
                continue
            _validate_param_value(row, _row_value(effective, row), errors)

    _selected_asset(model, effective, inspection, "config", "configs", errors)
    _selected_asset(model, effective, inspection, "weight", "weights", errors)
    _validate_custom_args(effective.get("custom_args"), errors)

    for nested_key in ("load_options", "session_options"):
        value = effective.get(nested_key)
        if value is not None and not isinstance(value, dict):
            errors.append(f"{nested_key} must be an object")

    if errors:
        raise ValueError("; ".join(dict.fromkeys(errors)))
    return {
        "errors": [],
        "warnings": warnings,
        "profile_fingerprint": profile.get("fingerprint"),
        "inspection": inspection,
    }

