"""Schema-driven validation for saved audio.cpp request defaults."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_request_policy import (
    build_request_policy,
    validate_instructions_against_policy,
)
from backend.audio_task_profiles import request_defaults_key_for


def _family_key(family: Optional[str]) -> str:
    return str(family or "").strip().lower()


def _task_key(task: Optional[str]) -> str:
    return str(task or "").strip().lower()


def _defaults_has_content(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key, item in value.items():
        if key == "options":
            if isinstance(item, dict) and any(
                nested is not None and nested != "" for nested in item.values()
            ):
                return True
            continue
        if item is not None and item != "" and item != []:
            return True
    return False


def _instructions_value(defaults: Optional[dict]) -> Optional[str]:
    if not isinstance(defaults, dict):
        return None
    instructions = defaults.get("instructions")
    if instructions is None or instructions == "":
        return None
    return str(instructions).strip() or None


def validate_saved_request_defaults(
    *,
    task: Optional[str],
    family: Optional[str],
    config: dict,
    inspection: Optional[dict] = None,
    model_profile: Optional[dict] = None,
    source_path: Optional[str] = None,
) -> List[str]:
    """Return user-facing errors for mismatched or invalid saved request defaults."""
    errors: List[str] = []
    family_key = _family_key(family)
    task_key = _task_key(task)
    policy = build_request_policy(
        task=task_key,
        family=family_key,
        inspection=inspection,
        model_profile=model_profile,
        source_path=source_path,
    )
    expected_key = str(
        policy.get("request_defaults_key")
        or request_defaults_key_for(task_key, family_key)
    )

    speech_defaults = config.get("speech_defaults")
    transcription_defaults = config.get("transcription_defaults")
    task_defaults = config.get("task_defaults")

    if expected_key != "speech_defaults" and _defaults_has_content(speech_defaults):
        errors.append(
            f"{family_key or 'model'} routes through {policy.get('api_endpoint')} and uses "
            f"{expected_key}, not speech_defaults. Move these fields to {expected_key}."
        )
    if expected_key != "transcription_defaults" and _defaults_has_content(
        transcription_defaults
    ):
        errors.append(
            f"{family_key or 'model'} does not use transcription_defaults for task '{task_key}'. "
            f"Use {expected_key} instead."
        )
    if expected_key != "task_defaults" and _defaults_has_content(task_defaults):
        errors.append(
            f"{family_key or 'model'} uses {expected_key}, not task_defaults. "
            "Move these fields to the matching defaults object."
        )

    active_defaults = config.get(expected_key)
    instructions = _instructions_value(active_defaults)
    if not instructions:
        return errors

    errors.extend(
        validate_instructions_against_policy(
            instructions,
            policy=str(policy.get("instructions_policy") or "none"),
            family=family_key,
            vocabulary=policy.get("instructions_vocabulary"),
        )
    )
    return errors
