"""Family-specific validation for saved audio.cpp request defaults."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_omnivoice_instruct import validate_omnivoice_instruct
from backend.audio_task_profiles import request_defaults_key_for

_TASK_RUN_INSTEAD_OF_SPEECH_FAMILIES = frozenset({"vevo2", "seed_vc", "miocodec"})


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
) -> List[str]:
    """Return user-facing errors for mismatched or invalid saved request defaults."""
    errors: List[str] = []
    family_key = _family_key(family)
    task_key = _task_key(task)
    expected_key = request_defaults_key_for(task_key, family_key)

    speech_defaults = config.get("speech_defaults")
    transcription_defaults = config.get("transcription_defaults")
    task_defaults = config.get("task_defaults")

    if expected_key != "speech_defaults" and _defaults_has_content(speech_defaults):
        errors.append(
            f"{family_key} routes through /v1/tasks/run and uses task_defaults, not "
            "speech_defaults. Move these fields to task_defaults."
        )
    if expected_key != "transcription_defaults" and _defaults_has_content(
        transcription_defaults
    ):
        errors.append(
            f"{family_key} does not use transcription_defaults for task '{task_key}'. "
            f"Use {expected_key} instead."
        )
    if expected_key != "task_defaults" and _defaults_has_content(task_defaults):
        errors.append(
            f"{family_key} uses {expected_key}, not task_defaults. "
            "Move these fields to the matching defaults object."
        )

    active_defaults = config.get(expected_key)
    instructions = _instructions_value(active_defaults)
    if not instructions:
        return errors

    if family_key == "omnivoice" and expected_key == "speech_defaults":
        errors.extend(validate_omnivoice_instruct(instructions))
    elif family_key == "voxcpm2" and expected_key == "speech_defaults":
        errors.append(
            "VoxCPM2 does not use instructions. Put style prefixes at the start of "
            "input text instead, e.g. '(calm) Hello world'."
        )
    elif family_key == "irodori_tts" and expected_key == "speech_defaults":
        errors.append(
            "Irodori voice design uses options.caption, not instructions. "
            "Set caption under speech_defaults.options.caption."
        )

    return errors
