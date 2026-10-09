"""TTS protocol helpers; family metadata belongs to the engine."""

from typing import Optional

from backend.engines.audio_cpp.contracts import load_family_contract
from backend.engines.audio_cpp.spec_fields import field_groups_from_contract, profile_from_contract


def is_tts_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in {"clon", "vdes", "tts"}


def tts_profile_for_family(
    family: Optional[str], source_path: Optional[str] = None
) -> Optional[dict]:
    return profile_from_contract(load_family_contract(source_path, str(family or "")))


def speech_request_field_groups(family: Optional[str], source_path: Optional[str] = None) -> list:
    return field_groups_from_contract(load_family_contract(source_path, str(family or "")))


def family_requires_session_voice(family: Optional[str], source_path: Optional[str] = None) -> bool:
    from backend.engines.audio_cpp.voices import spec_default_voice

    return bool(spec_default_voice(source_path, family))
