"""ANALYSIS protocol helpers; family metadata belongs to the engine."""

from typing import Optional

from backend.engines.audio_cpp.contracts import load_family_contract
from backend.engines.audio_cpp.spec_fields import field_groups_from_contract, profile_from_contract


def is_analysis_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in {"vad", "diar"}


def analysis_profile_for_family(
    family: Optional[str], source_path: Optional[str] = None
) -> Optional[dict]:
    return profile_from_contract(load_family_contract(source_path, str(family or "")))


def analysis_request_field_groups(family: Optional[str], source_path: Optional[str] = None) -> list:
    return field_groups_from_contract(load_family_contract(source_path, str(family or "")))


def is_vad_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() == "vad"


def is_diar_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() == "diar"
