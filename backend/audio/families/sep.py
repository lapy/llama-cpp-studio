"""SEP protocol helpers; family metadata belongs to the engine."""

from typing import Optional

from backend.engines.audio_cpp.contracts import load_family_contract
from backend.engines.audio_cpp.spec_fields import field_groups_from_contract, profile_from_contract


def is_sep_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in {"sep"}


def sep_profile_for_family(
    family: Optional[str], source_path: Optional[str] = None
) -> Optional[dict]:
    return profile_from_contract(load_family_contract(source_path, str(family or "")))


def separation_request_field_groups(
    family: Optional[str], source_path: Optional[str] = None
) -> list:
    return field_groups_from_contract(load_family_contract(source_path, str(family or "")))
