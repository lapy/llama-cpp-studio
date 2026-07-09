"""Curated source separation family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_profile_fields import field_spec

_SEP_TASKS = frozenset({"sep"})


def is_sep_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _SEP_TASKS


def sep_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    key = str(family or "").strip().lower()
    return _FAMILY_PROFILES.get(key) if key else None


def separation_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    profile = sep_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []
    for group_id, label, description in _GROUP_META:
        keys = profile.get(f"{group_id}_fields") or []
        fields = [field_spec(key) for key in keys]
        if fields:
            groups.append({"id": group_id, "label": label, "description": description, "fields": fields})
    return groups


_GROUP_META = [
    ("audio", "Audio input", "44.1 kHz mixture WAV for separation."),
]

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "htdemucs": {
        "label": "HTDemucs",
        "workflows": ["offline"],
        "summary": "Separate music mixtures into vocals, drums, bass, and other stems.",
        "audio_fields": ["audio"],
        "api_hint": "Use 44.1 kHz input. Stems are written to the output directory.",
    },
    "mel_band_roformer": {
        "label": "Mel-Band RoFormer",
        "workflows": ["offline"],
        "summary": "Vocal/source separation for 44.1 kHz mixtures.",
        "audio_fields": ["audio"],
        "api_hint": "Chunking behavior is internal; use 44.1 kHz WAV input.",
    },
}
