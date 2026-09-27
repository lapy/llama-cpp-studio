"""Curated source separation family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio.profiles import TaskProfileSet

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
    "bs_roformer": {
        "label": "BS-RoFormer",
        "workflows": ["offline"],
        "summary": "Band-split RoFormer vocal separation for 44.1 kHz mixtures.",
        "audio_fields": ["audio"],
        "api_hint": "Use 44.1 kHz input. Stems are written to the output directory.",
    },
}

SEP_PROFILES = TaskProfileSet(
    tasks=frozenset({"sep"}),
    profiles=_FAMILY_PROFILES,
    group_meta=_GROUP_META,
)


def is_sep_task(task: Optional[str]) -> bool:
    return SEP_PROFILES.matches_task(task)


def sep_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    return SEP_PROFILES.profile_for_family(family)


def separation_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    return SEP_PROFILES.field_groups(family)
