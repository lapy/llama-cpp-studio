"""Curated forced-alignment family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_profile_fields import field_spec

_ALIGN_TASKS = frozenset({"align"})


def is_align_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _ALIGN_TASKS


def align_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    key = str(family or "").strip().lower()
    return _FAMILY_PROFILES.get(key) if key else None


def alignment_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    profile = align_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []
    for group_id, label, description in _GROUP_META:
        keys = profile.get(f"{group_id}_fields") or []
        fields = [field_spec(key) for key in keys]
        if fields:
            groups.append({"id": group_id, "label": label, "description": description, "fields": fields})
    return groups


_GROUP_META = [
    ("audio", "Audio & transcript", "Speech audio and the exact transcript to align."),
    ("context", "Language", "Transcript language hint."),
    ("chunking", "Chunking", "Chunking mode for long audio."),
]

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "qwen3_forced_aligner": {
        "label": "Qwen3 Forced Aligner",
        "workflows": ["offline"],
        "summary": "Map an exact transcript onto speech audio to produce word timestamps.",
        "audio_fields": ["audio", "transcript"],
        "context_fields": ["language"],
        "chunking_fields": ["audio_chunk_mode"],
        "api_hint": "Not an ASR route — the transcript is required input. For long audio timestamps, use Qwen3 ASR with words_out.",
    },
}
