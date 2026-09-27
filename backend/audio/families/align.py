"""Curated forced-alignment family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio.profiles import TaskProfileSet


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
    "mms_forced_aligner": {
        "label": "MMS Forced Aligner",
        "workflows": ["offline"],
        "summary": "Map an exact transcript onto speech audio using MMS wav2vec2 CTC.",
        "audio_fields": ["audio", "transcript"],
        "context_fields": ["language"],
        "api_hint": "Not an ASR route — the transcript is required input. Language uses MMS codes such as eng or nld.",
    },
}

ALIGN_PROFILES = TaskProfileSet(
    tasks=frozenset({"align"}),
    profiles=_FAMILY_PROFILES,
    group_meta=_GROUP_META,
)


def is_align_task(task: Optional[str]) -> bool:
    return ALIGN_PROFILES.matches_task(task)


def align_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    return ALIGN_PROFILES.profile_for_family(family)


def alignment_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    return ALIGN_PROFILES.field_groups(family)
