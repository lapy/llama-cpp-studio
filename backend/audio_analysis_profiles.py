"""Curated VAD and diarization family guidance."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from backend.audio_profile_fields import field_spec

_VAD_TASKS = frozenset({"vad"})
_DIAR_TASKS = frozenset({"diar"})


def is_vad_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _VAD_TASKS


def is_diar_task(task: Optional[str]) -> bool:
    return str(task or "").strip().lower() in _DIAR_TASKS


def is_analysis_task(task: Optional[str]) -> bool:
    return is_vad_task(task) or is_diar_task(task)


def analysis_profile_for_family(family: Optional[str]) -> Optional[Dict[str, Any]]:
    key = str(family or "").strip().lower()
    return _FAMILY_PROFILES.get(key) if key else None


def analysis_request_field_groups(family: Optional[str]) -> List[Dict[str, Any]]:
    profile = analysis_profile_for_family(family) or {}
    groups: List[Dict[str, Any]] = []
    for group_id, label, description in _GROUP_META:
        keys = profile.get(f"{group_id}_fields") or []
        fields = [field_spec(key) for key in keys]
        if fields:
            groups.append({"id": group_id, "label": label, "description": description, "fields": fields})
    return groups


_GROUP_META = [
    ("audio", "Audio input", "Input audio for analysis."),
    ("session", "Session mode", "Streaming requires server mode=streaming."),
    ("chunking", "VAD chunk planning", "Offline VAD chunk window controls."),
    ("options", "Detection options", "Threshold and segment timing controls."),
]

_FAMILY_PROFILES: Dict[str, Dict[str, Any]] = {
    "silero_vad": {
        "label": "Silero VAD",
        "workflows": ["offline", "streaming"],
        "summary": "Speech activity detection with offline and streaming modes.",
        "audio_fields": ["audio"],
        "session_fields": ["stream"],
        "chunking_fields": [
            "vad_chunk_max_seconds",
            "vad_chunk_merge_gap_seconds",
            "vad_chunk_padding_seconds",
        ],
        "options_fields": [
            "threshold",
            "neg_threshold",
            "min_speech_duration_ms",
            "min_silence_duration_ms",
            "speech_pad_ms",
        ],
        "api_hint": "Use 16 kHz WAV. Streaming mode emits partial segments.",
    },
    "marblenet_vad": {
        "label": "MarbleNet VAD",
        "workflows": ["offline"],
        "summary": "Offline speech activity detection.",
        "audio_fields": ["audio"],
        "options_fields": ["threshold"],
        "api_hint": "Offline only; writes segment JSON.",
    },
    "marblenet": {
        "label": "MarbleNet VAD",
        "workflows": ["offline"],
        "summary": "Offline speech activity detection.",
        "audio_fields": ["audio"],
        "options_fields": ["threshold"],
    },
    "sortformer_diar": {
        "label": "Sortformer Diarization",
        "workflows": ["offline"],
        "summary": "Speaker diarization for meetings and conversations.",
        "audio_fields": ["audio"],
        "api_hint": "Default package supports up to 4 speakers.",
    },
    "sortformer": {
        "label": "Sortformer Diarization",
        "workflows": ["offline"],
        "summary": "Speaker diarization for meetings and conversations.",
        "audio_fields": ["audio"],
    },
}
