"""Audio runtime capability resolution.

Audio services are domain code, but their engine identity must come from the
definitions instead of a hardcoded id. An engine participates by declaring
``runtime_kind: audio_cpp`` (the protocol/audio service capability); the helpers
below are the only place that resolves those ids.
"""

from __future__ import annotations

from typing import Tuple

AUDIO_RUNTIME_KIND = "audio_cpp"


def audio_engine_ids() -> Tuple[str, ...]:
    """Ids of every definition declaring the audio runtime capability."""
    from backend.engines.registry import ENGINE_REGISTRY

    return tuple(
        sorted(
            engine_id
            for engine_id, spec in ENGINE_REGISTRY.items()
            if str(getattr(spec, "runtime_kind", "")) == AUDIO_RUNTIME_KIND
        )
    )


def default_audio_engine_id() -> str:
    """The primary audio engine id (definitions are the source of truth)."""
    ids = audio_engine_ids()
    if not ids:
        raise RuntimeError("No engine definition declares runtime_kind 'audio_cpp'")
    return ids[0]


def is_audio_engine_id(engine_id: str) -> bool:
    return str(engine_id or "") in audio_engine_ids()
