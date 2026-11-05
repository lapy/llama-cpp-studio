"""Request-default vocabulary accessor (engine data, spec ``request_defaults``).

The audio model surface normalizes Studio-only request defaults for each task
family. WHICH keys are accepted and what type each carries is engine data: it is
declared on the engine spec, exported into the definition, and read here.
"""

from __future__ import annotations

from backend.engines.audio_cpp.identity import default_audio_engine_id

from typing import Any, Dict


def request_defaults_vocab(section: str) -> Dict[str, Any]:
    """Allowed keys + type lists for one request-default surface."""
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(default_audio_engine_id())
    data = dict(spec.request_defaults) if spec is not None else {}
    vocab = dict(data.get(section) or {})
    if not vocab:
        raise RuntimeError(f"audio.cpp {section} request-default vocabulary is unavailable")
    return vocab
