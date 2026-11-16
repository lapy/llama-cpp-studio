"""Build-option defaults for the engines Studio builds from source.

One place that aggregates the definition-owned build-option catalogs so a
consumer needing *every* engine at once (the frontend build-default export,
docs tooling, tests) does not carry engine-id literals of its own: engine ids
stay inside ``backend/engines/`` by design.

The catalogs remain authoritative in their engine modules; this module only
shapes them into the payload the frontend imports.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.engines.audio_cpp import build_options as audio_options
from backend.engines.llama_cpp import build_options as llama_options


def _default_first(default: str, values: List[str]) -> List[str]:
    """Catalog values with ``default`` first (the build panels' historic order)."""
    return [default] + [value for value in values if value != default]


def llama_build_types() -> List[str]:
    return _default_first(llama_options.BUILD_TYPE_DEFAULT, list(llama_options.BUILD_TYPE_VALUES))


def audio_build_types() -> List[str]:
    values = list(audio_options.build_types())
    default = audio_options.default_build_settings()["build_type"]
    return _default_first(default, values)


def export_payload() -> Dict[str, Any]:
    """Payload for ``frontend/src/engines/build-defaults.generated.json``."""
    return {
        "defaults": llama_options.default_build_settings(),
        "build_types": llama_build_types(),
        "audio": {
            "defaults": audio_options.default_build_settings(),
            "build_types": audio_build_types(),
        },
    }


__all__ = [
    "audio_build_types",
    "export_payload",
    "llama_build_types",
]
