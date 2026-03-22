"""Studio-only model config fields (not necessarily from upstream --help)."""

from __future__ import annotations

from typing import Any, Dict, List


def studio_sections_for_engine(engine: str) -> List[Dict[str, Any]]:
    """Sections injected before CLI-derived params in the catalog API."""
    if engine in ("llama_cpp", "ik_llama"):
        return [
            {
                "id": "studio",
                "label": "Studio",
                "studio_only": True,
                "params": [
                    {
                        "key": "model_alias",
                        "label": "Model alias",
                        "type": "string",
                        "default": "",
                        "description": (
                            "Expose this model under a custom runtime ID instead of the "
                            "default Hugging Face-derived name (maps to --alias)."
                        ),
                        "flags": ["--alias"],
                    },
                ],
            }
        ]
    return []
