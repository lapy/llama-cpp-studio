"""Studio-only model config fields (deprecated; engine params now come from CLI scan)."""

from __future__ import annotations

from typing import Any, Dict, List


def studio_sections_for_engine(engine: str) -> List[Dict[str, Any]]:
    """Engine params are sourced from CLI help; infra-owned fields are rendered separately in the UI."""
    del engine
    return []
