"""Operator-controlled experimental feature gates."""

from __future__ import annotations

import os


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def audio_cpp_enabled() -> bool:
    """Kill switch for the audio.cpp integration."""
    return _env_bool("AUDIO_CPP_ENABLED", True)


def audio_cpp_heuristic_discovery(contract_grade: str | None = None) -> bool:
    """Allow fuzzy package→family / id heuristics when upstream JSON omits fields.

    Explicit ``AUDIO_CPP_HEURISTIC_DISCOVERY`` always wins. Otherwise ``full``
    contract pins default heuristics off; thin/partial pins keep them on.
    """
    if os.getenv("AUDIO_CPP_HEURISTIC_DISCOVERY") is not None:
        return _env_bool("AUDIO_CPP_HEURISTIC_DISCOVERY", True)
    grade = str(contract_grade or "").strip().lower()
    if grade == "full":
        return False
    return True

