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


def launch_manifests_enabled() -> bool:
    """One deployment-wide switch for versioned launch manifests.

    On unless ``LAUNCH_MANIFESTS_ENABLED`` is 0, false, no, off, or disabled.
    There is no per-engine fallback.
    """
    return _env_bool("LAUNCH_MANIFESTS_ENABLED", True)


def audio_cpp_source_option_discovery() -> bool:
    """Allow local-source fallback for options absent from the engine's help.

    Modern builds normally publish model-owned options via ``--help``. Some
    families still accept useful options that their model specs do not advertise,
    so a Studio-managed source checkout is used to fill only those missing rows.
    Operators can explicitly disable this fallback with the environment flag.
    """
    return _env_bool("AUDIO_CPP_SOURCE_OPTION_DISCOVERY", True)


def audio_cpp_heuristic_discovery(contract_grade: str | None = None) -> bool:
    """Allow fuzzy package→family / id heuristics when upstream JSON omits fields.

    Disabled by default for all builds: missing metadata stays unknown. Legacy
    installations can explicitly enable generic fuzzy matching if needed.
    """
    if os.getenv("AUDIO_CPP_HEURISTIC_DISCOVERY") is not None:
        return _env_bool("AUDIO_CPP_HEURISTIC_DISCOVERY", True)
    return False
