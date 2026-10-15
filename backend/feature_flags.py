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
