"""Shared filesystem roots for application state."""

from __future__ import annotations

import os


def studio_data_dir() -> str:
    """Return the authoritative data root for this process."""
    override = os.getenv("STUDIO_DATA_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    if os.path.exists("/app/data"):
        return "/app/data"
    return os.path.abspath("data")
