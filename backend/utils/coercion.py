"""Canonical coercion helpers (single source of truth for numeric parsing)."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

MAX_REASONABLE_INT = 1_000_000_000


def coerce_json_dict(value: Optional[Any], *, copy: bool = True) -> Dict[str, Any]:
    """
    Normalize optional JSON-ish config payloads to a flat dict.
    - dict: optional shallow copy
    - str: parse JSON object; invalid JSON logs a warning and returns {}
    - other / empty: {}
    """
    if not value:
        return {}
    if isinstance(value, dict):
        return dict(value) if copy else value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON dict from string")
            return {}
        return dict(parsed) if isinstance(parsed, dict) else {}
    return {}


def coerce_positive_int(value: Any) -> Optional[int]:
    """Parse a positive int from scalars or digit-containing strings; cap absurd values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        iv = int(value)
        if iv > MAX_REASONABLE_INT:
            logger.warning(
                "Unreasonably large value detected: %s, capping at %s",
                iv,
                MAX_REASONABLE_INT,
            )
            return None
        return iv if iv > 0 else None
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            candidate = int(cleaned)
            if candidate > MAX_REASONABLE_INT:
                logger.warning(
                    "Unreasonably large value detected: %s, capping at %s",
                    candidate,
                    MAX_REASONABLE_INT,
                )
                return None
            return candidate if candidate > 0 else None
        except ValueError:
            return None
    return None


def coerce_positive_int_lenient(value: Any) -> Optional[int]:
    """
    Like coerce_positive_int but extracts first digit run from strings (e.g. mixed text).
    No upper cap (used only where caller already bounds context).
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return int(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        match = re.search(r"\d+", cleaned)
        if match:
            try:
                candidate = int(match.group())
                return candidate if candidate > 0 else None
            except ValueError:
                return None
    return None


def coerce_positive_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and value > 0:
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        try:
            candidate = float(cleaned)
            return candidate if candidate > 0 else None
        except ValueError:
            return None
    return None
