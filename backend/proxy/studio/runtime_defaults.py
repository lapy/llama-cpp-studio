"""Default proxy runtime settings (restart-scoped)."""

from __future__ import annotations

from typing import Optional

RUNTIME_DEFAULTS = {
    "idle_ttl_seconds": None,
    "max_body_bytes": None,
    "max_loaded_models": None,
    "global_concurrency": None,
    "per_model_concurrency": None,
    "queue_timeout_seconds": 30,
    "request_timeout_seconds": None,
    "readiness_timeout_seconds": 2700,
    "drain_timeout_seconds": 30,
    "max_queue_size": 100,
    "stream_idle_timeout_seconds": None,
    "pool_timeout_seconds": 10,
    "max_connections": 200,
    "max_keepalive_connections": 40,
    "history_retention_days": 7,
    "history_max_rows": 100000,
    "default_output_tokens": 4096,
    "capture_request_payloads": False,
}

#: Proxy settings that apply to the running proxy without a restart. Routing
#: and access changes are published live, and GPU admission is read per start.
LIVE_APPLIED_SETTINGS = frozenset(
    {
        "routing_aliases",
        "default_model",
        "allow_anonymous",
        "vram_admission",
        "vram_safety_margin_mib",
        "vram_estimates",
    }
)


def positive_int(settings: dict, key: str) -> Optional[int]:
    try:
        value = int(settings.get(key))
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def positive_float(settings: dict, key: str, default: Optional[float]) -> Optional[float]:
    try:
        value = float(settings.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default
