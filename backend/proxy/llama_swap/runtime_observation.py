"""Last successful llama-swap running-model observation.

A failed lookup must not be reported as a verified empty process list. Callers
keep the previous snapshot and mark it stale until a new lookup succeeds.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

_lock = threading.Lock()
_snapshot: Dict[str, Any] = {"states": {}, "observed_at": None}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _view(states: Dict[str, str], observed_at: Optional[str], quality: str) -> Dict[str, Any]:
    return {
        "quality": quality,
        "observed_at": observed_at,
        "states": states,
    }


def remember_running_models(payload: Any) -> Dict[str, Any]:
    """Record a successful /running payload and return a verified observation."""
    items = payload if isinstance(payload, list) else (payload or {}).get("running") or []
    states: Dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("model")
        if not name:
            continue
        states[str(name)] = str(item.get("state") or "").lower()
    observed_at = _now()
    with _lock:
        _snapshot["states"] = states
        _snapshot["observed_at"] = observed_at
    return _view(dict(states), observed_at, "verified")


def stale_or_unreachable() -> Dict[str, Any]:
    """Return the last observation as stale, or unreachable when none exists."""
    with _lock:
        observed_at = _snapshot["observed_at"]
        if not observed_at:
            return _view({}, None, "unreachable")
        return _view(dict(_snapshot["states"]), observed_at, "stale")


def clear_runtime_observation() -> None:
    """Drop the in-process snapshot. Tests use this so lookups do not leak."""
    with _lock:
        _snapshot["states"] = {}
        _snapshot["observed_at"] = None


def runtime_fields_for(observation: Dict[str, Any], proxy_name: str) -> Dict[str, Any]:
    """Map an observation to the catalog fields for one model."""
    quality = observation.get("quality") or "unreachable"
    observed_at = observation.get("observed_at")
    if quality == "unreachable":
        return {
            "is_active": False,
            "status": None,
            "run_state": None,
            "runtime_quality": "unreachable",
            "runtime_observed_at": None,
        }
    raw_state = (observation.get("states") or {}).get(proxy_name) or None
    if raw_state == "loading":
        run_state = "loading"
    elif raw_state in ("running", "ready"):
        run_state = "running"
    else:
        run_state = None
        raw_state = None
    return {
        "is_active": run_state is not None,
        "status": raw_state,
        "run_state": run_state,
        "runtime_quality": quality,
        "runtime_observed_at": observed_at,
    }
