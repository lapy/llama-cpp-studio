"""Downloadable diagnostics.

The bundle copies an explicit field allowlist. Recursive redaction is only a
second pass over those permitted strings. Logs are a separate channel and are
not made safe by this export.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from backend.store_io import (
    DURABILITY_PHASES,
    PERSISTENCE_FAILED_DESCRIPTION,
    SAFE_PERSISTENCE_DESCRIPTIONS,
    persistence_status,
)

BUNDLE_SCHEMA_VERSION = 1

SETTINGS_ALLOWLIST = frozenset(
    {
        "proxy_port",
        "public_inference_url",
        "build_type",
        "cuda",
        "flash_attention",
    }
)
PROXY_FIELDS = ("healthy", "port", "status_code", "observed_at", "health_observed_at")
RUNTIME_FIELDS = ("quality", "observed_at", "age_seconds", "detail", "model_count")
EVENT_FIELDS = ("code", "category", "timestamp", "description", "phase", "committed")

_SENSITIVE_KEY = re.compile(
    r"(token|password|passwd|secret|api[_-]?key|access_token|authorization|credential|private_key)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:hf_[A-Za-z0-9_\-]{6,}|sk-[A-Za-z0-9_\-]{6,}|Bearer\s+\S+)",
    re.IGNORECASE,
)
_QUERY_IN_TEXT = re.compile(
    r"([?&](?:token|password|secret|api_key|access_token|authorization|key)=)[^&\s#]+",
    re.IGNORECASE,
)
_REDACTED = "[redacted]"


def allowlisted_settings(settings: dict | None) -> dict:
    """Copy known scalar settings. Nested objects and lists are not exported."""
    if not isinstance(settings, dict):
        return {}
    copied = {}
    for key in SETTINGS_ALLOWLIST:
        if key not in settings:
            continue
        value = settings[key]
        if isinstance(value, (dict, list)):
            continue
        copied[key] = value
    return copied


def _export_event(event: dict | None) -> dict | None:
    if not isinstance(event, dict):
        return None
    description = event.get("description")
    if description not in SAFE_PERSISTENCE_DESCRIPTIONS:
        description = PERSISTENCE_FAILED_DESCRIPTION
    code = event.get("code")
    category = event.get("category")
    timestamp = event.get("timestamp")
    exported = {
        "code": code if isinstance(code, str) else "PERSISTENCE_FAILED",
        "category": category if category in {"saturation", "durability", "unknown"} else "unknown",
        "timestamp": timestamp if isinstance(timestamp, str) else "",
        "description": description,
    }
    phase = event.get("phase")
    if phase in DURABILITY_PHASES:
        exported["phase"] = phase
    committed = event.get("committed")
    if committed is True or committed is False or committed == "unknown":
        exported["committed"] = committed
    return {key: exported[key] for key in EVENT_FIELDS if key in exported}


def _export_persistence(status: dict | None) -> dict:
    body = status if isinstance(status, dict) else {}
    latest = _export_event(body.get("latest_failure"))
    return {
        "pending_store_writes": body.get("pending_store_writes"),
        "max_pending_store_writes": body.get("max_pending_store_writes"),
        "saturated": bool(body.get("saturated")),
        "latest_failure": latest,
        "recent_failures": [
            event for event in (_export_event(item) for item in body.get("recent_failures") or []) if event
        ],
        "recent_saturation": [
            event for event in (_export_event(item) for item in body.get("recent_saturation") or []) if event
        ],
    }


_URL_IN_TEXT = re.compile(r"https?://[^\s]+", re.IGNORECASE)
_AUTH_ASSIGNMENT = re.compile(
    r"((?:authorization)\s*[:=]\s*)(?:(?:bearer|basic|digest|negotiate)\s+)?\S+",
    re.IGNORECASE,
)
_ASSIGNMENT_IN_TEXT = re.compile(
    r"((?:password|passwd|secret|token|api[_-]?key|access_token|credential|private_key)\s*[=:]\s*)\S+",
    re.IGNORECASE,
)


def _redact_url(value: str) -> str:
    parts = urlsplit(value)
    netloc = parts.netloc
    query_text = parts.query
    changed = False
    if parts.username or parts.password:
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        netloc = host
        changed = True
    if parts.query:
        query = []
        for name, item in parse_qsl(parts.query, keep_blank_values=True):
            if _SENSITIVE_KEY.search(name):
                query.append(f"{name}={_REDACTED}")
                changed = True
            else:
                query.append(f"{name}={item}")
        query_text = "&".join(query)
    if not changed:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query_text, parts.fragment))


def redact_text(value: str) -> str:
    redacted = _URL_IN_TEXT.sub(lambda match: _redact_url(match.group(0)), value)
    redacted = _AUTH_ASSIGNMENT.sub(rf"\1{_REDACTED}", redacted)
    redacted = _SECRET_VALUE.sub(_REDACTED, redacted)
    redacted = _ASSIGNMENT_IN_TEXT.sub(rf"\1{_REDACTED}", redacted)
    redacted = _QUERY_IN_TEXT.sub(rf"\1{_REDACTED}", redacted)
    return redacted


def sanitize(value: Any) -> Any:
    """Remove credentials from nested objects, query strings, and messages."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if _SENSITIVE_KEY.search(str(key)):
                cleaned[key] = _REDACTED
            else:
                cleaned[key] = sanitize(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize(item) for item in value]
    if isinstance(value, str):
        return redact_text(value)
    return value


def build_diagnostics_bundle(
    *,
    proxy_status: dict | None = None,
    runtime_observation: dict | None = None,
    settings: dict | None = None,
) -> dict:
    proxy = proxy_status or {}
    runtime = runtime_observation or {}
    health_observed_at = proxy.get("health_observed_at") or proxy.get("observed_at")
    bundle = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proxy_status": {key: proxy.get(key) for key in PROXY_FIELDS},
        "runtime_observation": {key: runtime.get(key) for key in RUNTIME_FIELDS},
        "persistence": _export_persistence(persistence_status()),
        "settings": allowlisted_settings(settings),
    }
    bundle["proxy_status"]["health_observed_at"] = health_observed_at
    return sanitize(bundle)
