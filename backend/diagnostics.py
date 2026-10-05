"""Downloadable diagnostics. Settings are allowlisted, then the whole bundle is redacted."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from backend.store_io import persistence_status

SETTINGS_ALLOWLIST = frozenset(
    {
        "proxy_port",
        "public_inference_url",
        "build_type",
        "cuda",
        "flash_attention",
        "studio_options",
    }
)

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
    """Copy only known settings fields. Nested values stay for the later redaction pass."""
    if not isinstance(settings, dict):
        return {}
    return {key: settings[key] for key in SETTINGS_ALLOWLIST if key in settings}


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
    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "proxy_status": {
            "healthy": (proxy_status or {}).get("healthy"),
            "port": (proxy_status or {}).get("port"),
            "status_code": (proxy_status or {}).get("status_code"),
            "observed_at": (proxy_status or {}).get("observed_at"),
            "health_observed_at": (proxy_status or {}).get("health_observed_at")
            or (proxy_status or {}).get("observed_at"),
        },
        "runtime_observation": runtime_observation or {},
        "persistence": persistence_status(),
        "settings": allowlisted_settings(settings),
    }
    return sanitize(bundle)
