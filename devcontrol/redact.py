"""Best-effort secret redaction for logs, diagnostics, and operation records."""

from __future__ import annotations

import re
from typing import Any

REDACTED = "[redacted]"

SENSITIVE_KEY = re.compile(
    r"(?:token|password|passwd|secret|api[_-]?key|access[_-]?token|authorization|"
    r"credential|private[_-]?key|hf[_-]?token)",
    re.IGNORECASE,
)
SECRET_VALUE = re.compile(
    r"(?:hf_[A-Za-z0-9_\-]{8,}|sk-[A-Za-z0-9_\-]{8,}|ghp_[A-Za-z0-9_\-]{8,}|"
    r"Bearer\s+[A-Za-z0-9._~+/=-]{8,})",
    re.IGNORECASE,
)
AUTH_HEADER = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer|basic|token)?\s*\S+")
ASSIGNMENT = re.compile(
    r"(?i)((?:password|passwd|secret|token|api[_-]?key|access[_-]?token|"
    r"private[_-]?key|hf[_-]?token)\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|\S+)"
)
QUERY_SECRET = re.compile(
    r"(?i)([?&](?:token|password|secret|api[_-]?key|access[_-]?token|key)=)[^&\s#]+"
)
GENERIC_HEX_SECRET = re.compile(r"\b[0-9a-f]{48,}\b", re.IGNORECASE)


def redact_text(value: str) -> str:
    if not value:
        return value
    text = SECRET_VALUE.sub(REDACTED, value)
    text = AUTH_HEADER.sub(r"\1" + REDACTED, text)
    text = ASSIGNMENT.sub(r"\1" + REDACTED, text)
    text = QUERY_SECRET.sub(r"\1" + REDACTED, text)
    text = GENERIC_HEX_SECRET.sub(REDACTED, text)
    return text


def redact_obj(value: Any, *, depth: int = 0) -> Any:
    if depth > 8:
        return "[truncated]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            sensitive_key = bool(SENSITIVE_KEY.search(str(key))) and isinstance(item, str)
            if sensitive_key:
                result[key] = REDACTED
            else:
                result[key] = redact_obj(item, depth=depth + 1)
        return result
    if isinstance(value, list):
        return [redact_obj(item, depth=depth + 1) for item in value[:1000]]
    if isinstance(value, tuple):
        return [redact_obj(item, depth=depth + 1) for item in value[:1000]]
    return value


def redact_argv(argv: list[str]) -> list[str]:
    """Hide token-like values in a recorded command line."""
    result = []
    for item in argv:
        result.append(redact_text(str(item)))
    return result
