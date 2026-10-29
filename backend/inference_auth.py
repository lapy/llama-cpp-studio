"""Opaque inference access keys shared by Studio and the native dataplane."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
from datetime import datetime, timezone
from typing import Mapping, Optional

from backend.data_store import get_store

_service_key = secrets.token_urlsafe(32)
_lock = threading.RLock()


def internal_service_key() -> str:
    """Ephemeral credential for control-plane calls through the public socket."""
    return _service_key


def service_headers() -> dict:
    """Headers for Studio-internal HTTP calls to the inference port."""
    return {"X-Studio-Service-Key": _service_key}


def create_key(name: str) -> tuple[dict, str]:
    label = str(name or "").strip()
    if not label:
        raise ValueError("name is required")
    if len(label) > 120:
        raise ValueError("name is too long")
    key_id = secrets.token_hex(8)
    token = "sk-studio-" + secrets.token_urlsafe(32)
    record = {
        "id": key_id,
        "name": label,
        "prefix": token[:20],
        "digest": _digest(token),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with _lock:
        get_store().add_inference_key(record)
    return _public(record), token


def list_keys() -> list[dict]:
    return [_public(row) for row in get_store().list_inference_keys()]


def delete_key(key_id: str) -> bool:
    with _lock:
        return get_store().delete_inference_key(str(key_id or ""))


def has_keys() -> bool:
    return bool(get_store().list_inference_keys())


def verify_token(token: str) -> bool:
    candidate = _digest(str(token or ""))
    valid = False
    # Do not return on the first match: keep comparison work independent of
    # the matching record's position.
    for row in get_store().list_inference_keys():
        digest = str(row.get("digest") or "")
        valid = hmac.compare_digest(candidate, digest) or valid
    return valid


def authorize_headers(headers: Mapping[str, str]) -> bool:
    service = str(headers.get("x-studio-service-key") or "")
    if service and hmac.compare_digest(service, _service_key):
        return True
    if not has_keys():
        return True
    token = bearer_or_api_key(headers)
    token = token or basic_password(headers)
    return bool(token and verify_token(token))


def authorize_admin_headers(headers: Mapping[str, str]) -> bool:
    """Admin tier: only the internal service credential qualifies.

    Deliberately stricter than authorize_headers: a valid inference key must
    never reach proxy administration (unload, raw logs, activity stream),
    and with no keys configured administration stays loopback-only via this
    credential rather than opening to the world.
    """
    service = str(headers.get("x-studio-service-key") or "")
    return bool(service) and hmac.compare_digest(service, _service_key)


def bearer_or_api_key(headers: Mapping[str, str]) -> str:
    authorization = str(headers.get("authorization") or "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return str(headers.get("x-api-key") or "")


def basic_password(headers: Mapping[str, str]) -> str:
    """llama-swap parity: the API key may arrive as a Basic auth password."""
    import base64

    authorization = str(headers.get("authorization") or "")
    if not authorization.lower().startswith("basic "):
        return ""
    try:
        decoded = base64.b64decode(authorization[6:].strip(), validate=True).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return ""
    _user, separator, password = decoded.partition(":")
    return password if separator else decoded


def _digest(token: str) -> str:
    # Keys contain 256 random bits, so a preimage-resistant digest is suitable
    # and avoids imposing a password-KDF cost on every inference request.
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _public(record: Mapping[str, object]) -> dict:
    return {
        "id": str(record.get("id") or ""),
        "name": str(record.get("name") or ""),
        "prefix": str(record.get("prefix") or ""),
        "created_at": str(record.get("created_at") or ""),
    }
