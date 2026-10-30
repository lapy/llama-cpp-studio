"""Opaque inference access keys shared by Studio and the native dataplane."""

from __future__ import annotations

import hashlib
import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Mapping, Optional

from backend.data_store import get_store

_service_key = secrets.token_urlsafe(32)
_lock = threading.RLock()
_usage: dict[str, deque[float]] = defaultdict(deque)
_last_touch: dict[str, float] = {}


def internal_service_key() -> str:
    """Ephemeral credential for control-plane calls through the public socket."""
    return _service_key


def service_headers() -> dict:
    """Headers for Studio-internal HTTP calls to the inference port."""
    return {"X-Studio-Service-Key": _service_key}


def create_key(
    name: str,
    *,
    expires_at: str | None = None,
    models: list[str] | None = None,
    scopes: list[str] | None = None,
    rate_limit_per_minute: int | None = None,
    **policy,
) -> tuple[dict, str]:
    label = str(name or "").strip()
    if not label:
        raise ValueError("name is required")
    if len(label) > 120:
        raise ValueError("name is too long")
    key_id = secrets.token_hex(8)
    token = "sk-studio-" + secrets.token_urlsafe(32)
    normalized_expiry = _expiry(expires_at)
    if models is not None and (
        not isinstance(models, list) or not all(isinstance(item, str) for item in models)
    ):
        raise ValueError("models must be a list of strings")
    if scopes is not None and (
        not isinstance(scopes, list)
        or not scopes
        or not all(isinstance(item, str) for item in scopes)
    ):
        raise ValueError("scopes must be a non-empty list of strings")
    normalized_models = sorted({str(item).strip() for item in (models or []) if str(item).strip()})
    normalized_scopes = sorted(
        {str(item).strip() for item in (scopes or ["inference"]) if str(item).strip()}
    )
    if not set(normalized_scopes) <= {"inference", "models:read"}:
        raise ValueError("unsupported key scope")
    if rate_limit_per_minute is not None and not 1 <= int(rate_limit_per_minute) <= 1_000_000:
        raise ValueError("rate_limit_per_minute must be between 1 and 1000000")
    from backend.proxy.studio.accounting import validate_policy

    policy = validate_policy({**policy, "rate_limit_per_minute": rate_limit_per_minute})
    record = {
        "id": key_id,
        "name": label,
        "prefix": token[:20],
        "digest": _digest(token),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": normalized_expiry,
        "models": normalized_models,
        "scopes": normalized_scopes,
        "rate_limit_per_minute": int(rate_limit_per_minute) if rate_limit_per_minute else None,
        "last_used_at": None,
        **policy,
    }
    with _lock:
        get_store().add_inference_key(record)
    return _public(record), token


def list_keys() -> list[dict]:
    return [_public(row) for row in get_store().list_inference_keys()]


def delete_key(key_id: str) -> bool:
    with _lock:
        return get_store().delete_inference_key(str(key_id or ""))


def rotate_key(key_id: str) -> tuple[dict, str] | None:
    """Issue a fresh secret for an existing key, keeping its name and policy.

    The old token stops working immediately (single stored digest); clients
    must switch to the returned secret. The new token is shown once.
    """
    key_id = str(key_id or "")
    with _lock:
        row = next(
            (r for r in get_store().list_inference_keys() if str(r.get("id") or "") == key_id),
            None,
        )
        if row is None:
            return None
        token = "sk-studio-" + secrets.token_urlsafe(32)
        updates = {
            "digest": _digest(token),
            "prefix": token[:20],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "last_used_at": None,
        }
        if not get_store().update_inference_key(key_id, updates):
            return None
        return _public({**row, **updates}), token


def has_keys() -> bool:
    return bool(get_store().list_inference_keys())


def verify_token(token: str) -> bool:
    return matching_key(token) is not None


def matching_key(token: str) -> dict | None:
    candidate = _digest(str(token or ""))
    match = None
    # Do not return on the first match: keep comparison work independent of
    # the matching record's position.
    for row in get_store().list_inference_keys():
        digest = str(row.get("digest") or "")
        if hmac.compare_digest(candidate, digest):
            match = row
    if match is None or _expired(match):
        return None
    return match


def authorize_token(token: str, *, model: str = "", scope: str = "inference") -> bool:
    row = matching_key(token)
    if row is None:
        return False
    scopes = row.get("scopes") or ["inference"]
    if scope not in scopes:
        return False
    allowed = [str(item) for item in (row.get("models") or [])]
    if model and allowed and model not in allowed:
        return False
    limit = row.get("rate_limit_per_minute")
    if limit:
        now = time.monotonic()
        bucket = _usage[str(row.get("id") or "")]
        while bucket and bucket[0] <= now - 60:
            bucket.popleft()
        if len(bucket) >= int(limit):
            return False
        bucket.append(now)
    _touch(row)
    return True


def authorize_headers(headers: Mapping[str, str]) -> bool:
    service = str(headers.get("x-studio-service-key") or "")
    if service and hmac.compare_digest(service, _service_key):
        return True
    settings = get_store().get_settings() or {}
    proxy = settings.get("proxy") if isinstance(settings.get("proxy"), dict) else {}
    if not has_keys() and bool(proxy.get("allow_anonymous", True)):
        return True
    token = bearer_or_api_key(headers)
    token = token or basic_password(headers)
    return bool(token and verify_token(token))


def authorize_model_headers(headers: Mapping[str, str], model: str) -> bool:
    service = str(headers.get("x-studio-service-key") or "")
    if service and hmac.compare_digest(service, _service_key):
        return True
    settings = get_store().get_settings() or {}
    proxy = settings.get("proxy") if isinstance(settings.get("proxy"), dict) else {}
    if not has_keys() and bool(proxy.get("allow_anonymous", True)):
        return True
    token = bearer_or_api_key(headers) or basic_password(headers)
    return bool(token and authorize_token(token, model=model))


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
    from backend.proxy.studio.accounting import POLICY_LIMITS

    return {
        **{name: record.get(name) for name in POLICY_LIMITS},
        "token_accounting": record.get("token_accounting", "auto"),
        "id": str(record.get("id") or ""),
        "name": str(record.get("name") or ""),
        "prefix": str(record.get("prefix") or ""),
        "created_at": str(record.get("created_at") or ""),
        "expires_at": record.get("expires_at"),
        "models": list(record.get("models") or []),
        "scopes": list(record.get("scopes") or ["inference"]),
        "rate_limit_per_minute": record.get("rate_limit_per_minute"),
        "last_used_at": record.get("last_used_at"),
    }


def _expiry(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("expires_at must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    if parsed <= datetime.now(timezone.utc):
        raise ValueError("expires_at must be in the future")
    return parsed.astimezone(timezone.utc).isoformat()


def _expired(record: Mapping[str, object]) -> bool:
    value = record.get("expires_at")
    if not value:
        return False
    try:
        return datetime.fromisoformat(str(value)).astimezone(timezone.utc) <= datetime.now(
            timezone.utc
        )
    except ValueError:
        return True


def _touch(record: Mapping[str, object]) -> None:
    key_id = str(record.get("id") or "")
    now = time.monotonic()
    if not key_id or now - _last_touch.get(key_id, 0) < 60:
        return
    _last_touch[key_id] = now
    get_store().update_inference_key(
        key_id, {"last_used_at": datetime.now(timezone.utc).isoformat()}
    )


def request_identity(headers: Mapping[str, str]) -> dict | None:
    """Resolve identity once; admission and accounting reuse this policy snapshot."""
    service = str(headers.get("x-studio-service-key") or "")
    if service and hmac.compare_digest(service, _service_key):
        return {"id": "internal", "name": "Studio", "internal": True}
    rows = get_store().list_inference_keys()
    proxy = (get_store().get_settings() or {}).get("proxy") or {}
    if not rows and proxy.get("allow_anonymous", True):
        return {"id": "anonymous", "name": "Anonymous", "anonymous": True}
    token = bearer_or_api_key(headers) or basic_password(headers)
    candidate = _digest(token)
    match = None
    for row in rows:
        if hmac.compare_digest(candidate, str(row.get("digest") or "")):
            match = row
    return dict(match) if match and not _expired(match) else None


def update_key_policy(key_id: str, body: dict) -> dict | None:
    from backend.proxy.studio.accounting import validate_policy

    allowed = {
        "token_accounting",
        "name",
        "expires_at",
        "models",
        "scopes",
        "rate_limit_per_minute",
        "input_tokens_per_minute",
        "output_tokens_per_minute",
        "daily_token_budget",
        "monthly_token_budget",
        "max_output_tokens",
        "max_concurrent_requests",
    }
    if set(body) - allowed:
        raise ValueError("Unsupported key policy field")
    with _lock:
        row = next((r for r in get_store().list_inference_keys() if r["id"] == key_id), None)
        if row is None:
            return None
        merged = {**row, **body}
        name = str(merged.get("name") or "").strip()
        if not name or len(name) > 120:
            raise ValueError("name must contain 1 to 120 characters")
        models, scopes = merged.get("models") or [], merged.get("scopes", ["inference"])
        if not scopes:
            raise ValueError("Select at least one scope")
        if (
            not isinstance(models, list)
            or not isinstance(scopes, list)
            or not all(isinstance(s, str) for s in models + scopes)
        ):
            raise ValueError("models and scopes must be lists of strings")
        if not set(scopes) <= {"inference", "models:read"}:
            raise ValueError("unsupported key scope")
        updates = {
            "name": name,
            "models": sorted(set(models)),
            "scopes": sorted(set(scopes)),
            **validate_policy(merged),
        }
        if "expires_at" in body:
            updates["expires_at"] = _expiry(body["expires_at"])
        get_store().update_inference_key(key_id, updates)
        return _public({**row, **updates})
