"""Local-only and remote-access policy for the management API.

Studio authentication does not cover llama-swap's own admin routes. Those stay
reachable only when the proxy listen address is loopback, unless an operator
explicitly sets ``STUDIO_PROXY_LISTEN``.
"""

from __future__ import annotations

import ipaddress
import os
import secrets
from typing import Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

LOOPBACK_HOSTS = {
    "127.0.0.1",
    "::1",
    "localhost",
    "testclient",
    "::ffff:127.0.0.1",
}
PUBLIC_PATHS = {"/api/live", "/api/ready", "/api/access"}
SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_sessions: dict[str, str] = {}


def access_mode() -> str:
    mode = os.getenv("STUDIO_ACCESS_MODE", "local").strip().lower()
    return mode if mode in {"local", "remote"} else "local"


def bind_host() -> str:
    """Host Uvicorn should bind. Containers listen on all interfaces; Compose publishes loopback."""
    explicit = os.getenv("STUDIO_BIND_HOST", "").strip()
    if explicit:
        return explicit
    if os.path.exists("/app/data"):
        return "0.0.0.0"
    return "127.0.0.1"


def proxy_listen_host() -> str:
    """llama-swap listen host. Loopback unless running in a container or overridden."""
    explicit = os.getenv("STUDIO_PROXY_LISTEN", "").strip()
    if explicit:
        return explicit
    if os.path.exists("/app/data"):
        return "0.0.0.0"
    return "127.0.0.1"


def proxy_listen_address(port: int) -> str:
    return f"{proxy_listen_host()}:{int(port)}"


def _token_path() -> str:
    from backend.data_store import _get_config_dir

    return os.path.join(_get_config_dir(), "studio_access.token")


def management_token() -> str:
    """Token for remote mode. Environment wins; otherwise a mode-0600 file is used."""
    env_token = os.getenv("STUDIO_API_TOKEN", "").strip()
    if env_token:
        return env_token
    path = _token_path()
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as handle:
                return handle.read().strip()
    except OSError:
        return ""
    if access_mode() != "remote":
        return ""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    generated = secrets.token_urlsafe(32)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(path, flags, 0o600)
    try:
        os.write(fd, generated.encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    return generated


def is_loopback(request: Request) -> bool:
    client = request.client
    if client is None:
        return True
    host = client.host or ""
    return host in LOOPBACK_HOSTS


def in_container() -> bool:
    """True when this process is the Compose/Docker app, which owns /app/data."""
    return os.path.exists("/app/data")


def _private_compose_peer(host: str) -> bool:
    """Docker publishes the host port through the bridge, so the in-container peer is the gateway."""
    bare = (host or "").split("%", 1)[0].strip()
    if bare.startswith("::ffff:"):
        bare = bare[7:]
    try:
        address = ipaddress.ip_address(bare)
    except ValueError:
        return False
    return address.is_private or address.is_link_local


def is_trusted_local_client(request: Request) -> bool:
    """Loopback, or a Compose bridge client when Studio itself is inside the container."""
    if is_loopback(request):
        return True
    client = request.client
    if client is None or not in_container():
        return False
    return _private_compose_peer(client.host or "")


def _bearer_token(request: Request) -> str:
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""


def _session_csrf(request: Request) -> Optional[str]:
    session_id = request.cookies.get("studio_session", "")
    if not session_id:
        return None
    return _sessions.get(session_id)


def request_is_authenticated(request: Request) -> bool:
    expected = management_token()
    if not expected:
        return False
    bearer = _bearer_token(request)
    if bearer and secrets.compare_digest(bearer, expected):
        return True
    csrf = _session_csrf(request)
    if csrf is None:
        return False
    if request.method.upper() in SAFE_METHODS:
        return True
    presented = request.headers.get("x-csrf-token", "")
    cookie_csrf = request.cookies.get("studio_csrf", "")
    return bool(
        presented
        and cookie_csrf
        and secrets.compare_digest(presented, csrf)
        and secrets.compare_digest(cookie_csrf, csrf)
    )


def establish_session(token: str) -> Optional[tuple[str, str]]:
    expected = management_token()
    if not expected or not secrets.compare_digest(token, expected):
        return None
    session_id = secrets.token_urlsafe(32)
    csrf = secrets.token_urlsafe(32)
    _sessions[session_id] = csrf
    return session_id, csrf


def access_status(request: Request) -> dict:
    mode = access_mode()
    return {
        "mode": mode,
        "authenticated": request_is_authenticated(request) if mode == "remote" else True,
        "loopback": is_trusted_local_client(request),
        "proxy_listen_host": proxy_listen_host(),
        "bind_host": bind_host(),
    }


class ManagementAccessMiddleware(BaseHTTPMiddleware):
    """Reject remote management calls that are outside the configured access mode."""

    async def dispatch(self, request: Request, call_next):
        if request.method.upper() == "OPTIONS":
            return await call_next(request)
        path = request.url.path
        if path in PUBLIC_PATHS or request.method.upper() == "POST" and path == "/api/session":
            return await call_next(request)
        protected = path.startswith("/api") or path.startswith("/v1") or path in {
            "/docs",
            "/redoc",
            "/openapi.json",
        }
        if not protected:
            return await call_next(request)
        if access_mode() == "local":
            if is_trusted_local_client(request):
                return await call_next(request)
            return JSONResponse(
                {
                    "detail": "Management API is loopback-only. Set STUDIO_ACCESS_MODE=remote to expose it.",
                },
                status_code=403,
            )
        if request_is_authenticated(request):
            return await call_next(request)
        return JSONResponse(
            {"detail": "Remote management requires authentication."},
            status_code=401,
        )


def require_remote_token(token: str) -> None:
    expected = management_token()
    if not expected or not secrets.compare_digest(token or "", expected):
        raise HTTPException(status_code=401, detail="Invalid management token")
