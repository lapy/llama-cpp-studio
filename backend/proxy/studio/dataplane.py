"""Streaming ASGI dataplane for the native Studio proxy.

Request routing, CORS, and auth behavior is adapted from llama-swap
(internal/server, MIT); see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.cookies import SimpleCookie
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote, unquote

import httpx

from backend.proxy.studio.filters import apply_request_filters
from backend.proxy.studio.registry import ProcessRegistry, RegistryError

ModelResolver = Callable[[str], Optional["ResolvedModel"]]
CatalogProvider = Callable[[], list[dict]]
Authorizer = Callable[[dict[str, str]], bool]

# Versionless and aliased route prefixes rewritten before upstream dispatch
# (llama-swap's stripVersionPrefix/stripAudioAPIPrefix, issue #728).
STRIPPED_ROUTE_PREFIXES = ("/v/", "/audioapi/")

# Cookie holding the inference key for browser navigation of engine UIs,
# which cannot attach Authorization headers to page navigation or asset GETs.
# HttpOnly (page scripts cannot read it), SameSite=Lax (no cross-site POST),
# and it only ever grants GET/HEAD passthrough — never admin or inference.
_UI_COOKIE = "studio_ui_grant"

DEFAULT_MAX_BODY_BYTES = 256 * 1024 * 1024


class BodyTooLarge(Exception):
    def __init__(self, limit: int) -> None:
        super().__init__(f"request body exceeds {limit} bytes")
        self.limit = limit


class RequestDisconnected(Exception):
    """The downstream vanished before its request body was complete."""


@dataclass
class ResolvedModel:
    model_id: str
    use_model_name: Optional[str] = None
    filters: Optional[dict] = None


class StudioDataPlane:
    """Resolve a model, acquire its generation and stream bytes unchanged."""

    def __init__(
        self,
        registry: ProcessRegistry,
        resolve_model: ModelResolver,
        authorize: Optional[Authorizer] = None,
        catalog: Optional[CatalogProvider] = None,
        max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
        authorize_admin: Optional[Authorizer] = None,
        verify_token: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self.registry = registry
        self.resolve_model = resolve_model
        self.authorize = authorize
        # Proxy administration (unload, logs, activity) needs more than an
        # inference key: the internal service credential. Without an injected
        # admin verifier, planes fall back to the plain authorizer (tests).
        self.authorize_admin = authorize_admin
        # Verifies ?key= / UI-grant-cookie credentials for browser contexts
        # that cannot attach Authorization headers.
        self.verify_token = verify_token
        self.catalog = catalog
        self.max_body_bytes = int(max_body_bytes)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None),
            limits=httpx.Limits(max_connections=200, max_keepalive_connections=40),
            trust_env=False,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await _json_response(send, 404, {"error": "not found"})
            return
        path = scope.get("path", "/")
        method = scope.get("method", "GET").upper()
        request_headers = {
            key.decode("latin-1").lower(): value.decode("latin-1")
            for key, value in scope.get("headers", [])
        }

        # CORS mirrors llama-swap's legacy permissive default: any origin, and
        # preflight is answered before auth because browsers send no key on it.
        origin = request_headers.get("origin", "")
        if method == "OPTIONS":
            await _cors_preflight(send, origin)
            return

        if path in ("/health", "/wol-health"):
            await _json_response(send, 200, {"status": "ok"}, origin=origin)
            return
        if path in ("/", "/index.html"):
            await _landing_page(send, origin=origin)
            return
        if path in ("/ui", "/ui/"):
            # llama-swap's admin UI namespace; the Studio landing page fills it.
            await send(
                {
                    "type": "http.response.start",
                    "status": 302,
                    "headers": [(b"location", b"/"), *_cors_headers(origin)],
                }
            )
            await send({"type": "http.response.body", "body": b""})
            return

        browser_key = _query_key(scope) if method in ("GET", "HEAD") else ""
        authorized = bool(self.authorize is None or self.authorize(request_headers))
        if not authorized and browser_key and self.verify_token is not None:
            authorized = bool(self.verify_token(browser_key))
        # Administration is only ever the service credential in a header:
        # never a query key, an inference key, or a UI grant cookie. Any plane
        # that enforces inference auth must inject its own admin verifier;
        # without one, administration stays closed rather than inheriting the
        # (weaker) inference authorizer.
        if self.authorize_admin is not None:
            admin = bool(self.authorize_admin(request_headers))
        else:
            admin = self.authorize is None
        # Browser entry into a model UI with ?key=: mint the namespace grant
        # cookie, then redirect so the key never persists in the URL bar.
        if (
            method == "GET"
            and browser_key
            and path.startswith("/ui/")
            and self.verify_token is not None
            and self.verify_token(browser_key)
        ):
            await _redirect_with_ui_cookie(send, scope, browser_key, origin)
            return
        ui_nav = (
            method in ("GET", "HEAD")
            and path.startswith("/ui/")
            and self.verify_token is not None
            and _ui_cookie_grants(request_headers, self.verify_token)
        )
        if not authorized and not ui_nav:
            await _json_response(
                send,
                401,
                {"error": "Invalid or missing inference key"},
                origin=origin,
                extra_headers=[(b"www-authenticate", b'Bearer realm="studio"')],
            )
            return

        if await self._admin_route(send, receive, scope, method, path, origin, admin=admin):
            return

        try:
            body = await _read_body(receive, self.max_body_bytes)
        except BodyTooLarge as exc:
            await _json_response(
                send, 413, {"error": str(exc), "limit_bytes": exc.limit}, origin=origin
            )
            return
        except RequestDisconnected:
            # There is no downstream left to answer. Most importantly, do not
            # resolve/start an engine using a truncated request body.
            return
        route_model, upstream_path = _model_route(path)
        upstream_passthrough = bool(route_model) or upstream_path != path
        requested = route_model or _requested_model(scope, body)
        resolved = self.resolve_model(requested)
        if resolved is None:
            await _json_response(
                send,
                404,
                {"error": f"unknown model: {requested or '<unspecified>'}"},
                origin=origin,
            )
            return
        if not upstream_passthrough:
            upstream_path = _strip_version_prefix(upstream_path)
            body = _prepare_body(body, resolved, requested)
        try:
            generation = await self.registry.acquire(resolved.model_id)
        except (RegistryError, ValueError, OSError) as exc:
            await _json_response(send, 502, {"error": str(exc)}, origin=origin)
            return
        try:
            await self._proxy(scope, send, generation, resolved, upstream_path, body, origin)
        finally:
            self.registry.release(generation)

    async def _admin_route(
        self,
        send: Callable,
        receive: Callable,
        scope: dict[str, Any],
        method: str,
        path: str,
        origin: str,
        *,
        admin: bool = False,
    ) -> bool:
        """Handle proxy-owned endpoints. True when the request was answered.

        ``admin`` marks the internal service credential. Inference keys may
        read the model/engines/running catalog but never unload models, tail
        raw engine logs, or watch the activity stream: those expose every
        model's traffic and lifecycle to any key holder.
        """
        sensitive = (
            path == "/api/models/unload"
            or path == "/unload"
            or path.startswith("/api/models/unload/")
            or path == "/api/events"
            or path == "/api/events/logs"
            or (path.startswith("/api/models/") and path.endswith("/logs"))
            or (path == "/api/profiles/active" and method == "PUT")
        )
        if sensitive and not admin:
            await _json_response(
                send,
                403,
                {"error": "proxy administration requires the Studio service credential"},
                origin=origin,
            )
            return True
        if path in ("/v1/models", "/models") and method == "GET":
            payload = self._catalog_payload()
            if payload is None:
                return False
            await _json_response(send, 200, payload, origin=origin)
            return True
        if path == "/api/models":
            await _json_response(send, 200, self.registry.snapshot(), origin=origin)
            return True
        if path == "/running":
            # llama-swap shape: internal callers and runtime observation parse
            # the "running" list and its per-model state ("running"/"loading").
            await _json_response(send, 200, {"running": self.registry.snapshot()}, origin=origin)
            return True
        if path == "/api/profiles":
            # Profiles are a llama-swap feature Studio does not implement. The
            # routing UI polls this passthrough; an empty set keeps it inert.
            await _json_response(send, 200, {"active": None, "profiles": []}, origin=origin)
            return True
        if path == "/api/profiles/active" and method == "PUT":
            # Clearing the active profile (name=null) is a no-op; selecting one
            # is refused because the Studio proxy has no profile feature.
            try:
                payload = _json_body(await _read_body(receive, 1_048_576))
            except BodyTooLarge as exc:
                await _json_response(send, 413, {"error": str(exc)}, origin=origin)
                return True
            if str(payload.get("name") or ""):
                await _json_response(
                    send,
                    400,
                    {"error": "profiles are not supported by the Studio proxy"},
                    origin=origin,
                )
            else:
                await _json_response(send, 200, {"active": None}, origin=origin)
            return True
        if path == "/api/engines":
            running = self.registry.snapshot()
            await _json_response(
                send,
                200,
                {
                    "engines": running,
                    "summary": {
                        "running": len(running),
                        "loading": sum(not item.get("ready", False) for item in running),
                    },
                },
                origin=origin,
            )
            return True
        if path == "/api/version":
            from backend.version import APP_VERSION

            await _json_response(
                send, 200, {"version": APP_VERSION, "proxy": "studio"}, origin=origin
            )
            return True
        if path == "/api/events":
            after = 0
            for key, value in scope.get("headers", []):
                if key.lower() == b"last-event-id":
                    try:
                        after = int(value)
                    except ValueError:
                        after = 0
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"cache-control", b"no-cache, no-store"),
                        (b"x-accel-buffering", b"no"),
                        *_cors_headers(origin),
                    ],
                }
            )
            async for item in self.registry.activities.subscribe(after):
                data = json.dumps(item.value, separators=(",", ":")).encode()
                chunk = b"id: " + str(item.sequence).encode() + b"\ndata: " + data + b"\n\n"
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            # Iterator end = disconnect or overflow close (bounded subscriber
            # queue). Close the chunked body; SSE clients resume via
            # Last-Event-ID replay from the ring.
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return True
        if path == "/api/events/logs":
            # llama-swap's per-monitor log stream. One consolidated stream per
            # running model keeps the UI's subscription model intact.
            await _json_response(
                send, 200, {"generations": self.registry.snapshot()}, origin=origin
            )
            return True
        if path.startswith("/api/models/") and path.endswith("/logs"):
            model_id = path[len("/api/models/") : -len("/logs")].strip("/")
            generation = self.registry.active(model_id)
            await _json_response(
                send,
                200,
                {"model": model_id, "lines": generation.logs.values() if generation else []},
                origin=origin,
            )
            return True
        if path == "/api/models/unload" or path == "/unload":
            stopped = 0
            for item in list(self.registry.snapshot()):
                stopped += int(await self.registry.stop(str(item.get("model") or "")))
            await _json_response(send, 200, {"stopped": stopped}, origin=origin)
            return True
        if path.startswith("/api/models/unload/"):
            model = unquote(path[len("/api/models/unload/") :])
            resolved = self.resolve_model(model)
            target = resolved.model_id if resolved else model
            if not await self.registry.stop(target):
                await _json_response(send, 404, {"error": "model is not running"}, origin=origin)
            else:
                await _json_response(send, 200, {"model": target, "stopped": True}, origin=origin)
            return True
        return False

    def _catalog_payload(self) -> Optional[dict]:
        if self.catalog is None:
            return None
        rows = self.catalog()
        created = 0
        data = []
        for row in rows:
            model_id = str(row.get("id") or "")
            if not model_id:
                continue
            entry = {
                "id": model_id,
                "object": "model",
                "created": created,
                "owned_by": str(row.get("engine") or "studio"),
            }
            data.append(entry)
            for alias in row.get("aliases") or []:
                data.append({**entry, "id": str(alias)})
        return {"object": "list", "data": data}

    async def _proxy(
        self,
        scope: dict[str, Any],
        send: Callable,
        generation: Any,
        resolved: ResolvedModel,
        upstream_path: str,
        body: bytes,
        origin: str,
    ) -> None:
        connection_tokens = {
            token.strip().lower().encode("latin-1")
            for key, value in scope.get("headers", [])
            if key.lower() == b"connection"
            for token in value.decode("latin-1").split(",")
            if token.strip()
        }
        excluded_headers = {
            b"host",
            b"content-length",
            b"connection",
            b"keep-alive",
            b"proxy-connection",
            b"te",
            b"trailer",
            b"transfer-encoding",
            b"upgrade",
            b"x-studio-service-key",
            b"authorization",
            b"x-api-key",
            b"x-api-key-id",
            *connection_tokens,
        }
        headers = {
            key.decode("latin-1"): value.decode("latin-1")
            for key, value in scope.get("headers", [])
            # Studio's credentials are consumed here; hop-by-hop fields only
            # describe the downstream connection.
            if key.lower() not in excluded_headers
        }
        # Keep the proxy's UI-grant cookie out of the engine; the page's own
        # cookies (if any) still pass through untouched.
        if _UI_COOKIE in headers.get("cookie", ""):
            pairs = [
                pair
                for pair in headers["cookie"].split(";")
                if pair.strip().split("=", 1)[0] != _UI_COOKIE
            ]
            if pairs:
                headers["cookie"] = ";".join(pairs)
            else:
                del headers["cookie"]
        headers["X-Studio-Model"] = resolved.model_id
        # Browser-supplied proxy credentials (?key=) stop at this boundary:
        # engines only ever see the business parameters of the request. The
        # raw pairs are kept verbatim so passthrough stays byte-faithful.
        query = _strip_credential_pairs(scope.get("query_string", b"").decode("latin-1"))
        url = generation.upstream + upstream_path + (("?" + query) if query else "")
        request = self.client.build_request(
            scope.get("method", "GET"), url, headers=headers, content=body
        )
        try:
            upstream = await self.client.send(request, stream=True)
        except httpx.PoolTimeout as exc:
            await _json_response(send, 429, {"error": str(exc)}, origin=origin)
            return
        except httpx.TimeoutException as exc:
            await _json_response(send, 504, {"error": str(exc)}, origin=origin)
            return
        except httpx.HTTPError as exc:
            await _json_response(send, 502, {"error": str(exc)}, origin=origin)
            return
        try:
            content_type = upstream.headers.get("content-type", "").lower()
            if "text/html" in content_type:
                html = await upstream.aread()
                html = _rewrite_html_base(html, resolved.model_id)
                response_headers = _safe_upstream_headers(
                    upstream, drop=("etag", "content-encoding")
                )
                response_headers.append(("content-length", str(len(html))))
                await _send_response_start(send, upstream.status_code, response_headers, origin)
                await send({"type": "http.response.body", "body": html})
                return
            await _send_response_start(
                send, upstream.status_code, _safe_upstream_headers(upstream), origin
            )
            async for chunk in upstream.aiter_raw():
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
        except httpx.PoolTimeout as exc:
            await _late_error(send, 429, str(exc))
        except httpx.TimeoutException as exc:
            await _late_error(send, 504, str(exc))
        except httpx.HTTPError as exc:
            await _late_error(send, 502, str(exc))
        finally:
            await upstream.aclose()


def _safe_upstream_headers(upstream: httpx.Response, *, drop: tuple[str, ...] = ()) -> list:
    # Strip hop-by-hop headers and the upstream's own CORS headers. The daemon
    # is the only CORS authority (llama-swap issue #85: duplicated
    # Access-Control-Allow-Origin values break strict clients).
    connection_tokens = {
        token.strip().lower()
        for token in upstream.headers.get("connection", "").split(",")
        if token.strip()
    }
    skip = {
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "proxy-connection",
        "te",
        "trailer",
        "upgrade",
        *connection_tokens,
    }
    skip.update(drop)
    return [
        (key, value)
        for key, value in upstream.headers.multi_items()
        if key.lower() not in skip and not key.lower().startswith("access-control-")
    ]


async def _send_response_start(send: Callable, status: int, headers: list, origin: str) -> None:
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(key.encode("latin-1"), value.encode("latin-1")) for key, value in headers]
            + _cors_headers(origin),
        }
    )


def _cors_headers(origin: str) -> list:
    if not origin:
        return []
    return [
        (b"access-control-allow-origin", b"*"),
        (b"vary", b"Origin"),
    ]


async def _cors_preflight(send: Callable, origin: str) -> None:
    headers = [
        (b"access-control-allow-methods", b"GET, POST, PUT, PATCH, DELETE, OPTIONS"),
        (b"access-control-allow-headers", b"*"),
        (b"access-control-max-age", b"600"),
        (b"content-length", b"0"),
        *_cors_headers(origin),
    ]
    await send({"type": "http.response.start", "status": 204, "headers": headers})
    await send({"type": "http.response.body", "body": b""})


def _requested_model(scope: dict[str, Any], body: bytes) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() in (b"x-studio-model", b"x-llama-swap-model"):
            return value.decode("utf-8", "replace").strip()
    for key, values in parse_qs(scope.get("query_string", b"").decode("latin-1")).items():
        if key.lower() == "model" and values:
            return values[0].strip()
    payload = _json_body(body)
    return str(payload.get("model") or "").strip()


def _prepare_body(body: bytes, resolved: ResolvedModel, requested: str) -> bytes:
    """Apply llama-swap filter semantics and useModelName to a JSON body."""
    if not body:
        return body
    payload = _json_body(body)
    if not payload:
        return body
    if resolved.filters:
        payload = apply_request_filters(payload, resolved.filters, requested)
    if resolved.use_model_name and payload.get("model") != resolved.use_model_name:
        payload["model"] = resolved.use_model_name
    return json.dumps(payload, separators=(",", ":")).encode()


def _strip_version_prefix(path: str) -> str:
    """Versionless /v/... routes reach the upstream as /... (llama-swap #728)."""
    if path.startswith("/v/"):
        return path[len("/v") :]
    if path.startswith("/audioapi/"):
        return path[len("/audioapi") :]
    return path


def _model_route(path: str) -> tuple[str, str]:
    prefix = next(
        (
            item
            for item in ("/ui/", "/upstream/", "/audiocpp/", "/comfyui/")
            if path.startswith(item)
        ),
        "",
    )
    if not prefix:
        return "", path
    if prefix in ("/audiocpp/", "/comfyui/"):
        # Fixed-model passthroughs from llama-swap. Studio has no fixed model;
        # resolve via ?model= or a single registered model in the resolver.
        return "", path[len(prefix) - 1 :]
    rest = path[len(prefix) :]
    encoded_model, separator, suffix = rest.partition("/")
    if not encoded_model:
        return "", path
    return unquote(encoded_model), "/" + suffix if separator else "/"


def _rewrite_html_base(body: bytes, model_id: str) -> bytes:
    """Pin relative UI assets and links to this model's proxy namespace."""
    base = f'<base href="/ui/{quote(model_id, safe="")}/">'.encode()
    lower = body.lower()
    start = lower.find(b"<base ")
    if start >= 0:
        end = lower.find(b">", start)
        if end >= 0:
            return body[:start] + base + body[end + 1 :]
    head = lower.find(b"<head")
    if head >= 0:
        end = lower.find(b">", head)
        if end >= 0:
            return body[: end + 1] + base + body[end + 1 :]
    return base + body


_LANDING = """<!doctype html>
<html><head><meta charset="utf-8"><title>Studio inference proxy</title>
<meta name="viewport" content="width=device-width, initial-scale=1"></head>
<body style="font-family: system-ui, sans-serif; margin: 2rem; max-width: 46rem">
<h1>Studio inference proxy</h1>
<p>This port is served by Studio's native proxy. Engine interfaces appear here
once their model is running.</p>
<p id="auth" hidden>Access keys are enabled. Paste an inference key (stored in
this browser only): <input id="key" type="password" size="40"
autocomplete="off"><button id="save">Save</button></p>
<ul id="models"></ul>
<script>
async function load() {
  const list = document.getElementById('models');
  const auth = document.getElementById('auth');
  const key = localStorage.getItem('studio-proxy-key') || '';
  const url = '/running' + (key ? '?key=' + encodeURIComponent(key) : '');
  let data;
  try {
    const r = await fetch(url);
    if (r.status === 401 || r.status === 403) { auth.hidden = false; return; }
    data = await r.json();
  } catch (e) { return; }
  auth.hidden = true;
  for (const item of data.running || []) {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = '/ui/' + encodeURIComponent(item.model) + '/' +
      (key ? '?key=' + encodeURIComponent(key) : '');
    a.textContent = item.model + ' (' + item.state + ')';
    li.appendChild(a);
    list.appendChild(li);
  }
}
document.getElementById('save').addEventListener('click', () => {
  localStorage.setItem('studio-proxy-key', document.getElementById('key').value.trim());
  document.getElementById('models').innerHTML = '';
  load();
});
load();
</script></body></html>
"""


async def _landing_page(send: Callable, *, origin: str = "") -> None:
    body = _LANDING.encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [
                (b"content-type", b"text/html; charset=utf-8"),
                (b"content-length", str(len(body)).encode()),
                *_cors_headers(origin),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _read_body(receive: Callable, limit: int = DEFAULT_MAX_BODY_BYTES) -> bytes:
    """Buffer a request body, refusing anything over ``limit`` (llama-swap 413).

    The cap is enforced while reading, so a hostile Content-Length (or none
    at all) cannot make this coroutine accumulate without bound.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        message = await receive()
        if message["type"] == "http.disconnect":
            raise RequestDisconnected()
        chunk = message.get("body", b"")
        total += len(chunk)
        if total > limit:
            raise BodyTooLarge(limit)
        chunks.append(chunk)
        if not message.get("more_body", False):
            break
    return b"".join(chunks)


def _query_key(scope: dict[str, Any]) -> str:
    for key, values in parse_qs(scope.get("query_string", b"").decode("latin-1")).items():
        if key.lower() in ("key", "api_key") and values:
            return values[0].strip()
    return ""


def _strip_credential_pairs(query: str) -> str:
    """Drop key/api_key pairs from a raw query string, keeping the rest intact."""
    return "&".join(
        pair
        for pair in query.split("&")
        if pair and pair.partition("=")[0].lower() not in ("key", "api_key")
    )


def _ui_cookie_value(request_headers: dict[str, str]) -> str:
    cookie = SimpleCookie()
    try:
        cookie.load(request_headers.get("cookie", ""))
    except ValueError:
        return ""
    morsel = cookie.get(_UI_COOKIE)
    return morsel.value if morsel else ""


def _ui_cookie_grants(request_headers: dict[str, str], verify_token: Callable[[str], bool]) -> bool:
    """True when the UI grant cookie carries a valid inference key.

    Grants browser navigation (GET/HEAD) only: callers must check the method
    before consulting this, and admin routes never see it.
    """
    value = _ui_cookie_value(request_headers)
    return bool(value and verify_token(value))


async def _redirect_with_ui_cookie(
    send: Callable, scope: dict[str, Any], key: str, origin: str
) -> None:
    """Set the UI grant cookie and re-enter the path with ?key= stripped."""
    query = _strip_credential_pairs(scope.get("query_string", b"").decode("latin-1"))
    location = scope.get("path", "/")
    if query:
        location += "?" + query
    await send(
        {
            "type": "http.response.start",
            "status": 302,
            "headers": [
                (b"location", location.encode("latin-1")),
                (
                    b"set-cookie",
                    (f"{_UI_COOKIE}={quote(key, safe='')}; Path=/; HttpOnly; SameSite=Lax").encode(
                        "latin-1"
                    ),
                ),
                (b"content-length", b"0"),
                *_cors_headers(origin),
            ],
        }
    )
    await send({"type": "http.response.body", "body": b""})


def _json_body(body: bytes) -> dict:
    if not body:
        return {}
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


async def _json_response(
    send: Callable,
    status: int,
    payload: Any,
    *,
    origin: str = "",
    extra_headers: Optional[list] = None,
) -> None:
    body = json.dumps(payload, separators=(",", ":")).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                *(extra_headers or []),
                *_cors_headers(origin),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def _late_error(send: Callable, status: int, detail: str) -> None:
    # A streaming response may already have started.  ASGI cannot replace its
    # status at that point, so closing the stream is the only valid operation.
    try:
        await _json_response(send, status, {"error": detail})
    except RuntimeError:
        await send({"type": "http.response.body", "body": b"", "more_body": False})
