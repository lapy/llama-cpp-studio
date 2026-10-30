"""Streaming ASGI dataplane for the native Studio proxy.

Request routing, CORS, and auth behavior is adapted from llama-swap
(internal/server, MIT); see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import json
import asyncio
import time
import uuid
from dataclasses import dataclass
from http.cookies import SimpleCookie
from http.cookiejar import CookieJar, DefaultCookiePolicy
from typing import Any, Callable, Optional
from urllib.parse import parse_qs, quote, unquote

import httpx

from backend.proxy.studio.accounting import Accounting, AdmissionError
from backend.proxy.studio.usage import UsageObserver, UsageContentDecoder
from backend.proxy.studio.transport import UpstreamTransport
from backend.proxy.studio.tokenization import count_input, supported_engine
from backend.proxy.studio.filters import apply_request_filters
from backend.proxy.studio.registry import ProcessRegistry, RegistryError

ModelResolver = Callable[[str], Optional["ResolvedModel"]]
CatalogProvider = Callable[[], list[dict]]
Authorizer = Callable[[dict[str, str]], bool]
ModelAuthorizer = Callable[[dict[str, str], str], bool]

# Versionless and aliased route prefixes rewritten before upstream dispatch
# (llama-swap's stripVersionPrefix/stripAudioAPIPrefix, issue #728).
STRIPPED_ROUTE_PREFIXES = ("/v/", "/audioapi/")

# Cookie holding the inference key for browser navigation of engine UIs,
# which cannot attach Authorization headers to page navigation or asset GETs.
# HttpOnly (page scripts cannot read it), SameSite=Lax (no cross-site POST),
# and it only ever grants GET/HEAD passthrough — never admin or inference.
_UI_COOKIE = "studio_ui_grant"

DEFAULT_MAX_BODY_BYTES = 256 * 1024 * 1024


class NoUpstreamCookieStorage(DefaultCookiePolicy):
    """A reverse proxy forwards caller cookies; it is not a shared browser."""

    def set_ok(self, cookie, request):
        return False


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
    routing_revision: Optional[str] = None
    engine_id: Optional[str] = None


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
        global_concurrency: Optional[int] = None,
        queue_timeout: float = 30.0,
        request_timeout: Optional[float] = None,
        authorize_model: Optional[ModelAuthorizer] = None,
        accounting: Accounting | None = None,
        identity_resolver: Callable | None = None,
        resolve_in_executor: bool = False,
        max_queue_size: int = 100,
        stream_idle_timeout: float | None = None,
        pool_timeout: float = 10,
        max_connections: int = 200,
        max_keepalive_connections: int = 40,
        history_retention_days: int = 7,
        history_max_rows: int = 100000,
        default_output_tokens: int = 4096,
    ) -> None:
        self.accounting = accounting
        self.identity_resolver = identity_resolver
        self.resolve_in_executor = resolve_in_executor
        self.request_timeout = request_timeout
        self.max_queue_size = max_queue_size
        self.max_inflight = max_connections + max_queue_size
        self._waiting = 0
        self._observed_count = 0
        self._queued: dict[str, int] = {}
        self._ticket = 0
        self._flush_task = None
        self.history_retention_days = history_retention_days
        self.history_max_rows = history_max_rows
        self.default_output_tokens = default_output_tokens
        self.registry = registry
        self.resolve_model = resolve_model
        self.authorize = authorize
        self.authorize_model = authorize_model
        # Proxy administration (unload, logs, activity) needs more than an
        # inference key: the internal service credential. Without an injected
        # admin verifier, planes fall back to the plain authorizer (tests).
        self.authorize_admin = authorize_admin
        # Verifies ?key= / UI-grant-cookie credentials for browser contexts
        # that cannot attach Authorization headers.
        self.verify_token = verify_token
        self.catalog = catalog
        self.max_body_bytes = int(max_body_bytes)
        self.queue_timeout = max(0.1, float(queue_timeout))
        self._global_slots = asyncio.Semaphore(
            min(int(global_concurrency), max_connections) if global_concurrency else max_connections
        )
        self.client = httpx.AsyncClient(
            transport=UpstreamTransport(max_connections, max_keepalive_connections),
            cookies=CookieJar(policy=NoUpstreamCookieStorage()),
            timeout=httpx.Timeout(
                connect=10.0,
                read=stream_idle_timeout,
                write=float(request_timeout) if request_timeout else None,
                pool=pool_timeout,
            ),
            limits=httpx.Limits(
                max_connections=max_connections, max_keepalive_connections=max_keepalive_connections
            ),
            trust_env=False,
        )

    async def close(self) -> None:
        if self._flush_task:
            self._flush_task.cancel()
            await asyncio.gather(self._flush_task, return_exceptions=True)
        try:
            if self.accounting:
                await asyncio.to_thread(
                    self.accounting.flush, self.history_retention_days, self.history_max_rows
                )
        finally:
            await self.client.aclose()

    async def _flush_history(self):
        from backend.logging_config import get_logger

        while True:
            await asyncio.sleep(1)
            try:
                await asyncio.to_thread(
                    self.accounting.flush, self.history_retention_days, self.history_max_rows
                )
            except Exception:
                get_logger(__name__).exception("Proxy history flush failed")

    async def __call__(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
        path = scope.get("path", "/")
        control = path in {
            "/",
            "/index.html",
            "/health",
            "/wol-health",
            "/ui",
            "/ui/",
            "/running",
            "/unload",
            "/api/models",
            "/api/models/unload",
            "/api/profiles",
            "/api/engines",
            "/api/version",
            "/api/events",
            "/api/events/logs",
        }
        control |= path in ("/models", "/v1/models") and scope.get("method") == "GET"
        control |= path.startswith("/api/models/unload/") or (
            path.startswith("/api/models/") and path.endswith("/logs")
        )
        control |= path == "/api/profiles/active" and scope.get("method") == "PUT"
        observed = scope.get("type") == "http" and scope.get("method") != "OPTIONS" and not control
        if not observed:
            return await self._handle(scope, receive, send)
        if self.accounting and (self._flush_task is None or self._flush_task.done()):
            self._flush_task = asyncio.create_task(self._flush_history())
        started = time.monotonic()
        trace = {
            "id": uuid.uuid4().hex,
            "started_at": time.time(),
            "state": "received",
            "method": scope.get("method"),
            "path": path[:4096],
            "input_bytes": 0,
            "output_bytes": 0,
            "input_tokens": None,
            "output_tokens": None,
            "usage_source": "unknown",
            "queue_ms": 0.0,
            "loading_ms": 0.0,
            "ttft_ms": None,
            "first_byte_ms": None,
            "generation_ms": None,
            "status": 0,
        }
        scope = {**scope, "studio.trace": trace, "studio.started": started}
        self._observed_count += 1
        if self.accounting:
            self.accounting.begin(trace)
        scope["studio.overloaded"] = self._observed_count > self.max_inflight
        body_done = asyncio.Event()
        response_started = False
        response_finished = False

        async def tracked_receive():
            message = await receive()
            if message["type"] == "http.disconnect" or (
                message["type"] == "http.request" and not message.get("more_body", False)
            ):
                body_done.set()
            return message

        async def tracked_send(message):
            nonlocal response_started, response_finished
            if message["type"] == "http.response.start":
                trace["status"] = message["status"]
                trace["http_status"] = message["status"]
                response_started = True
                message = {
                    **message,
                    "headers": [
                        *message.get("headers", []),
                        (b"x-request-id", trace["id"].encode()),
                        *scope.get("studio.limit_headers", []),
                    ],
                }
            elif message["type"] == "http.response.body":
                trace["output_bytes"] += len(message.get("body", b""))
                if not message.get("more_body", False):
                    response_finished = True
            try:
                await send(message)
            except OSError as exc:
                raise RequestDisconnected() from exc

        async def watch_disconnect():
            await body_done.wait()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    if response_finished:
                        await asyncio.Future()
                    return
                await asyncio.sleep(0.01)

        work = asyncio.create_task(self._handle(scope, tracked_receive, tracked_send))
        watcher = asyncio.create_task(watch_disconnect())
        try:
            done, _ = await asyncio.wait(
                (work, watcher), timeout=self.request_timeout, return_when=asyncio.FIRST_COMPLETED
            )
            if work in done:
                await work
                if trace["state"] != "cancelled":
                    trace["state"] = (
                        "completed"
                        if 0 < trace["status"] < 400
                        else ("failed" if trace.get("dispatched") else "rejected")
                    )
            elif watcher in done:
                trace.update(state="cancelled", status=499, error="client_disconnected")
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
            else:
                trace.update(state="failed", status=504, error="request_deadline_exceeded")
                work.cancel()
                await asyncio.gather(work, return_exceptions=True)
                if not response_started:
                    await tracked_send(
                        {
                            "type": "http.response.start",
                            "status": 504,
                            "headers": [(b"content-type", b"application/json")],
                        }
                    )
                    await tracked_send(
                        {
                            "type": "http.response.body",
                            "body": b'{"error":"request deadline exceeded"}',
                        }
                    )
                else:
                    await send({"type": "http.response.body", "body": b"", "more_body": False})
        except asyncio.CancelledError:
            trace.update(state="cancelled", status=499, error="request_cancelled")
            raise
        except RequestDisconnected:
            trace.update(state="cancelled", status=499, error="client_disconnected")
        except Exception as exc:
            trace.update(state="failed", status=502, error=type(exc).__name__)
            if not response_started:
                await tracked_send(
                    {
                        "type": "http.response.start",
                        "status": 502,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await tracked_send(
                    {"type": "http.response.body", "body": b'{"error":"proxy request failed"}'}
                )
            elif not response_finished:
                await send({"type": "http.response.body", "body": b"", "more_body": False})
        finally:
            work.cancel()
            watcher.cancel()
            await asyncio.gather(work, watcher, return_exceptions=True)
            trace["completed_at"] = time.time()
            trace["latency_ms"] = (time.monotonic() - started) * 1000
            if trace.get("first_byte_ms") is not None:
                trace["response_streaming_ms"] = max(
                    0, trace["latency_ms"] - trace["first_byte_ms"]
                )
            if trace.get("ttft_ms") is not None:
                trace["generation_ms"] = max(0, trace["latency_ms"] - trace["ttft_ms"])
                trace["output_tokens_per_second"] = (
                    trace["output_tokens"] / max(0.001, trace["generation_ms"] / 1000)
                    if trace["output_tokens"] is not None
                    else None
                )
            if trace.get("dispatched") and not trace.get("usage_complete"):
                trace["observed_input_tokens"] = trace["input_tokens"]
                trace["observed_output_tokens"] = trace["output_tokens"]
                trace["output_tokens"] = None
                trace["output_tokens_per_second"] = None
                trace["usage_source"] = "unknown"
            if (
                trace.get("dispatched")
                and trace["input_tokens"] is None
                and trace.get("counted_input_tokens") is not None
            ):
                trace["input_tokens"] = trace["counted_input_tokens"]
                if trace["output_tokens"] is not None:
                    trace["usage_source"] = "engine"
            trace.pop("phase_started_at", None)
            if not trace.get("dispatched"):
                trace.update(input_tokens=0, output_tokens=0, usage_source="not_dispatched")
            if trace.get("reserved"):
                settlement = (
                    trace["id"],
                    trace["input_tokens"],
                    trace["output_tokens"],
                    bool(trace.get("dispatched")),
                )
                try:
                    # Disk work cannot be cancelled; a durable reservation must
                    # always be reconciled or queued for a conservative retry.
                    await asyncio.shield(self.accounting.settle_async(*settlement))
                except Exception:
                    self.accounting.defer_settlement(*settlement)
                    trace["accounting_note"] = "Usage reconciliation was deferred"
            if (
                trace.get("reserved")
                and trace["usage_source"] == "unknown"
                and trace.get("dispatched")
                and (trace.get("input_reserved", 0) + trace.get("output_reserved", 0))
            ):
                trace["usage_source"] = "reservation"
            trace.pop("reserved", None)
            self._observed_count -= 1
            if self.accounting:
                self.accounting.finish(trace)
            if self.accounting and len(self.accounting._pending) >= 1000:
                try:
                    await asyncio.to_thread(
                        self.accounting.flush, self.history_retention_days, self.history_max_rows
                    )
                except Exception:
                    # The periodic writer retries; the bounded buffer exposes
                    # any dropped metadata in gateway diagnostics.
                    pass

    async def _handle(self, scope: dict[str, Any], receive: Callable, send: Callable) -> None:
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
        cookie_key = (
            _ui_cookie_value(request_headers)
            if method in ("GET", "HEAD") and path.startswith("/ui/")
            else ""
        )
        # Resolve all accepted credential transports into the same principal.
        # Explicit header credentials take precedence; an invalid header must
        # not fall through to a more permissive cookie or URL credential.
        has_header = bool(
            request_headers.get("authorization")
            or request_headers.get("x-api-key")
            or request_headers.get("x-studio-service-key")
        )
        identity_headers = request_headers
        if not has_header and (browser_key or cookie_key):
            identity_headers = {
                **request_headers,
                "authorization": f"Bearer {browser_key or cookie_key}",
            }
        identity = self.identity_resolver(identity_headers) if self.identity_resolver else None
        trace = scope.get("studio.trace", {})
        if identity:
            trace.update(key_id=identity["id"], key_name=identity["name"])
        authorized = (
            identity is not None
            if self.identity_resolver
            else bool(self.authorize is None or self.authorize(identity_headers))
        )
        if (
            not self.identity_resolver
            and not authorized
            and not has_header
            and (browser_key or cookie_key)
            and self.verify_token
        ):
            authorized = bool(self.verify_token(browser_key or cookie_key))
        # Browser credentials can never be the private service credential.
        if self.identity_resolver:
            admin = bool(identity and identity.get("internal") and has_header)
        elif self.authorize_admin:
            admin = bool(self.authorize_admin(request_headers))
        else:
            admin = self.authorize is None
        if not authorized:
            trace["error"] = "invalid_or_missing_key"
            await _json_response(
                send,
                401,
                {"error": "Invalid or missing inference key"},
                origin=origin,
                extra_headers=[(b"www-authenticate", b'Bearer realm="studio"')],
            )
            return

        if (
            self.identity_resolver
            and identity
            and not identity.get("internal")
            and not identity.get("anonymous")
            and path in ("/v1/models", "/models", "/api/models", "/running", "/api/engines")
            and "models:read" not in identity.get("scopes", [])
        ):
            await _json_response(
                send, 403, {"error": "Key lacks models:read permission"}, origin=origin
            )
            return
        scope["studio.identity"] = identity
        if await self._admin_route(send, receive, scope, method, path, origin, admin=admin):
            return

        if scope.get("studio.overloaded"):
            trace["error"] = "queue_full"
            await _json_response(
                send,
                429,
                {"error": "Proxy admission capacity is full", "code": "queue_full"},
                origin=origin,
                extra_headers=[(b"retry-after", b"1")],
            )
            return
        try:
            body = await _read_body(receive, self.max_body_bytes)
        except BodyTooLarge as exc:
            trace["error"] = "body_too_large"
            await _json_response(
                send, 413, {"error": str(exc), "limit_bytes": exc.limit}, origin=origin
            )
            return
        except RequestDisconnected:
            # There is no downstream left to answer. Most importantly, do not
            # resolve/start an engine using a truncated request body.
            trace.update(state="cancelled", status=499, error="client_disconnected")
            return
        trace["input_bytes"] = len(body)
        route_model, upstream_path = _model_route(path)
        upstream_passthrough = bool(route_model) or upstream_path != path
        requested = route_model or _requested_model(scope, body)
        if len(requested) > 4096:
            trace["error"] = "invalid_model_id"
            await _json_response(send, 400, {"error": "Model ID is too long"}, origin=origin)
            return
        trace["requested_model"] = requested
        resolved = (
            await asyncio.to_thread(self.resolve_model, requested)
            if self.resolve_in_executor
            else self.resolve_model(requested)
        )
        if resolved is None:
            trace["error"] = "unknown_model"
            await _json_response(
                send,
                404,
                {"error": f"unknown model: {requested or '<unspecified>'}"},
                origin=origin,
            )
            return
        trace["model"] = resolved.model_id
        trace["routing_revision"] = resolved.routing_revision
        if (
            self.identity_resolver
            and identity
            and not identity.get("internal")
            and not identity.get("anonymous")
        ):
            if "inference" not in (identity.get("scopes") or ["inference"]) or (
                identity.get("models") and resolved.model_id not in identity["models"]
            ):
                trace["error"] = "model_or_scope_forbidden"
                await _json_response(
                    send,
                    403,
                    {"error": "Key is not permitted for this model or scope"},
                    origin=origin,
                )
                return
        if (
            not self.identity_resolver
            and self.authorize_model is not None
            and not self.authorize_model(identity_headers, resolved.model_id)
        ):
            await _json_response(
                send,
                403,
                {"error": "inference key is not permitted for this model or is rate limited"},
                origin=origin,
            )
            return
        if method == "GET" and browser_key and path.startswith("/ui/"):
            # Only mint the cookie after canonical model/scope authorization.
            await _redirect_with_ui_cookie(send, scope, browser_key, origin)
            return
        if not upstream_passthrough:
            upstream_path = _strip_version_prefix(upstream_path)
            body = _prepare_body(body, resolved, requested)
        needs_count = False
        if (
            self.accounting
            and identity
            and not identity.get("internal")
            and not identity.get("anonymous")
        ):
            admission_started = time.monotonic()
            try:
                ui_asset = method in ("GET", "HEAD") and path.startswith("/ui/")
                if ui_asset:
                    safe_ui = upstream_path in (
                        "/",
                        "/index.html",
                        "/props",
                        "/health",
                        "/v1/models",
                        "/models",
                    ) or upstream_path.endswith(
                        (
                            ".js",
                            ".css",
                            ".ico",
                            ".svg",
                            ".png",
                            ".jpg",
                            ".woff",
                            ".woff2",
                            ".map",
                            ".html",
                        )
                    )
                    limited = any(
                        identity.get(name)
                        for name in (
                            "input_tokens_per_minute",
                            "output_tokens_per_minute",
                            "daily_token_budget",
                            "monthly_token_budget",
                            "max_output_tokens",
                        )
                    )
                    if limited and not safe_ui:
                        raise AdmissionError(
                            "This UI GET endpoint cannot be token-accounted",
                            "unsupported_accounting",
                            400,
                        )
                    estimate = output_reservation = 0
                else:
                    token_limited = any(
                        identity.get(name)
                        for name in (
                            "input_tokens_per_minute",
                            "output_tokens_per_minute",
                            "daily_token_budget",
                            "monthly_token_budget",
                        )
                    )
                    needs_count = token_limited and supported_engine(resolved.engine_id)
                    if (
                        token_limited
                        and identity.get("token_accounting") == "exact"
                        and not needs_count
                    ):
                        raise AdmissionError(
                            "This engine has no exact token-count adapter",
                            "exact_token_count_unavailable",
                            503,
                        )
                    trace["input_reserved_source"] = (
                        "pending_engine_count" if needs_count else "byte_estimate"
                    )
                    body, estimate, output_reservation = self._budget_body(
                        body, identity, upstream_path, engine_count=needs_count
                    )
                bounded_admission = any(
                    identity.get(name)
                    for name in (
                        "rate_limit_per_minute",
                        "input_tokens_per_minute",
                        "output_tokens_per_minute",
                        "daily_token_budget",
                        "monthly_token_budget",
                        "max_concurrent_requests",
                    )
                )
                if not bounded_admission:
                    # No allowance can be overspent. Record usage with the
                    # batched history transaction instead of syncing a zero
                    # reservation before every unlimited request.
                    reservation = {"input_reserved": 0, "output_reserved": 0}
                    trace.update(
                        reservation,
                        accounting_mode="batched_usage",
                        admission_ms=(time.monotonic() - admission_started) * 1000,
                    )
                else:
                    admission = asyncio.create_task(
                        self.accounting.reserve_async(
                            trace["id"], identity, estimate, output_reservation
                        )
                    )
                    try:
                        reservation = await asyncio.shield(admission)
                    except asyncio.CancelledError:
                        # A worker-thread transaction cannot be cancelled. Wait for
                        # its result so the outer finally can release its charge.
                        try:
                            reservation = await admission
                        except Exception:
                            raise asyncio.CancelledError() from None
                        trace.update(reservation, reserved=True)
                        raise
                    trace.update(
                        reservation,
                        reserved=True,
                        admission_ms=(time.monotonic() - admission_started) * 1000,
                    )
                scope["studio.limit_headers"] = _quota_headers(trace)
            except AdmissionError as exc:
                trace["error"] = exc.code
                await _json_response(
                    send,
                    exc.status,
                    {"error": str(exc), "code": exc.code},
                    origin=origin,
                    extra_headers=[(b"retry-after", str(exc.retry_after).encode())]
                    if exc.status == 429
                    else [],
                )
                return
        global_slot = model_slot = False
        generation = None
        queued = False
        queue_started = time.monotonic()
        loading_started = None
        try:
            # A bounded admission queue includes waiters for model slots even
            # when the global slot is available. No second hidden pool queue.
            if self._waiting >= self.max_queue_size:
                trace["error"] = "queue_full"
                await _json_response(
                    send,
                    429,
                    {"error": "Proxy queue is full", "code": "queue_full"},
                    origin=origin,
                    extra_headers=[(b"retry-after", b"1")],
                )
                return
            self._waiting += 1
            self._ticket += 1
            self._queued[trace.get("id", str(id(scope)))] = self._ticket
            queued = True
            trace.update(state="queued", queue_position=self._waiting, phase_started_at=time.time())
            async with asyncio.timeout(self.queue_timeout):
                if self._global_slots is not None:
                    await self._global_slots.acquire()
                    global_slot = True
                enter_request = getattr(self.registry, "enter_request", None)
                if enter_request is not None:
                    await enter_request(resolved.model_id)
                    model_slot = True
            self._waiting -= 1
            self._queued.pop(trace.get("id", str(id(scope))), None)
            queued = False
            trace.update(
                queue_ms=(time.monotonic() - queue_started) * 1000,
                queue_position=None,
                state="loading",
            )
            loading_started = time.monotonic()
            trace["phase_started_at"] = time.time()
            generation = await self.registry.acquire(resolved.model_id)
            trace.update(
                loading_ms=(time.monotonic() - loading_started) * 1000,
                engine=getattr(generation, "engine_id", ""),
                pid=getattr(getattr(generation, "process", None), "pid", None),
                state="forwarding",
            )
            if needs_count:
                trace["state"] = "counting"
                count_started = time.monotonic()
                counted = await count_input(
                    self.client,
                    generation.upstream,
                    resolved.engine_id,
                    upstream_path,
                    json.loads(body),
                )
                if counted is None:
                    if identity.get("token_accounting") == "exact":
                        raise AdmissionError(
                            "Installed engine does not support exact counts for this request",
                            "exact_token_count_unavailable",
                            503,
                        )
                    # Text fallback is visible. Complex/stateful requests still
                    # fail closed unless the engine counted the actual context.
                    _, exact_input, _ = self._budget_body(body, identity, upstream_path)
                    trace["input_reserved_source"] = "byte_estimate"
                else:
                    exact_input = counted.tokens
                    trace["input_reserved_source"] = "engine"
                    trace["counted_input_tokens"] = exact_input
                    trace["token_count_endpoint"] = counted.source
                adjustment = asyncio.create_task(
                    self.accounting.adjust_async(trace["id"], identity, exact_input)
                )
                try:
                    updated = await asyncio.shield(adjustment)
                except asyncio.CancelledError:
                    try:
                        trace.update(await adjustment)
                    except Exception:
                        pass
                    raise
                trace.update(
                    updated,
                    token_count_ms=(time.monotonic() - count_started) * 1000,
                    state="forwarding",
                )
            scope["studio.limit_headers"] = _quota_headers(trace)
            trace["dispatched"] = True
            status, output_bytes, input_tokens, output_tokens = await self._proxy(
                scope, send, generation, resolved, upstream_path, body, origin
            )
            record_request = getattr(self.registry, "record_request", None)
            trace["status"] = status
            if record_request is not None:
                record_request(
                    generation,
                    method=scope.get("method", ""),
                    path=path,
                    status=status,
                    latency_ms=(time.monotonic() - scope.get("studio.started", queue_started))
                    * 1000,
                    input_bytes=len(body),
                    output_bytes=output_bytes,
                    input_tokens=input_tokens or 0,
                    output_tokens=output_tokens or 0,
                )
        except AdmissionError as exc:
            trace["error"] = exc.code
            await _json_response(
                send,
                exc.status,
                {"error": str(exc), "code": exc.code},
                origin=origin,
                extra_headers=[(b"retry-after", str(exc.retry_after).encode())]
                if exc.status == 429
                else [],
            )
        except asyncio.TimeoutError:
            trace["error"] = "queue_timeout"
            await _json_response(
                send,
                429,
                {"error": "proxy request queue timed out", "code": "queue_timeout"},
                origin=origin,
                extra_headers=[(b"retry-after", b"1")],
            )
        except (RegistryError, ValueError, OSError) as exc:
            status = 429 if "queue" in str(exc) or "maximum loaded" in str(exc) else 502
            trace["error"] = "queue_timeout" if status == 429 else "engine_unavailable"
            await _json_response(send, status, {"error": str(exc)}, origin=origin)
        finally:
            if generation is None and loading_started is not None:
                trace["loading_ms"] = (time.monotonic() - loading_started) * 1000
            if queued:
                self._waiting -= 1
                self._queued.pop(trace.get("id", str(id(scope))), None)
                trace["queue_ms"] = (time.monotonic() - queue_started) * 1000
            for position, request_id in enumerate(self._queued, 1):
                if self.accounting and request_id in self.accounting.live:
                    self.accounting.live[request_id]["queue_position"] = position
            if generation is not None:
                self.registry.release(generation)
            if model_slot:
                self.registry.leave_request(resolved.model_id)
            if global_slot and self._global_slots is not None:
                self._global_slots.release()

    def _budget_body(
        self, body: bytes, key: dict, path: str, *, engine_count: bool = False
    ) -> tuple[bytes, int, int]:
        limited_tokens = any(
            key.get(name)
            for name in (
                "input_tokens_per_minute",
                "output_tokens_per_minute",
                "daily_token_budget",
                "monthly_token_budget",
            )
        )
        cap = key.get("max_output_tokens")
        if not limited_tokens and not cap:
            return body, 0, 0
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            raise AdmissionError(
                "Token-limited keys require a JSON inference request", "unsupported_accounting", 400
            ) from None
        if not isinstance(payload, dict) or not any(
            path.endswith(suffix)
            for suffix in ("/chat/completions", "/completions", "/embeddings", "/responses")
        ):
            raise AdmissionError(
                "Token budgets support text completions, responses and embeddings",
                "unsupported_accounting",
                400,
            )
        # A byte reservation deliberately overestimates ordinary text tokenizers.
        # Binary/multimodal tokenization is engine-specific and cannot be safely
        # admitted under text-token budgets without an engine tokenizer adapter.
        raw_input = json.dumps(
            {
                k: v
                for k, v in payload.items()
                if k in ("messages", "prompt", "input", "instructions", "tools")
            },
            ensure_ascii=False,
        ).encode()
        if not engine_count and any(
            marker in raw_input
            for marker in (
                b'"image_url"',
                b'"input_audio"',
                b'"input_image"',
                b'"video_url"',
                b'"file_id"',
            )
        ):
            raise AdmissionError(
                "Multimodal token budgets require engine tokenization support",
                "unsupported_accounting",
                400,
            )
        if (
            limited_tokens
            and not engine_count
            and any(payload.get(name) for name in ("previous_response_id", "conversation"))
        ):
            raise AdmissionError(
                "Stateful conversation context cannot be reserved without engine tokenization",
                "unsupported_accounting",
                400,
            )
        output = 0
        if not path.endswith("/embeddings"):
            fields = [
                name
                for name in ("max_tokens", "max_completion_tokens", "max_output_tokens")
                if name in payload
            ]
            for name in fields:
                if (
                    isinstance(payload[name], bool)
                    or not isinstance(payload[name], int)
                    or payload[name] < 1
                ):
                    raise AdmissionError(
                        "Output token limit must be a positive integer", "invalid_output_limit", 400
                    )
            requested = max(
                (payload[name] for name in fields), default=cap or self.default_output_tokens
            )
            if cap and requested > cap:
                raise AdmissionError(
                    "Requested output exceeds this key's output limit", "max_output_tokens", 400
                )
            output = requested
            if not fields:
                payload["max_output_tokens" if path.endswith("/responses") else "max_tokens"] = (
                    requested
                )
            # Usage is optional in OpenAI chat streams; request it for budgeted
            # traffic. Engines without support fail clearly, never free-charge.
            if payload.get("stream") and path.endswith("/chat/completions"):
                if payload.get("stream_options") is not None and not isinstance(
                    payload["stream_options"], dict
                ):
                    raise AdmissionError(
                        "stream_options must be an object", "invalid_stream_options", 400
                    )
                payload["stream_options"] = {
                    **(payload.get("stream_options") or {}),
                    "include_usage": True,
                }
            n = payload.get("n", 1)
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise AdmissionError("n must be a positive integer", "invalid_n", 400)
            prompts = payload.get("prompt")
            batch_size = (
                len(prompts)
                if path.endswith("/completions")
                and not path.endswith("/chat/completions")
                and isinstance(prompts, list)
                and prompts
                and (
                    all(isinstance(item, str) for item in prompts)
                    or all(isinstance(item, list) for item in prompts)
                )
                else 1
            )
            output *= n * batch_size
            if output > 10**12:
                raise AdmissionError("Output allowance is too large", "invalid_output_limit", 400)
            if cap and output > cap:
                raise AdmissionError(
                    "Combined output across choices exceeds the key output limit",
                    "max_output_tokens",
                    400,
                )
        return (
            json.dumps(payload).encode(),
            len(raw_input) if limited_tokens and not engine_count else 0,
            output if limited_tokens else 0,
        )

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
            identity = scope.get("studio.identity") or {}
            if identity.get("models"):
                allowed = set(identity["models"])
                permitted = {
                    row["id"]
                    for row in (self.catalog() if self.catalog else [])
                    if row["id"] in allowed
                }
                for row in self.catalog() if self.catalog else []:
                    if row["id"] in allowed:
                        permitted.update(row.get("aliases") or [])
                payload["data"] = [row for row in payload["data"] if row["id"] in permitted]
            await _json_response(send, 200, payload, origin=origin)
            return True

        def visible_running():
            identity = scope.get("studio.identity") or {}
            allowed = identity.get("models") or []
            return [
                row
                for row in self.registry.snapshot()
                if row.get("state") not in ("failed", "stopped")
                and (not allowed or row.get("model") in allowed)
            ]

        if path == "/api/models":
            await _json_response(send, 200, visible_running(), origin=origin)
            return True
        if path == "/running":
            # llama-swap shape: internal callers and runtime observation parse
            # the "running" list and its per-model state ("running"/"loading").
            await _json_response(send, 200, {"running": visible_running()}, origin=origin)
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
            running = visible_running()
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
    ) -> tuple[int, int, int, int]:
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
        trace = scope.get("studio.trace", {})
        try:
            upstream = await self.client.send(request, stream=True)
        except httpx.PoolTimeout as exc:
            trace["error"] = "upstream_pool_timeout"
            await _json_response(send, 429, {"error": str(exc)}, origin=origin)
            return 429, 0, 0, 0
        except httpx.TimeoutException as exc:
            trace["error"] = "upstream_timeout"
            await _json_response(send, 504, {"error": str(exc)}, origin=origin)
            return 504, 0, 0, 0
        except httpx.HTTPError as exc:
            trace["error"] = "upstream_connection_error"
            await _json_response(send, 502, {"error": str(exc)}, origin=origin)
            return 502, 0, 0, 0
        response_started = False
        try:
            output_bytes = 0
            observer = UsageObserver(
                "text/event-stream" in upstream.headers.get("content-type", "").lower()
            )
            trace = scope.get("studio.trace", {})
            trace["state"] = "streaming"
            content_type = upstream.headers.get("content-type", "").lower()
            observe_usage = "json" in content_type or "text/event-stream" in content_type
            usage_decoder = UsageContentDecoder(
                upstream.headers.get("content-encoding", "").lower()
            )
            if "text/html" in content_type:
                html = await upstream.aread()
                html = _rewrite_html_base(html, resolved.model_id)
                response_headers = _safe_upstream_headers(
                    upstream, drop=("etag", "content-encoding")
                )
                response_headers.append(("content-length", str(len(html))))
                await _send_response_start(send, upstream.status_code, response_headers, origin)
                response_started = True
                await send({"type": "http.response.body", "body": html})
                prompt_tokens, completion_tokens = _usage_tokens(html)
                return upstream.status_code, len(html), prompt_tokens, completion_tokens
            await _send_response_start(
                send, upstream.status_code, _safe_upstream_headers(upstream), origin
            )
            response_started = True
            async for chunk in upstream.aiter_raw():
                output_bytes += len(chunk)
                if trace.get("first_byte_ms") is None:
                    trace["first_byte_ms"] = (
                        time.monotonic() - scope.get("studio.started", time.monotonic())
                    ) * 1000
                if observe_usage:
                    for decoded in usage_decoder.chunks(chunk):
                        observer.feed(decoded)
                if observer.saw_token and trace.get("ttft_ms") is None:
                    trace["ttft_ms"] = (
                        time.monotonic() - scope.get("studio.started", time.monotonic())
                    ) * 1000
                trace.update(
                    input_tokens=observer.input_tokens,
                    output_tokens=observer.output_tokens,
                    usage_source="engine"
                    if observer.input_tokens is not None and observer.output_tokens is not None
                    else "unknown",
                )
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            observer.finish()
            trace["usage_complete"] = (
                observe_usage and usage_decoder.complete and (not observer.sse or observer.finished)
            )
            if upstream_path.endswith("/embeddings") and observer.input_tokens is not None:
                observer.output_tokens = 0
            trace.update(
                input_tokens=observer.input_tokens,
                output_tokens=observer.output_tokens,
                usage_source="engine"
                if observer.input_tokens is not None and observer.output_tokens is not None
                else "unknown",
            )
            return upstream.status_code, output_bytes, observer.input_tokens, observer.output_tokens
        except httpx.PoolTimeout as exc:
            trace["error"] = "upstream_pool_timeout"
            await _late_error(send, 429, str(exc), response_started)
            return 429, output_bytes, observer.input_tokens, observer.output_tokens
        except httpx.TimeoutException as exc:
            trace["error"] = "upstream_stream_timeout"
            await _late_error(send, 504, str(exc), response_started)
            return 504, output_bytes, observer.input_tokens, observer.output_tokens
        except httpx.HTTPError as exc:
            trace["error"] = "upstream_stream_error"
            await _late_error(send, 502, str(exc), response_started)
            return 502, output_bytes, observer.input_tokens, observer.output_tokens
        finally:
            await upstream.aclose()


def _quota_headers(record):
    return [
        (name.encode(), str(value).encode())
        for name, value in (
            ("x-ratelimit-remaining-requests", record.get("remaining_requests")),
            ("x-ratelimit-remaining-tokens-day", record.get("remaining_daily_tokens")),
            ("x-ratelimit-remaining-tokens-month", record.get("remaining_monthly_tokens")),
        )
        if value is not None
    ]


def _usage_tokens(body: bytes) -> tuple[int, int]:
    """Extract OpenAI usage from JSON or SSE without buffering unbounded output."""
    candidates = [body]
    if body.startswith(b"data:") or b"\ndata:" in body:
        candidates = [
            line[5:].strip()
            for line in body.splitlines()
            if line.startswith(b"data:") and line[5:].strip() not in (b"", b"[DONE]")
        ]
    prompt = completion = 0
    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except (ValueError, UnicodeDecodeError):
            continue
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if not isinstance(usage, dict):
            continue
        prompt = max(prompt, int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0))
        completion = max(
            completion,
            int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
        )
    return prompt, completion


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
        (
            b"access-control-expose-headers",
            b"X-Request-ID, Retry-After, X-RateLimit-Remaining-Requests, X-RateLimit-Remaining-Tokens-Day, X-RateLimit-Remaining-Tokens-Month",
        ),
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
                    (
                        f"{_UI_COOKIE}={quote(key, safe='')}; Path=/ui/{quote(_model_route(scope.get('path', '/'))[0], safe='')}/; HttpOnly; SameSite=Lax"
                    ).encode("latin-1"),
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


async def _late_error(send: Callable, status: int, detail: str, started: bool = True) -> None:
    # Never attempt a second response-start after streaming headers were sent.
    if started:
        await send({"type": "http.response.body", "body": b"", "more_body": False})
    else:
        await _json_response(send, status, {"error": detail})
