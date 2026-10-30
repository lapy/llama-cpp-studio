import asyncio
import json
import os
import sys

import httpx
import pytest

from backend.proxy.launch_spec import EnvSpec, LaunchSpec
from backend.proxy.studio.dataplane import (
    ResolvedModel,
    StudioDataPlane,
    _model_route,
    _prepare_body,
    _rewrite_html_base,
    _strip_version_prefix,
)
from backend.proxy.studio.filters import apply_request_filters
from backend.proxy.studio.registry import Generation, ProcessRegistry, _loading_progress
from backend.proxy.studio.ring import EventRing
from backend.inference_auth import authorize_headers, create_key, delete_key, verify_token


def _spec(model="demo", revision="a"):
    return (
        LaunchSpec(
            model_id=model,
            engine_id="fake",
            engine_install_id="fake",
            executable=sys.executable,
            argv=["-c", "import time; print('ready', flush=True); time.sleep(60)"],
            cwd=None,
            env=EnvSpec(),
        ),
        revision * 64,
        None,
    )


@pytest.mark.asyncio
async def test_ring_replays_then_fans_out():
    ring = EventRing[str](2)
    ring.append("one")
    second = ring.append("two")
    stream = ring.subscribe(second.sequence - 1)
    assert (await anext(stream)).value == "two"
    ring.append("three")
    assert (await anext(stream)).value == "three"
    await stream.aclose()
    assert ring.values() == ["two", "three"]


@pytest.mark.asyncio
async def test_registry_reuses_generation_and_kills_process_group(monkeypatch):
    async def factory(model):
        return _spec(model)

    registry = ProcessRegistry(factory, readiness_timeout=1)

    async def ready(_generation):
        return None

    monkeypatch.setattr(registry, "_wait_ready", ready)
    first = await registry.acquire("demo")
    second = await registry.acquire("demo")
    assert first is second
    assert first.inflight == 2
    registry.release(first)
    registry.release(second)
    pid = first.process.pid
    await registry.close()
    assert first.process.poll() is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_registry_swaps_revision_then_drains_old_generation(monkeypatch):
    revision = {"value": "a"}

    async def factory(model):
        return _spec(model, revision["value"])

    registry = ProcessRegistry(factory, readiness_timeout=1, drain_timeout=1)

    async def ready(_generation):
        return None

    monkeypatch.setattr(registry, "_wait_ready", ready)
    old = await registry.acquire("demo")
    revision["value"] = "b"
    new = await registry.acquire("demo")
    assert new is not old
    assert old.draining is True
    assert old.process.poll() is None
    registry.release(old)
    for _ in range(50):
        if old.process.poll() is not None:
            break
        await asyncio.sleep(0.02)
    assert old.process.poll() is not None
    registry.release(new)
    await registry.close()


@pytest.mark.asyncio
async def test_dataplane_streams_raw_sse_and_injects_model_header(monkeypatch):
    class Generation:
        upstream = "http://engine.test"

    class Registry:
        def __init__(self):
            self.released = False

        async def acquire(self, model):
            assert model == "demo"
            return Generation()

        def release(self, generation):
            self.released = True

        def snapshot(self):
            return []

    seen = {}

    def upstream(request):
        seen["header"] = request.headers["x-studio-model"]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=httpx.ByteStream(b"data: one\n\ndata: two\n\n"),
        )

    registry = Registry()
    plane = StudioDataPlane(registry, lambda name: ResolvedModel(name) if name == "demo" else None)
    await plane.client.aclose()
    plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post("/v1/chat/completions", json={"model": "demo"})
    await plane.close()
    assert response.content == b"data: one\n\ndata: two\n\n"
    assert response.headers["content-type"] == "text/event-stream"
    assert seen["header"] == "demo"
    assert registry.released is True


@pytest.mark.asyncio
async def test_dataplane_requires_configured_inference_key():
    class Registry:
        def snapshot(self):
            return []

    record, token = create_key("test client")
    try:
        plane = StudioDataPlane(
            Registry(), lambda _name: None, authorize=authorize_headers, catalog=lambda: []
        )
        transport = httpx.ASGITransport(app=plane)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            missing = await client.post("/v1/chat/completions", json={"model": "demo"})
            unauthenticated_catalog = await client.get("/v1/models")
            health = await client.get("/health")
            valid = await client.get("/v1/models", headers={"Authorization": f"Bearer {token}"})
            api_key = await client.get("/v1/models", headers={"x-api-key": token})
            basic = await client.get(
                "/v1/models",
                headers={"Authorization": f"Basic {_b64('user:' + token)}"},
            )
        await plane.close()
        assert missing.status_code == 401
        assert unauthenticated_catalog.status_code == 401
        # /health stays open for liveness probes, matching llama-swap.
        assert health.status_code == 200
        assert valid.status_code == 200
        assert api_key.status_code == 200
        assert basic.status_code == 200
    finally:
        delete_key(record["id"])


def _b64(text: str) -> str:
    import base64

    return base64.b64encode(text.encode()).decode()


@pytest.mark.asyncio
async def test_dataplane_answers_cors_preflight_before_auth():
    class Registry:
        def snapshot(self):
            return []

    record, token = create_key("browser")
    try:
        plane = StudioDataPlane(
            Registry(), lambda _name: None, authorize=authorize_headers, catalog=lambda: []
        )
        transport = httpx.ASGITransport(app=plane)
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            # A preflight carries no key: it must still succeed with CORS headers.
            preflight = await client.options(
                "/v1/chat/completions",
                headers={
                    "Origin": "http://dash.example",
                    "Access-Control-Request-Method": "POST",
                },
            )
            actual = await client.post(
                "/v1/chat/completions",
                json={"model": "demo"},
                headers={"Origin": "http://dash.example"},
            )
            no_origin = await client.post("/v1/chat/completions", json={"model": "demo"})
        await plane.close()
        assert preflight.status_code == 204
        assert preflight.headers["access-control-allow-origin"] == "*"
        assert preflight.headers["access-control-allow-headers"] == "*"
        assert actual.status_code == 401
        assert actual.headers["access-control-allow-origin"] == "*"
        assert actual.headers["www-authenticate"].startswith("Bearer")
        # Non-browser clients get no CORS headers at all (llama-swap #85).
        assert "access-control-allow-origin" not in no_origin.headers
    finally:
        delete_key(record["id"])


@pytest.mark.asyncio
async def test_dataplane_running_shape_and_catalog_lists_aliases():
    class Registry:
        def snapshot(self):
            return [{"model": "demo", "state": "ready", "ready": True}]

    plane = StudioDataPlane(
        Registry(),
        lambda _name: None,
        catalog=lambda: [{"id": "demo", "engine": "llama_cpp", "aliases": ["demo:fast"]}],
    )
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        running = await client.get("/running")
        catalog = await client.get("/v1/models")
        empty_profiles = await client.get("/api/profiles")
        deactivate = await client.put("/api/profiles/active", json={"name": None})
        select = await client.put("/api/profiles/active", json={"name": "gpu"})
    await plane.close()
    assert running.json() == {"running": [{"model": "demo", "state": "ready", "ready": True}]}
    ids = [row["id"] for row in catalog.json()["data"]]
    assert ids == ["demo", "demo:fast"]
    assert empty_profiles.json() == {"active": None, "profiles": []}
    assert deactivate.status_code == 200
    assert select.status_code == 400


def test_prepare_body_applies_filters_then_use_model_name():
    filters = {
        "setParamsByID": {"demo:fast": {"temperature?": 0.2}},
        "setParams": {"top_k?": 5},
    }
    payload = {"model": "demo:fast", "messages": []}
    result = _prepare_body(
        json.dumps(payload).encode(), ResolvedModel("demo", "native-id", filters), "demo:fast"
    )
    body = json.loads(result)
    assert body["temperature"] == 0.2
    assert body["top_k"] == 5
    # useModelName rewrites the model after filters key off the requested id.
    assert body["model"] == "native-id"
    # Request-provided values win over soft defaults.
    payload = {"model": "demo", "temperature": 1.0}
    body = json.loads(
        _prepare_body(json.dumps(payload).encode(), ResolvedModel("demo", None, filters), "demo")
    )
    assert body["temperature"] == 1.0
    assert body["top_k"] == 5


def test_request_filters_nested_paths_and_escapes():
    payload = {"voice": {"rate": 1.0}}
    result = apply_request_filters(
        payload,
        {"setParams": {"voice.pitch": 3, "engine\\.option": "x", "voice.rate?": 9}},
        "m",
    )
    assert result["voice"]["pitch"] == 3
    assert result["engine.option"] == "x"
    assert result["voice"]["rate"] == 1.0


def test_versionless_and_audioapi_prefixes_are_stripped():
    assert _strip_version_prefix("/v/chat/completions") == "/chat/completions"
    assert _strip_version_prefix("/audioapi/v1/tasks/run") == "/v1/tasks/run"
    assert _strip_version_prefix("/v1/chat/completions") == "/v1/chat/completions"


def test_inference_key_crud_returns_secret_once(client):
    created = client.post("/api/inference/keys", json={"name": "desktop"})
    assert created.status_code == 201
    payload = created.json()
    assert payload["key"].startswith("sk-studio-")
    listing = client.get("/api/inference/keys")
    assert listing.status_code == 200
    listed = listing.json()["keys"]
    assert len(listed) == 1
    assert {
        name: listed[0][name]
        for name in (
            "id",
            "name",
            "prefix",
            "created_at",
            "expires_at",
            "models",
            "scopes",
            "rate_limit_per_minute",
            "last_used_at",
        )
    } == {
        "id": payload["id"],
        "name": "desktop",
        "prefix": payload["prefix"],
        "created_at": payload["created_at"],
        "expires_at": None,
        "models": [],
        "scopes": ["inference"],
        "rate_limit_per_minute": None,
        "last_used_at": None,
    }
    assert payload["key"] not in listing.text
    removed = client.delete(f"/api/inference/keys/{payload['id']}")
    assert removed.status_code == 204
    assert client.get("/api/inference/keys").json() == {"keys": []}


def test_ui_namespace_and_base_rewrite_pin_assets_to_model():
    assert _model_route("/ui/model%20one/docs/index.html") == (
        "model one",
        "/docs/index.html",
    )
    assert _model_route("/upstream/model%20one/v1/models") == (
        "model one",
        "/v1/models",
    )
    rewritten = _rewrite_html_base(
        b"<html><head><base href='/old/'><title>Engine</title></head></html>",
        "model one",
    )
    assert b'<base href="/ui/model%20one/">' in rewritten
    assert b"/old/" not in rewritten


def test_loading_progress_parses_percent_and_fraction_lines():
    assert _loading_progress("loading model: 42.5%") == 42.5
    assert _loading_progress("Loading 3/12 tensors") == 25.0
    assert _loading_progress("server listening") is None


# --- regression coverage for the review findings ---------------------------


def test_manager_exposes_model_lookup():
    """_model must stay a real method: _launch calls it on every cold start."""
    from backend.proxy.studio.manager import StudioProxyManager

    assert callable(getattr(StudioProxyManager, "_model", None))


@pytest.mark.asyncio
async def test_registry_cancel_during_cold_start_kills_spawned_process(monkeypatch):
    """A disconnect mid-readiness must not orphan the engine (finding 2)."""

    started = asyncio.Event()

    async def hanging_ready(generation):
        started.set()
        await asyncio.sleep(30)

    async def factory(model):
        return _spec(model)

    registry = ProcessRegistry(factory, readiness_timeout=30)
    monkeypatch.setattr(registry, "_wait_ready", hanging_ready)
    task = asyncio.create_task(registry.acquire("demo"))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    for _ in range(150):
        if registry._generations:
            await asyncio.sleep(0.02)
            continue
        break
    assert registry._generations == {}
    assert registry._starting == {}
    await registry.close()


@pytest.mark.asyncio
async def test_registry_receipt_failure_aborts_and_kills_launch():
    """Never serve a process that crash recovery cannot identify as owned."""

    async def factory(model):
        return _spec(model)

    registry = ProcessRegistry(factory, readiness_timeout=30)

    def fail_receipt(_spec, _revision, _pid):
        raise OSError("disk full")

    registry.set_launch_observer(fail_receipt)
    with pytest.raises(OSError, match="disk full"):
        await registry.acquire("demo")
    assert registry._generations == {}
    assert registry._starting == {}
    await registry.close()


@pytest.mark.asyncio
async def test_registry_stop_unloads_loading_generation(monkeypatch):
    """stop() must reach cold-start processes, not only _active (finding 5)."""
    started = asyncio.Event()

    async def hanging_ready(generation):
        started.set()
        await asyncio.sleep(30)

    async def factory(model):
        return _spec(model)

    registry = ProcessRegistry(factory, readiness_timeout=30)
    monkeypatch.setattr(registry, "_wait_ready", hanging_ready)
    task = asyncio.create_task(registry.acquire("demo"))
    await started.wait()
    assert await registry.stop("demo") is True
    task.cancel()
    from backend.proxy.studio.registry import RegistryError

    with pytest.raises((asyncio.CancelledError, RegistryError)):
        await task
    assert registry._generations == {}
    assert registry._starting == {}
    await registry.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("health_status", [301, 401, 404, 503])
async def test_registry_readiness_requires_success_status(health_status):
    """401/404/503 on /health means still initializing, not ready (finding 6).

    A 200 on "/" (WebUI/redirect served before weights load) must NOT mark a
    normal engine ready: only allowlisted engines without /health get that
    liveness fallback.
    """

    class StubbornEngine:
        pid = 999_999  # never exists: process.poll() stays None via patch below

        def poll(self):
            return None

    registry = ProcessRegistry(lambda model: _spec(model), readiness_timeout=0.35)
    root_hits = 0

    class WebUiBeforeWeightsClient:
        """Answers non-2xx on /health but 200 on '/': must stay not-ready."""

        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            nonlocal root_hits
            if url.endswith("/health"):
                return httpx.Response(health_status, request=httpx.Request("GET", url))
            root_hits += 1
            return httpx.Response(200, request=httpx.Request("GET", url))

    class Shim:
        HTTPError = httpx.HTTPError
        AsyncClient = WebUiBeforeWeightsClient

    async def failing_wait():
        import backend.proxy.studio.registry as registry_module

        original = registry_module.httpx
        registry_module.httpx = Shim
        try:
            await registry._wait_ready(Generation("demo", "a" * 64, 1, StubbornEngine(), "fake"))
        finally:
            registry_module.httpx = original

    with pytest.raises(Exception) as raised:
        await failing_wait()
    assert "readiness timed out" in str(raised.value)
    assert root_hits == 0  # non-allowlisted engines never probe "/"


@pytest.mark.asyncio
async def test_registry_root_fallback_only_for_allowlisted_engines(monkeypatch):
    """ "/" liveness is reserved for engines known to lack /health."""
    import backend.proxy.studio.registry as registry_module

    class AliveEngine:
        def poll(self):
            return None

    registry = ProcessRegistry(lambda model: _spec(model), readiness_timeout=1.0)
    seen: list[str] = []

    class RootOnlyClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            seen.append(url)
            if url.endswith("/health"):
                return httpx.Response(404, request=httpx.Request("GET", url))
            return httpx.Response(200, request=httpx.Request("GET", url))

    class Shim:
        HTTPError = httpx.HTTPError
        AsyncClient = RootOnlyClient

    monkeypatch.setattr(
        registry_module, "ENGINES_WITHOUT_HEALTH_ROUTE", frozenset({"legacy_engine"})
    )
    original = registry_module.httpx
    registry_module.httpx = Shim
    try:
        # Allowlisted engine becomes ready via the "/" fallback.
        await registry._wait_ready(Generation("demo", "a" * 64, 1, AliveEngine(), "legacy_engine"))
        assert any(url.endswith("/") for url in seen)
    finally:
        registry_module.httpx = original


@pytest.mark.asyncio
async def test_dataplane_consumes_client_credentials_before_upstream():
    """Bearer/Basic/x-api-key must never reach engine processes (finding 3)."""

    class Generation:
        upstream = "http://engine.test"

    class Registry:
        async def acquire(self, model):
            return Generation()

        def release(self, generation):
            pass

        def snapshot(self):
            return []

    seen = {}

    def upstream(request):
        seen["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b'{"ok":true}'),
        )

    plane = StudioDataPlane(Registry(), lambda name: ResolvedModel(name))
    await plane.client.aclose()
    plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "demo"},
            headers={
                "Authorization": "Bearer sk-studio-secret",
                "x-api-key": "sk-studio-secret",
                "X-Studio-Service-Key": "internal",
            },
        )
    await plane.close()
    assert response.status_code == 200
    assert "authorization" not in seen["headers"]
    assert "x-api-key" not in seen["headers"]
    assert "x-studio-service-key" not in seen["headers"]
    assert seen["headers"]["x-studio-model"] == "demo"


@pytest.mark.asyncio
async def test_dataplane_strips_connection_nominated_headers_both_directions():
    class Generation:
        upstream = "http://engine.test"

    class Registry:
        async def acquire(self, model):
            return Generation()

        def release(self, generation):
            pass

        def snapshot(self):
            return []

    seen = {}

    def upstream(request):
        seen["headers"] = request.headers
        return httpx.Response(
            200,
            headers={"Connection": "X-Engine-Hop", "X-Engine-Hop": "private"},
            stream=httpx.ByteStream(b"ok"),
        )

    plane = StudioDataPlane(Registry(), lambda name: ResolvedModel(name))
    await plane.client.aclose()
    plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "demo"},
            headers={"Connection": "X-Client-Hop", "X-Client-Hop": "private"},
        )
    await plane.close()
    assert response.status_code == 200
    assert "x-client-hop" not in seen["headers"]
    # httpx may add its own loopback Connection: keep-alive; the downstream's
    # nominated field is what must not survive the boundary.
    assert "x-engine-hop" not in response.headers
    assert "connection" not in response.headers


@pytest.mark.asyncio
async def test_disconnect_during_request_body_never_starts_engine():
    class Registry:
        async def acquire(self, model):
            raise AssertionError("engine must not start after downstream disconnect")

        def snapshot(self):
            return []

    resolved = False

    def resolver(_name):
        nonlocal resolved
        resolved = True
        return ResolvedModel("demo")

    plane = StudioDataPlane(Registry(), resolver)
    messages = iter(
        [
            {"type": "http.request", "body": b'{"model":"demo",', "more_body": True},
            {"type": "http.disconnect"},
        ]
    )
    sent = []

    async def receive():
        return next(messages)

    async def send(message):
        sent.append(message)

    await plane(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/chat/completions",
            "headers": [],
            "query_string": b"",
        },
        receive,
        send,
    )
    await plane.close()
    assert resolved is False
    assert sent == []


@pytest.mark.asyncio
async def test_dataplane_maps_upstream_connect_failure_to_502():
    """Connect errors happen before response start and must be 502 (finding 4)."""

    class Generation:
        upstream = "http://engine.test"

    class Registry:
        async def acquire(self, model):
            return Generation()

        def release(self, generation):
            pass

        def snapshot(self):
            return []

    def upstream(request):
        raise httpx.ConnectError("connection refused", request=request)

    plane = StudioDataPlane(Registry(), lambda name: ResolvedModel(name))
    await plane.client.aclose()
    plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        response = await client.post("/v1/chat/completions", json={"model": "demo"})
    await plane.close()
    assert response.status_code == 502
    assert "error" in response.json()


class _FakeRegistry:
    def __init__(self):
        self.activities = EventRing(10)
        self.unloads: list[str] = []

    async def stop(self, model_id):
        self.unloads.append(model_id)
        return True

    def active(self, model_id):
        return None

    def snapshot(self):
        return [{"model": "demo", "state": "ready", "ready": True}]


@pytest.mark.asyncio
async def test_inference_keys_cannot_administer_the_proxy():
    """Unload/logs/activity need the service credential, not any key."""
    from backend.inference_auth import authorize_admin_headers

    registry = _FakeRegistry()
    record, token = create_key("client")
    try:
        plane = StudioDataPlane(
            registry,
            lambda name: ResolvedModel(name),
            authorize=authorize_headers,
            authorize_admin=authorize_admin_headers,
            verify_token=lambda value: value == token,
            catalog=lambda: [],
        )
        transport = httpx.ASGITransport(app=plane)
        key_headers = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
            # Read-only observation stays available to ordinary keys.
            running = await client.get("/running", headers=key_headers)
            unload_key = await client.post("/api/models/unload", headers=key_headers)
            unload_one = await client.post("/api/models/unload/demo", headers=key_headers)
            logs_key = await client.get("/api/models/demo/logs", headers=key_headers)
            events_key = await client.get("/api/events", headers=key_headers)
            from backend.inference_auth import internal_service_key

            service = {"X-Studio-Service-Key": internal_service_key()}
            unload_service = await client.post("/api/models/unload", headers=service)
            # A key passed as ?key= is also never an admin credential.
            unload_query = await client.post(
                "/api/models/unload", params={"key": token}, headers=key_headers
            )
        await plane.close()
        assert running.status_code == 200
        assert unload_key.status_code == 403
        assert unload_one.status_code == 403
        assert logs_key.status_code == 403
        assert events_key.status_code == 403
        assert unload_service.status_code == 200
        assert unload_query.status_code == 403
        assert registry.unloads == ["demo"]
    finally:
        delete_key(record["id"])


@pytest.mark.asyncio
async def test_dataplane_enforces_request_body_limit():
    class Generation:
        upstream = "http://engine.test"

    class Registry:
        async def acquire(self, model):
            return Generation()

        def release(self, generation):
            pass

        def snapshot(self):
            return []

    def upstream(request):
        return httpx.Response(200, stream=httpx.ByteStream(b"{}"))

    plane = StudioDataPlane(Registry(), lambda name: ResolvedModel(name))
    await plane.client.aclose()
    plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    plane.max_body_bytes = 32
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        over = await client.post("/v1/chat/completions", content=b"x" * 200)
        under = await client.post("/v1/chat/completions", content=b"x" * 8)
    await plane.close()
    assert over.status_code == 413
    assert over.json()["limit_bytes"] == 32
    assert under.status_code == 200


@pytest.mark.asyncio
async def test_stop_proxy_drains_server_before_closing_runtime(monkeypatch):
    """Shutdown order: server drain → upstream client → engine processes."""
    from backend.proxy.studio.manager import StudioProxyManager

    order: list[str] = []

    manager = StudioProxyManager(port=1)

    class FakeServer:
        started = True
        should_exit = False

    server = FakeServer()

    async def serve():
        while not server.should_exit:
            await asyncio.sleep(0.01)
        order.append("server")

    class FakeDataplane:
        async def close(self):
            order.append("dataplane")

    class FakeRegistry:
        async def close(self):
            order.append("registry")

    manager._server = server
    manager._task = asyncio.create_task(serve())
    manager.dataplane = FakeDataplane()  # type: ignore[assignment]
    manager.registry = FakeRegistry()  # type: ignore[assignment]
    await manager.stop_proxy()
    assert order == ["server", "dataplane", "registry"]


@pytest.mark.asyncio
async def test_manager_rebuilds_runtime_after_stop(monkeypatch):
    """A stopped manager restarts with fresh registry/client, not closed ones."""
    from backend.proxy.studio import manager as manager_module
    from backend.proxy.studio.manager import StudioProxyManager

    manager = StudioProxyManager(port=1)
    old_plane = manager.dataplane
    old_registry = manager.registry
    await manager.stop_proxy()
    assert old_plane.client.is_closed

    class FakeServer:
        def __init__(self, config):
            self.started = False
            self.should_exit = False

        async def serve(self):
            self.started = True
            while not self.should_exit:
                await asyncio.sleep(0.01)

    class FakeConfig:
        def __init__(self, app, **kwargs):
            pass

    monkeypatch.setattr(manager_module.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(manager_module.uvicorn, "Config", FakeConfig)
    await manager.start_proxy()
    assert manager.dataplane is not old_plane
    assert manager.registry is not old_registry
    assert not manager.dataplane.client.is_closed
    assert manager.registry._closed is False
    await manager.stop_proxy()


@pytest.mark.asyncio
async def test_ui_grant_cookie_flow_keeps_key_out_of_upstream():
    """?key= entry mints an HttpOnly cookie; nav works; engines never see it."""
    from backend.inference_auth import authorize_admin_headers

    class Generation:
        upstream = "http://engine.test"

    class Registry:
        async def acquire(self, model):
            return Generation()

        def release(self, generation):
            pass

        def snapshot(self):
            return []

    seen = {}

    def upstream(request):
        seen["query"] = str(request.url.query)
        seen["cookie"] = request.headers.get("cookie", "")
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            stream=httpx.ByteStream(b"[]"),
        )

    record, token = create_key("browser")
    try:
        plane = StudioDataPlane(
            Registry(),
            lambda name: ResolvedModel(name),
            authorize=authorize_headers,
            authorize_admin=authorize_admin_headers,
            verify_token=verify_token,
        )
        await plane.client.aclose()
        plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        transport = httpx.ASGITransport(app=plane)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://proxy", follow_redirects=False
        ) as client:
            entry = await client.get(f"/ui/demo/?key={token}&tab=1")
            assert entry.status_code == 302
            # Raw pairs are preserved verbatim; only credentials are dropped.
            assert entry.headers["location"].split("?")[0] == "/ui/demo/"
            assert "tab=1" in entry.headers["location"]
            assert "key=" not in entry.headers["location"]
            cookie = entry.headers["set-cookie"]
            assert "studio_ui_grant=" in cookie
            assert "HttpOnly" in cookie and "SameSite=Lax" in cookie
            value = cookie.split(";")[0]
            # Navigation with the grant cookie (no Authorization header).
            nav = await client.get("/ui/demo/v1/models?x=1", headers={"cookie": value})
            assert nav.status_code == 200
            assert "key" not in seen["query"] and "x=1" in seen["query"]
            assert "studio_ui_grant" not in seen["cookie"]
            # Cookie never grants admin or inference POSTs.
            admin = await client.post("/api/models/unload", headers={"cookie": value})
            post_infer = await client.post("/v1/chat/completions", json={"model": "demo"})
            query_post = await client.post("/v1/chat/completions", params={"key": token})
        await plane.close()
        assert admin.status_code == 401
        assert post_infer.status_code == 401
        # ?key= is a browser affordance for GETs only, never inference POSTs.
        assert query_post.status_code == 401
    finally:
        delete_key(record["id"])


@pytest.mark.asyncio
async def test_ring_closes_subscriber_that_falls_behind():
    """Bounded subscriber queues: slow SSE clients get closed, not buffered."""
    ring: EventRing[str] = EventRing(capacity=50, overflow_limit=2)
    stream = ring.subscribe(0)
    # Park the generator inside queue.get() so appends exercise the live
    # fan-out path (a parked getter receives items by direct hand-off).
    first = asyncio.create_task(anext(stream))
    parked = False
    for _ in range(50):
        if any(getattr(q, "_getters", None) for q in ring._subscribers):
            parked = True
            break
        await asyncio.sleep(0)
    assert parked
    ring.append("a")  # queued
    ring.append("b")  # queue now full (maxsize=2)
    ring.append("c")  # overflow: oldest ("a") dropped, sentinel queued
    ring.append("d")  # subscriber already detached; ring keeps it for replay
    assert (await first).value == "b"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    # A reconnecting client replays the full history from the ring itself.
    replay = ring.subscribe(0)
    replayed = [(await anext(replay)).value for _ in range(4)]
    await replay.aclose()
    assert replayed == ["a", "b", "c", "d"]


def test_v1_inference_grant_accepts_basic_password():
    """Studio's /v1 surface must accept the same credentials as :2000 (finding 7)."""
    import base64

    from backend.access_policy import _inference_granted

    class FakeRequest:
        def __init__(self, headers):
            self.headers = headers

    record, token = create_key("basic client")
    try:
        basic = "Basic " + base64.b64encode(f":{token}".encode()).decode()
        assert _inference_granted(FakeRequest({"authorization": f"Bearer {token}"})) is True
        assert _inference_granted(FakeRequest({"authorization": basic})) is True
        assert _inference_granted(FakeRequest({"x-api-key": token})) is True
        assert _inference_granted(FakeRequest({"authorization": "Bearer nope"})) is False
    finally:
        delete_key(record["id"])
