import asyncio
import threading
from types import SimpleNamespace

import httpx
import pytest

from backend.proxy.studio.dataplane import StudioDataPlane
from backend.proxy.studio.manager import StudioProxyManager
from backend.proxy.studio.registry import ProcessRegistry
from backend.tests.test_proxy_supervision import spec
from backend.tests.test_api_routes import _install_temp_store


@pytest.mark.asyncio
async def test_route_publication_preserves_active_stream_and_unrelated_processes(monkeypatch):
    models = [{"id": "a"}, {"id": "b"}]
    monkeypatch.setattr(
        "backend.proxy.studio.manager.get_store",
        lambda: SimpleNamespace(list_models=lambda: models),
    )
    monkeypatch.setattr(
        "backend.proxy.studio.manager.compile_model_runtime",
        lambda model: SimpleNamespace(
            proxy=SimpleNamespace(
                model_id=model["id"],
                catalog_id=model["id"],
                engine_id="fake",
                aliases=[],
                use_model_name=None,
                filters={},
            )
        ),
    )
    manager = object.__new__(StudioProxyManager)
    manager._route_lock = threading.Lock()
    manager._route_cursor = {}
    manager.registry = registry = ProcessRegistry(
        lambda model: asyncio.sleep(0, result=spec(model))
    )
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0))
    before = manager.apply_routing({"routing_aliases": {"fast": ["a"]}})
    app = StudioDataPlane(registry, manager._resolve)
    await app.client.aclose()
    started, finish = asyncio.Event(), asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            started.set()
            yield b'data: {"model":"a"}\n\n'
            await finish.wait()
            yield b"data: [DONE]\n\n"

    def upstream(request):
        if request.headers["x-studio-model"] == "a":
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=Stream()
            )
        return httpx.Response(200, stream=httpx.ByteStream(b'{"model":"b"}'))

    app.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            active = asyncio.create_task(
                client.post("/v1/chat/completions", json={"model": "fast", "stream": True})
            )
            await started.wait()
            first = registry.active("a")
            pid = first.process.pid
            after = manager.apply_routing({"routing_aliases": {"fast": ["b"]}})
            assert before != after
            next_request = await client.post("/v1/chat/completions", json={"model": "fast"})
            assert next_request.json()["model"] == "b"
            assert first.process.pid == pid and first.process.poll() is None and first.inflight == 1
            finish.set()
            assert b'"model":"a"' in (await active).content
            assert registry.active("a").process.pid == pid
    finally:
        finish.set()
        await app.close()
        await registry.close()


@pytest.mark.asyncio
async def test_settings_save_applies_routing_live_and_preserves_restart_requirement(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.update_settings({"proxy": {"backend": "studio"}})
    manager = StudioProxyManager(2000)
    manager._task = SimpleNamespace(done=lambda: False)
    manager._server = SimpleNamespace(started=True)
    monkeypatch.setattr("backend.routes.proxy_admin.get_studio_proxy_manager", lambda: manager)
    try:
        routing = client.put(
            "/api/proxy/settings", json={"backend": "studio", "routing_aliases": {"fast": ["demo"]}}
        ).json()
        assert routing["restart_required"] is False
        assert "routing_aliases" in routing["applied_live_fields"]
        assert manager._routing.aliases["fast"] == ("demo",)
        changed = client.put(
            "/api/proxy/settings", json={"backend": "studio", "request_timeout_seconds": 45}
        ).json()
        assert changed["restart_required"] is True
        changed_route = client.put(
            "/api/proxy/settings",
            json={"backend": "studio", "request_timeout_seconds": 45, "default_model": "demo"},
        ).json()
        assert (
            changed_route["restart_required"] is True
        )  # a route edit cannot clear pending serving settings
    finally:
        await manager.dataplane.close()
        await manager.registry.close()


def test_compiled_route_cache_invalidates_on_configuration_revision(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.write_document("models.yaml", {"models": [{"id": "a"}]})
    calls = []

    def compile_model(model):
        calls.append(model["id"])
        return SimpleNamespace(
            proxy=SimpleNamespace(
                model_id=model["id"],
                catalog_id=model["id"],
                engine_id="fake",
                aliases=[],
                use_model_name=None,
                filters={},
            )
        )

    monkeypatch.setattr("backend.proxy.studio.manager.compile_model_runtime", compile_model)
    manager = object.__new__(StudioProxyManager)
    manager._catalog_lock = threading.RLock()
    manager._compiled_cache = None
    first = manager._compiled_models()
    assert manager._compiled_models() is first
    assert calls == ["a"]
    store.write_document("models.yaml", {"models": [{"id": "a"}, {"id": "b"}]})
    assert len(manager._compiled_models()) == 2
    assert calls == ["a", "a", "b"]
