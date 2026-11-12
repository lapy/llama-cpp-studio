"""Engine-compatibility discovery routes served by the native proxy."""

from types import SimpleNamespace

import httpx
import pytest

from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane


class _Registry:
    def __init__(self, running=(), active=None):
        self.running = list(running)
        self.active_generation = active
        self.acquire_calls = []

    async def acquire(self, model):
        self.acquire_calls.append(model)
        raise AssertionError("a discovery route must not start an engine")

    def release(self, generation):
        return None

    def snapshot(self):
        return list(self.running)

    def active(self, model):
        if self.active_generation is None:
            return None
        return self.active_generation if model == "demo" else None


CATALOG = [
    {
        "id": "demo",
        "studio_model_id": "hf/demo",
        "engine": "llama_cpp",
        "aliases": ["demo:fast"],
        "display_name": "Demo model",
        "publisher": "hf",
        "quantization": "Q4_K_M",
        "format": "gguf",
        "model_type": "qwen",
        "file_size": 1234,
        "downloaded_at": "2026-01-02T03:04:05+00:00",
        "context_length": 8192,
        "capabilities": {"in": ["text"], "out": ["text"], "context": 8192},
    },
    {
        "id": "embed",
        "studio_model_id": "hf/embed",
        "engine": "llama_cpp",
        "aliases": [],
        "display_name": "Embedding model",
        "publisher": "hf",
        "quantization": "F16",
        "format": "gguf",
        "model_type": "bert",
        "file_size": 4321,
        "downloaded_at": "2026-01-03T03:04:05+00:00",
        "context_length": 512,
        "capabilities": {"in": ["text"], "out": ["embedding"], "context": 512},
    },
]


@pytest.mark.asyncio
async def test_compat_discovery_routes_answer_without_loading_models():
    registry = _Registry()
    plane = StudioDataPlane(
        registry, lambda _name: None, catalog=lambda: list(CATALOG)
    )
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        version = await client.get("/version")
        props = await client.get("/props", params={"model": "demo"})
        v1props = await client.get("/v1/props", params={"model": "demo"})
        tags = await client.get("/api/tags")
        lmstudio = await client.get("/api/v1/models")
        one = await client.get("/v1/models/demo:fast")
        missing = await client.get("/v1/models/absent")
        show = await client.post("/api/show", json={"name": "demo"})
        show_alias = await client.post("/api/show", json={"model": "demo:fast"})
        show_missing = await client.post("/api/show", json={"name": "absent"})
    await plane.close()

    assert registry.acquire_calls == []
    assert version.status_code == 200 and version.json().get("version")
    assert props.status_code == v1props.status_code == 200
    assert props.json()["default_generation_settings"]["n_ctx"] == 8192
    assert v1props.json()["default_generation_settings"]["n_ctx"] == 8192

    listed = tags.json()["models"]
    assert [item["name"] for item in listed] == ["demo", "embed"]
    assert listed[0]["details"]["quantization_level"] == "Q4_K_M"
    assert listed[0]["size"] == 1234

    lm_models = lmstudio.json()["models"]
    assert {item["key"] for item in lm_models} == {"demo", "embed"}
    assert lm_models[0]["max_context_length"] == 8192
    assert lm_models[1]["type"] == "embedding"

    assert one.status_code == 200
    assert one.json()["id"] == "demo:fast"
    assert one.json()["owned_by"] == "llama_cpp"
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "model_not_found"

    assert show.status_code == show_alias.status_code == 200
    shown = show.json()
    assert shown["model_info"]["qwen.context_length"] == 8192
    assert "num_ctx 8192" in shown["parameters"]
    assert "completion" in shown["capabilities"]
    assert show_missing.status_code == 404


@pytest.mark.asyncio
async def test_compat_routes_scope_the_catalog_to_the_calling_key():
    registry = _Registry()
    plane = StudioDataPlane(
        registry,
        lambda _name: None,
        catalog=lambda: list(CATALOG),
        identity_resolver=lambda _headers: {
            "id": "scoped",
            "name": "Scoped",
            "models": ["demo"],
            "scopes": ["models:read"],
        },
    )
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        tags = await client.get("/api/tags")
        lmstudio = await client.get("/api/v1/models")
        hidden = await client.get("/v1/models/embed")
    await plane.close()

    assert [item["name"] for item in tags.json()["models"]] == ["demo"]
    assert [item["key"] for item in lmstudio.json()["models"]] == ["demo"]
    assert hidden.status_code == 404


@pytest.mark.asyncio
async def test_props_and_version_prefer_a_running_engine_answer():
    generation = SimpleNamespace(upstream="http://engine.test")
    registry = _Registry(active=generation)
    plane = StudioDataPlane(registry, lambda name: ResolvedModel(name), catalog=lambda: list(CATALOG))
    await plane.client.aclose()
    seen = []

    def upstream(request):
        seen.append((request.url.path, request.headers.get("x-studio-model")))
        if request.url.path == "/props":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={
                    "default_generation_settings": {"n_ctx": 32768},
                    "engine": "llama.cpp",
                },
            )
        if request.url.path == "/version":
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                json={"version": "0.8.5"},
            )
        return httpx.Response(404, json={"error": "not found"})

    plane.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        props = await client.get("/props", params={"model": "demo"})
        version = await client.get("/version", params={"model": "demo"})
    await plane.close()

    assert props.json()["default_generation_settings"]["n_ctx"] == 32768
    assert version.json()["version"] == "0.8.5"
    assert seen == [("/props", "demo"), ("/version", "demo")]


@pytest.mark.asyncio
async def test_engine_404_falls_back_to_synthesized_compatibility_payload():
    generation = SimpleNamespace(upstream="http://engine.test")
    registry = _Registry(active=generation)
    plane = StudioDataPlane(registry, lambda name: ResolvedModel(name), catalog=lambda: list(CATALOG))
    await plane.client.aclose()
    plane.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(404, json={"error": "not found"})
        )
    )
    transport = httpx.ASGITransport(app=plane)
    async with httpx.AsyncClient(transport=transport, base_url="http://proxy") as client:
        props = await client.get("/props", params={"model": "demo"})
        version = await client.get("/version", params={"model": "demo"})
    await plane.close()

    assert props.status_code == 200
    assert props.json()["default_generation_settings"]["n_ctx"] == 8192
    assert version.status_code == 200
    assert version.json()["proxy"] == "studio"
