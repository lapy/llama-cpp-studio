import asyncio

import httpx

from backend.llama_swap_client import LlamaSwapClient


def _response(method: str, url: str, status_code: int = 200, text: str = "ok"):
    request = httpx.Request(method, url)
    return httpx.Response(status_code, request=request, text=text)


def test_unload_all_models_prefers_current_admin_api(monkeypatch):
    client = LlamaSwapClient()
    seen = []

    async def fake_request(method, path, *, timeout=10.0):
        seen.append((method, path, timeout))
        return _response(method, f"http://localhost:2000{path}", text='{"msg":"ok"}')

    monkeypatch.setattr(client, "request", fake_request)

    result = asyncio.run(client.unload_all_models())

    assert result == '{"msg":"ok"}'
    assert seen == [("POST", "/api/models/unload", 10)]


def test_unload_all_models_falls_back_to_legacy_endpoint(monkeypatch):
    client = LlamaSwapClient()
    seen = []

    async def fake_request(method, path, *, timeout=10.0):
        seen.append((method, path, timeout))
        if (method, path) == ("POST", "/api/models/unload"):
            return _response(method, "http://localhost:2000/api/models/unload", 404)
        return _response(method, "http://localhost:2000/unload", text="OK")

    monkeypatch.setattr(client, "request", fake_request)

    result = asyncio.run(client.unload_all_models())

    assert result == "OK"
    assert seen == [
        ("POST", "/api/models/unload", 10),
        ("GET", "/unload", 10),
    ]


def test_unload_all_models_does_not_mask_non_route_errors(monkeypatch):
    client = LlamaSwapClient()

    async def fake_request(method, path, *, timeout=10.0):
        return _response(method, f"http://localhost:2000{path}", 500, text="boom")

    monkeypatch.setattr(client, "request", fake_request)

    try:
        asyncio.run(client.unload_all_models())
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 500
    else:  # pragma: no cover
        raise AssertionError("Expected unload_all_models() to raise on 500")


def test_start_and_stop_passthrough_use_expected_llama_swap_routes(monkeypatch):
    client = LlamaSwapClient()
    seen = []

    async def fake_request(method, path, *, timeout=10.0):
        seen.append((method, path, timeout))
        return _response(method, f"http://localhost:2000{path}", text="ok")

    monkeypatch.setattr(client, "request", fake_request)

    asyncio.run(client.start_model_passthrough("demo-model"))
    asyncio.run(client.stop_model_passthrough("demo-model"))

    assert seen == [
        ("GET", "/upstream/demo-model/v1/models", 30),
        ("POST", "/api/models/unload/demo-model", 10),
    ]


def test_get_model_info_uses_upstream_route(monkeypatch):
    requested = {}

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, timeout):
            requested["url"] = url
            requested["timeout"] = timeout
            return _response("GET", url, text='{"data":true}')

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    result = asyncio.run(LlamaSwapClient().get_model_info("demo-model", "v1/models"))

    assert result == {"data": True}
    assert requested == {
        "url": "http://localhost:2000/upstream/demo-model/v1/models",
        "timeout": 5,
    }
