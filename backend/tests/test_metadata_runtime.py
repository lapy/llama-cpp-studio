"""CPU-only checks for metadata offload and truthful runtime observations."""

import asyncio
import threading
import time

import httpx
import pytest

from backend.tests.test_api_routes import _install_temp_store, _seed_model


def test_running_model_lookup_failure_is_not_an_empty_catalog(monkeypatch):
    from backend.proxy.llama_swap.client import LlamaSwapClient

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def get(self, url, timeout):
            raise RuntimeError(f"proxy down: {url}")

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)

    with pytest.raises(RuntimeError, match="proxy down"):
        asyncio.run(LlamaSwapClient().get_running_models())


def test_model_list_marks_an_unreachable_proxy(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)

    from backend.proxy.llama_swap.client import LlamaSwapClient

    async def unavailable(self):
        raise RuntimeError("proxy down")

    monkeypatch.setattr(LlamaSwapClient, "get_running_models", unavailable)

    response = client.get("/api/models")
    assert response.status_code == 200
    quant = response.json()[0]["quantizations"][0]
    assert quant["runtime_quality"] == "unreachable"
    assert quant["is_active"] is False
    assert quant["status"] is None
    assert quant["run_state"] is None
    assert quant["runtime_observed_at"] is None


def test_model_list_preserves_last_running_state_during_an_outage(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)

    from backend.proxy.llama_swap.client import LlamaSwapClient

    calls = {"n": 0}

    async def flaky(self):
        calls["n"] += 1
        if calls["n"] == 1:
            return {"running": [{"model": "org-model.q4_k_m", "state": "ready"}]}
        raise RuntimeError("proxy down")

    monkeypatch.setattr(LlamaSwapClient, "get_running_models", flaky)

    first = client.get("/api/models")
    assert first.status_code == 200
    verified = first.json()[0]["quantizations"][0]
    assert verified["runtime_quality"] == "verified"
    assert verified["is_active"] is True
    assert verified["status"] == "ready"
    assert verified["runtime_observed_at"]

    second = client.get("/api/models")
    assert second.status_code == 200
    stale = second.json()[0]["quantizations"][0]
    assert stale["runtime_quality"] == "stale"
    assert stale["is_active"] is True
    assert stale["status"] == "ready"
    assert stale["runtime_observed_at"] == verified["runtime_observed_at"]


@pytest.mark.asyncio
async def test_slow_file_size_lookup_leaves_liveness_responsive(monkeypatch):
    from backend.main import app
    from backend.routes import models as models_routes

    def slow(model_id, files):
        time.sleep(2)
        return {name: 1 for name in files}

    monkeypatch.setattr(models_routes, "get_accurate_file_sizes", slow)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        sizes = asyncio.create_task(
            client.get(
                "/api/models/search/org/repo/file-sizes",
                params={"filenames": "a.gguf"},
            )
        )
        await asyncio.sleep(0.05)
        gaps = []
        started = time.perf_counter()
        while time.perf_counter() - started < 1.2:
            tick = time.perf_counter()
            live = await client.get("/api/live")
            gaps.append(time.perf_counter() - tick)
            assert live.status_code == 200
            await asyncio.sleep(0.01)
        response = await sizes

    assert response.status_code == 200
    assert response.json()["sizes"] == {"a.gguf": 1}
    assert gaps
    assert max(gaps) < 0.5


@pytest.mark.asyncio
async def test_file_size_lookup_times_out_and_stays_bounded(monkeypatch):
    from backend.main import app
    from backend.models import metadata_workers
    from backend.routes import models as models_routes

    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    def slow(model_id, files):
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        try:
            time.sleep(0.3)
            return {name: 1 for name in files}
        finally:
            with lock:
                state["current"] -= 1

    monkeypatch.setattr(models_routes, "get_accurate_file_sizes", slow)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        responses = await asyncio.gather(
            *[
                client.get(
                    "/api/models/search/org/repo/file-sizes",
                    params={"filenames": f"file-{index}.gguf"},
                )
                for index in range(8)
            ]
        )

    assert all(response.status_code == 200 for response in responses)
    assert 2 <= state["peak"] <= metadata_workers.METADATA_WORKERS

    monkeypatch.setattr(metadata_workers, "METADATA_TIMEOUT_SECONDS", 0.05)

    def too_slow(model_id, files):
        time.sleep(0.4)
        return {name: 1 for name in files}

    monkeypatch.setattr(models_routes, "get_accurate_file_sizes", too_slow)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        timed_out = await client.get(
            "/api/models/search/org/repo/file-sizes",
            params={"filenames": "slow.gguf"},
        )
    assert timed_out.status_code == 504
    assert "timed out" in timed_out.json()["detail"]
    await asyncio.sleep(0.45)


@pytest.mark.asyncio
async def test_quantization_size_fallback_is_async_and_bounded(monkeypatch):
    from backend.main import app
    from backend.models import hub

    async def empty_lookup(huggingface_id, quantizations):
        return {}

    monkeypatch.setattr(hub, "get_quantization_sizes_from_hf", empty_lookup)

    state = {"current": 0, "peak": 0}
    lock = asyncio.Lock()

    class FakeClient:
        async def head(self, url, timeout=None):
            async with lock:
                state["current"] += 1
                state["peak"] = max(state["peak"], state["current"])
            try:
                await asyncio.sleep(0.15)
                return httpx.Response(
                    200,
                    headers={"content-length": "128"},
                    request=httpx.Request("HEAD", url),
                )
            finally:
                async with lock:
                    state["current"] -= 1

    monkeypatch.setattr(
        "backend.http_client.get_http_client",
        lambda: FakeClient(),
    )

    quantizations = {f"Q{index}": {"filename": f"q{index}.gguf"} for index in range(6)}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        response = await client.post(
            "/api/models/quantization-sizes",
            json={"huggingface_id": "org/repo", "quantizations": quantizations},
        )

    assert response.status_code == 200
    body = response.json()["quantizations"]
    assert set(body) == set(quantizations)
    assert 2 <= state["peak"] <= 4


@pytest.mark.asyncio
async def test_quantization_size_timeout_is_visible(monkeypatch):
    from backend.main import app
    from backend.models import hub, metadata_workers

    monkeypatch.setattr(metadata_workers, "METADATA_TIMEOUT_SECONDS", 0.05)

    def too_slow(huggingface_id, quantizations):
        time.sleep(0.4)
        return {}

    monkeypatch.setattr(hub, "_quantization_sizes_from_hf_blocking", too_slow)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        response = await client.post(
            "/api/models/quantization-sizes",
            json={
                "huggingface_id": "org/repo",
                "quantizations": {"Q4": {"filename": "model.gguf"}},
            },
        )
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]
    await asyncio.sleep(0.45)
