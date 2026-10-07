import asyncio
from types import SimpleNamespace

import pytest

from backend.data_store import DataStore
from backend.services import model_benchmark


class _StreamResponse:
    def __init__(self, lines):
        self.lines = lines

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        for line in self.lines:
            await asyncio.sleep(0.001)
            yield line


class _Client:
    def __init__(self, lines):
        self.lines = lines
        self.requests = []

    def stream(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return _StreamResponse(self.lines)


@pytest.mark.asyncio
async def test_benchmark_records_revision_speed_and_observed_memory(tmp_path, monkeypatch):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_model(
        {
            "id": "model-1",
            "huggingface_id": "org/model",
            "format": "gguf",
            "config": {"engine": "llama_cpp", "ctx_size": 2048},
        }
    )
    lines = [
        'data: {"choices":[{"delta":{"content":"blue"}}]}',
        'data: {"choices":[{"delta":{"content":" sky"}}],"usage":{"completion_tokens":2}}',
        "data: [DONE]",
    ]
    client = _Client(lines)

    async def running(_proxy_name):
        return None

    async def gpu_info():
        return {"gpus": [{"memory": {"used": 1024}}]}

    monkeypatch.setattr(model_benchmark, "_require_running", running)
    monkeypatch.setattr(model_benchmark, "get_gpu_info", gpu_info)
    monkeypatch.setattr(model_benchmark, "shared_proxy_client", lambda _url: client)
    monkeypatch.setattr(
        model_benchmark,
        "get_llama_swap_client",
        lambda: SimpleNamespace(base_url="http://proxy"),
    )

    result = await model_benchmark.run_model_benchmark(
        store,
        "model-1",
        prompt="Describe the sky",
        max_tokens=8,
    )
    assert result["output_preview"] == "blue sky"
    assert result["completion_tokens"] == 2
    assert result["tokens_per_second"] > 0
    assert result["peak_observed_gpu_memory_bytes"] == 1024
    assert result["config_fingerprint"]
    assert model_benchmark.list_model_benchmarks(store, "model-1") == [result]
    assert client.requests[0][2]["json"]["stream"] is True


@pytest.mark.asyncio
async def test_benchmark_requires_a_verified_running_model(tmp_path, monkeypatch):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_model({"id": "model-1", "format": "gguf", "config": {}})

    async def stopped(_proxy_name):
        raise model_benchmark.BenchmarkError(
            "BENCHMARK_MODEL_NOT_RUNNING",
            "Start the model first.",
            409,
        )

    monkeypatch.setattr(model_benchmark, "_require_running", stopped)
    with pytest.raises(model_benchmark.BenchmarkError) as caught:
        await model_benchmark.run_model_benchmark(
            store,
            "model-1",
            prompt="hello",
            max_tokens=4,
        )
    assert caught.value.status_code == 409
