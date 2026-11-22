import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from backend.data_store import DataStore
from backend.routes.benchmarks import BenchmarkResponse, BenchmarkRunBody
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
        "get_proxy_client",
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


_LINES = [
    'data: {"choices":[{"delta":{"content":"blue"}}]}',
    'data: {"choices":[{"delta":{"content":" sky"}}],"usage":{"completion_tokens":2}}',
    "data: [DONE]",
]


def _patched_benchmark_store(tmp_path, monkeypatch, client):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.add_model(
        {
            "id": "model-1",
            "huggingface_id": "org/model",
            "format": "gguf",
            "config": {"engine": "llama_cpp", "ctx_size": 2048},
        }
    )

    async def running(_proxy_name):
        return None

    async def gpu_info():
        return {"gpus": [{"memory": {"used": 2048}}]}

    monkeypatch.setattr(model_benchmark, "_require_running", running)
    monkeypatch.setattr(model_benchmark, "get_gpu_info", gpu_info)
    monkeypatch.setattr(model_benchmark, "shared_proxy_client", lambda _url: client)
    monkeypatch.setattr(
        model_benchmark,
        "get_proxy_client",
        lambda: SimpleNamespace(base_url="http://proxy"),
    )
    return store


@pytest.mark.asyncio
async def test_benchmark_warmup_and_repeats_aggregate_measured_runs(tmp_path, monkeypatch):
    client = _Client(_LINES)
    store = _patched_benchmark_store(tmp_path, monkeypatch, client)

    result = await model_benchmark.run_model_benchmark(
        store,
        "model-1",
        prompt="Describe the sky",
        max_tokens=8,
        warmup=2,
        repeats=3,
    )

    # Two discarded warmups plus three measured runs, all through the proxy.
    assert len(client.requests) == 5
    assert all(request[2]["json"]["stream"] is True for request in client.requests)
    assert result["warmup"] == 2
    assert result["repeats"] == 3
    assert [run["index"] for run in result["runs"]] == [1, 2, 3]
    assert all(run["completion_tokens"] == 2 for run in result["runs"])
    assert result["completion_tokens"] == result["runs"][0]["completion_tokens"]
    assert result["time_to_first_token_ms"] == result["runs"][0]["time_to_first_token_ms"]
    assert result["total_seconds"] == result["runs"][0]["total_seconds"]
    assert result["tokens_per_second"] == result["runs"][0]["tokens_per_second"]
    assert set(result["aggregate"]) == {
        "time_to_first_token_ms",
        "total_seconds",
        "tokens_per_second",
    }
    for stats in result["aggregate"].values():
        assert set(stats) == {"min", "p50", "p95", "max"}
        assert stats["min"] <= stats["p50"] <= stats["p95"] <= stats["max"]
    # The stored row keeps the aggregation; history returns the same shape.
    assert model_benchmark.list_model_benchmarks(store, "model-1") == [result]


def test_aggregate_stats_uses_nearest_rank_and_ignores_unknown_samples():
    assert model_benchmark._stats([30, 10, 20, None]) == {
        "min": 10.0,
        "p50": 20.0,
        "p95": 30.0,
        "max": 30.0,
    }
    assert model_benchmark._stats([5.0]) == {
        "min": 5.0,
        "p50": 5.0,
        "p95": 5.0,
        "max": 5.0,
    }
    assert model_benchmark._stats([None, "unknown", True]) == {
        "min": None,
        "p50": None,
        "p95": None,
        "max": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [("warmup", 4), ("warmup", -1), ("repeats", 0), ("repeats", 6)],
)
def test_benchmark_request_fields_are_bounded(field, value):
    with pytest.raises(ValidationError):
        BenchmarkRunBody(model_id="model-1", **{field: value})


def _benchmark_payload() -> dict:
    stats = {"min": 1.0, "p50": 1.0, "p95": 1.0, "max": 1.0}
    return {
        "id": "b",
        "created_at": 1.0,
        "model_id": "model-1",
        "proxy_name": "model-1",
        "config_revision": "r",
        "config_fingerprint": "f",
        "prompt": "p",
        "max_tokens": 4,
        "time_to_first_token_ms": 1.0,
        "total_seconds": 2.0,
        "completion_tokens": 2,
        "tokens_per_second": 1.0,
        "peak_observed_gpu_memory_bytes": None,
        "output_preview": "x",
        "warmup": 1,
        "repeats": 2,
        "runs": [
            {
                "index": 1,
                "time_to_first_token_ms": 1.0,
                "total_seconds": 2.0,
                "completion_tokens": 2,
                "tokens_per_second": 1.0,
            }
        ],
        "aggregate": {
            "time_to_first_token_ms": dict(stats),
            "total_seconds": dict(stats),
            "tokens_per_second": dict(stats),
        },
    }


def test_benchmark_response_exposes_aggregation_and_defaults_old_rows():
    dumped = BenchmarkResponse(**_benchmark_payload()).model_dump()
    assert dumped["warmup"] == 1
    assert dumped["repeats"] == 2
    assert dumped["runs"][0]["index"] == 1
    assert dumped["aggregate"]["total_seconds"]["p95"] == 1.0

    legacy = _benchmark_payload()
    for field in ("warmup", "repeats", "runs", "aggregate"):
        legacy.pop(field)
    old = BenchmarkResponse(**legacy).model_dump()
    assert old["warmup"] == 0
    assert old["repeats"] == 1
    assert old["runs"] == []
    assert old["aggregate"] is None


def test_benchmark_route_forwards_warmup_and_repeats(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from backend.routes import benchmarks

    captured = {}

    async def fake_run(store, model_id, *, prompt, max_tokens, warmup, repeats):
        captured.update(
            model_id=model_id,
            prompt=prompt,
            max_tokens=max_tokens,
            warmup=warmup,
            repeats=repeats,
        )
        return _benchmark_payload()

    monkeypatch.setattr(benchmarks, "run_model_benchmark", fake_run)
    monkeypatch.setattr(benchmarks, "get_store", lambda: object())
    app = FastAPI()
    app.include_router(benchmarks.router, prefix="/api")

    response = TestClient(app).post(
        "/api/benchmarks/run",
        json={"model_id": "model-1", "warmup": 2, "repeats": 3},
    )

    assert response.status_code == 200
    assert captured == {
        "model_id": "model-1",
        "prompt": "Reply with a short description of the sky.",
        "max_tokens": 64,
        "warmup": 2,
        "repeats": 3,
    }
    assert response.json()["runs"][0]["index"] == 1
