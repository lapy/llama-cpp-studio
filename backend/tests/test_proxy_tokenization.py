"""Exact engine counting, fallback policy and two-phase admission tests."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from backend.proxy.studio.accounting import Accounting, AdmissionError
from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane
from backend.proxy.studio.tokenization import count_input


def response(payload, status=200):
    return httpx.Response(
        status,
        headers={"content-type": "application/json"},
        stream=httpx.ByteStream(json.dumps(payload).encode()),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "engine,path,payload,endpoint,result,expected",
    [
        (
            "llama_cpp",
            "/chat/completions",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,fixture"},
                            }
                        ],
                    }
                ]
            },
            "/v1/chat/completions/input_tokens",
            {"input_tokens": 150},
            150,
        ),
        (
            "ik_llama",
            "/responses",
            {"previous_response_id": "context", "input": "continue"},
            "/v1/responses/input_tokens",
            {"input_tokens": 91},
            91,
        ),
        (
            "vllm",
            "/chat/completions",
            {"model": "real-model", "messages": [], "tools": [], "max_tokens": 4},
            "/tokenize",
            {"count": 17},
            17,
        ),
        ("1cat_vllm", "/completions", {"prompt": "hello"}, "/tokenize", {"count": 5}, 5),
        ("unsloth_llama", "/completions", {"prompt": "hi"}, "/tokenize", {"tokens": [1, 2, 3]}, 3),
        ("llama_cpp", "/embeddings", {"input": ["one", "two"]}, "/tokenize", {"tokens": [1, 2]}, 4),
    ],
)
async def test_engine_count_contract(engine, path, payload, endpoint, result, expected):
    calls = []

    def handler(request):
        assert request.url.path == endpoint
        calls.append(json.loads(request.content))
        return response(result)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        count = await count_input(client, "http://engine", engine, path, payload)
    assert count.tokens == expected and count.source == endpoint
    if engine == "vllm":
        assert "max_tokens" not in calls[0] and calls[0]["model"] == "real-model"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload,status",
    [
        ({"input_tokens": -1}, 200),
        ({"input_tokens": True}, 200),
        ({"input_tokens": "20"}, 200),
        ({"error": "not supported"}, 500),
    ],
)
async def test_bad_counts_are_not_treated_as_zero(payload, status):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: response(payload, status))
    ) as client:
        with pytest.raises(AdmissionError):
            await count_input(client, "http://engine", "llama_cpp", "/chat/completions", {})


async def setup(tmp_path, policy, count=20, count_status=200, engine="llama_cpp"):
    accounting = Accounting(str(tmp_path / "exact.sqlite3"))
    record = {
        "id": "exact-key",
        "name": "Exact client",
        "scopes": ["inference"],
        "daily_token_budget": 1000,
        "max_output_tokens": 10,
        **policy,
    }
    generation = SimpleNamespace(
        upstream="http://engine", engine_id=engine, process=SimpleNamespace(pid=42)
    )
    registry = SimpleNamespace(
        acquire=lambda model: asyncio.sleep(0, result=generation), release=lambda generation: None
    )
    app = StudioDataPlane(
        registry,
        lambda model: ResolvedModel("demo", engine_id=engine),
        accounting=accounting,
        identity_resolver=lambda headers: record,
    )
    await app.client.aclose()
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path.endswith("input_tokens"):
            return response({"input_tokens": count}, count_status)
        return response({"choices": [], "usage": {"completion_tokens": 3}})

    app.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return app, accounting, record, calls


@pytest.mark.asyncio
async def test_exact_multimodal_input_is_reserved_before_generation(tmp_path):
    app, accounting, record, calls = await setup(tmp_path, {"token_accounting": "exact"})
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={
                "model": "demo",
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "fixture"}}],
                    }
                ],
            },
        )
    await app.close()
    assert result.status_code == 200
    assert result.headers["x-ratelimit-remaining-tokens-day"] == "970"
    assert calls == ["/v1/chat/completions/input_tokens", "/v1/chat/completions"]
    row = accounting.query()["requests"][0]
    assert row["input_reserved"] == row["input_tokens"] == 20
    assert row["input_reserved_source"] == row["usage_source"] == "engine"
    assert accounting.key_usage(record)["daily_tokens"] == 23


@pytest.mark.asyncio
async def test_exact_count_can_reject_before_any_generation(tmp_path):
    app, accounting, record, calls = await setup(tmp_path, {"daily_token_budget": 25}, count=20)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post("/v1/chat/completions", json={"model": "demo", "messages": []})
    await app.close()
    assert result.status_code == 429
    assert calls == ["/v1/chat/completions/input_tokens"]
    assert accounting.key_usage(record)["daily_tokens"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,expected", [("auto", 200), ("exact", 503)])
async def test_old_engine_version_fallback_is_explicit(tmp_path, mode, expected):
    app, accounting, _, calls = await setup(tmp_path, {"token_accounting": mode}, count_status=404)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "demo", "messages": [{"role": "user", "content": "hello"}]},
        )
    await app.close()
    assert result.status_code == expected
    row = accounting.query()["requests"][0]
    if mode == "auto":
        assert row["input_reserved_source"] == "byte_estimate"
    else:
        assert calls == ["/v1/chat/completions/input_tokens"]


@pytest.mark.asyncio
async def test_complex_context_does_not_fall_back_to_byte_estimates(tmp_path):
    app, accounting, record, calls = await setup(
        tmp_path, {"token_accounting": "auto"}, count_status=404
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post(
            "/v1/responses",
            json={"model": "demo", "previous_response_id": "prior-context", "input": "continue"},
        )
    await app.close()
    assert result.status_code == 400 and len(calls) == 1
    assert accounting.key_usage(record)["daily_tokens"] == 0


@pytest.mark.asyncio
async def test_concurrent_exact_adjustments_do_not_overspend(tmp_path):
    accounting = Accounting(str(tmp_path / "adjust.sqlite3"))
    key = {"id": "client", "daily_token_budget": 100}
    await asyncio.gather(*(accounting.reserve_async(str(index), key, 0, 0) for index in range(20)))
    results = await asyncio.gather(
        *(accounting.adjust_async(str(index), key, 20) for index in range(20)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 5
    assert accounting.key_usage(key)["daily_tokens"] == 100


@pytest.mark.asyncio
async def test_active_reservations_remain_in_token_minute_window(tmp_path, monkeypatch):
    clock = 1900000000
    monkeypatch.setattr("backend.proxy.studio.accounting.time.time", lambda: clock)
    accounting = Accounting(str(tmp_path / "minute.sqlite3"))
    key = {"id": "client", "input_tokens_per_minute": 10}
    accounting.reserve("load", key, 8, 0)
    clock += 120
    with pytest.raises(AdmissionError):
        accounting.reserve("next", key, 8, 0)


@pytest.mark.asyncio
async def test_vllm_embedding_batch_counts_each_input():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return response({"count": 4})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        counted = await count_input(
            client,
            "http://engine",
            "vllm",
            "/embeddings",
            {"model": "embed", "input": ["one", "two"]},
        )
    assert counted.tokens == 8 and [row["prompt"] for row in seen] == ["one", "two"]


@pytest.mark.asyncio
async def test_output_reservations_include_all_batch_prompts_and_choices(tmp_path):
    app, _, _, _ = await setup(tmp_path, {})
    try:
        _, _, output = app._budget_body(
            b'{"prompt":["a","b"],"max_tokens":4,"n":3}',
            {"daily_token_budget": 1000},
            "/v1/completions",
        )
        assert output == 24
        with pytest.raises(AdmissionError):
            app._budget_body(
                b'{"prompt":["a","b"],"max_tokens":4,"n":3}',
                {"max_output_tokens": 20},
                "/v1/completions",
            )
    finally:
        await app.close()
