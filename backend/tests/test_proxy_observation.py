"""Observation contract, timing formulas, cache provenance and ledger components."""

import gzip
import json
from types import SimpleNamespace

import httpx
import pytest

from backend.proxy.studio.accounting import Accounting
from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane
from backend.proxy.studio.observation import (
    PROTOCOL_NATIVE_LLAMA,
    SCHEMA_VERSION,
    finalize_observation,
    normalize_cache,
    protocol_for_path,
)
from backend.proxy.studio.tokenization import count_input
from backend.proxy.studio.usage import UsageObserver


def observe(data: bytes, sse: bool = False) -> UsageObserver:
    observer = UsageObserver(sse)
    observer.feed(data)
    observer.finish()
    return observer


def test_stream_event_shapes_are_recognized():
    chat = observe(
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\ndata: [DONE]\n\n',
        sse=True,
    )
    assert chat.saw_token and chat.saw_text and chat.output_event_count == 1
    assert chat.terminal_kind == "done" and chat.finished

    completion = observe(b'data: {"choices":[{"text":"legacy"}]}\n\ndata: [DONE]\n\n', sse=True)
    assert completion.saw_token and completion.saw_text

    tool = observe(
        b'data: {"type":"response.function_call_arguments.delta","delta":"{}"}\n\ndata: [DONE]\n\n',
        sse=True,
    )
    assert tool.saw_token and not tool.saw_text

    reasoning = observe(
        b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n\ndata: [DONE]\n\n',
        sse=True,
    )
    assert reasoning.saw_token and not reasoning.saw_text

    native = observe(b'data: {"content":"native"}\n\ndata: [DONE]\n\n', sse=True)
    assert native.saw_token and native.saw_text

    done_only = observe(
        b'data: {"type":"response.output_text.delta","delta":"hi"}\n\n'
        b'data: {"type":"response.output_text.done","text":"hi"}\n\n'
        b'data: {"type":"response.completed"}\n\n',
        sse=True,
    )
    assert done_only.output_event_count == 1


def test_nonstream_protocol_usage_and_cache_details():
    responses = observe(
        json.dumps(
            {
                "response": {
                    "usage": {
                        "input_tokens": 900,
                        "input_tokens_details": {
                            "cached_tokens": 800,
                            "cache_write_tokens": 0,
                        },
                        "output_tokens": 5,
                        "output_tokens_details": {"reasoning_tokens": 3},
                    },
                    "status": "completed",
                }
            }
        ).encode()
    )
    assert responses.input_tokens == 900
    assert responses.input_tokens_cached == 800
    assert responses.output_tokens == 5
    assert responses.output_reasoning_tokens == 3
    assert responses.structural_complete and responses.usage_authoritative

    native = observe(
        b'{"tokens_evaluated":8000,"tokens_predicted":100,"tokens_cached":7500,'
        b'"timings":{"prompt_n":512,"predicted_per_second":42.5}}'
    )
    assert native.input_tokens == 8000
    assert native.input_tokens_cached == 7500
    assert native.input_tokens_processed == 512
    assert native.native_timing["predicted_per_second"] == 42.5
    assert native.usage_authoritative

    sglang = observe(
        b'{"text":"hi","meta_info":{"prompt_tokens":20,"completion_tokens":2,'
        b'"cached_tokens":16,"finish_reason":"stop"}}'
    )
    assert sglang.input_tokens == 20
    assert sglang.input_tokens_cached == 16
    assert sglang.output_tokens == 2


def test_nested_and_untrusted_usage_paths_are_ignored():
    observer = observe(b'{"output":{"usage":{"prompt_tokens":9,"completion_tokens":9}}}')
    assert observer.input_tokens is None and observer.output_tokens is None

    array_usage = observe(b'{"choices":[{"usage":{"prompt_tokens":999,"completion_tokens":99}}]}')
    assert array_usage.input_tokens is None and array_usage.output_tokens is None

    nested_response = observe(
        b'{"response":{"output":[{"usage":{"prompt_tokens":999,"completion_tokens":99}}]}}'
    )
    assert nested_response.input_tokens is None and nested_response.output_tokens is None

    supported = observe(
        b'{"response":{"usage":{"input_tokens":7,"output_tokens":2},"status":"completed"}}'
    )
    assert supported.input_tokens == 7 and supported.output_tokens == 2


def test_processed_input_is_order_independent_and_works_for_sse():
    usage_first = observe(
        b'{"usage":{"prompt_tokens":10,"completion_tokens":2},"timings":{"prompt_n":7}}'
    )
    timing_first = observe(
        b'{"timings":{"prompt_n":7},"usage":{"prompt_tokens":10,"completion_tokens":2}}'
    )
    assert usage_first.input_tokens_processed == 7
    assert timing_first.input_tokens_processed == 7

    sse = observe(
        b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2}}\n\n'
        b'data: {"timings":{"prompt_n":7}}\n\ndata: [DONE]\n\n',
        sse=True,
    )
    assert sse.input_tokens_processed == 7
    assert sse.native_timing["prompt_n"] == 7


def test_truncated_or_error_terminated_usage_is_not_authoritative():
    truncated = observe(b'{"content":"x","usage":{"prompt_tokens":3,"completion_tokens":4}')
    assert truncated.input_tokens == 3
    assert not truncated.structural_complete
    assert not truncated.usage_authoritative
    assert "truncated_json" in truncated.reasons

    error = observe(
        b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
        b'data: {"error":{"message":"boom"}}\n\ndata: [DONE]\n\n',
        sse=True,
    )
    assert error.input_tokens == 3 and error.output_tokens == 2
    assert error.terminal_error and error.error_after_usage
    assert not error.usage_authoritative


def test_output_after_usage_makes_output_snapshot_stale():
    observer = observe(
        b'data: {"choices":[{"delta":{"content":"a"}}],'
        b'"usage":{"prompt_tokens":8,"completion_tokens":1}}\n\n'
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\ndata: [DONE]\n\n',
        sse=True,
    )
    assert observer.usage_authoritative
    assert observer.usage_output_stale
    assert not observer.output_authoritative
    observation = finalize_observation(
        input_total=observer.input_tokens,
        output_total=observer.output_tokens,
        usage_reported=True,
        usage_authoritative=observer.usage_authoritative,
        output_authoritative=observer.output_authoritative,
        observed_output=observer.output_tokens,
        dispatched=True,
        transport_complete=True,
        terminal_kind="done",
        latency_ms=1000,
    )
    assert observation["input_tokens_total"] == 8
    assert observation["output_tokens"] is None
    assert observation["observed_output_tokens"] == 1
    assert observation["usage_status"] == "partial"
    assert "output_stale_after_usage" in observation["observation_reasons"]


def test_native_llama_stream_terminal_and_protocol():
    observer = observe(
        b'data: {"content":"hi","stop":false,"tokens_evaluated":10,"tokens_predicted":1}\n\n'
        b'data: {"stop":true,"tokens_evaluated":10,"tokens_predicted":2,'
        b'"tokens_cached":0,"timings":{"prompt_n":10,"predicted_per_second":5}}\n\n',
        sse=True,
    )
    assert observer.finished and observer.terminal_kind == "native_finished"
    assert observer.usage_authoritative and observer.output_authoritative
    assert observer.input_tokens == 10 and observer.output_tokens == 2
    assert observer.native_timing["predicted_per_second"] == 5
    assert protocol_for_path("/completion") == PROTOCOL_NATIVE_LLAMA


def test_final_snapshot_does_not_inherit_earlier_cache_detail():
    observer = observe(
        b'data: {"usage":{"prompt_tokens":8,"completion_tokens":2,'
        b'"prompt_tokens_details":{"cached_tokens":7}}}\n\n'
        b'data: {"usage":{"prompt_tokens":8,"completion_tokens":3}}\n\n'
        b"data: [DONE]\n\n",
        sse=True,
    )
    assert observer.input_tokens == 8
    assert observer.input_tokens_cached is None
    assert observer.output_tokens == 3
    assert observer.output_authoritative


def test_engine_timing_is_extracted_from_openai_and_metrics_envelopes():
    observer = observe(
        b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2},'
        b'"timings":{"prompt_n":2,"predicted_per_second":5}}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":8,"completion_tokens":2},'
        b'"metrics":{"generation_time_ms":10,"tokens_per_second":9}}\n\n'
        b"data: [DONE]\n\n",
        sse=True,
    )
    assert observer.native_timing["prompt_n"] == 2
    assert observer.native_timing["predicted_per_second"] == 5
    assert observer.native_timing["metrics.generation_time_ms"] == 10


def test_json_grammar_and_single_root_completeness():
    bad_literal = observe(b'{"usage":{"prompt_tokens":8,"completion_tokens":2},"x":garbage}')
    assert bad_literal.input_tokens == 8
    assert not bad_literal.structural_complete
    assert "malformed_json" in bad_literal.reasons

    concatenated = observe(b'{"usage":{"prompt_tokens":8,"completion_tokens":2}}{}')
    assert not concatenated.structural_complete
    assert "multiple_json_roots" in concatenated.reasons

    trailing_comma = observe(b'{"usage":{"prompt_tokens":8,"completion_tokens":2},}')
    assert not trailing_comma.structural_complete
    assert "trailing_json_comma" in trailing_comma.reasons

    bad_escape = observe(
        b'{"usage":{"prompt_tokens":8,"completion_tokens":2}}'[:-1] + b',"s":"\\x"}'
    )
    assert not bad_escape.structural_complete


@pytest.mark.asyncio
async def test_vllm_responses_counting_does_not_send_an_unrelated_shape():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"count": 3})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await count_input(
            client,
            "http://engine",
            "vllm",
            "/v1/responses",
            {"model": "demo", "input": "hello", "instructions": "system"},
            revision="rev-vllm-responses",
        )
    assert result is None
    assert calls == []


def test_legacy_ledger_migrates_to_per_field_components(tmp_path):
    import sqlite3
    import time

    from backend.proxy.studio.accounting import periods

    path = str(tmp_path / "legacy.sqlite3")
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE reservations (
            id TEXT PRIMARY KEY, key_id TEXT NOT NULL, started REAL NOT NULL,
            input INTEGER NOT NULL, output INTEGER NOT NULL, active INTEGER NOT NULL,
            known INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE usage_totals (
            key_id TEXT, period TEXT, started REAL, input INTEGER, output INTEGER,
            PRIMARY KEY(key_id, period, started)
        );
        CREATE TABLE requests (
            id TEXT PRIMARY KEY, started REAL NOT NULL, key_id TEXT, model TEXT,
            state TEXT, status INTEGER, payload TEXT NOT NULL
        );
        CREATE TABLE owners(id TEXT PRIMARY KEY, pid INTEGER, stamp TEXT);
        CREATE TABLE metadata(name TEXT PRIMARY KEY, value REAL);
        CREATE TABLE key_activity(id TEXT PRIMARY KEY, last_used REAL NOT NULL);
        """
    )
    now = time.time()
    day, month, _, _ = periods(now)
    connection.executemany(
        "INSERT INTO reservations VALUES(?,?,?,?,?,0,?)",
        [
            ("settled", "client", now, 20, 5, 1),
            ("retained", "client", now, 100, 50, 0),
        ],
    )
    connection.executemany(
        "INSERT INTO usage_totals VALUES('client',?,?,?,?)",
        [("day", day, 120, 55), ("month", month, 120, 55)],
    )
    connection.commit()
    connection.close()
    accounting = Accounting(path)
    usage = accounting.key_usage(key(daily_token_budget=1000))
    assert usage["daily_confirmed_tokens"] == 25
    assert usage["daily_retained_tokens"] == 150
    assert usage["daily_tokens"] == 175


def test_discarded_sse_event_requires_fresh_usage_before_output_authority():
    observer = UsageObserver(sse=True)
    observer.feed(b'data: {"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n')
    observer.feed(b'data: {"content":"' + b"x" * 300000 + b'"}\n\n')
    observer.feed(b"data: [DONE]\n\n")
    observer.finish()
    assert observer.usage_reported
    assert observer.observation_gap
    assert not observer.output_authoritative
    assert "oversized_event_discarded" in observer.reasons

    fresh = UsageObserver(sse=True)
    fresh.feed(b'data: {"content":"' + b"x" * 300000 + b'"}\n\n')
    fresh.feed(b'data: {"usage":{"prompt_tokens":4,"completion_tokens":2}}\n\n')
    fresh.feed(b"data: [DONE]\n\n")
    fresh.finish()
    assert fresh.observation_gap is False
    assert fresh.output_authoritative


def test_malformed_captured_object_invalidates_structural_usage():
    observer = observe(
        b'{"usage":{"prompt_tokens":8,"completion_tokens":2},"metrics":{"x":garbage}}'
    )
    assert "malformed_json_object" in observer.reasons
    assert not observer.structural_complete
    assert not observer.usage_authoritative
    assert not observer.output_authoritative


def test_observed_output_without_usage_is_partial():
    observation = finalize_observation(
        input_total=None,
        output_total=None,
        observed_output=None,
        usage_reported=False,
        usage_authoritative=False,
        dispatched=True,
        output_event_count=1,
        first_output_ms=10,
        last_output_ms=20,
        latency_ms=30,
    )
    assert observation["usage_status"] == "partial"
    assert "output_event_observed_without_usage" in observation["observation_reasons"]
    assert observation["output_tokens"] is None


def test_benchmark_response_model_keeps_observation_fields():
    from backend.routes.benchmarks import BenchmarkResponse

    payload = {
        "id": "b",
        "created_at": 1.0,
        "model_id": "m",
        "proxy_name": "m",
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
        "input_tokens_total": 8,
        "input_tokens_cached": 7,
        "engine_metrics": {"predicted_per_second": 4},
        "engine_tokens_per_second": 4.0,
        "tokens_per_second_source": "proxy_decode_estimate",
        "decode_rate_quality": "conventional_n_minus_one",
    }
    dumped = BenchmarkResponse(**payload).model_dump()
    assert dumped["input_tokens_total"] == 8
    assert dumped["engine_metrics"]["predicted_per_second"] == 4
    assert dumped["decode_rate_quality"] == "conventional_n_minus_one"


def test_finalize_separates_decode_effective_and_accounting_values():
    observation = finalize_observation(
        input_total=8000,
        input_cached=7500,
        output_total=101,
        usage_reported=True,
        usage_authoritative=True,
        first_output_ms=2000,
        last_output_ms=4000,
        terminal_ms=5000,
        ttft_ms=2000,
        latency_ms=5000,
        output_event_count=10,
        sequence_scope="single",
    )
    assert observation["schema_version"] == SCHEMA_VERSION
    assert observation["input_tokens_uncached"] == 500
    assert observation["proxy_decode_tokens_per_second"] == pytest.approx(50.0)
    assert observation["request_effective_tokens_per_second"] == pytest.approx(20.2)
    assert observation["output_tokens_per_second"] == pytest.approx(50.0)

    delayed = finalize_observation(
        input_total=8000,
        input_cached=7500,
        output_total=101,
        usage_reported=True,
        usage_authoritative=True,
        first_output_ms=2000,
        last_output_ms=4000,
        terminal_ms=7000,
        ttft_ms=2000,
        latency_ms=7000,
        output_event_count=10,
    )
    assert delayed["proxy_decode_tokens_per_second"] == pytest.approx(50.0)
    assert delayed["request_effective_tokens_per_second"] < 20.2


def test_cache_subset_validation_and_coverage_rules():
    cached, uncached, reasons = normalize_cache(100, 150)
    assert cached is None and uncached is None and "cache_exceeds_total" in reasons
    cached, uncached, reasons = normalize_cache(100, None)
    assert cached is None and uncached is None and "cache_not_reported" in reasons
    cached, uncached, _ = normalize_cache(100, 40)
    assert (cached, uncached) == (40, 60)

    observation = finalize_observation(
        input_total=100,
        input_cached=150,
        output_total=2,
        usage_reported=True,
        usage_authoritative=True,
        first_output_ms=1,
        last_output_ms=11,
        latency_ms=12,
        output_event_count=2,
    )
    assert observation["input_tokens_total"] == 100
    assert observation["input_tokens_cached"] is None
    assert "cache_exceeds_total" in observation["observation_reasons"]


def test_rate_is_suppressed_when_interval_is_unmeasurable():
    observation = finalize_observation(
        input_total=3,
        output_total=10,
        usage_reported=True,
        usage_authoritative=True,
        first_output_ms=100,
        last_output_ms=100,
        latency_ms=200,
        output_event_count=4,
    )
    assert observation["proxy_decode_tokens_per_second"] is None
    assert observation["decode_rate_quality"] == "unmeasurable_interval"


def key(**policy):
    return {"id": "client", "name": "Test client", "scopes": ["inference"], **policy}


def test_settlement_reconciles_fields_independently(tmp_path):
    accounting = Accounting(str(tmp_path / "accounting.sqlite3"))
    account = key(daily_token_budget=1000)
    accounting.reserve("one", account, 100, 100)
    accounting.settle("one", 20, None, True)
    usage = accounting.key_usage(account)
    assert usage["daily_tokens"] == 120
    assert usage["daily_confirmed_tokens"] == 20
    assert usage["daily_retained_tokens"] == 100
    assert usage["daily_confirmed_input_tokens"] == 20
    assert usage["daily_retained_output_tokens"] == 100
    # A stale retry cannot replace the first authoritative input value; the
    # field that was still unknown may be upgraded by a deferred retry.
    accounting.settle("one", 999, 5, True)
    usage = accounting.key_usage(account)
    assert usage["daily_tokens"] == 25
    assert usage["daily_confirmed_input_tokens"] == 20
    assert usage["daily_confirmed_output_tokens"] == 5
    assert usage["daily_retained_output_tokens"] == 0


class _Chunks(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class _Registry:
    def __init__(self):
        self.observed = []

    async def acquire(self, model):
        return SimpleNamespace(
            upstream="http://engine",
            engine_id="llama_cpp",
            revision="build-1",
            process=SimpleNamespace(pid=1),
        )

    def release(self, generation):
        return None

    def record_request(self, generation, **kwargs):
        self.observed.append(kwargs)


@pytest.mark.asyncio
async def test_streaming_trace_uses_finalized_observation(tmp_path):
    accounting = Accounting(str(tmp_path / "accounting.sqlite3"))
    registry = _Registry()
    app = StudioDataPlane(
        registry,
        lambda name: ResolvedModel("demo") if name in ("demo", "alias") else None,
        accounting=accounting,
        identity_resolver=lambda headers: key(daily_token_budget=10000, max_output_tokens=32),
    )
    await app.client.aclose()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
        b'data: {"usage":{"prompt_tokens":10,"completion_tokens":2,'
        b'"prompt_tokens_details":{"cached_tokens":7}}}\n\n',
        b"data: [DONE]\n\n",
    ]
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_Chunks(chunks),
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.close()
    assert result.status_code == 200
    assert registry.observed
    observation = registry.observed[0]["observation"]
    assert observation["usage_status"] == "authoritative"
    assert observation["input_tokens_total"] == 10
    assert observation["input_tokens_cached"] == 7
    assert observation["input_tokens_uncached"] == 3
    assert observation["output_tokens"] == 2
    assert observation["ttft_ms"] is not None
    assert observation["output_interval_ms"] is not None
    assert observation["proxy_decode_tokens_per_second"] is not None
    row = accounting.query()["requests"][0]
    assert row["input_tokens_cached"] == 7
    assert row["usage_status"] == "authoritative"
    assert row["observation_reasons"] == []


@pytest.mark.asyncio
async def test_output_after_usage_keeps_the_output_reservation(tmp_path):
    accounting = Accounting(str(tmp_path / "accounting.sqlite3"))
    registry = _Registry()
    app = StudioDataPlane(
        registry,
        lambda name: ResolvedModel("demo") if name in ("demo", "alias") else None,
        accounting=accounting,
        identity_resolver=lambda headers: key(daily_token_budget=10000, max_output_tokens=32),
    )
    await app.client.aclose()
    chunks = [
        b'data: {"choices":[{"delta":{"content":"a"}}],'
        b'"usage":{"prompt_tokens":10,"completion_tokens":1}}\n\n',
        b'data: {"choices":[{"delta":{"content":"b"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_Chunks(chunks),
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.close()
    assert result.status_code == 200
    row = accounting.query()["requests"][0]
    assert row["input_tokens_total"] == 10
    assert row["observed_output_tokens"] == 1
    assert row["output_tokens"] is None
    assert row["usage_status"] == "partial"
    assert row["retained_charge_reason"].startswith("Unknown output")
    usage = accounting.key_usage(key())
    assert usage["daily_confirmed_input_tokens"] == 10
    assert usage["daily_retained_output_tokens"] == 32


@pytest.mark.asyncio
async def test_truncated_compressed_usage_does_not_settle_output(tmp_path):
    accounting = Accounting(str(tmp_path / "accounting.sqlite3"))
    registry = _Registry()
    app = StudioDataPlane(
        registry,
        lambda name: ResolvedModel("demo") if name in ("demo", "alias") else None,
        accounting=accounting,
        identity_resolver=lambda headers: key(daily_token_budget=10000, max_output_tokens=32),
    )
    await app.client.aclose()
    payload = gzip.compress(
        b'{"choices":[{"message":{"content":"x"}}],'
        b'"usage":{"prompt_tokens":3,"completion_tokens":4}}'
    )
    truncated = payload[:-8]
    assert truncated
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                headers={"content-type": "application/json", "content-encoding": "gzip"},
                stream=_Chunks([truncated]),
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        result = await client.post(
            "/v1/chat/completions",
            json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
        )
    await app.close()
    assert result.status_code == 200
    row = accounting.query()["requests"][0]
    assert row["transport_complete"] is False
    assert row["usage_authoritative"] is False
    assert row["output_authoritative"] is False
    assert row["output_tokens"] is None
    assert "output_blocked_by_incomplete_transport" in row["observation_reasons"]
    usage = accounting.key_usage(key())
    assert usage["daily_confirmed_output_tokens"] == 0
    assert usage["daily_retained_output_tokens"] == 32


@pytest.mark.asyncio
async def test_sglang_tokenize_adapter_and_revision_scoped_probe(tmp_path):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        return httpx.Response(404, json={"error": "not found"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = await count_input(
            client,
            "http://engine",
            "sglang",
            "/v1/chat/completions",
            {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            revision="rev-1",
        )
        assert first is None
        second = await count_input(
            client,
            "http://engine",
            "sglang",
            "/v1/chat/completions",
            {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            revision="rev-1",
        )
        assert second is None and len(calls) == 1
        third = await count_input(
            client,
            "http://engine",
            "sglang",
            "/v1/chat/completions",
            {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            revision="rev-2",
        )
        assert third is None and len(calls) == 2


@pytest.mark.asyncio
async def test_sglang_chat_and_prompt_counts_match_the_endpoint_contract():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if "messages" in payload:
            return httpx.Response(200, json={"count": 11, "tokens": list(range(11))})
        return httpx.Response(200, json={"count": 4, "tokens": [1, 2, 3, 4]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        chat = await count_input(
            client,
            "http://engine",
            "sglang",
            "/v1/chat/completions",
            {"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            revision="rev-sglang-success",
        )
        prompt = await count_input(
            client,
            "http://engine",
            "sglang",
            "/v1/completions",
            {"model": "demo", "prompt": "hello"},
            revision="rev-sglang-success",
        )
    assert chat.tokens == 11
    assert prompt.tokens == 4
    assert seen[0]["messages"][0]["content"] == "hi"
    assert seen[1] == {"prompt": "hello", "add_special_tokens": True, "model": "demo"}
