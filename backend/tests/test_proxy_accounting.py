"""Quota admission, persistence, streaming accounting and request lifecycle tests."""

import asyncio
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from backend.proxy.studio.accounting import Accounting, AdmissionError, periods, validate_policy
from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane
from backend.proxy.studio.usage import UsageObserver
from backend.tests.test_api_routes import _install_temp_store


@pytest.fixture
def accounting(tmp_path):
    return Accounting(str(tmp_path / "accounting.sqlite3"))


def key(**policy):
    return {"id": "client", "name": "Test client", "scopes": ["inference", "models:read"], **policy}


@pytest.mark.parametrize(
    "field",
    [
        "input_tokens_per_minute",
        "output_tokens_per_minute",
        "daily_token_budget",
        "monthly_token_budget",
        "max_output_tokens",
        "max_concurrent_requests",
        "rate_limit_per_minute",
    ],
)
@pytest.mark.parametrize("value", [0, -1, True, "5", 1.5])
def test_policy_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        validate_policy({field: value})


@pytest.mark.parametrize(
    "policy,input_tokens,output_tokens",
    [
        ({"rate_limit_per_minute": 1}, 2, 3),
        ({"input_tokens_per_minute": 3}, 2, 0),
        ({"output_tokens_per_minute": 5}, 0, 3),
        ({"daily_token_budget": 8}, 2, 3),
        ({"monthly_token_budget": 8}, 2, 3),
        ({"max_concurrent_requests": 1}, 0, 0),
    ],
)
def test_limits_include_outstanding_reservations(accounting, policy, input_tokens, output_tokens):
    account = key(**policy)
    accounting.reserve("one", account, input_tokens, output_tokens)
    with pytest.raises(AdmissionError) as error:
        accounting.reserve("two", account, input_tokens, output_tokens)
    assert error.value.status == 429
    assert error.value.retry_after > 0


def test_atomic_reservation_under_concurrent_admission(accounting):
    gate = threading.Barrier(8)

    def attempt(index):
        gate.wait()
        try:
            accounting.reserve(str(index), key(daily_token_budget=100), 10, 40)
            return True
        except AdmissionError:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(attempt, range(8))) == 2


def test_reconcile_known_unknown_and_undispatched(accounting):
    account = key(daily_token_budget=1000, max_concurrent_requests=1)
    accounting.reserve("one", account, 100, 100)
    accounting.settle("one", 10, 20, True)
    assert accounting.key_usage(account)["daily_tokens"] == 30
    accounting.reserve("two", account, 100, 100)
    accounting.settle("two", None, None, True)
    assert accounting.key_usage(account)["daily_tokens"] == 230
    assert accounting.key_usage(account)["estimated_requests"] == 1
    accounting.reserve("three", account, 100, 100)
    accounting.settle("three", None, None, False)
    assert accounting.key_usage(account)["daily_tokens"] == 230
    assert accounting.key_usage(account)["concurrent_requests"] == 0


def test_budget_survives_restart_and_crash_releases_concurrency(accounting, monkeypatch):
    account = key(daily_token_budget=100, max_concurrent_requests=1)
    accounting.reserve("crashed", account, 20, 40)
    monkeypatch.setattr("backend.proxy.studio.accounting.process_stamp", lambda pid: "new-process")
    restarted = Accounting(accounting.path)
    assert restarted.query()["requests"][0]["error"] == "proxy_process_interrupted"
    assert restarted.key_usage(account)["daily_tokens"] == 60
    assert restarted.key_usage(account)["concurrent_requests"] == 0
    with pytest.raises(AdmissionError):
        restarted.reserve("new", account, 20, 40)


def test_utc_budget_windows_and_rolling_minute(accounting, monkeypatch):
    from datetime import datetime, timezone

    now = datetime(2026, 12, 31, 23, 59, 30, tzinfo=timezone.utc).timestamp()
    monkeypatch.setattr("backend.proxy.studio.accounting.time.time", lambda: now)
    account = key(daily_token_budget=100, monthly_token_budget=100, rate_limit_per_minute=1)
    accounting.reserve("old", account, 10, 20)
    accounting.settle("old", 10, 20, True)
    now += 40
    usage = accounting.key_usage(account)
    assert usage["daily_tokens"] == usage["monthly_tokens"] == 0
    assert usage["requests_per_minute"] == 1
    with pytest.raises(AdmissionError):
        accounting.reserve("new", account, 1, 1)
    now += 21
    accounting.reserve("new", account, 1, 1)
    assert periods(now)[2] - periods(now)[0] == 86400


def trace(index, **overrides):
    return {
        "id": str(index),
        "started_at": 1000 + (index if isinstance(index, int) else 0),
        "state": "completed",
        "status": 200,
        "model": "demo",
        "key_id": "client",
        "latency_ms": index,
        **overrides,
    }


def test_history_paging_filtering_retention_and_live(accounting, monkeypatch):
    monkeypatch.setattr("backend.proxy.studio.accounting.time.time", lambda: 1300)
    for index in range(220):
        accounting.finish(trace(index, state="failed" if index % 2 else "completed"))
    accounting.flush(max_rows=200)
    assert accounting.query()["total"] == 200
    assert len(accounting.query()["requests"]) == 100
    assert accounting.query(offset=100)["requests"][0]["id"] == "119"
    assert accounting.query(state="failed")["total"] == 100
    assert accounting.query(model="nope")["total"] == 0
    assert accounting.query(search="demo")["total"] == 200
    assert accounting.query(search="' OR 1=1 --")["total"] == 0
    accounting.begin(trace("live", started_at=1400, state="queued"))
    assert accounting.query(state="queued")["requests"][0]["id"] == "live"
    assert Accounting(accounting.path).query()["total"] == 200
    monkeypatch.setattr("backend.proxy.studio.accounting.time.time", lambda: 1000000)
    accounting.flush(retention_days=1)
    assert accounting.query()["total"] == 1  # live metadata stays available


def test_stats_reset_preserves_history_and_budgets(accounting):
    import time

    account = key(daily_token_budget=100)
    accounting.reserve("q", account, 10, 20)
    accounting.settle("q", 5, 10, True)
    accounting.finish(trace(1, started_at=time.time(), latency_ms=10))
    accounting.flush()
    assert accounting.summary()["requests"] == 1
    accounting.reset_stats()
    assert accounting.summary()["requests"] == 0
    assert accounting.query()["total"] == 1
    assert accounting.key_usage(account)["daily_tokens"] == 15


@pytest.mark.parametrize("sse", [True, False])
def test_incremental_usage_handles_chunk_boundaries_and_large_output(sse):
    observer = UsageObserver(sse)
    if sse:
        data = (
            b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
            + b'data: {"padding":"'
            + b"x" * 2100000
            + b'"}\n\n'
            + b'data: {"usage":{"prompt_tokens":12,"completion_tokens":34}}\n\ndata: [DONE]\n\n'
        )
    else:
        data = (
            b'{"content":"'
            + b"x" * 2100000
            + b'", "usage":{"prompt_tokens":12,"completion_tokens":34}}'
        )
    for offset in range(0, len(data), 137):
        observer.feed(data[offset : offset + 137])
    observer.finish()
    assert (observer.input_tokens, observer.output_tokens) == (12, 34)
    assert len(observer._line) < 262145


def test_usage_distinguishes_unknown_from_zero_and_ignores_content_strings():
    observer = UsageObserver()
    observer.feed(
        json.dumps({"content": '{"usage":{"prompt_tokens":99,"completion_tokens":99}}'}).encode()
    )
    assert observer.input_tokens is None
    observer.feed(b' {"usage":{"prompt_tokens":0,"completion_tokens":0}}')
    assert observer.input_tokens == observer.output_tokens == 0


class Registry:
    def __init__(self, delay=0):
        self.released = 0
        self.delay = delay

    async def acquire(self, model):
        await asyncio.sleep(self.delay)
        return SimpleNamespace(
            upstream="http://engine", engine_id="fake", process=SimpleNamespace(pid=1)
        )

    def release(self, generation):
        self.released += 1


async def plane(accounting, identity=None, **options):
    registry = Registry(options.pop("load_delay", 0))
    app = StudioDataPlane(
        registry,
        lambda name: ResolvedModel("demo") if name in ("demo", "alias") else None,
        accounting=accounting,
        identity_resolver=lambda headers: identity,
        **options,
    )
    await app.client.aclose()
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                headers={"content-type": "application/json"},
                stream=httpx.ByteStream(b'{"usage":{"prompt_tokens":3,"completion_tokens":4}}'),
            )
        )
    )
    return app, registry


@pytest.mark.asyncio
async def test_trace_attribution_timing_limits_and_rejections(accounting):
    app, registry = await plane(
        accounting, key(daily_token_budget=1000, max_output_tokens=10), load_delay=0.01
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        ok = await client.post(
            "/v1/chat/completions",
            json={"model": "alias", "messages": [{"role": "user", "content": "secret prompt"}]},
        )
        bad = await client.post("/v1/chat/completions", json={"model": "missing"})
        cap = await client.post("/v1/chat/completions", json={"model": "demo", "max_tokens": 11})
    await app.close()
    assert ok.status_code == 200 and ok.headers["x-request-id"]
    assert int(ok.headers["x-ratelimit-remaining-tokens-day"]) < 1000
    assert bad.status_code == 404 and cap.status_code == 400
    rows = accounting.query()["requests"]
    assert len(rows) == 3
    successful = next(row for row in rows if row["state"] == "completed")
    assert successful["key_id"] == "client" and successful["requested_model"] == "alias"
    assert successful["model"] == "demo" and successful["loading_ms"] >= 8
    assert successful["latency_ms"] >= successful["loading_ms"]
    assert successful["usage_source"] == "engine" and successful["output_tokens"] == 4
    assert "secret prompt" not in json.dumps(rows)
    assert accounting.key_usage(key())["daily_tokens"] == 7
    assert registry.released == 1


@pytest.mark.asyncio
async def test_unauthorized_forbidden_and_quota_denials_are_tracked(accounting):
    app, _ = await plane(accounting)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={"model": "demo"})
        ).status_code == 401
    await app.close()
    app, _ = await plane(accounting, key(models=["other"]))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={"model": "demo"})
        ).status_code == 403
    await app.close()
    app, _ = await plane(accounting, key(daily_token_budget=1))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        limited = await client.post("/v1/chat/completions", json={"model": "demo"})
        assert limited.status_code == 429 and int(limited.headers["retry-after"]) > 0
    await app.close()
    assert {row["status"] for row in accounting.query()["requests"]} == {401, 403, 429}


@pytest.mark.asyncio
async def test_total_deadline_cleans_up_and_refunds_undispatched_work(accounting):
    app, _ = await plane(
        accounting,
        key(daily_token_budget=10000),
        request_timeout=0.02,
        load_delay=1,
        global_concurrency=1,
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/chat/completions", json={"model": "demo"})
    await app.close()
    assert response.status_code == 504
    assert app._global_slots._value == 1
    assert accounting.key_usage(key())["daily_tokens"] == 0
    assert accounting.query()["requests"][0]["error"] == "request_deadline_exceeded"


@pytest.mark.asyncio
async def test_disconnect_cancels_loading_and_releases_reservation(accounting):
    app, _ = await plane(
        accounting, key(daily_token_budget=10000), load_delay=1, global_concurrency=1
    )
    disconnected = asyncio.Event()
    calls = 0

    async def receive():
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"type": "http.request", "body": b'{"model":"demo"}', "more_body": False}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        pass

    task = asyncio.create_task(
        app(
            {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
            receive,
            send,
        )
    )
    await asyncio.sleep(0.02)
    disconnected.set()
    await asyncio.wait_for(task, 1)
    await app.close()
    assert accounting.query()["requests"][0]["state"] == "cancelled"
    assert accounting.key_usage(key())["concurrent_requests"] == 0
    assert app._global_slots._value == 1


@pytest.mark.asyncio
async def test_bounded_queue_and_queue_timeout(accounting):
    app, _ = await plane(
        accounting, key(), global_concurrency=1, max_queue_size=1, queue_timeout=0.02
    )
    await app._global_slots.acquire()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", json={"model": "demo"}))
        await asyncio.sleep(0.005)
        second = await client.post("/v1/chat/completions", json={"model": "demo"})
        assert second.status_code == 429 and second.json()["code"] == "queue_full"
        assert (await first).status_code == 429
    app._global_slots.release()
    await app.close()
    assert app._waiting == 0
    assert accounting.key_usage(key())["concurrent_requests"] == 0


@pytest.mark.asyncio
async def test_output_caps_default_allowance_and_multimodal_refusal(accounting):
    app, _ = await plane(accounting, key(max_output_tokens=8, daily_token_budget=10000))
    body, _, reserved = app._budget_body(
        b'{"model":"demo","stream":true}',
        key(max_output_tokens=8, daily_token_budget=10000),
        "/chat/completions",
    )
    payload = json.loads(body)
    assert payload["max_tokens"] == 8 and payload["stream_options"]["include_usage"] is True
    assert reserved == 8
    with pytest.raises(AdmissionError):
        app._budget_body(
            b'{"messages":[{"image_url":"x"}]}', key(daily_token_budget=10000), "/chat/completions"
        )
    await app.close()


def test_policy_edit_preserves_secret_and_usage(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    created = client.post(
        "/api/inference/keys", json={"name": "client", "daily_token_budget": 100}
    ).json()
    response = client.patch(
        f"/api/inference/keys/{created['id']}",
        json={"name": "updated", "daily_token_budget": 200, "max_concurrent_requests": 2},
    )
    assert response.status_code == 200 and "key" not in response.json()
    from backend.inference_auth import verify_token

    assert verify_token(created["key"])
    assert client.get("/api/inference/keys").json()["keys"][0]["daily_token_budget"] == 200
    assert (
        client.patch(
            f"/api/inference/keys/{created['id']}", json={"max_output_tokens": -1}
        ).status_code
        == 400
    )
    assert client.patch("/api/inference/keys/missing", json={"name": "x"}).status_code == 404


@pytest.mark.asyncio
async def test_streaming_ttft_usage_and_partial_failure(accounting):
    class Stream(httpx.AsyncByteStream):
        def __init__(self, fail=False):
            self.closed = False
            self.fail = fail

        async def __aiter__(self):
            await asyncio.sleep(0.005)
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            if self.fail:
                raise httpx.ReadTimeout("stalled")
            yield b'data: {"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\ndata: [DONE]\n\n'

        async def aclose(self):
            self.closed = True

    for fail in (False, True):
        app, registry = await plane(accounting, key(daily_token_budget=10000, max_output_tokens=10))
        await app.client.aclose()
        stream = Stream(fail)
        app.client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda req: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=stream
                )
            )
        )
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            await client.post("/v1/chat/completions", json={"model": "demo", "stream": True})
        await app.close()
        row = accounting.query()["requests"][0]
        assert row["ttft_ms"] >= 4
        assert row["generation_ms"] is not None
        assert row["first_byte_ms"] <= row["ttft_ms"]
        assert stream.closed and registry.released == 1
        if fail:
            assert row["state"] == "failed" and row["status"] == 504
            assert row["usage_source"] == "reservation" and row["output_tokens"] is None
        else:
            assert row["state"] == "completed" and row["usage_source"] == "engine"
            assert row["output_tokens"] == 2


@pytest.mark.asyncio
async def test_request_deadline_includes_streaming_and_releases_upstream(accounting):
    closed = asyncio.Event()

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            await asyncio.sleep(2)
            yield b"data: [DONE]\n\n"

        async def aclose(self):
            closed.set()

    app, registry = await plane(accounting, key(daily_token_budget=10000), request_timeout=0.03)
    await app.client.aclose()
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=SlowStream()
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        response = await client.post("/v1/chat/completions", json={"model": "demo"})
    await app.close()
    assert response.status_code == 200  # Headers were already delivered.
    row = accounting.query()["requests"][0]
    assert row["status"] == 504 and row["state"] == "failed"
    assert closed.is_set() and registry.released == 1
    assert accounting.key_usage(key())["concurrent_requests"] == 0


@pytest.mark.asyncio
async def test_cancellation_during_reservation_does_not_leak_charge(accounting, monkeypatch):
    import time

    original = accounting.reserve

    def slow_reserve(*args):
        time.sleep(0.03)
        return original(*args)

    monkeypatch.setattr(accounting, "reserve", slow_reserve)
    app, _ = await plane(accounting, key(daily_token_budget=10000), request_timeout=0.005)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={"model": "demo"})
        ).status_code == 504
    await app.close()
    usage = accounting.key_usage(key())
    assert usage["concurrent_requests"] == usage["daily_tokens"] == 0


def test_requests_api_filters_and_validates_pagination(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    from backend.proxy.studio.accounting import get_accounting
    import time

    accounting = get_accounting()
    for index in range(120):
        accounting.finish(
            trace(
                index, started_at=time.time() + index, state="failed" if index % 2 else "completed"
            )
        )
    response = client.get("/api/proxy/requests?state=failed&limit=10&offset=10")
    assert response.status_code == 200
    assert response.json()["total"] == 60
    assert len(response.json()["requests"]) == 10
    assert all(row["state"] == "failed" for row in response.json()["requests"])
    assert client.get("/api/proxy/requests?limit=101").status_code == 422
    assert client.get("/api/proxy/requests?offset=-1").status_code == 422


def test_proxy_performance_settings_validation(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    for body in (
        {"max_queue_size": 0},
        {"pool_timeout_seconds": 0},
        {"history_retention_days": 0},
        {"history_max_rows": 99},
        {"max_connections": 2, "max_keepalive_connections": 3},
    ):
        assert client.put(
            "/api/proxy/settings", json={"backend": "studio", **body}
        ).status_code in (400, 422)
    body = {
        "backend": "studio",
        "max_queue_size": 7,
        "max_connections": 10,
        "max_keepalive_connections": 5,
        "history_retention_days": 3,
        "stream_idle_timeout_seconds": 2,
        "request_timeout_seconds": 30,
    }
    assert client.put("/api/proxy/settings", json=body).status_code == 200
    saved = client.get("/api/proxy").json()["settings"]
    assert all(saved[name] == value for name, value in body.items())


def test_second_worker_does_not_release_live_reservations(accounting):
    account = key(max_concurrent_requests=1)
    accounting.reserve("live-worker", account, 1, 1)
    other = Accounting(accounting.path)
    with pytest.raises(AdmissionError):
        other.reserve("other-worker", account, 1, 1)
    accounting.settle("live-worker", 1, 1, True)
    other.reserve("other-worker", account, 1, 1)


def test_multiline_sse_usage_and_invalid_usage_never_crashes():
    observer = UsageObserver(True)
    payload = b'event: completion\r\ndata: {"usage":\r\ndata: {"input_tokens":9,"output_tokens":2}}\r\n\r\n'
    for byte in payload:
        observer.feed(bytes([byte]))
    observer.finish()
    assert (observer.input_tokens, observer.output_tokens) == (9, 2)
    malformed = UsageObserver(True)
    malformed.feed(b'data: {"choices":123,"usage":{"input_tokens":-5,"output_tokens":true}}\n\n')
    assert malformed.input_tokens is malformed.output_tokens is None


def test_deferred_reconciliation_retries_idempotently(accounting):
    account = key(daily_token_budget=1000)
    accounting.reserve("retry", account, 100, 100)
    accounting.defer_settlement("retry", 5, 10, True)
    accounting.flush()
    assert accounting.key_usage(account)["daily_tokens"] == 15
    accounting.settle("retry", 5, 10, True)
    assert accounting.key_usage(account)["daily_tokens"] == 15


@pytest.mark.asyncio
async def test_deadline_wins_over_a_delayed_denied_reservation(accounting, monkeypatch):
    import time

    original = accounting.reserve

    def delayed(*args):
        time.sleep(0.02)
        return original(*args)

    monkeypatch.setattr(accounting, "reserve", delayed)
    app, _ = await plane(accounting, key(daily_token_budget=1), request_timeout=0.005)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={"model": "demo"})
        ).status_code == 504
    await app.close()


@pytest.mark.asyncio
async def test_token_policies_do_not_accept_unaccountable_state_or_unbounded_allowances(accounting):
    app = StudioDataPlane(Registry(), lambda _: None, accounting=accounting)
    for payload in ({"previous_response_id": "prior"}, {"max_tokens": 10**30}):
        with pytest.raises(AdmissionError):
            app._budget_body(
                json.dumps(payload).encode(), key(daily_token_budget=10000), "/responses"
            )
    await app.close()


def test_alias_cursor_advances_on_alias_not_resolved_target(monkeypatch):
    from backend.proxy.studio.manager import StudioProxyManager

    manager = object.__new__(StudioProxyManager)
    manager._route_cursor = {}
    manager._route_lock = threading.Lock()
    manager.registry = SimpleNamespace(snapshot=lambda: [])
    store = SimpleNamespace(
        get_settings=lambda: {"proxy": {"routing_aliases": {"fast": ["a", "b"]}}},
        list_models=lambda: [{"id": "a"}, {"id": "b"}],
    )
    monkeypatch.setattr("backend.proxy.studio.manager.get_store", lambda: store)
    monkeypatch.setattr(
        "backend.proxy.studio.manager.compile_model_runtime",
        lambda model: SimpleNamespace(
            proxy=SimpleNamespace(
                model_id=model["id"],
                catalog_id=model["id"],
                engine_id="llama_cpp",
                aliases=[],
                use_model_name=None,
                filters={},
            )
        ),
    )
    assert [manager._resolve("fast").model_id for _ in range(4)] == ["a", "b", "a", "b"]


@pytest.mark.asyncio
async def test_non_admin_api_passthrough_is_tracked_and_rate_limited(accounting):
    app, _ = await plane(accounting, key(rate_limit_per_minute=1))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (await client.post("/api/generate", json={"model": "demo"})).status_code == 200
        assert (await client.post("/api/generate", json={"model": "demo"})).status_code == 429
    await app.close()
    assert accounting.query()["total"] == 2


@pytest.mark.asyncio
async def test_settlement_failure_does_not_break_response_and_retries(accounting, monkeypatch):
    original = accounting.settle
    attempts = 0

    def transient(*args):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("temporary disk failure")
        return original(*args)

    monkeypatch.setattr(accounting, "settle", transient)
    app, _ = await plane(accounting, key(daily_token_budget=10000))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={"model": "demo"})
        ).status_code == 200
    await app.close()
    assert accounting.key_usage(key())["daily_tokens"] == 7
    assert accounting.key_usage(key())["concurrent_requests"] == 0
    assert (
        accounting.query()["requests"][0]["accounting_note"] == "Usage reconciliation was deferred"
    )


def test_live_wait_time_and_stats_window_persist(accounting, monkeypatch):
    monkeypatch.setattr("backend.proxy.studio.accounting.time.time", lambda: 2000)
    accounting.begin(trace("waiting", state="queued", phase_started_at=1999))
    row = accounting.query()["requests"][0]
    assert row["queue_ms"] == 1000
    assert "phase_started_at" not in row
    accounting.reset_stats()
    assert Accounting(accounting.path).summary()["since"] == 2000


def test_history_buffer_is_bounded_when_storage_cannot_keep_up(accounting):
    import time

    for index in range(1005):
        accounting.finish(trace(index, started_at=time.time()))
    assert len(accounting._pending) == 1000
    assert accounting.summary()["dropped_history"] == 5
    accounting.flush(max_rows=2000)
    assert accounting.query()["total"] == 1000


@pytest.mark.asyncio
async def test_unlimited_keys_batch_usage_without_a_reservation_write(accounting, monkeypatch):
    def must_not_reserve(*args):
        raise AssertionError("unlimited traffic must not block on quota writes")

    monkeypatch.setattr(accounting, "reserve", must_not_reserve)
    app, _ = await plane(accounting, key())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        assert (
            await client.post("/v1/chat/completions", json={"model": "demo"})
        ).status_code == 200
    await app.close()
    row = accounting.query()["requests"][0]
    assert row["accounting_mode"] == "batched_usage"
    assert accounting.key_usage(key())["daily_tokens"] == 7
    accounting.finish(row)  # A retried batch cannot double-charge usage.
    accounting.flush()
    assert accounting.key_usage(key())["daily_tokens"] == 7


@pytest.mark.asyncio
async def test_compressed_usage_is_observed_without_changing_forwarded_bytes(accounting):
    import gzip

    raw = b'{"usage":{"prompt_tokens":3,"completion_tokens":2}}'
    encoded = gzip.compress(raw, mtime=0)
    app, _ = await plane(accounting, key(daily_token_budget=10000))
    await app.client.aclose()
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                headers={"content-type": "application/json", "content-encoding": "gzip"},
                stream=httpx.ByteStream(encoded),
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        async with client.stream("POST", "/v1/chat/completions", json={"model": "demo"}) as result:
            observed = b"".join([chunk async for chunk in result.aiter_raw()])
    await app.close()
    assert observed == encoded
    assert accounting.key_usage(key())["daily_tokens"] == 5


@pytest.mark.asyncio
async def test_nonterminal_usage_is_not_settled_as_final(accounting):
    app, _ = await plane(accounting, key(daily_token_budget=10000, max_output_tokens=10))
    await app.client.aclose()
    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda req: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=httpx.ByteStream(
                    b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
                ),
            )
        )
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        await client.post("/v1/chat/completions", json={"model": "demo"})
    await app.close()
    row = accounting.query()["requests"][0]
    assert row["observed_output_tokens"] == 2
    assert row["output_tokens"] is None and row["usage_source"] == "reservation"
    assert (
        accounting.key_usage(key())["daily_tokens"]
        == row["input_reserved"] + row["output_reserved"]
    )


def test_nested_output_usage_is_not_engine_accounting():
    observer = UsageObserver()
    observer.feed(b'{"output":{"usage":{"prompt_tokens":999,"completion_tokens":999}}}')
    assert observer.input_tokens is None and observer.output_tokens is None


def test_summary_separates_client_responses_from_server_failures(accounting):
    import time

    for index, status in enumerate([200, 401, 404, 499, 500, 503]):
        accounting.finish(trace(str(index), started_at=time.time(), status=status))
    accounting.flush()
    summary = accounting.summary()
    assert summary["errors"] == 5
    assert summary["client_errors"] == 3
    assert summary["server_errors"] == 2
    assert accounting.query(outcome="4xx")["total"] == 3
    assert accounting.query(outcome="5xx")["total"] == 2
    assert accounting.query(outcome="errors")["total"] == 5
    accounting.begin(trace("live-failure", started_at=time.time(), status=502))
    assert accounting.query(outcome="5xx")["total"] == 3
    accounting.reset_stats()
    assert accounting.summary()["client_errors"] == 0
    assert accounting.summary()["server_errors"] == 0


def test_generation_http_outcome_counters_reset_independently():
    from types import SimpleNamespace
    from backend.proxy.studio.registry import Generation, ProcessRegistry

    registry = ProcessRegistry(None)
    generation = Generation("demo", "revision", 1234, SimpleNamespace(pid=42))
    registry._generations["demo"] = generation
    for status in [200, 401, 404, 503]:
        registry.record_request(
            generation, status=status, latency_ms=1, input_bytes=0, output_bytes=0
        )
    assert generation.request_count == 4
    assert generation.client_error_count == 2
    assert generation.server_error_count == 1
    registry.reset_stats()
    assert generation.error_count == 0
    assert generation.client_error_count == 0
    assert generation.server_error_count == 0
