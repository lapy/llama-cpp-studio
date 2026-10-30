"""Process-owned lifecycle and browser principal regression coverage.

Contracts are based on llama-swap v265 EnsureReady and Unix process supervision.
"""

import asyncio
import os
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest

from backend.inference_auth import create_key, request_identity, verify_token
from backend.proxy.launch_spec import EnvSpec, LaunchSpec
from backend.proxy.studio.accounting import Accounting
from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane
from backend.proxy.studio.registry import ProcessRegistry, RegistryError
from backend.tests.test_api_routes import _install_temp_store


def spec(model, code="import time; time.sleep(60)"):
    return (
        LaunchSpec(
            model_id=model,
            engine_id="fake",
            engine_install_id="fake",
            executable=sys.executable,
            argv=["-u", "-c", code],
            cwd=None,
            env=EnvSpec(),
        ),
        "a" * 64,
        None,
    )


@pytest.mark.asyncio
async def test_one_hundred_waiters_share_start_and_one_disconnect_does_not_kill_it(monkeypatch):
    calls = 0
    started, ready = asyncio.Event(), asyncio.Event()

    async def factory(model):
        nonlocal calls
        calls += 1
        return spec(model)

    async def wait(generation):
        started.set()
        await ready.wait()

    registry = ProcessRegistry(factory)
    monkeypatch.setattr(registry, "_wait_ready", wait)
    tasks = [asyncio.create_task(registry.acquire("demo")) for _ in range(100)]
    await started.wait()
    await asyncio.sleep(0)
    tasks[0].cancel()
    with pytest.raises(asyncio.CancelledError):
        await tasks[0]
    assert registry._starting["demo"].process.poll() is None
    ready.set()
    try:
        generations = await asyncio.gather(*tasks[1:])
        assert len({item.process.pid for item in generations}) == calls == 1
        assert generations[0].inflight == 99
        for generation in generations:
            registry.release(generation)
        assert generations[0].inflight == 0
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_last_waiter_cancellation_kills_start_and_next_request_can_retry(monkeypatch):
    started = asyncio.Event()

    async def wait(generation):
        started.set()
        await asyncio.sleep(30)

    registry = ProcessRegistry(lambda model: asyncio.sleep(0, result=spec(model)))
    monkeypatch.setattr(registry, "_wait_ready", wait)
    task = asyncio.create_task(registry.acquire("demo"))
    await started.wait()
    process = registry._starting["demo"].process
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert process.poll() is not None
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0))
    try:
        generation = await registry.acquire("demo")
        assert generation.process.pid != process.pid and generation.ready
        registry.release(generation)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_unexpected_exit_is_failed_and_recovers_on_next_start(monkeypatch):
    registry = ProcessRegistry(lambda model: asyncio.sleep(0, result=spec(model)))
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0))
    generation = await registry.acquire("demo")
    registry.release(generation)
    os.kill(generation.process.pid, signal.SIGKILL)
    try:
        for _ in range(100):
            if registry.snapshot()[0]["state"] == "failed" and generation._terminated:
                break
            await asyncio.sleep(0.01)
        row = registry.snapshot()[0]
        assert row["state"] == "failed" and not row["ready"]
        assert row["exit_code"] == -signal.SIGKILL
        assert registry.generation("demo") is generation  # failed logs remain inspectable
        replacement = await registry.acquire("demo")
        assert replacement.process.pid != generation.process.pid
        assert registry.snapshot()[0]["state"] == "ready"
        registry.release(replacement)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_quiet_logs_do_not_occupy_default_executor(monkeypatch):
    # A two-worker default pool used to be exhausted by one engine's two pipes.
    loop = asyncio.get_running_loop()
    pool = ThreadPoolExecutor(max_workers=2)
    loop.set_default_executor(pool)
    registry = ProcessRegistry(lambda model: asyncio.sleep(0, result=spec(model)))
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0))
    try:
        for index in range(8):
            generation = await registry.acquire(f"model-{index}")
            registry.release(generation)
        await asyncio.sleep(0.05)
        assert await asyncio.wait_for(asyncio.to_thread(lambda: 42), 0.5) == 42
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_acquire_during_stop_waits_for_drain(monkeypatch):
    registry = ProcessRegistry(lambda model: asyncio.sleep(0, result=spec(model)))
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0))
    first = await registry.acquire("demo")
    stop = asyncio.create_task(registry.stop("demo"))
    await asyncio.sleep(0.02)
    next_request = asyncio.create_task(registry.acquire("demo"))
    await asyncio.sleep(0.02)
    assert not next_request.done() and first.process.poll() is None
    registry.release(first)
    try:
        assert await stop
        replacement = await next_request
        assert replacement.process.pid != first.process.pid
        registry.release(replacement)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_log_line_buffers_are_bounded(monkeypatch):
    registry = ProcessRegistry(
        lambda model: asyncio.sleep(
            0, result=spec(model, "import time; print('x'*200000, flush=True); time.sleep(60)")
        )
    )
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0.1))
    try:
        generation = await registry.acquire("demo")
        assert generation.log_truncated_bytes > 100000
        assert max(map(len, generation.logs.values())) < 8300
        registry.release(generation)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_browser_credentials_have_model_scope_quota_and_key_attribution(
    monkeypatch, tmp_path
):
    _install_temp_store(monkeypatch, tmp_path)
    record, token = create_key(
        "Browser", models=["allowed"], rate_limit_per_minute=1, daily_token_budget=1000
    )
    accounting = Accounting(str(tmp_path / "browser.sqlite3"))
    generation = SimpleNamespace(
        upstream="http://engine", engine_id="fake", process=SimpleNamespace(pid=1)
    )
    registry = SimpleNamespace(
        acquire=lambda model: asyncio.sleep(0, result=generation), release=lambda generation: None
    )
    app = StudioDataPlane(
        registry,
        lambda name: ResolvedModel(name),
        accounting=accounting,
        identity_resolver=request_identity,
        verify_token=verify_token,
    )
    await app.client.aclose()
    seen = []

    def respond(request):
        seen.append(request)
        return httpx.Response(
            200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(b"{}")
        )

    app.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://proxy"
    ) as client:
        blocked = await client.get(f"/ui/forbidden/?key={token}")
        assert blocked.status_code == 403 and "set-cookie" not in blocked.headers
        entry = await client.get(f"/ui/allowed/?key={token}")
        assert entry.status_code == 302
        assert "Path=/ui/allowed/" in entry.headers["set-cookie"]
        cookie = entry.headers["set-cookie"].split(";")[0]
        assert (await client.get("/ui/forbidden/", headers={"cookie": cookie})).status_code == 403
        assert (await client.get("/ui/allowed/", headers={"cookie": cookie})).status_code == 200
        assert (
            await client.get("/ui/allowed/props", headers={"cookie": cookie})
        ).status_code == 429
        assert (
            await client.get(
                "/ui/allowed/", headers={"cookie": cookie, "authorization": "Bearer invalid"}
            )
        ).status_code == 401
        assert (
            await client.post(
                "/ui/allowed/v1/chat/completions", headers={"cookie": cookie}, json={}
            )
        ).status_code == 401
    await app.close()
    assert len(seen) == 1 and "studio_ui_grant" not in seen[0].headers.get("cookie", "")
    successful = next(row for row in accounting.query()["requests"] if row["status"] == 200)
    assert successful["key_id"] == record["id"]


@pytest.mark.asyncio
async def test_group_commit_preserves_atomic_limits_and_rollback(tmp_path):
    accounting = Accounting(str(tmp_path / "batch.sqlite3"))
    key = {"id": "batch", "daily_token_budget": 100}
    results = await asyncio.gather(
        *(accounting.reserve_async(str(index), key, 10, 10) for index in range(20)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, dict) for result in results) == 5
    assert accounting.key_usage(key)["daily_tokens"] == 100
    assert accounting._db_depth == 0


@pytest.mark.asyncio
async def test_fast_crash_retains_failure_and_last_log(monkeypatch):
    registry = ProcessRegistry(
        lambda model: asyncio.sleep(
            0, result=spec(model, "import sys; print('fatal-fixture', flush=True); sys.exit(17)")
        )
    )
    monkeypatch.setattr(registry, "_wait_ready", lambda generation: asyncio.sleep(0.15))
    try:
        with pytest.raises(RegistryError):
            await registry.acquire("demo")
        row = registry.snapshot()[0]
        assert row["state"] == "failed" and row["exit_code"] == 17
        assert any("fatal-fixture" in line for line in registry.generation("demo").logs.values())
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_retry_cohort_during_cancel_cleanup_still_shares_one_start(monkeypatch):
    calls = 0
    first_started = asyncio.Event()
    new_ready = asyncio.Event()

    async def factory(model):
        nonlocal calls
        calls += 1
        return spec(model)

    async def wait_ready(generation):
        if calls == 1:
            first_started.set()
            await asyncio.sleep(30)
        else:
            await new_ready.wait()

    registry = ProcessRegistry(factory)
    monkeypatch.setattr(registry, "_wait_ready", wait_ready)
    original = registry._terminate

    async def delayed_cleanup(generation):
        await asyncio.sleep(0.03)
        await original(generation)

    monkeypatch.setattr(registry, "_terminate", delayed_cleanup)
    first = asyncio.create_task(registry.acquire("demo"))
    await first_started.wait()
    first.cancel()
    await asyncio.sleep(0)
    retries = [asyncio.create_task(registry.acquire("demo")) for _ in range(10)]
    try:
        with pytest.raises(asyncio.CancelledError):
            await first
        new_ready.set()
        loaded = await asyncio.gather(*retries)
        assert calls == 2 and len({row.process.pid for row in loaded}) == 1
        for row in loaded:
            registry.release(row)
    finally:
        await registry.close()


@pytest.mark.asyncio
async def test_public_running_contract_omits_failed_and_unauthorized_models():
    rows = [
        {"model": "allowed", "state": "ready"},
        {"model": "other", "state": "ready"},
        {"model": "failed", "state": "failed"},
    ]
    registry = SimpleNamespace(snapshot=lambda: rows)
    app = StudioDataPlane(
        registry,
        lambda model: None,
        identity_resolver=lambda headers: {
            "id": "scoped",
            "name": "Scoped",
            "models": ["allowed", "failed"],
            "scopes": ["models:read"],
        },
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            running = (await client.get("/running")).json()["running"]
            engines = (await client.get("/api/engines")).json()["engines"]
        assert running == engines == [{"model": "allowed", "state": "ready"}]
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_upstream_cookies_are_not_shared_between_callers_or_engines():
    from backend.proxy.studio.dataplane import NoUpstreamCookieStorage
    from http.cookiejar import CookieJar

    generation = SimpleNamespace(upstream="http://engine")
    registry = SimpleNamespace(
        acquire=lambda model: asyncio.sleep(0, result=generation), release=lambda generation: None
    )
    app = StudioDataPlane(registry, lambda model: ResolvedModel(model))
    # Keep the production cookie policy when substituting the network transport.
    policy = app.client.cookies.jar._policy
    assert isinstance(policy, NoUpstreamCookieStorage)
    await app.client.aclose()
    seen = []

    def engine(request):
        seen.append(request.headers.get("cookie"))
        return httpx.Response(
            200,
            headers={"set-cookie": "engine_session=private; Path=/"},
            stream=httpx.ByteStream(b"ok"),
        )

    app.client = httpx.AsyncClient(
        transport=httpx.MockTransport(engine), cookies=CookieJar(policy=policy)
    )
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as first:
            await first.get("/upstream/a/props", headers={"cookie": "caller_cookie=one"})
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as second:
            await second.get("/upstream/b/props")
        assert seen == ["caller_cookie=one", None]
        assert not app.client.cookies
    finally:
        await app.close()
