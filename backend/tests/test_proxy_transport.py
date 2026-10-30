import asyncio
import socket

import httpx
import pytest
import pytest_asyncio
import uvicorn

from backend.proxy.studio.transport import UpstreamTransport


@pytest_asyncio.fixture
async def upstream():
    started, finish = asyncio.Event(), asyncio.Event()

    async def app(scope, receive, send):
        while (await receive()).get("more_body", False):
            pass
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"set-cookie", b"a=1"), (b"set-cookie", b"b=2")],
            }
        )
        await send({"type": "http.response.body", "body": b"first", "more_body": True})
        started.set()
        if scope["path"] == "/heartbeat":
            for _ in range(8):
                await asyncio.sleep(0.01)
                await send({"type": "http.response.body", "body": b".", "more_body": True})
        else:
            await finish.wait()
        await send({"type": "http.response.body", "body": b"last"})

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
    task = asyncio.create_task(server.serve(sockets=[sock]))
    while not server.started:
        await asyncio.sleep(0.005)
    try:
        yield f"http://127.0.0.1:{port}", started, finish
    finally:
        finish.set()
        server.should_exit = True
        await task
        sock.close()


@pytest.mark.asyncio
async def test_pool_leases_hold_until_body_close_and_cookies_are_not_stored(upstream):
    url, started, finish = upstream
    transport = UpstreamTransport(max_connections=2, max_keepalive_connections=1)
    async with httpx.AsyncClient(transport=transport) as client:
        responses = await asyncio.gather(
            *(client.send(client.build_request("GET", url), stream=True) for _ in range(2))
        )
        assert transport.active == 2
        assert responses[0].headers.get_list("set-cookie") == ["a=1", "b=2"]
        finish.set()
        await asyncio.gather(*(response.aread() for response in responses))
        await asyncio.gather(*(response.aclose() for response in responses))
        assert transport.active == 0
        connector = transport.session.connector
        assert sum(len(items) for items in connector._conns.values()) <= 1
        assert not transport.session.cookie_jar
    await transport.aclose()  # idempotent


@pytest.mark.asyncio
async def test_cancelled_pool_acquisition_does_not_leak_lease(upstream):
    url, started, finish = upstream
    transport = UpstreamTransport(max_connections=1, max_keepalive_connections=1)
    async with httpx.AsyncClient(transport=transport) as client:
        first = await client.send(client.build_request("GET", url), stream=True)
        second = asyncio.create_task(client.send(client.build_request("GET", url), stream=True))
        await asyncio.sleep(0.01)
        second.cancel()
        with pytest.raises(asyncio.CancelledError):
            await second
        assert transport.active == 1
        await first.aclose()
        assert transport.active == 0
        finish.set()


@pytest.mark.asyncio
async def test_pool_timeout_is_distinct_and_bounded(upstream):
    url, started, finish = upstream
    transport = UpstreamTransport(max_connections=1, max_keepalive_connections=1)
    async with httpx.AsyncClient(
        transport=transport, timeout=httpx.Timeout(2, pool=0.02)
    ) as client:
        first = await client.send(client.build_request("GET", url), stream=True)
        with pytest.raises(httpx.PoolTimeout):
            await client.get(url)
        await first.aclose()
        assert transport.active == 0
        finish.set()


@pytest.mark.asyncio
async def test_idle_timeout_does_not_limit_total_healthy_stream_duration(upstream):
    url, started, finish = upstream
    transport = UpstreamTransport()
    async with httpx.AsyncClient(transport=transport, timeout=0.05) as client:
        response = await client.get(url + "/heartbeat")
        assert response.content == b"first........last"
    assert transport.active == 0


@pytest.mark.asyncio
async def test_stalled_body_raises_read_timeout_and_releases_lease(upstream):
    url, started, finish = upstream
    transport = UpstreamTransport()
    async with httpx.AsyncClient(transport=transport, timeout=0.02) as client:
        with pytest.raises(httpx.ReadTimeout):
            await client.get(url)
    assert transport.active == 0
