"""Raw, bounded reverse-proxy transport using the existing aiohttp dependency.

The public HTTPX transport interface keeps request construction/callers stable.
Aiohttp supplies efficient streaming HTTP/1.1 I/O, cancellation and backpressure.
No redirects, cookie storage, automatic decompression or environment proxies.
"""

from __future__ import annotations

import asyncio

import aiohttp
import httpx
from multidict import CIMultiDict
from yarl import URL


class IdleCappedConnector(aiohttp.TCPConnector):
    """Apply the independent idle-count cap absent from aiohttp's public API.

    This small release hook is covered by real socket tests and the pinned
    aiohttp dependency. All other connection lifecycle remains library-owned.
    """

    def __init__(self, max_connections, max_idle):
        super().__init__(limit=max_connections, keepalive_timeout=5)
        self.max_idle = max_idle

    def _release(self, key, protocol, *, should_close=False):
        if sum(len(items) for items in self._conns.values()) >= self.max_idle:
            should_close = True
        super()._release(key, protocol, should_close=should_close)


class ResponseStream(httpx.AsyncByteStream):
    def __init__(self, response, release, request):
        self.response = response
        self.release = release
        self.request = request
        self.closed = False

    async def __aiter__(self):
        try:
            async for chunk in self.response.content.iter_any():
                yield chunk
        except asyncio.TimeoutError as exc:
            raise httpx.ReadTimeout(str(exc), request=self.request) from exc
        except aiohttp.ClientError as exc:
            raise httpx.RemoteProtocolError(str(exc), request=self.request) from exc

    async def aclose(self):
        if not self.closed:
            self.closed = True
            try:
                # Release reuses a fully consumed connection; otherwise it
                # closes immediately rather than draining a cancelled stream.
                self.response.release()
                await self.response.wait_for_close()
            finally:
                self.release()


class UpstreamTransport(httpx.AsyncBaseTransport):
    def __init__(self, max_connections=200, max_keepalive_connections=40):
        self.max_connections = max_connections
        self.max_idle = max_keepalive_connections
        self._slots = asyncio.Semaphore(max_connections)
        self.active = 0
        self.session = None
        self.closed = False

    def _session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession(
                connector=IdleCappedConnector(self.max_connections, self.max_idle),
                cookie_jar=aiohttp.DummyCookieJar(),
                auto_decompress=False,
                trust_env=False,
                skip_auto_headers={"Accept-Encoding", "Content-Type", "User-Agent"},
            )
        return self.session

    async def handle_async_request(self, request):
        if self.closed:
            raise httpx.TransportError("Upstream transport is closed")
        timeouts = request.extensions.get("timeout", {})
        acquired = False
        try:
            async with asyncio.timeout(timeouts.get("pool")):
                await self._slots.acquire()
                acquired = True
        except TimeoutError as exc:
            if acquired:
                self._slots.release()
            raise httpx.PoolTimeout(
                "Upstream connection pool wait timed out", request=request
            ) from exc
        self.active += 1

        def release():
            self.active -= 1
            self._slots.release()

        try:
            try:
                data = request.content or None
            except httpx.RequestNotRead:
                data = request.stream
            timeout = aiohttp.ClientTimeout(
                total=None, sock_connect=timeouts.get("connect"), sock_read=timeouts.get("read")
            )
            response = await self._session().request(
                request.method,
                URL(str(request.url), encoded=True),
                headers=CIMultiDict(request.headers.multi_items()),
                data=data,
                timeout=timeout,
                allow_redirects=False,
            )
        except BaseException as exc:
            release()
            if isinstance(exc, asyncio.TimeoutError):
                raise httpx.ReadTimeout(str(exc), request=request) from exc
            if isinstance(exc, aiohttp.ClientConnectorError):
                raise httpx.ConnectError(str(exc), request=request) from exc
            if isinstance(exc, aiohttp.ClientError):
                raise httpx.RemoteProtocolError(str(exc), request=request) from exc
            raise
        return httpx.Response(
            response.status,
            headers=response.raw_headers,
            stream=ResponseStream(response, release, request),
            extensions={
                "http_version": f"HTTP/{response.version.major}.{response.version.minor}".encode()
            },
        )

    async def aclose(self):
        if not self.closed:
            self.closed = True
            if self.session:
                await self.session.close()
