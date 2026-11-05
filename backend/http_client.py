"""Lifecycle-managed HTTP client for upstream calls that must not stall the API."""

from __future__ import annotations

from typing import Optional

import httpx

# Connect and read deadlines for GitHub, PyPI, and similar control-plane calls.
# Downloads of model weights use their own longer timeouts.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=5.0, read=20.0, write=10.0, pool=5.0)
UPSTREAM_LIMITS = httpx.Limits(max_connections=20, max_keepalive_connections=10)
UPSTREAM_RETRIES = 2

_client: Optional[httpx.AsyncClient] = None


def get_http_client() -> httpx.AsyncClient:
    """Return the process-wide async client, creating it on first use."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            timeout=UPSTREAM_TIMEOUT,
            limits=UPSTREAM_LIMITS,
            follow_redirects=True,
        )
    return _client


async def aclose_http_client() -> None:
    """Close the shared client. Safe to call when it was never opened."""
    global _client
    client = _client
    _client = None
    if client is not None and not client.is_closed:
        await client.aclose()


async def request_with_retries(
    method: str,
    url: str,
    *,
    retries: int = UPSTREAM_RETRIES,
) -> httpx.Response:
    """Issue a request with bounded retries for transport failures only."""
    client = get_http_client()
    attempt = 0
    while True:
        try:
            response = await client.request(method, url)
            return response
        except httpx.TransportError:
            if attempt >= retries:
                raise
            attempt += 1


async def get_text(url: str) -> httpx.Response:
    """GET a URL using the shared client and its deadlines."""
    return await request_with_retries("GET", url)
