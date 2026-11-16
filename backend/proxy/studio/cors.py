"""CORS headers for the native Studio proxy."""

from typing import Callable


def cors_headers(origin: str) -> list:
    if not origin:
        return []
    return [
        (b"access-control-allow-origin", b"*"),
        (b"vary", b"Origin"),
        (
            b"access-control-expose-headers",
            b"X-Request-ID, Retry-After, X-RateLimit-Remaining-Requests, X-RateLimit-Remaining-Tokens-Day, X-RateLimit-Remaining-Tokens-Month",
        ),
    ]


async def cors_preflight(send: Callable, origin: str) -> None:
    headers = [
        (b"access-control-allow-methods", b"GET, POST, PUT, PATCH, DELETE, OPTIONS"),
        (b"access-control-allow-headers", b"*"),
        (b"access-control-max-age", b"600"),
        (b"content-length", b"0"),
        *cors_headers(origin),
    ]
    await send({"type": "http.response.start", "status": 204, "headers": headers})
    await send({"type": "http.response.body", "body": b""})
