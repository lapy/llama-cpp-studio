#!/usr/bin/env python3
"""Compare observable inference behavior between two proxy backends.

Example:
  .venv/bin/python scripts/proxy_parity.py \
    --legacy http://127.0.0.1:2000 --studio http://127.0.0.1:2001 \
    --model my-model
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import hashlib
import zlib
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Observation:
    status: int
    content_type: str
    body: Any
    headers: dict = field(default_factory=dict)
    first_byte_ms: float | None = field(default=None, compare=False)
    duration_ms: float = field(default=0, compare=False)
    bytes_received: int = field(default=0, compare=False)
    chunks: int = field(default=0, compare=False)
    cancelled: bool = field(default=False, compare=False)
    incremental: bool = field(default=False, compare=False)

    def as_report(self):
        return self.__dict__


def canonical_response(value, ids=None):
    ids = {} if ids is None else ids
    if isinstance(value, list):
        return [canonical_response(item, ids) for item in value]
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in ("created", "created_at", "observed_at", "pid", "port"):
                continue
            if key in ("id", "tool_call_id") and isinstance(item, str):
                result[key] = ids.setdefault(item, f"identity-{len(ids)}")
            else:
                result[key] = canonical_response(item, ids)
        return result
    return value


async def observe(
    client: httpx.AsyncClient,
    base: str,
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
    content: bytes | None = None,
    slow_consumer: float = 0,
    cancel_after_bytes: int | None = None,
) -> Observation:
    """Observe raw stream arrival, bounded capture and cancellation, not .read()."""
    started = time.monotonic()
    first = None
    chunks = size = 0
    digest = hashlib.sha256()
    capture = bytearray()
    truncated = False
    kwargs = (
        {"content": content}
        if content is not None
        else {"json": payload}
        if payload is not None
        else {}
    )
    async with client.stream(
        method, base.rstrip("/") + path, headers=headers, **kwargs
    ) as response:
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        meaningful = {
            name: response.headers[name]
            for name in (
                "content-encoding",
                "content-disposition",
                "retry-after",
                "x-fixture",
                "access-control-allow-origin",
            )
            if name in response.headers
        }
        async for chunk in response.aiter_raw():
            if not chunk:
                continue
            if first is None:
                first = (time.monotonic() - started) * 1000
            chunks += 1
            size += len(chunk)
            digest.update(chunk)
            if len(capture) + len(chunk) <= 1024 * 1024:
                capture.extend(chunk)
            else:
                truncated = True
            if cancel_after_bytes is not None and size >= cancel_after_bytes:
                break
            if slow_consumer:
                await asyncio.sleep(slow_consumer)
    duration = (time.monotonic() - started) * 1000
    data = bytes(capture)
    if meaningful.get("content-encoding") == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        data = decoder.decompress(data, 1024 * 1024)
        truncated |= not decoder.eof
    body = None
    if not truncated:
        if content_type == "text/event-stream":
            frames, pending = [], []
            for line in data.splitlines() + [b""]:
                if line.startswith(b"data:"):
                    pending.append(line[5:].strip())
                elif not line and pending:
                    frame = b"\n".join(pending)
                    try:
                        frames.append(json.loads(frame))
                    except (ValueError, UnicodeDecodeError):
                        frames.append(frame.decode("utf-8", "replace"))
                    pending = []
            body = canonical_response(frames)
        else:
            try:
                body = canonical_response(json.loads(data))
            except (ValueError, UnicodeDecodeError):
                if content_type.startswith("text/"):
                    body = data.decode("utf-8", "replace")
    if body is None:
        body = {"sha256": digest.hexdigest(), "bytes": size, "capture_truncated": truncated}
    return Observation(
        response.status_code,
        content_type,
        body,
        meaningful,
        first,
        duration,
        size,
        chunks,
        cancel_after_bytes is not None and size >= cancel_after_bytes,
        chunks > 1 and first is not None and duration - first >= 20,
    )


def normalize(value: Any) -> Any:
    """Remove process-local values while retaining API shape and semantics."""
    if isinstance(value, list):
        return [normalize(item) for item in value]
    if isinstance(value, dict):
        ignored = {"pid", "port", "created_at", "started_at", "observed_at"}
        return {key: normalize(item) for key, item in sorted(value.items()) if key not in ignored}
    return value


async def main(args: argparse.Namespace) -> int:
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    checks: list[tuple[str, str, str, dict | None]] = [
        ("health", "GET", "/health", None),
        ("running", "GET", "/running", None),
    ]
    if args.model:
        checks.extend(
            [
                ("models", "GET", f"/upstream/{args.model}/v1/models", None),
                (
                    "chat-stream",
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": args.model,
                        "messages": [{"role": "user", "content": args.prompt}],
                        "max_tokens": 2,
                        "temperature": 0,
                        "stream": True,
                    },
                ),
            ]
        )
    failures = 0
    timeout = httpx.Timeout(args.timeout, read=args.timeout)
    async with httpx.AsyncClient(timeout=timeout) as client:
        for name, method, path, payload in checks:
            try:
                legacy, studio = await asyncio.gather(
                    observe(
                        client,
                        args.legacy,
                        method,
                        path,
                        payload=payload,
                        headers=headers,
                    ),
                    observe(
                        client,
                        args.studio,
                        method,
                        path,
                        payload=payload,
                        headers=headers,
                    ),
                )
            except Exception as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
                continue
            if legacy == studio:
                print(f"PASS {name}: {legacy.status} {legacy.content_type}")
                continue
            failures += 1
            print(f"FAIL {name}")
            print("  legacy:", json.dumps(legacy.__dict__, sort_keys=True))
            print("  studio:", json.dumps(studio.__dict__, sort_keys=True))
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", default="http://127.0.0.1:2000")
    parser.add_argument("--studio", default="http://127.0.0.1:2001")
    parser.add_argument("--model")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--prompt", default="Reply with OK")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(asyncio.run(main(parse_args())))
