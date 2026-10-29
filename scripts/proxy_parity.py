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
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Observation:
    status: int
    content_type: str
    body: Any


async def observe(
    client: httpx.AsyncClient,
    base: str,
    method: str,
    path: str,
    *,
    payload: dict | None = None,
    headers: dict[str, str] | None = None,
) -> Observation:
    response = await client.request(method, base.rstrip("/") + path, json=payload, headers=headers)
    content_type = response.headers.get("content-type", "").split(";", 1)[0]
    try:
        body = response.json()
    except ValueError:
        body = response.text
    return Observation(response.status_code, content_type, normalize(body))


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
