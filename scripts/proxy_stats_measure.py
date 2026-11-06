"""Measure real Studio proxy statistics against an already-running engine.

Run inside a Studio development instance (or any host that can reach the proxy
and management API). The script creates throwaway inference keys, sends
cold/warm/concurrent streaming chat requests through the proxy, then reads the
finalized request history and ledger. Output is metadata only.

Example:
    STUDIO_ADMIN_TOKEN=... python scripts/proxy_stats_measure.py \
        --model qwen-local --output /tmp/studio-metrics.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import statistics
import subprocess
import time

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-url", default="http://127.0.0.1:2000")
    parser.add_argument("--management-url", default="http://127.0.0.1:8081")
    parser.add_argument(
        "--admin-token",
        default=os.getenv("STUDIO_ADMIN_TOKEN", ""),
        help="Studio service credential (or STUDIO_ADMIN_TOKEN).",
    )
    parser.add_argument("--model", required=True, help="Running Studio proxy model id.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="/tmp/studio-proxy-metrics.json")
    parser.add_argument("--cold-requests", type=int, default=3)
    parser.add_argument("--warm-requests", type=int, default=5)
    parser.add_argument("--concurrent-requests", type=int, default=4)
    return parser.parse_args()


async def stream_chat(
    client: httpx.AsyncClient, args, base: str, text: str, *, key: str, model: str = ""
):
    body = {
        "model": model or args.model,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    started = time.perf_counter()
    first = None
    last = None
    chunks = 0
    usage = None
    timings = None
    request_id = None
    async with client.stream(
        "POST", f"{base}/v1/chat/completions", json=body, headers=headers
    ) as response:
        request_id = response.headers.get("x-request-id")
        if response.status_code >= 400:
            detail = (await response.aread()).decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"proxy returned {response.status_code}: {detail}")
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            choices = event.get("choices") or []
            content = ""
            if choices and isinstance(choices[0], dict):
                delta = choices[0].get("delta") or {}
                content = str(delta.get("content") or "")
            if content:
                now = time.perf_counter()
                first = now if first is None else first
                last = now
                chunks += 1
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            if isinstance(event.get("timings"), dict):
                timings = event["timings"]
    finished = time.perf_counter()
    interval = (last - first) if first is not None and last is not None else None
    completion = int((usage or {}).get("completion_tokens") or 0)
    return {
        "request_id": request_id,
        "first_token_ms": (first - started) * 1000 if first else None,
        "total_ms": (finished - started) * 1000,
        "interval_ms": interval * 1000 if interval else None,
        "chunks": chunks,
        "completion_tokens": completion,
        "prompt_tokens": int((usage or {}).get("prompt_tokens") or 0),
        "cached_tokens": int(
            ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        ),
        "engine_decode": (timings or {}).get("predicted_per_second"),
        "engine_prompt": (timings or {}).get("prompt_per_second"),
        "usage": usage,
        "timings": timings,
    }


async def create_key(client: httpx.AsyncClient, args, admin: dict, name: str) -> dict:
    response = await client.post(
        f"{args.management_url}/api/inference/keys",
        headers=admin,
        json={
            "name": name,
            "token_accounting": "auto",
            "daily_token_budget": 5_000_000,
            "max_output_tokens": max(256, args.max_tokens),
            "scopes": ["inference", "models:read"],
        },
    )
    response.raise_for_status()
    return response.json()


def warm_prompt() -> str:
    prefix = (
        "The following operational notes describe a fictional distributed system. "
        "They are deliberately repetitive so the prompt can be reused for cache measurements. "
    ) * 40
    return f"{prefix}\nSummarize the notes above in one short sentence."


def cold_prompt(tag: str) -> str:
    body = " ".join(secrets.token_hex(8) for _ in range(150))
    return f"{body}\nVariant {tag}. Summarize the notes above in one short sentence."


async def main() -> None:
    args = parse_args()
    if not args.admin_token:
        raise SystemExit("--admin-token or STUDIO_ADMIN_TOKEN is required")
    admin = {"Authorization": f"Bearer {args.admin_token}"}
    timeout = httpx.Timeout(180.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        warm_key = (await create_key(client, args, admin, f"metrics-warmup-{time.time_ns()}"))[
            "key"
        ]
        await stream_chat(client, args, args.proxy_url, warm_prompt(), key=warm_key)

        created = await create_key(client, args, admin, f"metrics-clean-{time.time_ns()}")
        key = created["key"]
        key_id = created["id"]
        await client.post(f"{args.management_url}/api/proxy/stats/reset", headers=admin)
        marker = time.time()
        await asyncio.sleep(0.1)

        cohorts: dict[str, list[dict]] = {"cold": [], "warm": [], "concurrent": []}
        for index in range(args.cold_requests):
            cohorts["cold"].append(
                await stream_chat(
                    client, args, args.proxy_url, cold_prompt(f"cold-{index}"), key=key
                )
            )

        text = warm_prompt()
        for _ in range(2):
            await stream_chat(client, args, args.proxy_url, text, key=key)
        for _ in range(args.warm_requests):
            cohorts["warm"].append(await stream_chat(client, args, args.proxy_url, text, key=key))

        concurrent_text = warm_prompt() + "\nConcurrency variant."
        for _ in range(args.concurrent_requests):
            await stream_chat(client, args, args.proxy_url, concurrent_text, key=key)
        cohorts["concurrent"] = await asyncio.gather(
            *(
                stream_chat(client, args, args.proxy_url, concurrent_text, key=key)
                for _ in range(args.concurrent_requests)
            )
        )

        await asyncio.sleep(1.5)
        gateway = (await client.get(f"{args.management_url}/api/proxy", headers=admin)).json()
        engine_port = next(
            (
                row.get("port")
                for row in gateway.get("running") or []
                if row.get("model") == args.model
            ),
            None,
        )
        direct_warm = None
        engine_model = args.model
        if engine_port:
            try:
                models = (await client.get(f"http://127.0.0.1:{engine_port}/v1/models")).json()
                engine_model = (models.get("data") or [{}])[0].get("id") or args.model
            except Exception:
                pass
            direct_warm = await stream_chat(
                client,
                args,
                f"http://127.0.0.1:{engine_port}",
                warm_prompt(),
                key="",
                model=engine_model,
            )
        history = (
            await client.get(
                f"{args.management_url}/api/proxy/requests",
                params={"limit": 100, "model": args.model},
                headers=admin,
            )
        ).json()
        rows = [row for row in history.get("requests") or [] if row.get("started_at", 0) >= marker]
        by_id = {row["id"]: row for row in rows}
        keys = (await client.get(f"{args.management_url}/api/inference/keys", headers=admin)).json()
        key_usage = next(
            (row.get("usage") for row in keys.get("keys", []) if row.get("id") == key_id),
            None,
        )

    def merge(cohort: list[dict]) -> list[dict]:
        return [{"client": item, "row": by_id.get(item.get("request_id"))} for item in cohort]

    result = {
        "measured_at": time.time(),
        "model": args.model,
        "gpu": subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip(),
        "engine_port": engine_port,
        "cohorts": {name: merge(items) for name, items in cohorts.items()},
        "direct_warm": direct_warm,
        "history_rows": rows,
        "summary": gateway.get("summary"),
        "key_usage": key_usage,
    }
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    def mean(items, field):
        values = [
            item["row"][field]
            for item in items
            if item.get("row") and isinstance(item["row"].get(field), (int, float))
        ]
        return statistics.fmean(values) if values else None

    report = {
        name: {
            "requests": len(items),
            "ttft_ms_mean": mean(items, "ttft_ms"),
            "decode_mean": mean(items, "proxy_decode_tokens_per_second"),
            "cache_ratio": (
                sum(item["row"].get("input_tokens_cached") or 0 for item in items if item["row"])
                / sum(item["row"].get("input_tokens_total") or 0 for item in items if item["row"])
                if any(item["row"] for item in items)
                else None
            ),
        }
        for name, items in result["cohorts"].items()
    }
    print(
        json.dumps(
            {"cohorts": report, "key_usage": key_usage, "summary": result["summary"]}, indent=2
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
