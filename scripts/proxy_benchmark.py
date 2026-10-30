"""Compare direct-engine and Studio-proxy latency over real HTTP connections.

Example: python -m scripts.proxy_benchmark --direct http://127.0.0.1:3001 \
  --proxy http://127.0.0.1:2000 --model demo --requests 100 --concurrency 8
Set STUDIO_BENCHMARK_KEY for inference authentication; it is never printed.
--fixture runs a deterministic local streaming engine and quota-enabled proxy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import resource
import socket
import tempfile
import time
from types import SimpleNamespace

import aiohttp
import uvicorn


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)] if values else None


async def measure(url, model, count, concurrency, token=""):
    from backend.proxy.studio.usage import UsageObserver

    sem = asyncio.Semaphore(concurrency)
    rows = []
    started = time.perf_counter()
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120),
        trust_env=False,
        connector=aiohttp.TCPConnector(limit=concurrency),
        cookie_jar=aiohttp.DummyCookieJar(),
    ) as client:

        async def request():
            async with sem:
                begin = time.perf_counter()
                first = None
                size = 0
                first_token = None
                async with client.post(
                    url.rstrip("/") + "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {token}"} if token else {},
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": "Reply hello."}],
                        "stream": True,
                        "stream_options": {"include_usage": True},
                        "max_tokens": 16,
                    },
                ) as response:
                    observer = UsageObserver(
                        "text/event-stream" in response.headers.get("content-type", "")
                    )
                    async for chunk in response.content.iter_any():
                        if chunk and first is None:
                            first = (time.perf_counter() - begin) * 1000
                        size += len(chunk)
                        observer.feed(chunk)
                        if first_token is None and observer.saw_token:
                            first_token = (time.perf_counter() - begin) * 1000
                    observer.finish()
                    rows.append(
                        {
                            "total_ms": (time.perf_counter() - begin) * 1000,
                            "first_byte_ms": first,
                            "first_token_ms": first_token,
                            "output_tokens": observer.output_tokens,
                            "status": response.status,
                            "bytes": size,
                        }
                    )

        # Warm the actual measured client's keepalive connections. A separate
        # warmup client would discard its pool and measure connection ramp-up.
        await asyncio.gather(*(request() for _ in range(concurrency)))
        await asyncio.sleep(0.01)
        rows.clear()
        started = time.perf_counter()
        await asyncio.gather(*(request() for _ in range(count)))
    elapsed = time.perf_counter() - started
    return {
        "load_generator": "aiohttp; warmed keepalive pool",
        "warmup_requests": concurrency,
        "requests": count,
        "concurrency": concurrency,
        "elapsed_seconds": elapsed,
        "requests_per_second": count / elapsed,
        "errors": sum(row["status"] >= 400 for row in rows),
        "p50_ms": percentile([r["total_ms"] for r in rows], 0.5),
        "p95_ms": percentile([r["total_ms"] for r in rows], 0.95),
        "first_byte_p50_ms": percentile(
            [r["first_byte_ms"] for r in rows if r["first_byte_ms"] is not None], 0.5
        ),
        "bytes_per_second": sum(row["bytes"] for row in rows) / elapsed,
        "first_token_p50_ms": percentile(
            [r["first_token_ms"] for r in rows if r["first_token_ms"] is not None], 0.5
        ),
        "output_tokens_per_second": sum(r["output_tokens"] for r in rows) / elapsed
        if all(r["output_tokens"] is not None for r in rows)
        else None,
    }


async def fixture(stack, accounting_mode="quota"):
    from backend.proxy.studio.accounting import Accounting
    from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane

    async def engine(scope, receive, send):
        while (await receive()).get("more_body", False):
            pass
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/event-stream")],
            }
        )
        for _ in range(4):
            await asyncio.sleep(0.002)
            await send(
                {
                    "type": "http.response.body",
                    "body": b'data: {"choices":[{"delta":{"content":"hello"}}]}\n\n',
                    "more_body": True,
                }
            )
        await send(
            {
                "type": "http.response.body",
                "body": b'data: {"usage":{"prompt_tokens":4,"completion_tokens":4}}\n\ndata: [DONE]\n\n',
            }
        )

    async def serve(app):
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(app, log_level="error", lifespan="off"))
        task = asyncio.create_task(server.serve(sockets=[sock]))
        while not server.started:
            if task.done():
                await task
            await asyncio.sleep(0.005)
        stack.append((server, task, sock))
        return f"http://127.0.0.1:{port}"

    direct = await serve(engine)

    class Registry:
        async def acquire(self, model):
            return SimpleNamespace(
                upstream=direct, engine_id="fixture", process=SimpleNamespace(pid=os.getpid())
            )

        def release(self, generation):
            pass

    temp = tempfile.TemporaryDirectory()
    plane = StudioDataPlane(
        Registry(),
        lambda name: ResolvedModel("fixture"),
        accounting=Accounting(temp.name + "/accounting.sqlite3")
        if accounting_mode != "off"
        else None,
        identity_resolver=lambda headers: {
            "id": "benchmark",
            "name": "Benchmark",
            "scopes": ["inference"],
            "daily_token_budget": 100000000 if accounting_mode == "quota" else None,
            "max_output_tokens": 64,
        },
    )
    proxy = await serve(plane)
    return direct, proxy, plane, temp


async def main(args):
    servers = []
    plane = temp = None
    try:
        if args.fixture:
            direct, proxy, plane, temp = await fixture(servers, args.accounting)
            model = "fixture"
        else:
            if not args.direct or not args.proxy or not args.model:
                raise ValueError("--direct, --proxy and --model are required without --fixture")
            direct, proxy, model = args.direct, args.proxy, args.model
        # Warm connection/engine paths before measuring.
        await measure(direct, model, 1, 1)
        await measure(proxy, model, 1, 1, os.getenv("STUDIO_BENCHMARK_KEY", ""))
        baseline = await measure(direct, model, args.requests, args.concurrency)
        proxied = await measure(
            proxy, model, args.requests, args.concurrency, os.getenv("STUDIO_BENCHMARK_KEY", "")
        )
        result = {
            "fixture": args.fixture,
            "accounting_mode": args.accounting if args.fixture else "server configuration",
            "direct": baseline,
            "proxy": proxied,
            "p50_overhead_ms": proxied["p50_ms"] - baseline["p50_ms"],
            "p95_overhead_ms": proxied["p95_ms"] - baseline["p95_ms"],
            "benchmark_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        }
        encoded = json.dumps(result, indent=2)
        if args.output:
            from pathlib import Path

            Path(args.output).write_text(encoded + "\n")
        print(encoded)
    finally:
        for server, _, _ in servers:
            server.should_exit = True
        await asyncio.gather(*(task for _, task, _ in servers))
        for _, _, sock in servers:
            sock.close()
        if plane:
            await plane.close()
        if temp:
            temp.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct")
    parser.add_argument("--proxy")
    parser.add_argument("--model")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument(
        "--accounting",
        choices=("quota", "tracking", "off"),
        default="quota",
        help="Fixture accounting mode; real servers keep their configured policies",
    )
    parser.add_argument("--output")
    arguments = parser.parse_args()
    if arguments.requests < 1 or arguments.concurrency < 1:
        parser.error("requests and concurrency must be positive")
    asyncio.run(main(arguments))
