"""Run native proxy streaming HTTP conformance against its direct upstream.

python -m scripts.proxy_conformance --report result.json
The fixture starts real child engines on loopback and requires no models or GPU.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import uvicorn
import yaml

from backend.proxy.launch_spec import EnvSpec, LaunchSpec
from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane
from backend.proxy.studio.registry import ProcessRegistry


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def ready(url, process=None):
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(200):
            if process and process.poll() is not None:
                raise RuntimeError("Studio proxy fixture exited before readiness")
            try:
                if (await client.get(url + "/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.025)
    raise RuntimeError("fixture readiness timed out")


async def run(report_path: str | None = None):
    with tempfile.TemporaryDirectory(prefix="studio-conformance-") as root:
        root = Path(root)
        studio_state = root / "studio"
        studio_state.mkdir()
        studio_port = free_port()

        async def factory(model):
            return (
                LaunchSpec(
                    model_id=model,
                    engine_id="fixture",
                    engine_install_id="fixture",
                    executable=sys.executable,
                    argv=[
                        "-m",
                        "scripts.proxy_fixture_engine",
                        "--port",
                        {"runtime": "port"},
                        "--state-dir",
                        str(studio_state),
                    ],
                    cwd=os.getcwd(),
                    env=EnvSpec(),
                ),
                "a" * 64,
                None,
            )

        registry = ProcessRegistry(factory, readiness_timeout=20)
        app = StudioDataPlane(
            registry, lambda model: ResolvedModel("fixture") if model == "fixture" else None
        )
        server = uvicorn.Server(
            uvicorn.Config(
                app, host="127.0.0.1", port=studio_port, log_level="error", lifespan="off"
            )
        )
        serving = asyncio.create_task(server.serve())
        results = []
        performance = {}
        try:
            studio = f"http://127.0.0.1:{studio_port}"
            await ready(studio)
            generation = await registry.acquire("fixture")
            registry.release(generation)
            direct = generation.upstream
            from scripts.proxy_parity import observe

            cases = [
                (
                    "chat-json",
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": "fixture",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": False,
                    },
                    None,
                ),
                (
                    "sse-fragmented-tools",
                    "POST",
                    "/v1/chat/completions",
                    {"model": "fixture", "messages": [], "stream": True},
                    None,
                ),
                ("upstream-status", "GET", "/upstream/fixture/fixture/error", None, None),
                ("compressed-body", "GET", "/upstream/fixture/fixture/gzip", None, None),
                ("binary-slow-consumer", "GET", "/upstream/fixture/fixture/binary", None, None),
                (
                    "hop-headers",
                    "POST",
                    "/upstream/fixture/fixture/echo?probe=1",
                    None,
                    b"raw bytes",
                ),
                (
                    "multipart-order",
                    "POST",
                    "/upstream/fixture/fixture/echo",
                    None,
                    b'--fixture-boundary\r\nContent-Disposition: form-data; name="file"; filename="sample.wav"\r\nContent-Type: audio/wav\r\n\r\nRIFF-fixture\r\n--fixture-boundary\r\nContent-Disposition: form-data; name="language"\r\n\r\nen\r\n--fixture-boundary\r\nContent-Disposition: form-data; name="language"\r\n\r\nfr\r\n--fixture-boundary\r\nContent-Disposition: form-data; name="model"\r\n\r\nfixture\r\n--fixture-boundary--\r\n',
                ),
            ]
            async with httpx.AsyncClient(trust_env=False, timeout=30) as client:
                for name, method, path, payload, content in cases:
                    headers = {
                        "x-probe": "fixture",
                        "connection": "x-drop",
                        "x-drop": "must-not-forward",
                    }
                    if name == "multipart-order":
                        headers["content-type"] = "multipart/form-data; boundary=fixture-boundary"
                    observations = await asyncio.gather(
                        *(
                            observe(
                                client,
                                base,
                                method,
                                path.replace("/upstream/fixture", "") if base == direct else path,
                                payload=payload,
                                headers={
                                    key: value
                                    for key, value in headers.items()
                                    if base != direct or key not in {"connection", "x-drop"}
                                },
                                content=content,
                                slow_consumer=0.002 if name == "binary-slow-consumer" else 0,
                            )
                            for base in (direct, studio)
                        )
                    )
                    left, right = observations
                    passed = left == right
                    if name == "sse-fragmented-tools":
                        passed &= left.incremental and right.incremental
                    results.append(
                        {
                            "name": name,
                            "passed": passed,
                            "upstream": left.as_report(),
                            "studio": right.as_report(),
                        }
                    )
                for label, base, state in (("studio", studio, studio_state),):
                    await observe(
                        client,
                        base,
                        "GET",
                        "/upstream/fixture/fixture/slow",
                        headers={"x-fixture-request": "cancel-probe"},
                        cancel_after_bytes=1,
                    )
                    disconnected = False
                    for _ in range(100):
                        events = [
                            json.loads(line)
                            for line in (state / "events.jsonl").read_text().splitlines()
                        ]
                        if any(
                            e["event"] == "disconnected" and e["request"] == "cancel-probe"
                            for e in events
                        ):
                            disconnected = True
                            break
                        await asyncio.sleep(0.02)
                    results.append(
                        {"name": f"{label}-upstream-cancellation", "passed": disconnected}
                    )
            from scripts.proxy_benchmark import measure

            performance["direct"] = await measure(
                registry.active("fixture").upstream, "fixture", 32, 8
            )
            performance["studio"] = await measure(studio, "fixture", 32, 8)
        finally:
            server.should_exit = True
            await serving
            await app.close()
            await registry.close()
        report = {
            "reference": "Direct fixture-engine HTTP responses",
            "performance": performance,
            "passed": all(r["passed"] for r in results)
            and not any(
                value.get("errors", 0) for value in performance.values() if isinstance(value, dict)
            ),
            "checks": results,
        }
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="proxy-conformance.json")
    args = parser.parse_args()
    result = asyncio.run(run(args.report))
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": [{"name": r["name"], "passed": r["passed"]} for r in result["checks"]],
                "report": args.report,
            },
            indent=2,
        )
    )
    sys.exit(0 if result["passed"] else 1)
