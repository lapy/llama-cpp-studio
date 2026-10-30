"""Run streaming HTTP conformance against a checksum-pinned llama-swap v265.

python -m scripts.proxy_conformance --llama-swap /path/to/llama-swap --report result.json
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

PINNED_BINARY_SHA256 = "35670b18fee5e9a243cd84a361c8c929d1e352567e3b66939781869ffa24dcaf"
PINNED_SOURCE = "d1e743fea64324c990e16e951ac2ea1678f02228"


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


async def ready(url, process=None):
    async with httpx.AsyncClient(trust_env=False) as client:
        for _ in range(200):
            if process and process.poll() is not None:
                raise RuntimeError("llama-swap fixture exited before readiness")
            try:
                if (await client.get(url + "/health")).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.025)
    raise RuntimeError("fixture readiness timed out")


async def run(binary: str, report_path: str | None = None):
    digest = hashlib.sha256(Path(binary).read_bytes()).hexdigest()
    if digest != PINNED_BINARY_SHA256:
        raise ValueError("Conformance requires the checksum-pinned Linux amd64 v265 binary")
    with tempfile.TemporaryDirectory(prefix="studio-parity-") as root:
        root = Path(root)
        legacy_state, studio_state = root / "legacy", root / "studio"
        legacy_state.mkdir()
        studio_state.mkdir()
        legacy_port, studio_port, start_port = free_port(), free_port(), free_port()
        cfg = root / "config.yaml"
        command = f"{shlex.quote(sys.executable)} -m scripts.proxy_fixture_engine --port ${{PORT}} --state-dir {shlex.quote(str(legacy_state))}"
        cfg.write_text(
            yaml.safe_dump(
                {
                    "healthCheckTimeout": 20,
                    "logLevel": "error",
                    "sendLoadingState": False,
                    "startPort": start_port,
                    "models": {
                        "fixture": {"cmd": command, "proxy": "http://127.0.0.1:${PORT}", "ttl": 0}
                    },
                    "groups": {
                        "default": {"swap": False, "exclusive": False, "members": ["fixture"]}
                    },
                }
            )
        )
        log = (root / "legacy.log").open("w+")
        process = subprocess.Popen(
            [binary, "-config", str(cfg), "-listen", f"127.0.0.1:{legacy_port}"],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )

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
            legacy, studio = f"http://127.0.0.1:{legacy_port}", f"http://127.0.0.1:{studio_port}"
            await asyncio.gather(ready(legacy, process), ready(studio))
            from scripts.proxy_parity import observe

            cases = [
                (
                    "cold-chat-json",
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
                                path,
                                payload=payload,
                                headers=headers,
                                content=content,
                                slow_consumer=0.002 if name == "binary-slow-consumer" else 0,
                            )
                            for base in (legacy, studio)
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
                            "legacy": left.as_report(),
                            "studio": right.as_report(),
                        }
                    )
                for label, base, state in (
                    ("legacy", legacy, legacy_state),
                    ("studio", studio, studio_state),
                ):
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
            performance["legacy"] = await measure(legacy, "fixture", 32, 8)
            performance["studio"] = await measure(studio, "fixture", 32, 8)
            performance["studio_vs_legacy_throughput"] = (
                performance["studio"]["requests_per_second"]
                / performance["legacy"]["requests_per_second"]
            )
        finally:
            server.should_exit = True
            await serving
            await app.close()
            await registry.close()
            process.send_signal(signal.SIGTERM)
            try:
                await asyncio.to_thread(process.wait, 5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                await asyncio.to_thread(process.wait)
            log.flush()
            log.seek(0)
            legacy_log = log.read()[-4000:]
            log.close()
        report = {
            "reference": {
                "version": "v265",
                "source_commit": PINNED_SOURCE,
                "binary_sha256": digest,
            },
            "performance": performance,
            "passed": all(r["passed"] for r in results)
            and not any(
                value.get("errors", 0) for value in performance.values() if isinstance(value, dict)
            ),
            "checks": results,
        }
        if not report["passed"]:
            report["legacy_log"] = legacy_log
        if report_path:
            Path(report_path).write_text(json.dumps(report, indent=2) + "\n")
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-swap", required=True)
    parser.add_argument("--report", default="proxy-conformance.json")
    args = parser.parse_args()
    result = asyncio.run(run(args.llama_swap, args.report))
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
