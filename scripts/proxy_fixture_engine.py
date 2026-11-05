"""Deterministic CPU-only HTTP engine for proxy conformance, never production."""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
from pathlib import Path

import uvicorn


def event(directory, kind, request=""):
    if directory:
        with (Path(directory) / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": kind, "request": request, "pid": os.getpid()}) + "\n")


class FixtureEngine:
    def __init__(self, directory=None):
        self.directory = directory

    async def __call__(self, scope, receive, send):
        path = scope["path"]
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if not message.get("more_body", False):
                break
        headers = {key.decode().lower(): value.decode() for key, value in scope.get("headers", [])}
        request_id = headers.get("x-fixture-request", "")
        if path == "/health":
            await self.respond(send, {"status": "ok"})
        elif path.endswith("/models"):
            await self.respond(
                send, {"object": "list", "data": [{"id": "fixture", "object": "model"}]}
            )
        elif path == "/fixture/echo":
            await self.respond(
                send,
                {
                    "method": scope["method"],
                    "path": path,
                    "query": scope.get("query_string", b"").decode(),
                    "sha256": hashlib.sha256(body).hexdigest(),
                    "content_type": headers.get("content-type"),
                    "x_probe": headers.get("x-probe"),
                    "hop_header": headers.get("x-drop"),
                },
            )
        elif path == "/fixture/error":
            await self.respond(send, {"error": "fixture upstream rejected request"}, 422)
        elif path == "/fixture/gzip":
            data = gzip.compress(
                b'{"fixture":"compressed","usage":{"prompt_tokens":3,"completion_tokens":2}}',
                mtime=0,
            )
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"content-encoding", b"gzip"),
                        (b"content-length", str(len(data)).encode()),
                    ],
                }
            )
            await send({"type": "http.response.body", "body": data})
        elif path == "/fixture/binary":
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/octet-stream")],
                }
            )
            for _ in range(64):
                await send(
                    {
                        "type": "http.response.body",
                        "body": bytes(range(256)) * 64,
                        "more_body": True,
                    }
                )
            await send({"type": "http.response.body", "body": b""})
        elif path.endswith("/chat/completions") or path == "/fixture/slow":
            try:
                payload = json.loads(body or b"{}")
            except ValueError:
                payload = {}
            if not payload.get("stream") and path != "/fixture/slow":
                await self.respond(
                    send,
                    {
                        "id": "fixed",
                        "object": "chat.completion",
                        "choices": [{"message": {"role": "assistant", "content": "héllo"}}],
                        "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    },
                )
                return
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"cache-control", b"no-cache"),
                        (b"x-fixture", b"stream"),
                    ],
                }
            )
            event(self.directory, "stream_started", request_id)

            async def watch():
                while True:
                    if (await receive())["type"] == "http.disconnect":
                        event(self.directory, "disconnected", request_id)
                        return

            watcher = asyncio.create_task(watch())
            try:
                frames = [
                    b'data: {"id":"fixed","choices":[{"delta":{"role":"assistant"}}]}\n\n',
                    'data: {"id":"fixed","choices":[{"delta":{"content":"héllo"}}]}\n\n'.encode(),
                    b'data: {"id":"fixed","choices":[{"delta":{"tool_calls":[{"index":0,"id":"tool-fixed","function":{"name":"lookup","arguments":"{\\"q\\":"}}]}}]}\n\n',
                    b'data: {"id":"fixed","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\"test\\"}"}}]}}]}\n\n',
                ]
                if path == "/fixture/slow":
                    frames *= 20
                for frame in frames:
                    # Split inside UTF-8 and JSON boundaries deliberately.
                    midpoint = frame.find(b"\xc3") + 1 if b"\xc3" in frame else len(frame) // 2
                    await send(
                        {"type": "http.response.body", "body": frame[:midpoint], "more_body": True}
                    )
                    await asyncio.sleep(0.015)
                    if watcher.done():
                        return
                    await send(
                        {"type": "http.response.body", "body": frame[midpoint:], "more_body": True}
                    )
                    await asyncio.sleep(0.015)
                    if watcher.done():
                        return
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'data: {"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\ndata: [DONE]\n\n',
                    }
                )
                event(self.directory, "stream_finished", request_id)
            except (OSError, asyncio.CancelledError):
                event(self.directory, "disconnected", request_id)
            finally:
                watcher.cancel()
                await asyncio.gather(watcher, return_exceptions=True)
        else:
            await self.respond(send, {"error": "fixture endpoint not found"}, 404)

    @staticmethod
    async def respond(send, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(data)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": data})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--state-dir", required=True)
    args = parser.parse_args()
    Path(args.state_dir).mkdir(parents=True, exist_ok=True)
    event(args.state_dir, "started")
    uvicorn.run(
        FixtureEngine(args.state_dir),
        host="127.0.0.1",
        port=args.port,
        log_level="error",
        lifespan="off",
    )
