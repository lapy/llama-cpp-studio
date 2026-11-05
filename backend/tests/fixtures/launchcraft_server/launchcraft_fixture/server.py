"""Launchcraft: the launch campaign's hermetic fixture SERVER.

A stdlib-only HTTP server implementing the contract Studio's native proxy
requires from ANY engine (docs/yaml-engine-launch-campaign.md, slices
R3/R4):

* ``GET /health`` → 2xx once ready (the ProcessRegistry readiness gate),
* ``GET /v1/models`` → OpenAI-style model list (the start route's
  passthrough target),
* ``POST /v1/chat/completions`` → deterministic completion; real SSE
  chunks when ``stream`` is true (the proxy round-trip proof),
* ``GET /__state`` → argv/cwd/pid/hits as observed BY THE SERVER.

``/__state`` is what makes the launch proof byte-honest: the test reads
the argv the OS actually passed to the executable after Studio's typed
port-placeholder substitution, and the hit counter proves each chat
completion really reached this process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _os_argv():
    """The argv the OS actually passed (CPython rewrites sys.argv for -m)."""
    try:
        raw = open("/proc/self/cmdline", "rb").read()
    except OSError:
        return list(sys.argv)
    return [part.decode("utf-8", "replace") for part in raw.split(b"\x00") if part]


class LaunchcraftHandler(BaseHTTPRequestHandler):
    server_version = "LaunchcraftFixture/0.1"
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep readiness polling quiet
        pass

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _read_payload(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _chunk(self, payload: dict) -> None:
        data = f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8")
        self.wfile.write(data)
        self.wfile.flush()

    def _stream_chat(self, model: str) -> None:
        # Real SSE: headers, incremental chunks, terminator, connection close.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        for piece in ("launchcraft", "-ok"):
            self._chunk(
                {
                    "id": "chatcmpl-launchcraft",
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}],
                }
            )
        self._chunk(
            {
                "id": "chatcmpl-launchcraft",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        )
        self._chunk_terminator()

    def _chunk_terminator(self) -> None:
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler contract
        path = self.path.split("?", 1)[0]
        self.server.hits[path] = self.server.hits.get(path, 0) + 1
        if path == "/health":
            self._send(200, {"status": "ok"})
        elif path == "/v1/models":
            self._send(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.server.model_ref, "object": "model"}],
                },
            )
        elif path == "/__state":
            self._send(
                200,
                {
                    "argv": list(sys.argv),
                    "os_argv": _os_argv(),
                    "cwd": os.getcwd(),
                    "pid": os.getpid(),
                    "model_ref": self.server.model_ref,
                    "hits": dict(self.server.hits),
                    "launchforge_mode": os.environ.get("LAUNCHFORGE_MODE"),
                },
            )
        else:
            self._send(404, {"error": "not found", "path": path})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler contract
        path = self.path.split("?", 1)[0]
        self.server.hits[path] = self.server.hits.get(path, 0) + 1
        if path != "/v1/chat/completions":
            self._send(404, {"error": "not found", "path": path})
            return
        request = self._read_payload()
        model = str(request.get("model") or self.server.model_ref)
        if request.get("stream"):
            self.server.stream_requests += 1
            self._stream_chat(model)
            return
        self._send(
            200,
            {
                "id": "chatcmpl-launchcraft",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "launchcraft-ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )


SCAN_HELP = """launchcraft-server 0.1.0

----- common params -----
-m, --model PATH            path to the model served by this fixture
--host HOST                 interface to bind (default: 127.0.0.1)
--port PORT                 port to listen on (default: ephemeral)
-b, --max-batch-size N      maximum concurrent requests (default: 8)
-t, --threads N             worker threads (default: 4)

----- launchcraft params -----
--scan-help                 print this scan help and exit
--log-level LEVEL           log verbosity (default: info)
"""


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--scan-help" in args:
        # Deliberately NOT --help: the YAML discovery block declares this
        # probe, so a successful scan proves the declared prefix was used.
        sys.stdout.write(SCAN_HELP)
        return 0
    parser = argparse.ArgumentParser(prog="launchcraft-server")
    # Two supported bindings, both exercised by shipped definitions:
    # * `--model PATH` (launchcraft: flag binding, console-script executable)
    # * positional PATH  (launchforge: positional binding, `python -m ...`)
    parser.add_argument("model_path", nargs="?", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parsed = parser.parse_args(args)
    model_ref = parsed.model_path or parsed.model or "fixture/model"
    server = ThreadingHTTPServer((parsed.host, parsed.port), LaunchcraftHandler)
    server.daemon_threads = True
    server.model_ref = model_ref
    server.hits = {}
    server.stream_requests = 0
    try:
        server.serve_forever(poll_interval=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
