"""HTTP gateway in front of audiocpp_server for llama-swap ``/upstream/{id}/``.

llama-swap binds the model process on ``${PORT}`` and strips ``/upstream/{id}``
before proxying. audio.cpp's SvelteKit UI still sees that prefix in the browser,
so this process:

1. Listens on ``${PORT}``
2. Runs ``audiocpp_server`` on a loopback port (not ``${PORT}``)
3. Rewrites HTML so ``base`` and ``/v1`` fetches use ``/upstream/{id}``
"""

from __future__ import annotations

import argparse
import http.client
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from backend.audio_cpp_ui_rewrite import rewrite_audio_cpp_ui_html
from backend.logging_config import get_logger

logger = get_logger(__name__)

_HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
    "accept-encoding",
}

_CHILD_READY_TIMEOUT_SEC = 600.0


def drop_port_flag(argv: Sequence[str]) -> List[str]:
    """Remove ``--port`` / ``-p`` so the child does not bind llama-swap's ``${PORT}``."""
    out: List[str] = []
    skip_next = False
    for token in argv:
        if skip_next:
            skip_next = False
            continue
        text = str(token)
        if text in ("--port", "-p"):
            skip_next = True
            continue
        if text.startswith("--port="):
            continue
        out.append(text)
    return out


def inject_child_port(argv: Sequence[str], port: int) -> List[str]:
    return drop_port_flag(argv) + ["--port", str(int(port))]


def allocate_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def parse_listen(value: str) -> Tuple[str, int]:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("listen address required")
    if raw.isdigit():
        return "127.0.0.1", int(raw)
    parsed = urlparse(f"//{raw}")
    host = parsed.hostname or "127.0.0.1"
    if parsed.port is None:
        raise ValueError(f"listen address must include a port: {value!r}")
    return host, int(parsed.port)


def _is_html(content_type: str, path: str) -> bool:
    lowered = (content_type or "").lower()
    if "text/html" in lowered:
        return True
    path_only = (path or "/").split("?", 1)[0]
    return path_only in {"", "/", "/index.html"}


def rewrite_proxied_html(
    content: bytes,
    *,
    content_type: str,
    path: str,
    prefix: str,
) -> Tuple[bytes, Optional[str]]:
    if not _is_html(content_type, path):
        return content, None
    try:
        return rewrite_audio_cpp_ui_html(content, prefix=prefix), "text/html; charset=utf-8"
    except (ValueError, UnicodeDecodeError):
        return content, None


def _filter_headers(
    headers: Iterable[Tuple[str, str]],
    *,
    extra_drop: Optional[Iterable[str]] = None,
) -> List[Tuple[str, str]]:
    drop = set(_HOP_BY_HOP)
    if extra_drop:
        drop.update(name.lower() for name in extra_drop)
    out: List[Tuple[str, str]] = []
    for key, value in headers:
        if str(key).lower() in drop:
            continue
        out.append((key, value))
    return out


def wait_for_port(
    host: str,
    port: int,
    *,
    proc: Optional[subprocess.Popen] = None,
    timeout: float = _CHILD_READY_TIMEOUT_SEC,
) -> None:
    deadline = time.monotonic() + max(0.1, timeout)
    last_error: Optional[BaseException] = None
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(
                f"audiocpp_server exited with {proc.returncode} before opening {host}:{port}"
            )
        try:
            with socket.create_connection((host, int(port)), timeout=1.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05)
    raise TimeoutError(f"audiocpp_server did not listen on {host}:{port}: {last_error}")


class _PrefixProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    inner_host = "127.0.0.1"
    inner_port = 0
    public_prefix = ""

    def log_message(self, format: str, *args) -> None:
        logger.info("%s - %s", self.address_string(), format % args)

    def do_GET(self) -> None:
        self._proxy()

    def do_HEAD(self) -> None:
        self._proxy()

    def do_POST(self) -> None:
        self._proxy()

    def do_PUT(self) -> None:
        self._proxy()

    def do_PATCH(self) -> None:
        self._proxy()

    def do_DELETE(self) -> None:
        self._proxy()

    def do_OPTIONS(self) -> None:
        self._proxy()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return b""
        return self.rfile.read(length)

    def _proxy(self) -> None:
        body = b""
        if self.command not in {"GET", "HEAD", "OPTIONS"}:
            body = self._read_body()
        headers = {
            key: value
            for key, value in _filter_headers(self.headers.items())
        }
        headers["Host"] = f"{self.inner_host}:{self.inner_port}"
        conn = http.client.HTTPConnection(self.inner_host, self.inner_port, timeout=300)
        try:
            conn.request(self.command, self.path, body=body or None, headers=headers)
            upstream = conn.getresponse()
            payload = upstream.read()
            content_type = upstream.getheader("content-type") or ""
            rewritten_type = None
            if self.command != "HEAD":
                payload, rewritten_type = rewrite_proxied_html(
                    payload,
                    content_type=content_type,
                    path=self.path,
                    prefix=self.public_prefix,
                )
            response_headers = _filter_headers(
                upstream.getheaders(),
                extra_drop=("content-encoding",) if rewritten_type else None,
            )
            if rewritten_type:
                response_headers = [
                    (key, value)
                    for key, value in response_headers
                    if str(key).lower() != "content-type"
                ]
                response_headers.append(("Content-Type", rewritten_type))
            self.send_response(upstream.status, upstream.reason)
            for key, value in response_headers:
                self.send_header(key, value)
            if self.command != "HEAD":
                self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(payload)
        except OSError as exc:
            logger.warning("audio.cpp UI gateway upstream error: %s", exc)
            self.send_error(502, f"audiocpp_server unreachable: {exc}")
        finally:
            conn.close()


def serve_gateway(
    *,
    listen_host: str,
    listen_port: int,
    inner_host: str,
    inner_port: int,
    public_prefix: str,
) -> ThreadingHTTPServer:
    handler = type(
        "AudioCppUiProxyHandler",
        (_PrefixProxyHandler,),
        {
            "inner_host": inner_host,
            "inner_port": int(inner_port),
            "public_prefix": str(public_prefix).rstrip("/"),
        },
    )
    server = ThreadingHTTPServer((listen_host, int(listen_port)), handler)
    server.daemon_threads = True
    return server


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--listen",
        required=True,
        help="Gateway bind port or host:port (llama-swap ${PORT})",
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument(
        "--public-prefix",
        required=True,
        help="Browser path prefix, e.g. /upstream/audio-cpp-pocket_tts_english_q8_0",
    )
    parser.add_argument("child", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv) if argv is not None else None)
    child = list(args.child or [])
    if child and child[0] == "--":
        child = child[1:]
    args.child = child
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.child:
        raise SystemExit("audiocpp_server command required after --")
    listen_host, listen_port = parse_listen(args.listen)
    if args.listen.isdigit():
        listen_host = str(args.listen_host or "127.0.0.1")
    prefix = str(args.public_prefix or "").strip().rstrip("/")
    if not prefix.startswith("/"):
        raise SystemExit("--public-prefix must be an absolute path")

    inner_port = allocate_loopback_port()
    child_argv = inject_child_port(args.child, inner_port)
    logger.info(
        "audio.cpp UI gateway listen=%s:%s prefix=%s child_port=%s",
        listen_host,
        listen_port,
        prefix,
        inner_port,
    )
    proc = subprocess.Popen(child_argv)

    def _shutdown(*_args) -> None:
        _terminate_process(proc)
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    def _reap() -> None:
        code = proc.wait()
        logger.warning("audiocpp_server exited with %s", code)
        os._exit(code or 1)

    threading.Thread(target=_reap, daemon=True).start()
    try:
        wait_for_port("127.0.0.1", inner_port, proc=proc)
        server = serve_gateway(
            listen_host=listen_host,
            listen_port=listen_port,
            inner_host="127.0.0.1",
            inner_port=inner_port,
            public_prefix=prefix,
        )
        try:
            server.serve_forever()
        finally:
            server.server_close()
    except Exception:
        _terminate_process(proc)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
