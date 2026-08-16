"""Gateway that rewrites audio.cpp HTML for llama-swap /upstream/{id}/."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from backend.audio_cpp_ui_gateway import (
    drop_port_flag,
    inject_child_port,
    parse_listen,
    rewrite_proxied_html,
    serve_gateway,
)


def test_drop_port_flag_removes_listen_port():
    assert drop_port_flag(
        ["audiocpp_server", "--host", "127.0.0.1", "--port", "${PORT}", "--ui-management"]
    ) == ["audiocpp_server", "--host", "127.0.0.1", "--ui-management"]
    assert inject_child_port(["audiocpp_server", "--port=2000"], 5801) == [
        "audiocpp_server",
        "--port",
        "5801",
    ]


def test_parse_listen_port_only():
    assert parse_listen("2000") == ("127.0.0.1", 2000)
    assert parse_listen("127.0.0.1:5801") == ("127.0.0.1", 5801)


def test_rewrite_proxied_html_only_touches_html():
    html = b'<html><head><script>__sveltekit_abc123 = { base: "" };</script></head></html>'
    out, ctype = rewrite_proxied_html(
        html,
        content_type="text/html",
        path="/",
        prefix="/upstream/audio-demo",
    )
    assert b'base: "/upstream/audio-demo"' in out
    assert ctype == "text/html; charset=utf-8"
    json_body, json_type = rewrite_proxied_html(
        b'{"status":"ok"}',
        content_type="application/json",
        path="/health",
        prefix="/upstream/audio-demo",
    )
    assert json_body == b'{"status":"ok"}'
    assert json_type is None


def test_gateway_rewrites_html_from_inner_server():
    html = b'<!doctype html><html><head><script>__sveltekit_abc123 = { base: "" };</script></head><body>ui</body></html>'

    class Inner(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            return

        def do_GET(self):
            payload = html if self.path in {"/", "/index.html"} else b'{"status":"ok"}'
            content_type = "text/html" if payload == html else "application/json"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    inner = HTTPServer(("127.0.0.1", 0), Inner)
    inner_thread = threading.Thread(target=inner.serve_forever, daemon=True)
    inner_thread.start()
    gateway = serve_gateway(
        listen_host="127.0.0.1",
        listen_port=0,
        inner_host="127.0.0.1",
        inner_port=inner.server_address[1],
        public_prefix="/upstream/audio-demo",
    )
    gateway_thread = threading.Thread(target=gateway.serve_forever, daemon=True)
    gateway_thread.start()
    try:
        import http.client

        port = gateway.server_address[1]
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/")
        response = conn.getresponse()
        body = response.read().decode("utf-8")
        assert response.status == 200
        assert 'base: "/upstream/audio-demo"' in body
        conn.request("GET", "/health")
        health = conn.getresponse()
        assert health.read() == b'{"status":"ok"}'
        conn.close()
    finally:
        gateway.shutdown()
        inner.shutdown()
