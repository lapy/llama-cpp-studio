"""Opt-in contract tests against a real released llama-swap executable.

Run with LLAMA_SWAP_TEST_BINARY=/path/to/llama-swap pytest this_file.py.
Everything is isolated under tmp_path with ephemeral ports; no Studio service,
model installation, active binary, or persistent configuration is changed.
"""

from __future__ import annotations

import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import httpx
import pytest
import yaml

from backend.llama_swap_config import _audio_cpp_swap_cmd, _swap_default_params


_SERVER = '''\
import base64
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.respond({"path": self.path})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if "multipart/form-data" in self.headers.get("Content-Type", ""):
            prefix = ("Content-Type: " + self.headers["Content-Type"] + "\\r\\n\\r\\n").encode()
            parts = BytesParser(policy=default).parsebytes(prefix + raw)
            fields = []
            for part in parts.iter_parts():
                value = part.get_payload(decode=True)
                filename = part.get_filename()
                fields.append({
                    "name": part.get_param("name", header="content-disposition"),
                    "value": base64.b64encode(value).decode() if filename else value.decode(),
                    "filename": filename,
                })
            self.respond({"path": self.path, "fields": fields})
        else:
            self.respond({"path": self.path, "payload": json.loads(raw)})

    def respond(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

ThreadingHTTPServer(("127.0.0.1", int(sys.argv[-1])), Handler).serve_forever()
'''


@pytest.fixture(scope="module")
def real_swap(tmp_path_factory):
    binary = os.environ.get("LLAMA_SWAP_TEST_BINARY")
    if not binary:
        pytest.skip("Set LLAMA_SWAP_TEST_BINARY to run released llama-swap contract tests")
    binary_path = Path(binary).resolve()
    assert binary_path.is_file(), f"Missing test binary: {binary_path}"
    root = tmp_path_factory.mktemp("real-llama-swap")
    script = root / "native_server.py"
    script.write_text(_SERVER)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    defaults = _swap_default_params({
        "temperature": 0.7,
        "options": {"num_beams": 4, "qwen3_asr.preserve_punctuation": True},
    })
    config = {
        "healthCheckTimeout": 10,
        "models": {"audio-model": {
            "cmd": _audio_cpp_swap_cmd({
                "cmd_argv": [sys.executable, str(script), "--port", "${PORT}"],
                "cmd_cwd": str(root),
            }),
            "aliases": ["voice", "voice:warm"],
            "useModelName": "native-model",
            "filters": {
                "setParams": defaults,
                "setParamsByID": {"voice:warm": {"temperature": 0.2}},
            },
        }},
    }
    config_path = root / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    log_path = root / "llama-swap.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [str(binary_path), "--config", str(config_path), "--listen", f"127.0.0.1:{port}"],
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=15, trust_env=False) as client:
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text()
                    try:
                        if client.get("/health").is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.05)
                else:
                    pytest.fail(f"llama-swap did not become ready: {log_path.read_text()}")
                yield client
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("path,upstream_path", [
    ("/v1/audio/speech", "/v1/audio/speech"),
    ("/audioapi/v1/tasks/run", "/v1/tasks/run"),
])
def test_real_swap_aliases_and_native_defaults_preserve_client_values(real_swap, path, upstream_path):
    result = real_swap.post(path, json={
        "model": "voice",
        "temperature": 0,
        "options": {"qwen3_asr.preserve_punctuation": False, "custom": "kept"},
    })
    result.raise_for_status()
    body = result.json()
    assert body["path"] == upstream_path
    assert body["payload"] == {
        "model": "native-model",
        "temperature": 0,
        "options": {
            "num_beams": 4,
            "qwen3_asr.preserve_punctuation": False,
            "custom": "kept",
        },
    }
    defaults = real_swap.post(path, json={"model": "voice"}).json()["payload"]
    assert defaults["temperature"] == 0.7
    assert defaults["options"]["qwen3_asr.preserve_punctuation"] is True
    variant = real_swap.post(path, json={"model": "voice:warm", "temperature": 1}).json()["payload"]
    assert variant["temperature"] == 0.2


def test_real_swap_transcription_alias_rewrites_only_model(real_swap):
    result = real_swap.post("/v1/audio/transcriptions", files=[
        ("model", (None, "voice")),
        ("timestamp_granularities[]", (None, "word")),
        ("timestamp_granularities[]", (None, "segment")),
        ("file", ("input.wav", b"RIFFaudio", "audio/wav")),
    ])
    result.raise_for_status()
    fields = result.json()["fields"]
    assert next(f["value"] for f in fields if f["name"] == "model") == "native-model"
    assert [f["value"] for f in fields if f["name"] == "timestamp_granularities[]"] == ["word", "segment"]
    assert not any(f["name"] == "temperature" for f in fields)


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions/details", "/v1/audio/alignments"])
def test_real_swap_generic_passthrough_preserves_new_native_multipart_endpoints(real_swap, path):
    result = real_swap.post(f"/upstream/audio-model{path}?detail=full", files=[
        ("model", (None, "native-model")),
        ("text", (None, "Align this text.")),
        ("file", ("input.wav", b"RIFFaudio", "audio/wav")),
    ])
    result.raise_for_status()
    body = result.json()
    assert body["path"] == f"{path}?detail=full"
    assert next(f["value"] for f in body["fields"] if f["name"] == "text") == "Align this text."


def test_real_swap_generic_passthrough_reaches_model_capability_api(real_swap):
    path = "/v1/models/native-model/capabilities"
    result = real_swap.get(f"/upstream/audio-model{path}")
    result.raise_for_status()
    assert result.json()["path"] == path
