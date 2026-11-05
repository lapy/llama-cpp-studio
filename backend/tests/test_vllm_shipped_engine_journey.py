"""Shipped-engine journey: vLLM install → discovery → launch spec.

Hermetic: a hand-built local ``vllm`` wheel provides the console script and
module the YAML profile/discovery names; pip gets a wheelhouse; no network, no
GPU. The real definition recipes, discovery scanner and launch compiler run.
"""

from __future__ import annotations

import base64
import hashlib
import os
import zipfile
from pathlib import Path

import pytest

import httpx

import backend.data_store as ds
from backend.data_store import DataStore
from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python import generic as gen
from backend.engines.python import installer as venv_installer
from backend.engines.python.generic import GenericVenvInstaller
from backend.engines.scan import scanner as scanner_mod

VERSION = "0.1.0"
WHEEL = f"vllm-{VERSION}-py3-none-any.whl"

FAKE_API_SERVER = '''"""Hermetic vLLM stand-in: declared help probe + health server."""
from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HELP = """usage: vllm serve [options]

options:
  --port PORT           Port number for the server. (default: 8000)
  --max-model-len MAX_MODEL_LEN
                        Model context length.
  --tensor-parallel-size TENSOR_PARALLEL_SIZE
                        Number of tensor parallel replicas. (default: 1)
"""


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if any(item.startswith("--help") for item in args):
        print(HELP)
        return 0
    model = ""
    port = 8000
    index = 0
    while index < len(args):
        if args[index] == "--model" and index + 1 < len(args):
            model = args[index + 1]
            index += 2
            continue
        if args[index] == "--port" and index + 1 < len(args):
            port = int(args[index + 1])
            index += 2
            continue
        index += 1

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return None

        def do_GET(self):
            body = {"data": [{"id": model, "object": "model"}]} if self.path.startswith("/v1/models") else {"status": "ok"}
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                'data: {"choices":[{"delta":{"content":"vllm"}}]}\\n\\n',
                'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\\n\\n',
                "data: [DONE]\\n\\n",
            ):
                self.wfile.write(chunk.encode())
                self.wfile.flush()

    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _record(files: dict[str, bytes]) -> bytes:
    lines = []
    for name, data in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        lines.append(f"{name},sha256={digest},{len(data)}")
    lines.append("vllm-0.1.0.dist-info/RECORD,,")
    return ("\n".join(lines) + "\n").encode()


def _build_wheel(wheelhouse: Path) -> Path:
    files = {
        "vllm/__init__.py": b'__version__ = "0.1.0"\n',
        "vllm/entrypoints/__init__.py": b"",
        "vllm/entrypoints/openai/__init__.py": b"",
        "vllm/entrypoints/openai/api_server.py": FAKE_API_SERVER.encode(),
        "vllm-0.1.0.dist-info/METADATA": (
            "Metadata-Version: 2.1\nName: vllm\nVersion: 0.1.0\n\n"
            "Hermetic vLLM stand-in for the shipped-engine journey.\n"
        ).encode(),
        "vllm-0.1.0.dist-info/WHEEL": (
            "Wheel-Version: 1.0\nGenerator: handrolled\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ).encode(),
        "vllm-0.1.0.dist-info/entry_points.txt": (
            "[console_scripts]\nvllm = vllm.entrypoints.openai.api_server:main\n"
        ).encode(),
    }
    files["vllm-0.1.0.dist-info/RECORD"] = _record(files)
    path = wheelhouse / WHEEL
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return path


@pytest.mark.asyncio
async def test_vllm_install_discovery_and_launch_from_the_definition(tmp_path, monkeypatch):
    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(venv_installer, "studio_data_dir", lambda: str(tmp_path / "data"))
    monkeypatch.setattr(venv_installer, "get_store", lambda: store)
    monkeypatch.setattr(gen, "get_store", lambda: store)
    monkeypatch.setattr(ds, "get_store", lambda: store)
    monkeypatch.setattr(ds, "_store", store)

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    assert _build_wheel(wheelhouse).is_file()

    real_run_pip = GenericVenvInstaller._run_pip

    async def pip_with_wheelhouse(
        self, args, operation, ensure_venv=True, *, cwd=None, env=None, append=True
    ):
        argv = list(args)
        if argv and argv[0] == "install":
            argv += ["--no-index", "--find-links", str(wheelhouse)]
        return await real_run_pip(
            self, argv, operation, ensure_venv, cwd=cwd, env=env, append=append
        )

    monkeypatch.setattr(GenericVenvInstaller, "_run_pip", pip_with_wheelhouse)

    async def fake_activate(*args, **kwargs):
        return None

    monkeypatch.setattr(gen, "activate_engine_version", fake_activate)

    definition = load_default_engine_definitions().definitions["vllm"]
    installer = GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "vllm.log"),
        base_dir=str(tmp_path / "vllm"),
    )
    holder: dict = {}
    installer._create_task = lambda coro: holder.__setitem__("coro", coro)

    await installer.install_release(version=VERSION)
    await holder["coro"]
    assert installer._last_error is None

    rows = store.get_engine_versions("vllm")
    assert len(rows) == 1, rows
    row = rows[0]
    assert row["version"] == VERSION
    assert os.path.isfile(os.path.join(row["venv_path"], "bin", "vllm"))
    snapshots = store.list_install_snapshots("vllm")
    assert any(s["version"] == VERSION and s["phase"] == "published" for s in snapshots)

    # Discovery runs the YAML probe (vllm serve --help=all) through the
    # definition scanner against the installed console script.
    entry = scanner_mod.scan_definition_engine_version("vllm", row)
    assert entry["scan_error"] is None, entry
    params = {param["key"] for section in entry["sections"] for param in section["params"]}
    assert {"port", "max_model_len", "tensor_parallel_size"} <= params

    # Launch uses the shipped profile (venv python -m vllm.entrypoints...).
    store.set_active_engine_version("vllm", VERSION)
    store.add_model(
        {
            "id": "vllm-model",
            "name": "vLLM Model",
            "display_name": "vLLM Model",
            "huggingface_id": "org/model",
            "format": "safetensors",
            "config": {"engine": "vllm", "engines": {"vllm": {}}},
        }
    )
    from backend.proxy.launch_spec import compile_model_runtime

    runtime = compile_model_runtime(store.get_model("vllm-model"))
    assert runtime.launch.executable.endswith("venv/bin/python")
    argv = runtime.launch.argv
    assert argv[:2] == ["-m", "vllm.entrypoints.openai.api_server"], argv
    assert "--model" in argv
    assert argv[argv.index("--model") + 1] == "org/model"

    # Launch, readiness and one streamed request through the native process
    # owner the shipped profile actually targets.
    from backend.proxy.studio.registry import ProcessRegistry

    async def launch_factory(model_id):
        return runtime.launch, runtime.revision, None

    registry = ProcessRegistry(
        launch_factory,
        readiness_timeout=30.0,
    )
    try:
        generation = await registry.acquire("vllm-model")
        assert generation.ready is True
        assert generation.process.poll() is None
        async with httpx.AsyncClient(timeout=5.0) as client:
            health = await client.get(f"http://127.0.0.1:{generation.port}/health")
            assert health.status_code == 200
            chat = await client.post(
                f"http://127.0.0.1:{generation.port}/v1/chat/completions",
                json={
                    "model": "org/model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
            )
        assert chat.status_code == 200
        assert "vllm" in chat.text
        assert "[DONE]" in chat.text
    finally:
        await registry.close()
    assert generation.process.poll() is not None

    # Lifecycle removal through the same definition-owned installer.
    assert store.delete_model("vllm-model") is True
    holder.clear()
    await installer.remove()
    await holder["coro"]
    assert installer._last_error is None, installer._last_error
    assert store.get_engine_versions("vllm") == []
    assert store.get_active_engine_version("vllm") is None
    assert not os.path.exists(row["venv_path"])
