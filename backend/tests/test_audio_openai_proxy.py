"""Tests for Studio /v1/audio transport (format adapter + native-route forwarding)."""

from __future__ import annotations

import io
import json
import wave
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient


def _minimal_wav() -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(b"\x00\x00" * 800)
    return buf.getvalue()


class _FakeUpstream:
    def __init__(self, status_code, *, content=b"", headers=None, json_body=None):
        if json_body is not None:
            content = json.dumps(json_body).encode()
            headers = {**(headers or {}), "content-type": "application/json"}
        self.status_code = status_code
        self.headers = httpx.Headers(headers or {})
        self.content = content

    async def aiter_raw(self):
        yield self.content

    async def aiter_bytes(self):
        yield self.content

    async def aclose(self):
        return None


def _install_upstream(monkeypatch, handler):
    from backend.routes import audio_openai_proxy as proxy

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, method, url, **kwargs):
            return SimpleNamespace(method=method, url=url, kwargs=kwargs)

        async def send(self, request, stream=False):
            return await handler(request, request.kwargs)

        async def aclose(self):
            return None

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)


@pytest.fixture
def client(monkeypatch):
    from backend.main import app
    from backend.routes import audio_openai_proxy as proxy

    monkeypatch.setattr(proxy, "_get_proxy_port", lambda: 2000)
    return TestClient(app)


def test_transcriptions_converts_non_wav_and_forwards(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    wav = _minimal_wav()
    seen: dict[str, Any] = {}

    def fake_ensure(content, *, filename=None, content_type=None):
        assert content == b"ogg-bytes"
        return wav, "voice.wav"

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["files"] = kwargs.get("files")
        return _FakeUpstream(200, json_body={"text": "hello"})

    monkeypatch.setattr(proxy, "ensure_wav_bytes_http", fake_ensure)
    _install_upstream(monkeypatch, handler)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-demo"},
        files={"file": ("memo.ogg", b"ogg-bytes", "audio/ogg")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    assert seen["url"].startswith("http://127.0.0.1:2000/v1/audio/transcriptions")
    files = seen["files"]
    assert files
    file_part = next(part for part in files if part[0] == "file")
    assert file_part[1][0] == "voice.wav"
    assert file_part[1][2] == "audio/wav"


def test_speech_passthrough(client, monkeypatch):
    async def handler(request, _kwargs):
        assert str(request.url) == "http://127.0.0.1:2000/v1/audio/speech"
        return _FakeUpstream(
            200,
            content=b"RIFF....WAVE",
            headers={"content-type": "audio/wav"},
        )

    _install_upstream(monkeypatch, handler)
    response = client.post(
        "/v1/audio/speech",
        json={"model": "tts-demo", "input": "hi"},
    )
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")


def test_speech_converts_opus_from_wav(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    wav = _minimal_wav()

    async def handler(request, _kwargs):
        return _FakeUpstream(
            200,
            content=wav,
            headers={"content-type": "audio/wav"},
        )

    _install_upstream(monkeypatch, handler)
    monkeypatch.setattr(
        proxy,
        "encode_wav_speech_format",
        lambda _content, fmt: (b"OPUS", "audio/opus") if fmt == "opus" else (_content, "audio/wav"),
    )
    response = client.post(
        "/v1/audio/speech",
        json={
            "model": "audio-cpp-pocket_tts_english_q8_0",
            "input": "Hello",
            "voice": "azelma",
            "response_format": "opus",
        },
    )
    assert response.status_code == 200
    assert response.content == b"OPUS"
    assert response.headers["content-type"].startswith("audio/opus")


def test_speech_passes_engine_errors_through(client, monkeypatch):
    payload = {"error": {"message": "failed to open embeddings/azelma.safetensors", "type": "server_error"}}

    async def handler(request, _kwargs):
        return _FakeUpstream(500, json_body=payload)

    _install_upstream(monkeypatch, handler)
    response = client.post(
        "/v1/audio/speech",
        json={"model": "pocket", "input": "hi", "voice": "azelma"},
    )
    assert response.status_code == 500
    assert response.json()["error"]["message"] == payload["error"]["message"]


def test_alignments_rewrites_to_upstream_passthrough(client, monkeypatch):
    from backend.audio_cpp_proxy_routing import AudioUpstreamTarget
    from backend.routes import audio_openai_proxy as proxy

    seen: dict[str, Any] = {}

    def fake_target(model, native_path, store=None):
        assert model == "voice"
        assert native_path == "/v1/audio/alignments"
        return AudioUpstreamTarget(path="/upstream/audio-demo/v1/audio/alignments", model="audio-demo")

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["files"] = kwargs.get("files")
        return _FakeUpstream(200, json_body={"words": []})

    monkeypatch.setattr(
        "backend.audio_cpp_proxy_routing.resolve_audio_upstream_target",
        fake_target,
    )
    monkeypatch.setattr(proxy, "ensure_wav_bytes_http", lambda content, **_k: (content, "clip.wav"))
    _install_upstream(monkeypatch, handler)

    response = client.post(
        "/v1/audio/alignments",
        data={"model": "voice", "text": "hello"},
        files={"file": ("clip.wav", _minimal_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    assert seen["url"].startswith("http://127.0.0.1:2000/upstream/audio-demo/v1/audio/alignments")
    model_part = next(part for part in seen["files"] if part[0] == "model")
    assert model_part[1][1] == "audio-demo"


def test_transcription_details_uses_upstream_passthrough(client, monkeypatch):
    from backend.audio_cpp_proxy_routing import AudioUpstreamTarget
    from backend.routes import audio_openai_proxy as proxy

    seen: dict[str, Any] = {}

    def fake_target(model, native_path, store=None):
        return AudioUpstreamTarget(
            path="/upstream/audio-asr/v1/audio/transcriptions/details",
            model="audio-asr",
        )

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["files"] = kwargs.get("files")
        return _FakeUpstream(200, json_body={"text": "hi", "words": []})

    monkeypatch.setattr(
        "backend.audio_cpp_proxy_routing.resolve_audio_upstream_target",
        fake_target,
    )
    monkeypatch.setattr(proxy, "ensure_wav_bytes_http", lambda content, **_k: (content, "memo.wav"))
    _install_upstream(monkeypatch, handler)

    response = client.post(
        "/v1/audio/transcriptions/details",
        data={"model": "asr-demo"},
        files={"file": ("memo.wav", _minimal_wav(), "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hi"
    assert seen["url"].startswith(
        "http://127.0.0.1:2000/upstream/audio-asr/v1/audio/transcriptions/details"
    )


def test_tasks_run_forwards_to_llama_swap_audioapi(client, monkeypatch):
    seen: dict[str, Any] = {}

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        body = kwargs.get("content")
        if hasattr(body, "__aiter__"):
            chunks = [chunk async for chunk in body]
            seen["payload"] = json.loads(b"".join(chunks))
        else:
            seen["payload"] = json.loads(body)
        return _FakeUpstream(200, json_body={"ok": True})

    _install_upstream(monkeypatch, handler)
    response = client.post(
        "/v1/tasks/run",
        json={"model": "vad-demo", "request": {"audio": "/tmp/a.wav"}},
    )
    assert response.status_code == 200
    assert seen["url"].startswith("http://127.0.0.1:2000/audioapi/v1/tasks/run")
    assert seen["payload"]["model"] == "vad-demo"
