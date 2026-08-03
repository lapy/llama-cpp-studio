"""Tests for Studio /v1/audio OpenAI proxy."""

from __future__ import annotations

import io
import wave

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


@pytest.fixture
def client(monkeypatch):
    from backend.main import app
    from backend.routes import audio_openai_proxy as proxy

    monkeypatch.setattr(proxy, "_get_proxy_port", lambda: 2000)
    return TestClient(app)


def test_transcriptions_converts_non_wav_and_forwards(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    wav = _minimal_wav()
    seen = {}

    def fake_ensure(content, *, filename=None, content_type=None):
        assert content == b"ogg-bytes"
        return wav, "voice.wav"

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, data=None, files=None):
            seen["url"] = url
            seen["data"] = data
            seen["files"] = files
            return httpx.Response(
                200,
                json={"text": "hello"},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(proxy, "ensure_wav_bytes_http", fake_ensure)
    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)

    response = client.post(
        "/v1/audio/transcriptions",
        data={"model": "asr-demo"},
        files={"file": ("memo.ogg", b"ogg-bytes", "audio/ogg")},
    )
    assert response.status_code == 200
    assert response.json()["text"] == "hello"
    assert seen["url"] == "http://127.0.0.1:2000/v1/audio/transcriptions"
    assert seen["data"] == {"model": "asr-demo"}
    assert "file" in seen["files"]
    assert seen["files"]["file"][0] == "voice.wav"
    assert seen["files"]["file"][2] == "audio/wav"


def test_speech_passthrough(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def request(self, method, url, headers=None, content=None):
            assert method == "POST"
            assert url == "http://127.0.0.1:2000/v1/audio/speech"
            return httpx.Response(
                200,
                content=b"RIFF....WAVE",
                headers={"content-type": "audio/wav"},
                request=httpx.Request(method, url),
            )

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    response = client.post(
        "/v1/audio/speech",
        json={"model": "tts-demo", "input": "hi"},
    )
    assert response.status_code == 200
    assert response.content.startswith(b"RIFF")


def test_tasks_run_forwards_to_upstream(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    seen = {}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def post(self, url, headers=None, json=None):
            seen["url"] = url
            seen["json"] = json
            return httpx.Response(
                200,
                json={"ok": True, "segments": []},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    response = client.post(
        "/v1/audio/tasks/run",
        json={
            "model": "vad-demo",
            "task": "vad",
            "input": {"audio_path": "/data/a.wav"},
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert seen["url"] == "http://127.0.0.1:2000/upstream/vad-demo/v1/tasks/run"
    assert seen["json"] == {"task": "vad", "input": {"audio_path": "/data/a.wav"}}


def test_tasks_run_requires_model_and_task(client):
    assert client.post("/v1/audio/tasks/run", json={"task": "vad"}).status_code == 400
    assert client.post(
        "/v1/audio/tasks/run",
        json={"model": "x"},
    ).status_code == 400
