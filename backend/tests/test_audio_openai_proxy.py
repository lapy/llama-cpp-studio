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


def _install_upstream(monkeypatch, handler, seen=None):
    from backend.routes import audio_openai_proxy as proxy

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, method, url, **kwargs):
            return SimpleNamespace(method=method, url=url, kwargs=kwargs)

        async def send(self, request, stream=False):
            if seen is not None:
                seen["send_stream"] = stream
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
    payload = {
        "error": {"message": "failed to open embeddings/azelma.safetensors", "type": "server_error"}
    }

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
    from backend.engines.audio_cpp.proxy_routing import AudioUpstreamTarget
    from backend.routes import audio_openai_proxy as proxy

    seen: dict[str, Any] = {}

    def fake_target(model, native_path, store=None):
        assert model == "voice"
        assert native_path == "/v1/audio/alignments"
        return AudioUpstreamTarget(
            path="/upstream/audio-demo/v1/audio/alignments", model="audio-demo"
        )

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["files"] = kwargs.get("files")
        return _FakeUpstream(200, json_body={"words": []})

    monkeypatch.setattr(
        "backend.engines.audio_cpp.proxy_routing.resolve_audio_upstream_target",
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
    from backend.engines.audio_cpp.proxy_routing import AudioUpstreamTarget
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
        "backend.engines.audio_cpp.proxy_routing.resolve_audio_upstream_target",
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


def test_batch_transcriptions_rewrite_to_upstream(client, monkeypatch):
    from backend.engines.audio_cpp.proxy_routing import AudioUpstreamTarget
    from backend.routes import audio_openai_proxy as proxy

    seen: dict[str, Any] = {}

    def fake_target(model, native_path, store=None):
        assert model == "meeting"
        assert native_path == "/v1/batches/transcriptions"
        return AudioUpstreamTarget(
            path="/upstream/audio-asr/v1/batches/transcriptions",
            model="audio-asr",
        )

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["files"] = kwargs.get("files")
        return _FakeUpstream(
            200, content=b"data: {}\n\n", headers={"content-type": "text/event-stream"}
        )

    monkeypatch.setattr(
        "backend.engines.audio_cpp.proxy_routing.resolve_audio_upstream_target",
        fake_target,
    )
    monkeypatch.setattr(proxy, "ensure_wav_bytes_http", lambda content, **_k: (content, "clip.wav"))
    _install_upstream(monkeypatch, handler)

    wav = _minimal_wav()
    response = client.post(
        "/v1/batches/transcriptions",
        data={"model": "meeting"},
        files=[
            ("file", ("a.wav", wav, "audio/wav")),
            ("file", ("b.wav", wav, "audio/wav")),
        ],
    )
    assert response.status_code == 200
    assert seen["url"].startswith(
        "http://127.0.0.1:2000/upstream/audio-asr/v1/batches/transcriptions"
    )
    file_parts = [part for part in seen["files"] if part[0] == "file"]
    assert len(file_parts) == 2
    model_part = next(part for part in seen["files"] if part[0] == "model")
    assert model_part[1][1] == "audio-asr"


def test_tasks_run_forwards_to_native_proxy_audioapi(client, monkeypatch):
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


# --- v0.9.5 generic task transports + engine-memory management ----------------


def _resolve_recorder(monkeypatch, mapping):
    """Patch the alias resolver and record every (model, native_path) lookup."""
    from backend.engines.audio_cpp.proxy_routing import AudioUpstreamTarget

    calls = []

    def fake_target(model, native_path, store=None):
        calls.append((model, native_path))
        stable = mapping[model]
        return AudioUpstreamTarget(path=f"/upstream/{stable}{native_path}", model=stable)

    monkeypatch.setattr(
        "backend.engines.audio_cpp.proxy_routing.resolve_audio_upstream_target",
        fake_target,
    )
    return calls


def _empty_audio_store(monkeypatch):
    """Make the real resolver see no registered models (unknown -> 404)."""
    from backend import data_store

    monkeypatch.setattr(
        data_store,
        "get_store",
        lambda: SimpleNamespace(list_models=lambda: []),
    )


def test_tasks_stream_rewrites_alias_and_streams_sse(client, monkeypatch):
    seen: dict[str, Any] = {}
    calls = _resolve_recorder(monkeypatch, {"s2s alias": "audio-s2s"})
    body = b'data: {"type":"task.stream.done"}\n\ndata: [DONE]\n\n'

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(kwargs["content"])
        seen["headers"] = dict(kwargs["headers"])
        return _FakeUpstream(
            200,
            content=body,
            headers={"content-type": "text/event-stream", "x-accel-buffering": "no"},
        )

    _install_upstream(monkeypatch, handler, seen)
    response = client.post(
        "/v1/tasks/stream?trace=1",
        json={"model": "s2s alias", "stream_format": "sse", "request": {"audio": "/q.wav"}},
    )
    assert response.status_code == 200
    assert response.content == body
    assert seen["send_stream"] is True
    assert seen["url"] == "http://127.0.0.1:2000/upstream/audio-s2s/v1/tasks/stream?trace=1"
    assert seen["payload"] == {
        "model": "audio-s2s",
        "stream_format": "sse",
        "request": {"audio": "/q.wav"},
    }
    # The upstream request keeps the client's Content-Type; the internal service
    # key is the only header Studio adds.
    assert seen["headers"]["content-type"] == "application/json"
    assert calls == [("s2s alias", "/v1/tasks/stream")]
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["cache-control"] == "no-cache"


def test_tasks_stream_forwards_json_response(client, monkeypatch):
    calls = _resolve_recorder(monkeypatch, {"s2s": "audio-s2s"})

    async def handler(request, _kwargs):
        return _FakeUpstream(200, json_body={"events": [], "result": {"text": "hi"}})

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/stream", json={"model": "s2s", "request": {}})
    assert response.status_code == 200
    assert response.json() == {"events": [], "result": {"text": "hi"}}
    assert calls == [("s2s", "/v1/tasks/stream")]


@pytest.mark.parametrize(
    "body",
    [b"not json", b"[]", b"{}", b'{"model": 1}', b'{"model": ""}'],
)
def test_tasks_stream_rejects_invalid_body_or_missing_model(client, monkeypatch, body):
    async def handler(_request, _kwargs):  # pragma: no cover - must not run
        raise AssertionError("invalid requests must not reach upstream")

    _install_upstream(monkeypatch, handler)
    response = client.post(
        "/v1/tasks/stream", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


def test_tasks_stream_unknown_model_returns_404(client, monkeypatch):
    _empty_audio_store(monkeypatch)
    response = client.post("/v1/tasks/stream", json={"model": "ghost", "request": {}})
    assert response.status_code == 404
    assert "Unknown Studio audio model" in response.json()["detail"]


def test_tasks_stream_rejects_oversized_body_by_content_length(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    monkeypatch.setattr(proxy, "MAX_AUDIO_UPLOAD_BYTES", 32)
    response = client.post("/v1/tasks/stream", json={"model": "x", "request": {"text": "y" * 128}})
    assert response.status_code == 413


def test_tasks_stream_rejects_oversized_chunked_body(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    monkeypatch.setattr(proxy, "MAX_AUDIO_UPLOAD_BYTES", 16)

    def chunks():
        yield b'{"model":"x",'
        yield b'"request":{}}'

    response = client.post(
        "/v1/tasks/stream",
        content=chunks(),
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 413


def test_tasks_batch_rewrites_alias_and_forwards_requests(client, monkeypatch):
    seen: dict[str, Any] = {}
    calls = _resolve_recorder(monkeypatch, {"asr-batch alias": "audio-batch"})

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(kwargs["content"])
        return _FakeUpstream(
            200,
            content=b'data: {"type":"task.batch.done"}\n\ndata: [DONE]\n\n',
            headers={"content-type": "text/event-stream"},
        )

    _install_upstream(monkeypatch, handler)
    requests = [{"audio": "/a.wav"}, {"audio": "/b.wav"}]
    response = client.post(
        "/v1/tasks/batch", json={"model": "asr-batch alias", "requests": requests}
    )
    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:2000/upstream/audio-batch/v1/tasks/batch"
    assert seen["payload"] == {"model": "audio-batch", "requests": requests}
    assert calls == [("asr-batch alias", "/v1/tasks/batch")]
    assert response.headers["x-accel-buffering"] == "no"


def test_unload_models_rewrites_each_id_and_pins_first(client, monkeypatch):
    seen: dict[str, Any] = {}
    calls = _resolve_recorder(monkeypatch, {"voice": "audio-demo", "asr": "audio-asr"})

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(kwargs["content"])
        return _FakeUpstream(
            200, json_body={"unloaded": ["audio-demo"], "not_found": ["audio-asr"]}
        )

    _install_upstream(monkeypatch, handler)
    response = client.post(
        "/v1/tasks/unload_models",
        json={"model_ids": ["voice", "asr"], "reason": "test"},
    )
    assert response.status_code == 200
    assert response.json() == {"unloaded": ["audio-demo"], "not_found": ["audio-asr"]}
    assert seen["url"] == "http://127.0.0.1:2000/upstream/audio-demo/v1/tasks/unload_models"
    assert seen["payload"] == {
        "model_ids": ["audio-demo", "audio-asr"],
        # Studio only rewrites the ids; unrelated fields pass through untouched.
        "reason": "test",
    }
    assert calls == [
        ("voice", "/v1/tasks/unload_models"),
        ("asr", "/v1/tasks/unload_models"),
    ]


@pytest.mark.parametrize(
    "body",
    [
        b"nope",
        b"[]",
        b"{}",
        b'{"model_ids":"x"}',
        b'{"model_ids":[1]}',
        b'{"model_ids":[""]}',
    ],
)
def test_unload_models_rejects_invalid_shape(client, monkeypatch, body):
    async def handler(_request, _kwargs):  # pragma: no cover - must not run
        raise AssertionError("invalid requests must not reach upstream")

    _install_upstream(monkeypatch, handler)
    response = client.post(
        "/v1/tasks/unload_models", content=body, headers={"content-type": "application/json"}
    )
    assert response.status_code == 400


def test_unload_models_empty_list_requires_an_explicit_pin(client, monkeypatch):
    async def handler(_request, _kwargs):  # pragma: no cover - must not run
        raise AssertionError("unpinned empty lists must not reach upstream")

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/unload_models", json={"model_ids": []})
    assert response.status_code == 400
    assert "model_ids" in response.json()["detail"]


def test_unload_models_empty_list_with_pin_forwards_no_ids(client, monkeypatch):
    seen: dict[str, Any] = {}
    calls = _resolve_recorder(monkeypatch, {"voice": "audio-demo"})

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(kwargs["content"])
        return _FakeUpstream(200, json_body={"unloaded": [], "not_found": []})

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/unload_models?model=voice", json={"model_ids": []})
    assert response.status_code == 200
    assert seen["url"] == "http://127.0.0.1:2000/upstream/audio-demo/v1/tasks/unload_models"
    assert seen["payload"] == {"model_ids": []}
    assert calls == [("voice", "/v1/tasks/unload_models")]


def test_unload_models_explicit_pin_selects_the_engine_instance(client, monkeypatch):
    seen: dict[str, Any] = {}
    _resolve_recorder(monkeypatch, {"voice": "audio-demo", "asr": "audio-asr"})

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(kwargs["content"])
        return _FakeUpstream(200, json_body={"unloaded": [], "not_found": ["audio-demo"]})

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/unload_models?model=asr", json={"model_ids": ["voice"]})
    assert response.status_code == 200
    # The explicit pin chooses the engine instance; the body still lists the
    # rewritten ids the caller asked about.
    assert seen["url"] == "http://127.0.0.1:2000/upstream/audio-asr/v1/tasks/unload_models"
    assert seen["payload"] == {"model_ids": ["audio-demo"]}


def test_unload_models_unknown_id_returns_404(client, monkeypatch):
    _empty_audio_store(monkeypatch)
    response = client.post("/v1/tasks/unload_models", json={"model_ids": ["ghost"]})
    assert response.status_code == 404


def test_unload_models_rejects_oversized_body(client, monkeypatch):
    from backend.routes import audio_openai_proxy as proxy

    monkeypatch.setattr(proxy, "MAX_AUDIO_UPLOAD_BYTES", 16)
    response = client.post("/v1/tasks/unload_models", json={"model_ids": ["voice"]})
    assert response.status_code == 413


def test_unload_all_models_requires_a_model_pin(client, monkeypatch):
    async def handler(_request, _kwargs):  # pragma: no cover - must not run
        raise AssertionError("unpinned unload-all must not reach upstream")

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/unload_all_models")
    assert response.status_code == 400
    assert "model query parameter" in response.json()["detail"]


def test_unload_all_models_forwards_and_strips_the_pin(client, monkeypatch):
    seen: dict[str, Any] = {}
    calls = _resolve_recorder(monkeypatch, {"voice": "audio-demo"})

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["content"] = kwargs["content"]
        return _FakeUpstream(200, json_body={"unloaded": ["audio-demo"]})

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/unload_all_models?model=voice&trace=1")
    assert response.status_code == 200
    assert response.json() == {"unloaded": ["audio-demo"]}
    # The Studio-only `model` pin is not forwarded; other query fields are.
    assert seen["url"] == (
        "http://127.0.0.1:2000/upstream/audio-demo/v1/tasks/unload_all_models?trace=1"
    )
    assert seen["content"] == b""
    assert calls == [("voice", "/v1/tasks/unload_all_models")]


def test_unload_all_models_unknown_model_returns_404(client, monkeypatch):
    _empty_audio_store(monkeypatch)
    response = client.post("/v1/tasks/unload_all_models?model=ghost")
    assert response.status_code == 404


def test_new_task_routes_are_not_mounted_under_v1_audio(client, monkeypatch):
    seen: dict[str, Any] = {}

    async def handler(request, _kwargs):
        seen["url"] = str(request.url)
        return _FakeUpstream(200, json_body={"ok": True})

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/audio/tasks/stream", json={"model": "voice"})
    assert response.status_code == 200
    # The /v1/audio catch-all forwards the path literally: no alias rewrite, no
    # /upstream model scoping, no SSE header injection.
    assert seen["url"] == "http://127.0.0.1:2000/v1/audio/tasks/stream"
    assert "x-accel-buffering" not in response.headers
    assert "cache-control" not in response.headers


def test_existing_task_route_does_not_gain_sse_headers(client, monkeypatch):
    async def handler(_request, _kwargs):
        return _FakeUpstream(200, json_body={"ok": True})

    _install_upstream(monkeypatch, handler)
    response = client.post("/v1/tasks/run", json={"model": "vad", "request": {}})
    assert response.status_code == 200
    assert "x-accel-buffering" not in response.headers
    assert "cache-control" not in response.headers


async def test_stream_response_is_lazy_and_keeps_upstream_sse_headers():
    from backend.routes import audio_openai_proxy as proxy

    pulled: list[bytes] = []
    chunks = [b'data: {"type":"task.stream.event"}\n\n', b"data: [DONE]\n\n"]

    class Upstream:
        status_code = 200
        headers = httpx.Headers(
            {
                "content-type": "text/event-stream",
                "cache-control": "private",
                "x-accel-buffering": "no",
            }
        )

        async def aiter_raw(self):
            for chunk in chunks:
                pulled.append(chunk)
                yield chunk

        async def aclose(self):
            return None

    class Client:
        async def aclose(self):
            return None

    response = proxy._stream_response(Client(), Upstream(), sse=True)
    # Building the response must not consume the upstream stream.
    assert pulled == []
    # Upstream's own SSE headers win over Studio's defaults.
    assert response.headers["cache-control"] == "private"
    assert response.headers["x-accel-buffering"] == "no"

    iterator = response.body_iterator
    assert await iterator.__anext__() == chunks[0]
    assert pulled == chunks[:1]
    assert await iterator.__anext__() == chunks[1]
    await iterator.aclose()


# --- v0.9.5 live audio: chunked request bodies through the /v1/audio catch-all --


@pytest.mark.parametrize(
    "path,query,chunks",
    [
        (
            "/v1/audio/transcriptions/live",
            "model=asr-live&sample_rate=16000&channels=1&sample_format=s16le",
            [b'data: {"type":"transcript.text.delta"}\n\n', b"data: [DONE]\n\n"],
        ),
        (
            "/v1/audio/speech/live",
            "model=s2s-live&return_text=true",
            [b'data: {"type":"speech.text.delta"}\n\n', b"data: [DONE]\n\n"],
        ),
    ],
)
def test_live_audio_routes_forward_chunked_pcm_and_stream_sse(
    client, monkeypatch, path, query, chunks
):
    seen: dict[str, Any] = {}

    async def handler(request, kwargs):
        seen["url"] = str(request.url)
        seen["content"] = kwargs["content"]
        seen["content_length"] = any(key.lower() == "content-length" for key in kwargs["headers"])
        received = []
        async for chunk in kwargs["content"]:
            received.append(chunk)
        seen["received"] = b"".join(received)
        return _FakeUpstream(
            200, content=b"".join(chunks), headers={"content-type": "text/event-stream"}
        )

    _install_upstream(monkeypatch, handler)
    pcm = b"\x00\x01\x02\x03" * 64
    response = client.post(
        f"{path}?{query}",
        content=iter([pcm[:64], pcm[64:]]),
        headers={"content-type": "application/octet-stream"},
    )
    assert response.status_code == 200
    # The catch-all forwards the native path and the model query untouched: the
    # upstream engine reads its model from the query on these routes.
    assert seen["url"] == f"http://127.0.0.1:2000{path}?{query}"
    assert hasattr(seen["content"], "__aiter__")
    assert seen["content_length"] is False
    assert seen["received"] == pcm
    assert response.headers["content-type"].startswith("text/event-stream")


async def test_live_audio_passthrough_is_incrementally_forwarded(monkeypatch):
    """No read-ahead buffering: each client chunk reaches httpx as it arrives."""
    from starlette.requests import Request

    from backend.routes import audio_openai_proxy as proxy

    chunks = [b"pcm-one", b"pcm-two", b"pcm-three"]
    pulled: list[bytes] = []
    pending = list(chunks)
    observed: list[tuple[bytes, tuple[bytes, ...]]] = []

    async def receive():
        if pending:
            chunk = pending.pop(0)
            pulled.append(chunk)
            return {"type": "http.request", "body": chunk, "more_body": True}
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/v1/audio/transcriptions/live",
        "raw_path": b"/v1/audio/transcriptions/live",
        "query_string": b"model=asr-live&sample_rate=16000",
        "root_path": "",
        "headers": [(b"content-type", b"application/octet-stream")],
        "server": ("testserver", 80),
        "client": ("testclient", 50000),
    }
    request = Request(scope, receive)

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def build_request(self, method, url, **kwargs):
            return SimpleNamespace(method=method, url=url, kwargs=kwargs)

        async def send(self, built, stream=False):
            content = built.kwargs["content"]
            assert hasattr(content, "__aiter__"), "body must not be buffered"
            assert "content-length" not in {key.lower() for key in built.kwargs["headers"]}
            async for chunk in content:
                observed.append((chunk, tuple(pulled)))
            return _FakeUpstream(
                200, content=b"data: [DONE]\n\n", headers={"content-type": "text/event-stream"}
            )

        async def aclose(self):
            return None

    monkeypatch.setattr(proxy.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(proxy, "_get_proxy_port", lambda: 2000)

    response = await proxy._passthrough(request, "/v1/audio/transcriptions/live")
    assert response.status_code == 200
    # Starlette may hand over a terminating empty chunk; it carries no audio.
    forwarded = [(chunk, pulled_at_yield) for chunk, pulled_at_yield in observed if chunk]
    assert [chunk for chunk, _ in forwarded] == chunks
    # Consuming chunk N reads only up to chunk N; nothing is read ahead.
    for index, (_, pulled_at_yield) in enumerate(forwarded):
        assert pulled_at_yield == tuple(chunks[: index + 1])
    assert str(request.url) == (
        "http://testserver/v1/audio/transcriptions/live?model=asr-live&sample_rate=16000"
    )
