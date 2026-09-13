"""Audio API transport; adapt only formats the engine cannot currently decode.

The native audio.cpp contract owns request validation and response schemas. v0.7.4
accepts WAV uploads and returns WAV for offline speech; those are the only format
boundaries requiring Studio conversion. SSE and native audio responses stream.
"""

from __future__ import annotations

import json
from functools import partial
from typing import AsyncIterator, Iterable

import anyio
import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.responses import StreamingResponse

from backend.audio_format_convert import (
    AudioConvertError,
    MAX_AUDIO_UPLOAD_BYTES,
    SPEECH_PASSTHROUGH_FORMATS,
    encode_wav_speech_format,
    ensure_wav_bytes_http,
    is_wav_content,
    normalize_speech_response_format,
)
from backend.llama_swap_client import get_proxy_port
from backend.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()
tasks_router = APIRouter()

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}
PASSTHROUGH_TIMEOUT = httpx.Timeout(30.0, read=300.0)
MAX_MULTIPART_FILES = 8
MAX_MULTIPART_FIELDS = 128
MAX_MULTIPART_FIELD_BYTES = 64 * 1024
_conversion_limiter = anyio.CapacityLimiter(2)
_get_proxy_port = get_proxy_port


def _upstream_base() -> str:
    return f"http://127.0.0.1:{_get_proxy_port()}"


def _filter_headers(headers: Iterable[tuple[str, str]], *, multipart: bool = False) -> dict[str, str]:
    values = list(headers)
    excluded = set(HOP_BY_HOP)
    for key, value in values:
        if key.lower() == "connection":
            excluded.update(token.strip().lower() for token in value.split(","))
    if multipart:
        excluded.add("content-type")
    return {key: value for key, value in values if key.lower() not in excluded}


class _UploadTooLarge(MultiPartException):
    pass


def _check_content_length(request: Request) -> None:
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise HTTPException(400, "Invalid Content-Length") from exc
    if length < 0:
        raise HTTPException(400, "Invalid Content-Length")
    if length > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(413, "Audio request exceeds 60 MiB limit")


async def _bounded_request_stream(request: Request, *, multipart: bool = False) -> AsyncIterator[bytes]:
    _check_content_length(request)
    count = 0
    async for chunk in request.stream():
        count += len(chunk)
        if count > MAX_AUDIO_UPLOAD_BYTES:
            # MultiPartParser closes every temporary upload on this exception.
            if multipart:
                raise _UploadTooLarge("Audio request exceeds 60 MiB limit")
            raise HTTPException(413, "Audio request exceeds 60 MiB limit")
        yield chunk


async def _read_body(request: Request) -> bytes:
    return b"".join([chunk async for chunk in _bounded_request_stream(request)])


async def _close_upstream(client: httpx.AsyncClient, upstream: httpx.Response) -> None:
    # Response disconnects cancel the streaming task. Cleanup must survive that
    # cancellation so a busy engine does not retain an abandoned connection.
    with anyio.CancelScope(shield=True):
        try:
            await upstream.aclose()
        finally:
            await client.aclose()


async def _open_upstream(request: Request, upstream_path: str, **kwargs) -> tuple[httpx.AsyncClient, httpx.Response]:
    url = f"{_upstream_base()}{upstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    client = httpx.AsyncClient(timeout=PASSTHROUGH_TIMEOUT, trust_env=False)
    try:
        outgoing = client.build_request(request.method, url, **kwargs)
        upstream = await client.send(outgoing, stream=True)
    except BaseException as exc:
        with anyio.CancelScope(shield=True):
            await client.aclose()
        if isinstance(exc, httpx.RequestError):
            logger.warning("audio proxy upstream error for %s: %s", upstream_path, exc)
            raise HTTPException(502, "Audio upstream is unavailable") from exc
        raise
    return client, upstream


def _stream_response(client: httpx.AsyncClient, upstream: httpx.Response) -> StreamingResponse:
    async def chunks():
        try:
            # Raw bytes preserve Content-Encoding. Decoding while retaining this
            # header makes gzip responses fail at the downstream client.
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await _close_upstream(client, upstream)

    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        headers=_filter_headers(upstream.headers.items()),
        background=BackgroundTask(_close_upstream, client, upstream),
    )


async def _passthrough(request: Request, upstream_path: str, *, body: bytes | None = None) -> Response:
    _check_content_length(request)
    client, upstream = await _open_upstream(
        request, upstream_path,
        headers=_filter_headers(request.headers.items()),
        content=body if body is not None else _bounded_request_stream(request),
    )
    return _stream_response(client, upstream)


async def _resolve_target(model: str, native_path: str):
    from backend.audio_cpp_proxy_routing import resolve_audio_upstream_target

    return await anyio.to_thread.run_sync(partial(resolve_audio_upstream_target, model, native_path))


async def _forward_audio_multipart(request: Request, native_path: str, *, generic_route: bool) -> Response:
    _check_content_length(request)
    parser = MultiPartParser(
        request.headers, _bounded_request_stream(request, multipart=True),
        max_files=MAX_MULTIPART_FILES, max_fields=MAX_MULTIPART_FIELDS,
        max_part_size=MAX_MULTIPART_FIELD_BYTES,
    )
    try:
        form = await parser.parse()
    except _UploadTooLarge as exc:
        raise HTTPException(413, exc.message) from exc
    except MultiPartException as exc:
        raise HTTPException(400, exc.message) from exc
    try:
        upstream_path = native_path
        native_model = None
        if generic_route:
            models = form.getlist("model")
            if len(models) != 1 or not isinstance(models[0], str):
                raise HTTPException(400, "Exactly one model field is required")
            target = await _resolve_target(models[0], native_path)
            upstream_path, native_model = target.path, target.model
        # All parts use httpx's files sequence, including filename=None text
        # parts. This preserves duplicate names and order without a sync body.
        parts = []
        total = 0
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                raw = await value.read(MAX_AUDIO_UPLOAD_BYTES + 1)
                if key == "file":
                    raw, name = await anyio.to_thread.run_sync(
                        partial(ensure_wav_bytes_http, raw, filename=value.filename,
                                content_type=value.content_type),
                        limiter=_conversion_limiter,
                    )
                    part = (name, raw, "audio/wav")
                else:
                    part = (value.filename or "upload", raw, value.content_type)
                total += len(raw)
            else:
                text = native_model if key == "model" and native_model else str(value)
                part = (None, text)
                total += len(text.encode("utf-8"))
            if total > MAX_AUDIO_UPLOAD_BYTES:
                raise HTTPException(413, "Converted audio request exceeds 60 MiB limit")
            parts.append((key, part))
        client, upstream = await _open_upstream(
            request, upstream_path,
            headers=_filter_headers(request.headers.items(), multipart=True), files=parts,
        )
    finally:
        await form.close()
    return _stream_response(client, upstream)


async def _audio_upload(request: Request, native_path: str, *, generic_route: bool = False):
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        return await _forward_audio_multipart(request, native_path, generic_route=generic_route)
    if not generic_route:
        return await _passthrough(request, native_path)
    body = await _read_body(request)
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, "Expected JSON or multipart audio request") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
        raise HTTPException(400, "A model field is required")
    target = await _resolve_target(payload["model"], native_path)
    payload["model"] = target.model
    return await _passthrough(request, target.path, body=json.dumps(payload).encode("utf-8"))


@router.post("/transcriptions")
async def proxy_transcriptions(request: Request):
    return await _audio_upload(request, "/v1/audio/transcriptions")


@router.post("/transcriptions/details")
async def proxy_transcription_details(request: Request):
    return await _audio_upload(request, "/v1/audio/transcriptions/details", generic_route=True)


@router.post("/alignments")
async def proxy_alignments(request: Request):
    return await _audio_upload(request, "/v1/audio/alignments", generic_route=True)


@router.post("/transcriptions/live")
@router.post("/speech/live")
async def proxy_live_audio(request: Request):
    raise HTTPException(
        501,
        "Live audio requires simultaneous request and response streaming. "
        "Connect a native client directly to audio.cpp; Studio's HTTP transport "
        "supports streamed responses after the upload completes.",
    )


def _speech_format_from_body(body: bytes) -> str:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError, TypeError):
        return ""
    return normalize_speech_response_format(payload.get("response_format")) if isinstance(payload, dict) else ""


def _openai_speech_error(status_code: int, message: str) -> Response:
    return Response(
        content=json.dumps({"error": {"message": message, "type": "server_error"}}),
        status_code=status_code, media_type="application/json",
    )


@router.post("/speech")
async def proxy_speech(request: Request):
    body = await _read_body(request)
    response_format = _speech_format_from_body(body)
    client, upstream = await _open_upstream(
        request, "/v1/audio/speech", headers=_filter_headers(request.headers.items()), content=body,
    )
    media_type = upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if (upstream.status_code >= 400 or response_format in SPEECH_PASSTHROUGH_FORMATS
            or media_type not in {"audio/wav", "audio/wave", "audio/x-wav"}):
        return _stream_response(client, upstream)
    # Offline v0.7.4 ignores compressed response formats and returns WAV. If a
    # later engine implements a requested format, it bypasses this fallback.
    try:
        payload = bytearray()
        async for chunk in upstream.aiter_bytes():
            if len(payload) + len(chunk) > MAX_AUDIO_UPLOAD_BYTES:
                raise AudioConvertError("Speech conversion input exceeds 60 MiB limit", status_code=413)
            payload.extend(chunk)
        encoded = bytes(payload)
        if is_wav_content(encoded):
            encoded, media_type = await anyio.to_thread.run_sync(
                partial(encode_wav_speech_format, encoded, response_format),
                limiter=_conversion_limiter,
            )
    except AudioConvertError as exc:
        return _openai_speech_error(exc.status_code, str(exc))
    except httpx.RequestError as exc:
        return _openai_speech_error(502, "Audio upstream response was interrupted")
    finally:
        await _close_upstream(client, upstream)
    headers = _filter_headers(upstream.headers.items())
    for key in list(headers):
        if key.lower() in {"content-type", "content-encoding", "etag", "content-md5", "digest"}:
            del headers[key]
    return Response(encoded, status_code=upstream.status_code, headers=headers, media_type=media_type)


@tasks_router.post("/tasks/run")
async def proxy_tasks_run(request: Request):
    """Keep generic task JSON and llama-swap filters on the native mapped route."""
    return await _passthrough(request, "/audioapi/v1/tasks/run")


@router.api_route(
    "/{rest:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_audio_passthrough(request: Request, rest: str = ""):
    return await _passthrough(request, f"/v1/audio/{rest}" if rest else "/v1/audio")
