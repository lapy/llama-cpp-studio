"""Audio API transport; adapt only formats the engine cannot currently decode.

The native audio.cpp contract owns request validation and response schemas.
The core server accepts WAV uploads and returns WAV for offline speech. Studio
converts other uploads to WAV at the source sample rate, and converts speech
only when the engine still returns WAV for a compressed response_format.
SSE and native audio responses stream.

Live ingest is routed but not usable yet: ``/v1/audio/speech/live`` and
``/v1/audio/transcriptions/live`` reach the catch-all below, yet the
inference-origin dataplane reads the whole request body before forwarding
(``backend/proxy/studio/dataplane.py``: ``StudioDataPlane._handle`` ->
``_read_body`` at L641, then ``StudioDataPlane._proxy`` ->
``build_request(..., content=body)`` at L1569), while the engine's live
handlers require a chunked body. Fixing that belongs in the dataplane; the
contract registry marks both operations ``served=False`` until it does.
"""

from __future__ import annotations

import json
from functools import partial
from typing import AsyncIterator, Iterable
from urllib.parse import parse_qsl, urlencode

import anyio
import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser
from starlette.responses import StreamingResponse

from backend.audio.format_convert import (
    AudioConvertError,
    MAX_AUDIO_UPLOAD_BYTES,
    SPEECH_PASSTHROUGH_FORMATS,
    audio_limit_detail,
    encode_wav_speech_format,
    ensure_wav_bytes_http,
    is_wav_content,
    normalize_speech_response_format,
)
from backend.proxy.client import get_proxy_port
from backend.engines.audio_cpp.integration import (
    NATIVE_AUDIO_BASE,
    TASKS_RUN_ALIAS,
    operation_by_id,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)
router = APIRouter()
tasks_router = APIRouter()
task_admin_router = APIRouter()
batches_router = APIRouter()

HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
PASSTHROUGH_TIMEOUT = httpx.Timeout(30.0, read=300.0)
MAX_MULTIPART_FILES = 8
MAX_BATCH_FILES = 32
MAX_MULTIPART_FIELDS = 128
MAX_MULTIPART_FIELD_BYTES = 64 * 1024
# Added to streamed task responses unless upstream already named them, so an
# intermediary between Studio and the client does not buffer the event stream.
# Upstream already sends X-Accel-Buffering: no on its own SSE routes.
SSE_RESPONSE_HEADERS = {
    "cache-control": "no-cache",
    "x-accel-buffering": "no",
}
_conversion_limiter = anyio.CapacityLimiter(2)
_get_proxy_port = get_proxy_port


def _upstream_base() -> str:
    return f"http://127.0.0.1:{_get_proxy_port()}"


def _native_operation(operation_id: str):
    """Contract operation for an id (registry is declarative; None = coding error)."""
    operation = operation_by_id(operation_id)
    if operation is None:  # pragma: no cover
        raise HTTPException(500, f"Unknown audio.cpp operation {operation_id!r}")
    return operation


def _native_path(operation_id: str) -> str:
    """Native engine path for an operation id, resolved from the contract registry."""
    return _native_operation(operation_id).native_path


def _filter_headers(
    headers: Iterable[tuple[str, str]], *, multipart: bool = False
) -> dict[str, str]:
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
        raise HTTPException(413, audio_limit_detail("Audio request"))


async def _bounded_request_stream(
    request: Request, *, multipart: bool = False
) -> AsyncIterator[bytes]:
    _check_content_length(request)
    count = 0
    async for chunk in request.stream():
        count += len(chunk)
        if count > MAX_AUDIO_UPLOAD_BYTES:
            # MultiPartParser closes every temporary upload on this exception.
            if multipart:
                raise _UploadTooLarge(audio_limit_detail("Audio request"))
            raise HTTPException(413, audio_limit_detail("Audio request"))
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


async def _open_upstream(
    request: Request, upstream_path: str, *, query: str | None = None, **kwargs
) -> tuple[httpx.AsyncClient, httpx.Response]:
    from backend.inference_auth import internal_service_key

    # ``query=None`` keeps the client query string byte-for-byte. An explicit
    # string replaces it, which is how a Studio-only routing pin such as
    # ``?model=`` is kept from reaching the engine.
    forwarded_query = request.url.query if query is None else query
    url = f"{_upstream_base()}{upstream_path}"
    if forwarded_query:
        url = f"{url}?{forwarded_query}"
    client = httpx.AsyncClient(timeout=PASSTHROUGH_TIMEOUT, trust_env=False)
    try:
        headers = dict(kwargs.pop("headers", {}) or {})
        headers["X-Studio-Service-Key"] = internal_service_key()
        kwargs["headers"] = headers
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


def _stream_response(
    client: httpx.AsyncClient, upstream: httpx.Response, *, sse: bool = False
) -> StreamingResponse:
    async def chunks():
        try:
            # Raw bytes preserve Content-Encoding. Decoding while retaining this
            # header makes gzip responses fail at the downstream client.
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await _close_upstream(client, upstream)

    headers = _filter_headers(upstream.headers.items())
    if sse:
        present = {key.lower() for key in headers}
        for key, value in SSE_RESPONSE_HEADERS.items():
            if key not in present:
                headers[key] = value
    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        headers=headers,
        background=BackgroundTask(_close_upstream, client, upstream),
    )


async def _passthrough(
    request: Request,
    upstream_path: str,
    *,
    body: bytes | None = None,
    query: str | None = None,
    sse: bool = False,
) -> Response:
    _check_content_length(request)
    client, upstream = await _open_upstream(
        request,
        upstream_path,
        query=query,
        headers=_filter_headers(request.headers.items()),
        content=body if body is not None else _bounded_request_stream(request),
    )
    return _stream_response(client, upstream, sse=sse)


async def _resolve_target(model: str, native_path: str):
    from backend.engines.audio_cpp.proxy_routing import resolve_audio_upstream_target

    return await anyio.to_thread.run_sync(
        partial(resolve_audio_upstream_target, model, native_path)
    )


def _json_object(body: bytes, *, message: str) -> dict:
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(400, message) from exc
    if not isinstance(payload, dict):
        raise HTTPException(400, message)
    return payload


def _query_without_model(request: Request) -> str:
    """Client query string minus the Studio-only ``model`` routing pin.

    The pin selects a Studio engine instance; forwarding it to the engine would
    expose a Studio alias the engine does not know and can change what a route
    such as ``/v1/audio/voices?model=`` (which reads ``model``) does.
    """
    pairs = parse_qsl(request.url.query, keep_blank_values=True)
    return urlencode([(key, value) for key, value in pairs if key.lower() != "model"])


async def _json_upstream_target(request: Request, operation_id: str):
    """Read a bounded JSON body, resolve its ``model``, rewrite it natively."""
    body = await _read_body(request)
    payload = _json_object(body, message="Expected a JSON audio task request")
    model = payload.get("model")
    if not isinstance(model, str) or not model.strip():
        raise HTTPException(400, "A model field is required")
    target = await _resolve_target(model, _native_path(operation_id))
    payload["model"] = target.model
    return target, json.dumps(payload).encode("utf-8")


async def _forward_audio_multipart(
    request: Request,
    native_path: str,
    *,
    generic_route: bool,
    max_files: int = MAX_MULTIPART_FILES,
) -> Response:
    _check_content_length(request)
    parser = MultiPartParser(
        request.headers,
        _bounded_request_stream(request, multipart=True),
        max_files=max_files,
        max_fields=MAX_MULTIPART_FIELDS,
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
                        partial(
                            ensure_wav_bytes_http,
                            raw,
                            filename=value.filename,
                            content_type=value.content_type,
                        ),
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
                raise HTTPException(413, audio_limit_detail("Converted audio request"))
            parts.append((key, part))
        client, upstream = await _open_upstream(
            request,
            upstream_path,
            headers=_filter_headers(request.headers.items(), multipart=True),
            files=parts,
        )
    finally:
        await form.close()
    return _stream_response(client, upstream)


async def _audio_upload(
    request: Request,
    native_path: str,
    *,
    generic_route: bool = False,
    max_files: int = MAX_MULTIPART_FILES,
):
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" in content_type:
        return await _forward_audio_multipart(
            request,
            native_path,
            generic_route=generic_route,
            max_files=max_files,
        )
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
    return await _audio_upload(request, _native_path("transcriptions"))


@router.post("/transcriptions/details")
async def proxy_transcription_details(request: Request):
    return await _audio_upload(request, _native_path("transcriptions_details"), generic_route=True)


@router.post("/alignments")
async def proxy_alignments(request: Request):
    return await _audio_upload(request, _native_path("alignments"), generic_route=True)


def _speech_format_from_body(body: bytes) -> str:
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError, TypeError):
        return ""
    return (
        normalize_speech_response_format(payload.get("response_format"))
        if isinstance(payload, dict)
        else ""
    )


def _openai_speech_error(status_code: int, message: str) -> Response:
    return Response(
        content=json.dumps({"error": {"message": message, "type": "server_error"}}),
        status_code=status_code,
        media_type="application/json",
    )


@router.post("/speech")
async def proxy_speech(request: Request):
    body = await _read_body(request)
    response_format = _speech_format_from_body(body)
    client, upstream = await _open_upstream(
        request,
        _native_path("speech"),
        headers=_filter_headers(request.headers.items()),
        content=body,
    )
    media_type = upstream.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if (
        upstream.status_code >= 400
        or response_format in SPEECH_PASSTHROUGH_FORMATS
        or media_type not in {"audio/wav", "audio/wave", "audio/x-wav"}
    ):
        return _stream_response(client, upstream)
    # The engine still returns WAV when it ignores a compressed response_format.
    # A response whose content type is already the requested format is streamed
    # above and does not pass through this conversion.
    try:
        payload = bytearray()
        async for chunk in upstream.aiter_bytes():
            if len(payload) + len(chunk) > MAX_AUDIO_UPLOAD_BYTES:
                raise AudioConvertError(
                    audio_limit_detail("Speech conversion input"), status_code=413
                )
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
    return Response(
        encoded, status_code=upstream.status_code, headers=headers, media_type=media_type
    )


@batches_router.post("/batches/transcriptions")
async def proxy_batch_transcriptions(request: Request):
    """Native multi-file batch. Studio proxy has no route, so use /upstream."""
    return await _audio_upload(
        request,
        _native_path("batch_transcriptions"),
        generic_route=True,
        max_files=MAX_BATCH_FILES,
    )


@tasks_router.post("/tasks/run")
async def proxy_tasks_run(request: Request):
    """Keep generic task JSON and Studio proxy filters on the native mapped route."""
    return await _passthrough(request, TASKS_RUN_ALIAS)


@tasks_router.post("/tasks/stream")
async def proxy_tasks_stream(request: Request):
    """Generic streaming task (upstream app/server/api.md @ v0.9.5).

    The body is the /v1/tasks/run body; ``model`` is rewritten to the stable
    native id and the upstream response is streamed unbuffered, so
    ``stream_format: "sse"`` reaches the client event by event.
    """
    target, body = await _json_upstream_target(request, "tasks_stream")
    return await _passthrough(request, target.path, body=body, sse=True)


@tasks_router.post("/tasks/batch")
async def proxy_tasks_batch(request: Request):
    """Native generic batch (upstream app/server/api.md @ v0.9.5).

    ``requests`` is forwarded untouched; the engine owns its shape and answers
    with SSE, so the response is streamed unbuffered with SSE headers.
    """
    target, body = await _json_upstream_target(request, "tasks_batch")
    return await _passthrough(request, target.path, body=body, sse=True)


@task_admin_router.post("/tasks/unload_models")
async def proxy_tasks_unload_models(request: Request):
    """Release engine memory for named models (upstream api.md @ v0.9.5).

    Upstream takes ``{"model_ids": ["<engine id>", ...]}``. Studio resolves
    every id to its stable native id and pins the engine instance by the first
    one; when the list is empty (or to choose the instance explicitly) an
    ``?model=`` pin does the same job. The pin is Studio routing metadata and
    never reaches the engine.
    """
    body = await _read_body(request)
    payload = _json_object(body, message="Expected a JSON unload request")
    raw_ids = payload.get("model_ids")
    if not isinstance(raw_ids, list):
        raise HTTPException(400, "request requires a 'model_ids' string array")
    stable_ids = []
    pinned = None
    for item in raw_ids:
        if not isinstance(item, str) or not item.strip():
            raise HTTPException(400, "each element of 'model_ids' must be a non-empty string")
        target = await _resolve_target(item, _native_path("tasks_unload_models"))
        pinned = pinned or target
        stable_ids.append(target.model)
    pin = str(request.query_params.get("model") or "").strip()
    if pin:
        pinned = await _resolve_target(pin, _native_path("tasks_unload_models"))
    if pinned is None:
        raise HTTPException(400, "model_ids must name at least one audio model, or pass ?model=")
    payload["model_ids"] = stable_ids
    return await _passthrough(
        request,
        pinned.path,
        body=json.dumps(payload).encode("utf-8"),
        query=_query_without_model(request),
    )


@task_admin_router.post("/tasks/unload_all_models")
async def proxy_tasks_unload_all_models(request: Request):
    """Release all engine memory on one instance (upstream api.md @ v0.9.5).

    Upstream takes no body and names no model. Studio's engine routes are
    model-scoped, so the route requires a concrete ``?model=`` pin to select
    the engine instance; the pin is Studio routing metadata and is stripped
    before forwarding.
    """
    pin = str(request.query_params.get("model") or "").strip()
    if not pin:
        raise HTTPException(
            400,
            "This audio endpoint requires a model query parameter to select the engine instance",
        )
    # Upstream ignores a body here; forward whatever the client sent under the
    # same bounded read as every other audio transport.
    body = await _read_body(request)
    target = await _resolve_target(pin, _native_path("tasks_unload_all_models"))
    return await _passthrough(
        request,
        target.path,
        body=body,
        query=_query_without_model(request),
    )


async def proxy_audio_passthrough(request: Request, rest: str = ""):
    # Live-ingest paths land here too, but this shim is not enough for them:
    # the proxy dataplane buffers the request body before forwarding
    # (backend/proxy/studio/dataplane.py: _read_body at L641, content=body at
    # L1569), so the engine never sees the chunked PCM its live handlers
    # require. The registry keeps speech_live/transcriptions_live served=False.
    return await _passthrough(request, f"{NATIVE_AUDIO_BASE}/{rest}" if rest else NATIVE_AUDIO_BASE)


for _method in ("GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"):
    router.add_api_route(
        "/{rest:path}",
        proxy_audio_passthrough,
        methods=[_method],
        operation_id=f"proxy_audio_passthrough_{_method.lower()}",
        name=f"proxy_audio_passthrough_{_method.lower()}",
    )
