"""OpenAI-compatible /v1/audio proxy with non-WAV → WAV conversion for ASR."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import httpx
from fastapi import APIRouter, HTTPException, Request, Response
from starlette.datastructures import UploadFile

from backend.audio_format_convert import ensure_wav_bytes_http
from backend.logging_config import get_logger
from backend.routes.status import _get_proxy_port

logger = get_logger(__name__)

router = APIRouter()

HOP_BY_HOP = {
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
}

PASSTHROUGH_TIMEOUT = httpx.Timeout(30.0, read=300.0)
ASR_TIMEOUT = httpx.Timeout(30.0, read=300.0)


def _upstream_base() -> str:
    return f"http://127.0.0.1:{_get_proxy_port()}"


def _filter_request_headers_for_multipart(
    headers: Iterable[Tuple[str, str]],
) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers:
        lower = key.lower()
        if lower in HOP_BY_HOP or lower == "content-type":
            continue
        out[key] = value
    return out


def _filter_response_headers(headers: httpx.Headers) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for key, value in headers.items():
        if key.lower() in HOP_BY_HOP:
            continue
        out[key] = value
    return out


async def _passthrough(request: Request, upstream_path: str) -> Response:
    url = f"{_upstream_base()}{upstream_path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in HOP_BY_HOP
    }
    body = await request.body()

    try:
        async with httpx.AsyncClient(timeout=PASSTHROUGH_TIMEOUT) as client:
            upstream = await client.request(
                request.method,
                url,
                headers=headers,
                content=body if body else None,
            )
    except httpx.RequestError as exc:
        logger.warning("audio proxy upstream error for %s: %s", upstream_path, exc)
        raise HTTPException(
            status_code=502,
            detail=f"llama-swap unreachable at {_upstream_base()}: {exc}",
        ) from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_filter_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


async def _forward_transcriptions_multipart(request: Request) -> Response:
    form = await request.form()
    # httpx AsyncClient requires form fields as a dict (or Mapping). A list of
    # (key, value) tuples is treated as a sync byte-stream body and raises:
    # RuntimeError: Attempted to send an sync request with an AsyncClient instance
    data: Dict[str, str] = {}
    files: Dict[str, Tuple[str, bytes, str]] = {}

    for key, value in form.multi_items():
        if isinstance(value, UploadFile):
            raw = await value.read()
            wav_bytes, wav_name = ensure_wav_bytes_http(
                raw,
                filename=value.filename,
                content_type=value.content_type,
            )
            files[key] = (wav_name, wav_bytes, "audio/wav")
        else:
            data[key] = str(value)

    if not files:
        raise HTTPException(
            status_code=400,
            detail="multipart transcription requires a file field",
        )

    headers = _filter_request_headers_for_multipart(request.headers.items())
    url = f"{_upstream_base()}/v1/audio/transcriptions"
    if request.url.query:
        url = f"{url}?{request.url.query}"

    try:
        async with httpx.AsyncClient(timeout=ASR_TIMEOUT) as client:
            upstream = await client.post(
                url,
                headers=headers,
                data=data,
                files=files,
            )
    except httpx.RequestError as exc:
        logger.warning("audio transcription proxy upstream error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"llama-swap unreachable at {_upstream_base()}: {exc}",
        ) from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_filter_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


@router.post("/transcriptions")
async def proxy_transcriptions(request: Request):
    """Convert non-WAV multipart uploads, then forward to llama-swap."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "multipart/form-data" in content_type:
        return await _forward_transcriptions_multipart(request)
    return await _passthrough(request, "/v1/audio/transcriptions")


@router.post("/tasks/run")
async def proxy_tasks_run(request: Request):
    """Forward generic audio tasks via llama-swap upstream (avoids browser CORS).

    Body: ``{"model": "<id>", "task": "<name>", "input": {...}}``
    Upstream: ``POST /upstream/{model}/v1/tasks/run`` with ``{"task","input"}``.
    """
    try:
        payload = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="JSON body required") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="JSON object required")

    model = str(payload.get("model") or "").strip()
    task = payload.get("task")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    if task is None or task == "":
        raise HTTPException(status_code=400, detail="task is required")

    input_obj = payload.get("input")
    if input_obj is None:
        input_obj = {}
    if not isinstance(input_obj, dict):
        raise HTTPException(status_code=400, detail="input must be an object")

    upstream_path = f"/upstream/{model}/v1/tasks/run"
    url = f"{_upstream_base()}{upstream_path}"
    headers = _filter_request_headers_for_multipart(request.headers.items())
    headers["Content-Type"] = "application/json"

    try:
        async with httpx.AsyncClient(timeout=PASSTHROUGH_TIMEOUT) as client:
            upstream = await client.post(
                url,
                headers=headers,
                json={"task": task, "input": input_obj},
            )
    except httpx.RequestError as exc:
        logger.warning("audio tasks/run proxy upstream error: %s", exc)
        raise HTTPException(
            status_code=502,
            detail=f"llama-swap unreachable at {_upstream_base()}: {exc}",
        ) from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        headers=_filter_response_headers(upstream.headers),
        media_type=upstream.headers.get("content-type"),
    )


@router.api_route(
    "/{rest:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"],
)
async def proxy_audio_passthrough(request: Request, rest: str = ""):
    """Passthrough other OpenAI audio routes to llama-swap."""
    upstream_path = f"/v1/audio/{rest}" if rest else "/v1/audio"
    return await _passthrough(request, upstream_path)
