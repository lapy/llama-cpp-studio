"""Redirect Studio's audio.cpp UI path to llama-swap ``/upstream/{model}/``."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import RedirectResponse

from backend.audio_cpp_ui_rewrite import llama_swap_upstream_prefix
from backend.llama_swap_client import get_proxy_port

router = APIRouter()
_get_proxy_port = get_proxy_port


def _request_hostname(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-host")
    host = (forwarded or request.headers.get("host") or request.url.hostname or "").split(",")[
        0
    ].strip()
    if host.startswith("[") and "]" in host:
        return host.split("]", 1)[0] + "]"
    if host.count(":") == 1:
        return host.rsplit(":", 1)[0]
    return host or "localhost"


def llama_swap_public_origin(request: Request) -> str:
    forwarded_proto = request.headers.get("x-forwarded-proto")
    scheme = (
        forwarded_proto.split(",")[0].strip()
        if forwarded_proto
        else (request.url.scheme or "http")
    )
    hostname = _request_hostname(request)
    return f"{scheme}://{hostname}:{_get_proxy_port()}"


def llama_swap_upstream_url(request: Request, model_id: str, rest: str = "") -> str:
    prefix = llama_swap_upstream_prefix(model_id)
    leftover = str(rest or "").lstrip("/")
    path = f"{prefix}/{leftover}" if leftover else f"{prefix}/"
    url = f"{llama_swap_public_origin(request)}{path}"
    if request.url.query:
        url = f"{url}?{request.url.query}"
    return url


@router.api_route("/{model}", methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def audio_ui_index_redirect(model: str, request: Request):
    model_id = str(model or "").strip().strip("/")
    if not model_id:
        raise HTTPException(status_code=404, detail="model id required")
    return RedirectResponse(url=llama_swap_upstream_url(request, model_id), status_code=307)


@router.api_route(
    "/{model}/",
    methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def audio_ui_root_redirect(model: str, request: Request):
    return await audio_ui_index_redirect(model, request)


@router.api_route(
    "/{model}/{rest:path}",
    methods=["GET", "HEAD", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
)
async def audio_ui_redirect(model: str, request: Request, rest: str = ""):
    model_id = str(model or "").strip().strip("/")
    if not model_id:
        raise HTTPException(status_code=404, detail="model id required")
    return RedirectResponse(
        url=llama_swap_upstream_url(request, model_id, rest),
        status_code=307,
    )
