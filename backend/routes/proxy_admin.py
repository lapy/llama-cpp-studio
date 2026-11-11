"""Studio control-plane endpoints for proxy configuration and observation."""

import json
import os
import asyncio
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from backend.access_policy import proxy_listen_host
from backend.concurrency import run_blocking
from backend.data_store import get_store
from backend.inference_auth import rotate_key
from backend.proxy.studio.manager import configured_proxy_backend, get_studio_proxy_manager

router = APIRouter()


class ProxySettingsBody(BaseModel):
    proxy_port: int = Field(default=2000, ge=1, le=65535)
    idle_ttl_seconds: float | None = Field(default=None, ge=0)
    max_body_bytes: int | None = Field(default=None, ge=1)
    allow_anonymous: bool = True
    max_loaded_models: int | None = Field(default=None, ge=1, le=1000)
    global_concurrency: int | None = Field(default=None, ge=1, le=100000)
    per_model_concurrency: int | None = Field(default=None, ge=1, le=100000)
    queue_timeout_seconds: float = Field(default=30, gt=0, le=3600)
    request_timeout_seconds: float | None = Field(default=None, gt=0, le=86400)
    readiness_timeout_seconds: float = Field(default=2700, gt=0, le=86400)
    drain_timeout_seconds: float = Field(default=30, gt=0, le=3600)
    max_queue_size: int = Field(default=100, ge=1, le=100000)
    stream_idle_timeout_seconds: float | None = Field(default=None, gt=0, le=86400)
    pool_timeout_seconds: float = Field(default=10, gt=0, le=3600)
    max_connections: int = Field(default=200, ge=1, le=100000)
    max_keepalive_connections: int = Field(default=40, ge=1, le=100000)
    history_retention_days: int = Field(default=7, ge=1, le=365)
    history_max_rows: int = Field(default=100000, ge=100, le=1000000)
    default_output_tokens: int = Field(default=4096, ge=1, le=1000000)
    default_model: str | None = None
    routing_aliases: dict[str, str | list[str]] = Field(default_factory=dict)


def _settings_payload() -> dict[str, Any]:
    settings = get_store().get_settings() or {}
    proxy = settings.get("proxy") if isinstance(settings.get("proxy"), dict) else {}
    return {
        **{
            name: proxy.get(name, default)
            for name, default in {
                "max_queue_size": 100,
                "stream_idle_timeout_seconds": None,
                "pool_timeout_seconds": 10,
                "max_connections": 200,
                "max_keepalive_connections": 40,
                "history_retention_days": 7,
                "history_max_rows": 100000,
                "default_output_tokens": 4096,
            }.items()
        },
        "backend": configured_proxy_backend(),
        "idle_ttl_seconds": proxy.get("idle_ttl_seconds"),
        "max_body_bytes": proxy.get("max_body_bytes"),
        "proxy_port": settings.get("proxy_port", 2000),
        "allow_anonymous": bool(proxy.get("allow_anonymous", True)),
        "max_loaded_models": proxy.get("max_loaded_models"),
        "global_concurrency": proxy.get("global_concurrency"),
        "per_model_concurrency": proxy.get("per_model_concurrency"),
        "queue_timeout_seconds": proxy.get("queue_timeout_seconds", 30),
        "request_timeout_seconds": proxy.get("request_timeout_seconds"),
        "readiness_timeout_seconds": proxy.get("readiness_timeout_seconds", 2700),
        "drain_timeout_seconds": proxy.get("drain_timeout_seconds", 30),
        "default_model": proxy.get("default_model"),
        "routing_aliases": proxy.get("routing_aliases", {}),
    }


@router.get("")
async def get_proxy_admin():
    from backend.routes.status import _public_inference_url

    settings = _settings_payload()
    studio = get_studio_proxy_manager()
    running: list[dict[str, Any]] = []
    activity: list[dict[str, Any]] = []
    if settings["backend"] == "studio":
        running = studio.registry.snapshot()
        activity = studio.registry.activities.values()[-100:]
    studio_running = studio.running
    active_backend = "studio" if studio_running else ""
    from backend.proxy.studio.accounting import get_accounting

    accounting = get_accounting()
    await asyncio.to_thread(
        accounting.flush, settings["history_retention_days"], settings["history_max_rows"]
    )
    history, summary, catalog = await asyncio.gather(
        asyncio.to_thread(accounting.query),
        asyncio.to_thread(accounting.summary),
        asyncio.to_thread(studio._catalog),
    )
    from backend.engines.capabilities import ENGINE_CAPABILITIES
    from backend.proxy.studio.observation import METRIC_SEMANTICS, SCHEMA_VERSION

    return {
        "summary": summary,
        "request_total": history["total"],
        "observation_schema_version": SCHEMA_VERSION,
        "metric_semantics": METRIC_SEMANTICS,
        "engine_capabilities": {
            name: capabilities.as_dict() for name, capabilities in ENGINE_CAPABILITIES.items()
        },
        "settings": settings,
        "active_backend": active_backend,
        "routing_revision": getattr(getattr(studio, "_routing", None), "revision", None),
        "restart_required": studio.runtime_restart_required(settings)
        if studio_running and hasattr(studio, "runtime_restart_required")
        else False,
        "configured_backend": settings["backend"],
        "proxy_running": studio_running,
        "studio_running": studio_running,
        "running": running,
        "activity": activity,
        "requests": history["requests"] if settings["backend"] == "studio" else [],
        "catalog": catalog,
        "network": {
            "listen_host": proxy_listen_host(),
            # The bound port of the running proxy if any; otherwise the saved
            # one (it binds here after the next start/switch).
            "port": int(studio.port if studio_running else settings["proxy_port"]),
            "public_url": _public_inference_url(),
            "listen_source": (
                "STUDIO_PROXY_LISTEN"
                if os.getenv("STUDIO_PROXY_LISTEN", "").strip()
                else "container default"
                if os.path.exists("/app/data")
                else "loopback default"
            ),
            "tls_note": "Terminate TLS at a reverse proxy; Studio serves this port as plain HTTP.",
        },
    }


@router.put("/settings")
async def put_proxy_settings(body: ProxySettingsBody):
    backend = "studio"
    if body.max_keepalive_connections > body.max_connections:
        raise HTTPException(
            status_code=400, detail="Keepalive connections cannot exceed total connections"
        )
    before = _settings_payload()
    proxy: dict[str, Any] = body.model_dump()
    proxy_port = proxy.pop("proxy_port")
    proxy["backend"] = backend
    if any(not name.strip() for name in body.routing_aliases):
        raise HTTPException(status_code=400, detail="routing aliases cannot be empty")
    await run_blocking(get_store().update_settings, {"proxy": proxy, "proxy_port": proxy_port})
    after = _settings_payload()
    changed = {name for name in after if before.get(name) != after[name]}
    live_fields = changed & {"routing_aliases", "default_model", "allow_anonymous"}
    restart_required = bool(changed - live_fields)
    routing_revision = None
    manager = get_studio_proxy_manager()
    if hasattr(manager, "apply_routing"):
        routing_revision = manager.apply_routing(after)
        if manager.running:
            restart_required = manager.runtime_restart_required(after)
    return {
        "settings": after,
        "restart_required": restart_required,
        "applied_live_fields": sorted(live_fields),
        "routing_revision": routing_revision,
    }


_LOG_LEVELS = ("error", "warn", "info")


def _filter_log_lines(lines: list[str], search: str, level: str) -> list[str]:
    if search:
        needle = search.lower()
        lines = [line for line in lines if needle in line.lower()]
    if level == "error":
        lines = [line for line in lines if "error" in line.lower() or " stderr:" in line.lower()]
    elif level == "warn":
        lines = [line for line in lines if "error" in line.lower() or "warn" in line.lower()]
    return lines


@router.get("/models/{model_id:path}/logs")
async def get_proxy_model_logs(
    model_id: str,
    search: str = "",
    level: str = "",
    download: bool = False,
    limit: int = 5000,
):
    if level and level not in _LOG_LEVELS:
        raise HTTPException(status_code=400, detail="level must be error, warn, or info")
    model_id = unquote(model_id)
    generation = get_studio_proxy_manager().registry.generation(model_id)
    if generation is None:
        raise HTTPException(status_code=404, detail="Model is not running")
    lines = _filter_log_lines(list(generation.logs.values()), search=search, level=level)[
        -max(1, min(limit, 100_000)) :
    ]
    if download:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in model_id)
        return PlainTextResponse(
            "\n".join(lines) + ("\n" if lines else ""),
            headers={"content-disposition": f'attachment; filename="{safe}.log"'},
        )
    return {"model": model_id, "lines": lines, "sequence": generation.logs.last_sequence}


@router.get("/models/{model_id:path}/logs/stream")
async def stream_proxy_model_logs(model_id: str, request: Request):
    """Stream live engine output over SSE, replaying from the ring cursor.

    The browser sends ``Last-Event-ID`` automatically after a reconnect; the
    explicit ``last_event_id`` query is used for the first connection so a
    freshly opened panel does not receive the lines it already fetched.
    """
    model_id = unquote(model_id)
    generation = get_studio_proxy_manager().registry.generation(model_id)
    if generation is None:
        raise HTTPException(status_code=404, detail="Model is not running")
    cursor = (
        request.headers.get("last-event-id") or request.query_params.get("last_event_id") or "0"
    )
    try:
        sequence = max(0, int(cursor))
    except (TypeError, ValueError):
        sequence = 0

    async def events():
        yield "retry: 1000\n: connected\n\n"
        # Re-send the cursor line as well: an inline terminal redraw keeps its
        # sequence number, so a reconnect must be able to replace the last
        # value it saw instead of missing the updated progress line.
        replay_from = max(0, sequence - 1)
        async for item in generation.logs.subscribe(replay_from, heartbeat_seconds=15.0):
            if item is None:
                yield ": heartbeat\n\n"
                continue
            payload = json.dumps({"line": item.value, "seq": item.sequence})
            yield f"id: {item.sequence}\nevent: log\ndata: {payload}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.delete("/models/{model_id:path}/logs", status_code=204)
async def clear_proxy_model_logs(model_id: str):
    model_id = unquote(model_id)
    generation = get_studio_proxy_manager().registry.generation(model_id)
    if generation is None:
        raise HTTPException(status_code=404, detail="Model is not running")
    generation.logs.clear()
    return None


@router.delete("/models/{model_id:path}/instance")
async def remove_proxy_model_instance(model_id: str):
    """Remove a failed model instance and clear the status it contributed."""
    from backend.proxy.runtime_observation import forget_runtime_model

    model_id = unquote(model_id)
    registry = get_studio_proxy_manager().registry
    if registry.generation(model_id) is None:
        raise HTTPException(status_code=404, detail="Model instance is not retained")
    if not registry.remove_failed_instance(model_id):
        raise HTTPException(
            status_code=409,
            detail="Only a failed engine instance can be removed",
        )
    forget_runtime_model(model_id)
    return {"model": model_id, "removed": True}


@router.post("/models/{model_id:path}/unload")
async def unload_proxy_model(model_id: str):
    model_id = unquote(model_id)
    if not await get_studio_proxy_manager().registry.stop(model_id):
        raise HTTPException(status_code=404, detail="Model is not running")
    return {"model": model_id, "stopped": True}


@router.post("/models/{model_id:path}/preload")
async def preload_proxy_model(model_id: str, warm: bool = False):
    """Load a model now instead of on first request.

    ``warm`` additionally issues a tiny best-effort completion so the engine
    allocates its KV cache and JITs sampling paths before real traffic.
    """
    model_id = unquote(model_id)
    manager = get_studio_proxy_manager()
    try:
        generation = await manager.registry.acquire(model_id)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    warmed = False
    if warm:
        warmed = await _warm_generation(generation)
    manager.registry.release(generation)
    return {"model": model_id, "ready": generation.ready, "warmed": warmed}


async def _warm_generation(generation) -> bool:
    """One-token completion through the loopback engine; failures are silent."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
            response = await client.post(
                f"{generation.upstream}/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "warmup"}],
                    "max_tokens": 1,
                },
            )
        return response.status_code < 400
    except Exception:
        return False


@router.post("/models/{model_id:path}/reload")
async def reload_proxy_model(model_id: str):
    model_id = unquote(model_id)
    manager = get_studio_proxy_manager()
    await manager.registry.stop(model_id)
    try:
        generation = await manager.registry.acquire(model_id)
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    manager.registry.release(generation)
    return {"model": model_id, "ready": generation.ready}


@router.post("/lifecycle/{action}")
async def proxy_lifecycle(action: str):
    """Control the native listener and drain engines when settings are reapplied."""
    backend = "studio"
    if action not in {"start", "stop", "restart", "apply"}:
        raise HTTPException(status_code=404, detail="Unknown lifecycle action")
    manager = get_studio_proxy_manager()
    if action in {"start", "apply"}:
        # Rebuild the runtime so new limits and timeouts take effect.
        await manager.restart_proxy() if action == "apply" else await manager.start_proxy()
    elif action == "stop":
        await manager.stop_proxy()
    else:
        await manager.restart_proxy()
    running = bool(
        manager.running
        if hasattr(manager, "running")
        else getattr(manager, "process", None) and manager.process.poll() is None
    )
    return {"action": action, "backend": backend, "running": running}


@router.post("/stats/reset")
async def reset_proxy_stats():
    from backend.proxy.studio.accounting import get_accounting

    get_studio_proxy_manager().registry.reset_stats()
    await asyncio.to_thread(get_accounting().reset_stats)
    return {"reset": True}


@router.get("/requests")
async def get_proxy_requests(
    limit: int = Query(default=100, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    model: str = "",
    key_id: str = "",
    state: str = "",
    status: int | None = None,
    outcome: str = Query(default="", pattern=r"^(|4xx|5xx|errors)$"),
    search: str = "",
):
    from backend.proxy.studio.accounting import get_accounting

    accounting = get_accounting()
    settings = _settings_payload()
    await asyncio.to_thread(
        accounting.flush, settings["history_retention_days"], settings["history_max_rows"]
    )
    return await asyncio.to_thread(
        accounting.query,
        limit=limit,
        offset=offset,
        model=model,
        key_id=key_id,
        state=state,
        status=status,
        outcome=outcome,
        search=search,
    )
