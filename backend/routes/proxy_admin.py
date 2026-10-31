"""Studio control-plane endpoints for proxy configuration and observation."""

import os
import asyncio
from typing import Any
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.access_policy import proxy_listen_host
from backend.data_store import get_store
from backend.inference_auth import rotate_key
from backend.proxy.studio.manager import configured_proxy_backend, get_studio_proxy_manager

router = APIRouter()


class ProxySettingsBody(BaseModel):
    backend: str
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
    from backend.proxy.llama_swap.manager import get_llama_swap_manager
    from backend.routes.status import _public_inference_url

    settings = _settings_payload()
    studio = get_studio_proxy_manager()
    running: list[dict[str, Any]] = []
    activity: list[dict[str, Any]] = []
    if settings["backend"] == "studio":
        running = studio.registry.snapshot()
        activity = studio.registry.activities.values()[-100:]
    studio_running = studio.running
    swap_running = False
    try:
        manager = get_llama_swap_manager()
        swap_running = bool(manager.process and manager.process.poll() is None)
    except Exception:
        swap_running = False
    if studio_running:
        active_backend = "studio"
    elif swap_running:
        active_backend = "llama_swap"
    else:
        active_backend = ""
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
    return {
        "summary": summary,
        "request_total": history["total"],
        "settings": settings,
        "active_backend": active_backend,
        "routing_revision": getattr(getattr(studio, "_routing", None), "revision", None),
        "restart_required": studio.runtime_restart_required(settings)
        if studio_running and hasattr(studio, "runtime_restart_required")
        else False,
        "configured_backend": settings["backend"],
        "proxy_running": studio_running or swap_running,
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
    backend = body.backend.strip().lower()
    if backend not in {"llama_swap", "studio"}:
        raise HTTPException(status_code=400, detail="backend must be llama_swap or studio")
    if body.max_keepalive_connections > body.max_connections:
        raise HTTPException(
            status_code=400, detail="Keepalive connections cannot exceed total connections"
        )
    before = _settings_payload()
    proxy: dict[str, Any] = body.model_dump()
    proxy_port = proxy.pop("proxy_port")
    proxy["backend"] = backend
    aliases = proxy.get("routing_aliases") or {}
    if any(not str(alias).strip() for alias in aliases):
        raise HTTPException(status_code=400, detail="routing aliases cannot be empty")
    get_store().update_settings({"proxy": proxy, "proxy_port": proxy_port})
    after = _settings_payload()
    changed = {name for name in after if before.get(name) != after[name]}
    live_fields = changed & {"routing_aliases", "default_model", "allow_anonymous"}
    restart_required = bool(changed - live_fields)
    routing_revision = None
    if backend == "studio":
        manager = get_studio_proxy_manager()
        if hasattr(manager, "apply_routing"):
            routing_revision = manager.apply_routing(after)
            if manager.running:
                restart_required = manager.runtime_restart_required(after)
    else:
        # Legacy reload semantics are owned by llama-swap.
        restart_required = bool(changed)
        live_fields = set()
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
    if configured_proxy_backend() != "studio":
        raise HTTPException(status_code=409, detail="Native proxy logs require the Studio backend")
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
    return {"model": model_id, "lines": lines}


@router.delete("/models/{model_id:path}/logs", status_code=204)
async def clear_proxy_model_logs(model_id: str):
    if configured_proxy_backend() != "studio":
        raise HTTPException(status_code=409, detail="Native proxy logs require the Studio backend")
    model_id = unquote(model_id)
    generation = get_studio_proxy_manager().registry.generation(model_id)
    if generation is None:
        raise HTTPException(status_code=404, detail="Model is not running")
    generation.logs.clear()
    return None


@router.post("/models/{model_id:path}/unload")
async def unload_proxy_model(model_id: str):
    if configured_proxy_backend() != "studio":
        raise HTTPException(status_code=409, detail="Native unload requires the Studio backend")
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
    if configured_proxy_backend() != "studio":
        raise HTTPException(status_code=409, detail="Preload requires the Studio backend")
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
    if configured_proxy_backend() != "studio":
        # Without the guard, a reload in legacy mode would make the native
        # registry launch a second engine llama-swap knows nothing about.
        raise HTTPException(status_code=409, detail="Native reload requires the Studio backend")
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
    """Start/stop/restart the *configured* proxy without restarting Studio.

    Works for both backends: the native Studio proxy restarts in-process and
    re-reads its settings (config apply), while llama-swap restarts its
    subprocess. Running models survive a Studio-proxy restart only via the
    llama-swap path's own reload semantics; the native path deliberately
    drains and re-launches engines so applied limits and timeouts take hold.
    """
    backend = configured_proxy_backend()
    if action not in {"start", "stop", "restart", "apply"}:
        raise HTTPException(status_code=404, detail="Unknown lifecycle action")
    if backend == "studio":
        manager = get_studio_proxy_manager()
    else:
        from backend.proxy.llama_swap.manager import get_llama_swap_manager

        manager = get_llama_swap_manager()
    if action in {"start", "apply"}:
        # "apply" re-reads settings: for the native proxy that means a full
        # restart cycle so new limits/timeouts bind; llama-swap applies via
        # its own process restart (its global reload caveat stays legacy).
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


@router.post("/switch")
async def proxy_switch(body: dict):
    """Durable backend change plus controlled switch, in one operation.

    Saves proxy.backend, stops the proxy that is currently active, and starts
    the newly configured one. Engine processes of the previous backend stay
    owned by that backend's lifecycle; the switch is explicit about it.
    """
    backend = str(body.get("backend") or "").strip().lower()
    if backend not in {"llama_swap", "studio"}:
        raise HTTPException(status_code=400, detail="backend must be llama_swap or studio")
    from backend.proxy.llama_swap.manager import get_llama_swap_manager

    settings = get_store().get_settings() or {}
    proxy = settings.get("proxy") if isinstance(settings.get("proxy"), dict) else {}
    get_store().update_settings({"proxy": {**proxy, "backend": backend}})
    stopped: list[str] = []
    studio = get_studio_proxy_manager()
    if studio.running:
        await studio.stop_proxy()
        stopped.append("studio")
    swap = get_llama_swap_manager()
    if swap.process and swap.process.poll() is None:
        await swap.stop_proxy()
        stopped.append("llama_swap")
    manager = studio if backend == "studio" else swap
    try:
        await manager.start_proxy()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"proxy failed to start: {exc}") from exc
    return {
        "backend": backend,
        "stopped": stopped,
        "running": True,
        "restart_required": False,
    }


@router.post("/stats/reset")
async def reset_proxy_stats():
    if configured_proxy_backend() != "studio":
        raise HTTPException(status_code=409, detail="Stats reset requires the Studio backend")
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
