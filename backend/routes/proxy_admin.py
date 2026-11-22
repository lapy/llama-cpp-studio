"""Studio control-plane endpoints for proxy configuration and observation."""

import json
import os
import asyncio
import time

import httpx
from typing import Any, Literal
from urllib.parse import unquote

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.access_policy import proxy_listen_host
from backend.concurrency import run_blocking
from backend.data_store import get_store
from backend.inference_auth import rotate_key
from backend.proxy.studio.manager import configured_proxy_backend, get_studio_proxy_manager
from backend.proxy.studio.runtime_defaults import LIVE_APPLIED_SETTINGS
from backend.proxy.studio.vram import VramAdmissionError
from backend.proxy.studio.vram_estimates import (
    MAX_ESTIMATE_ENTRIES,
    MAX_ESTIMATE_MIB,
    MAX_SAFETY_MARGIN_MIB,
    MIN_ESTIMATE_MIB,
)

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
    capture_request_payloads: bool = False
    default_model: str | None = None
    routing_aliases: dict[str, str | list[str]] = Field(default_factory=dict)
    # GPU admission is applied live: a mode or estimate change takes effect on
    # the next cold start without restarting the proxy or any engine.
    vram_admission: Literal["off", "warn", "enforce"] = "off"
    vram_safety_margin_mib: int = Field(default=1024, ge=0, le=MAX_SAFETY_MARGIN_MIB)
    vram_estimates: dict[str, int] = Field(default_factory=dict)

    @field_validator("vram_estimates")
    @classmethod
    def _validate_vram_estimates(cls, value: dict[str, int]) -> dict[str, int]:
        if len(value) > MAX_ESTIMATE_ENTRIES:
            raise ValueError(f"At most {MAX_ESTIMATE_ENTRIES} model estimates are allowed")
        cleaned: dict[str, int] = {}
        for model_id, estimate in value.items():
            name = str(model_id).strip()
            if not name:
                raise ValueError("A VRAM estimate needs a model id")
            if not MIN_ESTIMATE_MIB <= int(estimate) <= MAX_ESTIMATE_MIB:
                raise ValueError(
                    f"VRAM estimate for {name} must be between "
                    f"{MIN_ESTIMATE_MIB} and {MAX_ESTIMATE_MIB} MiB"
                )
            cleaned[name] = int(estimate)
        return cleaned


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
                "capture_request_payloads": False,
                "vram_admission": "off",
                "vram_safety_margin_mib": 1024,
                "vram_estimates": {},
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


def _vram_snapshot() -> dict[str, Any]:
    manager = get_studio_proxy_manager()
    snapshot = getattr(manager, "vram_snapshot", None)
    if callable(snapshot):
        return snapshot()
    admission = getattr(manager, "vram", None)
    if admission is not None and hasattr(admission, "snapshot"):
        return admission.snapshot()
    return {
        "mode": "off",
        "safety_margin_mib": 1024,
        "estimates_configured": {},
        "observed_estimates": {},
        "probe": {"status": "never"},
        "reservations": [],
        "last_admission": None,
    }


def _vram_refusal(exc: VramAdmissionError) -> JSONResponse:
    """409 body for a refused start: ``{error, code, detail}``."""
    return JSONResponse(status_code=409, content=exc.payload())


def _with_replay_payloads(rows: list[dict]) -> list[dict]:
    """Attach the memory-only capture fields to metadata rows.

    Both the overview and the paginated request list must carry the same
    capture fields, otherwise a background overview refresh silently strips
    Replay from a request that still has a payload.
    """
    manager = get_studio_proxy_manager()
    registry = getattr(manager, "registry", None)
    payloads = getattr(registry, "replay_payloads", None) if registry is not None else None
    if not payloads:
        return rows
    for row in rows:
        payload = payloads.get(str(row.get("id") or ""))
        if not payload:
            continue
        row["replayable"] = True
        row["captured_headers"] = payload.get("headers") or {}
        row["captured_redacted_headers"] = payload.get("redacted_headers") or []
        row["captured_body"] = payload.get("body")
        row["captured_body_truncated"] = bool(payload.get("body_truncated"))
    return rows


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
        "vram": _vram_snapshot(),
        "requests": _with_replay_payloads(history["requests"])
        if settings["backend"] == "studio"
        else [],
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


@router.get("/vram")
async def get_proxy_vram():
    """GPU admission observation: mode, probe, estimates and last decision.

    Read-only and additive. The probe is never run here; it reports the last
    cached probe so the page load cannot block on NVML or nvidia-smi.
    """
    return _vram_snapshot()


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
    live_fields = changed & LIVE_APPLIED_SETTINGS
    restart_required = bool(changed - live_fields)
    routing_revision = None
    manager = get_studio_proxy_manager()
    if hasattr(manager, "apply_routing"):
        routing_revision = manager.apply_routing(after)
        if manager.running:
            restart_required = manager.runtime_restart_required(after)
    apply_vram = getattr(manager, "apply_vram_settings", None)
    if callable(apply_vram):
        apply_vram(after)
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
    except VramAdmissionError as exc:
        return _vram_refusal(exc)
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
    except VramAdmissionError as exc:
        return _vram_refusal(exc)
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
    result = await asyncio.to_thread(
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
    result["requests"] = _with_replay_payloads(result.get("requests", []))
    return result


def _record_replay_activity(
    registry: Any,
    payload: dict[str, Any],
    *,
    status: int | None = None,
    error: str = "",
) -> None:
    """Leave an audit trail without ever storing the payload itself."""
    activities = getattr(registry, "activities", None)
    if activities is None:
        return
    try:
        activities.append(
            {
                "type": "replay" if status is not None and status < 400 else "replay_failed",
                "model": payload.get("model"),
                "request_id": payload.get("id"),
                "path": payload.get("path"),
                "status": status,
                "error": error,
                "timestamp": time.time(),
            }
        )
    except Exception:  # noqa: BLE001 - auditing must never fail the replay
        pass


@router.post("/requests/{request_id}/replay")
async def replay_proxy_request(request_id: str):
    """Re-send a captured request to the still-loaded engine.

    Captured payloads are memory-only and opt-in. Replay is an operator action:
    it bypasses quota accounting and reaches the loaded engine directly. It is
    intentionally rejected when the model is no longer loaded so a replay can
    never silently start a new process or hit a different model.
    """
    manager = get_studio_proxy_manager()
    registry = getattr(manager, "registry", None)
    payload = registry.replay_payload(request_id) if registry is not None else None
    if not payload:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "PAYLOAD_NOT_CAPTURED",
                "message": "Request payload is no longer available. Enable payload capture before the request.",
            },
        )
    generation = registry.active(payload.get("model") or "") if registry is not None else None
    if generation is None:
        _record_replay_activity(registry, payload, error="model_not_loaded")
        raise HTTPException(
            status_code=409,
            detail={
                "code": "MODEL_NOT_LOADED",
                "message": f"Start {payload.get('model') or 'the model'} before replaying this request.",
            },
        )
    headers = {
        str(name): str(value)
        for name, value in (payload.get("headers") or {}).items()
        if str(name).lower()
        not in {"host", "content-length", "connection", "transfer-encoding", "accept-encoding"}
    }
    body = payload.get("body")
    content = body.encode("utf-8") if body is not None else None
    url = generation.upstream.rstrip("/") + "/" + str(payload.get("path") or "/").lstrip("/")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.request(
                str(payload.get("method") or "POST").upper(),
                url,
                headers=headers,
                content=content,
            )
    except httpx.HTTPError as exc:
        _record_replay_activity(registry, payload, error=type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail={
                "code": "REPLAY_FAILED",
                "message": f"Replay request failed: {type(exc).__name__}",
            },
        ) from exc
    _record_replay_activity(registry, payload, status=response.status_code)
    limit = 512 * 1024
    raw = response.content or b""
    truncated = len(raw) > limit
    text = raw[:limit].decode("utf-8", "replace")
    return {
        "replayed": True,
        "source_request_id": request_id,
        "model": payload.get("model"),
        "method": payload.get("method"),
        "path": payload.get("path"),
        "status": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "body": text,
        "body_truncated": truncated,
    }
