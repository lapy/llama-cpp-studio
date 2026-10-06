import json
from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse, Response
import psutil
import os

from backend.diagnostics import build_diagnostics_bundle
from backend.inference_url import normalize_public_inference_url
from backend.proxy.llama_swap.client import get_llama_swap_client, get_proxy_port
from backend.proxy.llama_swap.runtime_observation import runtime_observation_report
from backend.ops_metrics import snapshot_metrics
from backend.paths import studio_data_dir
from backend.store_io import persistence_status

router = APIRouter()


@router.get("/operations/recovery")
async def operation_recovery():
    """Report the last restart reconciliation. It does not start work."""
    from backend.operations.supervisor import get_supervisor

    return get_supervisor().recovery_status()


@router.post("/operations/reconcile")
async def reconcile_operations():
    """Reconcile durable operations with this process. Does not replay them."""
    from backend.operations.supervisor import get_supervisor

    result = get_supervisor().reconcile_startup()
    if result.get("outcome") == "unknown":
        return JSONResponse(
            status_code=500,
            content={
                "code": "RECONCILE_UNKNOWN",
                "committed": "unknown",
                "outcome": "unknown",
                "detail": result.get("detail")
                or "The recovery outcome could not be established. Refresh before trying again.",
            },
        )
    return result


@router.get("/live")
async def live():
    """Cheap liveness probe. It does not check the proxy or model runtimes."""
    return {"live": True}


@router.get("/ready")
async def ready():
    """Readiness. First-run with no engine installed is ready; a required proxy failure is not."""
    from backend.data_store import StorageCorruptionError, get_store
    from backend.proxy.llama_swap.config import any_active_runtime_in_db

    try:
        get_store().get_settings()
    except StorageCorruptionError:
        return JSONResponse(
            {"ready": False, "reason": "configuration is unreadable"},
            status_code=503,
        )
    if not any_active_runtime_in_db():
        return {"ready": True, "proxy_required": False}
    client = get_llama_swap_client()
    try:
        health = await client.check_health()
    except Exception:
        health = {"healthy": False}
    if isinstance(health, dict) and health.get("healthy"):
        return {"ready": True, "proxy_required": True}
    return JSONResponse(
        {
            "ready": False,
            "proxy_required": True,
            "reason": "inference proxy is not healthy",
        },
        status_code=503,
    )


@router.get("/status")
async def get_system_status():
    """Get system status and running instances (from llama-swap)."""
    proxy_port = get_proxy_port()
    client = get_llama_swap_client()
    runtime_known = True
    try:
        running_data = await client.get_running_models()
    except Exception:
        running_data = {"running": []}
        runtime_known = False
    if isinstance(running_data, list):
        running_list = running_data
    else:
        running_list = (running_data or {}).get("running") or []

    try:
        proxy_health = await client.check_health()
    except Exception:
        proxy_health = {"healthy": False, "status_code": None, "loading_models": []}
    if not isinstance(proxy_health, dict):
        proxy_health = {
            "healthy": bool(proxy_health),
            "status_code": None,
            "loading_models": [],
        }

    active_instances = []
    for i, item in enumerate(running_list):
        if not isinstance(item, dict):
            continue
        proxy_model_name = item.get("model", "")
        state = item.get("state", "")
        runtime_type = "lmdeploy" if state == "lmdeploy" else "llama_cpp"
        # All traffic is served via the unified llama-swap proxy on proxy_port.
        port = proxy_port
        active_instances.append(
            {
                "id": i,
                "model_id": proxy_model_name,
                "port": port,
                "runtime_type": runtime_type,
                "proxy_model_name": proxy_model_name,
                "started_at": None,
            }
        )

    try:
        cpu_percent = psutil.cpu_percent(interval=None)
    except Exception:
        cpu_percent = 0.0

    try:
        memory = psutil.virtual_memory()
        memory_payload = {
            "total": memory.total,
            "used": memory.used,
            "available": memory.available,
            "percent": memory.percent,
        }
    except Exception:
        memory_payload = {
            "total": 0,
            "used": 0,
            "available": 0,
            "percent": 0.0,
        }

    health_observed_at = datetime.now(timezone.utc).isoformat()
    runtime_observation = runtime_observation_report()
    data_dir = studio_data_dir()
    disk = None
    for path in (data_dir, "/"):
        try:
            disk = psutil.disk_usage(path)
            break
        except Exception:
            continue
    disk_payload = {
        "total": disk.total if disk else 0,
        "used": disk.used if disk else 0,
        "free": disk.free if disk else 0,
        "percent": ((disk.used / disk.total) * 100) if disk and disk.total else 0.0,
    }

    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory": memory_payload,
            "disk": disk_payload,
        },
        "running_instances": active_instances,
        "proxy_status": {
            "enabled": True,
            "port": proxy_port,
            "healthy": proxy_health.get("healthy", False),
            "status_code": proxy_health.get("status_code"),
            "loading_models": proxy_health.get("loading_models", []),
            "public_inference_url": _public_inference_url(),
            "observed_at": health_observed_at,
            "health_observed_at": health_observed_at,
            "observation": "current",
            "runtime_known": runtime_known,
        },
        "runtime_observation": runtime_observation,
        "persistence": persistence_status(),
        "ready": _status_ready(proxy_health),
        "metrics": snapshot_metrics(),
    }


def _public_inference_url() -> str:
    try:
        from backend.data_store import get_store

        settings = get_store().get_settings() or {}
    except Exception:
        return ""
    try:
        return normalize_public_inference_url(settings.get("public_inference_url"))
    except ValueError:
        return ""


@router.get("/settings/inference")
async def get_inference_settings():
    """Return the proxy port and optional public URL clients should call."""
    return {
        "proxy_port": get_proxy_port(),
        "public_inference_url": _public_inference_url(),
    }


@router.put("/settings/inference")
async def put_inference_settings(body: dict):
    """Store the URL remote clients use. An empty string clears it."""
    from fastapi import HTTPException
    from backend.data_store import get_store

    try:
        url = normalize_public_inference_url((body or {}).get("public_inference_url"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    get_store().update_settings({"public_inference_url": url})
    return {"proxy_port": get_proxy_port(), "public_inference_url": url}


@router.get("/diagnostics/bundle")
async def diagnostics_bundle():
    """Download a redacted JSON bundle. It does not include document contents."""
    from backend.data_store import get_store

    try:
        settings = get_store().get_settings() or {}
    except Exception:
        settings = {}
    status = await get_system_status()
    bundle = build_diagnostics_bundle(
        proxy_status=status.get("proxy_status"),
        runtime_observation=status.get("runtime_observation"),
        settings=settings,
    )
    body = json.dumps(bundle, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="studio-diagnostics.json"'},
    )


def _status_ready(proxy_health: dict) -> bool:
    try:
        from backend.proxy.llama_swap.config import any_active_runtime_in_db

        if not any_active_runtime_in_db():
            return True
    except Exception:
        return False
    return bool(proxy_health.get("healthy", False))
