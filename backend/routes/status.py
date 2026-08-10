from fastapi import APIRouter
import psutil
import os

from backend.llama_swap_client import get_llama_swap_client, get_proxy_port

router = APIRouter()

# Backward-compatible aliases for callers/tests that imported from this module.
_get_proxy_port = get_proxy_port


@router.get("/status")
async def get_system_status():
    """Get system status and running instances (from llama-swap)."""
    proxy_port = get_proxy_port()
    client = get_llama_swap_client()
    try:
        running_data = await client.get_running_models()
    except Exception:
        running_data = {"running": []}
    if isinstance(running_data, list):
        running_list = running_data
    else:
        running_list = running_data.get("running") or []

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

    data_dir = "data" if os.path.exists("data") else "/app/data"
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
        },
    }
