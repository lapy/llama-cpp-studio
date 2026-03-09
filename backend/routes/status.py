from fastapi import APIRouter
import psutil
import os

from backend.llama_swap_client import LlamaSwapClient

router = APIRouter()

DEFAULT_PROXY_PORT = 2000


@router.get("/status")
async def get_system_status():
    """Get system status and running instances (from llama-swap)."""
    client = LlamaSwapClient()
    try:
        running_data = await client.get_running_models()
    except Exception:
        running_data = {"running": []}
    if isinstance(running_data, list):
        running_list = running_data
    else:
        running_list = running_data.get("running") or []

    proxy_health = await client.check_health()

    active_instances = []
    for i, item in enumerate(running_list):
        proxy_model_name = item.get("model", "")
        state = item.get("state", "")
        runtime_type = "lmdeploy" if state == "lmdeploy" else "llama_cpp"
        # All traffic is served via the unified llama-swap proxy on DEFAULT_PROXY_PORT.
        port = DEFAULT_PROXY_PORT
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

    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    data_dir = "data" if os.path.exists("data") else "/app/data"
    try:
        disk = psutil.disk_usage(data_dir)
    except FileNotFoundError:
        disk = psutil.disk_usage("/")

    return {
        "system": {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent,
            },
            "disk": {
                "total": disk.total,
                "used": disk.used,
                "free": disk.free,
                "percent": (disk.used / disk.total) * 100,
            },
        },
        "running_instances": active_instances,
        "proxy_status": {
            "enabled": True,
            "port": DEFAULT_PROXY_PORT,
            "endpoint": "http://localhost:2000/v1/chat/completions",
            "healthy": proxy_health.get("healthy", False),
            "status_code": proxy_health.get("status_code"),
            "loading_models": proxy_health.get("loading_models", []),
        },
    }
