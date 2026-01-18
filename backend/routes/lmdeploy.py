from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

from backend.lmdeploy_installer import get_lmdeploy_installer
from backend.lmdeploy_manager import get_lmdeploy_manager

router = APIRouter()


@router.get("/lmdeploy/status")
async def lmdeploy_installer_status() -> Dict:
    installer = get_lmdeploy_installer()
    return installer.status()


@router.post("/lmdeploy/install")
async def lmdeploy_install(request: Optional[Dict[str, str]] = None) -> Dict:
    installer = get_lmdeploy_installer()
    payload = request or {}
    version = payload.get("version")
    force_reinstall = bool(payload.get("force_reinstall"))
    try:
        return await installer.install(version=version, force_reinstall=force_reinstall)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/lmdeploy/remove")
async def lmdeploy_remove() -> Dict:
    installer = get_lmdeploy_installer()
    try:
        return await installer.remove()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.get("/lmdeploy/logs")
async def lmdeploy_logs(max_bytes: int = 8192) -> Dict[str, str]:
    """Get LMDeploy installer logs."""
    installer = get_lmdeploy_installer()
    max_bytes = max(1024, min(max_bytes, 1024 * 1024))
    return {"log": installer.read_log_tail(max_bytes)}


@router.get("/lmdeploy/runtime-logs")
async def lmdeploy_runtime_logs(max_bytes: int = 8192) -> Dict[str, str]:
    """Get LMDeploy runtime logs (from running server instances)."""
    manager = get_lmdeploy_manager()
    max_bytes = max(1024, min(max_bytes, 1024 * 1024))
    return {"log": manager.read_log_tail(max_bytes)}
