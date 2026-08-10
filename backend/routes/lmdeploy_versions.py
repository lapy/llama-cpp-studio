from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException

from backend.data_store import get_store
from backend.lmdeploy_manager import get_lmdeploy_manager
from backend.logging_config import get_logger
from backend.venv_install_settings import coerce_install_settings

router = APIRouter()
logger = get_logger(__name__)

_ENGINE_ID = "lmdeploy"


@router.get("/lmdeploy/check-updates")
async def lmdeploy_check_updates() -> Dict:
    """Check PyPI for latest LMDeploy version."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get("https://pypi.org/pypi/lmdeploy/json", timeout=10.0)
            r.raise_for_status()
            data = r.json()
            info = data.get("info", {})
            return {
                "latest_version": info.get("version"),
                "releases": list(data.get("releases", {}).keys()),
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check PyPI: {exc}")


@router.get("/lmdeploy/status")
async def lmdeploy_installer_status() -> Dict:
    manager = get_lmdeploy_manager()
    try:
        return manager.status()
    except Exception as exc:
        logger.warning("lmdeploy/status: %s", exc)
        # Never fail the whole app load if status probing throws (permissions, corrupt state, etc.).
        return {
            "installed": False,
            "version": None,
            "binary_path": None,
            "venv_path": None,
            "installed_at": None,
            "removed_at": None,
            "operation": None,
            "operation_started_at": None,
            "last_error": str(exc),
            "log_path": None,
            "install_type": None,
            "source_repo": None,
            "source_branch": None,
        }


@router.get("/lmdeploy/build-settings")
async def lmdeploy_get_build_settings() -> Dict:
    """Return persisted install/build defaults for LMDeploy."""
    store = get_store()
    saved = store.get_engine_build_settings(_ENGINE_ID) or {}
    return coerce_install_settings(_ENGINE_ID, saved)


@router.put("/lmdeploy/build-settings")
async def lmdeploy_save_build_settings(settings: Optional[Dict] = Body(None)) -> Dict:
    """Persist install/build defaults for LMDeploy (edit/save without installing)."""
    if settings is not None and not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    coerced = coerce_install_settings(_ENGINE_ID, settings or {})
    store = get_store()
    store.replace_engine_build_settings(_ENGINE_ID, coerced)
    return coerce_install_settings(_ENGINE_ID, coerced)


@router.post("/lmdeploy/install")
async def lmdeploy_install(request: Optional[Dict[str, str]] = None) -> Dict:
    manager = get_lmdeploy_manager()
    payload = request or {}
    saved = coerce_install_settings(
        _ENGINE_ID, get_store().get_engine_build_settings(_ENGINE_ID) or {}
    )
    version = payload.get("version")
    if version is None or str(version).strip() == "":
        version = saved.get("pip_version") or None
    force_reinstall = bool(payload.get("force_reinstall"))
    try:
        return await manager.install_release(
            version=version or None, force_reinstall=force_reinstall
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/lmdeploy/install-source")
async def lmdeploy_install_source(request: Optional[Dict[str, str]] = None) -> Dict:
    """Install LMDeploy from a git repo and branch (for development)."""
    manager = get_lmdeploy_manager()
    payload = request or {}
    saved = coerce_install_settings(
        _ENGINE_ID, get_store().get_engine_build_settings(_ENGINE_ID) or {}
    )
    repo_url = payload.get("repo_url") or saved["source_repo"]
    branch = payload.get("branch") or saved["source_branch"]
    try:
        return await manager.install_from_source(repo_url=repo_url, branch=branch)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/lmdeploy/remove")
async def lmdeploy_remove() -> Dict:
    manager = get_lmdeploy_manager()
    try:
        return await manager.remove()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@router.post("/lmdeploy/cancel")
async def lmdeploy_cancel(payload: dict = Body(...)) -> Dict:
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return get_lmdeploy_manager().cancel_task(str(task_id))
