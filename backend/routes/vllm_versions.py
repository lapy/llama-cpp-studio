"""Install, status, and settings endpoints for vanilla vLLM."""

from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException

from backend.data_store import get_store
from backend.venv_install_settings import coerce_install_settings
from backend.vllm_manager import get_vllm_manager


router = APIRouter()
ENGINE_ID = "vllm"


@router.get("/vllm/check-updates")
async def check_updates() -> Dict:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("https://pypi.org/pypi/vllm/json", timeout=15.0)
            response.raise_for_status()
            payload = response.json()
        active = get_store().get_active_engine_version(ENGINE_ID) or {}
        latest = (payload.get("info") or {}).get("version")
        current = active.get("package_version") or active.get("version")
        return {
            "latest_version": latest,
            "current_version": current,
            "update_available": bool(latest and latest != current),
            "releases": list((payload.get("releases") or {}).keys()),
            "url": "https://pypi.org/project/vllm/",
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Failed to check vLLM updates: {exc}") from exc


@router.get("/vllm/status")
async def status() -> Dict:
    return get_vllm_manager().status()


@router.get("/vllm/build-settings")
async def get_build_settings() -> Dict:
    return coerce_install_settings(
        ENGINE_ID, get_store().get_engine_build_settings(ENGINE_ID) or {}
    )


@router.put("/vllm/build-settings")
async def save_build_settings(settings: Optional[Dict] = Body(None)) -> Dict:
    if settings is not None and not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    coerced = coerce_install_settings(ENGINE_ID, settings or {})
    get_store().replace_engine_build_settings(ENGINE_ID, coerced)
    return coerced


@router.post("/vllm/install")
async def install(request: Optional[Dict] = None) -> Dict:
    payload = request or {}
    saved = await get_build_settings()
    version = payload.get("version") or saved.get("pip_version") or None
    try:
        return await get_vllm_manager().install_release(
            str(version) if version else None,
            force_reinstall=bool(payload.get("force_reinstall")),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/vllm/install-source")
async def install_source(request: Optional[Dict] = None) -> Dict:
    payload = request or {}
    saved = await get_build_settings()
    try:
        return await get_vllm_manager().install_from_source(
            str(payload.get("repo_url") or saved["source_repo"]),
            str(payload.get("branch") or saved["source_branch"]),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/vllm/remove")
async def remove() -> Dict:
    try:
        return await get_vllm_manager().remove()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/vllm/cancel")
async def cancel(payload: dict = Body(...)) -> Dict:
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return get_vllm_manager().cancel_task(str(task_id))
