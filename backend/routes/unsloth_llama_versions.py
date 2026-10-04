from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from backend.engines.unsloth_llama.installer import (
    ENGINE_ID,
    get_unsloth_llama_manager,
)
from backend.routes.engine_versions import _check_updates, github_release_updates


router = APIRouter()


@router.get("/unsloth-llama/check-updates", operation_id="unsloth_llama_check_updates")
async def check_updates() -> Dict[str, Any]:
    return await _check_updates(ENGINE_ID, github_release_updates("unslothai/llama.cpp"))


@router.get("/unsloth-llama/status", operation_id="unsloth_llama_status")
async def status() -> Dict[str, Any]:
    try:
        return get_unsloth_llama_manager().status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/unsloth-llama/install", operation_id="unsloth_llama_install")
async def install(request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = request or {}
    tag = payload.get("tag_name") or payload.get("version")
    try:
        return await get_unsloth_llama_manager().install_release(
            tag_name=str(tag).strip() if tag else None
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/unsloth-llama/cancel", operation_id="unsloth_llama_cancel")
async def cancel(payload: dict = Body(...)) -> Dict[str, Any]:
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return get_unsloth_llama_manager().cancel_task(str(task_id))
