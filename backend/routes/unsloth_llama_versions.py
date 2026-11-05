from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from backend.data_store import get_store
from backend.engines.unsloth_llama.installer import (
    ENGINE_ID,
    get_unsloth_llama_manager,
)
from backend.engines.unsloth_llama.prebuilt import is_behind


router = APIRouter()


@router.get("/unsloth-llama/check-updates", operation_id="unsloth_llama_check_updates")
async def check_updates() -> Dict[str, Any]:
    manager = get_unsloth_llama_manager()
    active = get_store().get_active_engine_version(ENGINE_ID) or {}
    current = str(active.get("source_ref") or active.get("version") or "").strip() or None
    try:
        latest = await manager.latest_published_tag()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail=f"Failed to check unsloth_llama updates: {exc}"
        ) from exc
    release_url = "https://github.com/unslothai/llama.cpp/releases"
    if latest:
        release_url = f"{release_url}/tag/{latest}"
    return {
        "latest_version": latest,
        "current_version": current,
        "update_available": is_behind(current, latest),
        "url": release_url,
        "release_url": release_url,
    }


@router.post("/unsloth-llama/install", operation_id="unsloth_llama_install")
async def install(request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    payload = request or {}
    from backend.operations.action_recovery import (
        ActionAdmissionError,
        bind_action_confirmation,
    )

    bind_action_confirmation(payload)
    tag = payload.get("tag_name") or payload.get("version")
    try:
        return await get_unsloth_llama_manager().install_release(
            tag_name=str(tag).strip() if tag else None
        )
    except ActionAdmissionError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/unsloth-llama/cancel", operation_id="unsloth_llama_cancel")
async def cancel(payload: dict = Body(...)) -> Dict[str, Any]:
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return get_unsloth_llama_manager().cancel_task(str(task_id))
