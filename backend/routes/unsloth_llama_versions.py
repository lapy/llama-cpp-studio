from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from backend.data_store import get_store
from backend.engines.python.release_host import get_release_installer

ENGINE_ID = "unsloth_llama"
from backend.engines.unsloth_llama.prebuilt import is_behind
from backend.routes.engine_versions import LIFECYCLE, EngineLifecycle


router = APIRouter()


@router.get("/unsloth-llama/check-updates", operation_id="unsloth_llama_check_updates")
async def check_updates() -> Dict[str, Any]:
    manager = get_release_installer(ENGINE_ID)
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
        return await get_release_installer(ENGINE_ID).install_release(
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
    return get_release_installer(ENGINE_ID).cancel_task(str(task_id))


# ── Generic lifecycle registry (Unsloth llama.cpp) ────────────────────
# Unsloth ships release installs only: no build-settings dialog, no source
# build, no per-engine status (versions ride the shared version list), and
# deletion is version-scoped. Those verbs answer 405 on the generic
# surface; the three implemented ones delegate to the route functions
# above so both surfaces are identical by construction.
LIFECYCLE[ENGINE_ID] = EngineLifecycle(
    engine_id=ENGINE_ID,
    url_prefix="/api/unsloth-llama",
    verbs={
        "check_updates": lambda: check_updates(),
        "install": lambda request: install(request),
        "cancel": lambda payload: cancel(payload),
    },
)
