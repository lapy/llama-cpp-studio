from typing import Any, Dict, Optional

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.proxy.llama_swap.client import get_llama_swap_client
from backend.proxy.llama_swap.manager import get_llama_swap_manager, mark_swap_config_stale
from backend.proxy.llama_swap.routing import (
    get_routing_document,
    routing_warnings,
    save_routing_document,
)
from backend.logging_config import get_logger

router = APIRouter()
logger = get_logger(__name__)


class LlamaSwapRoutingPayload(BaseModel):
    profiles: Dict[str, Any] = Field(default_factory=dict)
    selectors: Dict[str, Any] = Field(default_factory=dict)


class ActiveProfilePayload(BaseModel):
    name: Optional[str] = None


def _proxy_http_error(exc: Exception, *, action: str) -> HTTPException:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        status = exc.response.status_code
        detail = (exc.response.text or "").strip() or str(exc)
        if status in (400, 404):
            return HTTPException(status_code=status, detail=detail)
        return HTTPException(
            status_code=502,
            detail=f"llama-swap {action} failed ({status}): {detail}",
        )
    return HTTPException(status_code=502, detail=f"llama-swap {action} unavailable: {exc}")


@router.get("/llama-swap/pending")
async def llama_swap_pending() -> Dict[str, Any]:
    """
    Compare the on-disk llama-swap config to what would be generated from the current DB.
    """
    manager = get_llama_swap_manager()
    return await manager.get_config_pending_state()


@router.get("/llama-swap/stale")
async def llama_swap_stale() -> Dict[str, Any]:
    """
    Cheap flag for the UI: studio has changes that may require rewriting llama-swap-config.yaml.
    Use GET /llama-swap/pending only when the user opens “apply” or needs a diff summary.
    """
    manager = get_llama_swap_manager()
    return manager.get_swap_config_stale_state()


class LaunchApplyEntry(BaseModel):
    catalog_id: str
    expected_desired_revision: Optional[str] = None
    expected_published_revision: Optional[str] = None
    idempotency_key: str


class LaunchApplyBody(BaseModel):
    mode: str
    models: list[LaunchApplyEntry]


@router.post("/llama-swap/apply-launch")
async def llama_swap_apply_launch(body: LaunchApplyBody) -> Dict[str, Any]:
    """Apply launch-only revisions one model at a time. Proxy edits stay on apply-config."""
    from backend.data_store import get_store
    from backend.proxy.llama_swap.manager import _configs_semantically_equal
    from backend.services.model_runtime_apply import (
        ApplyRejected,
        LlamaSwapRuntimeGateway,
        apply_many,
    )

    manager = get_llama_swap_manager()
    store = get_store()
    models = {str(row.get("id")): row for row in store.list_models() if row.get("id")}
    try:
        async with manager._apply_lock:
            desired = await manager.compute_desired_config_content()
            disk = manager._read_config_text()
            yaml_differs = not _configs_semantically_equal(disk, desired or "")
            if yaml_differs:
                raise ApplyRejected(
                    409,
                    {
                        "error": "global_apply_required",
                        "message": "Reload proxy — affects all loaded models.",
                    },
                )
            return await apply_many(
                [entry.model_dump() for entry in body.models],
                mode=body.mode,
                gateway=LlamaSwapRuntimeGateway(manager._client()),
                models_by_id=models,
                yaml_differs=False,
                disk_yaml=disk,
            )
    except ApplyRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/llama-swap/apply-config")
async def llama_swap_apply_config() -> Dict[str, str]:
    """
    Unload all models via llama-swap, then regenerate and write llama-swap-config.yaml.
    """
    manager = get_llama_swap_manager()
    try:
        await manager.user_apply_regenerate_config()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("apply-config failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"message": "llama-swap configuration applied"}


@router.get("/llama-swap/routing")
async def llama_swap_get_routing() -> Dict[str, Any]:
    """Return Studio-managed profiles and selectors for llama-swap YAML generation."""
    doc = get_routing_document()
    return {
        **doc,
        "warnings": routing_warnings(doc),
    }


@router.put("/llama-swap/routing")
async def llama_swap_put_routing(payload: LlamaSwapRoutingPayload) -> Dict[str, Any]:
    """Persist profiles/selectors and mark llama-swap config stale for apply."""
    try:
        doc = save_routing_document(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    mark_swap_config_stale()
    return {
        **doc,
        "warnings": routing_warnings(doc),
        "stale": True,
    }


@router.get("/llama-swap/profiles")
async def llama_swap_live_profiles() -> Dict[str, Any]:
    """Passthrough: configured profiles + active profile from the running proxy."""
    try:
        return await get_llama_swap_client().get_profiles()
    except Exception as exc:
        raise _proxy_http_error(exc, action="profiles") from exc


@router.put("/llama-swap/profiles/active")
async def llama_swap_set_active_profile(
    payload: ActiveProfilePayload,
) -> Dict[str, Any]:
    """Passthrough: activate a profile (or clear with name=null) on the running proxy."""
    name = payload.name
    if isinstance(name, str):
        name = name.strip() or None
    try:
        return await get_llama_swap_client().set_active_profile(name)
    except Exception as exc:
        raise _proxy_http_error(exc, action="active profile") from exc
