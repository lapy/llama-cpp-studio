"""Per-model native launch plans. No external proxy configuration is generated."""

import asyncio
from typing import Literal
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.data_store import get_store
from backend.proxy.studio.manager import get_studio_proxy_manager
from backend.services.model_runtime_apply import build_apply_plan

router = APIRouter()


@router.get("/pending")
async def runtime_pending():
    manager = get_studio_proxy_manager()
    states = {row["model"]: row.get("state", "unknown") for row in manager.registry.snapshot()}
    plan = await asyncio.to_thread(build_apply_plan, get_store().list_models(), states=states)
    return {
        **plan,
        "applicable": True,
        "pending": bool(plan["models"]),
        "changes": [],
        "backend": "studio",
    }


@router.get("/stale")
async def runtime_stale():
    plan = await runtime_pending()
    return {"applicable": True, "stale": plan["pending"], "backend": "studio"}


class LaunchApplyEntry(BaseModel):
    catalog_id: str
    expected_desired_revision: str | None = None
    expected_published_revision: str | None = None
    idempotency_key: str


class LaunchApplyBody(BaseModel):
    mode: Literal["restart_now", "next_start"]
    models: list[LaunchApplyEntry]


@router.post("/apply-launch")
async def apply_launch_changes(body: LaunchApplyBody):
    """Preserve native sequential batch apply without an external proxy or YAML."""
    from backend.services.model_runtime_apply import ApplyRejected, NativeRuntimeGateway, apply_many

    manager = get_studio_proxy_manager()
    models = {str(row["id"]): row for row in get_store().list_models() if row.get("id")}
    try:
        async with manager._apply_lock:
            return await apply_many(
                [entry.model_dump() for entry in body.models],
                mode=body.mode,
                gateway=NativeRuntimeGateway(manager._client()),
                models_by_id=models,
            )
    except ApplyRejected as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


compatibility_router = APIRouter()
# Previously available with the native backend. Retain the URL for existing clients.
compatibility_router.add_api_route(
    "/llama-swap/apply-launch", apply_launch_changes, methods=["POST"], include_in_schema=False
)
