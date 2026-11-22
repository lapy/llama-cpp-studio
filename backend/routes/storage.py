"""Storage inventory and guarded cleanup endpoints.

The API never accepts a filesystem path.  Inventory rows carry opaque unit
ids; a cleanup plan names only a category and those ids.  Every scan and every
delete runs on the shared blocking pool so a multi-GB cache tree cannot stall
the event loop.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from backend.concurrency import run_blocking
from backend.operations.exclusive import exclusive_action, exclusive_http_error
from backend.services.storage_cleanup import (
    StorageCleanupError,
    apply_cleanup,
    preview_cleanup,
)
from backend.services.storage_inventory import collect_storage_inventory

router = APIRouter()

StorageCategoryId = Literal[
    "models",
    "hf_cache",
    "engine_installs",
    "cuda_toolkits",
    "jit_caches",
    "history_backups",
    "accounting_db",
]


class StorageCleanupPreviewBody(BaseModel):
    category: StorageCategoryId
    unit_ids: list[str] = Field(default_factory=list, max_length=200)
    keep_history: int | None = Field(default=None, ge=1, le=100)


class StorageCleanupApplyBody(BaseModel):
    plan_id: str = Field(min_length=8, max_length=128)
    confirm: bool = False


def _cleanup_http_error(exc: StorageCleanupError) -> HTTPException:
    return HTTPException(status_code=exc.status_code, detail=exc.detail)


@router.get("/storage/inventory")
async def storage_inventory(refresh: bool = Query(default=False)):
    """Return the cached (or freshly scanned) storage inventory."""
    return await run_blocking(collect_storage_inventory, refresh=refresh)


@router.post("/storage/cleanup/preview")
async def storage_cleanup_preview(body: StorageCleanupPreviewBody):
    """Resolve opaque unit ids into a short-lived, guarded cleanup plan."""
    try:
        return await run_blocking(
            preview_cleanup,
            body.category,
            body.unit_ids,
            body.keep_history,
        )
    except StorageCleanupError as exc:
        raise _cleanup_http_error(exc) from exc


@router.post("/storage/cleanup/apply")
async def storage_cleanup_apply(body: StorageCleanupApplyBody):
    """Apply a previewed plan after re-scanning and revalidating every unit."""
    try:
        async with exclusive_action(
            "storage_cleanup",
            "storage:cleanup",
            detail={"plan_id": body.plan_id},
        ):
            return await run_blocking(apply_cleanup, body.plan_id, confirm=body.confirm)
    except StorageCleanupError as exc:
        raise _cleanup_http_error(exc) from exc
    except Exception as exc:
        mapped = exclusive_http_error(exc)
        if mapped is not None:
            raise mapped from exc
        raise
