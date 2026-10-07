"""Export and restore a versioned configuration backup."""

import json
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from backend.config_backup import (
    ConfigBackupError,
    MAX_BACKUP_BYTES,
    apply_backup,
    export_backup,
    preview_backup,
    reconcile_config_restore,
)
from backend.data_store import get_store

router = APIRouter()


class BackupItemResponse(BaseModel):
    kind: str
    id: str
    action: str
    local_id: str | None = None
    reason: str | None = None


class BackupPreviewResponse(BaseModel):
    schema_version: int
    plan_id: str | None
    applicable: bool
    notice: str
    revisions: dict[str, str]
    items: list[BackupItemResponse]
    limits: dict[str, list[str]]


class BackupApplyResponse(BaseModel):
    plan_id: str
    outcome: Literal["completed"]
    notice: str


class BackupReconcileResponse(BaseModel):
    outcome: str
    replayed: int | None = None


async def _payload(request: Request) -> dict:
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 2 * MAX_BACKUP_BYTES:
            raise ConfigBackupError("BACKUP_TOO_LARGE", "The restore request is too large.", 413)
    try:
        value = json.loads(body)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ConfigBackupError("BACKUP_MALFORMED", "The restore request must be valid JSON.") from exc
    if not isinstance(value, dict):
        raise ConfigBackupError("BACKUP_MALFORMED", "The restore request must be an object.")
    return value


def _body_error(exc: ConfigBackupError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": exc.detail},
    )


@router.get("/config-backup")
def download_config_backup():
    """Download the portable configuration. Credentials and runtime state are omitted."""
    try:
        document = export_backup(get_store())
    except ConfigBackupError as exc:
        return _body_error(exc)
    body = json.dumps(document, separators=(",", ":"), ensure_ascii=False)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="studio-config-backup.json"'},
    )


@router.post("/config-backup/preview", response_model=BackupPreviewResponse)
async def preview_config_backup(request: Request):
    """Validate a backup and return a read-only restore plan."""
    try:
        payload = await _payload(request)
        return await run_in_threadpool(
            preview_backup,
            get_store(),
            payload.get("backup"),
            decisions=payload.get("decisions"),
            mapping=payload.get("mapping"),
        )
    except ConfigBackupError as exc:
        return _body_error(exc)


@router.post("/config-backup/reconcile", response_model=BackupReconcileResponse)
def reconcile_config_backup():
    """Finish an interrupted restore. This does not apply a backup again."""
    try:
        return reconcile_config_restore(get_store())
    except ConfigBackupError as exc:
        return _body_error(exc)


@router.post("/config-backup/apply", response_model=BackupApplyResponse)
async def apply_config_backup(request: Request):
    """Apply the exact previewed plan. A stale preview is rejected."""
    try:
        payload = await _payload(request)
        return await run_in_threadpool(
            apply_backup,
            get_store(),
            payload.get("backup"),
            plan_id=str(payload.get("plan_id") or ""),
            decisions=payload.get("decisions"),
            mapping=payload.get("mapping"),
        )
    except ConfigBackupError as exc:
        return _body_error(exc)
