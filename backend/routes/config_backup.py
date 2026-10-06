"""Export and restore a versioned configuration backup."""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from backend.config_backup import (
    ConfigBackupError,
    apply_backup,
    export_backup,
    preview_backup,
    reconcile_config_restore,
)
from backend.data_store import get_store

router = APIRouter()


def _body_error(exc: ConfigBackupError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": exc.detail},
    )


@router.get("/config-backup")
async def download_config_backup():
    """Download the portable configuration. Credentials and runtime state are omitted."""
    import json

    try:
        document = export_backup(get_store())
    except ConfigBackupError as exc:
        return _body_error(exc)
    body = json.dumps(document, indent=2)
    return Response(
        content=body,
        media_type="application/json",
        headers={"Content-Disposition": 'attachment; filename="studio-config-backup.json"'},
    )


@router.post("/config-backup/preview")
async def preview_config_backup(request: Request):
    """Validate a backup and return a read-only restore plan."""
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    try:
        return preview_backup(
            get_store(),
            payload.get("backup"),
            decisions=payload.get("decisions"),
            mapping=payload.get("mapping"),
        )
    except ConfigBackupError as exc:
        return _body_error(exc)


@router.post("/config-backup/reconcile")
async def reconcile_config_backup():
    """Finish an interrupted restore. This does not apply a backup again."""
    try:
        return reconcile_config_restore(get_store())
    except ConfigBackupError as exc:
        return _body_error(exc)


@router.post("/config-backup/apply")
async def apply_config_backup(request: Request):
    """Apply the exact previewed plan. A stale preview is rejected."""
    payload = await request.json()
    if not isinstance(payload, dict):
        payload = {}
    try:
        return apply_backup(
            get_store(),
            payload.get("backup"),
            plan_id=str(payload.get("plan_id") or ""),
            decisions=payload.get("decisions"),
            mapping=payload.get("mapping"),
        )
    except ConfigBackupError as exc:
        return _body_error(exc)
