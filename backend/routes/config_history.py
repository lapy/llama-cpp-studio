"""Browse and selectively restore local configuration history."""

from typing import Any, Literal

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config_history import (
    ConfigHistoryError,
    config_history_diff,
    list_config_history,
    restore_config_history_item,
)
from backend.data_store import get_store


router = APIRouter()


class HistoryEntryResponse(BaseModel):
    id: str
    created_at: str
    document: str
    reason: str


class HistoryChangeResponse(BaseModel):
    path: str
    before: Any
    after: Any


class HistoryDiffResponse(HistoryEntryResponse):
    current_revision: str
    changes: list[HistoryChangeResponse]


class HistoryRestoreBody(BaseModel):
    kind: Literal["preference", "model", "template", "profile", "selector"]
    item_id: str = Field(min_length=1, max_length=240)
    expected_revision: str = Field(min_length=1, max_length=200)


class HistoryRestoreResponse(BaseModel):
    outcome: Literal["completed"]
    document: str
    kind: str
    item_id: str
    revision: str
    notice: str


def _error(exc: ConfigHistoryError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"code": exc.code, "detail": exc.detail},
    )


@router.get("/config-history", response_model=list[HistoryEntryResponse])
def history_entries(limit: int = Query(default=30, ge=1, le=100)):
    return list_config_history(get_store(), limit=limit)


@router.get("/config-history/{entry_id}", response_model=HistoryDiffResponse)
def history_diff(entry_id: str):
    try:
        return config_history_diff(get_store(), entry_id)
    except ConfigHistoryError as exc:
        return _error(exc)


@router.post(
    "/config-history/{entry_id}/restore",
    response_model=HistoryRestoreResponse,
)
def restore_history_item(entry_id: str, body: HistoryRestoreBody):
    try:
        return restore_config_history_item(
            get_store(),
            entry_id,
            kind=body.kind,
            item_id=body.item_id,
            expected_revision=body.expected_revision,
        )
    except ConfigHistoryError as exc:
        return _error(exc)
