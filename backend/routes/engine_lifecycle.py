"""Generic ``/api/engines/{engine_id}`` lifecycle router (refactoring plan Phase 3/4).

ONE router over the shared ``EngineLifecycle`` records that the compatibility
routers and native lifecycle registrations create (see
``routes/engine_versions.py``). Every body and error contract is produced by
the same ``lifecycle_*`` callables the legacy per-engine prefixes run, so the
two surfaces are identical BY CONSTRUCTION — pinned by
``tests/test_generic_engine_router.py`` (same-payload equality of both
surfaces for every verb, including error statuses).

Response typing follows docs/engine-surface-contract-matrix.md: the typed
models from ``backend/models/engine_contracts.py`` are attached via
``responses=`` (OpenAPI documentation only). ``response_model=`` is
deliberately NOT used — it would re-serialize and inject null keys for
fields producers legitimately omit (one family's status has no ``engine``
key), changing bytes on a brand-new surface that promises legacy equality.

Membership: a registered ``EngineLifecycle`` record. The five python-venv
engines register the shared installer machinery; the three native
llama-family engines register per-verb overrides delegating to their own
route functions (see ``EngineLifecycle`` docstring) — verbs they do not
implement answer 405 rather than being faked (no per-engine ``status``, no
engine-wide ``remove``). The audio engine stays on its flag-gated,
drift-aware router as the plan's declared extension (see matrix §7).
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, HTTPException

from backend.models.engine_contracts import (
    CancelAccepted,
    InstallAccepted,
    VenvEngineStatus,
)
from backend.routes.engine_versions import (
    LIFECYCLE,
    EngineLifecycle,
    lifecycle_cancel,
    lifecycle_check_updates,
    lifecycle_get_build_settings,
    lifecycle_install,
    lifecycle_install_source,
    lifecycle_remove,
    lifecycle_save_build_settings,
    lifecycle_status,
)


router = APIRouter()


def _lifecycle(engine_id: str) -> EngineLifecycle:
    lc = LIFECYCLE.get(engine_id)
    if lc is None:
        raise HTTPException(status_code=404, detail=f"Unknown engine: {engine_id}")
    return lc


@router.get(
    "/{engine_id}/check-updates",
    operation_id="generic_engine_check_updates",
)
async def check_updates(engine_id: str) -> Dict:
    return await lifecycle_check_updates(_lifecycle(engine_id))


@router.get(
    "/{engine_id}/status",
    operation_id="generic_engine_status",
    responses={200: {"model": VenvEngineStatus}},
)
async def status(engine_id: str) -> Dict:
    return await lifecycle_status(_lifecycle(engine_id))


@router.get("/{engine_id}/build-settings", operation_id="generic_engine_get_build_settings")
async def get_build_settings(engine_id: str) -> Dict:
    return await lifecycle_get_build_settings(_lifecycle(engine_id))


@router.put("/{engine_id}/build-settings", operation_id="generic_engine_save_build_settings")
async def save_build_settings(engine_id: str, settings: Optional[Dict] = Body(None)) -> Dict:
    return await lifecycle_save_build_settings(_lifecycle(engine_id), settings)


@router.post(
    "/{engine_id}/install",
    operation_id="generic_engine_install",
    responses={200: {"model": InstallAccepted}},
)
async def install(engine_id: str, request: Optional[Dict[str, Any]] = None) -> Dict:
    return await lifecycle_install(_lifecycle(engine_id), request)


@router.post(
    "/{engine_id}/install-source",
    operation_id="generic_engine_install_source",
    responses={200: {"model": InstallAccepted}},
)
async def install_source(engine_id: str, request: Optional[Dict[str, Any]] = None) -> Dict:
    return await lifecycle_install_source(_lifecycle(engine_id), request)


@router.post(
    "/{engine_id}/remove",
    operation_id="generic_engine_remove",
    responses={200: {"model": InstallAccepted}},
)
async def remove(engine_id: str, payload: Optional[dict] = Body(default=None)) -> Dict:
    return await lifecycle_remove(_lifecycle(engine_id), payload)


@router.post(
    "/{engine_id}/cancel",
    operation_id="generic_engine_cancel",
    responses={200: {"model": CancelAccepted}},
)
async def cancel(engine_id: str, payload: dict = Body(...)) -> Dict:
    return await lifecycle_cancel(_lifecycle(engine_id), payload)
