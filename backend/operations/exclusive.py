"""Open one side-effecting action and keep its resource until it finishes."""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Mapping, Optional

from backend.operations.action_recovery import ActionAdmissionError, bind_action_confirmation
from backend.operations.supervisor import ResourceBusyError, get_supervisor
from backend.store_io import StoreDurabilityError


def _refusal_before_effect(exc: BaseException) -> bool:
    """True when this exception, or the one it was raised from, is a finished refusal.

    Wrapping a refusal as HTTP 500 must not leave the resource locked. A later
    confirmation would only enter the same refusal again.
    """
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if getattr(current, "before_side_effect", False):
            return True
        current = current.__cause__
    return False


@asynccontextmanager
async def exclusive_action(
    kind: str,
    resource_key: str,
    *,
    detail: Optional[Mapping] = None,
    payload: Optional[Mapping] = None,
    operation_id: Optional[str] = None,
    covered_by: Optional[str] = None,
):
    """Reserve a resource, fence the effect, then finish the row.

    The caller mutates runtime or configuration inside the block. A raised
    exception leaves the row unknown so a later attempt must confirm it,
    unless the caller already stored a definite rejection. Cancellation is
    not this helper: cancelling must not finish the row.

    ``covered_by`` is the operation already executing this step. When that
    operation holds an overlapping resource, the block runs inside it and
    does not open another row. Callers must pass their own operation id,
    never a value taken from a request.
    """
    supervisor = get_supervisor()
    if covered_by and supervisor.covers(str(covered_by), resource_key):
        yield str(covered_by)
        return
    bind_action_confirmation(payload)
    operation_id = operation_id or uuid.uuid4().hex
    body = dict(detail or {})
    body["effect_started"] = False
    supervisor.start_operation(operation_id, kind, resource_key, detail=body)
    await supervisor.fence_effect_started(operation_id)
    try:
        yield operation_id
    except Exception as exc:
        # ``failed`` releases the reservation only when this rejection happened
        # before any possible side effect. A timeout or status error after
        # dispatch stays unknown. A row already stored as terminal is left as it is.
        current = supervisor._get(operation_id)
        status = str((current or {}).get("status") or "")
        if status not in {"succeeded", "failed", "cancelled", "interrupted", "unknown"}:
            if _refusal_before_effect(exc):
                supervisor.finish_operation(
                    operation_id,
                    "failed",
                    str(exc) or "The action was rejected before it changed anything.",
                )
            else:
                supervisor.finish_operation(
                    operation_id,
                    "unknown",
                    "Whether this action finished could not be established. It was not run again.",
                )
        raise
    else:
        supervisor.finish_operation(operation_id, "succeeded", "")


def exclusive_http_error(exc: BaseException):
    """Map admission and durability failures to an HTTP error, or return None."""
    from fastapi import HTTPException

    if isinstance(exc, ActionAdmissionError):
        return HTTPException(status_code=409, detail=exc.detail)
    if isinstance(exc, ResourceBusyError):
        return HTTPException(
            status_code=409,
            detail={
                "code": "ACTION_IN_FLIGHT",
                "message": str(exc),
            },
        )
    if isinstance(exc, StoreDurabilityError):
        return HTTPException(
            status_code=503 if exc.committed is False else 500,
            detail={
                "code": "ACTION_NOT_SENT",
                "committed": exc.committed,
                "message": (
                    "The action had not started. It was not run."
                    if exc.committed is False
                    else "Whether this action finished could not be established. It was not run."
                ),
            },
        )
    return None
