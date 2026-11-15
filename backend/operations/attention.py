"""Read-only projection of interrupted and unknown operations.

Restart reconciliation stores an ``interrupted`` or ``unknown`` outcome for
work this process could not prove finished. Those durable rows are the
evidence a later attempt must not repeat silently, but the API only exposed
aggregate counts (``GET /api/operations/recovery``) and the SSE activity
snapshot: there was no list of the rows themselves.

This module turns the durable rows into the facts a user needs — status,
reason, time, and the actions that are already supported — and nothing more.
It never replays work, never consumes a confirmation, and never dismisses a
row. Every action it advertises points at an endpoint that already exists.
"""

from __future__ import annotations

from typing import List, Mapping, Optional

from pydantic import BaseModel, Field

from backend.operations.action_recovery import classify_action

ATTENTION_STATUSES = frozenset({"interrupted", "unknown"})
# Each advertised action must map to an existing, already-audited endpoint.
# ``view_logs``: GET /api/operations/tasks/{operation_id}/logs (read-only).
# ``dismiss``: POST /api/tasks/dismiss, offered only when the row is not an
# admission fence and the classification says a retry is already safe.
ACTION_VIEW_LOGS = "view_logs"
ACTION_DISMISS = "dismiss"


class OperationRecoveryDetail(BaseModel):
    """The decision restart reconciliation stored on the row, when present."""

    status: str = ""
    evidence: str = ""
    retry: str = ""
    message: str = ""
    replayed: int = 0


class OperationAttentionItem(BaseModel):
    operation_id: str
    task_id: str
    kind: str
    status: str
    description: str
    message: str
    resource_key: str
    evidence: str
    retry: str
    needs_confirmation: bool
    updated_at: float
    actions: List[str] = Field(default_factory=list)
    recovery: Optional[OperationRecoveryDetail] = None


class OperationAttentionResponse(BaseModel):
    operations: List[OperationAttentionItem] = Field(default_factory=list)
    count: int = 0


def attention_item(row: Mapping) -> Optional[OperationAttentionItem]:
    """One attention item, or None when the row does not need attention.

    Acknowledged rows (``confirmation_consumed``) are settled and stay out of
    the list. A row whose side effect never started is still reported: it is
    interrupted work, but it is not an admission fence, so acknowledging it is
    safe and is offered.
    """
    status = str(row.get("status") or "")
    if status not in ATTENTION_STATUSES:
        return None
    operation_id = str(row.get("operation_id") or "").strip()
    if not operation_id:
        return None
    raw_detail = row.get("detail")
    detail = dict(raw_detail) if isinstance(raw_detail, dict) else {}
    if detail.get("confirmation_consumed"):
        return None

    kind = str(row.get("kind") or "operation")
    decision = classify_action(kind, detail)
    # Mirrors the retention fence: an unresolved row must not be dropped by a
    # dismiss, because dropping it would silently admit an unproven retry.
    unresolved_fence = bool(row.get("resource_key")) and (
        status == "unknown"
        or (status == "interrupted" and detail.get("effect_started") is not False)
    )
    # The stored status wins over an inconsistent classification: an "unknown"
    # row is a fence even when some detail claims the effect never started.
    retry = "withheld" if unresolved_fence else str(decision["retry"])
    needs_confirmation = retry != "safe"
    actions = [ACTION_VIEW_LOGS]
    if not needs_confirmation:
        actions.append(ACTION_DISMISS)

    return OperationAttentionItem(
        operation_id=operation_id,
        task_id=operation_id,
        kind=kind,
        status=status,
        description=str(detail.get("description") or kind),
        message=str(row.get("message") or decision["message"]),
        resource_key=str(row.get("resource_key") or ""),
        evidence=str(decision["evidence"]),
        retry=retry,
        needs_confirmation=needs_confirmation,
        updated_at=_updated_at(row),
        actions=actions,
        recovery=_stored_recovery(detail),
    )


def attention_report(rows) -> OperationAttentionResponse:
    """Newest first, with a stable operation-id tiebreak; read-only."""
    items = [item for item in (attention_item(row) for row in rows or []) if item is not None]
    items.sort(key=lambda item: (-item.updated_at, item.operation_id))
    return OperationAttentionResponse(operations=items, count=len(items))


def _updated_at(row: Mapping) -> float:
    try:
        return float(row.get("updated_at") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stored_recovery(detail: Mapping) -> Optional[OperationRecoveryDetail]:
    stored = detail.get("recovery")
    if not isinstance(stored, dict):
        return None
    try:
        replayed = int(stored.get("replayed") or 0)
    except (TypeError, ValueError):
        replayed = 0
    return OperationRecoveryDetail(
        status=str(stored.get("status") or ""),
        evidence=str(stored.get("evidence") or ""),
        retry=str(stored.get("retry") or ""),
        message=str(stored.get("message") or ""),
        replayed=replayed,
    )
