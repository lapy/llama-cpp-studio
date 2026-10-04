"""Age and count limits for durable operation history.

Active work is never dropped. Recent failures stay so a restart can still
show them. Finished successes are bookkeeping and are not restored.
"""

from __future__ import annotations

import time
from typing import Any

ACTIVE_OPERATION_STATES = {"queued", "running", "cancelling"}
RECOVERABLE_OPERATION_STATES = {"failed", "cancelled", "canceled", "interrupted"}
SUCCEEDED_OPERATION_STATES = {"succeeded", "completed"}
MAX_RETAINED_TERMINAL_OPERATIONS = 200
TERMINAL_OPERATION_MAX_AGE_SECONDS = 7 * 24 * 60 * 60


def _updated_at(row: dict) -> float:
    try:
        return float(row.get("updated_at") or 0)
    except (TypeError, ValueError):
        return 0.0


def prune_operation_rows(rows: list, *, now: float | None = None) -> list:
    """Return the rows that should remain in operations.yaml."""
    current = time.time() if now is None else now
    active: list[dict] = []
    recoverable: list[dict] = []
    unknown: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "")
        if status in ACTIVE_OPERATION_STATES:
            active.append(row)
            continue
        if status in SUCCEEDED_OPERATION_STATES:
            continue
        if status in RECOVERABLE_OPERATION_STATES:
            if current - _updated_at(row) <= TERMINAL_OPERATION_MAX_AGE_SECONDS:
                recoverable.append(row)
            continue
        unknown.append(row)
    recoverable.sort(key=_updated_at, reverse=True)
    unknown.sort(key=_updated_at, reverse=True)
    terminal = (recoverable + unknown)[:MAX_RETAINED_TERMINAL_OPERATIONS]
    return active + terminal


def retention_changed(before: list, after: list) -> bool:
    """True when pruning dropped or reordered a durable row."""
    if len(before) != len(after):
        return True
    before_ids = [str(row.get("operation_id") or "") for row in before if isinstance(row, dict)]
    after_ids = [str(row.get("operation_id") or "") for row in after if isinstance(row, dict)]
    return before_ids != after_ids
