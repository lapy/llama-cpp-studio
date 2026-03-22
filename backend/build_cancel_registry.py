"""Register in-flight source builds so clients can request cancellation."""

from __future__ import annotations

import asyncio
from typing import Dict, Optional

_events: Dict[str, asyncio.Event] = {}


def register_build_cancel(task_id: str) -> asyncio.Event:
    ev = asyncio.Event()
    _events[task_id] = ev
    return ev


def unregister_build_cancel(task_id: str) -> None:
    _events.pop(task_id, None)


def request_build_cancel(task_id: str) -> bool:
    ev = _events.get(task_id)
    if ev and not ev.is_set():
        ev.set()
        return True
    return False


def is_build_cancel_requested(task_id: Optional[str]) -> bool:
    if not task_id:
        return False
    ev = _events.get(task_id)
    return bool(ev and ev.is_set())


class BuildCancelledError(Exception):
    """Raised when the user cancels a source build."""

    pass
