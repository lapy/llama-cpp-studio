"""Helpers for cancelling long-running installer/manager operations."""

from __future__ import annotations

import asyncio
from typing import Optional


def cancel_running_operation(
    *,
    operation: Optional[str],
    current_task: Optional[asyncio.Task],
    active_process: Optional[asyncio.subprocess.Process],
) -> bool:
    if not operation:
        return False

    cancelled = False
    if active_process is not None and active_process.returncode is None:
        try:
            active_process.terminate()
        except ProcessLookupError:
            pass
        cancelled = True

    if current_task is not None and not current_task.done():
        current_task.cancel()
        cancelled = True

    return cancelled
