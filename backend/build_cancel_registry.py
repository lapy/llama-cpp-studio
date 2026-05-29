"""Backward-compatible aliases for source build cancellation."""

from backend.task_cancel_registry import (
    TaskCancelledError as BuildCancelledError,
    is_task_cancel_requested as is_build_cancel_requested,
    register_task_cancel as register_build_cancel,
    request_task_cancel as request_build_cancel,
    unregister_task_cancel as unregister_build_cancel,
)

__all__ = [
    "BuildCancelledError",
    "is_build_cancel_requested",
    "register_build_cancel",
    "request_build_cancel",
    "unregister_build_cancel",
]
