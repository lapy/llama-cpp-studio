"""In-process counters for operations, queue depth, and event-loop delay."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)

#: An overrun longer than this is a stall a user feels (an API call, a stream
#: chunk or the UI's poll did not answer in time), not scheduler noise.
LOOP_STALL_WARN_SECONDS = 2.0
#: One warning per window keeps a long stall from filling the log.
LOOP_STALL_WARN_INTERVAL_SECONDS = 30.0

_metrics: Dict[str, Any] = {
    "operations_started": 0,
    "operations_succeeded": 0,
    "operations_failed": 0,
    "operations_cancelled": 0,
    "operations_interrupted": 0,
    "event_loop_delay_seconds_max": 0.0,
    "event_loop_stalls": 0,
    "sse_queue_depth_max": 0,
    "cancellation_latency_seconds_max": 0.0,
    "config_revision": 0,
}

_last_stall_warning = 0.0


def record_operation(status: str) -> None:
    key = {
        "running": "operations_started",
        "succeeded": "operations_succeeded",
        "failed": "operations_failed",
        "cancelled": "operations_cancelled",
        "interrupted": "operations_interrupted",
    }.get(status)
    if key:
        _metrics[key] = int(_metrics.get(key, 0)) + 1


def observe_loop_delay(delay_seconds: float) -> None:
    current = float(_metrics.get("event_loop_delay_seconds_max") or 0.0)
    if delay_seconds > current:
        _metrics["event_loop_delay_seconds_max"] = round(delay_seconds, 6)


def warn_loop_stall(delay_seconds: float) -> bool:
    """Log (rate-limited) when the loop was blocked long enough to be felt.

    The stall delay, the memory headroom and the build budget are the facts
    needed to tell a memory-thrashing engine build from a blocked call, and the
    proxy/API recovering on its own with no restart is otherwise invisible in
    the logs.
    """
    global _last_stall_warning

    threshold = float(os.getenv("STUDIO_LOOP_STALL_WARN_SECONDS", "") or LOOP_STALL_WARN_SECONDS)
    if delay_seconds < threshold:
        return False
    now = time.monotonic()
    if now - _last_stall_warning < LOOP_STALL_WARN_INTERVAL_SECONDS:
        return False
    _last_stall_warning = now
    _metrics["event_loop_stalls"] = int(_metrics.get("event_loop_stalls") or 0) + 1
    try:
        from backend.concurrency import KIB_PER_GIB, available_memory_kib, build_job_budget

        memory_gib = available_memory_kib() / KIB_PER_GIB
        budget = build_job_budget()
        detail = f"available memory {memory_gib:.1f} GiB, build budget {budget} jobs"
    except Exception:  # pragma: no cover - diagnostics must never raise
        detail = "host budget unavailable"
    logger.warning(
        "Event loop stalled for %.1fs (%s); the API and proxy answer late or not "
        "at all until the load drops",
        delay_seconds,
        detail,
    )
    return True


def observe_queue_depth(depth: int) -> None:
    current = int(_metrics.get("sse_queue_depth_max") or 0)
    if depth > current:
        _metrics["sse_queue_depth_max"] = depth


def observe_cancellation_latency(delay_seconds: float) -> None:
    current = float(_metrics.get("cancellation_latency_seconds_max") or 0.0)
    if delay_seconds > current:
        _metrics["cancellation_latency_seconds_max"] = round(delay_seconds, 6)


def set_config_revision(revision: int) -> None:
    _metrics["config_revision"] = int(revision)


def snapshot_metrics() -> Dict[str, Any]:
    return dict(_metrics)


async def monitor_event_loop(interval: float = 0.05) -> None:
    """Record how far a short sleep overruns, which shows event-loop stalls."""
    while True:
        started = time.perf_counter()
        await asyncio.sleep(interval)
        delay = max(0.0, time.perf_counter() - started - interval)
        observe_loop_delay(delay)
        warn_loop_stall(delay)
