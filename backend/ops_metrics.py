"""In-process counters for operations, queue depth, and event-loop delay."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict

_metrics: Dict[str, Any] = {
    "operations_started": 0,
    "operations_succeeded": 0,
    "operations_failed": 0,
    "operations_cancelled": 0,
    "operations_interrupted": 0,
    "event_loop_delay_seconds_max": 0.0,
    "sse_queue_depth_max": 0,
    "cancellation_latency_seconds_max": 0.0,
    "config_revision": 0,
}


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
        observe_loop_delay(max(0.0, time.perf_counter() - started - interval))
