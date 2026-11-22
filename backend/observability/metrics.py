"""Prometheus text for Studio's management process.

The exposition reads only in-process state and the observations other services
already cache. Anything that can block -- the proxy accounting query, the disk
stat, GPU probing -- runs on the blocking pool or returns the last cached
sample; an unavailable value is omitted instead of being reported as zero.

Metric families are fixed and ordered: application, proxy, models, operations,
persistence, disk, GPU. The route is authenticated with the management policy
because the values describe the operator's host and traffic.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Callable, Iterable

from backend.concurrency import run_blocking
from backend.observability.audit import audit_stats
from backend.ops_metrics import snapshot_metrics
from backend.paths import studio_data_dir
from backend.store_io import persistence_status
from backend.version import APP_VERSION

#: Wall-clock start of this process. Used only for the process metrics.
PROCESS_STARTED_AT = time.time()

#: ``(metric, snapshot key, help)`` for the in-process operation counters.
_OPERATION_COUNTERS: tuple[tuple[str, str, str], ...] = (
    (
        "studio_operations_started_total",
        "operations_started",
        "Operations admitted since this process started.",
    ),
    (
        "studio_operations_succeeded_total",
        "operations_succeeded",
        "Operations that finished successfully since this process started.",
    ),
    (
        "studio_operations_failed_total",
        "operations_failed",
        "Operations that failed since this process started.",
    ),
    (
        "studio_operations_cancelled_total",
        "operations_cancelled",
        "Operations cancelled by the operator since this process started.",
    ),
    (
        "studio_operations_interrupted_total",
        "operations_interrupted",
        "Operations interrupted by a restart since this process started.",
    ),
)

_OPERATION_GAUGES: tuple[tuple[str, str, str], ...] = (
    (
        "studio_event_loop_delay_seconds_max",
        "event_loop_delay_seconds_max",
        "Largest observed event-loop scheduling delay in seconds.",
    ),
    (
        "studio_event_loop_stalls_total",
        "event_loop_stalls",
        "Event-loop stalls long enough to warn about since this process started.",
    ),
    (
        "studio_sse_queue_depth_max",
        "sse_queue_depth_max",
        "Largest observed progress-subscription queue depth.",
    ),
    (
        "studio_cancellation_latency_seconds_max",
        "cancellation_latency_seconds_max",
        "Slowest observed cancellation handoff in seconds.",
    ),
    (
        "studio_config_revision",
        "config_revision",
        "Configuration revision currently loaded by this process.",
    ),
)

#: ``(metric, accounting summary key, help)`` for proxy request counters.
_PROXY_COUNTERS: tuple[tuple[str, str, str], ...] = (
    (
        "studio_proxy_requests_total",
        "requests",
        "Inference requests recorded by the proxy since its stats window began.",
    ),
    (
        "studio_proxy_request_errors_total",
        "errors",
        "Failed inference requests recorded by the proxy since its stats window began.",
    ),
    (
        "studio_proxy_client_errors_total",
        "client_errors",
        "Inference requests rejected with a 4xx status.",
    ),
    (
        "studio_proxy_server_errors_total",
        "server_errors",
        "Inference requests failed with a 5xx status.",
    ),
    (
        "studio_proxy_input_tokens_total",
        "input_tokens",
        "Input tokens attributed by the proxy when it knew the count.",
    ),
    (
        "studio_proxy_output_tokens_total",
        "output_tokens",
        "Output tokens attributed by the proxy when it knew the count.",
    ),
)

#: ``(metric, accounting summary key, help)`` for proxy request gauges.
_PROXY_GAUGES: tuple[tuple[str, str, str], ...] = (
    ("studio_proxy_requests_active", "active", "Inference requests in flight now."),
    ("studio_proxy_requests_queued", "queued", "Inference requests waiting for admission now."),
    ("studio_proxy_requests_loading", "loading", "Inference requests waiting for a model to load."),
    (
        "studio_proxy_requests_streaming",
        "streaming",
        "Inference responses currently streaming.",
    ),
    (
        "studio_proxy_latency_p50_ms",
        "p50_latency_ms",
        "Median completed-request latency in the proxy stats window.",
    ),
    (
        "studio_proxy_latency_p95_ms",
        "p95_latency_ms",
        "95th-percentile completed-request latency in the proxy stats window.",
    ),
    (
        "studio_proxy_ttft_p50_ms",
        "p50_ttft_ms",
        "Median time to first token in the proxy stats window.",
    ),
    (
        "studio_proxy_ttft_p95_ms",
        "p95_ttft_ms",
        "95th-percentile time to first token in the proxy stats window.",
    ),
)


class _MetricsText:
    """Accumulate Prometheus families and render the text exposition format."""

    def __init__(self) -> None:
        self._blocks: list[str] = []

    def metric(
        self,
        name: str,
        kind: str,
        help_text: str,
        samples: Iterable[tuple[dict[str, str] | None, Any]],
    ) -> None:
        rows: list[str] = []
        for labels, value in samples:
            rendered = _format_value(value)
            if rendered is None:
                continue
            rows.append(f"{name}{_render_labels(labels)} {rendered}")
        if not rows:
            return
        self._blocks.append(
            "\n".join([f"# HELP {name} {help_text}", f"# TYPE {name} {kind}", *rows])
        )

    def render(self) -> str:
        if not self._blocks:
            return ""
        return "\n\n".join(self._blocks) + "\n"


def _format_value(value: Any) -> str | None:
    """Render a finite Prometheus sample value, or ``None`` when unknown."""
    if value is None or isinstance(value, (dict, list, tuple, set)):
        return None
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    rendered = f"{number:.6f}".rstrip("0").rstrip(".")
    return rendered or "0"


def _render_labels(labels: dict[str, str] | None) -> str:
    if not labels:
        return ""
    parts = []
    for name, value in labels.items():
        escaped = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        parts.append(f'{name}="{escaped}"')
    return "{" + ",".join(parts) + "}"


async def collect_metrics_text() -> str:
    """Build the Prometheus exposition for this process."""
    proxy_summary, disk, gpu_rows = await asyncio.gather(
        _run_blocking_optional(_proxy_summary),
        _run_blocking_optional(_disk_usage),
        _gpu_rows(),
    )

    text = _MetricsText()
    _add_application_metrics(text)
    _add_proxy_metrics(text, proxy_summary)
    _add_model_metrics(text, _model_state_counts())
    _add_operation_metrics(text)
    _add_persistence_metrics(text)
    _add_audit_metrics(text, audit_stats())
    _add_disk_metrics(text, disk)
    _add_gpu_metrics(text, gpu_rows)
    return text.render()


async def _run_blocking_optional(func: Callable[[], Any]) -> Any:
    try:
        return await run_blocking(func)
    except Exception:
        return None


def _add_application_metrics(text: _MetricsText) -> None:
    text.metric(
        "studio_app_info",
        "gauge",
        "Build information for the running Studio management process.",
        [({"version": APP_VERSION}, 1)],
    )
    text.metric(
        "studio_process_start_time_seconds",
        "gauge",
        "Unix time when this management process started.",
        [(None, PROCESS_STARTED_AT)],
    )
    text.metric(
        "studio_uptime_seconds",
        "gauge",
        "Seconds since this management process started.",
        [(None, max(0.0, time.time() - PROCESS_STARTED_AT))],
    )


def _add_proxy_metrics(text: _MetricsText, summary: Any) -> None:
    if not isinstance(summary, dict):
        return
    text.metric(
        "studio_proxy_stats_since_timestamp_seconds",
        "gauge",
        "Unix time when the proxy statistics window began (reset clears it).",
        [(None, summary.get("since"))],
    )
    for name, key, help_text in _PROXY_COUNTERS:
        text.metric(name, "counter", help_text, [(None, summary.get(key))])
    for name, key, help_text in _PROXY_GAUGES:
        text.metric(name, "gauge", help_text, [(None, summary.get(key))])
    text.metric(
        "studio_proxy_usage_unknown_total",
        "counter",
        "Requests whose token usage was unknown or only reserved in the stats window.",
        [(None, summary.get("unknown_usage"))],
    )


def _model_state_counts() -> dict[str, int] | None:
    """Count loaded, starting and failed proxy generations from memory."""
    try:
        from backend.proxy.studio.manager import get_studio_proxy_manager

        registry = getattr(get_studio_proxy_manager(), "registry", None)
        rows = registry.snapshot() if registry is not None else None
    except Exception:
        return None
    if not isinstance(rows, list):
        return None
    counts = {"loaded": 0, "starting": 0, "failed": 0, "total": 0}
    for row in rows:
        if not isinstance(row, dict):
            continue
        counts["total"] += 1
        state = str(row.get("state") or "").lower()
        if state == "ready":
            counts["loaded"] += 1
        elif state in {"loading", "starting"}:
            counts["starting"] += 1
        elif state == "failed":
            counts["failed"] += 1
    return counts


def _add_model_metrics(text: _MetricsText, counts: dict[str, int] | None) -> None:
    if not isinstance(counts, dict):
        return
    text.metric(
        "studio_models_loaded",
        "gauge",
        "Proxy model generations ready to serve inference now.",
        [(None, counts.get("loaded"))],
    )
    text.metric(
        "studio_models_starting",
        "gauge",
        "Proxy model generations currently loading.",
        [(None, counts.get("starting"))],
    )
    text.metric(
        "studio_models_failed",
        "gauge",
        "Proxy model generations retained after a failed start.",
        [(None, counts.get("failed"))],
    )
    text.metric(
        "studio_models_managed",
        "gauge",
        "Proxy model generations tracked by this process (all states).",
        [(None, counts.get("total"))],
    )


def _add_operation_metrics(text: _MetricsText) -> None:
    try:
        snapshot = snapshot_metrics()
    except Exception:
        return
    if not isinstance(snapshot, dict):
        return
    for name, key, help_text in _OPERATION_COUNTERS:
        text.metric(name, "counter", help_text, [(None, snapshot.get(key))])
    for name, key, help_text in _OPERATION_GAUGES:
        text.metric(name, "gauge", help_text, [(None, snapshot.get(key))])


def _add_persistence_metrics(text: _MetricsText) -> None:
    try:
        status = persistence_status()
    except Exception:
        return
    if not isinstance(status, dict):
        return
    text.metric(
        "studio_persistence_pending_writes",
        "gauge",
        "Configuration writes accepted by the store pool and not yet durable.",
        [(None, status.get("pending_store_writes"))],
    )
    text.metric(
        "studio_persistence_max_pending_writes",
        "gauge",
        "Configuration write slots before the persistence gate rejects work.",
        [(None, status.get("max_pending_store_writes"))],
    )
    text.metric(
        "studio_persistence_saturated",
        "gauge",
        "1 when the persistence write gate is currently saturated.",
        [(None, status.get("saturated"))],
    )
    recent = status.get("recent_failures")
    text.metric(
        "studio_persistence_recent_failures",
        "gauge",
        "Recent persistence failures retained in memory (bounded history).",
        [(None, len(recent) if isinstance(recent, list) else None)],
    )


def _add_audit_metrics(text: _MetricsText, stats: Any) -> None:
    if not isinstance(stats, dict):
        return
    text.metric(
        "studio_audit_records_written_total",
        "counter",
        "Management mutation audit records written since this process started.",
        [(None, stats.get("written"))],
    )
    text.metric(
        "studio_audit_records_dropped_total",
        "counter",
        "Audit records dropped because the bounded queue was full.",
        [(None, stats.get("dropped"))],
    )
    text.metric(
        "studio_audit_write_failures_total",
        "counter",
        "Audit records that could not be written to disk.",
        [(None, stats.get("failures"))],
    )


def _add_disk_metrics(text: _MetricsText, disk: Any) -> None:
    if not isinstance(disk, dict):
        return
    text.metric(
        "studio_disk_total_bytes",
        "gauge",
        "Size of the filesystem holding the Studio data directory.",
        [(None, disk.get("total"))],
    )
    text.metric(
        "studio_disk_free_bytes",
        "gauge",
        "Free bytes on the filesystem holding the Studio data directory.",
        [(None, disk.get("free"))],
    )
    text.metric(
        "studio_disk_used_bytes",
        "gauge",
        "Used bytes on the filesystem holding the Studio data directory.",
        [(None, disk.get("used"))],
    )


def _add_gpu_metrics(text: _MetricsText, rows: Any) -> None:
    if not isinstance(rows, list):
        return
    total_samples: list[tuple[dict[str, str], int]] = []
    free_samples: list[tuple[dict[str, str], int]] = []
    for position, gpu in enumerate(rows):
        if not isinstance(gpu, dict):
            continue
        memory = gpu.get("memory") if isinstance(gpu.get("memory"), dict) else {}
        total = _non_negative_int(memory.get("total"))
        if not total:
            # A zero/unknown total means the probe did not report memory.
            continue
        labels = {
            "index": str(gpu.get("index", position)),
            "name": str(gpu.get("name") or "unknown")[:64],
        }
        total_samples.append((labels, total))
        free = _non_negative_int(memory.get("free"))
        if free is not None:
            free_samples.append((labels, free))
    text.metric(
        "studio_gpu_memory_total_bytes",
        "gauge",
        "Total memory reported by the last cached GPU observation, per device.",
        total_samples,
    )
    text.metric(
        "studio_gpu_memory_free_bytes",
        "gauge",
        "Free memory reported by the last cached GPU observation, per device.",
        free_samples,
    )


def _non_negative_int(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _proxy_summary() -> dict | None:
    """Blocking accounting summary. Runs on the blocking pool."""
    try:
        from backend.proxy.studio.accounting import get_accounting

        summary = get_accounting().summary()
    except Exception:
        return None
    return summary if isinstance(summary, dict) else None


def _disk_usage() -> dict | None:
    """Blocking stat of the data filesystem. Runs on the blocking pool."""
    try:
        import psutil
    except Exception:  # pragma: no cover - psutil is a runtime dependency
        return None
    for path in (studio_data_dir(), os.sep):
        try:
            usage = psutil.disk_usage(path)
        except Exception:
            continue
        return {
            "total": int(usage.total),
            "used": int(usage.used),
            "free": int(usage.free),
        }
    return None


async def _gpu_rows() -> list[dict] | None:
    """The last cached GPU observation, without waiting for a slow probe.

    ``initial_wait_seconds=0`` starts a background refresh if the cache is cold
    and returns immediately with the previous value (or nothing). A probe never
    blocks a scrape and an unknown device set is omitted, not reported as zero.
    """
    try:
        from backend.services.model_metadata import get_cached_gpu_info

        info = await get_cached_gpu_info(initial_wait_seconds=0)
    except Exception:
        return None
    if not isinstance(info, dict):
        return None
    if info.get("detecting") or info.get("error"):
        return None
    rows = info.get("gpus")
    if not isinstance(rows, list) or not rows:
        return None
    return [row for row in rows if isinstance(row, dict)]
