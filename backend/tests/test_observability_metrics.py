"""Prometheus metrics route: exposition shape, sources, and unknown omission."""

import math
import threading

from backend.observability import metrics as observability_metrics
from backend.version import APP_VERSION


def _patch_sources(
    monkeypatch,
    *,
    summary=None,
    disk=None,
    gpu=None,
    models=None,
):
    monkeypatch.setattr(observability_metrics, "_proxy_summary", lambda: summary)
    monkeypatch.setattr(observability_metrics, "_disk_usage", lambda: disk)
    monkeypatch.setattr(observability_metrics, "_model_state_counts", lambda: models)

    async def gpu_rows():
        return gpu

    monkeypatch.setattr(observability_metrics, "_gpu_rows", gpu_rows)


def _declared_types(text: str) -> dict[str, str]:
    declared = {}
    for line in text.splitlines():
        if not line.startswith("# TYPE "):
            continue
        name, _, kind = line[len("# TYPE ") :].partition(" ")
        declared[name] = kind
    return declared


def _sample_lines(text: str) -> list[tuple[str, str, float]]:
    samples = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        series, _, value = line.rpartition(" ")
        name = series.split("{", 1)[0]
        number = float(value)
        assert math.isfinite(number), line
        samples.append((name, series, number))
    return samples


def test_metrics_route_returns_prometheus_text_from_cached_services(client, monkeypatch):
    _patch_sources(
        monkeypatch,
        summary={
            "since": 1700000000.0,
            "requests": 12,
            "errors": 2,
            "client_errors": 1,
            "server_errors": 1,
            "input_tokens": 900,
            "output_tokens": 300,
            "active": 1,
            "queued": 2,
            "loading": 0,
            "streaming": 1,
            "p50_latency_ms": 40.5,
            "p95_latency_ms": 120.25,
            "p50_ttft_ms": 10.0,
            "p95_ttft_ms": 30.0,
            "unknown_usage": 3,
        },
        disk={"total": 1000, "used": 400, "free": 600},
        gpu=[{"index": 0, "name": "Fake GPU", "memory": {"total": 1024, "free": 512}}],
        models={"loaded": 1, "starting": 2, "failed": 0, "total": 3},
    )

    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-store"
    text = response.text
    assert f'studio_app_info{{version="{APP_VERSION}"}} 1' in text
    assert "studio_uptime_seconds " in text
    assert "studio_proxy_requests_total 12" in text
    assert "studio_proxy_requests_queued 2" in text
    assert "studio_proxy_latency_p95_ms 120.25" in text
    assert "studio_models_loaded 1" in text
    assert "studio_models_starting 2" in text
    assert "studio_models_managed 3" in text
    assert "studio_disk_free_bytes 600" in text
    assert 'studio_gpu_memory_total_bytes{index="0",name="Fake GPU"} 1024' in text
    assert 'studio_gpu_memory_free_bytes{index="0",name="Fake GPU"} 512' in text
    assert "studio_persistence_pending_writes " in text
    assert "studio_operations_started_total " in text

    declared = _declared_types(text)
    samples = _sample_lines(text)
    assert samples
    for name, _series, _value in samples:
        assert name in declared, f"{name} has no # TYPE line"
    assert text.endswith("\n")


def test_metrics_omits_unknown_values_instead_of_reporting_zero(client, monkeypatch):
    _patch_sources(monkeypatch, summary=None, disk=None, gpu=None, models=None)

    response = client.get("/api/metrics")

    assert response.status_code == 200
    text = response.text
    for absent in (
        "studio_proxy_requests_total",
        "studio_proxy_stats_since_timestamp_seconds",
        "studio_models_loaded",
        "studio_disk_free_bytes",
        "studio_gpu_memory_total_bytes",
        "studio_gpu_memory_free_bytes",
    ):
        assert absent not in text
    # The in-process families never depend on a probe and stay available.
    assert "studio_app_info" in text
    assert "studio_uptime_seconds" in text
    assert "studio_operations_started_total " in text
    assert "studio_persistence_pending_writes " in text


def test_metrics_renders_known_zero_but_omits_unknown_gpu_memory(client, monkeypatch):
    _patch_sources(
        monkeypatch,
        summary=None,
        disk=None,
        gpu=[
            {"index": 0, "name": "Busy GPU", "memory": {"total": 2048, "free": 0}},
            {"index": 1, "name": 'Odd "GPU"\\name', "memory": {"total": 2048, "free": 4}},
            {"index": 2, "name": "Unreported", "memory": {"total": 0, "free": 0}},
        ],
        models=None,
    )

    text = client.get("/api/metrics").text

    assert 'studio_gpu_memory_free_bytes{index="0",name="Busy GPU"} 0' in text
    assert 'studio_gpu_memory_total_bytes{index="1",name="Odd \\"GPU\\"\\\\name"} 2048' in text
    assert 'index="2"' not in text


def test_model_state_counts_reads_the_registry_snapshot(monkeypatch):
    class _Registry:
        def snapshot(self):
            return [
                {"state": "ready"},
                {"state": "loading"},
                {"state": "failed"},
                {"state": "draining"},
            ]

    class _Manager:
        registry = _Registry()

    from backend.proxy.studio import manager as studio_manager

    monkeypatch.setattr(studio_manager, "get_studio_proxy_manager", lambda: _Manager())

    counts = observability_metrics._model_state_counts()

    assert counts == {"loaded": 1, "starting": 1, "failed": 1, "total": 4}


def test_metrics_remote_mode_requires_management_authorization(client, monkeypatch):
    _patch_sources(monkeypatch, summary=None, disk=None, gpu=None, models=None)
    monkeypatch.setenv("STUDIO_ACCESS_MODE", "remote")
    monkeypatch.setenv("STUDIO_API_TOKEN", "management-secret")

    denied = client.get("/api/metrics")
    allowed = client.get("/api/metrics", headers={"Authorization": "Bearer management-secret"})

    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert "studio_app_info" in allowed.text
    assert "management-secret" not in allowed.text


async def test_metrics_reads_accounting_and_disk_off_the_event_loop(monkeypatch):
    loop_thread = threading.get_ident()
    calls: list[str] = []

    def summary():
        calls.append(f"summary:{threading.get_ident()}")
        return None

    def disk():
        calls.append(f"disk:{threading.get_ident()}")
        return None

    async def gpu_rows():
        return None

    monkeypatch.setattr(observability_metrics, "_proxy_summary", summary)
    monkeypatch.setattr(observability_metrics, "_disk_usage", disk)
    monkeypatch.setattr(observability_metrics, "_gpu_rows", gpu_rows)
    monkeypatch.setattr(observability_metrics, "_model_state_counts", lambda: None)

    await observability_metrics.collect_metrics_text()

    assert len(calls) == 2
    for call in calls:
        assert call.split(":", 1)[1] != str(loop_thread)
