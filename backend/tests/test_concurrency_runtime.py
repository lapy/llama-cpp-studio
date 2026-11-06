"""Regression coverage for control-plane threading and loop responsiveness.

These tests are deliberately process-local: they use blocking fakes that release
the GIL (``time.sleep``) and assert that the API event loop keeps running while
the work happens. They do not claim a multi-core speedup for pure Python.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from backend import concurrency
from backend.routes.engine_versions import LIFECYCLE, lifecycle_status


class _SlowInstaller:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.calls = 0

    def status(self) -> dict:
        self.calls += 1
        time.sleep(self.seconds)
        return {"installed": False}


def _lifecycle(installer: _SlowInstaller) -> SimpleNamespace:
    return SimpleNamespace(
        engine_id="slow",
        verbs={},
        get_installer=lambda: installer,
        url_prefix="/api/slow",
    )


@pytest.mark.asyncio
async def test_independent_engine_status_probes_run_concurrently():
    installers = [_SlowInstaller(0.2) for _ in range(4)]

    started = time.perf_counter()
    results = await asyncio.gather(
        *(lifecycle_status(_lifecycle(installer)) for installer in installers)
    )
    elapsed = time.perf_counter() - started

    assert results == [{"installed": False}] * 4
    assert [installer.calls for installer in installers] == [1, 1, 1, 1]
    # Serial execution would take >= 0.8 s. Leave generous CI headroom.
    assert elapsed < 0.65


@pytest.mark.asyncio
async def test_slow_engine_status_does_not_stall_the_event_loop():
    probe = asyncio.create_task(lifecycle_status(_lifecycle(_SlowInstaller(0.35))))
    await asyncio.sleep(0.02)

    overruns = []
    started = time.perf_counter()
    while not probe.done() and time.perf_counter() - started < 2.0:
        tick = time.perf_counter()
        await asyncio.sleep(0.01)
        overruns.append(max(0.0, (time.perf_counter() - tick) - 0.01))
    result = await probe

    assert result == {"installed": False}
    assert overruns
    assert max(overruns) < 0.15


@pytest.mark.asyncio
async def test_blocking_pool_overlaps_independent_work():
    state = {"current": 0, "peak": 0}
    lock = threading.Lock()

    def work() -> bool:
        with lock:
            state["current"] += 1
            state["peak"] = max(state["peak"], state["current"])
        try:
            time.sleep(0.15)
            return True
        finally:
            with lock:
                state["current"] -= 1

    results = await asyncio.gather(*(concurrency.run_blocking(work) for _ in range(4)))

    assert results == [True] * 4
    assert state["peak"] >= 2


def test_blocking_worker_count_honours_override(monkeypatch):
    monkeypatch.setenv("STUDIO_BLOCKING_WORKERS", "3")
    assert concurrency.blocking_worker_count() == 3
    assert concurrency.effective_cpu_count() >= 1


def test_build_parallelism_uses_every_available_cpu(monkeypatch):
    from backend.engines.python.cmake_build import CmakeBuildHost

    host = object.__new__(CmakeBuildHost)
    monkeypatch.delenv("STUDIO_BUILD_THREADS", raising=False)
    assert host.get_optimal_build_threads() == concurrency.effective_cpu_count()

    monkeypatch.setenv("STUDIO_BUILD_THREADS", "6")
    assert host.get_optimal_build_threads() == 6


def test_admission_cache_tracks_the_settings_revision(tmp_path, monkeypatch):
    from backend import data_store, inference_auth
    from backend.data_store import DataStore

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    inference_auth.clear_policy_cache()

    assert inference_auth.request_identity({}) == {
        "id": "anonymous",
        "name": "Anonymous",
        "anonymous": True,
    }

    store.write_document(
        "settings.yaml",
        {"proxy": {"backend": "studio", "allow_anonymous": False}},
    )
    assert inference_auth.request_identity({}) is None


@pytest.mark.asyncio
async def test_engine_status_endpoint_overlaps_probes_and_keeps_liveness():
    """The Engines page requests every engine's status at once.

    Before the status probe moved to the blocking pool these requests
    serialized on the event loop: four 200 ms probes took ~800 ms and blocked
    every other request for the duration of each probe.
    """
    from backend.main import app

    original = LIFECYCLE["vllm"]
    LIFECYCLE["vllm"] = dataclasses.replace(original, get_installer=lambda: _SlowInstaller(0.2))
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
            started = time.perf_counter()
            responses = await asyncio.gather(
                *(client.get("/api/engines/vllm/status") for _ in range(4))
            )
            elapsed = time.perf_counter() - started

            gaps = []
            status = asyncio.create_task(client.get("/api/engines/vllm/status"))
            await asyncio.sleep(0.02)
            while not status.done():
                tick = time.perf_counter()
                live = await client.get("/api/live")
                gaps.append(time.perf_counter() - tick)
                assert live.status_code == 200
                await asyncio.sleep(0.01)
            assert (await status).status_code == 200
    finally:
        LIFECYCLE["vllm"] = original

    assert [response.status_code for response in responses] == [200, 200, 200, 200]
    # Four serial 200 ms probes would need >= 0.8 s.
    assert elapsed < 0.65
    assert gaps
    assert max(gaps) < 0.15
