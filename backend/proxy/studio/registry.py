"""Process generations for Studio's native inference proxy.

Behavior adapted from llama-swap's process/log monitoring design (MIT), using
Studio's structured LaunchSpec as the source of truth.
"""

from __future__ import annotations

import asyncio
import math
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

import httpx

from backend.proxy.launch_spec import LaunchSpec
from backend.proxy.studio.ring import EventRing


class RegistryError(RuntimeError):
    pass


# Engines explicitly known to expose no /health route. For these only, a
# persistent 404 on /health falls back to probing "/" for liveness. Every
# engine Studio ships (llama.cpp family, vllm, sglang, lmdeploy, audio.cpp)
# serves /health that reports load completion, so this set is deliberately
# empty; add an engine id only after verifying it has no health endpoint.
ENGINES_WITHOUT_HEALTH_ROUTE: frozenset[str] = frozenset()


@dataclass
class Generation:
    model_id: str
    revision: str
    port: int
    process: subprocess.Popen[str]
    engine_id: str = ""
    created_at: float = field(default_factory=time.time)
    ready: bool = False
    loading_progress: Optional[float] = None
    draining: bool = False
    inflight: int = 0
    request_count: int = 0
    error_count: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_total_ms: float = 0.0
    latency_samples: EventRing[float] = field(default_factory=lambda: EventRing(1000))
    last_used: float = field(default_factory=time.monotonic)
    logs: EventRing[str] = field(default_factory=lambda: EventRing(2000))
    _drained: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _log_tasks: list[asyncio.Task[None]] = field(default_factory=list, repr=False)

    @property
    def upstream(self) -> str:
        return f"http://127.0.0.1:{self.port}"


LaunchFactory = Callable[[str], Awaitable[tuple[LaunchSpec, str, Optional[str]]]]


class ProcessRegistry:
    def __init__(
        self,
        launch_factory: LaunchFactory,
        *,
        readiness_timeout: float = 2700.0,
        drain_timeout: float = 30.0,
        idle_ttl: Optional[float] = None,
        max_loaded_models: Optional[int] = None,
        per_model_concurrency: Optional[int] = None,
        queue_timeout: float = 30.0,
    ) -> None:
        self._launch_factory = launch_factory
        self.readiness_timeout = readiness_timeout
        self.drain_timeout = drain_timeout
        self.idle_ttl = idle_ttl
        self.max_loaded_models = max_loaded_models
        self.per_model_concurrency = per_model_concurrency
        self.queue_timeout = queue_timeout
        self._active: dict[str, Generation] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._request_slots: dict[str, asyncio.Semaphore] = {}
        # Generations whose process is spawned but not yet ready. Reported in
        # snapshots with llama-swap's "loading" state.
        self._starting: dict[str, Generation] = {}
        # Includes draining generations, not only the currently routed one.
        # Shutdown must never leave a replaced process group behind.
        self._generations: dict[int, Generation] = {}
        self.activities: EventRing[dict[str, Any]] = EventRing(1000)
        self._closed = False
        self._reaper: Optional[asyncio.Task[None]] = None
        self._background: set[asyncio.Task[None]] = set()
        self._observer: Optional[Callable[[LaunchSpec, str, int], None]] = None

    def set_launch_observer(
        self, observer: Optional[Callable[[LaunchSpec, str, int], None]]
    ) -> None:
        """Called after a process spawns with (spec, revision, pid).

        The manager writes the durable launch receipt here so apply journals
        and restart reconciliation recognize natively-launched generations.
        """
        self._observer = observer

    def active(self, model_id: str) -> Optional[Generation]:
        generation = self._active.get(model_id)
        if generation and generation.process.poll() is None:
            return generation
        return None

    def generation(self, model_id: str) -> Optional[Generation]:
        """Return a live ready or starting generation for observation."""
        return self.active(model_id) or self._starting.get(model_id)

    def snapshot(self) -> list[dict[str, Any]]:
        rows = [self.describe(item) for item in self._active.values()]
        for model_id, item in self._starting.items():
            if self._active.get(model_id) is not item:
                rows.append(self.describe(item))
        return rows

    @staticmethod
    def describe(item: Generation) -> dict[str, Any]:
        samples = sorted(item.latency_samples.values())
        def percentile(fraction: float) -> Optional[float]:
            if not samples:
                return None
            return samples[min(len(samples) - 1, max(0, math.ceil(len(samples) * fraction) - 1))]
        return {
            "model": item.model_id,
            "engine": item.engine_id,
            "revision": item.revision,
            "pid": item.process.pid,
            "port": item.port,
            "ready": item.ready,
            "state": "ready" if item.ready else "loading",
            "loading_progress": item.loading_progress,
            "draining": item.draining,
            "inflight": item.inflight,
            "created_at": item.created_at,
            "request_count": item.request_count,
            "error_count": item.error_count,
            "input_bytes": item.input_bytes,
            "output_bytes": item.output_bytes,
            "input_tokens": item.input_tokens,
            "output_tokens": item.output_tokens,
            "output_tokens_per_second": (
                item.output_tokens / max(0.001, time.time() - item.created_at)
            ),
            "average_latency_ms": (
                item.latency_total_ms / item.request_count if item.request_count else None
            ),
            "p50_latency_ms": percentile(0.50),
            "p95_latency_ms": percentile(0.95),
        }

    async def enter_request(self, model_id: str) -> None:
        if not self.per_model_concurrency:
            return
        slot = self._request_slots.setdefault(
            model_id, asyncio.Semaphore(self.per_model_concurrency)
        )
        try:
            await asyncio.wait_for(slot.acquire(), timeout=self.queue_timeout)
        except asyncio.TimeoutError as exc:
            raise RegistryError("model request queue timed out") from exc

    def leave_request(self, model_id: str) -> None:
        if self.per_model_concurrency:
            slot = self._request_slots.get(model_id)
            if slot is not None:
                slot.release()

    def record_request(
        self,
        generation: Generation,
        *,
        status: int,
        latency_ms: float,
        input_bytes: int,
        output_bytes: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        generation.request_count += 1
        generation.error_count += int(status >= 400)
        generation.input_bytes += max(0, input_bytes)
        generation.output_bytes += max(0, output_bytes)
        generation.input_tokens += max(0, input_tokens)
        generation.output_tokens += max(0, output_tokens)
        generation.latency_total_ms += max(0.0, latency_ms)
        generation.latency_samples.append(max(0.0, latency_ms))

    async def acquire(self, model_id: str) -> Generation:
        current = self.active(model_id)
        self._ensure_capacity(model_id)
        spec, revision, artifact_dir = await self._launch_factory(model_id)
        if current and current.revision == revision and not current.draining:
            current.inflight += 1
            current.last_used = time.monotonic()
            return current
        lock = self._locks.setdefault(model_id, asyncio.Lock())
        async with lock:
            current = self.active(model_id)
            if current and current.revision == revision and not current.draining:
                current.inflight += 1
                current.last_used = time.monotonic()
                return current
            self._ensure_capacity(model_id)
            replacement = await self._start(spec, revision, artifact_dir)
            old = self._active.get(model_id)
            self._active[model_id] = replacement
            replacement.inflight = 1
            if old:
                old.draining = True
                self._spawn_background(self._drain_and_stop(old))
            self._ensure_reaper()
            return replacement

    def _ensure_capacity(self, model_id: str) -> None:
        live_models = {*self._active.keys(), *self._starting.keys()}
        if (
            self.max_loaded_models
            and model_id not in live_models
            and len(live_models) >= self.max_loaded_models
        ):
            raise RegistryError(
                f"maximum loaded models reached ({self.max_loaded_models})"
            )

    def release(self, generation: Generation) -> None:
        generation.inflight = max(0, generation.inflight - 1)
        generation.last_used = time.monotonic()
        if generation.inflight == 0:
            generation._drained.set()

    async def stop(self, model_id: str) -> bool:
        generation = self._active.pop(model_id, None)
        if generation is None:
            # A cold-starting generation is not in _active yet but is still a
            # live process group the caller asked to unload.
            generation = self._starting.get(model_id)
        if generation is None:
            return False
        generation.draining = True
        await self._drain_and_stop(generation)
        return True

    async def close(self) -> None:
        self._closed = True
        if self._reaper:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
            self._reaper = None
        tasks = tuple(self._background)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        generations = list(self._generations.values())
        self._active.clear()
        await asyncio.gather(*(self._terminate(item) for item in generations))

    async def _start(
        self, spec: LaunchSpec, revision: str, artifact_dir: Optional[str]
    ) -> Generation:
        port = _free_loopback_port()
        argv = [_resolve_arg(item, port, artifact_dir) for item in spec.argv]
        env = dict(os.environ)
        env.update({key: str(value) for key, value in spec.env.set.items()})
        for key in spec.env.unset:
            env.pop(key, None)
        process = subprocess.Popen(
            [spec.executable, *argv],
            cwd=spec.cwd or None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        generation = Generation(spec.model_id, revision, port, process, spec.engine_id)
        self._generations[process.pid] = generation
        self._starting[spec.model_id] = generation
        generation._log_tasks = [
            asyncio.create_task(self._pump(stream, generation, label))
            for stream, label in ((process.stdout, "stdout"), (process.stderr, "stderr"))
            if stream is not None
        ]
        self._emit("starting", generation)
        try:
            # A launch without its durable receipt cannot be reconciled after
            # a crash. Treat receipt persistence as part of starting and kill
            # the process if it fails instead of serving an unowned runtime.
            if self._observer:
                self._observer(spec, revision, process.pid)
            await self._wait_ready(generation)
        except BaseException:
            # Covers asyncio.CancelledError (a BaseException): a request that
            # disconnects mid-cold-start must still tree-kill the spawn, or
            # the engine is orphaned and the next request launches a second
            # generation for the same model.
            self._starting.pop(spec.model_id, None)
            cleanup = asyncio.create_task(self._terminate(generation))
            try:
                await asyncio.shield(cleanup)
            except BaseException:
                pass  # cleanup keeps running detached; the error re-raises
            raise
        generation.ready = True
        self._starting.pop(spec.model_id, None)
        self._emit("ready", generation)
        return generation

    async def _wait_ready(self, generation: Generation) -> None:
        # Only a success status on the engine's /health proves model weights
        # finished loading. A 401/404/503 there means still initializing and
        # must keep us waiting — never route early. "/" is not a readiness
        # signal for any engine (WebUIs and redirects can answer long before
        # the model loads), so it is only used as a fallback for engine ids
        # explicitly known to lack /health entirely.
        allow_root_fallback = generation.engine_id in ENGINES_WITHOUT_HEALTH_ROUTE
        deadline = time.monotonic() + self.readiness_timeout
        consecutive_missing = 0
        async with httpx.AsyncClient(timeout=1.0) as client:
            while time.monotonic() < deadline:
                code = generation.process.poll()
                if code is not None:
                    tail = "\n".join(generation.logs.values()[-20:])
                    raise RegistryError(f"engine exited with status {code}\n{tail}")
                try:
                    response = await client.get(generation.upstream + "/health")
                except httpx.HTTPError:
                    consecutive_missing = 0
                    await asyncio.sleep(0.1)
                    continue
                if 200 <= response.status_code < 300:
                    return
                if allow_root_fallback and response.status_code == 404:
                    consecutive_missing += 1
                    if consecutive_missing % 50 == 1:
                        try:
                            root = await client.get(generation.upstream + "/")
                        except httpx.HTTPError:
                            root = None
                        # Strictly 2xx: a 3xx redirect answers before the
                        # model has finished loading just as often as after.
                        if root is not None and root.status_code < 300:
                            return
                else:
                    consecutive_missing = 0
                await asyncio.sleep(0.1)
        raise RegistryError(f"engine readiness timed out after {self.readiness_timeout:g}s")

    async def _pump(self, stream: Any, generation: Generation, label: str) -> None:
        while True:
            line = await asyncio.to_thread(stream.readline)
            if not line:
                return
            clean = line.rstrip()
            generation.logs.append(f"{label}: {clean}")
            progress = _loading_progress(clean)
            if progress is not None and progress != generation.loading_progress:
                generation.loading_progress = progress
                self._emit("loading_progress", generation)

    async def _drain_and_stop(self, generation: Generation) -> None:
        if generation.inflight:
            generation._drained.clear()
            try:
                await asyncio.wait_for(generation._drained.wait(), self.drain_timeout)
            except asyncio.TimeoutError:
                self._emit("drain_timeout", generation)
        await self._terminate(generation)

    async def _terminate(self, generation: Generation) -> None:
        process = generation.process
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(asyncio.to_thread(process.wait), 5.0)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await asyncio.to_thread(process.wait)
        for task in generation._log_tasks:
            if not task.done():
                task.cancel()
        self._generations.pop(process.pid, None)
        if self._starting.get(generation.model_id) is generation:
            self._starting.pop(generation.model_id, None)
        self._emit("stopped", generation)

    def _emit(self, kind: str, generation: Generation) -> None:
        self.activities.append({"type": kind, **self.describe(generation)})

    def _ensure_reaper(self) -> None:
        if self.idle_ttl and (self._reaper is None or self._reaper.done()):
            self._reaper = asyncio.create_task(self._reap_idle())

    def _spawn_background(self, coroutine: Awaitable[None]) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _reap_idle(self) -> None:
        while not self._closed:
            await asyncio.sleep(min(max(self.idle_ttl or 1.0, 0.1), 5.0))
            now = time.monotonic()
            expired = [
                model_id
                for model_id, generation in self._active.items()
                if not generation.inflight
                and now - generation.last_used >= (self.idle_ttl or float("inf"))
            ]
            for model_id in expired:
                await self.stop(model_id)


def _free_loopback_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _resolve_arg(item: object, port: int, artifact_dir: Optional[str]) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping) and item.get("runtime") == "port":
        return str(port)
    if isinstance(item, Mapping) and item.get("artifact") and artifact_dir:
        name = str(item["artifact"])
        candidate = os.path.realpath(os.path.join(artifact_dir, name))
        root = os.path.realpath(artifact_dir)
        if candidate != root and not candidate.startswith(root + os.sep):
            raise RegistryError("artifact path escapes generation")
        return candidate
    raise RegistryError(f"unsupported launch argument: {item!r}")


_PERCENT_PROGRESS = re.compile(r"(?<![\d.])(\d{1,3}(?:\.\d+)?)\s*%")
_FRACTION_PROGRESS = re.compile(r"(?:load|loading)[^\d]{0,20}(\d+)\s*/\s*(\d+)", re.I)


def _loading_progress(line: str) -> Optional[float]:
    """Parse common engine progress lines without coupling to one logger."""
    match = _PERCENT_PROGRESS.search(line)
    if match:
        return min(100.0, float(match.group(1)))
    match = _FRACTION_PROGRESS.search(line)
    if match and int(match.group(2)) > 0:
        return min(100.0, 100.0 * int(match.group(1)) / int(match.group(2)))
    return None
