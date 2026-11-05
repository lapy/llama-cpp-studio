"""Process generations for Studio's native inference proxy.

Behavior adapted from llama-swap's process/log monitoring design (MIT), using
Studio's structured LaunchSpec as the source of truth.
"""

from __future__ import annotations

import asyncio
import codecs
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
    stats_since: float = field(default_factory=time.time)
    ready: bool = False
    loading_progress: Optional[float] = None
    draining: bool = False
    inflight: int = 0
    request_count: int = 0
    error_count: int = 0
    client_error_count: int = 0
    server_error_count: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_total_ms: float = 0.0
    latency_samples: EventRing[float] = field(default_factory=lambda: EventRing(1000))
    last_used: float = field(default_factory=time.monotonic)
    logs: EventRing[str] = field(default_factory=lambda: EventRing(2000))
    exit_code: Optional[int] = None
    failure_reason: Optional[str] = None
    log_truncated_bytes: int = 0
    _terminated: bool = False
    _terminate_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)
    _watch_task: Optional[asyncio.Task] = field(default=None, repr=False)
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
        self._loads: dict[str, asyncio.Task] = {}
        self._waiters: dict[asyncio.Task, int] = {}
        self._stops: dict[str, asyncio.Task] = {}
        self._failed: dict[str, Generation] = {}
        self._request_slots: dict[str, asyncio.Semaphore] = {}
        # Generations whose process is spawned but not yet ready. Reported in
        # snapshots with llama-swap's "loading" state.
        self._starting: dict[str, Generation] = {}
        # Includes draining generations, not only the currently routed one.
        # Shutdown must never leave a replaced process group behind.
        self._generations: dict[int, Generation] = {}
        self.activities: EventRing[dict[str, Any]] = EventRing(1000)
        self.requests: EventRing[dict[str, Any]] = EventRing(100)
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
        return self.active(model_id) or self._starting.get(model_id) or self._failed.get(model_id)

    def snapshot(self) -> list[dict[str, Any]]:
        rows = [self.describe(item) for item in self._active.values()]
        for model_id, item in self._starting.items():
            if self._active.get(model_id) is not item:
                rows.append(self.describe(item))
        for model_id, item in self._failed.items():
            if model_id not in self._active and model_id not in self._starting:
                rows.append(self.describe(item))
        return rows

    @staticmethod
    def describe(item: Generation) -> dict[str, Any]:
        exited = item.process.poll()
        ready = item.ready and exited is None and not item._terminated
        state = (
            "failed"
            if item.failure_reason or (exited is not None and not item.draining)
            else "stopped"
            if item._terminated
            else "draining"
            if item.draining
            else "ready"
            if ready
            else "loading"
        )
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
            "ready": ready,
            "state": state,
            "exit_code": item.exit_code if item.exit_code is not None else exited,
            "failure_reason": item.failure_reason,
            "log_truncated_bytes": item.log_truncated_bytes,
            "loading_progress": item.loading_progress,
            "draining": item.draining,
            "inflight": item.inflight,
            "created_at": item.created_at,
            "request_count": item.request_count,
            "error_count": item.error_count,
            "client_error_count": item.client_error_count,
            "server_error_count": item.server_error_count,
            "input_bytes": item.input_bytes,
            "output_bytes": item.output_bytes,
            "input_tokens": item.input_tokens,
            "output_tokens": item.output_tokens,
            "output_tokens_per_second": (
                item.output_tokens / max(0.001, time.time() - item.stats_since)
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
        method: str = "",
        path: str = "",
    ) -> None:
        self.requests.append(
            {
                "id": uuid.uuid4().hex,
                "completed_at": time.time(),
                "model": generation.model_id,
                "engine": generation.engine_id,
                "method": method,
                "path": path,
                "status": status,
                "latency_ms": latency_ms,
                "input_bytes": input_bytes,
                "output_bytes": output_bytes,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "pid": generation.process.pid,
            }
        )
        generation.request_count += 1
        generation.error_count += int(status >= 400)
        generation.client_error_count += int(400 <= status < 500)
        generation.server_error_count += int(status >= 500)
        generation.input_bytes += max(0, input_bytes)
        generation.output_bytes += max(0, output_bytes)
        generation.input_tokens += max(0, input_tokens)
        generation.output_tokens += max(0, output_tokens)
        generation.latency_total_ms += max(0.0, latency_ms)
        generation.latency_samples.append(max(0.0, latency_ms))

    def reset_stats(self) -> None:
        for generation in self._generations.values():
            for name in (
                "request_count",
                "error_count",
                "client_error_count",
                "server_error_count",
                "input_bytes",
                "output_bytes",
                "input_tokens",
                "output_tokens",
                "latency_total_ms",
            ):
                setattr(generation, name, 0)
            generation.latency_samples.clear()
            generation.stats_since = time.time()

    async def acquire(self, model_id: str) -> Generation:
        """Share one process-owned start; cancelling a waiter does not cancel peers.

        Adapted from llama-swap v265's EnsureReady/run-loop guarantees. A stop
        owns teardown and subsequent acquires wait for it rather than launching
        a replacement underneath the drain.
        """
        while True:
            if self._closed:
                raise RegistryError("proxy registry is closed")
            stop = self._stops.get(model_id)
            if stop is not None and not stop.done():
                await asyncio.shield(stop)
                continue
            task = self._loads.get(model_id)
            if task is not None and (
                task.cancelling()
                or (task.done() and (task.cancelled() or task.exception() is not None))
            ):
                await asyncio.shield(asyncio.gather(task, return_exceptions=True))
                if self._loads.get(model_id) is task:
                    self._loads.pop(model_id, None)
                # Recheck stop/load publication after every asynchronous wait.
                continue
            if task is None:
                task = asyncio.create_task(self._load(model_id))
                self._loads[model_id] = task
            break
        self._waiters[task] = self._waiters.get(task, 0) + 1
        try:
            try:
                generation = await asyncio.shield(task)
            except asyncio.CancelledError:
                if not asyncio.current_task().cancelling():
                    raise RegistryError("engine startup was stopped") from None
                raise
            if self._closed or generation.draining or generation.process.poll() is not None:
                raise RegistryError("engine is no longer available")
            generation.inflight += 1
            generation._drained.clear()
            generation.last_used = time.monotonic()
            return generation
        finally:
            remaining = self._waiters[task] - 1
            if remaining:
                self._waiters[task] = remaining
            else:
                self._waiters.pop(task, None)
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                if self._loads.get(model_id) is task:
                    self._loads.pop(model_id, None)

    async def _load(self, model_id: str) -> Generation:
        async with self._locks.setdefault(model_id, asyncio.Lock()):
            if self._closed:
                raise RegistryError("proxy registry is closed")
            self._ensure_capacity(model_id)
            spec, revision, artifact_dir = await self._launch_factory(model_id)
            current = self.active(model_id)
            if current and current.revision == revision and not current.draining:
                return current
            self._ensure_capacity(model_id)
            replacement = await self._start(spec, revision, artifact_dir)
            old = self._active.get(model_id)
            self._active[model_id] = replacement
            self._failed.pop(model_id, None)
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
            raise RegistryError(f"maximum loaded models reached ({self.max_loaded_models})")

    def release(self, generation: Generation) -> None:
        generation.inflight = max(0, generation.inflight - 1)
        generation.last_used = time.monotonic()
        if generation.inflight == 0:
            generation._drained.set()

    async def stop(self, model_id: str) -> bool:
        task = self._stops.get(model_id)
        if task is None:
            task = asyncio.create_task(self._stop_model(model_id))
            self._stops[model_id] = task
            task.add_done_callback(
                lambda done: (
                    self._stops.pop(model_id, None) if self._stops.get(model_id) is done else None
                )
            )
        return await asyncio.shield(task)

    async def _stop_model(self, model_id: str) -> bool:
        load = self._loads.get(model_id)
        was_loading = load is not None and not load.done()
        if was_loading:
            load.cancel()
            await asyncio.gather(load, return_exceptions=True)
        async with self._locks.setdefault(model_id, asyncio.Lock()):
            generation = self._active.pop(model_id, None) or self._starting.get(model_id)
            if generation is None:
                return was_loading
            generation.draining = True
            await self._drain_and_stop(generation)
            return True

    async def close(self) -> None:
        self._closed = True
        if self._reaper:
            self._reaper.cancel()
            await asyncio.gather(self._reaper, return_exceptions=True)
            self._reaper = None
        loads = tuple(self._loads.values())
        for task in loads:
            task.cancel()
        await asyncio.gather(*loads, return_exceptions=True)
        tasks = tuple(self._background) + tuple(self._stops.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        generations = list(self._generations.values())
        self._active.clear()
        for item in generations:
            item.draining = True
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
        generation._watch_task = asyncio.create_task(self._watch_exit(generation))
        self._emit("starting", generation)
        try:
            # A launch without its durable receipt cannot be reconciled after
            # a crash. Treat receipt persistence as part of starting and kill
            # the process if it fails instead of serving an unowned runtime.
            if self._observer:
                self._observer(spec, revision, process.pid)
            await self._wait_ready(generation)
            if generation.process.poll() is not None:
                raise RegistryError(
                    f"engine exited before readiness (code {generation.process.returncode})"
                )
        except BaseException as exc:
            if not isinstance(exc, asyncio.CancelledError):
                generation.failure_reason = str(exc)[:512] or type(exc).__name__
                self._remember_failure(generation)
            else:
                generation.draining = True
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
        generation.last_used = time.monotonic()
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
        """Drain bounded pipe chunks without occupying an executor worker."""
        reader = asyncio.StreamReader(limit=65536)
        protocol = asyncio.StreamReaderProtocol(reader)
        transport = None
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        try:
            transport, _ = await asyncio.get_running_loop().connect_read_pipe(
                lambda: protocol, stream
            )
            while True:
                chunk = await reader.read(16384)
                text = decoder.decode(chunk, final=not chunk)
                pending += text
                while "\n" in pending:
                    line, pending = pending.split("\n", 1)
                    self._log_line(generation, label, line[:8192])
                    generation.log_truncated_bytes += max(0, len(line) - 8192)
                if len(pending) > 8192:
                    generation.log_truncated_bytes += len(pending) - 8192
                    pending = pending[:8192]
                if not chunk:
                    if pending:
                        self._log_line(generation, label, pending)
                    return
        finally:
            if transport:
                transport.close()

    def _log_line(self, generation: Generation, label: str, line: str):
        clean = line.rstrip()
        generation.logs.append(f"{label}: {clean}")
        progress = _loading_progress(clean)
        if progress is not None and progress != generation.loading_progress:
            generation.loading_progress = progress
            self._emit("loading_progress", generation)

    def _remember_failure(self, generation: Generation):
        generation.ready = False
        self._failed[generation.model_id] = generation
        if len(self._failed) > 100:
            self._failed.pop(next(iter(self._failed)))

    async def _watch_exit(self, generation: Generation):
        while generation.process.poll() is None:
            await asyncio.sleep(0.05)
        generation.exit_code = generation.process.returncode
        if not generation.draining and not self._closed:
            generation.failure_reason = f"Engine exited unexpectedly (code {generation.exit_code})"
            self._remember_failure(generation)
            if self._active.get(generation.model_id) is generation:
                self._active.pop(generation.model_id, None)
            self._emit("exited", generation)
            await self._terminate(generation)

    async def _drain_and_stop(self, generation: Generation) -> None:
        if generation.inflight:
            generation._drained.clear()
            try:
                await asyncio.wait_for(generation._drained.wait(), self.drain_timeout)
            except asyncio.TimeoutError:
                self._emit("drain_timeout", generation)
        await self._terminate(generation)

    async def _terminate(self, generation: Generation) -> None:
        async with generation._terminate_lock:
            if generation._terminated:
                return
            process = generation.process
            # Like llama-swap's Unix runtime, signal the process group and cap
            # shutdown even when descendants retain stdout/stderr pipes.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while process.poll() is None and time.monotonic() < deadline:
                await asyncio.sleep(0.02)
            try:
                # Also clean descendants after a wrapper/root exited first.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            while process.poll() is None:
                await asyncio.sleep(0.01)
            generation.exit_code = process.returncode
            generation._terminated = True
            generation.ready = False
            if generation._log_tasks:
                # Preserve final crash diagnostics, but never wait indefinitely
                # on a descendant retaining an inherited pipe (v265 WaitDelay).
                await asyncio.wait(generation._log_tasks, timeout=0.1)
            for task in generation._log_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*generation._log_tasks, return_exceptions=True)
            watch = generation._watch_task
            if watch and watch is not asyncio.current_task() and not watch.done():
                watch.cancel()
                await asyncio.gather(watch, return_exceptions=True)
            self._generations.pop(process.pid, None)
            if self._starting.get(generation.model_id) is generation:
                self._starting.pop(generation.model_id, None)
            if self._active.get(generation.model_id) is generation:
                self._active.pop(generation.model_id, None)
            self._emit("failed" if generation.failure_reason else "stopped", generation)

    def _emit(self, kind: str, generation: Generation) -> None:
        self.activities.append(
            {"type": kind, **self.describe(generation), "timestamp": time.time()}
        )

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
