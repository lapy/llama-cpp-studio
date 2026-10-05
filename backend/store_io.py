"""Run YAML mutations off the event loop without returning before they are durable.

Request handlers may queue a write and keep serving other work. That write is
fsynced before the HTTP response starts. A background task does not wait on
the event loop; shutdown waits until its write has finished.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# One worker performs the fsync. This caps queued work plus that worker so a
# burst of background writes cannot grow the submission queue without bound.
MAX_PENDING_STORE_WRITES = 32

_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="store-io")
_LOCAL = threading.local()
_INFLIGHT: set[Future] = set()
_INFLIGHT_LOCK = threading.Lock()
_SLOTS = threading.BoundedSemaphore(MAX_PENDING_STORE_WRITES)
_PENDING = 0
_PENDING_LOCK = threading.Lock()
_GATE: ContextVar[Optional["StoreGate"]] = ContextVar("store_io_gate", default=None)
_EVENTS: deque[dict[str, str]] = deque(maxlen=16)
_EVENTS_LOCK = threading.Lock()


class StoreIoBusy(RuntimeError):
    """The persistence queue is full.

    Callers on the event loop fail instead of waiting, because waiting there
    would stall every other request. Callers off the loop wait for a free slot.
    """


class StoreGate:
    """Writes from the request task, closed before the response is sent."""

    def __init__(self, owner: Optional[asyncio.Task] = None) -> None:
        self.owner = owner
        self._lock = threading.Lock()
        self.accepting = True
        self.futures: list[Future] = []

    def add(self, future: Future) -> bool:
        with self._lock:
            if not self.accepting:
                return False
            self.futures.append(future)
            return True

    def close(self) -> list[Future]:
        with self._lock:
            self.accepting = False
            return list(self.futures)


def _track(future: Future) -> None:
    with _INFLIGHT_LOCK:
        _INFLIGHT.add(future)

    def _done(finished: Future) -> None:
        with _INFLIGHT_LOCK:
            _INFLIGHT.discard(finished)

    future.add_done_callback(_done)


def pending_store_writes() -> int:
    """Writes accepted by the pool and not yet finished."""
    with _PENDING_LOCK:
        return _PENDING


def _record_persistence_event(kind: str, error: BaseException | None = None) -> None:
    """Remember a failure or saturation. The YAML document is not stored."""
    message = ""
    exception_type = "StoreIoBusy" if kind == "saturation" else "Exception"
    if error is not None:
        exception_type = type(error).__name__
        message = " ".join(str(error).split())[:500]
    elif kind == "saturation":
        message = "Persistence queue is full"
    entry = {
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "exception_type": exception_type,
        "message": message,
    }
    with _EVENTS_LOCK:
        _EVENTS.append(entry)


def clear_persistence_events() -> None:
    """Drop the in-memory failure ring. Tests use this so events do not leak."""
    with _EVENTS_LOCK:
        _EVENTS.clear()


def persistence_status() -> dict[str, Any]:
    """Queue depth and recent failures. Saturated is true while the cap is held."""
    with _EVENTS_LOCK:
        events = list(_EVENTS)
    failures = [event for event in events if event["kind"] == "failure"]
    saturations = [event for event in events if event["kind"] == "saturation"]
    pending = pending_store_writes()
    return {
        "pending_store_writes": pending,
        "max_pending_store_writes": MAX_PENDING_STORE_WRITES,
        "saturated": pending >= MAX_PENDING_STORE_WRITES,
        "latest_failure": failures[-1] if failures else None,
        "recent_failures": failures[-8:],
        "recent_saturation": saturations[-8:],
    }


def _acquire_slot(*, blocking: bool) -> bool:
    global _PENDING
    if blocking:
        _SLOTS.acquire()
    elif not _SLOTS.acquire(blocking=False):
        return False
    with _PENDING_LOCK:
        _PENDING += 1
    return True


def _release_slot(_future: Future | None = None) -> None:
    global _PENDING
    with _PENDING_LOCK:
        _PENDING = max(0, _PENDING - 1)
    _SLOTS.release()


def run_store(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``func`` on the single store thread.

    Off the event loop, the caller waits so the write is finished before
    return. On the event loop, waiting would stall every other request, so
    the write is only waited for by the request gate or by shutdown. A full
    queue waits off the loop and raises :class:`StoreIoBusy` on the loop.
    """
    if getattr(_LOCAL, "in_store", False):
        return func(*args, **kwargs)
    try:
        loop = asyncio.get_running_loop()
        on_loop = True
    except RuntimeError:
        loop = None
        on_loop = False

    if not _acquire_slot(blocking=not on_loop):
        busy = StoreIoBusy(
            "Persistence queue is full; retry after the current writes finish"
        )
        _record_persistence_event("saturation", busy)
        raise busy

    def wrapped() -> Any:
        _LOCAL.in_store = True
        try:
            return func(*args, **kwargs)
        except Exception as error:
            # Record before the future is marked done. future.result() can
            # wake the caller before a completion callback runs.
            logger.error("store write failed: %s", error)
            _record_persistence_event("failure", error)
            raise
        finally:
            _LOCAL.in_store = False

    try:
        future = _POOL.submit(wrapped)
    except Exception:
        _release_slot()
        raise
    future.add_done_callback(_release_slot)
    _track(future)
    if not on_loop:
        return future.result()
    gate = _GATE.get()
    # Only the request task's own writes delay the response. A child task
    # inherits the gate, but waiting for it here would freeze the loop, and
    # attaching it would make the response depend on later background work.
    if gate is not None and gate.owner is asyncio.current_task(loop) and gate.add(future):
        return None
    return None


async def wait_for_store_writes(futures: list[Future]) -> None:
    pending = [future for future in futures if not future.done()]
    if not pending:
        for future in futures:
            future.result()
        return
    await asyncio.gather(*(asyncio.wrap_future(future) for future in pending))
    for future in futures:
        future.result()


async def drain_store_io() -> None:
    """Block shutdown until every queued YAML write has finished."""
    while True:
        with _INFLIGHT_LOCK:
            pending = [future for future in _INFLIGHT if not future.done()]
        if not pending:
            return
        await wait_for_store_writes(pending)


class StoreIoMiddleware:
    """Fsync queued store writes before the first response byte."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        gate = StoreGate(asyncio.current_task())
        token = _GATE.set(gate)
        flushed = False

        async def flush() -> None:
            nonlocal flushed
            if flushed:
                return
            flushed = True
            await wait_for_store_writes(gate.close())

        async def send_after_flush(message: dict) -> None:
            if message.get("type") == "http.response.start":
                await flush()
            await send(message)

        try:
            await self.app(scope, receive, send_after_flush)
        finally:
            try:
                await flush()
            finally:
                _GATE.reset(token)
