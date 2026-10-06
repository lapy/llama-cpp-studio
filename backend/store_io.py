"""Run YAML mutations off the event loop without returning before they are durable.

Request handlers may queue a write and keep serving other work. That write is
fsynced before the HTTP response starts. A background task does not wait on
the event loop; shutdown waits until its write has finished.
"""

from __future__ import annotations

import asyncio
import json
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
_EVENTS: deque[dict[str, Any]] = deque(maxlen=16)
_EVENTS_LOCK = threading.Lock()


class StoreIoBusy(RuntimeError):
    """The persistence queue is full.

    Callers on the event loop fail instead of waiting, because waiting there
    would stall every other request. Callers off the loop wait for a free slot.
    """


class StoreDurabilityError(OSError):
    """A document write failed at a named durability boundary.

    ``committed`` is ``True`` only when this process observed ``os.replace``
    return. That is atomic replacement: readers see the complete previous
    document or the complete new one. It is not, by itself, crash durability.
    Crash durability also needs the new file synced before the replace and the
    directory entry synced after it. A later power loss can still lose a
    rename whose directory entry was not synced.

    ``committed`` is ``False`` when this process observed that the replacement
    did not happen. The previous document is unchanged.

    ``committed`` is ``"unknown"`` when this process cannot establish which of
    those happened. Callers refresh and must not repeat a side effect until
    that refresh succeeds.
    """

    def __init__(self, message: str, *, phase: str, committed: bool | str) -> None:
        super().__init__(message)
        self.phase = phase
        if committed is True or committed is False or committed == "unknown":
            self.committed = committed
        else:
            self.committed = "unknown"


QUEUE_FULL_DESCRIPTION = (
    "Persistence queue is full. Retry after the current writes finish."
)
WRITE_REPLACED_DESCRIPTION = (
    "The document was replaced, but acknowledgement failed. "
    "Refresh before trying again."
)
WRITE_UNCHANGED_DESCRIPTION = (
    "The save was not stored. The previous state is unchanged. "
    "If this was the first save, no document was written."
)
WRITE_UNKNOWN_DESCRIPTION = (
    "The save outcome could not be established. Refresh before trying again."
)
PERSISTENCE_FAILED_DESCRIPTION = (
    "A persistence error occurred. Its details were not exported."
)
SAFE_PERSISTENCE_DESCRIPTIONS = frozenset(
    {
        QUEUE_FULL_DESCRIPTION,
        WRITE_REPLACED_DESCRIPTION,
        WRITE_UNCHANGED_DESCRIPTION,
        WRITE_UNKNOWN_DESCRIPTION,
        PERSISTENCE_FAILED_DESCRIPTION,
    }
)
DURABILITY_PHASES = frozenset({"temp_write", "fsync", "replace", "acknowledge"})


def persistence_http_error(exc: BaseException) -> tuple[int, dict[str, Any]]:
    """Stable API body for queue rejection and durability failures.

    The wording is fixed. Raw exception text stays in the log.
    """
    if isinstance(exc, StoreIoBusy):
        return 503, {
            "code": "STORE_QUEUE_FULL",
            "committed": False,
            "detail": QUEUE_FULL_DESCRIPTION,
        }
    if isinstance(exc, StoreDurabilityError):
        committed = exc.committed if exc.committed in (True, False, "unknown") else "unknown"
        if committed is True:
            detail = WRITE_REPLACED_DESCRIPTION
        elif committed is False:
            detail = WRITE_UNCHANGED_DESCRIPTION
        else:
            detail = WRITE_UNKNOWN_DESCRIPTION
        body: dict[str, Any] = {
            "code": "STORE_WRITE_FAILED",
            "committed": committed,
            "detail": detail,
        }
        if exc.phase in DURABILITY_PHASES:
            body["phase"] = exc.phase
        return 500, body
    raise TypeError(f"{type(exc).__name__} is not a persistence failure")


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
    """Remember a failure or saturation without the exception text or document."""
    timestamp = datetime.now(timezone.utc).isoformat()
    if kind == "saturation" or isinstance(error, StoreIoBusy):
        entry: dict[str, Any] = {
            "kind": "saturation",
            "code": "STORE_QUEUE_FULL",
            "category": "saturation",
            "timestamp": timestamp,
            "description": QUEUE_FULL_DESCRIPTION,
        }
    elif isinstance(error, StoreDurabilityError):
        _status, body = persistence_http_error(error)
        entry = {
            "kind": "failure",
            "code": "STORE_WRITE_FAILED",
            "category": "durability",
            "timestamp": timestamp,
            "description": body["detail"],
            "committed": body["committed"],
        }
        if "phase" in body:
            entry["phase"] = body["phase"]
    else:
        entry = {
            "kind": "failure",
            "code": "PERSISTENCE_FAILED",
            "category": "unknown",
            "timestamp": timestamp,
            "description": PERSISTENCE_FAILED_DESCRIPTION,
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


async def _send_persistence_error(send: Any, exc: BaseException) -> None:
    status, payload = persistence_http_error(exc)
    body = json.dumps(payload).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


async def run_store_durable(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Run ``func`` and return only after that write has finished.

    This is the fence before a side effect. It waits on the event loop
    without blocking other tasks, and it does not leave the future on the
    response gate. A caller that handles a durability error must not continue
    into the side effect.
    """
    if getattr(_LOCAL, "in_store", False):
        return func(*args, **kwargs)
    try:
        asyncio.get_running_loop()
        on_loop = True
    except RuntimeError:
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
    if on_loop:
        await wait_for_store_writes([future])
    return future.result()


async def wait_for_store_writes(futures: list[Future]) -> None:
    pending = [future for future in futures if not future.done()]
    if not pending:
        for future in futures:
            future.result()
        return
    await asyncio.gather(*(asyncio.wrap_future(future) for future in pending))
    for future in futures:
        future.result()


def _pending_store_futures() -> list[Future]:
    with _INFLIGHT_LOCK:
        return [future for future in _INFLIGHT if not future.done()]


def drain_store_io_blocking() -> None:
    """Wait until queued YAML writes finish.

    Test teardown drops in-memory operation state. Without this wait, the next
    test can read a document that still shows the previous active row.
    """
    while True:
        pending = _pending_store_futures()
        if not pending:
            return
        for future in pending:
            future.result()


async def drain_store_io() -> None:
    """Block shutdown until every queued YAML write has finished."""
    while True:
        pending = _pending_store_futures()
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

        suppressed = False

        async def flush() -> None:
            nonlocal flushed
            if flushed:
                return
            flushed = True
            await wait_for_store_writes(gate.close())

        async def send_after_flush(message: dict) -> None:
            nonlocal suppressed
            if suppressed:
                return
            if message.get("type") == "http.response.start":
                try:
                    await flush()
                except (StoreIoBusy, StoreDurabilityError) as exc:
                    # Replacement can already have succeeded. Do not send the
                    # route's success response over that uncertainty.
                    suppressed = True
                    await _send_persistence_error(send, exc)
                    return
            await send(message)

        try:
            await self.app(scope, receive, send_after_flush)
        finally:
            try:
                await flush()
            finally:
                _GATE.reset(token)
