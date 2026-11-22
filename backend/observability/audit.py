"""Bounded, redacted audit trail for management mutations.

Every non-GET ``/api/*`` request appends one JSON line under
``<data>/logs/audit.jsonl``. The middleware is pure ASGI and only enqueues a
small record; a dedicated daemon thread owns the file so a slow disk never
blocks the event loop that serves inference and the UI.

The trail is deliberately narrow: timestamp, request id, method, path (never
the query string), status, duration and actor kind. Request bodies, headers,
cookies, credentials and query values are never read into the record, and a
path that embeds a secret-looking value is redacted. The queue and the line
length are bounded, so a burst cannot grow memory without limit, and the file
rotates at 5 MiB keeping five backups.

Auditing is best effort: a full queue or an I/O error drops the entry and
increments a counter. It must never fail the mutation it observes.
"""

from __future__ import annotations

import json
import math
import os
import queue
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from backend.paths import studio_data_dir

SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
AUDIT_DIRNAME = "logs"
AUDIT_FILENAME = "audit.jsonl"
#: Rotate before a write would push the active file past this size.
MAX_AUDIT_BYTES = 5 * 1024 * 1024
#: Rotated files retained next to the active file (audit.jsonl.1 ... .5).
AUDIT_BACKUP_COUNT = 5
#: A path longer than this is truncated; it cannot be a useful selector anyway.
MAX_AUDIT_PATH_CHARS = 512
#: Pending records kept before new ones are dropped instead of blocking.
MAX_AUDIT_QUEUE = 2048
#: Scope prefix for the mutation surface this middleware covers.
AUDIT_SCOPE_PREFIX = "/api"

_REDACTED = "redacted"
# A secret-looking assignment in a path (``token=``, ``api-key:`` ...). The
# query string never reaches the record; this catches values embedded in the
# path itself, such as a proxy or webhook path.
_SECRET_PATTERN = re.compile(
    r"(?i)\b((?:api[_-]?)?key|token|secret|password|authorization|bearer)"
    r"(?:([=:/])\s*|\s+)([^/&\s?#]+)"
)
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _redact_secret(match: re.Match[str]) -> str:
    separator = match.group(2) or " "
    return f"{match.group(1)}{separator}{_REDACTED}"


def default_audit_path() -> str:
    """The audit file under the configured Studio data root."""
    return os.path.join(studio_data_dir(), AUDIT_DIRNAME, AUDIT_FILENAME)


def redact_path(path: Any) -> str:
    """Return a bounded, single-line path with embedded secrets redacted."""
    text = _CONTROL_CHARS.sub("", str(path or ""))
    text = _SECRET_PATTERN.sub(_redact_secret, text)
    if len(text) > MAX_AUDIT_PATH_CHARS:
        text = text[:MAX_AUDIT_PATH_CHARS] + "..."
    return text


def utc_timestamp() -> str:
    """Timestamp with millisecond precision and a ``Z`` suffix."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def clean_record(record: dict[str, Any]) -> dict[str, Any]:
    """Reduce a record to the audited fields, bounded and redacted.

    Extra keys are dropped rather than serialized, so a caller cannot
    accidentally persist a body, header, or credential.
    """
    try:
        status = int(record.get("status") or 0)
    except (TypeError, ValueError):
        status = 0
    try:
        duration_ms = round(float(record.get("duration_ms") or 0.0), 3)
    except (TypeError, ValueError):
        duration_ms = 0.0
    if not math.isfinite(duration_ms):
        duration_ms = 0.0
    return {
        "timestamp": str(record.get("timestamp") or utc_timestamp())[:40],
        "request_id": str(record.get("request_id") or "")[:64],
        "method": str(record.get("method") or "")[:16].upper(),
        "path": redact_path(record.get("path")),
        "status": status,
        "duration_ms": duration_ms,
        "actor": str(record.get("actor") or "unknown")[:16],
    }


class _Flush:
    """Queue sentinel asking the writer to set ``done`` after earlier records."""

    __slots__ = ("done",)

    def __init__(self) -> None:
        self.done = threading.Event()


class AuditLog:
    """Bound a JSONL audit file with one background writer thread."""

    def __init__(
        self,
        path: str,
        *,
        max_bytes: int = MAX_AUDIT_BYTES,
        backup_count: int = AUDIT_BACKUP_COUNT,
        queue_size: int = MAX_AUDIT_QUEUE,
    ) -> None:
        self.path = str(path)
        self.max_bytes = max(1, int(max_bytes))
        self.backup_count = max(0, int(backup_count))
        self.queue_size = max(1, int(queue_size))
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=self.queue_size)
        self._thread: threading.Thread | None = None
        self._thread_lock = threading.Lock()
        self._size: int | None = None
        self.written = 0
        self.dropped = 0
        self.failures = 0

    def record(self, record: dict[str, Any]) -> None:
        """Enqueue one record. Never blocks and never raises."""
        try:
            line = json.dumps(clean_record(record), separators=(",", ":")) + "\n"
        except Exception:  # pragma: no cover - clean_record always serializes
            self.failures += 1
            return
        self._ensure_thread()
        try:
            self._queue.put_nowait(line)
        except queue.Full:
            self.dropped += 1

    def flush(self, timeout: float = 5.0) -> None:
        """Wait for records queued before this call to reach the file."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            return
        # Always enqueue the marker: the writer may already be appending the
        # last record, which would make an ``empty()`` check return too early.
        marker = _Flush()
        try:
            self._queue.put_nowait(marker)
        except queue.Full:
            return
        marker.done.wait(max(0.0, float(timeout)))

    def close(self, timeout: float = 5.0) -> None:
        """Flush and stop the writer thread. Used by tests and shutdown paths."""
        thread = self._thread
        if thread is None or not thread.is_alive():
            self._thread = None
            return
        self.flush(timeout)
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            return
        thread.join(max(0.0, float(timeout)))
        self._thread = None

    def _ensure_thread(self) -> None:
        thread = self._thread
        if thread is not None and thread.is_alive():
            return
        with self._thread_lock:
            thread = self._thread
            if thread is not None and thread.is_alive():
                return
            thread = threading.Thread(target=self._run, name="studio-audit", daemon=True)
            self._thread = thread
            thread.start()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            if isinstance(item, _Flush):
                item.done.set()
                continue
            try:
                self._append(item)
                self.written += 1
            except Exception:
                self.failures += 1

    def _append(self, line: str) -> None:
        data = line.encode("utf-8")
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, mode=0o700, exist_ok=True)
        if self._current_size() + len(data) > self.max_bytes:
            self._rotate()
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            written = 0
            while written < len(data):
                written += os.write(descriptor, data[written:])
        finally:
            os.close(descriptor)
        self._size = (self._size or 0) + len(data)

    def _current_size(self) -> int:
        if self._size is None:
            try:
                self._size = os.path.getsize(self.path)
            except OSError:
                self._size = 0
        return self._size

    def _rotate(self) -> None:
        if self.backup_count <= 0:
            try:
                os.remove(self.path)
            except FileNotFoundError:
                pass
            self._size = 0
            return
        for index in range(self.backup_count - 1, 0, -1):
            source = f"{self.path}.{index}"
            if os.path.exists(source):
                os.replace(source, f"{self.path}.{index + 1}")
        try:
            os.replace(self.path, f"{self.path}.1")
        except FileNotFoundError:
            pass
        self._size = 0


_default_log: AuditLog | None = None
_default_lock = threading.Lock()


def get_audit_log() -> AuditLog:
    """The process-wide audit log rooted at the current data directory."""
    global _default_log

    log = _default_log
    if log is not None:
        return log
    with _default_lock:
        if _default_log is None:
            _default_log = AuditLog(default_audit_path())
        return _default_log


def reset_audit_log() -> None:
    """Drop the process-wide log. Tests use this to isolate the audit file."""
    global _default_log

    with _default_lock:
        log = _default_log
        _default_log = None
    if log is not None:
        log.close()


def flush_audit_log(timeout: float = 2.0) -> None:
    """Flush pending audit records from the process-wide log."""
    log = _default_log
    if log is not None:
        log.flush(timeout)


def audit_stats() -> dict[str, int] | None:
    """Writer counters for a metrics scrape, without creating the log."""
    log = _default_log
    if log is None:
        return None
    return {"written": log.written, "dropped": log.dropped, "failures": log.failures}


def _is_management_mutation(scope: Scope) -> bool:
    if str(scope.get("method") or "GET").upper() in SAFE_METHODS:
        return False
    path = str(scope.get("path") or "")
    return path == AUDIT_SCOPE_PREFIX or path.startswith(f"{AUDIT_SCOPE_PREFIX}/")


class AuditMiddleware:
    """Append one bounded JSONL record per non-GET ``/api/*`` request.

    Registered outside the access-control middleware so a rejected mutation is
    still recorded (with its status and the actor kind that was presented).
    """

    def __init__(self, app: ASGIApp, audit_log: AuditLog | None = None) -> None:
        self.app = app
        self._audit_log = audit_log

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http" or not _is_management_mutation(scope):
            await self.app(scope, receive, send)
            return

        # Imported lazily so importing this middleware never pulls the access
        # policy (and its data-directory token file) into unrelated processes.
        from backend.access_policy import request_actor_kind

        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        status = 0

        async def send_with_status(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = int(message.get("status") or 0)
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        except BaseException:
            if not status:
                status = 500
            raise
        finally:
            try:
                request = Request(scope)
                actor = request_actor_kind(request)
            except Exception:  # pragma: no cover - an audit record must not fail a response
                actor = "unknown"
            try:
                (self._audit_log or get_audit_log()).record(
                    {
                        "timestamp": utc_timestamp(),
                        "request_id": request_id,
                        "method": scope.get("method") or "",
                        "path": scope.get("path") or "",
                        "status": status,
                        "duration_ms": (time.perf_counter() - started) * 1000,
                        "actor": actor,
                    }
                )
            except Exception:  # pragma: no cover - never fail the request
                pass
