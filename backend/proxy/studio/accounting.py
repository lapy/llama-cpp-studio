"""Persistent quota reservations and bounded, metadata-only request telemetry.

All SQLite work is invoked off the ASGI event loop. Reservations are committed
before dispatch; unfinished reservations survive crashes as conservative charges.
UTC calendar budgets and rolling minute limits include outstanding reservations.
"""

from __future__ import annotations

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

POLICY_LIMITS = (
    "input_tokens_per_minute",
    "output_tokens_per_minute",
    "daily_token_budget",
    "monthly_token_budget",
    "max_output_tokens",
    "max_concurrent_requests",
)


class AdmissionError(Exception):
    def __init__(
        self, message: str, code: str = "quota_exceeded", status: int = 429, retry_after: int = 60
    ):
        super().__init__(message)
        self.code, self.status, self.retry_after = code, status, retry_after


def validate_policy(body: dict) -> dict:
    result = {}
    for name in (*POLICY_LIMITS, "rate_limit_per_minute"):
        value = body.get(name)
        if value is None or value == "":
            result[name] = None
        elif isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10**12:
            raise ValueError(f"{name} must be a positive integer or null")
        else:
            result[name] = value
    mode = body.get("token_accounting", "auto")
    if mode not in ("auto", "exact"):
        raise ValueError("token_accounting must be auto or exact")
    result["token_accounting"] = mode
    return result


def periods(now: float) -> tuple[float, float, float, float]:
    stamp = datetime.fromtimestamp(now, timezone.utc)
    day = stamp.replace(hour=0, minute=0, second=0, microsecond=0)
    month = day.replace(day=1)
    next_month = month.replace(
        year=month.year + (month.month == 12), month=1 if month.month == 12 else month.month + 1
    )
    return day.timestamp(), month.timestamp(), day.timestamp() + 86400, next_month.timestamp()


def process_stamp(pid: int) -> str | None:
    try:
        # Linux process birth time distinguishes reused PIDs after a restart.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        try:
            os.kill(pid, 0)
            return "alive"
        except OSError:
            return None


class Accounting:
    def __init__(self, path: str):
        self.path = path
        self._owner = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._db_lock = threading.RLock()
        self._db_depth = 0
        self._admission_cache = None
        self._admission_now = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="studio-accounting")
        self._batchers: dict = {}
        self.live: dict[str, dict] = {}
        self._pending: list[dict] = []
        self._settlements: list[tuple] = []
        self.dropped_history = 0
        self._stats_since = time.time()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=FULL")
        with self._db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS reservations (
                    id TEXT PRIMARY KEY, key_id TEXT NOT NULL, started REAL NOT NULL,
                    input INTEGER NOT NULL, output INTEGER NOT NULL, active INTEGER NOT NULL,
                    known INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS reservation_key_time ON reservations(key_id, started);
                CREATE INDEX IF NOT EXISTS reservation_active ON reservations(key_id, active);
                CREATE INDEX IF NOT EXISTS reservation_estimated ON reservations(key_id, known, active, started);
                CREATE TABLE IF NOT EXISTS usage_totals (
                    key_id TEXT, period TEXT, started REAL, input INTEGER, output INTEGER,
                    PRIMARY KEY(key_id, period, started)
                );
                CREATE TABLE IF NOT EXISTS requests (
                    id TEXT PRIMARY KEY, started REAL NOT NULL, key_id TEXT, model TEXT,
                    state TEXT, status INTEGER, payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS request_time ON requests(started DESC);
                CREATE TABLE IF NOT EXISTS owners(id TEXT PRIMARY KEY, pid INTEGER, stamp TEXT);
                CREATE TABLE IF NOT EXISTS metadata(name TEXT PRIMARY KEY, value REAL);
                CREATE TABLE IF NOT EXISTS key_activity(id TEXT PRIMARY KEY, last_used REAL NOT NULL);
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {row[1] for row in db.execute("PRAGMA table_info(reservations)")}
            if "owner" not in columns:
                db.execute("ALTER TABLE reservations ADD COLUMN owner TEXT")
            if not db.execute("SELECT 1 FROM metadata WHERE name='totals_migrated'").fetchone():
                for name, modifier in (("day", "start of day"), ("month", "start of month")):
                    db.execute(
                        """INSERT OR IGNORE INTO usage_totals
                        SELECT key_id, ?, CAST(strftime('%s',started,'unixepoch',?) AS REAL),sum(input),sum(output)
                        FROM reservations GROUP BY key_id,strftime('%s',started,'unixepoch',?)""",
                        (name, modifier, modifier),
                    )
                db.execute("INSERT INTO metadata VALUES('totals_migrated',1)")
            stale = [
                row[0]
                for row in db.execute("SELECT id,pid,stamp FROM owners")
                if process_stamp(row[1]) != row[2]
            ]
            for owner in [*stale, None]:
                rows = db.execute(
                    "SELECT id,key_id,started,input,output FROM reservations WHERE active=1 AND owner IS ?",
                    (owner,),
                ).fetchall()
                for request_id, key_id, started, input_tokens, output_tokens in rows:
                    trace = {
                        "id": request_id,
                        "key_id": key_id,
                        "started_at": started,
                        "completed_at": time.time(),
                        "state": "cancelled",
                        "status": 499,
                        "error": "proxy_process_interrupted",
                        "usage_source": "reservation",
                        "input_tokens": None,
                        "output_tokens": None,
                        "input_reserved": input_tokens,
                        "output_reserved": output_tokens,
                    }
                    db.execute(
                        "INSERT OR IGNORE INTO requests VALUES(?,?,?,?,?,?,?)",
                        (
                            request_id,
                            started,
                            key_id,
                            None,
                            "cancelled",
                            499,
                            json.dumps(trace, ensure_ascii=False),
                        ),
                    )
                db.execute("UPDATE reservations SET active=0 WHERE owner IS ?", (owner,))
                db.execute("DELETE FROM owners WHERE id IS ?", (owner,))
            db.execute(
                "INSERT INTO owners VALUES(?,?,?)",
                (self._owner, os.getpid(), process_stamp(os.getpid())),
            )
            db.execute(
                "INSERT OR IGNORE INTO metadata VALUES('stats_since',?)", (self._stats_since,)
            )
            self._stats_since = db.execute(
                "SELECT value FROM metadata WHERE name='stats_since'"
            ).fetchone()[0]

    @contextmanager
    def _db(self):
        # Keep WAL open across transactions. Reopening the final connection
        # for every request forces repeated checkpoints and journal setup.
        # This lock is separate from live-state snapshots: disk sync never
        # blocks the event loop's begin/finish/update operations.
        with self._db_lock:
            outer = self._db_depth == 0
            self._db_depth += 1
            try:
                yield self._connection
                if outer:
                    self._connection.commit()
            except BaseException:
                if outer:
                    self._connection.rollback()
                raise
            finally:
                self._db_depth -= 1

    async def reserve_async(self, *args):
        return await self._submit("reserve", args)

    async def adjust_async(self, *args):
        return await self._submit("adjust", args)

    async def settle_async(self, *args):
        return await self._submit("settle", args)

    async def _submit(self, operation, args):
        loop = asyncio.get_running_loop()
        state = self._batchers.setdefault(loop, {"jobs": [], "task": None})
        future = loop.create_future()
        state["jobs"].append((operation, args, future))
        if state["task"] is None or state["task"].done():
            state["task"] = loop.create_task(self._drain_batch(state))
        return await asyncio.shield(future)

    async def _drain_batch(self, state):
        loop = asyncio.get_running_loop()
        while state["jobs"]:
            # Coalesce work already runnable on this event-loop turn. No fixed
            # delay on an otherwise idle proxy. One FULL-sync commit durably
            # admits the whole batch before any caller can forward upstream.
            await asyncio.sleep(0)
            jobs, state["jobs"] = state["jobs"][:128], state["jobs"][128:]
            try:
                results = await loop.run_in_executor(self._executor, self._transact, jobs)
            except Exception as exc:
                results = [exc] * len(jobs)
            for (_, _, future), result in zip(jobs, results):
                if not future.done():
                    if isinstance(result, Exception):
                        future.set_exception(result)
                    else:
                        future.set_result(result)

    def _transact(self, jobs):
        results = []
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            # The write transaction excludes other workers. Reuse counters
            # across consecutive admissions for the same key, not across
            # transactions. Reconciliation invalidates this short-lived cache.
            self._admission_cache = {}
            self._admission_now = time.time()
            try:
                for operation, args, _ in jobs:
                    if operation != "reserve":
                        self._admission_cache.clear()
                    try:
                        results.append(getattr(self, operation)(*args))
                    except AdmissionError as exc:
                        results.append(exc)
            finally:
                self._admission_cache = None
                self._admission_now = None
        return results

    def begin(self, trace: dict):
        with self._lock:
            self.live[trace["id"]] = trace

    def finish(self, trace: dict):
        with self._lock:
            self.live.pop(trace["id"], None)
            self._pending.append(dict(trace))
            if len(self._pending) > 1000:
                del self._pending[0]
                self.dropped_history += 1

    def defer_settlement(self, *settlement):
        with self._lock:
            self._settlements.append(settlement)

    def flush(self, retention_days: int = 7, max_rows: int = 100000):
        with self._lock:
            settlements, self._settlements = self._settlements, []
        try:
            for settlement in settlements:
                self.settle(*settlement)
        except Exception:
            with self._lock:
                self._settlements = settlements + self._settlements
            raise
        with self._lock:
            batch, self._pending = self._pending, []
        try:
            with self._db() as db:
                db.executemany(
                    "INSERT OR REPLACE INTO requests VALUES (?,?,?,?,?,?,?)",
                    [
                        (
                            r["id"],
                            r["started_at"],
                            r.get("key_id"),
                            r.get("model"),
                            r["state"],
                            r.get("status"),
                            json.dumps(r, ensure_ascii=False),
                        )
                        for r in batch
                    ],
                )
                db.execute(
                    "DELETE FROM requests WHERE started < ?",
                    (time.time() - retention_days * 86400,),
                )
                db.execute(
                    "DELETE FROM requests WHERE id IN (SELECT id FROM requests ORDER BY started DESC LIMIT -1 OFFSET ?)",
                    (max_rows,),
                )
                for row in batch:
                    if row.get("accounting_mode") == "batched_usage" and row.get("dispatched"):
                        known = (
                            row.get("input_tokens") is not None
                            and row.get("output_tokens") is not None
                        )
                        input_tokens, output_tokens = (
                            row.get("input_tokens") or 0,
                            row.get("output_tokens") or 0,
                        )
                        added = db.execute(
                            "INSERT OR IGNORE INTO reservations(id,key_id,started,input,output,active,known,owner) VALUES(?,?,?,?,?,0,?,?)",
                            (
                                row["id"],
                                row["key_id"],
                                row["started_at"],
                                input_tokens,
                                output_tokens,
                                int(known),
                                self._owner,
                            ),
                        )
                        if added.rowcount:
                            self._charge(
                                db, row["key_id"], row["started_at"], input_tokens, output_tokens
                            )
                            db.execute(
                                "INSERT INTO key_activity VALUES(?,?) ON CONFLICT(id) DO UPDATE SET last_used=max(last_used,excluded.last_used)",
                                (row["key_id"], row["started_at"]),
                            )
                # Budget history must outlive telemetry retention. Last month's
                # reservations can be removed without affecting current windows.
                _, month, _, _ = periods(time.time())
                db.execute("DELETE FROM reservations WHERE started < ? AND active=0", (month - 60,))
                db.execute(
                    "DELETE FROM usage_totals WHERE started < ? AND NOT EXISTS (SELECT 1 FROM reservations WHERE active=1 AND reservations.key_id=usage_totals.key_id AND reservations.started < ?)",
                    (month, month),
                )
        except Exception:
            with self._lock:
                combined = batch + self._pending
                self.dropped_history += max(0, len(combined) - 1000)
                self._pending = combined[-1000:]
            raise

    def reserve(self, request_id: str, key: dict, input_tokens: int, output_tokens: int) -> dict:
        key_id = key["id"]
        with self._db() as db:
            if not db.in_transaction:
                db.execute("BEGIN IMMEDIATE")
            now = self._admission_now if self._admission_cache is not None else time.time()
            day, month, day_reset, month_reset = periods(now)
            counters = (
                self._admission_cache.get(key_id) if self._admission_cache is not None else None
            )
            if counters is None:
                minute = db.execute(
                    "SELECT coalesce(sum(started>?),0),coalesce(sum(input),0),coalesce(sum(output),0) FROM reservations WHERE key_id=? AND (started>? OR active=1)",
                    (now - 60, key_id, now - 60),
                ).fetchone()
                active = db.execute(
                    "SELECT count(*) FROM reservations WHERE key_id=? AND active=1", (key_id,)
                ).fetchone()[0]
                daily = self._total(db, key_id, "day", day)
                monthly = self._total(db, key_id, "month", month)
            else:
                minute, active, daily, monthly = counters
            checks = (
                ("rate_limit_per_minute", minute[0] + 1, 60),
                ("input_tokens_per_minute", minute[1] + input_tokens, 60),
                ("output_tokens_per_minute", minute[2] + output_tokens, 60),
                ("max_concurrent_requests", active + 1, 1),
                (
                    "daily_token_budget",
                    daily + input_tokens + output_tokens,
                    max(1, int(day_reset - now)),
                ),
                (
                    "monthly_token_budget",
                    monthly + input_tokens + output_tokens,
                    max(1, int(month_reset - now)),
                ),
            )
            for name, usage, retry in checks:
                if key.get(name) and usage > key[name]:
                    raise AdmissionError(f"Key limit reached: {name}", name, retry_after=retry)
            db.execute(
                "INSERT INTO reservations(id,key_id,started,input,output,active,owner) VALUES(?,?,?,?,?,1,?)",
                (request_id, key_id, now, input_tokens, output_tokens, self._owner),
            )
            self._charge(db, key_id, now, input_tokens, output_tokens)
            db.execute(
                "INSERT INTO key_activity VALUES(?,?) ON CONFLICT(id) DO UPDATE SET last_used=excluded.last_used",
                (key_id, now),
            )
            if self._admission_cache is not None:
                self._admission_cache[key_id] = (
                    (minute[0] + 1, minute[1] + input_tokens, minute[2] + output_tokens),
                    active + 1,
                    daily + input_tokens + output_tokens,
                    monthly + input_tokens + output_tokens,
                )
            return {
                "input_reserved": input_tokens,
                "output_reserved": output_tokens,
                "remaining_requests": max(0, key["rate_limit_per_minute"] - minute[0] - 1)
                if key.get("rate_limit_per_minute")
                else None,
                "remaining_daily_tokens": max(
                    0, key["daily_token_budget"] - daily - input_tokens - output_tokens
                )
                if key.get("daily_token_budget")
                else None,
                "remaining_monthly_tokens": max(
                    0, key["monthly_token_budget"] - monthly - input_tokens - output_tokens
                )
                if key.get("monthly_token_budget")
                else None,
            }

    @staticmethod
    def _total(db, key_id: str, period: str, started: float) -> int:
        row = db.execute(
            "SELECT input+output FROM usage_totals WHERE key_id=? AND period=? AND started=?",
            (key_id, period, started),
        ).fetchone()
        return row[0] if row else 0

    @staticmethod
    def _charge(db, key_id: str, started: float, input_tokens: int, output_tokens: int):
        day, month, _, _ = periods(started)
        for name, window in (("day", day), ("month", month)):
            db.execute(
                """INSERT INTO usage_totals VALUES(?,?,?,?,?)
                ON CONFLICT(key_id,period,started) DO UPDATE SET
                input=usage_totals.input+excluded.input,output=usage_totals.output+excluded.output""",
                (key_id, name, window, input_tokens, output_tokens),
            )

    def adjust(self, request_id: str, key: dict, input_tokens: int):
        """Replace a provisional prompt reservation atomically before dispatch."""
        with self._db() as db:
            if not db.in_transaction:
                db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT key_id,started,input,output,active FROM reservations WHERE id=?",
                (request_id,),
            ).fetchone()
            if row is None or not row[4] or row[0] != key["id"]:
                raise AdmissionError(
                    "Reservation is no longer active", "reservation_unavailable", 503
                )
            delta = input_tokens - row[2]
            day, month, day_reset, month_reset = periods(row[1])
            minute = db.execute(
                "SELECT coalesce(sum(input),0) FROM reservations WHERE key_id=? AND (started>? OR active=1)",
                (row[0], time.time() - 60),
            ).fetchone()[0]
            for name, usage, retry in (
                ("input_tokens_per_minute", minute + delta, 60),
                (
                    "daily_token_budget",
                    self._total(db, row[0], "day", day) + delta,
                    max(1, int(day_reset - time.time())),
                ),
                (
                    "monthly_token_budget",
                    self._total(db, row[0], "month", month) + delta,
                    max(1, int(month_reset - time.time())),
                ),
            ):
                if key.get(name) and usage > key[name]:
                    raise AdmissionError(f"Key limit reached: {name}", name, retry_after=retry)
            self._charge(db, row[0], row[1], delta, 0)
            db.execute("UPDATE reservations SET input=? WHERE id=?", (input_tokens, request_id))
            return {
                "input_reserved": input_tokens,
                "output_reserved": row[3],
                "remaining_daily_tokens": max(
                    0, key["daily_token_budget"] - self._total(db, row[0], "day", day)
                )
                if key.get("daily_token_budget")
                else None,
                "remaining_monthly_tokens": max(
                    0, key["monthly_token_budget"] - self._total(db, row[0], "month", month)
                )
                if key.get("monthly_token_budget")
                else None,
            }

    def settle(
        self, request_id: str, input_tokens: int | None, output_tokens: int | None, dispatched: bool
    ):
        with self._db() as db:
            if not db.in_transaction:
                db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT key_id,started,input,output FROM reservations WHERE id=?", (request_id,)
            ).fetchone()
            if row is None:
                return
            if not dispatched:
                input_tokens = output_tokens = 0
            if input_tokens is not None and output_tokens is not None:
                self._charge(db, row[0], row[1], input_tokens - row[2], output_tokens - row[3])
                db.execute(
                    "UPDATE reservations SET input=?,output=?,active=0,known=1 WHERE id=?",
                    (input_tokens, output_tokens, request_id),
                )
            else:
                db.execute("UPDATE reservations SET active=0 WHERE id=?", (request_id,))

    def key_usage(self, key: dict) -> dict:
        now = time.time()
        day, month, day_reset, month_reset = periods(now)
        with self._db() as db:
            minute = db.execute(
                "SELECT coalesce(sum(started>?),0),coalesce(sum(input),0),coalesce(sum(output),0) FROM reservations WHERE key_id=? AND (started>? OR active=1)",
                (now - 60, key["id"], now - 60),
            ).fetchone()
            daily = self._total(db, key["id"], "day", day)
            monthly = self._total(db, key["id"], "month", month)
            active = db.execute(
                "SELECT count(*) FROM reservations WHERE key_id=? AND active=1", (key["id"],)
            ).fetchone()[0]
            estimated = db.execute(
                "SELECT count(*) FROM reservations WHERE key_id=? AND known=0 AND active=0 AND started>=?",
                (key["id"], month),
            ).fetchone()[0]
            activity = db.execute(
                "SELECT last_used FROM key_activity WHERE id=?", (key["id"],)
            ).fetchone()
        with self._lock:
            active += sum(
                row.get("accounting_mode") == "batched_usage" and row.get("key_id") == key["id"]
                for row in self.live.values()
            )
        return {
            "requests_per_minute": minute[0],
            "input_tokens_per_minute": minute[1],
            "output_tokens_per_minute": minute[2],
            "concurrent_requests": active,
            "daily_tokens": daily,
            "monthly_tokens": monthly,
            "daily_remaining": max(0, key["daily_token_budget"] - daily)
            if key.get("daily_token_budget")
            else None,
            "monthly_remaining": max(0, key["monthly_token_budget"] - monthly)
            if key.get("monthly_token_budget")
            else None,
            "day_resets_at": day_reset,
            "month_resets_at": month_reset,
            "estimated_requests": estimated,
            "last_used_at": activity[0] if activity else None,
        }

    def query(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        model: str = "",
        key_id: str = "",
        state: str = "",
        status: int | None = None,
        search: str = "",
    ) -> dict:
        filters, values = [], []
        for name, value in (
            ("model", model),
            ("key_id", key_id),
            ("state", state),
            ("status", status),
        ):
            if value is not None and value != "":
                filters.append(f"{name}=?")
                values.append(value)
        if search:
            filters.append("lower(payload) LIKE ? ESCAPE '\\'")
            values.append(
                "%"
                + search.lower().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                + "%"
            )
        where = " WHERE " + " AND ".join(filters) if filters else ""
        with self._db() as db:
            total = db.execute("SELECT count(*) FROM requests" + where, values).fetchone()[0]
            rows = db.execute(
                "SELECT payload FROM requests" + where + " ORDER BY started DESC LIMIT ?",
                [*values, offset + limit],
            ).fetchall()
        with self._lock:
            live = [dict(r) for r in self.live.values()]

        def matches(r):
            return all(
                r.get(name) == value
                for name, value in (
                    ("model", model),
                    ("key_id", key_id),
                    ("state", state),
                    ("status", status),
                )
                if value is not None and value != ""
            ) and (not search or search.lower() in json.dumps(r, ensure_ascii=False).lower())

        for row in live:
            if row.get("phase_started_at") and row["state"] in ("queued", "loading"):
                row["queue_ms" if row["state"] == "queued" else "loading_ms"] = max(
                    0, (time.time() - row["phase_started_at"]) * 1000
                )
            row.pop("phase_started_at", None)
        live = [r for r in live if matches(r)]
        combined = sorted(
            live + [json.loads(r[0]) for r in rows], key=lambda r: r["started_at"], reverse=True
        )
        return {"requests": combined[offset : offset + limit], "total": total + len(live)}

    def summary(self) -> dict:
        with self._lock:
            live = [dict(r) for r in self.live.values()]
        with self._db() as db:
            self._stats_since = db.execute(
                "SELECT value FROM metadata WHERE name='stats_since'"
            ).fetchone()[0]
            counts = db.execute(
                """SELECT count(*),coalesce(sum(status>=400),0),
                coalesce(sum(json_extract(payload,'$.usage_source') IN ('unknown','reservation')),0),
                coalesce(sum(json_extract(payload,'$.input_tokens')),0),
                coalesce(sum(json_extract(payload,'$.output_tokens')),0)
                FROM requests WHERE started>=?""",
                (self._stats_since,),
            ).fetchone()
            latencies = [
                r[0]
                for r in db.execute(
                    "SELECT json_extract(payload,'$.latency_ms') FROM requests WHERE started>=? AND json_extract(payload,'$.latency_ms') IS NOT NULL ORDER BY json_extract(payload,'$.latency_ms')",
                    (self._stats_since,),
                )
            ]

        def percentile(q):
            import math

            return latencies[max(0, math.ceil(len(latencies) * q) - 1)] if latencies else None

        return {
            "dropped_history": self.dropped_history,
            "since": self._stats_since,
            "requests": counts[0],
            "active": len(live),
            "queued": sum(r["state"] == "queued" for r in live),
            "loading": sum(r["state"] == "loading" for r in live),
            "streaming": sum(r["state"] == "streaming" for r in live),
            "errors": counts[1],
            "p50_latency_ms": percentile(0.5),
            "p95_latency_ms": percentile(0.95),
            "unknown_usage": counts[2],
            "input_tokens": counts[3],
            "output_tokens": counts[4],
        }

    def reset_stats(self):
        self._stats_since = time.time()
        with self._db() as db:
            db.execute("UPDATE metadata SET value=? WHERE name='stats_since'", (self._stats_since,))


_instances: dict[str, Accounting] = {}
_instances_lock = threading.Lock()


def get_accounting() -> Accounting:
    from backend.data_store import get_store

    path = os.path.join(get_store()._config_dir, "proxy-accounting.sqlite3")
    with _instances_lock:
        if path not in _instances:
            _instances[path] = Accounting(path)
        return _instances[path]
