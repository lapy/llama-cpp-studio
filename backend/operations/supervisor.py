"""In-process supervisor for builds, downloads, and installs.

Restart reconciles durable rows with what this process can still observe.
It marks unresolved work interrupted or unknown, drops in-memory reservations,
and does not start that work again. A terminal row already stored stays as it
was. Only operations explicitly marked resumable are eligible for a later
manual continue; restart itself does not replay them.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "interrupted", "unknown"}
ACTIVE_STATES = {"queued", "running", "cancelling"}

# What restart is allowed to conclude about a side effect.
# "unproven" means a missing result file, a dead process, or a partial
# install directory does not prove the side effect did not happen. An active
# row of that kind becomes "unknown" and is not replayed.
# No current kind has reliable negative evidence, so a missing file is never
# "interrupted". A result file that exists only shows a result was recorded;
# an active row is still unknown rather than succeeded.
# "interrupted" remains a stored status for a future kind that can prove the
# side effect did not occur, and for rows already saved with that status.
# A kind added later stays "unproven" until it is listed here with evidence
# that the side effect did not happen. Absence from this map is not proof.
OPERATION_COMPLETION_EVIDENCE = {
    "build": "unproven",
    "download": "unproven",
    "install": "unproven",
    "install_source": "unproven",
    "remove": "unproven",
    "sync": "unproven",
    "sync_source": "unproven",
    "update": "unproven",
    "runtime_apply": "unproven",
    "operation": "unproven",
}


class ResourceBusyError(RuntimeError):
    def __init__(self, resource_key: str, owner: str):
        super().__init__(
            f"{resource_key} is already in use by operation {owner}"
        )
        self.resource_key = resource_key
        self.owner = owner


class OperationSupervisor:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._resources: Dict[str, str] = {}
        # Latest record for this process. Disk can lag behind a queued fsync,
        # so a follow-up read must not rebuild the row from an older file.
        self._records: Dict[str, dict] = {}
        # Operation ids whose latest disk write has not finished. Those rows
        # stay in memory so a read cannot revive the older file.
        self._pending_writes: Dict[str, int] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._cancellers: Dict[str, Callable[[], Awaitable[None]]] = {}
        self._cleanup_tasks: set[asyncio.Task] = set()
        self._last_reconcile: Optional[dict] = None

    def start_operation(
        self,
        operation_id: str,
        kind: str,
        resource_key: Optional[str] = None,
        *,
        resumable: bool = False,
        detail: Optional[dict] = None,
    ) -> dict:
        record = {
            "operation_id": operation_id,
            "kind": kind,
            "status": "running",
            "resource_key": resource_key or "",
            "resumable": bool(resumable),
            "detail": detail or {},
            "message": "",
            "updated_at": time.time(),
        }
        confirmed_id = self._admit_build_or_install(kind, resource_key, detail)
        claimed_keys = self._claim_keys(operation_id, resource_key, detail)
        companion = self._confirmation_row(confirmed_id) if confirmed_id else None
        try:
            self._persist(record, companion)
        except Exception:
            if claimed_keys and not self._kept_submission(operation_id, record):
                for key in claimed_keys:
                    self._release_claim(key, operation_id)
            raise
        from backend.ops_metrics import record_operation

        record_operation("running")
        logger.info(
            "operation started",
            extra={
                "operation_id": operation_id,
                "task_id": operation_id,
                "engine": (detail or {}).get("engine"),
                "model_id": (detail or {}).get("model_id"),
            },
        )
        return record

    def _claim_keys(self, operation_id: str, resource_key: Optional[str], detail: Optional[dict]) -> list:
        from backend.operations.action_recovery import keys_overlap, resource_names

        depends = (detail or {}).get("depends_on")
        keys = resource_names(resource_key or "", depends)
        claimed = []
        with self._lock:
            for key in keys:
                owner = next(
                    (
                        owner
                        for held, owner in self._resources.items()
                        if keys_overlap(key, held) and owner != operation_id
                    ),
                    None,
                )
                if owner and owner != operation_id:
                    for claimed_key in claimed:
                        if self._resources.get(claimed_key) == operation_id:
                            self._resources.pop(claimed_key, None)
                    raise ResourceBusyError(key, owner)
                if key not in self._resources:
                    self._resources[key] = operation_id
                    claimed.append(key)
        return claimed

    def _admit_build_or_install(
        self,
        kind: str,
        resource_key: Optional[str],
        detail: Optional[dict] = None,
    ) -> Optional[str]:
        from backend.operations.action_recovery import (
            _GATED_KINDS,
            ActionAdmissionError,
            admit_action,
            current_confirmation,
        )

        if kind not in _GATED_KINDS or not resource_key:
            return None
        from backend.data_store import get_store

        confirmation = current_confirmation() or {}
        # The store queue may not yet contain this process's latest rows.
        rows = {str(row.get("operation_id")): row for row in get_store().list_operations()}
        with self._lock:
            rows.update({key: dict(row) for key, row in self._records.items()})
        decision = admit_action(
            list(rows.values()),
            resource_key or "",
            confirm_operation_id=confirmation.get("operation_id"),
            confirm_state=confirmation.get("state_token"),
            depends_on=(detail or {}).get("depends_on"),
        )
        if not decision["admit"]:
            if decision.get("code") == "ACTION_IN_FLIGHT":
                raise ResourceBusyError(
                    resource_key or "",
                    str(decision.get("operation_id") or ""),
                )
            raise ActionAdmissionError(decision)
        if decision.get("retry") == "confirmed":
            return str(decision.get("operation_id") or "") or None
        return None

    def _consume_confirmation(self, operation_id: str) -> None:
        """Queue a token change. A later durable fence waits behind this write."""
        current = self._confirmation_row(operation_id)
        if current is None:
            return
        self._persist(current)

    async def consume_action_confirmation(self, operation_id: str) -> None:
        """Durably change the confirmed row before another attempt is dispatched."""
        from backend.store_io import run_store_durable

        current = self._confirmation_row(operation_id)
        if current is None:
            return
        await self._persist_durable(current, run_store_durable)

    def _confirmation_row(self, operation_id: str) -> Optional[dict]:
        from backend.data_store import get_store

        current = None
        for row in get_store().list_operations():
            if str(row.get("operation_id") or "") == operation_id:
                current = dict(row)
                break
        if not current:
            return None
        detail = dict(current.get("detail") or {})
        detail["confirmation_consumed"] = True
        current["detail"] = detail
        current["updated_at"] = time.time()
        return current

    async def fence_effect_started(self, operation_id: str) -> None:
        """Durably record that the effect may have started, before any launch.

        ``committed is False`` finishes the row as not started. Any other
        durability failure finishes it unknown. Either way the caller must
        not launch the process or send the request.
        """
        from backend.store_io import StoreDurabilityError

        try:
            await self.note_effect_started(operation_id)
        except StoreDurabilityError as exc:
            if exc.committed is False:
                self.finish_operation(
                    operation_id,
                    "interrupted",
                    "The action had not started. It was not run again.",
                )
            else:
                current = self._get(operation_id)
                if current is not None:
                    detail = dict(current.get("detail") or {})
                    detail["effect_started"] = True
                    current = dict(current)
                    current["detail"] = detail
                    with self._lock:
                        self._records[operation_id] = current
                self.finish_operation(
                    operation_id,
                    "unknown",
                    "Whether this action finished could not be established. It was not run again.",
                )
            raise

    async def note_effect_started(self, operation_id: str) -> None:
        """Durably record that the side effect may have started.

        Returns only after that replacement has finished. Callers must not
        send the request or launch the process if this raises. A crash before
        this replacement can leave ``effect_started: false``. A crash after
        it stays unknown, including when the process was not actually launched.
        """
        from backend.store_io import run_store_durable

        current = self._get(operation_id)
        if not current:
            raise KeyError(operation_id)
        detail = dict(current.get("detail") or {})
        detail["effect_started"] = True
        current = dict(current)
        current["detail"] = detail
        current["updated_at"] = time.time()
        await self._persist_durable(current, run_store_durable)

    def note_cancellation(self, operation_id: str) -> None:
        """Record that cancellation was requested without releasing the resource.

        The row stays active until a later finish observes that the work has
        stopped. A second request leaves this state unchanged.
        """
        current = self._get(operation_id)
        if not current or str(current.get("status") or "") in TERMINAL_STATES:
            return
        if str(current.get("status") or "") == "cancelling":
            return
        current = dict(current)
        current["status"] = "cancelling"
        current["message"] = (
            "Cancellation was requested. Termination has not been verified."
        )
        current["updated_at"] = time.time()
        self._persist(current)

    def finish_operation(
        self,
        operation_id: str,
        status: str,
        message: str = "",
    ) -> None:
        if status not in TERMINAL_STATES:
            status = "failed"
        with self._lock:
            release = [
                key for key, owner in self._resources.items() if owner == operation_id
            ]
            for key in release:
                self._resources.pop(key, None)
        current = self._get(operation_id) or {
            "operation_id": operation_id,
            "kind": "operation",
            "resource_key": "",
            "resumable": False,
            "detail": {},
        }
        current["status"] = status
        current["message"] = message
        current["updated_at"] = time.time()
        try:
            self._persist(current)
        except Exception:
            # A rejected terminal write never reached disk. Put the resource
            # back when the running row was restored, so it cannot sit on disk
            # with no owner. A write that was queued keeps the release.
            if not self._kept_submission(operation_id, current):
                with self._lock:
                    for key in release:
                        self._resources.setdefault(key, operation_id)
            raise
        from backend.ops_metrics import record_operation

        record_operation(status)

    def note_known(self, operation_id: str) -> bool:
        with self._lock:
            return operation_id in set(self._resources.values()) or bool(self._get(operation_id))

    def covers(self, operation_id: str, resource_key: str) -> bool:
        """True when this active operation already holds *resource_key*.

        A build owns ``engine:audio_cpp:/install``. Activation of that same
        engine is part of the build, so it must not open a second reservation
        on ``engine:audio_cpp``.
        """
        from backend.operations.action_recovery import keys_overlap

        operation_id = str(operation_id or "").strip()
        resource_key = str(resource_key or "").strip()
        if not operation_id or not resource_key:
            return False
        with self._lock:
            owned = [
                key for key, owner in self._resources.items() if owner == operation_id
            ]
            record = self._records.get(operation_id)
        if not owned:
            return False
        if record is not None and str(record.get("status") or "") not in ACTIVE_STATES:
            return False
        return any(keys_overlap(resource_key, key) for key in owned)

    def spawn(
        self,
        operation_id: str,
        awaitable: Awaitable[Any],
        *,
        cancel: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> asyncio.Task:
        """Own an operation coroutine until it finishes or shutdown drains it."""
        with self._lock:
            existing = self._tasks.get(operation_id)
            if existing is not None and not existing.done():
                if hasattr(awaitable, "close"):
                    awaitable.close()  # type: ignore[attr-defined]
                raise RuntimeError(f"Operation {operation_id} is already running")
            task = asyncio.get_running_loop().create_task(awaitable)
            self._tasks[operation_id] = task
            if cancel is not None:
                self._cancellers[operation_id] = cancel

        def _done(finished: asyncio.Task) -> None:
            with self._lock:
                if self._tasks.get(operation_id) is finished:
                    self._tasks.pop(operation_id, None)
                    self._cancellers.pop(operation_id, None)
            if finished.cancelled():
                return
            try:
                exc = finished.exception()
            except asyncio.CancelledError:
                return
            if exc is None:
                return
            logger.exception(
                "supervised operation failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={"operation_id": operation_id, "task_id": operation_id},
            )
            try:
                from backend.operations.progress import get_progress_manager

                progress = get_progress_manager()
                tracked = progress.get_task(operation_id)
                if tracked and tracked.get("status") not in {
                    "completed",
                    "failed",
                    "cancelled",
                    "interrupted",
                }:
                    progress.fail_task(operation_id, str(exc))
            except Exception:
                logger.exception(
                    "failed to publish supervised operation failure",
                    extra={"operation_id": operation_id},
                )

        task.add_done_callback(_done)
        return task

    def track_cleanup(self, operation_id: str, task: asyncio.Task) -> None:
        """Own cancellation/process cleanup independently of the worker task."""
        with self._lock:
            self._cleanup_tasks.add(task)

        def _done(finished: asyncio.Task) -> None:
            with self._lock:
                self._cleanup_tasks.discard(finished)
            if finished.cancelled():
                return
            try:
                exc = finished.exception()
            except asyncio.CancelledError:
                return
            if exc is not None:
                logger.error(
                    "operation cleanup failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                    extra={"operation_id": operation_id, "task_id": operation_id},
                )

        task.add_done_callback(_done)

    async def drain(self, *, cancel: bool = True) -> None:
        """Wait for every owned operation; cancel them first during shutdown."""
        with self._lock:
            items = [
                (operation_id, task, self._cancellers.get(operation_id))
                for operation_id, task in self._tasks.items()
                if not task.done()
            ]
        if cancel:
            cancellations = [callback() for _, _, callback in items if callback]
            if cancellations:
                await asyncio.gather(*cancellations, return_exceptions=True)
            for _, task, callback in items:
                if callback is None and not task.done():
                    task.cancel()
        tasks = [task for _, task, _ in items]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        while True:
            with self._lock:
                cleanups = [
                    task for task in self._cleanup_tasks if not task.done()
                ]
            if not cleanups:
                break
            await asyncio.gather(*cleanups, return_exceptions=True)

    def recovery_status(self) -> dict:
        """Last restart reconciliation, or a settled report when none has run."""
        if self._last_reconcile is None:
            return {
                "outcome": "reconciled",
                "interrupted": 0,
                "unknown": 0,
                "kept_terminal": 0,
                "replayed": 0,
                "detail": "No restart reconciliation has run in this process.",
            }
        return dict(self._last_reconcile)

    def _runtime_observation(self, store, record: dict) -> str:
        """Evidence besides the operation row.

        ``running`` means this process still has the task. ``finished`` means
        a result file recorded for the operation is present while the row is
        still active. ``absent`` means there is no live task and no result
        file. ``unknown`` means that evidence could not be read. None of these
        start the operation again.
        """
        operation_id = str(record.get("operation_id") or "")
        with self._lock:
            task = self._tasks.get(operation_id)
        if task is not None and not task.done():
            return "running"
        detail = record.get("detail") if isinstance(record.get("detail"), dict) else {}
        result_path = str(detail.get("result_path") or "").strip()
        if not result_path:
            return "absent"
        root = os.path.realpath(store._config_dir)
        try:
            candidate = os.path.realpath(result_path)
        except OSError:
            return "unknown"
        if candidate != root and not candidate.startswith(root + os.sep):
            return "unknown"
        try:
            finished = os.path.isfile(candidate)
        except OSError:
            return "unknown"
        return "finished" if finished else "absent"

    def reconcile_startup(self) -> dict:
        """Match durable rows to dead in-process work. Does not replay it."""
        from backend.data_store import get_store
        from backend.engines.lifecycle import repair_stale_building_versions
        from backend.operations.retention import prune_operation_rows, retention_changed

        store = get_store()
        try:
            rows = store.list_operations()
        except Exception as exc:
            logger.warning("Could not read durable operations: %s", exc)
            result = {
                "outcome": "unknown",
                "interrupted": 0,
                "unknown": 0,
                "kept_terminal": 0,
                "replayed": 0,
                "detail": (
                    "Operation records could not be read. "
                    "Refresh before trying again."
                ),
            }
            with self._lock:
                self._resources.clear()
                self._last_reconcile = dict(result)
            return result

        from backend.operations.action_recovery import classify_action

        revised = []
        interrupted = 0
        unknown = 0
        kept_terminal = 0
        for row in rows:
            status = str(row.get("status") or "")
            if status not in ACTIVE_STATES:
                revised.append(row)
                if status in TERMINAL_STATES:
                    kept_terminal += 1
                continue
            # Cancellation requested is not verified termination. A restart
            # that cannot see the worker stop leaves the reservation owned.
            if status == "cancelling":
                revised.append(row)
                continue
            observation = self._runtime_observation(store, row)
            if observation == "running":
                revised.append(row)
                continue
            kind = str(row.get("kind") or "operation")
            detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
            # Absence from OPERATION_COMPLETION_EVIDENCE stays unproven.
            # effect_started False is the only general negative proof, and a
            # saved settings document is not completion. A map value of
            # "negative" is explicit proof the side effect did not happen.
            if OPERATION_COMPLETION_EVIDENCE.get(kind, "unproven") == "negative":
                detail = {**detail, "effect_started": False}
            decision = classify_action(kind, detail)
            updated = dict(row)
            updated["updated_at"] = time.time()
            updated["status"] = decision["status"]
            updated["message"] = decision["message"]
            updated["detail"] = {**detail, "recovery": decision}
            if decision["status"] == "interrupted":
                interrupted += 1
            elif decision["status"] == "unknown":
                unknown += 1
            else:
                kept_terminal += 1
            revised.append(updated)
        pruned = prune_operation_rows(revised)
        durable = pruned
        outcome = "reconciled"
        detail = "Restart reconciliation stored the interrupted and unknown rows."
        if interrupted or unknown or retention_changed(rows, pruned):
            try:
                store.replace_operations(pruned)
            except Exception as exc:
                logger.warning("Could not reconcile durable operations: %s", exc)
                durable = rows
                interrupted = 0
                unknown = 0
                outcome = "unknown"
                detail = (
                    "Reconciliation could not be stored. "
                    "Refresh before trying again."
                )
        with self._lock:
            self._resources.clear()
            self._records = {
                str(row.get("operation_id") or ""): dict(row)
                for row in durable
                if str(row.get("operation_id") or "")
            }
            if outcome == "reconciled":
                from backend.operations.action_recovery import resource_names

                for row in durable:
                    if str(row.get("status") or "") not in ACTIVE_STATES:
                        continue
                    operation_id = str(row.get("operation_id") or "")
                    detail = row.get("detail") if isinstance(row.get("detail"), dict) else {}
                    for key in resource_names(
                        str(row.get("resource_key") or ""),
                        detail.get("depends_on"),
                    ):
                        self._resources.setdefault(key, operation_id)
        if outcome == "reconciled":
            try:
                repair_stale_building_versions(store, get_task=lambda _task_id: None)
            except Exception as exc:
                logger.warning("Could not repair stale engine builds: %s", exc)
            try:
                from backend.operations.progress import get_progress_manager

                get_progress_manager().restore_operations(store.list_operations())
            except Exception as exc:
                logger.warning("Could not restore reconciled progress outcomes: %s", exc)
        result = {
            "outcome": outcome,
            "interrupted": interrupted,
            "unknown": unknown,
            "kept_terminal": kept_terminal,
            "replayed": 0,
            "detail": detail,
        }
        with self._lock:
            self._last_reconcile = dict(result)
        return result

    def forget_operation(self, operation_id: str) -> None:
        """Drop a record in memory and queue the disk delete behind any pending write."""
        operation_id = str(operation_id or "").strip()
        if not operation_id:
            return
        with self._lock:
            self._records.pop(operation_id, None)
        from backend.data_store import get_store
        from backend.store_io import run_store

        run_store(get_store().delete_operation, operation_id)

    def _evict_terminal_records(self) -> None:
        """Drop finished rows the durable retention policy would not keep.

        Active work stays. A row with a queued fsync stays too, so the memory
        copy remains newer than the file until that write finishes.
        """
        from backend.operations.retention import prune_operation_rows

        with self._lock:
            kept = {
                str(row.get("operation_id") or "")
                for row in prune_operation_rows(list(self._records.values()))
            }
            kept.update(self._pending_writes)
            for operation_id in list(self._records):
                if operation_id not in kept:
                    self._records.pop(operation_id, None)

    def _kept_submission(self, operation_id: str, record: dict) -> bool:
        with self._lock:
            cached = self._records.get(operation_id)
        return (
            isinstance(cached, dict)
            and cached.get("status") == record.get("status")
            and cached.get("updated_at") == record.get("updated_at")
        )

    def _release_claim(self, resource_key: Optional[str], operation_id: str) -> None:
        if not resource_key:
            return
        with self._lock:
            if self._resources.get(resource_key) == operation_id:
                self._resources.pop(resource_key, None)

    async def _persist_durable(self, record: dict, run_store_durable) -> None:
        """Same reservation as ``_persist``, but the caller waits for the write."""
        from backend.store_io import StoreDurabilityError

        stored = dict(record)
        operation_id = str(stored.get("operation_id") or "")
        previous = None
        had_record = False
        with self._lock:
            if operation_id:
                had_record = operation_id in self._records
                if had_record:
                    previous = dict(self._records[operation_id])
                self._records[operation_id] = stored
                self._pending_writes[operation_id] = (
                    self._pending_writes.get(operation_id, 0) + 1
                )

        def write() -> None:
            try:
                from backend.data_store import get_store

                get_store().upsert_operation(stored)
            except StoreDurabilityError as exc:
                if exc.committed is False and operation_id:
                    with self._lock:
                        if had_record and previous is not None:
                            self._records[operation_id] = previous
                        else:
                            self._records.pop(operation_id, None)
                raise
            finally:
                with self._lock:
                    if operation_id:
                        remaining = self._pending_writes.get(operation_id, 0) - 1
                        if remaining <= 0:
                            self._pending_writes.pop(operation_id, None)
                        else:
                            self._pending_writes[operation_id] = remaining
                self._evict_terminal_records()

        await run_store_durable(write)

    def _persist(self, record: dict, companion: Optional[dict] = None) -> None:
        from backend.data_store import get_store
        from backend.store_io import StoreDurabilityError, run_store

        stored = dict(record)
        operation_id = str(stored.get("operation_id") or "")
        companion_stored = dict(companion) if companion else None
        companion_id = str((companion_stored or {}).get("operation_id") or "")
        previous = None
        had_record = False
        companion_previous = None
        companion_had = False
        with self._lock:
            if operation_id:
                had_record = operation_id in self._records
                if had_record:
                    previous = dict(self._records[operation_id])
                self._records[operation_id] = stored
                self._pending_writes[operation_id] = (
                    self._pending_writes.get(operation_id, 0) + 1
                )
            if companion_id:
                companion_had = companion_id in self._records
                if companion_had:
                    companion_previous = dict(self._records[companion_id])
                self._records[companion_id] = companion_stored
        submitted = False

        def restore_companion() -> None:
            if not companion_id:
                return
            with self._lock:
                if companion_had and companion_previous is not None:
                    self._records[companion_id] = companion_previous
                else:
                    self._records.pop(companion_id, None)

        def write() -> None:
            nonlocal submitted
            submitted = True
            try:
                if companion_stored is None:
                    get_store().upsert_operation(stored)
                else:
                    store = get_store()

                    def replace_pair(document):
                        rows = [
                            dict(row)
                            for row in document.get("operations", [])
                            if str(row.get("operation_id") or "")
                            not in {operation_id, companion_id}
                        ]
                        rows.extend((dict(companion_stored), dict(stored)))
                        store._store_operation_rows(document, rows)

                    store._mutate("operations.yaml", replace_pair)
            except StoreDurabilityError as exc:
                # The request is still on the event loop, so this failure is
                # observed later by the response gate. Undo an uncommitted
                # write here, or the in-memory reservation outlives the disk.
                if exc.committed is False and operation_id:
                    with self._lock:
                        if had_record and previous is not None:
                            self._records[operation_id] = previous
                            resource_key = str(previous.get("resource_key") or "")
                            if resource_key and previous.get("status") not in TERMINAL_STATES:
                                self._resources.setdefault(resource_key, operation_id)
                        else:
                            self._records.pop(operation_id, None)
                            for key, owner in list(self._resources.items()):
                                if owner == operation_id:
                                    self._resources.pop(key, None)
                    restore_companion()
                raise
            finally:
                with self._lock:
                    if operation_id:
                        remaining = self._pending_writes.get(operation_id, 0) - 1
                        if remaining <= 0:
                            self._pending_writes.pop(operation_id, None)
                        else:
                            self._pending_writes[operation_id] = remaining
                self._evict_terminal_records()

        # Queued on the store thread. The request gate fsyncs it before the
        # response; background work stays on the loop and is drained at shutdown.
        # A rejected submission never enters ``write``, so the reservation made
        # above has to be undone here or the id stays pinned forever.
        try:
            run_store(write)
        except Exception:
            if not submitted and operation_id:
                with self._lock:
                    remaining = self._pending_writes.get(operation_id, 0) - 1
                    if remaining <= 0:
                        self._pending_writes.pop(operation_id, None)
                    else:
                        self._pending_writes[operation_id] = remaining
                    if had_record:
                        self._records[operation_id] = previous
                    else:
                        self._records.pop(operation_id, None)
                    restore_companion()
            raise

    def _get(self, operation_id: str) -> Optional[dict]:
        with self._lock:
            cached = self._records.get(operation_id)
        if cached is not None:
            return dict(cached)
        from backend.data_store import get_store

        try:
            rows = get_store().list_operations()
        except Exception:
            return None
        for row in rows:
            if str(row.get("operation_id") or "") == operation_id:
                return dict(row)
        return None


_supervisor: Optional[OperationSupervisor] = None


def get_supervisor() -> OperationSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = OperationSupervisor()
    return _supervisor
