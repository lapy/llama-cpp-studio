"""Guarded cleanup plans for the storage inventory.

Storage governance never accepts a filesystem path.  A caller may only name a
cleanable category and opaque unit ids that came from the inventory.  Preview
resolves those ids to the current scan, applies every refusal guard, and returns
a short-lived plan id.  Apply re-scans the tree, verifies that the selected
units are unchanged (the plan id is a digest of category, keep count and each
unit's identity), and only then deletes.

Safety invariants:

* Every deletion target is re-derived from a fresh inventory and validated with
  ``realpath`` containment under its category root; a symlink is never a
  deletion target.
* A category root, the data root, or any other protected root is never deleted.
* A running model, a retained launch generation, an active download/build/
  install, a configuration restore in progress and a scope cap all refuse the
  affected units instead of forcing a delete.
* Only the four cleanable categories have a deletion path.  Catalog model
  payloads, engine installs, CUDA toolkits and the accounting database stay
  report-only.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from backend.logging_config import get_logger
from backend.services import storage_inventory
from backend.utils.fs_ops import (
    FilesystemRefusal,
    release_launch_hold,
    robust_rmtree,
)

logger = get_logger(__name__)

MAX_CLEANUP_UNITS = 200
PLAN_TTL_SECONDS = 300.0
MAX_RETAINED_PLANS = 32
MAX_UNIT_ID_LENGTH = 128

CATEGORY_NOUNS: Dict[str, str] = {
    "models": "MODEL DIRECTORIES",
    "hf_cache": "HUGGING FACE CACHE DIRECTORIES",
    "jit_caches": "CACHE DIRECTORIES",
    "history_backups": "CONFIGURATION REVISIONS",
}

_ALLOWED_KINDS: Dict[str, frozenset] = {
    "models": frozenset({"model_payload"}),
    "hf_cache": frozenset({"hf_repository"}),
    "jit_caches": frozenset({"engine_cache", "build_workspace", "ccache", "cuda_installer_cache"}),
    "history_backups": frozenset({"history_revision"}),
}

CODE_CATEGORY_NOT_CLEANABLE = "CATEGORY_NOT_CLEANABLE"
CODE_CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"
CODE_PLAN_STALE = "PLAN_STALE"
CODE_SCOPE_TOO_LARGE = "CLEANUP_SCOPE_TOO_LARGE"
CODE_CLEANUP_IN_FLIGHT = "CLEANUP_IN_FLIGHT"
CODE_NOTHING_TO_CLEAN = "NOTHING_TO_CLEAN"
CODE_UNKNOWN_UNIT = "UNKNOWN_UNIT"
CODE_INVALID_TARGET = "INVALID_TARGET"
CODE_CLEANUP_FAILED = "CLEANUP_FAILED"
CODE_SCAN_FAILED = "STORAGE_SCAN_FAILED"

_PLANS: Dict[str, dict] = {}
_PLANS_LOCK = threading.Lock()
_APPLY_LOCK = threading.Lock()


class StorageCleanupError(RuntimeError):
    """A cleanup request that was refused before any file was removed."""

    before_side_effect = True

    def __init__(self, code: str, message: str, *, status_code: int = 409) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.status_code = int(status_code)

    @property
    def detail(self) -> dict:
        return {"code": self.code, "message": self.message}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _plan_digest(category: str, keep_history: Optional[int], entries: List[List[Any]]) -> str:
    canonical = json.dumps(
        {"category": category, "keep_history": keep_history, "units": entries},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _unit_brief(unit: dict) -> dict:
    return {
        "id": unit.get("id"),
        "label": unit.get("label"),
        "bytes": int(unit.get("bytes") or 0),
    }


def _refusal(unit: dict, code: str, message: str) -> dict:
    return {
        **_unit_brief(unit),
        "code": str(code),
        "message": str(message),
    }


def _confirmation_phrase(category: str, count: int) -> str:
    verb = "PRUNE" if category == "history_backups" else "DELETE"
    noun = CATEGORY_NOUNS.get(category, "ITEMS")
    if count <= 0:
        return f"NOTHING TO {verb} FROM {noun}"
    return f"{verb} {count} {noun}"


def _validate_category(category: str) -> str:
    value = str(category or "").strip()
    if value in storage_inventory.CLEANABLE_CATEGORIES:
        return value
    if value in storage_inventory.CATEGORY_ORDER:
        raise StorageCleanupError(
            CODE_CATEGORY_NOT_CLEANABLE,
            "That storage category is managed by another Studio surface and cannot be cleaned here.",
            status_code=400,
        )
    raise StorageCleanupError(
        CODE_CATEGORY_NOT_CLEANABLE,
        "Unknown storage category.",
        status_code=400,
    )


def _validate_keep_history(category: str, keep_history: Optional[int]) -> Optional[int]:
    if category != "history_backups":
        return None
    if keep_history is None:
        return storage_inventory.DEFAULT_HISTORY_KEEP
    try:
        value = int(keep_history)
    except (TypeError, ValueError) as exc:
        raise StorageCleanupError(
            CODE_SCOPE_TOO_LARGE,
            "keep_history must be an integer between 1 and 100.",
            status_code=400,
        ) from exc
    if value < 1 or value > storage_inventory.MAX_HISTORY_ENTRIES:
        raise StorageCleanupError(
            CODE_SCOPE_TOO_LARGE,
            "keep_history must be an integer between 1 and 100.",
            status_code=400,
        )
    return value


def _history_recent_ids(units: Dict[str, dict], keep: int) -> set:
    ordered = sorted(
        units.values(),
        key=lambda unit: (-int(unit.get("mtime_ns") or 0), str(unit.get("label") or "")),
    )
    return {unit["id"] for unit in ordered[: max(0, keep)]}


def _effective_blocked(
    category: str,
    unit: dict,
    *,
    keep_history: Optional[int],
    history_recent: set,
) -> Optional[dict]:
    blocked = unit.get("blocked")
    if category == "history_backups":
        # The scanner marks the default keep window. A plan may choose a
        # different keep count, so recentness is recomputed here.
        if blocked and blocked.get("code") == "RECENT_HISTORY":
            blocked = None
        if blocked:
            return blocked
        if unit.get("id") in history_recent:
            return {
                "code": "RECENT_HISTORY",
                "message": (
                    "Recent revisions are retained. Prune only revisions older than the keep count."
                ),
            }
        return None
    return blocked


def _validate_target(unit: dict) -> Optional[str]:
    """Return a refusal message when an indexed unit is unsafe to delete."""
    path = str(unit.get("path") or "")
    root = str(unit.get("root") or "")
    if not path or not root:
        return "This unit has no safe filesystem target."
    try:
        path_real = os.path.realpath(path)
        root_real = os.path.realpath(root)
    except (OSError, ValueError):
        return "This unit path could not be resolved safely."
    if path_real == root_real:
        return "Refusing to delete a storage root."
    try:
        if os.path.commonpath([root_real, path_real]) != root_real:
            return "This unit resolves outside its storage root."
    except ValueError:
        return "This unit resolves outside its storage root."
    if os.path.islink(path):
        return "Refusing to delete a symlink."
    if not os.path.lexists(path):
        return "This unit no longer exists."
    return None


def _entry_for(unit: dict) -> List[Any]:
    return [str(unit.get("id") or ""), int(unit.get("bytes") or 0), int(unit.get("mtime_ns") or 0)]


def _remember_plan(plan: dict) -> None:
    now = time.monotonic()
    with _PLANS_LOCK:
        for plan_id in [
            key
            for key, value in _PLANS.items()
            if float(value.get("expires_monotonic") or 0) <= now
        ]:
            _PLANS.pop(plan_id, None)
        _PLANS[plan["plan_id"]] = plan
        while len(_PLANS) > MAX_RETAINED_PLANS:
            oldest = min(_PLANS, key=lambda key: float(_PLANS[key].get("created_monotonic") or 0))
            _PLANS.pop(oldest, None)


def _load_plan(plan_id: str) -> Optional[dict]:
    with _PLANS_LOCK:
        plan = _PLANS.get(plan_id)
    if plan is None:
        return None
    if float(plan.get("expires_monotonic") or 0) <= time.monotonic():
        with _PLANS_LOCK:
            _PLANS.pop(plan_id, None)
        return None
    return plan


def _consume_plan(plan_id: str) -> None:
    with _PLANS_LOCK:
        _PLANS.pop(plan_id, None)


def clear_storage_cleanup_plans() -> None:
    """Drop all pending plans (tests, shutdown)."""
    with _PLANS_LOCK:
        _PLANS.clear()


def preview_cleanup(
    category: str,
    unit_ids: Optional[Iterable[str]] = None,
    keep_history: Optional[int] = None,
) -> dict:
    """Build a short-lived cleanup plan from opaque inventory unit ids."""
    category = _validate_category(category)
    keep = _validate_keep_history(category, keep_history)
    requested = [str(value).strip() for value in (unit_ids or []) if str(value).strip()]
    if len(requested) > MAX_CLEANUP_UNITS:
        raise StorageCleanupError(
            CODE_SCOPE_TOO_LARGE,
            f"At most {MAX_CLEANUP_UNITS} units can be cleaned at once.",
        )
    if len(requested) != len(set(requested)):
        deduped: List[str] = []
        for unit_id in requested:
            if unit_id not in deduped:
                deduped.append(unit_id)
        requested = deduped

    inventory, index = storage_inventory.storage_inventory_index(refresh=False)
    del inventory  # only the unit index is needed to build a plan
    category_index = (index.get("categories") or {}).get(category) or {}
    units: Dict[str, dict] = category_index.get("units") or {}
    if category == "history_backups":
        recent = _history_recent_ids(units, int(keep or storage_inventory.DEFAULT_HISTORY_KEEP))
    else:
        recent = set()

    refused: List[dict] = []
    accepted: List[dict] = []
    selected: List[dict] = []
    if requested:
        for unit_id in requested:
            if len(unit_id) > MAX_UNIT_ID_LENGTH or unit_id not in units:
                refused.append(
                    _refusal(
                        {"id": unit_id, "label": unit_id, "bytes": 0},
                        CODE_UNKNOWN_UNIT,
                        "That unit is not part of the current storage inventory.",
                    )
                )
                continue
            selected.append(units[unit_id])
    else:
        selected = list(units.values())

    for unit in selected:
        blocked = _effective_blocked(
            category,
            unit,
            keep_history=keep,
            history_recent=recent,
        )
        if blocked:
            refused.append(_refusal(unit, blocked["code"], blocked["message"]))
            continue
        problem = _validate_target(unit)
        if problem:
            refused.append(_refusal(unit, CODE_INVALID_TARGET, problem))
            continue
        accepted.append(unit)

    if len(accepted) > MAX_CLEANUP_UNITS:
        raise StorageCleanupError(
            CODE_SCOPE_TOO_LARGE,
            f"At most {MAX_CLEANUP_UNITS} units can be cleaned at once. Select fewer units.",
        )

    entries = [_entry_for(unit) for unit in accepted]
    plan_id = _plan_digest(category, keep, entries)
    now = time.monotonic()
    plan = {
        "plan_id": plan_id,
        "category": category,
        "keep_history": keep,
        "unit_ids": [unit["id"] for unit in accepted],
        "entries": entries,
        "created_monotonic": now,
        "expires_monotonic": now + PLAN_TTL_SECONDS,
    }
    _remember_plan(plan)
    reclaimable = sum(int(unit.get("bytes") or 0) for unit in accepted)
    notice = (
        "Nothing is selected; the plan cannot be applied."
        if not accepted
        else "Review the selected units. The plan is refused if the inventory changes before it is applied."
    )
    return {
        "plan_id": plan_id,
        "category": category,
        "category_label": storage_inventory.CATEGORY_LABELS[category],
        "keep_history": keep,
        "units": [_unit_brief(unit) for unit in accepted],
        "refused": refused,
        "reclaimable_bytes": int(reclaimable),
        "confirmation_phrase": _confirmation_phrase(category, len(accepted)),
        "scope_cap_units": MAX_CLEANUP_UNITS,
        "expires_at": datetime.fromtimestamp(
            time.time() + PLAN_TTL_SECONDS, tz=timezone.utc
        ).isoformat(),
        "notice": notice,
    }


def _measure_freed(path: str) -> int:
    if not os.path.lexists(path):
        return 0
    if os.path.isfile(path) and not os.path.islink(path):
        try:
            return int(os.stat(path, follow_symlinks=False).st_size)
        except OSError:
            return 0
    total = 0
    for current, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [name for name in dirnames if not os.path.islink(os.path.join(current, name))]
        for name in filenames:
            file_path = os.path.join(current, name)
            try:
                if os.path.islink(file_path):
                    continue
                total += int(os.stat(file_path, follow_symlinks=False).st_size)
            except OSError:
                continue
    return total


def _remove_unit(unit: dict, category: str) -> int:
    """Delete one verified unit and return the bytes removed."""
    kind = str(unit.get("kind") or "")
    if kind not in _ALLOWED_KINDS.get(category, frozenset()):
        raise StorageCleanupError(
            CODE_INVALID_TARGET,
            "That unit is not a cleanable target for its category.",
        )
    path = str(unit.get("path") or "")
    problem = _validate_target(unit)
    if problem:
        raise StorageCleanupError(CODE_INVALID_TARGET, problem)
    # Belt and braces: the launch-hold check is repeated at delete time so a
    # generation published after the scan still refuses the delete.
    release_launch_hold(path, retire_references=False)
    if category == "history_backups":
        freed = _measure_freed(path)
        os.remove(path)
        return freed
    before = _measure_freed(path)
    try:
        robust_rmtree(path, retire_references=False)
    except FilesystemRefusal:
        raise
    except Exception:
        after = _measure_freed(path) if os.path.lexists(path) else 0
        if after == before:
            raise
        raise _PartialRemoval(max(0, before - after), path) from None
    return before


class _PartialRemoval(OSError):
    """A directory removal failed after some bytes were already freed."""

    def __init__(self, freed: int, path: str) -> None:
        super().__init__(f"Could not remove every file under {os.path.basename(path)}")
        self.freed = int(freed)


def _protected_roots() -> List[str]:
    roots = [os.path.realpath(os.getcwd())]
    from backend.paths import studio_data_dir

    data_root = os.path.realpath(studio_data_dir())
    roots.append(data_root)
    roots.append(os.path.realpath(os.path.join(data_root, "models")))
    roots.append(os.path.realpath(os.path.join(data_root, "cuda")))
    roots.append(os.path.realpath(os.path.join(data_root, "build-workspaces")))
    roots.append(os.path.realpath(os.path.join(data_root, "temp")))
    try:
        from backend.engines.lifecycle import discover_engine_install_roots

        roots.extend(
            os.path.realpath(str(root))
            for root in (discover_engine_install_roots() or {}).values()
            if str(root or "").strip()
        )
    except Exception:
        pass
    try:
        from backend.models import hub

        roots.append(os.path.realpath(hub._hf_hub_cache_root()))
    except Exception:
        pass
    try:
        from backend.data_store import get_store

        roots.append(os.path.realpath(str(getattr(get_store(), "_config_dir", "") or "")))
    except Exception:
        pass
    return [root for root in roots if root]


def apply_cleanup(plan_id: str, *, confirm: bool) -> dict:
    """Apply a previously previewed plan after revalidation."""
    if confirm is not True:
        raise StorageCleanupError(
            CODE_CONFIRMATION_REQUIRED,
            "Confirmation is required before cleaning storage.",
            status_code=400,
        )
    plan_id = str(plan_id or "").strip()
    if not plan_id:
        raise StorageCleanupError(CODE_PLAN_STALE, "That cleanup plan is no longer available.")
    if not _APPLY_LOCK.acquire(blocking=False):
        raise StorageCleanupError(
            CODE_CLEANUP_IN_FLIGHT,
            "Another storage cleanup is already running.",
        )
    try:
        plan = _load_plan(plan_id)
        if plan is None:
            raise StorageCleanupError(
                CODE_PLAN_STALE,
                "That cleanup plan expired or was replaced. Preview the storage inventory again.",
            )
        category = str(plan.get("category") or "")
        keep = plan.get("keep_history")
        try:
            _, index = storage_inventory.storage_inventory_index(refresh=True)
        except Exception as exc:
            raise StorageCleanupError(
                CODE_SCAN_FAILED,
                "The storage inventory could not be rescanned, so nothing was deleted.",
                status_code=503,
            ) from exc
        category_index = (index.get("categories") or {}).get(category) or {}
        units: Dict[str, dict] = category_index.get("units") or {}
        if category not in _ALLOWED_KINDS or not units:
            raise StorageCleanupError(
                CODE_PLAN_STALE,
                "The storage inventory changed. Preview the cleanup again.",
            )
        current_entries: List[List[Any]] = []
        current_units: List[dict] = []
        for unit_id in plan.get("unit_ids") or []:
            unit = units.get(str(unit_id))
            if unit is None:
                raise StorageCleanupError(
                    CODE_PLAN_STALE,
                    "A selected unit changed since the preview. Preview the cleanup again.",
                )
            current_entries.append(_entry_for(unit))
            current_units.append(unit)
        if _plan_digest(category, keep, current_entries) != plan_id:
            raise StorageCleanupError(
                CODE_PLAN_STALE,
                "The selected storage changed since the preview. Preview the cleanup again.",
            )
        if len(current_units) > MAX_CLEANUP_UNITS:
            raise StorageCleanupError(
                CODE_SCOPE_TOO_LARGE,
                f"At most {MAX_CLEANUP_UNITS} units can be cleaned at once.",
            )
        if not current_units:
            raise StorageCleanupError(
                CODE_NOTHING_TO_CLEAN,
                "This plan has no selected units. Preview the storage inventory again.",
            )
        history_recent = (
            _history_recent_ids(units, int(keep or storage_inventory.DEFAULT_HISTORY_KEEP))
            if category == "history_backups"
            else set()
        )
        _consume_plan(plan_id)
        protected = set(_protected_roots())
        removed: List[dict] = []
        failed: List[dict] = []
        freed_total = 0
        for unit in current_units:
            blocked = _effective_blocked(
                category,
                unit,
                keep_history=keep,
                history_recent=history_recent,
            )
            if blocked:
                failed.append(_refusal(unit, blocked["code"], blocked["message"]))
                continue
            path = str(unit.get("path") or "")
            real = os.path.realpath(path) if path else ""
            if real and real in protected:
                failed.append(
                    _refusal(
                        unit,
                        CODE_INVALID_TARGET,
                        "Refusing to delete a protected storage root.",
                    )
                )
                continue
            try:
                freed = _remove_unit(unit, category)
            except _PartialRemoval as exc:
                freed_total += exc.freed
                failed.append(
                    _refusal(
                        unit,
                        CODE_CLEANUP_FAILED,
                        "Only part of this unit could be removed. The remaining files stay.",
                    )
                )
                logger.warning("Partial storage cleanup for %s: %s", unit.get("label"), exc)
                continue
            except FilesystemRefusal as exc:
                code = str((exc.detail or {}).get("code") or CODE_CLEANUP_FAILED)
                failed.append(_refusal(unit, code, str(exc)))
                continue
            except (StorageCleanupError, OSError) as exc:
                code = exc.code if isinstance(exc, StorageCleanupError) else CODE_CLEANUP_FAILED
                failed.append(_refusal(unit, code, str(exc)))
                logger.warning("Storage cleanup failed for %s: %s", unit.get("label"), exc)
                continue
            freed_total += int(freed)
            removed.append({**_unit_brief(unit), "bytes": int(freed)})
        storage_inventory.clear_storage_inventory_cache()
        if removed and failed:
            outcome = "partial"
        elif removed:
            outcome = "completed"
        else:
            outcome = "refused"
        notice = {
            "completed": "Everything selected was removed.",
            "partial": "Some units were removed; the rest stayed and are listed below.",
            "refused": "Nothing was removed. Resolve the listed refusals and preview again.",
        }[outcome]
        return {
            "plan_id": plan_id,
            "category": category,
            "category_label": storage_inventory.CATEGORY_LABELS[category],
            "outcome": outcome,
            "freed_bytes": int(freed_total),
            "removed": removed,
            "failed": failed,
            "notice": notice,
        }
    finally:
        _APPLY_LOCK.release()
