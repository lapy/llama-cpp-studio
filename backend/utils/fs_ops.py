"""Low-level filesystem helpers (e.g. robust tree removal on Windows)."""

from __future__ import annotations

import os
import shutil
import stat
import time
from typing import Callable

from backend.logging_config import get_logger

logger = get_logger(__name__)


def remove_readonly(func: Callable, path: str, exc) -> None:
    """shutil.rmtree onerror: clear read-only bit then retry (common on Windows)."""
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as e:
        logger.warning("Could not remove %s: %s", path, e)


class FilesystemRefusal(PermissionError):
    """A delete that finished as a refusal before any file was removed."""

    before_side_effect = True

    def __init__(self, message: str, *, code: str):
        super().__init__(message)
        self.detail = {"code": code, "message": message}


def running_model_ids(path: str) -> list:
    """Model ids whose live process still uses a generation overlapping ``path``."""
    from backend.proxy.manifests import LaunchManifestStore
    from backend.services.model_runtime_apply import verified_running_revision

    store = LaunchManifestStore()
    hits = store.overlapping_generations(path)
    live = []
    seen = set()
    for hit in hits:
        model_id = hit["model_id"]
        if model_id in seen:
            continue
        seen.add(model_id)
        running = verified_running_revision(store, model_id)
        if running and any(
            row["revision"] == running and row["model_id"] == model_id for row in hits
        ):
            live.append(model_id)
    return live


def launch_hold_detail(path: str | None, *, confirmed: bool) -> dict | None:
    """Describe a retained launch hold without retiring it."""
    if not path:
        return None
    from backend.proxy.manifests import deletion_block_reason

    if not deletion_block_reason(path):
        return None
    live = running_model_ids(path)
    if live:
        names = ", ".join(live)
        return {
            "code": "LAUNCH_REFERENCE_RUNNING",
            "message": (
                f"Stop {names} before deleting this path. "
                "A running launch still uses it."
            ),
        }
    if not confirmed:
        return {
            "code": "RETAINED_LAUNCH_REFERENCE",
            "message": (
                "This path is still referenced by a published or retained "
                "launch generation. Confirm to retire those generations and delete it."
            ),
        }
    return None


def release_launch_hold(path: str | None, *, retire_references: bool) -> None:
    """Refuse a held path, or retire stopped generations when the caller confirmed."""
    refusal = launch_hold_detail(path, confirmed=retire_references)
    if refusal:
        raise FilesystemRefusal(refusal["message"], code=refusal["code"])
    if not retire_references or not path:
        return
    from backend.proxy.manifests import LaunchManifestStore, deletion_block_reason

    if not deletion_block_reason(path):
        return
    LaunchManifestStore().retire_overlapping_generations(path)
    remaining = deletion_block_reason(path)
    if remaining:
        raise FilesystemRefusal(remaining, code="LAUNCH_REFERENCE_REMAINING")


def robust_rmtree(path: str, max_retries: int = 3, *, retire_references: bool = False) -> None:
    """Robustly remove a directory tree, handling Windows file locks."""
    if not os.path.exists(path):
        return

    release_launch_hold(path, retire_references=retire_references)

    for attempt in range(max_retries):
        try:
            shutil.rmtree(path, onerror=remove_readonly)
            logger.info("Successfully deleted directory: %s", path)
            return
        except PermissionError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    "Permission error deleting %s, attempt %s/%s: %s",
                    path,
                    attempt + 1,
                    max_retries,
                    e,
                )
                time.sleep(0.5)
            else:
                logger.error(
                    "Failed to delete %s after %s attempts: %s", path, max_retries, e
                )
                raise
        except OSError as e:
            if attempt < max_retries - 1:
                logger.warning(
                    "OS error deleting %s, attempt %s/%s: %s",
                    path,
                    attempt + 1,
                    max_retries,
                    e,
                )
                time.sleep(0.5)
            else:
                logger.error(
                    "Failed to delete %s after %s attempts: %s", path, max_retries, e
                )
                raise
