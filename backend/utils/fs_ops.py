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


def robust_rmtree(path: str, max_retries: int = 3) -> None:
    """Robustly remove a directory tree, handling Windows file locks."""
    if not os.path.exists(path):
        return

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
