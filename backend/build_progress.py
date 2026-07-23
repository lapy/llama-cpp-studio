"""Parse compiler/build log step counters into UI progress percentages.

Shared by all engine installers (llama.cpp / ik_llama.cpp via ``llama_manager``,
audio.cpp, and Python-venv engines that may stream ninja/cmake ``[x/y]`` lines).
"""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

# Ninja / CMake --build style: "[123/456] Building CXX object ..."
# Also accept spaced forms like "[ 12/ 90 ]".
_BUILD_STEP_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")

# Compile is the longest phase — give it most of the bar.
# Other stages stay short bookmarks around it.
CMAKE_STAGE_WINDOWS: Dict[str, Tuple[int, int]] = {
    "init": (0, 2),
    "clone": (2, 12),
    "checkout": (12, 16),
    "patch": (16, 18),
    "sync": (2, 12),
    "fetch": (2, 10),
    "configure": (18, 28),
    "build": (28, 92),
    "verify": (92, 98),
    "validate": (92, 98),
    "complete": (100, 100),
    "error": (0, 0),
}

# Python-venv installs (LMDeploy / 1Cat) spend most time in compile/wheel build when
# doing source installs; map [x/y] into the same heavy window.
PYTHON_INSTALL_BUILD_WINDOW: Tuple[int, int] = (20, 92)
PYTHON_INSTALL_CREEP_CEIL: float = 28.0


def cmake_stage_window(stage: str) -> Tuple[int, int]:
    """Return ``(floor, ceil)`` for a cmake engine stage."""
    return CMAKE_STAGE_WINDOWS.get(str(stage or ""), (0, 100))


def cmake_stage_start(stage: str) -> int:
    """Starting progress percent when entering a cmake stage."""
    return cmake_stage_window(stage)[0]


def apply_cmake_stage(ctx: dict, stage: str, *, message: str = "", base_message: str = "") -> dict:
    """Mutate a log-batcher ctx for a new cmake stage and return it."""
    floor, ceil = cmake_stage_window(stage)
    ctx["stage"] = stage
    ctx["progress"] = floor
    ctx["progress_floor"] = floor
    ctx["progress_ceil"] = ceil
    if base_message:
        ctx["base_message"] = base_message
    if message:
        ctx["message"] = message
    elif base_message:
        ctx["message"] = base_message
    return ctx


def parse_build_step_ratio(line: str) -> Optional[Tuple[int, int]]:
    """Return ``(current, total)`` from a ``[x/y]`` build log line, if present."""
    if not line:
        return None
    match = _BUILD_STEP_RE.search(line)
    if not match:
        return None
    current = int(match.group(1))
    total = int(match.group(2))
    if total <= 0 or current < 0:
        return None
    if current > total:
        current = total
    return current, total


def map_build_step_progress(
    current: int,
    total: int,
    *,
    floor: int,
    ceil: int,
) -> int:
    """Map a build step ratio into ``[floor, ceil]`` (inclusive)."""
    if total <= 0:
        return int(floor)
    low = max(0, min(100, int(floor)))
    high = max(low, min(100, int(ceil)))
    if high == low:
        return low
    ratio = max(0.0, min(1.0, float(current) / float(total)))
    return low + int(round((high - low) * ratio))


def apply_build_step_progress(
    line: str,
    *,
    current_progress: int,
    floor: int,
    ceil: int,
) -> Optional[Tuple[int, str]]:
    """If ``line`` has ``[x/y]``, return ``(progress, message_suffix)``.

    Progress never decreases below ``current_progress``.
    """
    ratio = parse_build_step_ratio(line)
    if not ratio:
        return None
    current, total = ratio
    mapped = map_build_step_progress(current, total, floor=floor, ceil=ceil)
    progress = max(int(current_progress), mapped)
    return progress, f"[{current}/{total}]"


def progress_from_install_log(
    line: str,
    *,
    current_progress: float,
    log_count: int,
) -> Tuple[float, str]:
    """Resolve progress for Python-venv install log lines.

    Prefers ``[x/y]`` compile counters (mapped into the heavy build window).
    Falls back to a slow pre-build creep so pip noise does not fake completion.
    """
    floor, ceil = PYTHON_INSTALL_BUILD_WINDOW
    step = apply_build_step_progress(
        line,
        current_progress=int(current_progress),
        floor=floor,
        ceil=ceil,
    )
    if step:
        progress, suffix = step
        return float(progress), suffix
    creep = min(
        PYTHON_INSTALL_CREEP_CEIL,
        max(float(current_progress or 0), 8.0 + float(log_count) * 0.4),
    )
    return creep, ""
