"""Parse compiler/build log counters into UI progress percentages.

Shared by all engine installers (llama.cpp / ik_llama.cpp via ``llama_manager``,
audio.cpp, and Python-venv engines). Understands both:

- Ninja / CMake ``[x/y]`` step counters (single global counter across targets)
- Unix Makefiles ``[ N%]`` percent counters (may restart per target/pass)
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Ninja / CMake --build style: "[123/456] Building CXX object ..."
# Also accept spaced forms like "[ 12/ 90 ]".
_BUILD_STEP_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")

# Unix Makefiles cmake --build style: "[ 46%] Built target ggml-cuda"
_BUILD_PERCENT_RE = re.compile(r"\[\s*(\d{1,3})\s*%\s*\]")

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
# doing source installs; map [x/y] / [N%] into the same heavy window.
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


def prefer_ninja_generator(cmake_args: List[str]) -> List[str]:
    """Append ``-G Ninja`` when ninja is available and no generator was chosen.

    Ninja emits a single ``[x/y]`` counter across all targets, which avoids the
    Makefile behavior of restarting ``[ N%]`` for each ``--target``.
    """
    args = list(cmake_args or [])
    for index, arg in enumerate(args):
        if arg == "-G" or arg.startswith("-G"):
            return args
        if arg == "--" and index > 0:
            break
    if not shutil.which("ninja"):
        return args
    return args + ["-G", "Ninja"]


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


def parse_build_percent(line: str) -> Optional[int]:
    """Return ``0..100`` from a Makefile-style ``[ N%]`` build log line, if present."""
    if not line:
        return None
    match = _BUILD_PERCENT_RE.search(line)
    if not match:
        return None
    percent = int(match.group(1))
    if percent < 0 or percent > 100:
        return None
    return percent


def parse_build_progress_ratio(line: str) -> Optional[Tuple[int, int]]:
    """Return ``(current, total)`` from ``[x/y]`` or synthesized from ``[N%]``.

    Prefer ninja step counters when both forms somehow appear on one line.
    """
    step = parse_build_step_ratio(line)
    if step:
        return step
    percent = parse_build_percent(line)
    if percent is None:
        return None
    return percent, 100


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


def _map_makefile_multipass_progress(
    *,
    pass_index: int,
    percent: int,
    floor: int,
    ceil: int,
) -> int:
    """Map Makefile ``[N%]`` that may restart per target into ``[floor, ceil]``.

    Uses an asymptotic multi-pass curve so the first target's ``100%`` does not
    consume the entire stage window (audio.cpp builds ``cli`` then ``server``).
    """
    low = max(0, min(100, int(floor)))
    high = max(low, min(100, int(ceil)))
    if high == low:
        return low
    span = high - low
    unit = max(0, int(pass_index)) + max(0.0, min(1.0, float(percent) / 100.0))
    # 1 - 0.5^unit → 0 at start, 0.5 after first 100%, 0.75 after second, …
    fraction = 1.0 - (0.5**unit)
    return low + int(round(span * fraction))


@dataclass
class BuildProgressTracker:
    """Stateful mapper for ninja ``[x/y]`` and multi-pass Makefile ``[N%]``."""

    floor: int
    ceil: int
    progress: int = 0
    _last_percent: Optional[int] = field(default=None, repr=False)
    _pass_index: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.floor = int(self.floor)
        self.ceil = max(self.floor, int(self.ceil))
        self.progress = max(int(self.progress), self.floor)

    def apply_line(self, line: str) -> Optional[Tuple[int, str]]:
        """If ``line`` has a build counter, update and return ``(progress, suffix)``."""
        step = parse_build_step_ratio(line)
        if step:
            current, total = step
            mapped = map_build_step_progress(
                current, total, floor=self.floor, ceil=self.ceil
            )
            self.progress = max(self.progress, mapped)
            # Global ninja counters supersede Makefile pass tracking.
            self._last_percent = None
            self._pass_index = 0
            return self.progress, f"[{current}/{total}]"

        percent = parse_build_percent(line)
        if percent is None:
            return None

        if self._last_percent is not None and percent + 5 < self._last_percent:
            self._pass_index += 1
        self._last_percent = percent

        mapped = _map_makefile_multipass_progress(
            pass_index=self._pass_index,
            percent=percent,
            floor=self.floor,
            ceil=self.ceil,
        )
        self.progress = max(self.progress, mapped)
        return self.progress, f"[{percent}%]"

    def complete(self) -> int:
        """Snap to the stage ceil when the phase finishes successfully."""
        self.progress = self.ceil
        return self.progress


def apply_build_step_progress(
    line: str,
    *,
    current_progress: int,
    floor: int,
    ceil: int,
) -> Optional[Tuple[int, str]]:
    """Stateless helper: map one line into ``[floor, ceil]``.

    Prefer :class:`BuildProgressTracker` for live builds so Makefile percent
    restarts across targets stay monotonic within the stage window.
    """
    tracker = BuildProgressTracker(
        floor=floor, ceil=ceil, progress=current_progress
    )
    return tracker.apply_line(line)


def progress_from_install_log(
    line: str,
    *,
    current_progress: float,
    log_count: int,
) -> Tuple[float, str]:
    """Resolve progress for Python-venv install log lines.

    Prefers ``[x/y]`` / ``[N%]`` compile counters (mapped into the heavy build window).
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
