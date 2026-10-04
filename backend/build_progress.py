"""Parse compiler/build/clone/pip log counters into UI progress percentages.

Shared by all engine installers (llama.cpp / ik_llama.cpp via ``llama_manager``,
audio.cpp, and Python-venv engines). Understands:

- Git clone ``--progress`` phases (``Receiving objects: 46% (x/y)``, …)
- Ninja / CMake ``[x/y]`` step counters (single global counter across targets)
- Unix Makefiles ``[ N%]`` percent counters (may restart per target/pass)
- Pip release installs (bootstrap, resolve, wheel downloads, backtracking, install)
"""

from __future__ import annotations

import re
import shutil
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# Ninja / CMake --build style: "[123/456] Building CXX object ..."
# Also accept spaced forms like "[ 12/ 90 ]".
_BUILD_STEP_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")

# Unix Makefiles cmake --build style: "[ 46%] Built target ggml-cuda"
_BUILD_PERCENT_RE = re.compile(r"\[\s*(\d{1,3})\s*%\s*\]")

# Git clone --progress (see llama-install.log):
#   remote: Counting objects:  50% (99/197)
#   remote: Compressing objects:  50% (53/106)
#   Receiving objects:  46% (48213/104810), 153.47 MiB | 23.38 MiB/s
#   Resolving deltas:  50% (36700/73399)
#   Updating files: 100% (3258/3258)
_GIT_PROGRESS_RE = re.compile(
    r"(?P<label>"
    r"remote:\s+Counting objects|"
    r"remote:\s+Compressing objects|"
    r"Counting objects|"
    r"Compressing objects|"
    r"Receiving objects|"
    r"Resolving deltas|"
    r"Updating files|"
    r"Checking out files"
    r"):\s*(?P<percent>\d{1,3})%"
    r"(?:\s*\((?P<current>\d+)/(?P<total>\d+)\))?",
    re.IGNORECASE,
)

# Relative spans inside the current stage window (clone/sync/fetch).
# Receiving dominates wall time on a cold clone (~400 MiB in the sample log).
_GIT_PHASE_SPANS: Dict[str, Tuple[float, float]] = {
    "counting": (0.00, 0.05),
    "compressing": (0.05, 0.12),
    "receiving": (0.12, 0.78),
    "resolving": (0.78, 0.95),
    "updating": (0.95, 1.00),
}

_GIT_LABEL_TO_PHASE: Dict[str, str] = {
    "remote: counting objects": "counting",
    "counting objects": "counting",
    "remote: compressing objects": "compressing",
    "compressing objects": "compressing",
    "receiving objects": "receiving",
    "resolving deltas": "resolving",
    "updating files": "updating",
    "checking out files": "updating",
}

# Compile is the longest phase — give it most of the bar.
# Clone owns a meaningful slice so Receiving objects: N% is visible.
CMAKE_STAGE_WINDOWS: Dict[str, Tuple[int, int]] = {
    "init": (0, 2),
    "clone": (2, 18),
    "checkout": (18, 20),
    "patch": (20, 22),
    "sync": (2, 18),
    "fetch": (2, 14),
    "configure": (22, 28),
    "build": (28, 92),
    "verify": (92, 98),
    "validate": (92, 98),
    "complete": (100, 100),
    "error": (0, 0),
}

# Python-venv installs (LMDeploy / 1Cat) spend most time in compile/wheel build when
# doing source installs; map [x/y] / [N%] into the same heavy window.
PYTHON_INSTALL_BUILD_WINDOW: Tuple[int, int] = (20, 92)
PYTHON_INSTALL_CREEP_CEIL: float = 8.0

# Release-wheel installs (the 1Cat-vLLM cu128 log) download about 5.4 GB of
# wheels, then pip backtracks, then a second download wave unpacks. Bytes use
# pip's decimal units (kB/MB/GB = 1000-based) so bar fractions match the log.
PIP_DOWNLOAD_BUDGET_BYTES = 5_400_000_000
# Downloads and resolver backtracking finish before the long silent unpack.
# That pause — after "Successfully uninstalled setuptools-…" and before
# "Successfully installed" — is the largest slice of the bar.
PIP_DOWNLOAD_FLOOR = 8.0
PIP_DOWNLOAD_CEIL = 32.0
PIP_PRE_UNPACK_CEIL = 42.0
PIP_UNPACK_TARGET = 92.0
PIP_UNPACK_TIME_CONSTANT = 90.0
_PIP_UNIT_SCALE = {"b": 1, "kb": 1_000, "mb": 1_000_000, "gb": 1_000_000_000}

_PIP_DOWNLOAD_RE = re.compile(
    r"^Downloading\s+(?P<file>\S+?)"
    r"(?:\s+\((?P<size>\d+(?:\.\d+)?)\s*(?P<unit>kB|MB|GB|B)\))?\s*$",
    re.IGNORECASE,
)
_PIP_BAR_RE = re.compile(
    r"(?P<current>\d+(?:\.\d+)?)\s*/\s*(?P<total>\d+(?:\.\d+)?) "
    r"\s*(?P<unit>kB|MB|GB|B)\b",
    re.IGNORECASE,
)
_PIP_SPEED_RE = re.compile(
    r"(?P<rate>\d+(?:\.\d+)?)\s*(?P<unit>kB|MB|GB)/s",
    re.IGNORECASE,
)
_PIP_BACKTRACK_RE = re.compile(
    r"(?:still\s+)?looking at multiple versions of\s+(?P<name>\S+)",
    re.IGNORECASE,
)


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


# CMake CLI ``-W`` is a diagnostic switch (``-Wdev``, ``-WCMD_DEPRECATED``, …).
# nvcc's ``-Wno-deprecated-gpu-targets`` is a compiler flag; if it reaches the
# cmake argv, configure fails with: warning category "deprecated-gpu-targets"
# is not known.
_CMAKE_WARNING_CATEGORIES = frozenset(
    {
        "dev",
        "deprecated",
        "uninitialized",
        "CMD_AUTHOR",
        "CMD_DEPRECATED",
        "CMD_EXPERIMENTAL",
        "CMD_INSTALL_ABSOLUTE_DESTINATION",
        "CMD_POLICY",
        "CMD_UNINITIALIZED",
        "CMD_UNUSED_CLI",
    }
)
_CMAKE_WARNING_OPTION_RE = re.compile(
    r"^-W(?:no-)?(?:error=)?(?P<category>[A-Za-z][A-Za-z0-9_-]*)$"
)


def is_cmake_warning_option(arg: str) -> bool:
    match = _CMAKE_WARNING_OPTION_RE.fullmatch(str(arg or ""))
    return bool(match and match.group("category") in _CMAKE_WARNING_CATEGORIES)


def split_cmake_cli_warning_flags(
    args: Optional[Sequence[str]],
) -> Tuple[List[str], List[str]]:
    """Keep CMake ``-Wdev`` / ``-Wdeprecated``; relocate compiler ``-W`` flags."""
    kept: List[str] = []
    relocated: List[str] = []
    for arg in args or []:
        text = str(arg)
        if text.startswith("-W") and not is_cmake_warning_option(text):
            relocated.append(text)
            continue
        kept.append(text)
    return kept, relocated


def apply_relocated_cuda_warning_flags(
    env: Optional[Dict[str, str]], flags: Sequence[str]
) -> Dict[str, str]:
    """Append relocated nvcc ``-W`` flags to CUDA compiler env vars."""
    merged = dict(env or {})
    extra = " ".join(str(flag) for flag in flags if flag).strip()
    if not extra:
        return merged
    for key in ("CMAKE_CUDA_FLAGS", "CUDAFLAGS"):
        current = str(merged.get(key) or "").strip()
        merged[key] = f"{current} {extra}".strip() if current else extra
    return merged


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


def parse_git_progress(line: str) -> Optional[Tuple[str, int, Optional[int], Optional[int]]]:
    """Return ``(phase, percent, current, total)`` from a git ``--progress`` line."""
    if not line:
        return None
    match = _GIT_PROGRESS_RE.search(line)
    if not match:
        return None
    percent = int(match.group("percent"))
    if percent < 0 or percent > 100:
        return None
    label = re.sub(r"\s+", " ", match.group("label")).strip().lower()
    phase = _GIT_LABEL_TO_PHASE.get(label)
    if not phase:
        return None
    current_s = match.group("current")
    total_s = match.group("total")
    current = int(current_s) if current_s is not None else None
    total = int(total_s) if total_s is not None else None
    if total is not None and total <= 0:
        total = None
        current = None
    if current is not None and total is not None and current > total:
        current = total
    return phase, percent, current, total


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


def _map_fraction_progress(fraction: float, *, floor: int, ceil: int) -> int:
    low = max(0, min(100, int(floor)))
    high = max(low, min(100, int(ceil)))
    if high == low:
        return low
    ratio = max(0.0, min(1.0, float(fraction)))
    return low + int(round((high - low) * ratio))


def map_git_phase_progress(
    phase: str,
    percent: int,
    *,
    floor: int,
    ceil: int,
) -> int:
    """Map a git clone phase percent into ``[floor, ceil]`` using phase weights."""
    span = _GIT_PHASE_SPANS.get(phase, (0.0, 1.0))
    start, end = span
    unit = start + (end - start) * (max(0, min(100, int(percent))) / 100.0)
    return _map_fraction_progress(unit, floor=floor, ceil=ceil)


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
    """Stateful mapper for git clone %, ninja ``[x/y]``, and multi-pass Makefile ``[N%]``."""

    floor: int
    ceil: int
    progress: int = 0
    _last_percent: Optional[int] = field(default=None, repr=False)
    _pass_index: int = field(default=0, repr=False)
    _git_phase: Optional[str] = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.floor = int(self.floor)
        self.ceil = max(self.floor, int(self.ceil))
        self.progress = max(int(self.progress), self.floor)

    def set_window(self, floor: int, ceil: int, *, progress: Optional[int] = None) -> None:
        """Retarget the tracker when the cmake stage changes."""
        self.floor = int(floor)
        self.ceil = max(self.floor, int(ceil))
        if progress is None:
            self.progress = self.floor
        else:
            self.progress = max(self.floor, min(self.ceil, int(progress)))
        self._last_percent = None
        self._pass_index = 0
        self._git_phase = None

    def apply_line(self, line: str) -> Optional[Tuple[int, str]]:
        """If ``line`` has a progress counter, update and return ``(progress, suffix)``."""
        step = parse_build_step_ratio(line)
        if step:
            current, total = step
            mapped = map_build_step_progress(
                current, total, floor=self.floor, ceil=self.ceil
            )
            self.progress = max(self.progress, mapped)
            # Global ninja counters supersede Makefile / git pass tracking.
            self._last_percent = None
            self._pass_index = 0
            self._git_phase = None
            return self.progress, f"[{current}/{total}]"

        git = parse_git_progress(line)
        if git:
            phase, percent, current, total = git
            mapped = map_git_phase_progress(
                phase, percent, floor=self.floor, ceil=self.ceil
            )
            self.progress = max(self.progress, mapped)
            self._git_phase = phase
            if current is not None and total is not None:
                suffix = f"{percent}% {phase} ({current}/{total})"
            else:
                suffix = f"{percent}% {phase}"
            return self.progress, suffix

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


def _pip_unit_bytes(amount: str, unit: str) -> int:
    scale = _PIP_UNIT_SCALE.get(str(unit or "").lower(), 1)
    try:
        return max(0, int(round(float(amount) * scale)))
    except ValueError:
        return 0


def _pip_dist_name(filename: str) -> str:
    """PEP 427 wheel stem → distribution name (``torch-2.10…whl`` → ``torch``)."""
    base = str(filename or "").rsplit("/", 1)[-1]
    base = base.replace("%2B", "+").replace("%2b", "+")
    for suffix in (".whl.metadata", ".whl", ".tar.gz", ".zip", ".metadata"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    name_parts: List[str] = []
    for part in base.split("-"):
        if part[:1].isdigit():
            break
        name_parts.append(part)
    raw = "-".join(name_parts) if name_parts else base
    return raw.replace("_", "-") or "package"


def is_partial_pip_download(line: str) -> bool:
    """True when ``line`` is an in-progress pip bar (``400/916.9 MB``), not the final tick."""
    match = _PIP_BAR_RE.search(str(line or ""))
    if not match:
        return False
    try:
        current = float(match.group("current"))
        total = float(match.group("total"))
    except ValueError:
        return False
    return total > 0 and current + 0.05 < total


def is_compiler_progress_label(label: str) -> bool:
    """True for ninja/make suffixes and git ``46% receiving`` labels."""
    text = str(label or "").strip()
    if not text:
        return False
    if text.startswith("[") and text.endswith("]"):
        return True
    return bool(re.match(r"^\d{1,3}% \w+", text))


def _pip_error_message(line: str) -> Optional[str]:
    text = str(line or "").strip()
    if text.startswith("ERROR:") or text.startswith("error:"):
        return text[:180]
    if text.startswith("Traceback (most recent call last)"):
        return text[:180]
    return None


@dataclass
class PipInstallProgressTracker:
    """Map a pip release-wheel log onto one monotonic install percentage.

    Tuned to the 1Cat-vLLM cu128 install order: upgrade pip, resolve, download
    the torch/NVIDIA wheels (~4 GB), resolver backtracking, a second download
    wave, then ``Installing collected packages``.
    """

    progress: float = 0.0
    phase: str = "start"
    message: str = ""
    seen_main: bool = False
    completed_bytes: int = 0
    in_file_bytes: int = 0
    files_completed: int = 0
    resolve_lines: int = 0
    backtrack_lines: int = 0
    backtrack_anchor: Optional[float] = None
    files_at_backtrack: int = 0
    _backtrack_hold: str = field(default="", repr=False)
    _unpack_floor: float = field(default=0.0, repr=False)
    _unpack_started: Optional[float] = field(default=None, repr=False)
    current_name: str = ""
    _file_open: bool = field(default=False, repr=False)
    _counted_this_file: bool = field(default=False, repr=False)
    _pending_total: int = field(default=0, repr=False)
    _pending_bar: bool = field(default=False, repr=False)

    def observe(self, line: str, *, log_count: int = 0) -> Tuple[float, str]:
        """Advance from one pip log line. Returns ``(progress, stage message)``."""
        text = str(line or "").strip()
        if not text:
            return self.progress, self.message
        if self._apply(text):
            return self.progress, self.message
        if self.phase == "start":
            creep = min(
                PYTHON_INSTALL_CREEP_CEIL,
                max(self.progress, 4.0 + float(log_count) * 0.15),
            )
            self.progress = max(self.progress, creep)
        return self.progress, self.message

    def _bump(self, value: float, *, ceil: float = 97.0) -> None:
        self.progress = max(self.progress, min(float(ceil), float(value)))

    def _apply(self, line: str) -> bool:
        error = _pip_error_message(line)
        if error:
            self.message = error
            return True

        if line.startswith("$") and "pip" in line and "install" in line:
            return self._apply_command(line)

        backtrack = _PIP_BACKTRACK_RE.search(line)
        if backtrack and "looking at multiple versions" in line.lower():
            kind = "still" if "still looking" in line.lower() else "looking"
            self._advance_backtrack(kind)
            name = backtrack.group("name").rstrip(".,")
            self.current_name = name
            if kind == "still":
                self._backtrack_hold = "still"
                self.message = f"Still resolving {name} versions"
            elif self._backtrack_hold != "longer":
                self._backtrack_hold = ""
                self.message = f"Resolving {name} versions"
            return True
        if "taking longer than usual" in line.lower():
            self._advance_backtrack("longer")
            self._backtrack_hold = "longer"
            self.message = "Dependency resolution is taking longer than usual"
            return True

        if line.startswith("Installing collected packages"):
            return self._apply_install_start()
        if line.startswith("Attempting uninstall") or line.startswith("Successfully uninstalled"):
            return self._apply_uninstall()
        if line.startswith("Successfully installed"):
            return self._apply_installed()

        download = _PIP_DOWNLOAD_RE.match(line)
        if download:
            return self._apply_download_announcement(download)

        bar = _PIP_BAR_RE.search(line)
        if bar and "Downloading" not in line:
            return self._apply_download_bar(bar)

        if (
            line.startswith("Collecting ")
            or line.startswith("Requirement already satisfied")
            or line.startswith("Looking in indexes")
        ):
            return self._apply_resolve()
        return False

    def _apply_command(self, line: str) -> bool:
        if "--upgrade" in line and "setuptools" in line.lower():
            self.phase = "bootstrap"
            self.message = "Upgrading pip"
            self._bump(2.0, ceil=8.0)
            return True
        self.seen_main = True
        self.phase = "resolve"
        self.message = "Resolving dependencies"
        self._bump(8.0, ceil=20.0)
        return True

    def _apply_install_start(self) -> bool:
        self._flush_open_file()
        if not self.seen_main:
            self.phase = "bootstrap"
            self.message = "Upgrading pip"
            self._bump(6.0, ceil=8.0)
            return True
        self._begin_unpack()
        return True

    def _apply_uninstall(self) -> bool:
        if not self.seen_main:
            self.phase = "bootstrap"
            self.message = "Upgrading pip"
            self._bump(6.5, ceil=8.0)
            return True
        # The long silent stretch starts once pip has removed the previous
        # setuptools (or any other dist) and begins unpacking the new wheels.
        self._begin_unpack()
        return True

    def _apply_installed(self) -> bool:
        self._flush_open_file()
        if not self.seen_main:
            self.phase = "bootstrap"
            self.message = "Pip tools ready"
            self._bump(8.0, ceil=10.0)
            return True
        self.phase = "install"
        self._unpack_started = None
        self.message = "Packages installed"
        self._bump(96.0, ceil=97.0)
        return True

    def _begin_unpack(self) -> None:
        """Start (or restart) the silent wheel-unpack wait."""
        self._backtrack_hold = ""
        if self.phase != "unpack":
            self._bump(40.0, ceil=PIP_PRE_UNPACK_CEIL)
        self.phase = "unpack"
        self._unpack_floor = self.progress
        self._unpack_started = time.monotonic()
        self.message = "Installing packages"

    def note_unpack_wait(self) -> Tuple[float, str]:
        """Creep through the silent gap after pip stops logging.

        Pip prints ``Successfully uninstalled`` and then writes nothing while it
        unpacks the collected wheels. Leave headroom for ``Successfully installed``.
        """
        if self.phase != "unpack" or self._unpack_started is None:
            return self.progress, self.message
        elapsed = max(0.0, time.monotonic() - self._unpack_started)
        fraction = 1.0 - (2.718281828 ** (-elapsed / PIP_UNPACK_TIME_CONSTANT))
        span = max(0.0, PIP_UNPACK_TARGET - self._unpack_floor)
        self.message = "Installing packages"
        self._bump(self._unpack_floor + span * fraction, ceil=PIP_UNPACK_TARGET)
        return self.progress, self.message

    def _apply_resolve(self) -> bool:
        if self.phase == "backtrack" and not self._file_open:
            self._advance_backtrack("tick")
            return True
        if self._file_open:
            return True
        self.resolve_lines += 1
        fraction = 1.0 - (2.718281828 ** (-self.resolve_lines / 25.0))
        if not self.seen_main:
            if self.phase != "bootstrap":
                self.phase = "resolve"
                self.message = "Resolving dependencies"
            self._bump(2.0 + 4.0 * fraction, ceil=7.5)
            return True
        if self.phase not in ("install", "unpack"):
            self.phase = "resolve"
            self.message = "Resolving dependencies"
        self._bump(8.0 + 10.0 * fraction, ceil=20.0)
        return True

    def _apply_download_announcement(self, match: re.Match) -> bool:
        filename = match.group("file") or ""
        if ".metadata" in filename.lower():
            if self.phase == "backtrack":
                self._advance_backtrack("tick")
                name = _pip_dist_name(filename)
                if name and not self._backtrack_hold:
                    self.message = f"Resolving {name} versions"
                return True
            return self._apply_resolve()

        self._flush_open_file()
        self._backtrack_hold = ""
        self._file_open = True
        self._counted_this_file = False
        self._pending_bar = False
        size = match.group("size")
        unit = match.group("unit") or ""
        self._pending_total = _pip_unit_bytes(size, unit) if size else 0
        self.in_file_bytes = 0
        self.current_name = _pip_dist_name(filename)
        shown = f" ({size} {unit})" if size else ""
        self.message = f"Downloading {self.current_name}{shown}"
        if self.seen_main:
            self.phase = "download"
            self._advance_download()
        else:
            self.phase = "bootstrap"
            self._advance_bootstrap_download()
        return True

    def _apply_download_bar(self, match: re.Match) -> bool:
        current_s = match.group("current")
        total_s = match.group("total")
        unit = match.group("unit")
        try:
            current = float(current_s)
            total = float(total_s)
        except ValueError:
            return False
        if total <= 0:
            return False
        total_bytes = _pip_unit_bytes(total_s, unit)
        complete = current + 0.05 >= total
        self._pending_bar = True
        if not self._file_open:
            self._file_open = True
            self._counted_this_file = False
        if complete and not self._counted_this_file:
            self.completed_bytes += total_bytes
            self.files_completed += 1
            self._counted_this_file = True
            self.in_file_bytes = 0
            self._file_open = False
        elif not self._counted_this_file:
            self.in_file_bytes = _pip_unit_bytes(current_s, unit)

        name = self.current_name or "packages"
        speed = _PIP_SPEED_RE.search(match.string)
        self.message = f"Downloading {name} · {current_s}/{total_s} {unit}"
        if speed:
            self.message += f" · {speed.group('rate')} {speed.group('unit')}/s"
        if self.seen_main:
            self.phase = "download"
            self._advance_download()
        else:
            self.phase = "bootstrap"
            self._advance_bootstrap_download()
        return True

    def _flush_open_file(self) -> None:
        if self._file_open and not self._counted_this_file:
            pending = self.in_file_bytes if self._pending_bar else self._pending_total
            if pending > 0:
                self.completed_bytes += pending
                self.files_completed += 1
        self._file_open = False
        self._counted_this_file = False
        self._pending_bar = False
        self._pending_total = 0
        self.in_file_bytes = 0

    def _shown_bytes(self) -> int:
        inflight = 0 if self._counted_this_file else self.in_file_bytes
        return self.completed_bytes + inflight

    def _advance_bootstrap_download(self) -> None:
        ratio = min(1.0, self._shown_bytes() / 5_000_000)
        self._bump(2.0 + 5.0 * ratio, ceil=8.0)

    def _advance_download(self) -> None:
        shown = self._shown_bytes()
        ratio = min(1.0, shown / float(PIP_DOWNLOAD_BUDGET_BYTES))
        mapped = PIP_DOWNLOAD_FLOOR + (PIP_DOWNLOAD_CEIL - PIP_DOWNLOAD_FLOOR) * ratio
        if self.backtrack_anchor is not None:
            extra_files = max(0, self.files_completed - self.files_at_backtrack)
            if extra_files:
                lifted = self.backtrack_anchor + 8.0 + extra_files * 0.15
                mapped = max(mapped, lifted)
        self._bump(mapped, ceil=36.0)

    def _advance_backtrack(self, kind: str) -> None:
        if self.backtrack_anchor is None:
            self.backtrack_anchor = self.progress
            self.files_at_backtrack = self.files_completed
        self.backtrack_lines += 1
        fraction = 1.0 - (2.718281828 ** (-self.backtrack_lines / 6.0))
        lifted = self.backtrack_anchor + 12.0 * fraction
        if kind == "still":
            lifted = max(lifted, self.backtrack_anchor + 6.0)
        elif kind == "longer":
            lifted = max(lifted, self.backtrack_anchor + 10.0)
        self.phase = "backtrack"
        self._bump(lifted, ceil=36.0)


def progress_from_install_log(
    line: str,
    *,
    current_progress: float,
    log_count: int,
    tracker: Optional["PipInstallProgressTracker"] = None,
) -> Tuple[float, str]:
    """Resolve progress for Python-venv install log lines.

    Prefers ``[x/y]`` / ``[N%]`` / git counters (mapped into the heavy build window).
    Otherwise follows pip's release-install phases. Pass the same ``tracker`` for
    every line of an install so download bytes and resolver backtracking accumulate.
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
        progress = max(float(current_progress or 0), float(progress))
        if tracker is not None:
            tracker.progress = max(tracker.progress, progress)
        return progress, suffix

    if tracker is None:
        tracker = PipInstallProgressTracker()
    tracker.progress = max(tracker.progress, float(current_progress or 0))
    return tracker.observe(line, log_count=log_count)
