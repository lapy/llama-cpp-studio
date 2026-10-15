"""Reclaim inference servers that escaped llama-swap's process supervision.

llama-swap supervises the models it launches with process-group kills and
``PR_SET_PDEATHSIG`` on its direct child. Python multi-process engines (SGLang,
vLLM) can break that containment: when the intermediate wrapper dies first, the
server tree is reparented to PID 1, keeps its GPU memory, and becomes invisible
to llama-swap (pdeathsig is not inherited and the escapees sit in their own
session, so group kills miss them).

This module finds engine processes reparented to PID 1 — only a dead parent can
cause reparenting, so any such process is leftover from a previous llama-swap
lifecycle (config-reload teardown, crash, or a kill that hit the wrapper) — and
kills their whole process group so the VRAM is returned.
"""

from __future__ import annotations

import os
import signal
import time
from typing import Any, Dict, List, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


# Command-line markers of the server entrypoints llama-swap is meant to own.
_LAUNCH_MARKERS = (
    "sglang.launch_server",
    "sglang.srt",
    "vllm.entrypoints",
    "lmdeploy serve",
)

# Renamed multiprocess roles (setproctitle) — leaderless when their launcher died.
_COMM_PREFIXES = ("sglang::", "vllm::")

# Spawned children and the resource tracker are plain-named python processes
# running inside a managed engine venv; the module path marker identifies them
# without matching unrelated venv usage such as ``pip install``.
_CHILD_MARKERS = ("multiprocessing",)

# Detached audio.cpp sidecars are owned by the audio manager on purpose.
_EXCLUDED_MARKERS = ("audiocpp_server", "audio-cpp")

# Seconds to wait for a graceful exit before escalating to SIGKILL.
DEFAULT_TERM_GRACE_SECONDS = 3.0
DEFAULT_KILL_GRACE_SECONDS = 2.0


def _engine_data_roots() -> List[str]:
    """Directories under which Studio-managed engine venvs live (e.g. /app/data/)."""
    try:
        from backend.paths import studio_data_dir

        root = os.path.realpath(studio_data_dir())
    except Exception:
        return []
    if not root or root == os.path.sep:
        return []
    return [root + os.path.sep]


def _is_launch_marker(name: str, cmdline: str) -> bool:
    """True for server entrypoints and renamed scheduler/detokenizer roles."""
    if any(marker in cmdline for marker in _LAUNCH_MARKERS):
        return True
    return any(name.startswith(prefix) for prefix in _COMM_PREFIXES)


def _matches_engine_signature(name: str, cmdline: str) -> bool:
    if any(marker in cmdline for marker in _EXCLUDED_MARKERS):
        return False
    # Renamed roles match on comm: zombies keep the name but expose no cmdline.
    if any(name.startswith(prefix) for prefix in _COMM_PREFIXES):
        return True
    if not cmdline:
        return False
    if any(marker in cmdline for marker in _LAUNCH_MARKERS):
        return True
    if not any(root in cmdline for root in _engine_data_roots()):
        return False
    return any(marker in cmdline for marker in _CHILD_MARKERS)


def _pgid(pid: int) -> Optional[int]:
    """Process group id via getpgid(2); None when the process is already gone.

    psutil exposes ``ppid`` but has no ``pgid`` attribute: passing ``"pgid"``
    in ``attrs=`` raises ``ValueError: invalid attr name 'pgid'`` for the whole
    iteration, so the group id must come from the syscall.
    """
    try:
        return os.getpgid(pid)
    except OSError:
        return None


def _process_rows():
    """Yield ``(pid, ppid, pgid, name, cmdline, create_time, status)`` for every process."""
    import psutil

    attrs = ["pid", "ppid", "name", "cmdline", "create_time", "status"]
    for proc in psutil.process_iter(attrs=attrs, ad_value=None):
        info = proc.info
        pid = info.get("pid")
        yield (
            pid,
            info.get("ppid"),
            _pgid(pid) if pid is not None else None,
            info.get("name") or "",
            " ".join(info.get("cmdline") or []),
            info.get("create_time"),
            (info.get("status") or "").lower(),
        )


def detect_engine_orphans() -> Dict[str, Any]:
    """Report engine processes reparented to PID 1 plus zombie leftovers.

    Reparenting only happens when the original parent died, so a matching
    process outside llama-swap's tree cannot be a live managed server: managed
    descendants always trace back to the llama-swap child it supervises.
    """
    report: Dict[str, Any] = {
        "supported": os.name != "nt",
        "orphans": [],
        "zombie_orphans": 0,
    }
    if not report["supported"]:
        report["detail"] = "Orphan detection is only available on POSIX systems."
        report["count"] = 0
        return report
    now = time.time()
    rows = list(_process_rows())
    orphans: Dict[int, Dict[str, Any]] = {}
    group_pids: Dict[int, List[Dict[str, Any]]] = {}
    for pid, ppid, pgid, name, cmdline, create_time, status in rows:
        if pid is None or pid <= 1 or ppid != 1:
            continue
        if not _matches_engine_signature(name, cmdline):
            continue
        if status == "zombie":
            # Zombies hold a PID slot only; tini/PID-1 init reaps them, Studio
            # cannot reap foreign children itself.
            report["zombie_orphans"] += 1
            continue
        row = {
            "pid": pid,
            "pgid": pgid or pid,
            "name": name,
            "cmdline": cmdline[:400],
            "age_seconds": max(0, int(now - create_time)) if create_time else None,
        }
        orphans[pid] = row
        group_pids.setdefault(row["pgid"], []).append(row)
    # Second pass: spawned workers keep the launcher's pgid but may carry plain
    # names with no engine marker. Same (still-orphaned) process group is proof
    # enough that they belong to the escapee tree.
    engine_groups = {
        pgid for pgid, rows_in_group in group_pids.items()
        if any(_is_launch_marker(row["name"], row["cmdline"]) for row in rows_in_group)
        or any(row["pid"] == pgid for row in rows_in_group)
    }
    if engine_groups:
        for pid, ppid, pgid, name, cmdline, create_time, status in rows:
            if pid is None or pid in orphans or ppid != 1 or status == "zombie":
                continue
            if (pgid or pid) not in engine_groups:
                continue
            if any(marker in cmdline for marker in _EXCLUDED_MARKERS):
                continue
            orphans[pid] = {
                "pid": pid,
                "pgid": pgid or pid,
                "name": name,
                "cmdline": cmdline[:400],
                "age_seconds": max(0, int(now - create_time)) if create_time else None,
            }
    report["orphans"] = list(orphans.values())
    report["count"] = len(report["orphans"])
    return report


def _os_kill(pid: int, sig: int) -> None:
    os.kill(pid, sig)


def _os_killpg(pgid: int, sig: int) -> None:
    os.killpg(pgid, sig)


def _alive(pid: int) -> bool:
    try:
        _os_kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _any_alive(pids: List[int]) -> bool:
    return any(_alive(pid) for pid in pids)


def _signal_all(pids: List[int], sig: int) -> None:
    for pid in pids:
        try:
            _os_kill(pid, sig)
        except (ProcessLookupError, PermissionError):
            pass


def _signal_group(pgid: int, pids: List[int], sig: int, *, group_leader_present: bool) -> None:
    """Signal the whole group when its leader is among the escapees, else pid-by-pid.

    A killpg on a group whose leader lives elsewhere (for example a still
    supervised llama-swap group) would hit processes we do not own — never do that.
    """
    if os.name != "nt" and pgid and group_leader_present:
        try:
            _os_killpg(pgid, sig)
            return
        except (ProcessLookupError, PermissionError):
            pass
    _signal_all(pids, sig)


def sweep_engine_orphans(
    *,
    dry_run: bool = False,
    term_grace: float = DEFAULT_TERM_GRACE_SECONDS,
    kill_grace: float = DEFAULT_KILL_GRACE_SECONDS,
) -> Dict[str, Any]:
    """Terminate (then kill) every engine process group reparented to PID 1.

    ``dry_run`` reports candidates without signalling anything.
    """
    detection = detect_engine_orphans()
    report: Dict[str, Any] = {
        "dry_run": dry_run,
        "supported": detection["supported"],
        "candidates": list(detection["orphans"]),
        "killed": [],
        "survivors": [],
        "zombie_orphans": detection["zombie_orphans"],
    }
    if not detection["supported"] or not detection["orphans"] or dry_run:
        return report

    # Escapees from one engine generation share a process group (the launcher
    # pgid), so one group signal covers launcher, schedulers, and workers.
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for row in detection["orphans"]:
        groups.setdefault(row["pgid"], []).append(row)

    killed_pids: set = set()
    survivor_pids: set = set()
    for pgid, rows in groups.items():
        pids = [row["pid"] for row in rows]
        leader_present = pgid in pids
        _signal_group(pgid, pids, signal.SIGTERM, group_leader_present=leader_present)
        deadline = time.monotonic() + max(0.0, term_grace)
        while _any_alive(pids) and time.monotonic() < deadline:
            time.sleep(0.05)
        if _any_alive(pids):
            _signal_group(pgid, pids, signal.SIGKILL, group_leader_present=leader_present)
            kill_deadline = time.monotonic() + max(0.0, kill_grace)
            while _any_alive(pids) and time.monotonic() < kill_deadline:
                time.sleep(0.05)
        for row in rows:
            if _alive(row["pid"]):
                survivor_pids.add(row["pid"])
            else:
                killed_pids.add(row["pid"])

    report["killed"] = [row for row in detection["orphans"] if row["pid"] in killed_pids]
    report["survivors"] = sorted(survivor_pids)
    if report["killed"]:
        logger.warning(
            "Reclaimed %s orphaned inference process(es) reparented to PID 1 (pgids=%s)",
            len(report["killed"]),
            sorted({row["pgid"] for row in report["killed"]}),
        )
    if report["survivors"]:
        logger.error(
            "Orphaned inference processes survived the sweep: %s", report["survivors"]
        )
    return report
