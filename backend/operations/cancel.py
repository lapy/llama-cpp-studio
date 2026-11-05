"""Helpers for cancelling long-running installer/manager operations."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Optional

from backend.ops_metrics import observe_cancellation_latency


def _descendant_pids(pid: int) -> list[int]:
    """Child processes, including ones that left the parent's process group."""
    try:
        import psutil
    except ImportError:
        return []
    try:
        process = psutil.Process(pid)
        return [child.pid for child in process.children(recursive=True)]
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_tree(
    pid: int,
    *,
    term_timeout: float = 3.0,
    kill_timeout: float = 2.0,
) -> None:
    """Stop an installer and every descendant, then kill whatever is still alive."""
    if not pid:
        return
    started = time.perf_counter()
    tracked = {pid}

    def _refresh() -> None:
        tracked.update(_descendant_pids(pid))
        for child in list(tracked):
            tracked.update(_descendant_pids(child))

    def _signal(sig: int) -> None:
        _refresh()
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            pgid = None
        if pgid == pid and os.name != "nt":
            try:
                # Only signal the group when this pid leads it. Otherwise killpg
                # would hit the parent process group, including Studio itself.
                os.killpg(pgid, sig)
            except (PermissionError, ProcessLookupError, OSError):
                pass
        for target in list(tracked):
            try:
                os.kill(target, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def _any_alive() -> bool:
        _refresh()
        return any(_pid_alive(target) for target in tracked)

    _signal(signal.SIGTERM)
    deadline = time.monotonic() + term_timeout
    while _any_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    if _any_alive():
        _signal(signal.SIGKILL)
        kill_deadline = time.monotonic() + kill_timeout
        while _any_alive() and time.monotonic() < kill_deadline:
            time.sleep(0.05)
    observe_cancellation_latency(time.perf_counter() - started)


async def cancel_running_operation(
    *,
    operation: Optional[str],
    current_task: Optional[asyncio.Task],
    active_process: Optional[asyncio.subprocess.Process],
    term_timeout: float = 3.0,
    kill_timeout: float = 2.0,
) -> bool:
    """Terminate and reap the process tree before cancelling its owner task."""
    if not operation:
        return False

    cancelled = False
    if active_process is not None and active_process.returncode is None:
        pid = getattr(active_process, "pid", None)
        if pid:
            await asyncio.to_thread(
                terminate_process_tree,
                pid,
                term_timeout=term_timeout,
                kill_timeout=kill_timeout,
            )
            try:
                await active_process.wait()
            except (ChildProcessError, ProcessLookupError):
                pass
        else:
            try:
                active_process.terminate()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(active_process.wait(), timeout=term_timeout)
            except asyncio.TimeoutError:
                try:
                    active_process.kill()
                except ProcessLookupError:
                    pass
                await active_process.wait()
        cancelled = True

    if current_task is not None and not current_task.done():
        current_task.cancel()
        try:
            await current_task
        except asyncio.CancelledError:
            pass
        except Exception:
            # The operation may fail naturally as its process exits. Cancellation
            # still owns final state publication and resource release.
            pass
        cancelled = True

    return cancelled
