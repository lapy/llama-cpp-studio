"""Bounded workers for blocking control-plane work.

The API runs on one asyncio event loop. Anything that blocks that loop -- a
subprocess probe, a directory scan, a full-document YAML write with ``fsync`` --
stops every other request and every task update. This module keeps that work
off the loop and lets independent blocking probes run at the same time.

Two related facts shape the design:

* CPython threads do not execute pure-Python bytecode in parallel because of
  the GIL. A thread is therefore used for *responsiveness* and for overlapping
  work that releases the GIL (subprocesses, file/socket IO), not to claim a
  multi-core speedup for Python computation.
* The CPU-heavy parts of Studio are native subprocesses (CMake, llama.cpp and
  the other engine servers). They already parallelize themselves. The control
  plane must stop serializing their launches and progress plumbing so those
  subprocesses can use every core.

The pool is deliberately separate from the event loop's default executor and
from the bounded Hugging Face metadata pool: a burst of slow engine probes must
not occupy the workers that serve ordinary synchronous routes or download
metadata.
"""

from __future__ import annotations

import asyncio
import contextvars
import functools
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")

DEFAULT_BLOCKING_WORKERS = 32
MAX_BLOCKING_WORKERS = 128

_executor: Optional[ThreadPoolExecutor] = None


def _cgroup_cpu_quota() -> Optional[int]:
    """CPU limit reported by cgroup v2 or v1, if the process is throttled."""
    try:
        with open("/sys/fs/cgroup/cpu.max", "r", encoding="utf-8") as handle:
            quota, period = handle.read().split()[:2]
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        quota_path = "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
        period_path = "/sys/fs/cgroup/cpu/cpu.cfs_period_us"
        with open(quota_path, "r", encoding="utf-8") as handle:
            quota = int(handle.read().strip())
        with open(period_path, "r", encoding="utf-8") as handle:
            period = int(handle.read().strip())
        if quota > 0 and period > 0:
            return max(1, int(quota / period))
    except (OSError, ValueError):
        pass
    return None


def effective_cpu_count() -> int:
    """CPUs this process may actually run on.

    ``os.cpu_count()`` can report every host core inside a CPU-limited
    container, which makes a build oversubscribe the quota. Affinity and the
    cgroup quota are the usable bounds when they are available.
    """
    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        try:
            import psutil

            count = len(psutil.Process().cpu_affinity())
        except Exception:
            count = os.cpu_count() or 1
    count = max(1, int(count))
    quota = _cgroup_cpu_quota()
    if quota is not None:
        count = min(count, quota)
    return max(1, count)


def blocking_worker_count() -> int:
    """Size of the shared blocking-work pool.

    ``STUDIO_BLOCKING_WORKERS`` overrides the default for constrained hosts.
    The default is generous enough for several concurrent installs, scans and
    metadata probes, but bounded so a task storm cannot spawn unbounded
    threads.
    """
    configured = os.getenv("STUDIO_BLOCKING_WORKERS", "").strip()
    if configured:
        try:
            return max(1, min(MAX_BLOCKING_WORKERS, int(configured)))
        except ValueError:
            pass
    return max(
        8,
        min(
            MAX_BLOCKING_WORKERS,
            max(DEFAULT_BLOCKING_WORKERS, effective_cpu_count() * 4),
        ),
    )


def get_blocking_executor() -> ThreadPoolExecutor:
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=blocking_worker_count(),
            thread_name_prefix="studio-blocking",
        )
    return _executor


async def run_blocking(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run ``func`` on the shared blocking pool and await its result.

    Context variables are copied into the worker, matching
    :func:`asyncio.to_thread`. Using a dedicated pool keeps long probes from
    starving the event loop's default executor, which also serves synchronous
    route handlers.
    """
    loop = asyncio.get_running_loop()
    context = contextvars.copy_context()
    call = functools.partial(context.run, func, *args, **kwargs)
    return await loop.run_in_executor(get_blocking_executor(), call)


def shutdown_blocking(wait: bool = False) -> None:
    """Release worker threads. Used at application shutdown and in tests."""
    global _executor
    executor = _executor
    _executor = None
    if executor is not None:
        executor.shutdown(wait=wait, cancel_futures=True)
