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


#: KiB in one GiB, for the memory budgets below.
KIB_PER_GIB = 1024 * 1024


def _read_meminfo_available_kib() -> int:
    """``MemAvailable`` from /proc/meminfo, or 0 when it cannot be read."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _read_cgroup_available_kib() -> int:
    """Memory headroom left by the container's cgroup limit, or 0.

    The page cache the cgroup already accounts for is not free memory: the
    kernel reclaims it on demand, so subtracting it would understate how much
    a build can safely claim.
    """
    maximum = 0
    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as handle:
            raw = handle.read().strip()
        maximum = 0 if raw in {"", "max"} else int(raw)
    except (OSError, ValueError):
        maximum = 0
    if maximum <= 0:
        try:
            with open(
                "/sys/fs/cgroup/memory/memory.limit_in_bytes", "r", encoding="utf-8"
            ) as handle:
                maximum = int(handle.read().strip())
        except (OSError, ValueError):
            return 0
        # cgroup v1 reports "unlimited" as a huge page-aligned value.
        if maximum >= 1 << 60:
            return 0
    try:
        with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as handle:
            current = int(handle.read().strip())
        file_cache = 0
        try:
            with open("/sys/fs/cgroup/memory.stat", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, _, value = line.partition(" ")
                    if key in {"active_file", "inactive_file"}:
                        file_cache += int(value.strip())
        except (OSError, ValueError):
            file_cache = 0
        usage = max(0, current - file_cache)
        if maximum > usage:
            return (maximum - usage) // 1024
    except (OSError, ValueError):
        return 0
    return 0


def available_memory_kib() -> int:
    """Memory a build can claim right now: host headroom capped by the cgroup."""
    available = _read_meminfo_available_kib()
    cgroup_available = _read_cgroup_available_kib()
    if cgroup_available > 0 and (available <= 0 or cgroup_available < available):
        return cgroup_available
    return available


def build_job_budget(
    *,
    per_job_kib: int = 4 * KIB_PER_GIB,
    reserve_kib: Optional[int] = None,
    available_kib: Optional[int] = None,
) -> int:
    """Parallel compiler jobs that fit this host without starving Studio.

    Engine builds are the most memory-hungry work Studio starts: a parallel
    CUDA compile can allocate several GiB per nvcc process, and an unbounded
    ``cmake --build --parallel`` turns the box into a swapping host where the
    API, the proxy and every SSE stream stop answering (they recover when the
    build ends). The budget is the smaller of the CPUs this process may use
    (affinity and cgroup quota) and what fits in memory after a reserve for
    the control plane and any loaded model.

    ``per_job_kib`` is the worst-case allocation of one compile job (CUDA is
    heavier than a host C++ compile); ``available_kib`` lets a caller pass a
    memory measurement it already made; ``STUDIO_BUILD_JOBS`` overrides the
    result for hosts that want a different tradeoff.
    """
    configured = os.getenv("STUDIO_BUILD_JOBS", "").strip()
    if configured:
        try:
            return max(1, int(configured))
        except ValueError:
            pass
    cpu_jobs = max(1, effective_cpu_count())
    available = available_memory_kib() if available_kib is None else int(available_kib)
    if available <= 0:
        return cpu_jobs
    if reserve_kib is None:
        # Leave the control plane (and a resident model) room to keep
        # answering: at least 4 GiB, and an eighth of the headroom on hosts
        # large enough for a big compile burst to fill it.
        reserve_kib = min(max(4 * KIB_PER_GIB, available // 8), available // 2)
    budget = max(1, (available - reserve_kib) // max(1, int(per_job_kib)))
    return max(1, min(cpu_jobs, budget))
