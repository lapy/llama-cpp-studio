"""Reusable fork-build environment preparation (``build.prepare_fork``).

Some engine forks need to compile against the managed toolchain instead of the
upstream host bootstrap. That mechanism is declared by a recipe step and
implemented ONCE here: controlled HOME/cargo dirs, ccache layering, compiler
selection within a declared major range, memory-aware parallelism, required
tools, CUDA toolkit validation, and preservation of dependency checkouts that a
previous failed/partial build left in an unusable state.

All inputs come from the step's ``with`` mapping (validated below); the step
returns ``{"install_env": {...}}`` for the builder command.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Mapping, Optional


class ForkBuildError(RuntimeError):
    """The declared fork-build environment cannot be composed safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ForkBuildError(message)


def _string_list(value: Any, where: str) -> list:
    _require(isinstance(value, list) and all(isinstance(item, str) for item in value), where)
    return list(value)


def _compiler_major(executable: str) -> Optional[int]:
    try:
        output = subprocess.check_output(
            [executable, "-dumpfullversion", "-dumpversion"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    match = re.match(r"(\d+)", output)
    return int(match.group(1)) if match else None


def _select_compiler(path: str, minimum: int, maximum: int, error: str) -> str:
    candidates = [
        os.environ.get("CUDAHOSTCXX"),
        shutil.which("g++-12", path=path),
        shutil.which("g++-11", path=path),
        shutil.which("g++-10", path=path),
        shutil.which("g++-14", path=path),
        shutil.which("g++-13", path=path),
        shutil.which("g++", path=path),
        shutil.which("c++", path=path),
    ]
    checked = []
    for compiler in candidates:
        if not compiler or compiler in checked:
            continue
        checked.append(compiler)
        major = _compiler_major(compiler)
        if major is not None and minimum <= major <= maximum:
            return compiler
    raise ForkBuildError(error)


def _read_meminfo_available_kib() -> int:
    """Host ``MemAvailable`` in KiB (kept as a seam for tests and callers)."""
    from backend.concurrency import _read_meminfo_available_kib as read_meminfo

    return read_meminfo()


def _read_cgroup_available_kib() -> int:
    """Container memory headroom in KiB (kept as a seam for tests and callers)."""
    from backend.concurrency import _read_cgroup_available_kib as read_cgroup

    return read_cgroup()


def safe_jobs() -> int:
    """Compiler parallelism budget using host and cgroup memory limits.

    The math lives in :func:`backend.concurrency.build_job_budget` so the CMake
    builds (audio.cpp, llama-family) and the venv fork builds size their job
    count the same way; this name stays the fork-build entry point and keeps
    its historical reading order (host MemAvailable, capped by the cgroup).
    """
    from backend.concurrency import KIB_PER_GIB, build_job_budget

    available_kib = _read_meminfo_available_kib()
    cgroup_available_kib = _read_cgroup_available_kib()
    if cgroup_available_kib > 0 and (available_kib <= 0 or cgroup_available_kib < available_kib):
        available_kib = cgroup_available_kib
    if available_kib <= 0:
        return 1
    return build_job_budget(
        per_job_kib=4 * KIB_PER_GIB,
        available_kib=available_kib,
    )


def cuda_arch_count(raw: Any) -> int:
    """Device architectures a build targets, from TORCH_CUDA_ARCH_LIST."""
    text = str(raw or "").strip()
    if not text:
        return 1
    return max(1, len([part for part in re.split(r"[;,\s]+", text) if part.strip()]))


def nvcc_threads_for(budget: int, archs: int) -> int:
    """Per-compile nvcc parallelism that still fits the host budget.

    ``nvcc --threads`` only parallelizes the work INSIDE one compilation (its
    per-architecture passes), so it buys nothing for a single-architecture
    build where the file-level jobs already fill the host. When a build targets
    several architectures it is split across them, bounded so
    ``jobs * threads`` does not oversubscribe the budget, and capped because
    each nvcc worker is memory-hungry.
    """
    if archs <= 1 or budget < 8:
        return 1
    return max(1, min(archs, budget // 8, 4))


def host_nvcc_threads(raw_arch_list: Any = "", budget: Optional[int] = None) -> int:
    """Per-compile nvcc parallelism for a build environment on THIS host.

    The single policy every Studio build environment uses, so an engine that
    reads ``NVCC_THREADS`` never needs a hardcoded value in its definition: the
    file-level job budget (``safe_jobs``: affinity, cgroup quota and RAM) and
    the declared architecture count decide it.
    """
    budget_value = int(budget) if budget is not None else safe_jobs()
    return nvcc_threads_for(budget_value, cuda_arch_count(raw_arch_list))


async def prepare_fork_build(
    host,
    options: Mapping[str, Any],
    *,
    checkout: str,
    base_env: Mapping[str, Any],
) -> Dict[str, str]:
    """Compose the builder environment described by one recipe step."""
    _require(isinstance(options, Mapping), "build.prepare_fork options must be a mapping")
    unknown = set(options) - {
        "ccache",
        "variables",
        "create_dirs",
        "dependencies",
        "rustup_home_fallback",
        "required_tools",
        "toolchain",
        "concurrency",
        "caches",
        "require_cuda_nvcc",
        "cuda_error",
    }
    _require(not unknown, f"build.prepare_fork has unknown options {sorted(unknown)}")

    host._ensure_venv()
    env = {str(key): str(value) for key, value in dict(base_env).items()}
    root = str(getattr(host, "_base_dir", ""))
    venv = str(getattr(host, "_venv_path", ""))
    venv_python = str(host._venv_python())
    cuda_home = str(env.get("CUDA_HOME") or "")

    if options.get("require_cuda_nvcc"):
        if not cuda_home or not os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
            raise ForkBuildError(
                str(options.get("cuda_error") or "the managed CUDA toolkit is unavailable")
            )

    if options.get("ccache"):
        from backend.engines.build_workspace import ccache_base_dir, ccache_environment

        env.update(ccache_environment(ccache_base_dir(checkout), launchers=True, cuda=True))
    else:
        # A fork's OWN build system may turn ccache on without Studio asking
        # for it: the SGLang SM70 fork's sglang-kernel CMakeLists does exactly
        # that whenever a `ccache` binary is on PATH. Such a build inherits
        # whatever CCACHE_DIR the service environment carries, and a foreign
        # value (the standalone image exports one under another user's data
        # root) makes every cache miss die with "ccache: error: Permission
        # denied". Normalize only the cache LOCATION: the launchers stay off,
        # so this does not opt an engine into ccache, it just gives a build
        # that opts itself in a writable cache.
        from backend.engines.build_workspace import ccache_base_dir, ccache_environment

        env.update(ccache_environment(ccache_base_dir(checkout), launchers=False))

    def expand(template: Any) -> str:
        return (
            str(template)
            .replace("{root}", root)
            .replace("{venv}", venv)
            .replace("{venv_python}", venv_python)
            .replace("{cuda_home}", cuda_home)
            .replace("{dependencies}", os.path.join(root, "dependencies"))
        )

    for path in options.get("create_dirs") or ():
        expanded = expand(path)
        os.makedirs(expanded, exist_ok=True)
    for name, value in dict(options.get("variables") or {}).items():
        env[str(name)] = expand(value)

    for dependency in options.get("dependencies") or ():
        _require(isinstance(dependency, Mapping), "dependency entry must be a mapping")
        unknown = set(dependency) - {
            "path",
            "preserve_if_file",
            "require_files",
            "preserve_if_missing",
            "reason",
        }
        _require(not unknown, f"dependency entry has unknown keys {sorted(unknown)}")
        path = expand(dependency.get("path"))
        reason = str(dependency.get("reason") or "dependency checkout is unusable")
        preserve = False
        if dependency.get("preserve_if_file"):
            preserve = os.path.isfile(os.path.join(path, str(dependency["preserve_if_file"])))
        elif dependency.get("preserve_if_missing"):
            required = _string_list(
                dependency.get("require_files") or [], "dependency require_files"
            )
            preserve = os.path.isdir(path) and not all(
                os.path.isfile(os.path.join(path, name)) for name in required
            )
        if preserve:
            await _preserve_path(host, path, reason)

    rustup_fallback = options.get("rustup_home_fallback")
    if rustup_fallback:
        rustup_home = env.get("RUSTUP_HOME") or ""
        if rustup_home and os.path.isdir(rustup_home) and not os.access(rustup_home, os.W_OK):
            fallback = expand(rustup_fallback)
            os.makedirs(fallback, exist_ok=True)
            env["RUSTUP_HOME"] = fallback

    toolchain = dict(options.get("toolchain") or {})
    if toolchain:
        unknown = set(toolchain) - {"major_min", "major_max", "env_vars", "error"}
        _require(not unknown, f"toolchain has unknown keys {sorted(unknown)}")
        minimum = int(toolchain.get("major_min") or 0)
        maximum = int(toolchain.get("major_max") or 999)
        compiler = _select_compiler(
            str(env.get("PATH") or ""),
            minimum,
            maximum,
            str(toolchain.get("error") or "no usable C++ compiler in the declared range"),
        )
        for name in _string_list(toolchain.get("env_vars") or [], "toolchain env_vars"):
            env[name] = compiler

    concurrency = dict(options.get("concurrency") or {})
    budget = safe_jobs()
    declared_threads = concurrency.get("nvcc_threads")
    auto_threads = declared_threads is None or str(declared_threads).strip().lower() == "auto"
    threads = (
        nvcc_threads_for(budget, cuda_arch_count(env.get("TORCH_CUDA_ARCH_LIST")))
        if auto_threads
        else max(1, int(declared_threads))
    )
    if concurrency:
        unknown = set(concurrency) - {"memory_aware", "env_vars", "nvcc_threads"}
        _require(not unknown, f"concurrency has unknown keys {sorted(unknown)}")
        if concurrency.get("memory_aware"):
            # The per-file knob multiplies the file-level jobs, so the jobs are
            # rebalanced against it: jobs * nvcc_threads stays inside the budget
            # (a single-architecture build keeps the whole budget for jobs).
            limit = max(1, budget // threads)
            for name in _string_list(concurrency.get("env_vars") or [], "concurrency env_vars"):
                try:
                    configured = int(env.get(name, limit))
                except (TypeError, ValueError):
                    configured = limit
                env[name] = str(max(1, min(configured, limit)))
    # Host-derived by default: a definition only overrides NVCC_THREADS to pin
    # it. The fork scripts read it for `nvcc --threads` and the kernel build's
    # internal compile-thread count.
    env["NVCC_THREADS"] = str(threads)

    cached_vars = _string_list(options.get("caches") or [], "caches")
    if cached_vars:
        # Same managed cache root the launch profile resolves, so the builder's
        # precompile/smoke step fills what the server will read on start.
        from backend.engines.caches import cache_environment, ensure_cache_dirs

        ensure_cache_dirs(root, cached_vars)
        env.update(cache_environment(root, cached_vars))

    missing = [
        tool
        for tool in _string_list(options.get("required_tools") or [], "required_tools")
        if not shutil.which(tool, path=str(env.get("PATH") or ""))
    ]
    if missing:
        raise ForkBuildError("missing build tools: " + ", ".join(missing))
    return env


async def _preserve_path(host, path: str, reason: str) -> None:
    """Move an incompatible dependency checkout aside without losing edits."""
    if not os.path.isdir(path):
        return
    backup_dir = tempfile.mkdtemp(
        prefix=os.path.basename(path) + ".preserved-",
        dir=os.path.dirname(path),
    )
    preserved = os.path.join(backup_dir, "source")
    try:
        os.rename(path, preserved)
    except OSError:
        os.rmdir(backup_dir)
        raise
    await host._broadcast_log_line(
        f"Preserved dependency checkout at {preserved}: {reason}. "
        "The fork installer will create a fresh checkout."
    )
