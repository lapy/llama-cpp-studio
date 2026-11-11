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
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def _read_cgroup_available_kib() -> int:
    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as handle:
            maximum = handle.read().strip()
        if not maximum.isdigit():
            return 0
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
        maximum_bytes = int(maximum)
        if maximum_bytes > usage:
            return (maximum_bytes - usage) // 1024
    except (OSError, ValueError):
        return 0
    return 0


def safe_jobs() -> int:
    """Compiler parallelism budget using host and cgroup memory limits."""
    from backend.concurrency import effective_cpu_count

    cpu_jobs = max(1, effective_cpu_count())
    available_kib = _read_meminfo_available_kib()
    cgroup_available_kib = _read_cgroup_available_kib()
    if cgroup_available_kib > 0 and (available_kib <= 0 or cgroup_available_kib < available_kib):
        available_kib = cgroup_available_kib
    if available_kib <= 0:
        return 1
    gib_kib = 1024 * 1024
    reserve_kib = min(4 * gib_kib, available_kib // 4)
    memory_jobs = max(1, (available_kib - reserve_kib) // (4 * gib_kib))
    return max(1, min(cpu_jobs, memory_jobs))


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
    if concurrency:
        unknown = set(concurrency) - {"memory_aware", "env_vars", "nvcc_threads"}
        _require(not unknown, f"concurrency has unknown keys {sorted(unknown)}")
        if concurrency.get("memory_aware"):
            limit = safe_jobs()
            for name in _string_list(concurrency.get("env_vars") or [], "concurrency env_vars"):
                try:
                    configured = int(env.get(name, limit))
                except (TypeError, ValueError):
                    configured = limit
                env[name] = str(max(1, min(configured, limit)))
        if concurrency.get("nvcc_threads") is not None:
            env["NVCC_THREADS"] = str(concurrency["nvcc_threads"])

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
