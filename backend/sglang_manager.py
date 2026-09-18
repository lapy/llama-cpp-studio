"""Install and manage versioned Python inference-server environments.

Every install uses a Studio-owned, versioned Python virtual environment.  For
the V100 fork, Studio also supplies its managed CUDA 12.8 toolkit and removes
the repository installer's host bootstrap (apt, CUDA, Miniconda).  The fork
continues to own its Python pins, patched dependencies, SM70 builds, and smoke
checks.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import sys
import time
from asyncio.subprocess import PIPE, STDOUT
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from backend.cancellable_operation_manager import CancellableOperationManager
from backend.data_store import get_store
from backend.llama_swap_manager import mark_swap_config_stale
from backend.logging_config import get_logger
from backend.progress_manager import get_progress_manager
from backend.utils.fs_ops import robust_rmtree


logger = get_logger(__name__)

SGLANG_ENGINE_IDS = frozenset({"sglang", "sglang_v100"})
PYTHON_SERVER_ENGINE_IDS = frozenset({*SGLANG_ENGINE_IDS, "vllm"})
SGLANG_REPOSITORIES = {
    "sglang": "https://github.com/sgl-project/sglang.git",
    "sglang_v100": "https://github.com/haohervchb/sglang-V100.git",
    "vllm": "https://github.com/vllm-project/vllm.git",
}
SGLANG_LABELS = {
    "sglang": "SGLang",
    "sglang_v100": "SGLang V100",
    "vllm": "vLLM",
}
SGLANG_ROOT_NAMES = {
    "sglang": "sglang",
    "sglang_v100": "sglang-v100",
    "vllm": "vllm",
}
V100_CUDA_VERSION = "12.8"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_variant(engine_id: str) -> str:
    value = str(engine_id or "").strip()
    if value not in PYTHON_SERVER_ENGINE_IDS:
        raise ValueError(f"Unknown Python inference engine: {engine_id}")
    return value


def _unique_version_name(store: Any, engine_id: str, base: str) -> str:
    existing = {
        str(row.get("version"))
        for row in store.get_engine_versions(engine_id)
        if row.get("version")
    }
    if base not in existing:
        return base
    stamp = int(time.time())
    for suffix in range(10000):
        candidate = f"{base}-{stamp}" if suffix == 0 else f"{base}-{stamp}-{suffix}"
        if candidate not in existing:
            return candidate
    return f"{base}-{stamp}-x"


_manager_instances: Dict[str, "SglangManager"] = {}


def get_sglang_manager(engine_id: str = "sglang") -> "SglangManager":
    if str(engine_id or "").strip() not in SGLANG_ENGINE_IDS:
        raise ValueError(f"Unknown SGLang engine: {engine_id}")
    engine_id = _safe_variant(engine_id)
    manager = _manager_instances.get(engine_id)
    if manager is None:
        manager = SglangManager(engine_id)
        _manager_instances[engine_id] = manager
    return manager


class SglangManager(CancellableOperationManager):
    """Shared versioned installer for SGLang variants and vanilla vLLM."""

    def __init__(
        self,
        engine_id: str = "sglang",
        *,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.engine_id = _safe_variant(engine_id)
        self.label = SGLANG_LABELS[self.engine_id]
        self.is_v100 = self.engine_id == "sglang_v100"
        self.is_vllm = self.engine_id == "vllm"
        self.MANAGER_NAME = self.engine_id
        self.LEGACY_STATUS_EVENT = f"{self.engine_id}_install_status"
        self.LEGACY_LOG_EVENT = f"{self.engine_id}_install_log"

        data_root = os.path.abspath("data")
        self._root_dir = os.path.abspath(
            base_dir or os.path.join(data_root, SGLANG_ROOT_NAMES[self.engine_id])
        )
        self._base_dir = self._root_dir
        self._venv_path = os.path.join(self._base_dir, "venv")
        self._log_path = os.path.abspath(
            log_path
            or os.path.join(data_root, "logs", f"{self.engine_id}_install.log")
        )
        self._ensure_directories()

    @property
    def operation_descriptions(self) -> Dict[str, str]:
        return {
            "install": f"Install {self.label}",
            "install_source": f"Install {self.label} from Source",
            "sync_source": f"Sync {self.label} Source",
            "remove": f"Remove {self.label}",
        }

    def _ensure_directories(self) -> None:
        os.makedirs(self._base_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)

    def _venv_bin(self, executable: str) -> str:
        if os.name == "nt":
            executable = (
                executable
                if executable.lower().endswith(".exe")
                else f"{executable}.exe"
            )
            return os.path.join(self._venv_path, "Scripts", executable)
        return os.path.join(self._venv_path, "bin", executable)

    def _venv_python(self) -> str:
        return self._venv_bin("python")

    def _ensure_venv(self) -> None:
        running = sys.version_info[:2]
        unsupported = (
            running != (3, 12)
            if self.is_v100
            else running < (3, 10) or (self.is_vllm and running > (3, 13))
        )
        if unsupported:
            requirement = (
                "3.12" if self.is_v100 else "3.10–3.13" if self.is_vllm else "3.10+"
            )
            raise RuntimeError(
                f"{self.label} requires Python {requirement}; "
                f"Studio is running Python {sys.version_info.major}.{sys.version_info.minor}"
            )
        if os.path.isfile(self._venv_python()):
            return
        os.makedirs(self._base_dir, exist_ok=True)
        subprocess.run([sys.executable, "-m", "venv", self._venv_path], check=True)

    @staticmethod
    def _prepend_path(env: Dict[str, str], directory: str) -> None:
        directory = os.path.abspath(directory)
        current = env.get("PATH", "")
        parts = [part for part in current.split(os.pathsep) if part]
        if directory not in parts:
            env["PATH"] = os.pathsep.join([directory, *parts])

    @staticmethod
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

    def _select_v100_compiler(self, env: Dict[str, str]) -> str:
        candidates = [
            env.get("CUDAHOSTCXX"),
            shutil.which("g++-12", path=env.get("PATH")),
            shutil.which("g++-11", path=env.get("PATH")),
            shutil.which("g++-10", path=env.get("PATH")),
            shutil.which("g++", path=env.get("PATH")),
            shutil.which("c++", path=env.get("PATH")),
        ]
        checked = []
        for compiler in candidates:
            if not compiler or compiler in checked:
                continue
            checked.append(compiler)
            major = self._compiler_major(compiler)
            if major is not None and 10 <= major <= 12:
                return compiler
        raise RuntimeError(
            "SGLang V100 requires GCC/G++ 10–12; GCC 13 is incompatible with "
            "the fork's CUDA 12.8 kernels. Rebuild the Studio image with g++-12."
        )

    @staticmethod
    def _v100_safe_jobs() -> int:
        """Choose compiler parallelism using host and cgroup memory limits."""
        cpu_jobs = max(1, os.cpu_count() or 1)
        available_kib = 0
        try:
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemAvailable:"):
                        available_kib = int(line.split()[1])
                        break
        except (OSError, ValueError, IndexError):
            pass

        try:
            with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as handle:
                maximum = handle.read().strip()
            with open("/sys/fs/cgroup/memory.current", "r", encoding="utf-8") as handle:
                current = int(handle.read().strip())
            if maximum.isdigit() and int(maximum) > current:
                cgroup_available_kib = (int(maximum) - current) // 1024
                if available_kib <= 0 or cgroup_available_kib < available_kib:
                    available_kib = cgroup_available_kib
        except (OSError, ValueError):
            pass

        if available_kib <= 0:
            return 1
        # Match the fork's conservative budget: reserve 16 GiB, then allow
        # roughly 4 GiB for every concurrent compiler process.
        memory_jobs = max(
            1,
            (available_kib - 16 * 1024 * 1024) // (4 * 1024 * 1024),
        )
        return max(1, min(cpu_jobs, memory_jobs))

    def _v100_build_environment(self) -> Dict[str, str]:
        """Bind the fork build to Studio's Python venv and CUDA 12.8 install."""
        self._ensure_venv()
        from backend.cuda_installer import get_cuda_installer

        cuda_env = get_cuda_installer().get_cuda_env(V100_CUDA_VERSION)
        cuda_home = str(cuda_env.get("CUDA_HOME") or "").strip()
        if not cuda_home or not os.path.isfile(os.path.join(cuda_home, "bin", "nvcc")):
            raise RuntimeError(
                "SGLang V100 requires Studio-managed CUDA 12.8. "
                "Install CUDA 12.8 from Engines > NVIDIA CUDA before building the fork."
            )

        env = dict(os.environ)
        env.update(cuda_env)
        env.update(
            {
                "VIRTUAL_ENV": self._venv_path,
                "SGLANG_STUDIO_PYTHON": self._venv_python(),
                "SGLANG_V100_PYTHON": self._venv_python(),
                "SGLANG_STUDIO_CUDA_HOME": cuda_home,
                "TORCH_CUDA_ARCH_LIST": "7.0",
            }
        )
        self._prepend_path(env, os.path.dirname(self._venv_python()))

        compiler = self._select_v100_compiler(env)
        env["CUDAHOSTCXX"] = compiler
        env["SGLANG_STUDIO_CUDAHOSTCXX"] = compiler
        safe_jobs = self._v100_safe_jobs()
        for key in ("MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL"):
            try:
                configured = int(env.get(key, safe_jobs))
            except (TypeError, ValueError):
                configured = safe_jobs
            env[key] = str(max(1, min(configured, safe_jobs)))
        # Multiple NVCC frontend threads multiply memory use per build job.
        env["NVCC_THREADS"] = "1"

        missing = [
            tool
            for tool in ("git", "cmake", "ninja", "curl")
            if not shutil.which(tool, path=env.get("PATH"))
        ]
        if missing:
            raise RuntimeError(
                "SGLang V100 is missing Studio build tools: " + ", ".join(missing)
            )
        return env

    def _prepare_versioned_paths(self, label: str) -> str:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        version_dir = f"{timestamp}-{label}"
        self._base_dir = os.path.join(self._root_dir, version_dir)
        self._venv_path = os.path.join(self._base_dir, "venv")
        self._ensure_directories()
        return version_dir

    def _bind_install_dir(self, reuse_dir: Optional[str], label: str) -> str:
        if reuse_dir:
            self._base_dir = os.path.abspath(reuse_dir)
            self._venv_path = os.path.join(self._base_dir, "venv")
            self._ensure_directories()
            return os.path.basename(self._base_dir)
        return self._prepare_versioned_paths(label)

    def _register_pending(self, version: str, extra: Dict[str, Any]) -> None:
        from backend.engine_version_lifecycle import mark_engine_version_building

        mark_engine_version_building(
            get_store(),
            self.engine_id,
            {
                "version": version,
                "venv_path": self._venv_path,
                "install_dir": self._base_dir,
                "installed_at": _utcnow(),
                **extra,
            },
            task_id=self._progress_task_id,
        )

    def _mark_failed(
        self,
        version: str,
        error: str,
        extra: Optional[Dict[str, Any]] = None,
        *,
        cancelled: bool = False,
    ) -> None:
        from backend.engine_version_lifecycle import mark_engine_version_failed

        mark_engine_version_failed(
            get_store(),
            self.engine_id,
            version,
            error=error,
            cancelled=cancelled,
            extra={
                "venv_path": self._venv_path,
                "install_dir": self._base_dir,
                **(extra or {}),
            },
        )

    def _mark_ready(self, pending_version: str, meta: Dict[str, Any]) -> str:
        from backend.engine_version_lifecycle import mark_engine_version_ready

        store = get_store()
        final_version = str(meta.get("version") or pending_version)
        if final_version != pending_version:
            store.delete_engine_version(self.engine_id, pending_version)
        mark_engine_version_ready(
            store,
            self.engine_id,
            {
                **meta,
                "version": final_version,
                "venv_path": self._venv_path,
                "install_dir": self._base_dir,
            },
        )
        return final_version

    def _detect_installed_version(self, venv_path: Optional[str] = None) -> Optional[str]:
        venv = os.path.abspath(venv_path or self._venv_path)
        python_bin = os.path.join(
            venv,
            "Scripts" if os.name == "nt" else "bin",
            "python.exe" if os.name == "nt" else "python",
        )
        if not os.path.isfile(python_bin):
            return None
        distribution = "vllm" if self.is_vllm else "sglang"
        script = (
            "from importlib import metadata\n"
            "try:\n"
            f" print(metadata.version({distribution!r}))\n"
            "except metadata.PackageNotFoundError:\n"
            " raise SystemExit(1)\n"
        )
        try:
            value = subprocess.check_output(
                [python_bin, "-c", script], text=True, stderr=subprocess.DEVNULL
            ).strip()
            return value or None
        except (OSError, subprocess.CalledProcessError):
            return None

    async def _broadcast_log_line(self, line: str) -> None:
        await self._append_task_log(line)
        await self._emit_legacy_log(line)
        if not self._progress_task_id:
            return
        task = get_progress_manager().get_task(self._progress_task_id) or {}
        count = int((task.get("metadata") or {}).get("log_count", 0)) + 1
        current = int(round(float(task.get("progress") or 0)))
        progress, stage = self._progress_stage_for_line(line, current)
        if progress > current:
            await self._update_progress_task(
                progress,
                stage,
                metadata_update={"log_count": count, "stage": stage},
            )
        else:
            # Log volume is unrelated to completed work. Keep it for diagnostics
            # without broadcasting noisy fractional progress updates per line.
            get_progress_manager().update_task(
                self._progress_task_id,
                metadata_update={"log_count": count},
                broadcast=False,
            )

    async def _update_progress_task(
        self,
        progress: float,
        message: str = "",
        metadata_update: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Publish rounded SGLang milestones without allowing regressions."""
        if self._progress_task_id:
            task = get_progress_manager().get_task(self._progress_task_id) or {}
            progress = max(float(task.get("progress") or 0), float(progress))
        await super()._update_progress_task(
            round(progress),
            message,
            metadata_update=metadata_update,
        )

    def _progress_stage_for_line(self, line: str, current: int) -> tuple[int, str]:
        """Return a monotonic integer progress milestone for installer output."""
        text = str(line or "").lower()
        if self.is_v100:
            milestones = (
                ("using studio python", 10, "Preparing Studio environment"),
                ("downloading ", 16, "Downloading V100 dependencies"),
                ("installing collected packages", 19, "Installing V100 dependencies"),
                ("building editable for sglang", 22, "Installing SGLang dependencies"),
                ("cloning flashinfer", 34, "Fetching FlashInfer SM70"),
                ("installing the proven flashinfer", 42, "Building FlashInfer SM70"),
                ("fetching the attributed", 50, "Fetching V100 kernel sources"),
                ("building the attributed turbomind", 58, "Building TurboMind SM70"),
                ("building lean sm70-only sglang-kernel", 70, "Building SGLang SM70 kernels"),
                ("restoring the cuda 12 nccl", 78, "Installing CUDA 12 NCCL bindings"),
                ("building v100 marlin", 85, "Building V100 Marlin kernels"),
                ("running sm70 smoke checks", 93, "Running V100 smoke checks"),
                ("complete. studio environment", 98, "Finalizing SGLang V100"),
            )
        else:
            milestones = (
                ("collecting ", 15, f"Resolving {self.label} dependencies"),
                ("downloading ", 25, f"Downloading {self.label} dependencies"),
                ("building editable", 55, f"Building {self.label}"),
                ("installing collected packages", 75, f"Installing {self.label}"),
                ("successfully installed", 92, f"Validating {self.label}"),
            )
        for marker, value, label in reversed(milestones):
            if marker in text:
                return max(current, value), label
        return current, "Working"

    async def _run_logged(
        self,
        argv: list[str],
        operation: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        mode = "a" if append else "w"
        with open(self._log_path, mode, encoding="utf-8") as log_file:
            log_file.write(f"[{_utcnow()}] {self.label} {operation}: {' '.join(argv)}\n")
        await self._broadcast_log_line(f"$ {' '.join(argv)}")
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=PIPE,
            stderr=STDOUT,
            cwd=cwd,
            env=env,
        )
        self._track_process(process)
        recent_lines: deque[str] = deque(maxlen=12)
        failure_context: deque[str] = deque(maxlen=120)

        async def stream() -> None:
            if process.stdout is None:
                return
            with open(self._log_path, "a", encoding="utf-8", buffering=1) as handle:
                while True:
                    chunk = await process.stdout.readline()
                    if not chunk:
                        break
                    line = chunk.decode("utf-8", errors="replace")
                    handle.write(line)
                    clean_line = line.rstrip("\n")
                    recent_lines.append(clean_line)
                    if self._is_failure_context_line(clean_line):
                        failure_context.extend(recent_lines)
                    await self._broadcast_log_line(clean_line)

        await asyncio.gather(process.wait(), stream())
        self._clear_active_process()
        returncode = process.returncode or 0
        if returncode != 0 and failure_context:
            # Parallel CUDA jobs can continue printing thousands of ptxas lines
            # after an earlier command fails. Replay the useful diagnostic at
            # the end so it remains visible in both the UI and log-tail API.
            summary = ["--- captured failure context ---"]
            seen = set()
            for context_line in failure_context:
                if context_line and context_line not in seen:
                    summary.append(context_line)
                    seen.add(context_line)
            with open(self._log_path, "a", encoding="utf-8") as handle:
                handle.write("\n" + "\n".join(summary) + "\n")
            for context_line in summary:
                await self._broadcast_log_line(context_line)
        return returncode

    @staticmethod
    def _is_failure_context_line(line: str) -> bool:
        text = str(line or "").lower()
        return any(
            marker in text
            for marker in (
                "failed:",
                "fatal error:",
                "ptxas fatal",
                " error:",
                "killed signal",
                "out of memory",
                "no space left on device",
                "subcommand failed",
            )
        )

    async def _run_pip(
        self,
        args: list[str],
        operation: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        append: bool = True,
    ) -> int:
        self._ensure_venv()
        return await self._run_logged(
            [self._venv_python(), "-m", "pip", *args],
            operation,
            cwd=cwd,
            env=env,
            append=append,
        )

    async def _clone(self, repo_url: str, branch: str, clone_dir: str) -> None:
        if os.path.exists(clone_dir):
            shutil.rmtree(clone_dir)
        await self._update_progress_task(3, "Cloning source")
        code = await self._run_logged(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                branch,
                repo_url,
                clone_dir,
            ],
            "clone",
            append=True,
        )
        if code != 0:
            raise RuntimeError(f"git clone failed with status {code}")
        await self._update_progress_task(8, "Source checkout ready")

    async def _git_head(self, clone_dir: str) -> Optional[str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                "git",
                "rev-parse",
                "HEAD",
                stdout=PIPE,
                stderr=STDOUT,
                cwd=clone_dir,
            )
            stdout, _ = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="replace").strip() or None
        except OSError:
            pass
        return None

    async def _sync_checkout(self, clone_dir: str, branch: str) -> None:
        if not os.path.isdir(os.path.join(clone_dir, ".git")):
            raise RuntimeError(f"Source checkout not found: {clone_dir}")
        for argv in (
            ["git", "fetch", "--depth", "1", "origin", branch],
            ["git", "checkout", "-B", branch, "FETCH_HEAD"],
            ["git", "reset", "--hard", "FETCH_HEAD"],
        ):
            code = await self._run_logged(argv, "sync_source", cwd=clone_dir)
            if code != 0:
                raise RuntimeError(f"{' '.join(argv[:2])} failed with status {code}")

    def _write_v100_prefix_installer(self, clone_dir: str) -> str:
        """Rewrite only the fork's host bootstrap to consume Studio resources."""
        source = os.path.join(clone_dir, "scripts", "install_v100.sh")
        if not os.path.isfile(source):
            raise RuntimeError("SGLang-V100 checkout has no scripts/install_v100.sh")
        with open(source, "r", encoding="utf-8") as handle:
            script = handle.read()
        bootstrap_start = "if [[ ${EUID} -eq 0 ]]; then"
        build_start = "# Use every CPU only when RAM can sustain"
        if script.count(bootstrap_start) != 1 or script.count(build_start) != 1:
            raise RuntimeError(
                "SGLang-V100 installer layout changed; refusing to rewrite its host bootstrap"
            )
        start = script.index(bootstrap_start)
        end = script.index(build_start)
        if end <= start:
            raise RuntimeError("SGLang-V100 installer bootstrap markers are out of order")
        studio_bootstrap = """# Studio owns host packages, Python, and CUDA.
[[ -n \"${SGLANG_STUDIO_PYTHON:-}\" ]] || die \"SGLANG_STUDIO_PYTHON is required.\"
[[ -x \"$SGLANG_STUDIO_PYTHON\" ]] || die \"Studio Python is not executable: $SGLANG_STUDIO_PYTHON\"
[[ -n \"${SGLANG_STUDIO_CUDA_HOME:-}\" ]] || die \"SGLANG_STUDIO_CUDA_HOME is required.\"
[[ -x \"$SGLANG_STUDIO_CUDA_HOME/bin/nvcc\" ]] || die \"Studio CUDA nvcc is missing.\"
export VIRTUAL_ENV=\"$(dirname \"$(dirname \"$SGLANG_STUDIO_PYTHON\")\")\"
export CUDA_HOME=\"$SGLANG_STUDIO_CUDA_HOME\"
export CUDA_PATH=\"$CUDA_HOME\"
export PATH=\"$(dirname \"$SGLANG_STUDIO_PYTHON\"):$CUDA_HOME/bin:$PATH\"
export CUDAHOSTCXX=\"${SGLANG_STUDIO_CUDAHOSTCXX:-${CUDAHOSTCXX:-}}\"
export TORCH_CUDA_ARCH_LIST=7.0
log \"Using Studio Python: $SGLANG_STUDIO_PYTHON\"
log \"Using Studio CUDA: $CUDA_HOME\"
# A failed pre-adapter attempt may have populated this versioned venv with
# CUDA 13 Python-toolkit packages. These are not the NVIDIA driver and are not
# used with Studio's managed CUDA 12.8 toolkit.
python -m pip uninstall -y \
  cuda-python cuda-bindings cuda-core cuda-pathfinder cuda-toolkit \
  nvidia-cuda-crt nvidia-cuda-nvcc nvidia-cuda-runtime \
  nvidia-cuda-tileiras nvidia-nvjitlink nvidia-nvvm || true

"""
        patched = script[:start] + studio_bootstrap + script[end:]
        marlin_stage = 'log "Building V100 Marlin GPTQ/AWQ kernels"'
        if marlin_stage in patched:
            patched = patched.replace(
                marlin_stage,
                f"{marlin_stage}\n# Do not leak sglang-kernel-only CMake flags into Marlin.\nunset CMAKE_ARGS",
                1,
            )
        # The TurboMind subset includes the repository's LICENSE file as well
        # as directories. Git sparse-checkout defaults to cone mode, where
        # every argument is validated as a directory, and aborts on LICENSE.
        # Non-cone mode supports the fork's mixed file/directory pattern list
        # and also works when retrying the partially initialized checkout.
        sparse_checkout = 'git -C "$destination" sparse-checkout set "$@"'
        if sparse_checkout in patched:
            patched = patched.replace(
                sparse_checkout,
                'git -C "$destination" sparse-checkout set --no-cone "$@"',
                1,
            )
        patched = patched.replace(
            'log "Complete. Run: conda activate sglang-v100"',
            'log "Complete. Studio environment: $VIRTUAL_ENV"',
        )
        # Keep the patched entry point beside the upstream script.  The fork
        # derives REPO_ROOT from BASH_SOURCE, so moving it outside scripts/
        # would make the otherwise unchanged installer target the wrong tree.
        destination = os.path.join(clone_dir, "scripts", "install_v100_studio.sh")
        with open(destination, "w", encoding="utf-8") as handle:
            handle.write(patched)
        os.chmod(destination, 0o755)
        return destination

    @staticmethod
    def _patch_v100_python_metadata(clone_dir: str) -> None:
        """Keep the fork's Python CUDA bindings aligned with Studio CUDA 12.8."""
        pyproject = os.path.join(clone_dir, "python", "pyproject.toml")
        if not os.path.isfile(pyproject):
            raise RuntimeError("SGLang-V100 checkout has no python/pyproject.toml")
        with open(pyproject, "r", encoding="utf-8") as handle:
            text = handle.read()
        marker = "# Studio SM70 dependency overrides applied"
        if marker in text:
            return
        replacements = {
            '"cuda-python>=13.0",': '"cuda-python==12.8.0",',
            # These stock wheels are replaced by pinned SM70 source builds.
            '"flashinfer_python==0.6.12",': "",
            '"flashinfer_cubin==0.6.11.post1",': "",
            '"sglang-kernel==0.4.3",': "",
        }
        missing = [
            original
            for original in replacements
            if text.count(original) != 1
        ]
        if missing:
            raise RuntimeError(
                "SGLang-V100 Python dependencies changed; refusing an unsafe "
                "automatic SM70 metadata rewrite: " + ", ".join(missing)
            )
        for original, replacement in replacements.items():
            text = text.replace(original, replacement, 1)
        with open(pyproject, "w", encoding="utf-8") as handle:
            handle.write(f"{marker}\n{text}")

    async def _install_source_checkout(
        self, clone_dir: str, build_env: Optional[Dict[str, str]] = None
    ) -> None:
        if not self.is_v100:
            code = await self._run_pip(
                ["install", "--upgrade", "pip"], "install_source", append=True
            )
            if code != 0:
                raise RuntimeError(f"pip upgrade failed with status {code}")
            python_package = (
                clone_dir if self.is_vllm else os.path.join(clone_dir, "python")
            )
            if not os.path.isdir(python_package):
                raise RuntimeError(f"{self.label} source package directory is unavailable")
            code = await self._run_pip(
                ["install", "-v", "-e", python_package],
                "install_source",
                cwd=clone_dir,
                env=build_env,
            )
            if code != 0:
                raise RuntimeError(f"{self.label} source install failed with status {code}")
            return

        env = dict(build_env or self._v100_build_environment())
        installer = self._write_v100_prefix_installer(clone_dir)
        self._patch_v100_python_metadata(clone_dir)
        controlled_home = os.path.join(self._base_dir, "home")
        dependencies_dir = os.path.join(self._base_dir, "dependencies")
        marlin_repo = os.path.join(dependencies_dir, "marlin-v100")
        legacy_bf16_header = os.path.join(marlin_repo, "csrc", "sm70_bf16_compat.h")
        os.makedirs(controlled_home, exist_ok=True)
        # The fork's compatibility patch is intended for CUDA toolkits whose
        # SM70 headers omit BF16 helpers. Studio's managed CUDA 12.8 already
        # defines them, so applying that patch causes ten redefinition errors.
        # A retry may retain the previously patched managed checkout; discard
        # just that dependency so the fork can clone it again without the shim.
        if os.path.isfile(legacy_bf16_header):
            robust_rmtree(marlin_repo)
        env.update(
            {
                "HOME": controlled_home,
                "SGLANG_V100_PYTHON": env["SGLANG_STUDIO_PYTHON"],
                "SGLANG_V100_DEPS_DIR": dependencies_dir,
                "MARLIN_V100_REPO": marlin_repo,
                "MARLIN_V100_SKIP_BF16_COMPAT": "1",
                "CARGO_HOME": os.path.join(self._base_dir, "cargo-home"),
                "CARGO_TARGET_DIR": os.path.join(self._base_dir, "cargo-target"),
            }
        )
        os.makedirs(env["CARGO_HOME"], exist_ok=True)
        os.makedirs(env["CARGO_TARGET_DIR"], exist_ok=True)
        code = await self._run_logged(
            ["bash", installer],
            "install_source",
            cwd=clone_dir,
            env=env,
        )
        if code != 0:
            raise RuntimeError(f"SGLang-V100 installer failed with status {code}")

    async def _begin(self, operation: str) -> None:
        await self._begin_operation(
            operation,
            self.operation_descriptions.get(operation, f"Install {self.label}"),
            {"engine": self.engine_id},
        )

    async def _finalize_install(
        self,
        pending_version: str,
        meta: Dict[str, Any],
        success_message: str,
    ) -> None:
        await self._update_progress_task(98, f"Validating {self.label} installation")
        detected = self._detect_installed_version()
        if not detected:
            raise RuntimeError(f"Installed {self.label} package could not be imported")
        store = get_store()
        final_base = str(meta.get("version") or detected or pending_version)
        final_version = (
            pending_version
            if meta.get("reuse_existing")
            else _unique_version_name(store, self.engine_id, final_base)
        )
        meta = {
            **meta,
            "version": final_version,
            "package_version": detected,
            "installed_at": _utcnow(),
        }
        meta.pop("reuse_existing", None)
        final_version = self._mark_ready(pending_version, meta)
        store.set_active_engine_version(self.engine_id, final_version)
        try:
            from backend.engine_param_scanner import scan_engine_version

            scan_engine_version(store, self.engine_id, meta)
        except Exception as exc:
            logger.warning("%s parameter scan failed: %s", self.label, exc)
        mark_swap_config_stale()
        await self._finish_operation(True, success_message)

    async def install_release(
        self,
        version: Optional[str] = None,
        force_reinstall: bool = False,
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.is_v100:
            return await self.install_from_source(
                SGLANG_REPOSITORIES[self.engine_id],
                "main",
                reuse_dir=reuse_dir,
                existing_version=existing_version,
            )
        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            await self._begin("install")
            dirname = self._bind_install_dir(reuse_dir, "pip")
            pending = existing_version or dirname
            self._register_pending(
                pending, {"install_type": "pip", "package_version": version}
            )

            async def runner() -> None:
                extra = {"install_type": "pip", "package_version": version}
                try:
                    args = ["install", "--upgrade"]
                    if not self.is_vllm:
                        args.append("--pre")
                    if force_reinstall:
                        args.append("--force-reinstall")
                    package = "vllm" if self.is_vllm else "sglang"
                    args.append(f"{package}=={version}" if version else package)
                    code = await self._run_pip(args, "install", append=False)
                    if code != 0:
                        raise RuntimeError(f"pip install failed with status {code}")
                    detected = self._detect_installed_version()
                    await self._finalize_install(
                        pending,
                        {
                            **extra,
                            "version": existing_version or detected or pending,
                            "reuse_existing": bool(existing_version),
                        },
                        f"{self.label} installed",
                    )
                except asyncio.CancelledError:
                    self._mark_failed(pending, "Operation cancelled by user", extra, cancelled=True)
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._mark_failed(pending, str(exc), extra)
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(f"{self.label} installation started")

    async def install_from_source(
        self,
        repo_url: Optional[str] = None,
        branch: str = "main",
        *,
        reuse_dir: Optional[str] = None,
        existing_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        repo_url = str(repo_url or SGLANG_REPOSITORIES[self.engine_id]).strip()
        branch = str(branch or "main").strip()
        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            await self._begin("install_source")
            dirname = self._bind_install_dir(reuse_dir, "source")
            pending = existing_version or dirname
            from backend.repo_identity import source_build_type_labels_for_engine

            labels = source_build_type_labels_for_engine(self.engine_id, repo_url)
            extra = {
                "type": labels["type"],
                "install_type": labels["install_type"],
                "is_fork": labels["is_fork"],
                "source_repo": repo_url,
                "source_branch": branch,
            }
            self._register_pending(pending, extra)
            clone_dir = os.path.join(self._base_dir, "source")

            async def runner() -> None:
                runtime_meta: Dict[str, Any] = {}
                try:
                    build_env = None
                    if self.is_v100:
                        build_env = self._v100_build_environment()
                        runtime_meta = {
                            "cuda_version": V100_CUDA_VERSION,
                            "cuda_path": build_env["CUDA_HOME"],
                            "python_version": (
                                f"{sys.version_info.major}.{sys.version_info.minor}"
                            ),
                        }
                    elif self.is_vllm:
                        from backend.cuda_installer import get_cuda_installer

                        cuda_installer = get_cuda_installer()
                        cuda = cuda_installer.status()
                        build_env = dict(os.environ)
                        build_env.update(cuda_installer.get_cuda_env())
                        build_env["VIRTUAL_ENV"] = self._venv_path
                        self._prepend_path(build_env, os.path.dirname(self._venv_python()))
                        runtime_meta = {
                            "cuda_version": cuda.get("version"),
                            "cuda_path": cuda.get("cuda_path"),
                            "python_version": (
                                f"{sys.version_info.major}.{sys.version_info.minor}"
                            ),
                        }
                    await self._clone(repo_url, branch, clone_dir)
                    await self._install_source_checkout(clone_dir, build_env)
                    commit = await self._git_head(clone_dir)
                    detected = self._detect_installed_version()
                    base = f"{detected or branch}-{(commit or '')[:8]}".rstrip("-")
                    await self._finalize_install(
                        pending,
                        {
                            **extra,
                            "version": existing_version or base or pending,
                            "source_commit": commit,
                            "reuse_existing": bool(existing_version),
                            **runtime_meta,
                        },
                        f"{self.label} installed from {branch}",
                    )
                except asyncio.CancelledError:
                    self._mark_failed(
                        pending,
                        "Operation cancelled by user",
                        {**extra, **runtime_meta},
                        cancelled=True,
                    )
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._mark_failed(pending, str(exc), {**extra, **runtime_meta})
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(
                f"{self.label} source installation started",
                repo=repo_url,
                branch=branch,
            )

    async def retry_existing_install(self, row: Dict[str, Any]) -> Dict[str, Any]:
        install_dir = str(row.get("install_dir") or "").strip()
        version = str(row.get("version") or "").strip()
        if not install_dir or not version:
            raise ValueError("Version is missing its install directory or name")
        kind = str(row.get("install_type") or row.get("type") or "").lower()
        if kind in {"source", "fork", "patched", "local"} or self.is_v100:
            return await self.install_from_source(
                row.get("source_repo") or SGLANG_REPOSITORIES[self.engine_id],
                row.get("source_branch") or "main",
                reuse_dir=install_dir,
                existing_version=version,
            )
        return await self.install_release(
            row.get("package_version"),
            force_reinstall=True,
            reuse_dir=install_dir,
            existing_version=version,
        )

    async def sync_source_version(self, row: Dict[str, Any]) -> Dict[str, Any]:
        install_dir = str(row.get("install_dir") or "").strip()
        version = str(row.get("version") or "").strip()
        branch = str(row.get("source_branch") or "").strip()
        if not install_dir or not version or not branch:
            raise ValueError("Source version is missing install directory, name, or branch")
        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            self._base_dir = os.path.abspath(install_dir)
            self._venv_path = os.path.join(self._base_dir, "venv")
            clone_dir = os.path.join(self._base_dir, "source")
            await self._begin("sync_source")
            self._register_pending(version, dict(row))

            async def runner() -> None:
                runtime_meta: Dict[str, Any] = {}
                try:
                    build_env = None
                    if self.is_v100:
                        build_env = self._v100_build_environment()
                        runtime_meta = {
                            "cuda_version": V100_CUDA_VERSION,
                            "cuda_path": build_env["CUDA_HOME"],
                            "python_version": (
                                f"{sys.version_info.major}.{sys.version_info.minor}"
                            ),
                        }
                    elif self.is_vllm:
                        from backend.cuda_installer import get_cuda_installer

                        cuda_installer = get_cuda_installer()
                        cuda = cuda_installer.status()
                        build_env = dict(os.environ)
                        build_env.update(cuda_installer.get_cuda_env())
                        build_env["VIRTUAL_ENV"] = self._venv_path
                        self._prepend_path(build_env, os.path.dirname(self._venv_python()))
                        runtime_meta = {
                            "cuda_version": cuda.get("version"),
                            "cuda_path": cuda.get("cuda_path"),
                            "python_version": (
                                f"{sys.version_info.major}.{sys.version_info.minor}"
                            ),
                        }
                    await self._sync_checkout(clone_dir, branch)
                    await self._install_source_checkout(clone_dir, build_env)
                    commit = await self._git_head(clone_dir)
                    await self._finalize_install(
                        version,
                        {
                            **row,
                            "version": version,
                            "source_commit": commit,
                            "reuse_existing": True,
                            **runtime_meta,
                        },
                        f"{self.label} source synchronized",
                    )
                except asyncio.CancelledError:
                    self._mark_failed(
                        version,
                        "Operation cancelled by user",
                        {**row, **runtime_meta},
                        cancelled=True,
                    )
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._mark_failed(version, str(exc), {**row, **runtime_meta})
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(
                f"{self.label} source sync started", version=version
            )

    async def remove(self) -> Dict[str, Any]:
        async with self._lock:
            if self._operation:
                raise RuntimeError(f"Another {self.label} operation is already running")
            await self._begin("remove")
            active = get_store().get_active_engine_version(self.engine_id)

            async def runner() -> None:
                try:
                    install_dir = os.path.realpath(
                        str((active or {}).get("install_dir") or "")
                    )
                    root_dir = os.path.realpath(self._root_dir)
                    if install_dir and os.path.dirname(install_dir) == root_dir:
                        robust_rmtree(install_dir)
                    if active and active.get("version"):
                        get_store().delete_engine_version(
                            self.engine_id, str(active["version"])
                        )
                    mark_swap_config_stale()
                    await self._finish_operation(True, f"{self.label} removed")
                except Exception as exc:
                    self._last_error = str(exc)
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response(f"{self.label} removal started")

    def status(self) -> Dict[str, Any]:
        active = get_store().get_active_engine_version(self.engine_id)
        venv_path = str((active or {}).get("venv_path") or self._venv_path)
        version = self._detect_installed_version(venv_path)
        python_bin = os.path.join(
            venv_path,
            "Scripts" if os.name == "nt" else "bin",
            "python.exe" if os.name == "nt" else "python",
        )
        return {
            "engine": self.engine_id,
            "installed": bool(version and os.path.isfile(python_bin)),
            "version": version,
            "binary_path": python_bin if os.path.isfile(python_bin) else None,
            "venv_path": venv_path,
            "installed_at": (active or {}).get("installed_at"),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "progress_task_id": self._progress_task_id,
            "last_error": self._last_error,
            "log_path": self._log_path,
            "install_type": (active or {}).get("install_type"),
            "source_repo": (active or {}).get("source_repo"),
            "source_branch": (active or {}).get("source_branch"),
            "source_commit": (active or {}).get("source_commit"),
            "cuda_version": (active or {}).get("cuda_version"),
            "cuda_path": (active or {}).get("cuda_path"),
            "python_version": (active or {}).get("python_version"),
            "v100": self.is_v100,
        }

    def read_log_tail(self, max_bytes: int = 65536) -> str:
        if not os.path.isfile(self._log_path):
            return ""
        with open(self._log_path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - max_bytes))
            text = handle.read().decode("utf-8", errors="replace")
        return text.split("\n", 1)[-1].strip() if size > max_bytes else text.strip()

    def _on_task_error(self, exc: Exception) -> None:
        logger.error("%s manager task error: %s", self.label, exc)
