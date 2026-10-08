import os
import re
import shlex
import subprocess
import shutil
import time
import multiprocessing
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import asyncio
import aiohttp
from backend.logging_config import get_logger
from backend.paths import studio_data_dir
from backend.task_cancel_registry import (
    TaskCancelledError,
    register_task_cancel,
    unregister_task_cancel,
)

logger = get_logger(__name__)


def _build_config_defaults() -> Dict[str, Any]:
    from backend.engines.llama_cpp.build_options import build_config_field_defaults

    return build_config_field_defaults()


_BC = _build_config_defaults()


@dataclass
class BuildConfig:
    """Configuration for building llama.cpp from source.

    Fields mirror upstream GGML_* / LLAMA_* CMake options (see llama_build_options).
    """

    build_type: str = _BC["build_type"]

    # Backends (Studio: CPU + CUDA only)
    enable_cuda: bool = _BC["enable_cuda"]
    enable_blas: bool = _BC["enable_blas"]
    # IQK (ik_llama.cpp)
    enable_iqk_mul_mat: bool = _BC.get("enable_iqk_mul_mat", True)
    enable_iqk_flash_attention: bool = _BC.get("enable_iqk_flash_attention", True)
    enable_iqk_fa_all_quants: bool = _BC.get("enable_iqk_fa_all_quants", True)
    enable_expert_chunking: bool = _BC.get("enable_expert_chunking", True)
    enable_nccl: bool = _BC.get("enable_nccl", True)
    max_contexts: str = _BC.get("max_contexts", "")

    # CUDA
    enable_cuda_fa: bool = _BC["enable_cuda_fa"]
    enable_flash_attention: bool = _BC["enable_flash_attention"]  # FA_ALL_QUANTS
    enable_cuda_graphs: bool = _BC["enable_cuda_graphs"]
    enable_cuda_force_mmq: bool = _BC["enable_cuda_force_mmq"]
    enable_cuda_force_cublas: bool = _BC["enable_cuda_force_cublas"]
    enable_cuda_force_dmmv: bool = _BC.get("enable_cuda_force_dmmv", False)
    enable_cuda_iqk_force_bf16: bool = _BC.get("enable_cuda_iqk_force_bf16", False)
    enable_cuda_f16: bool = _BC.get("enable_cuda_f16", False)
    enable_cuda_no_peer_copy: bool = _BC["enable_cuda_no_peer_copy"]
    enable_cuda_no_vmm: bool = _BC["enable_cuda_no_vmm"]
    enable_cuda_nccl: bool = _BC["enable_cuda_nccl"]
    cuda_architectures: str = _BC["cuda_architectures"]
    cuda_fa_quants: str = _BC.get(
        "cuda_fa_quants", "q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16"
    )
    cuda_peer_max_batch_size: str = _BC["cuda_peer_max_batch_size"]
    cuda_min_batch_offload: str = _BC.get("cuda_min_batch_offload", "32")
    cuda_dmmv_x: str = _BC.get("cuda_dmmv_x", "32")
    cuda_mmv_y: str = _BC.get("cuda_mmv_y", "1")
    cuda_kquants_iter: str = _BC.get("cuda_kquants_iter", "2")
    cuda_fusion: str = _BC.get("cuda_fusion", "1")
    cuda_compression_mode: str = _BC["cuda_compression_mode"]

    # CPU / BLAS
    enable_cpu: bool = _BC["enable_cpu"]
    enable_openmp: bool = _BC["enable_openmp"]
    enable_openmp_fetch: bool = _BC.get("enable_openmp_fetch", False)
    enable_accelerate: bool = _BC["enable_accelerate"]
    enable_llamafile: bool = _BC["enable_llamafile"]
    enable_cpu_hbm: bool = _BC["enable_cpu_hbm"]
    enable_cpu_repack: bool = _BC["enable_cpu_repack"]
    enable_cpu_kleidiai: bool = _BC["enable_cpu_kleidiai"]
    blas_vendor: str = _BC["blas_vendor"]

    # Artifacts
    build_common: bool = _BC["build_common"]
    build_tests: bool = _BC["build_tests"]
    build_tools: bool = _BC["build_tools"]
    build_examples: bool = _BC["build_examples"]
    build_server: bool = _BC["build_server"]
    build_app: bool = _BC["build_app"]
    build_ui: bool = _BC["build_ui"]
    use_prebuilt_ui: bool = _BC["use_prebuilt_ui"]
    build_mtmd: bool = _BC["build_mtmd"]
    install_tools: bool = _BC["install_tools"]
    install_tests: bool = _BC["install_tests"]
    enable_openssl: bool = _BC["enable_openssl"]
    enable_subprocess: bool = _BC["enable_subprocess"]
    enable_llguidance: bool = _BC["enable_llguidance"]

    # GGML general
    enable_native: bool = _BC["enable_native"]
    enable_backend_dl: bool = _BC["enable_backend_dl"]
    enable_cpu_all_variants: bool = _BC["enable_cpu_all_variants"]
    enable_lto: bool = _BC["enable_lto"]
    enable_ccache: bool = _BC["enable_ccache"]
    enable_static: bool = _BC["enable_static"]
    enable_sched_no_realloc: bool = _BC["enable_sched_no_realloc"]
    backend_dir: str = _BC["backend_dir"]
    sched_max_copies: str = _BC["sched_max_copies"]
    cpu_arm_arch: str = _BC["cpu_arm_arch"]
    cpu_powerpc_cputype: str = _BC["cpu_powerpc_cputype"]

    # CPU ISA
    enable_sse42: bool = _BC["enable_sse42"]
    enable_avx: bool = _BC["enable_avx"]
    enable_avx_vnni: bool = _BC["enable_avx_vnni"]
    enable_avx2: bool = _BC["enable_avx2"]
    enable_bmi2: bool = _BC["enable_bmi2"]
    enable_fma: bool = _BC["enable_fma"]
    enable_f16c: bool = _BC["enable_f16c"]
    enable_avx512: bool = _BC["enable_avx512"]
    enable_avx512_vbmi: bool = _BC["enable_avx512_vbmi"]
    enable_avx512_vnni: bool = _BC["enable_avx512_vnni"]
    enable_avx512_bf16: bool = _BC["enable_avx512_bf16"]
    enable_amx_tile: bool = _BC["enable_amx_tile"]
    enable_amx_int8: bool = _BC["enable_amx_int8"]
    enable_amx_bf16: bool = _BC["enable_amx_bf16"]
    enable_sve: bool = _BC.get("enable_sve", False)
    enable_lasx: bool = _BC["enable_lasx"]
    enable_lsx: bool = _BC["enable_lsx"]
    enable_rvv: bool = _BC["enable_rvv"]
    enable_rv_zfh: bool = _BC["enable_rv_zfh"]
    enable_rv_zvfh: bool = _BC["enable_rv_zvfh"]
    enable_rv_zicbop: bool = _BC["enable_rv_zicbop"]
    enable_rv_zihintpause: bool = _BC["enable_rv_zihintpause"]
    enable_rv_zvfbfwma: bool = _BC["enable_rv_zvfbfwma"]
    enable_xtheadvector: bool = _BC["enable_xtheadvector"]
    enable_vxe: bool = _BC["enable_vxe"]

    # Debug
    enable_all_warnings: bool = _BC["enable_all_warnings"]
    enable_fatal_warnings: bool = _BC["enable_fatal_warnings"]
    enable_sanitize_thread: bool = _BC["enable_sanitize_thread"]
    enable_sanitize_address: bool = _BC["enable_sanitize_address"]
    enable_sanitize_undefined: bool = _BC["enable_sanitize_undefined"]
    enable_gprof: bool = _BC["enable_gprof"]

    # Freeform
    custom_cmake_args: str = _BC["custom_cmake_args"]
    cflags: str = _BC["cflags"]
    cxxflags: str = _BC["cxxflags"]
    env_vars: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        self.normalize()

    def normalize(self):
        """
        Normalize combinations that are known to be incompatible in upstream
        CMake configuration. See ggml/src/ggml-cpu/CMakeLists.txt.
        """
        if self.enable_backend_dl:
            if self.enable_native:
                logger.warning(
                    "GGML_BACKEND_DL is enabled; disabling GGML_NATIVE to avoid CMake build failure."
                )
                self.enable_native = False
            if not self.enable_cpu_all_variants:
                logger.info(
                    "GGML_BACKEND_DL is enabled; enabling GGML_CPU_ALL_VARIANTS to ensure CPU variants are available."
                )
                self.enable_cpu_all_variants = True


class LlamaManager:
    # Repository URLs
    LLAMA_CPP_REPO = "https://github.com/ggerganov/llama.cpp.git"
    IK_LLAMA_CPP_REPO = "https://github.com/ikawrakow/ik_llama.cpp.git"
    # Pre-built CUDA releases (ai-dock builds; used for "Install Release")
    LLAMA_CPP_CUDA_RELEASES_API = (
        "https://api.github.com/repos/ai-dock/llama.cpp-cuda/releases"
    )

    REPOSITORY_SOURCES = {
        "llama.cpp": LLAMA_CPP_REPO,
        "ik_llama.cpp": IK_LLAMA_CPP_REPO,
    }

    # Build options: llama.cpp vs ik_llama.cpp
    # - Studio only exposes CPU + CUDA. Other ggml backends are forced OFF.
    # - ik_llama.cpp adds IQK options (GGML_IQK_*) and uses GGML_CUDA_USE_GRAPHS.
    # - ik_llama.cpp puts the server binary under examples/, so LLAMA_BUILD_EXAMPLES must be ON.

    def __init__(self):
        # Use absolute path so clone/build work regardless of process cwd (e.g. --app-dir backend)
        self.llama_dir = os.path.join(studio_data_dir(), "llama-cpp")
        os.makedirs(self.llama_dir, exist_ok=True)
        # Ensure directory has proper permissions (read, write, execute for owner)
        try:
            import stat

            os.chmod(
                self.llama_dir,
                stat.S_IRWXU
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
        except Exception as e:
            logger.warning(f"Could not set permissions on {self.llama_dir}: {e}")
        self._cached_cuda_architectures: Optional[str] = None
        self._cached_cmake_path: Optional[str] = None

    def _check_cuda_toolkit_available(
        self,
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Check if CUDA Toolkit is available on the system.

        Returns:
            Tuple of (is_available, cuda_root, error_message)
            - is_available: True if CUDA Toolkit is found
            - cuda_root: Path to CUDA root directory if found, None otherwise
            - error_message: Error message if not available, None otherwise
        """
        # First, check CUDA installer for installations in data directory
        try:
            from backend.cuda_installer import get_cuda_installer

            installer = get_cuda_installer()
            cuda_path = installer._get_cuda_path()
            if cuda_path and os.path.exists(cuda_path):
                # Verify it has nvcc
                nvcc_path = os.path.join(cuda_path, "bin", "nvcc")
                if os.path.exists(nvcc_path):
                    # Verify toolkit completeness
                    is_complete, missing = self._verify_cuda_toolkit_complete(cuda_path)
                    if is_complete:
                        return (True, cuda_path, None)
                    else:
                        logger.warning(
                            f"CUDA toolkit at {cuda_path} is incomplete, missing: {missing}"
                        )
        except (ImportError, Exception):
            # If CUDA installer is not available or fails, continue with standard checks
            pass

        env = os.environ.copy()
        data_cuda_root = os.path.join(studio_data_dir(), "cuda")
        possible_cuda_roots = [
            env.get("CUDA_PATH"),
            env.get("CUDA_HOME"),
        ]

        # Filter out None values and check if paths exist
        for cuda_root in possible_cuda_roots:
            if not cuda_root or not os.path.exists(cuda_root):
                continue
            if not os.path.abspath(cuda_root).startswith(data_cuda_root):
                continue

            # Check for nvcc compiler
            nvcc_path = os.path.join(cuda_root, "bin", "nvcc")
            if not os.path.exists(nvcc_path):
                # On Windows, nvcc might be in a different location
                if os.name == "nt":
                    nvcc_path = os.path.join(cuda_root, "bin", "nvcc.exe")
                    if not os.path.exists(nvcc_path):
                        continue
                else:
                    continue

            # Verify toolkit completeness (includes headers, libs, etc.)
            is_complete, missing = self._verify_cuda_toolkit_complete(cuda_root)
            if is_complete:
                return (True, cuda_root, None)
            else:
                logger.warning(
                    f"CUDA toolkit at {cuda_root} is incomplete, missing: {missing}"
                )

        # Try to find nvcc in PATH as a fallback
        try:
            result = subprocess.run(
                ["nvcc", "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # nvcc found in PATH, try to determine CUDA root
                nvcc_path = shutil.which("nvcc")
                if nvcc_path:
                    # nvcc is typically in <CUDA_ROOT>/bin/nvcc
                    potential_root = os.path.dirname(os.path.dirname(nvcc_path))
                    if os.path.exists(potential_root) and os.path.abspath(
                        potential_root
                    ).startswith(data_cuda_root):
                        is_complete, missing = self._verify_cuda_toolkit_complete(
                            potential_root
                        )
                        if is_complete:
                            return (True, potential_root, None)
                        else:
                            logger.warning(
                                f"CUDA toolkit at {potential_root} is incomplete, missing: {missing}"
                            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        error_msg = (
            "CUDA Toolkit not found or incomplete. Please either:\n"
            "1. Install CUDA Toolkit via the app's CUDA installer (installs under /app/data)\n"
            "2. Set CUDA_PATH to a CUDA install under /app/data\n"
            "3. Disable CUDA in build configuration (set enable_cuda: false)"
        )
        return (False, None, error_msg)

    def _verify_cuda_toolkit_complete(self, cuda_root: str) -> Tuple[bool, List[str]]:
        """
        Verify that CUDA toolkit has all required components for building.

        Returns:
            Tuple of (is_complete, missing_components)
        """
        missing = []

        # Check for nvcc compiler
        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
        nvcc_path = os.path.join(cuda_root, "bin", nvcc_name)
        if not os.path.exists(nvcc_path):
            missing.append("nvcc compiler")

        # Check for CUDA headers (required for CUDA language support)
        include_dir = os.path.join(cuda_root, "include")
        if not os.path.exists(include_dir):
            missing.append("include directory")
        else:
            # Check for key headers
            key_headers = ["cuda.h", "cuda_runtime.h"]
            for header in key_headers:
                header_path = os.path.join(include_dir, header)
                if not os.path.exists(header_path):
                    missing.append(f"header: {header}")

        # Check for CUDA libraries (both shared and static)
        lib_dirs = ["lib64", "lib"] if os.name != "nt" else ["lib/x64", "lib"]
        has_shared_libs = False
        has_static_libs = False
        cuda_lib_dir = None

        # Also check stubs and targets directories
        all_lib_dirs = lib_dirs + [
            "lib64/stubs",
            "lib/stubs",
            "targets/x86_64-linux/lib",
        ]

        for lib_dir in all_lib_dirs:
            lib_path = os.path.join(cuda_root, lib_dir)
            if os.path.exists(lib_path):
                if cuda_lib_dir is None:
                    cuda_lib_dir = lib_path
                try:
                    lib_files = os.listdir(lib_path)
                    # Check for shared libraries
                    if os.name == "nt":
                        if any("cudart" in f and f.endswith(".dll") for f in lib_files):
                            has_shared_libs = True
                    else:
                        if any("libcudart.so" in f for f in lib_files):
                            has_shared_libs = True

                    # Check for static libraries (required for linking llama.cpp)
                    if any(
                        "cudart_static" in f and f.endswith(".a") for f in lib_files
                    ):
                        has_static_libs = True
                    if any("cudadevrt" in f for f in lib_files):
                        has_static_libs = True
                except OSError:
                    pass

        if not has_shared_libs and not has_static_libs:
            missing.append("CUDA runtime library (cudart)")
        elif not has_static_libs:
            # Log warning but don't mark as missing - might still work
            logger.warning(
                f"CUDA static libraries (cudart_static, cudadevrt) not found in {cuda_root}. "
                "llama.cpp CUDA builds may fail. Install full CUDA toolkit: apt install cuda-toolkit-12-9"
            )

        # Check for version.txt or version.json (indicates full toolkit)
        version_files = ["version.txt", "version.json"]
        has_version = any(
            os.path.exists(os.path.join(cuda_root, vf)) for vf in version_files
        )
        if not has_version:
            # Not critical, just log
            logger.debug(
                f"CUDA toolkit at {cuda_root} missing version file (not critical)"
            )

        # Check for NCCL (optional but recommended for multi-GPU)
        # NCCL can be in the CUDA directory or system directories
        nccl_found = False
        nccl_search_paths = [
            os.path.join(cuda_root, "include", "nccl.h"),
            os.path.join(cuda_root, "include", "nccl_net.h"),
            "/usr/include/nccl.h",
            "/usr/local/include/nccl.h",
        ]
        for nccl_path in nccl_search_paths:
            if os.path.exists(nccl_path):
                nccl_found = True
                break

        # Also check for NCCL library
        if not nccl_found:
            nccl_lib_paths = [
                os.path.join(cuda_root, "lib64"),
                os.path.join(cuda_root, "lib"),
                "/usr/lib/x86_64-linux-gnu",
                "/usr/local/lib",
            ]
            for lib_dir in nccl_lib_paths:
                if os.path.exists(lib_dir):
                    try:
                        lib_files = os.listdir(lib_dir)
                        if any("libnccl" in f for f in lib_files):
                            nccl_found = True
                            break
                    except OSError:
                        pass

        if not nccl_found:
            # NCCL is optional, just log a warning
            logger.info(
                "NCCL not found - multi-GPU support may be limited. Build will continue."
            )
        else:
            logger.debug("NCCL found - multi-GPU support available")

        return (len(missing) == 0, missing)

    def _get_cmake_version(self) -> Optional[Tuple[int, int, int]]:
        """Get CMake version as tuple (major, minor, patch)."""
        try:
            cmake_exe = self._find_cmake_executable()
            if not cmake_exe:
                return None
            result = subprocess.run(
                [cmake_exe, "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Parse "cmake version X.Y.Z"
                match = re.search(r"cmake version (\d+)\.(\d+)\.(\d+)", result.stdout)
                if match:
                    return (
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                    )
        except Exception:
            pass
        return None

    def _find_cmake_executable(self) -> Optional[str]:
        """Find a usable cmake executable from env or PATH."""
        if self._cached_cmake_path and os.path.exists(self._cached_cmake_path):
            return self._cached_cmake_path

        candidates = [
            os.getenv("CMAKE"),
            os.getenv("CMAKE_EXECUTABLE"),
            shutil.which("cmake"),
        ]

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                self._cached_cmake_path = candidate
                return candidate

        return None

    def _get_cuda_version(self, nvcc_path: str) -> Optional[Tuple[int, int]]:
        """Get CUDA version from nvcc as tuple (major, minor)."""
        try:
            result = subprocess.run(
                [nvcc_path, "--version"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                # Parse "release X.Y"
                match = re.search(r"release (\d+)\.(\d+)", result.stdout)
                if match:
                    return (int(match.group(1)), int(match.group(2)))
        except Exception:
            pass
        return None

    async def _detect_cuda_architectures(self) -> Optional[str]:
        """
        Determine CUDA architectures for the current environment by querying GPU capabilities.
        Results are cached because GPU detection can be relatively expensive.
        """
        if self._cached_cuda_architectures is not None:
            return self._cached_cuda_architectures

        try:
            from backend.gpu_detector import get_gpu_info
        except ImportError:
            return None

        try:
            gpu_info = await get_gpu_info()
        except Exception as exc:
            logger.debug(f"Failed to detect GPU architectures: {exc}")
            return None

        if gpu_info.get("vendor") != "nvidia":
            return None

        architectures = []
        for gpu in gpu_info.get("gpus", []):
            compute_capability = gpu.get("compute_capability")
            if not compute_capability:
                continue
            parts = compute_capability.replace(" ", "").split(".")
            if len(parts) != 2:
                continue
            major, minor = parts
            if major.isdigit() and minor.isdigit():
                architectures.append(f"{major}{minor}")

        if not architectures:
            return None

        # Ensure uniqueness and deterministic order
        unique_arches = sorted(set(architectures))
        detected = ";".join(unique_arches)
        self._cached_cuda_architectures = detected
        return detected

    def get_optimal_build_threads(self) -> int:
        """Get optimal number of threads for building based on CPU cores"""
        try:
            cpu_count = multiprocessing.cpu_count()
            # Use 75% of cores, minimum 1, maximum cpu_count
            optimal = max(1, min(cpu_count, int(cpu_count * 0.75)))
            return optimal
        except BaseException:
            return 1  # Fallback to single thread

    async def _run_command(
        self,
        *args,
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        timeout: Optional[int] = None,
        merge_stderr: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a subprocess in a thread for cross-platform compatibility."""

        def _runner():
            return subprocess.run(
                list(args),
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT if merge_stderr else subprocess.PIPE,
                timeout=timeout,
                check=False,
            )

        return await asyncio.to_thread(_runner)

    def _create_build_log_batcher(self, progress_manager, task_id: str):
        """Batch streamed lines into periodic SSE build_progress events."""
        from backend.build_progress import BuildProgressTracker, apply_cmake_stage

        buf: List[str] = []
        last_flush = [0.0]
        # Start in init; callers call apply_cmake_stage before each phase.
        # Tracker window is resynced from ctx whenever the stage changes.
        ctx = apply_cmake_stage(
            {
                "stage": "init",
                "progress": 0,
                "message": "",
                "base_message": "",
                "progress_floor": 0,
                "progress_ceil": 2,
            },
            "init",
            base_message="Starting",
        )
        tracker = BuildProgressTracker(
            floor=int(ctx["progress_floor"]),
            ceil=int(ctx["progress_ceil"]),
            progress=int(ctx["progress"]),
        )

        def _sync_tracker_window() -> None:
            floor = int(ctx.get("progress_floor") or 0)
            ceil = int(ctx.get("progress_ceil") or 100)
            if floor != tracker.floor or ceil != tracker.ceil:
                tracker.set_window(
                    floor,
                    ceil,
                    progress=int(ctx.get("progress") or floor),
                )

        async def emit_line(line: str) -> None:
            if not (progress_manager and task_id and line):
                return
            _sync_tracker_window()
            step = tracker.apply_line(line)
            if step:
                mapped, suffix = step
                ctx["progress"] = mapped
                base = str(ctx.get("base_message") or ctx.get("message") or "Building")
                # Keep the latest git % / [x/y] / [N%] visible in the task message.
                ctx["message"] = f"{base} {suffix}".strip()
            buf.append(line)
            now = time.monotonic()
            if len(buf) >= 48 or (now - last_flush[0]) >= 0.5:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage=str(ctx["stage"]),
                    progress=int(ctx["progress"]),
                    message=str(ctx["message"]),
                    log_lines=list(buf),
                )
                buf.clear()
                last_flush[0] = now

        async def flush(*, complete_stage: bool = False) -> None:
            # Intermediate flushes must NOT snap to ceil — that used to jump the
            # bar to ~92% after git clone finished. Only snap when a stage ends
            # successfully and the caller asks for it (e.g. end of compile).
            if complete_stage:
                _sync_tracker_window()
                ctx["progress"] = tracker.complete()
            if buf and progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage=str(ctx["stage"]),
                    progress=int(ctx["progress"]),
                    message=str(ctx["message"]),
                    log_lines=list(buf),
                )
                buf.clear()

        return ctx, emit_line, flush

    async def _run_command_streaming(
        self,
        args: List[str],
        cwd: Optional[str] = None,
        env: Optional[dict] = None,
        cancel_event: Optional[asyncio.Event] = None,
        on_line: Optional[Callable[[str], Awaitable[None]]] = None,
        merge_stderr: bool = True,
        timeout: Optional[float] = None,
    ) -> SimpleNamespace:
        """Run a subprocess and stream stdout (and optionally merged stderr) line-by-line."""
        if not args:
            raise ValueError("args required")

        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
            if merge_stderr
            else asyncio.subprocess.PIPE,
            start_new_session=(os.name != "nt"),
        )

        deadline = time.monotonic() + timeout if timeout else None
        all_lines: List[str] = []

        async def _kill_proc() -> None:
            if proc.returncode is not None:
                return
            from backend.operations.cancel import terminate_process_tree

            if proc.pid:
                await asyncio.to_thread(
                    terminate_process_tree,
                    proc.pid,
                    term_timeout=3.0,
                    kill_timeout=2.0,
                )
            if proc.returncode is None:
                try:
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
                except (asyncio.TimeoutError, ProcessLookupError, OSError):
                    try:
                        proc.kill()
                    except ProcessLookupError:
                        pass
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=2.0)
                    except Exception:
                        pass

        assert proc.stdout is not None
        try:
            while True:
                if deadline is not None and time.monotonic() > deadline:
                    await _kill_proc()
                    raise asyncio.TimeoutError()
                if cancel_event is not None and cancel_event.is_set():
                    await _kill_proc()
                    raise TaskCancelledError("Build cancelled by user")

                try:
                    line_b = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
                except asyncio.TimeoutError:
                    if proc.returncode is not None:
                        break
                    continue

                if not line_b:
                    break
                text = line_b.decode("utf-8", errors="replace").rstrip("\n\r")
                if text.strip():
                    all_lines.append(text)
                    if on_line:
                        await on_line(text)

            rc = await proc.wait()
            return SimpleNamespace(returncode=rc, lines=all_lines)
        except TaskCancelledError:
            await _kill_proc()
            raise
        except asyncio.TimeoutError:
            await _kill_proc()
            raise
        except asyncio.CancelledError:
            await _kill_proc()
            raise
        except Exception:
            await _kill_proc()
            raise

    async def validate_build(
        self, binary_path: str, progress_manager=None, task_id: str = None
    ) -> bool:
        """Run basic validation on built binary"""
        try:
            # Test 1: Check binary exists and is executable
            if not os.path.exists(binary_path) or not os.access(binary_path, os.X_OK):
                return False

            # Test 2: Run --version command
            process = await self._run_command(binary_path, "--version", timeout=10)
            stdout = process.stdout or b""
            stderr = process.stderr or b""

            if process.returncode != 0:
                return False

            # Test 3: Check for expected output (either "llama" or "version:" string)
            output = stdout.decode() + stderr.decode()
            if "llama" not in output.lower() and "version:" not in output.lower():
                logger.debug(f"Validation output: {output}")
                return False

            return True
        except Exception as e:
            logger.error(f"Build validation failed: {e}")
            return False

    async def _sync_existing_checkout(
        self,
        clone_dir: str,
        branch: str,
        *,
        progress_manager=None,
        task_id: str = None,
        cancel_event: Optional[asyncio.Event] = None,
        log_ctx: Optional[dict] = None,
        emit_line: Optional[Callable[[str], Awaitable[None]]] = None,
        flush_logs: Optional[Callable[[], Awaitable[None]]] = None,
    ) -> None:
        """Fetch and hard-reset a managed source checkout, keeping build cache dirs."""
        branch = str(branch or "").strip()
        if not branch or "\0" in branch:
            raise Exception("A source branch is required for sync")
        if not os.path.isdir(os.path.join(clone_dir, ".git")):
            raise Exception(f"Existing source checkout not found: {clone_dir}")

        async def _noop_line(_line: str = "") -> None:
            return None

        async def _noop_flush(**_kwargs) -> None:
            return None

        emit = emit_line or _noop_line
        flush = flush_logs or _noop_flush
        ctx = log_ctx if log_ctx is not None else {}

        async def run_git(args: List[str], timeout: float = 300.0) -> SimpleNamespace:
            result = await self._run_command_streaming(
                ["git", *args],
                cwd=clone_dir,
                env=os.environ.copy(),
                cancel_event=cancel_event,
                on_line=emit,
                merge_stderr=True,
                timeout=timeout,
            )
            await flush()
            return result

        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="sync",
                progress=2,
                message=f"Fetching latest changes from {branch}...",
                log_lines=[f"Syncing existing checkout at {clone_dir}"],
            )
        from backend.build_progress import apply_cmake_stage

        apply_cmake_stage(
            ctx,
            "sync",
            message=f"Fetching origin/{branch}...",
            base_message=f"Syncing {branch}",
        )

        fetch = await run_git(["fetch", "--prune", "origin", branch])
        if fetch.returncode != 0:
            tail = "\n".join(fetch.lines[-40:]) if fetch.lines else ""
            raise Exception(f"Git fetch failed: {tail or 'unknown error'}")

        if cancel_event is not None and cancel_event.is_set():
            raise TaskCancelledError("Build cancelled by user")

        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="sync",
                progress=8,
                message=f"Resetting checkout to origin/{branch}...",
                log_lines=["Discarding managed checkout changes before rebuild..."],
            )
        ctx["progress"] = max(int(ctx.get("progress") or 0), 8)
        ctx["message"] = f"Resetting checkout to origin/{branch}..."

        checkout = await run_git(["checkout", "-B", branch, "FETCH_HEAD"], timeout=120.0)
        if checkout.returncode != 0:
            if progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage="sync",
                    progress=28,
                    message="Cleaning source conflicts before retry...",
                    log_lines=[
                        "Checkout had local conflicts; cleaning untracked source files while keeping build/."
                    ],
                )
            ctx["progress"] = 28
            ctx["message"] = "Cleaning source conflicts before retry..."
            clean = await run_git(
                [
                    "clean",
                    "-fd",
                    "-e",
                    "build/",
                    "-e",
                    "build",
                    "-e",
                    ".cache/",
                    "-e",
                    ".cache",
                    "-e",
                    "dist/",
                    "-e",
                    "dist",
                ],
                timeout=120.0,
            )
            if clean.returncode != 0:
                tail = "\n".join(clean.lines[-40:]) if clean.lines else ""
                raise Exception(f"Git clean failed: {tail or 'unknown error'}")
            checkout = await run_git(
                ["checkout", "-B", branch, "FETCH_HEAD"], timeout=120.0
            )
            if checkout.returncode != 0:
                tail = "\n".join(checkout.lines[-40:]) if checkout.lines else ""
                raise Exception(f"Git checkout failed after clean: {tail or 'unknown error'}")

        reset = await run_git(["reset", "--hard", "FETCH_HEAD"], timeout=120.0)
        if reset.returncode != 0:
            tail = "\n".join(reset.lines[-40:]) if reset.lines else ""
            raise Exception(f"Git reset failed: {tail or 'unknown error'}")

        logger.info("Synced existing checkout %s to origin/%s", clone_dir, branch)


    async def build_source(
        self,
        commit_sha: str,
        patches: List[str] = None,
        build_config: BuildConfig = None,
        progress_manager=None,
        task_id: str = None,
        repository_url: str = None,
        version_name: str = None,
        reuse_existing_checkout: bool = False,
        source_branch: str = None,
        use_workspace: bool = False,
    ) -> str:
        """Adapt caller arguments onto the source-build contract."""
        from backend.engines.llama_cpp.source_build import run_source_build

        return await run_source_build(
            self,
            commit_sha=commit_sha,
            patches=patches,
            build_config=build_config,
            progress_manager=progress_manager,
            task_id=task_id,
            repository_url=repository_url,
            version_name=version_name,
            reuse_existing_checkout=reuse_existing_checkout,
            source_branch=source_branch,
            use_workspace=use_workspace,
        )

    async def _apply_patch(self, repo_dir: str, patch_url: str):
        """Apply a patch from URL"""
        try:
            if patch_url.startswith("https://github.com/"):
                # GitHub PR URL - convert to patch URL
                if "/pull/" in patch_url:
                    patch_url = patch_url.replace("/pull/", "/pull/").replace(
                        "/files", ".patch"
                    )
                elif not patch_url.endswith(".patch"):
                    patch_url += ".patch"

            # Download patch
            async with aiohttp.ClientSession() as session:
                async with session.get(patch_url) as response:
                    patch_content = await response.text()

            # Apply patch
            patch_file = os.path.join(repo_dir, "temp.patch")
            with open(patch_file, "w") as f:
                f.write(patch_content)

            apply_process = await self._run_command(
                "git",
                "apply",
                patch_file,
                cwd=repo_dir,
                timeout=60,
            )

            if apply_process.returncode != 0:
                apply_stderr = apply_process.stderr or b""
                raise Exception(f"Failed to apply patch: {apply_stderr.decode()}")

            os.remove(patch_file)

        except Exception as e:
            raise Exception(f"Failed to apply patch {patch_url}: {e}")

    def list_installed_versions(self) -> List[str]:
        """List all installed llama.cpp versions"""
        versions = []
        if os.path.exists(self.llama_dir):
            for item in os.listdir(self.llama_dir):
                version_path = os.path.join(self.llama_dir, item)
                if os.path.isdir(version_path):
                    # Check if it has a server binary
                    binary_path = os.path.join(version_path, "server")
                    if os.path.exists(binary_path) and os.access(binary_path, os.X_OK):
                        versions.append(item)
        return versions

    def get_version_path(self, version_name: str) -> Optional[str]:
        """Get the path to a specific version's server binary"""
        version_path = os.path.join(self.llama_dir, version_name)
        if os.path.exists(version_path):
            # Look for server binary in the version directory
            server_path = os.path.join(version_path, "server")
            if os.path.exists(server_path) and os.access(server_path, os.X_OK):
                return server_path

            # Look for llama-server binary in subdirectories
            for root, _, files in os.walk(version_path):
                if "llama-server" in files:
                    llama_server_path = os.path.join(root, "llama-server")
                    if os.path.exists(llama_server_path) and os.access(
                        llama_server_path, os.X_OK
                    ):
                        return llama_server_path
        return None

    def delete_version(self, version_name: str) -> bool:
        """Delete a specific version"""
        version_path = os.path.join(self.llama_dir, version_name)
        if os.path.exists(version_path):
            from backend.utils.fs_ops import FilesystemRefusal, robust_rmtree

            try:
                robust_rmtree(version_path)
                return True
            except FilesystemRefusal:
                raise
            except Exception as e:
                logger.error(f"Failed to delete version {version_name}: {e}")
                return False
        return False

    def verify_installation(self, version_name: str) -> Dict[str, bool]:
        """Verify that all required llama.cpp commands are available for a version"""
        version_path = self.get_version_path(version_name)
        if not version_path:
            return {"llama-server": False, "llama-cli": False, "llama-quantize": False}

        # Check for commands in the same directory
        binary_dir = os.path.dirname(version_path)
        commands = {
            "llama-server": os.path.exists(os.path.join(binary_dir, "llama-server")),
            "llama-cli": os.path.exists(os.path.join(binary_dir, "llama-cli")),
            "llama-quantize": os.path.exists(
                os.path.join(binary_dir, "llama-quantize")
            ),
        }

        return commands

    def get_all_commands(self, version_name: str) -> Dict[str, str]:
        """Get all available commands for a specific version with their full paths"""
        version_path = self.get_version_path(version_name)
        if not version_path:
            return {}

        binary_dir = os.path.dirname(version_path)
        commands = {}

        for cmd in ["llama-server", "llama-cli", "llama-quantize"]:
            cmd_path = os.path.join(binary_dir, cmd)
            if os.path.exists(cmd_path) and os.access(cmd_path, os.X_OK):
                commands[cmd] = cmd_path

        return commands


LlamaCppManager = LlamaManager
