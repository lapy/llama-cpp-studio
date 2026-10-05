"""llama.cpp source build workflow.

LlamaManager.build_source is the adapter. Each build command goes through
run_checked_command with the command runner, the checkout path, and a
cancellation check. Build configuration is a BuildConfig object. Progress
updates use the caller's progress callback. CMake and CUDA lookups stay on
the manager so install-release, CUDA detection, and asset selection are
unchanged.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any, Awaitable, Callable, List, Optional

from backend.engines.llama_cpp.manager import BuildConfig
from backend.logging_config import get_logger
from backend.task_cancel_registry import (
    TaskCancelledError,
    register_task_cancel,
    unregister_task_cancel,
)

logger = get_logger(__name__)


def cleanup_partial_build(checkout_path: str | Path) -> None:
    """Remove an incomplete checkout. A finished install is not this directory."""
    path = Path(checkout_path)
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _cancellation_check(cancel_event) -> Callable[[], bool]:
    return lambda: cancel_event is not None and cancel_event.is_set()


async def run_checked_command(
    run_command: Callable[..., Awaitable[Any]],
    *args: Any,
    cancelled: Callable[[], bool],
    checkout_path: str | Path,
    cleanup_on_failure: bool = True,
    **kwargs: Any,
) -> Any:
    """Run one build command. Cancellation and failure both remove a partial checkout."""
    if cancelled():
        cleanup_partial_build(checkout_path)
        raise TaskCancelledError("Build cancelled by user")
    result = await run_command(*args, **kwargs)
    if cleanup_on_failure and getattr(result, "returncode", 0) != 0:
        cleanup_partial_build(checkout_path)
        detail = getattr(result, "stderr", b"") or b""
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        raise RuntimeError(detail or "build command failed")
    return result


async def _checked_streaming(
    host,
    args: list,
    *,
    cancel_event,
    checkout_path: str,
    cleanup_on_failure: bool = True,
    **kwargs: Any,
) -> Any:
    return await run_checked_command(
        host._run_command_streaming,
        args,
        cancelled=_cancellation_check(cancel_event),
        checkout_path=checkout_path,
        cleanup_on_failure=cleanup_on_failure,
        cancel_event=cancel_event,
        **kwargs,
    )


async def _checked_command(
    host,
    *args: str,
    cancel_event,
    checkout_path: str,
    cleanup_on_failure: bool = True,
    **kwargs: Any,
) -> Any:
    return await run_checked_command(
        host._run_command,
        *args,
        cancelled=_cancellation_check(cancel_event),
        checkout_path=checkout_path,
        cleanup_on_failure=cleanup_on_failure,
        **kwargs,
    )


async def run_source_build(
    host,
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
    """Build llama.cpp from source following official documentation - simplified approach"""
    cancel_event = register_task_cancel(task_id) if task_id else None
    workspace = None
    try:
        if progress_manager and task_id:
            log_ctx, emit_line, flush_logs = host._create_build_log_batcher(
                progress_manager, task_id
            )
        else:
            log_ctx = {
                "stage": "init",
                "progress": 0,
                "message": "",
                "base_message": "",
                "progress_floor": 0,
                "progress_ceil": 2,
            }

            async def emit_line(_=""):
                pass

            async def flush_logs(**_kwargs):
                pass

        # Use default repository if not specified
        if repository_url is None:
            repository_url = host.LLAMA_CPP_REPO

        # Determine repository source name for logging
        repo_source_name = "llama.cpp"
        for source_name, repo_url in host.REPOSITORY_SOURCES.items():
            if repo_url == repository_url:
                repo_source_name = source_name
                break

        # Send initial progress
        if progress_manager and task_id:
            action = "Syncing and rebuilding" if reuse_existing_checkout else "Building"
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="init",
                progress=0,
                message=f"Starting {repo_source_name} source {'sync' if reuse_existing_checkout else 'build'}...",
                log_lines=[f"{action} {repo_source_name} from {commit_sha}"],
            )

        # Use provided version_name or generate default (shouldn't happen, but fallback)
        if version_name is None:
            version_name = f"source-{commit_sha[:8]}"
            logger.warning(
                f"No version_name provided, using default: {version_name}"
            )

        version_dir = os.path.join(host.llama_dir, version_name)

        # Don't clean up existing directory - let API handle uniqueness check
        # This allows multiple builds of the same commit with different names
        os.makedirs(version_dir, exist_ok=True)
        # Ensure directory has proper permissions (read, write, execute for owner)
        try:
            import stat

            os.chmod(
                version_dir,
                stat.S_IRWXU
                | stat.S_IRGRP
                | stat.S_IXGRP
                | stat.S_IROTH
                | stat.S_IXOTH,
            )
        except Exception as e:
            logger.warning(f"Could not set permissions on {version_dir}: {e}")

        clone_dir = os.path.join(version_dir, "llama.cpp")
        if use_workspace and not reuse_existing_checkout:
            from dataclasses import asdict

            from backend.engines.build_workspace import BuildWorkspace

            if build_config is None:
                build_config = BuildConfig()
            else:
                if build_config.enable_cpu_all_variants:
                    build_config.enable_backend_dl = True
                build_config.normalize()
            engine_key = (
                "ik_llama" if repo_source_name == "ik_llama.cpp" else "llama_cpp"
            )
            config_payload = asdict(build_config)
            workspace = BuildWorkspace.open(
                engine_key,
                repository_url,
                config_payload,
                patches or [],
            )
            workspace.acquire()
            clone_dir = workspace.checkout_dir
            workspace.note("building", version_name=version_name, ref=commit_sha)

        if reuse_existing_checkout:
            await host._sync_existing_checkout(
                clone_dir,
                source_branch or commit_sha,
                progress_manager=progress_manager,
                task_id=task_id,
                cancel_event=cancel_event,
                log_ctx=log_ctx,
                emit_line=emit_line,
                flush_logs=flush_logs,
            )
        elif workspace is not None:
            from backend.build_progress import apply_cmake_stage, cmake_stage_start

            apply_cmake_stage(
                log_ctx,
                "clone",
                message=f"Updating the {repo_source_name} build workspace...",
                base_message=f"Updating {repo_source_name}",
            )
            if progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage="clone",
                    progress=cmake_stage_start("clone"),
                    message=log_ctx["message"],
                    log_lines=[
                        "Fetching into the persistent build workspace. "
                        "The installed version is not modified."
                    ],
                )
            await asyncio.to_thread(workspace.sync_git, repository_url, commit_sha)
            await flush_logs(complete_stage=True)
            if cancel_event is not None and cancel_event.is_set():
                cleanup_partial_build(clone_dir)
                raise TaskCancelledError("Build cancelled by user")
        else:
            # Stage 1: Clone repository (stream git --progress to SSE).
            # Retry of a failed build may already have a checkout; resume it.
            from backend.build_progress import apply_cmake_stage, cmake_stage_start

            existing_git = os.path.isdir(os.path.join(clone_dir, ".git"))
            if os.path.isdir(clone_dir) and not existing_git:
                cleanup_partial_build(clone_dir)

            apply_cmake_stage(
                log_ctx,
                "clone",
                message=(
                    f"Reusing existing {repo_source_name} checkout..."
                    if existing_git
                    else f"Cloning {repo_source_name} repository..."
                ),
                base_message=(
                    f"Reusing {repo_source_name}"
                    if existing_git
                    else f"Cloning {repo_source_name}"
                ),
            )
            if progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage="clone",
                    progress=cmake_stage_start("clone"),
                    message=log_ctx["message"],
                    log_lines=[
                        (
                            f"Resuming existing checkout at {clone_dir}"
                            if existing_git
                            else f"Cloning {repo_source_name} repository..."
                        )
                    ],
                )

            if not existing_git:
                try:
                    clone_result = await _checked_streaming(
                        host,
                        [
                            "git",
                            "clone",
                            "--progress",
                            repository_url,
                            clone_dir,
                        ],
                        cancel_event=cancel_event,
                        checkout_path=clone_dir,
                        cleanup_on_failure=False,
                        env=os.environ.copy(),
                        on_line=emit_line,
                        merge_stderr=True,
                        timeout=300.0,
                    )
                    await flush_logs(complete_stage=True)
                    if clone_result.returncode != 0:
                        tail = (
                            "\n".join(clone_result.lines[-40:])
                            if clone_result.lines
                            else ""
                        )
                        cleanup_partial_build(clone_dir)
                        raise Exception(
                            f"Git clone failed: {tail or 'unknown error'}"
                        )
                    logger.info("Repository cloned successfully")
                except asyncio.TimeoutError:
                    logger.error("Git clone timed out")
                    raise Exception("Git clone timed out - network issues")
            else:
                await flush_logs(complete_stage=True)
                logger.info("Reusing existing source checkout at %s", clone_dir)

            if cancel_event is not None and cancel_event.is_set():
                cleanup_partial_build(clone_dir)
                raise TaskCancelledError("Build cancelled by user")

            # Stage 2: Checkout specific commit/branch (simplified)
            from backend.build_progress import apply_cmake_stage, cmake_stage_start

            apply_cmake_stage(
                log_ctx,
                "checkout",
                message=f"Checking out {commit_sha}...",
                base_message="Checking out",
            )
            if progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage="checkout",
                    progress=cmake_stage_start("checkout"),
                    message=f"Checking out {commit_sha}...",
                    log_lines=[f"Checking out {commit_sha}..."],
                )

            try:
                checkout_process = await _checked_command(
                    host,
                    "git",
                    "checkout",
                    commit_sha,
                    cancel_event=cancel_event,
                    checkout_path=clone_dir,
                    cleanup_on_failure=False,
                    cwd=clone_dir,
                    timeout=60,
                )

                if checkout_process.returncode != 0:
                    checkout_stderr = checkout_process.stderr or b""
                    error_msg = checkout_stderr.decode().strip()
                    # Try main when the default branch is not named master
                    if commit_sha == "master":
                        logger.info("Failed to checkout 'master', trying 'main'")
                        main_process = await _checked_command(
                            host,
                            "git",
                            "checkout",
                            "main",
                            cancel_event=cancel_event,
                            checkout_path=clone_dir,
                            cleanup_on_failure=False,
                            cwd=clone_dir,
                            timeout=60,
                        )

                        if main_process.returncode != 0:
                            main_stderr = main_process.stderr or b""
                            raise Exception(
                                f"Failed to checkout both 'master' and 'main': {main_stderr.decode()}"
                            )
                    else:
                        raise Exception(f"Failed to checkout {commit_sha}: {error_msg}")

                logger.info(f"Successfully checked out {commit_sha}")

            except asyncio.TimeoutError:
                raise Exception("Git checkout timed out")

        # Stage 3: Apply patches (if any)
        if cancel_event is not None and cancel_event.is_set():
            raise TaskCancelledError("Build cancelled by user")

        if patches:
            from backend.build_progress import apply_cmake_stage, cmake_stage_start

            apply_cmake_stage(
                log_ctx,
                "patch",
                message=f"Applying {len(patches)} patches...",
                base_message="Applying patches",
            )
            if progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage="patch",
                    progress=cmake_stage_start("patch"),
                    message=f"Applying {len(patches)} patches...",
                    log_lines=[f"Applying {len(patches)} patches..."],
                )

            for patch_url in patches:
                if cancel_event is not None and cancel_event.is_set():
                    raise TaskCancelledError("Build cancelled by user")
                await host._apply_patch(clone_dir, patch_url)

        # Stage 4: Build following official documentation
        from backend.build_progress import cmake_stage_start

        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="configure",
                progress=cmake_stage_start("configure"),
                message="Configuring build with CMake...",
                log_lines=["Running CMake configuration..."],
            )

        # Create build directory
        build_dir = os.path.join(clone_dir, "build")
        os.makedirs(build_dir, exist_ok=True)

        # Prepare build configuration
        if build_config is None:
            build_config = BuildConfig()  # Use defaults
        else:
            # Ensure dependent options stay in sync
            if build_config.enable_cpu_all_variants:
                build_config.enable_backend_dl = True
            build_config.normalize()

        # Enforce build_examples for ik_llama.cpp (server is in examples directory)
        if repo_source_name == "ik_llama.cpp" and not build_config.build_examples:
            logger.warning(
                "ik_llama.cpp requires LLAMA_BUILD_EXAMPLES=ON (server is in examples directory). Enabling automatically."
            )
            build_config.build_examples = True

        # Validate CUDA Toolkit availability if CUDA is enabled
        validated_cuda_root = None
        if build_config.enable_cuda:
            cuda_available, cuda_root, cuda_error = (
                host._check_cuda_toolkit_available()
            )
            if not cuda_available:
                # Check if CUDA installer is available
                try:
                    from backend.cuda_installer import get_cuda_installer

                    installer = get_cuda_installer()
                    installer_status = installer.status()
                    if not installer_status.get("installed"):
                        error_msg = (
                            f"CUDA build requested but CUDA Toolkit not found.\n\n"
                            f"{cuda_error}\n\n"
                            f"You can install CUDA Toolkit using the CUDA installer in the LlamaCpp Manager, "
                            f"or install it manually from https://developer.nvidia.com/cuda-downloads"
                        )
                    else:
                        error_msg = f"CUDA build requested but CUDA Toolkit not found.\n\n{cuda_error}"
                except ImportError:
                    error_msg = f"CUDA build requested but CUDA Toolkit not found.\n\n{cuda_error}"

                logger.error(error_msg)
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="configure",
                        progress=18,
                        message="CUDA Toolkit validation failed",
                        log_lines=[error_msg],
                    )
                raise Exception(error_msg)

            # Verify nvcc is actually executable
            nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
            nvcc_path = os.path.join(cuda_root, "bin", nvcc_name)
            if os.path.exists(nvcc_path):
                try:
                    # Test if nvcc can actually run
                    result = subprocess.run(
                        [nvcc_path, "--version"],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    if result.returncode != 0:
                        error_msg = f"nvcc found at {
                            nvcc_path
                        } but failed to execute (exit code {result.returncode})"
                        logger.error(error_msg)
                        if progress_manager and task_id:
                            await progress_manager.send_build_progress(
                                task_id=task_id,
                                stage="configure",
                                progress=18,
                                message="CUDA compiler verification failed",
                                log_lines=[error_msg],
                            )
                        raise Exception(error_msg)
                    else:
                        logger.info(
                            f"CUDA Toolkit verified at: {cuda_root} (nvcc version: {
                                result.stdout.split(chr(10))[3]
                                if len(result.stdout.split(chr(10))) > 3
                                else 'unknown'
                            })"
                        )
                except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
                    error_msg = f"Failed to verify nvcc at {nvcc_path}: {e}"
                    logger.error(error_msg)
                    if progress_manager and task_id:
                        await progress_manager.send_build_progress(
                            task_id=task_id,
                            stage="configure",
                            progress=18,
                            message="CUDA compiler verification failed",
                            log_lines=[error_msg],
                        )
                    raise Exception(error_msg)
            else:
                error_msg = f"nvcc not found at expected path {nvcc_path} (CUDA root: {cuda_root})"
                logger.error(error_msg)
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="configure",
                        progress=18,
                        message="CUDA compiler not found",
                        log_lines=[error_msg],
                    )
                    raise Exception(error_msg)

            # Store validated CUDA root for later use
            validated_cuda_root = cuda_root

        cmake_exe = host._find_cmake_executable()
        if not cmake_exe:
            raise Exception(
                "CMake was not found. Install CMake or set CMAKE/CMAKE_EXECUTABLE to its path."
            )

        # Build CMake arguments
        cmake_args = [cmake_exe, ".."]

        # Add build type
        cmake_args.append(f"-DCMAKE_BUILD_TYPE={build_config.build_type}")

        def set_flag(flag: str, value: bool):
            state = "ON" if value else "OFF"
            cmake_args.append(f"-D{flag}={state}")

        # Add GPU/compute backends
        set_flag("GGML_CUDA", build_config.enable_cuda)
        from backend.engines.llama_cpp.build_options import append_generic_cmake_flags

        engine_for_flags = (
            "ik_llama" if repo_source_name == "ik_llama.cpp" else "llama_cpp"
        )
        append_generic_cmake_flags(
            cmake_args,
            build_config,
            set_flag=set_flag,
            engine=engine_for_flags,
        )
        from backend.engines.build_workspace import origin_cmake_args

        cmake_args.extend(origin_cmake_args())

        # Explicitly disable CUDA language if CUDA is disabled to prevent auto-detection
        if not build_config.enable_cuda:
            cmake_args.append(
                "-DCMAKE_CUDA_COMPILER="
            )  # Empty string disables CUDA language
        elif validated_cuda_root:
            # === BULLETPROOF CUDA CONFIGURATION ===
            nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
            nvcc_path = os.path.join(validated_cuda_root, "bin", nvcc_name)

            if os.path.exists(nvcc_path):
                # 1. Verify CMake version (3.18+ required for CUDA, 3.20+ for CUDA20)
                cmake_version = host._get_cmake_version()
                if cmake_version:
                    logger.info(
                        f"CMake version: {cmake_version[0]}.{cmake_version[1]}.{cmake_version[2]}"
                    )
                    if cmake_version[0] < 3 or (
                        cmake_version[0] == 3 and cmake_version[1] < 18
                    ):
                        error_msg = (
                            f"CMake version {cmake_version[0]}.{cmake_version[1]}.{cmake_version[2]} is too old for CUDA builds.\n"
                            "CUDA builds require CMake 3.18 or newer (3.20+ recommended for CUDA20).\n"
                            "Please upgrade CMake or disable CUDA in build configuration."
                        )
                        raise Exception(error_msg)

                # 2. Verify CUDA version (11.2+ required for CUDA20 standard)
                cuda_version = host._get_cuda_version(nvcc_path)
                if cuda_version:
                    logger.info(
                        f"CUDA version: {cuda_version[0]}.{cuda_version[1]}"
                    )
                    if cuda_version[0] < 11 or (
                        cuda_version[0] == 11 and cuda_version[1] < 2
                    ):
                        logger.warning(
                            f"CUDA {cuda_version[0]}.{cuda_version[1]} may not fully support CUDA20 standard. "
                            "CUDA 11.2+ is recommended. Build may fail."
                        )

                # 3. Set CMAKE_CUDA_COMPILER (primary way to tell CMake where nvcc is)
                cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc_path}")

                # 4. Set all CUDA toolkit path variables (different CMake versions use different ones)
                cmake_args.append(f"-DCUDAToolkit_ROOT={validated_cuda_root}")
                cmake_args.append(f"-DCUDA_TOOLKIT_ROOT_DIR={validated_cuda_root}")
                cmake_args.append(
                    f"-DCMAKE_CUDA_COMPILER_TOOLKIT_ROOT={validated_cuda_root}"
                )

                # 5. Set CUDA include directories explicitly
                cuda_include = os.path.join(validated_cuda_root, "include")
                if os.path.exists(cuda_include):
                    cmake_args.append(
                        f"-DCMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES={cuda_include}"
                    )

                # 6. === BULLETPROOF CUDA LIBRARY CONFIGURATION ===
                # Collect ALL possible CUDA library directories
                cuda_lib_dirs = []

                # Standard library directories
                for lib_dir in ["lib64", "lib"]:
                    cuda_lib = os.path.join(validated_cuda_root, lib_dir)
                    if os.path.exists(cuda_lib):
                        cuda_lib_dirs.append(cuda_lib)

                # Stubs directory (contains stub libraries for linking)
                for lib_dir in ["lib64/stubs", "lib/stubs"]:
                    stubs_lib = os.path.join(validated_cuda_root, lib_dir)
                    if os.path.exists(stubs_lib):
                        cuda_lib_dirs.append(stubs_lib)

                # Target-specific directory (CUDA 11+)
                targets_lib = os.path.join(
                    validated_cuda_root, "targets", "x86_64-linux", "lib"
                )
                if os.path.exists(targets_lib):
                    cuda_lib_dirs.append(targets_lib)

                # Extras directory
                extras_lib = os.path.join(
                    validated_cuda_root, "extras", "CUPTI", "lib64"
                )
                if os.path.exists(extras_lib):
                    cuda_lib_dirs.append(extras_lib)

                logger.info(f"CUDA library search directories: {cuda_lib_dirs}")

                # Find where static libraries are located
                cudart_static_path = None
                cudadevrt_path = None

                for lib_dir in cuda_lib_dirs:
                    try:
                        if not os.path.exists(lib_dir):
                            continue
                        lib_files = os.listdir(lib_dir)

                        # Look for cudart_static
                        if not cudart_static_path:
                            for f in lib_files:
                                if "cudart_static" in f and f.endswith(".a"):
                                    cudart_static_path = os.path.join(lib_dir, f)
                                    logger.info(
                                        f"Found cudart_static at: {cudart_static_path}"
                                    )
                                    break

                        # Look for cudadevrt
                        if not cudadevrt_path:
                            for f in lib_files:
                                if "cudadevrt" in f and (
                                    f.endswith(".a") or f.endswith(".so")
                                ):
                                    cudadevrt_path = os.path.join(lib_dir, f)
                                    logger.info(
                                        f"Found cudadevrt at: {cudadevrt_path}"
                                    )
                                    break
                    except OSError as e:
                        logger.warning(f"Error scanning {lib_dir}: {e}")

                # Log warnings if libraries are missing
                if not cudart_static_path:
                    logger.warning(
                        "cudart_static not found! This will cause linking errors. "
                        "Ensure full CUDA toolkit is installed (not just runtime). "
                        "Try installing with: apt install cuda-toolkit-12-9"
                    )
                if not cudadevrt_path:
                    logger.warning(
                        "cudadevrt not found! This will cause linking errors. "
                        "Ensure full CUDA toolkit is installed (not just runtime)."
                    )

                # Set ALL CMake library path variables (different CMake versions use different ones)
                unique_lib_dirs = list(
                    dict.fromkeys(cuda_lib_dirs)
                )  # Remove duplicates, preserve order
                lib_paths_cmake = ";".join(unique_lib_dirs)

                cmake_args.append(
                    f"-DCMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES={lib_paths_cmake}"
                )
                cmake_args.append(f"-DCMAKE_LIBRARY_PATH={lib_paths_cmake}")
                cmake_args.append(f"-DCUDA_LIBRARY_PATH={lib_paths_cmake}")
                cmake_args.append(f"-DCUDA_LIB_PATH={lib_paths_cmake}")

                # Set CUDA library paths for FindCUDA module
                cmake_args.append(
                    f"-DCUDA_CUDART_LIBRARY={cudart_static_path or ''}"
                )
                cmake_args.append(
                    f"-DCUDA_cudadevrt_LIBRARY={cudadevrt_path or ''}"
                )

                # Set linker flags directly in CMake
                linker_flags = " ".join([f"-L{d}" for d in unique_lib_dirs])
                cmake_args.append(f'-DCMAKE_EXE_LINKER_FLAGS="{linker_flags}"')
                cmake_args.append(f'-DCMAKE_SHARED_LINKER_FLAGS="{linker_flags}"')

                logger.info(
                    f"CUDA library configuration complete: {len(unique_lib_dirs)} directories"
                )

                # 7. Set CUDA host compiler explicitly (use system gcc/g++)
                if os.name != "nt":
                    gcc_path = shutil.which("gcc")
                    gxx_path = shutil.which("g++")
                    if gcc_path and gxx_path:
                        cmake_args.append(f"-DCMAKE_CUDA_HOST_COMPILER={gxx_path}")

                logger.info(
                    f"CUDA configuration: compiler={nvcc_path}, toolkit={validated_cuda_root}"
                )
        # BLAS (llama.cpp only; ik_llama has no GGML_BLAS)
        if engine_for_flags == "llama_cpp":
            set_flag(
                "GGML_BLAS",
                bool(build_config.enable_blas),
            )
            if build_config.enable_blas:
                vendor = (
                    build_config.blas_vendor or "OpenBLAS"
                ).strip() or "OpenBLAS"
                cmake_args.append(f"-DGGML_BLAS_VENDOR={vendor}")

        set_flag(
            "GGML_CUDA_FA_ALL_QUANTS",
            build_config.enable_flash_attention and build_config.enable_cuda,
        )

        # Auto-detect CUDA architectures when building in NVIDIA containers
        if build_config.enable_cuda:
            cuda_arch = build_config.cuda_architectures.strip()
            if not cuda_arch:
                cuda_arch = await host._detect_cuda_architectures()
            if cuda_arch:
                cmake_args.append(f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}")

        # Disable CURL if not available (llama.cpp requires it by default, but we can disable it)
        # Try to detect if CURL dev headers are available
        try:
            result = subprocess.run(
                ["pkg-config", "--exists", "libcurl"],
                capture_output=True,
                timeout=2,
            )
            if result.returncode != 0:
                # CURL not found, disable it
                set_flag("LLAMA_CURL", False)
                logger.warning(
                    "CURL development headers not found, disabling LLAMA_CURL"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            # pkg-config not available or timeout, try to disable CURL to avoid build failure
            set_flag("LLAMA_CURL", False)
            logger.warning(
                "Could not check for CURL, disabling LLAMA_CURL to avoid build failure"
            )

        # Stable vX.Y.Z tags are release builds. Nightly bNNNN tags,
        # branches, and commit SHAs keep the upstream default (ON).
        if engine_for_flags == "llama_cpp":
            from backend.engines.llama_cpp.github_refs import is_stable_release_tag

            if is_stable_release_tag(commit_sha):
                set_flag("LLAMA_BUILD_IS_DEV", False)

        # Add custom CMake args if provided (can override the flag above)
        if build_config.custom_cmake_args:
            cmake_args.extend(shlex.split(build_config.custom_cmake_args))

        from backend.build_progress import (
            apply_relocated_cuda_warning_flags,
            prefer_ninja_generator,
            split_cmake_cli_warning_flags,
        )

        cmake_args = prefer_ninja_generator(cmake_args)
        cmake_args, relocated_w_flags = split_cmake_cli_warning_flags(cmake_args)

        # Simple CMake configuration following official docs
        try:
            # Set environment variables if provided
            env = os.environ.copy()
            if build_config.env_vars:
                env.update(build_config.env_vars)
            if build_config.enable_ccache:
                from backend.engines.build_workspace import ccache_environment

                env.update(
                    ccache_environment(
                        clone_dir,
                        launchers=False,
                        cuda=bool(build_config.enable_cuda),
                    )
                )
            extra_cmake = str(env.get("CMAKE_ARGS") or "").strip()
            if extra_cmake:
                extra_kept, extra_relocated = split_cmake_cli_warning_flags(
                    shlex.split(extra_cmake)
                )
                relocated_w_flags = list(relocated_w_flags) + extra_relocated
                if extra_relocated:
                    env["CMAKE_ARGS"] = " ".join(extra_kept)
            if relocated_w_flags:
                env = apply_relocated_cuda_warning_flags(env, relocated_w_flags)
                logger.info(
                    "Moved non-CMake -W flags to CMAKE_CUDA_FLAGS: %s",
                    " ".join(relocated_w_flags),
                )

            # Add compiler flags
            if build_config.cflags:
                env["CFLAGS"] = build_config.cflags
            if build_config.cxxflags:
                env["CXXFLAGS"] = build_config.cxxflags

            # === BULLETPROOF CUDA ENVIRONMENT SETUP ===
            if build_config.enable_cuda and validated_cuda_root:
                cuda_root = validated_cuda_root
                nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
                nvcc_path = os.path.join(cuda_root, "bin", nvcc_name)

                # 1. Set ALL CUDA-related environment variables
                env["CUDA_PATH"] = cuda_root
                env["CUDA_HOME"] = cuda_root
                env["CUDA_ROOT"] = cuda_root
                env["CUDACXX"] = nvcc_path
                env["CUDA_COMPILER"] = nvcc_path

                # 2. Add CUDA bin to PATH (at the front for priority)
                cuda_bin = os.path.join(cuda_root, "bin")
                current_path = env.get("PATH", "")
                if cuda_bin not in current_path:
                    env["PATH"] = (
                        f"{cuda_bin}{os.pathsep}{current_path}"
                        if current_path
                        else cuda_bin
                    )

                # 3. === BULLETPROOF LIBRARY PATH CONFIGURATION ===
                if os.name != "nt":
                    # Collect ALL CUDA library directories
                    cuda_lib_paths = []

                    # Standard library directories
                    for lib_dir in ("lib64", "lib"):
                        cuda_lib = os.path.join(cuda_root, lib_dir)
                        if os.path.exists(cuda_lib):
                            cuda_lib_paths.append(cuda_lib)

                    # Stubs directory (critical for linking)
                    for lib_dir in ("lib64/stubs", "lib/stubs"):
                        stubs_lib = os.path.join(cuda_root, lib_dir)
                        if os.path.exists(stubs_lib):
                            cuda_lib_paths.append(stubs_lib)

                    # Target-specific directory
                    targets_lib = os.path.join(
                        cuda_root, "targets", "x86_64-linux", "lib"
                    )
                    if os.path.exists(targets_lib):
                        cuda_lib_paths.append(targets_lib)

                    # Add all CUDA library directories to environment paths
                    for cuda_lib in cuda_lib_paths:
                        # LD_LIBRARY_PATH for runtime
                        current_ld = env.get("LD_LIBRARY_PATH", "")
                        if cuda_lib not in current_ld:
                            env["LD_LIBRARY_PATH"] = (
                                f"{cuda_lib}{os.pathsep}{current_ld}"
                                if current_ld
                                else cuda_lib
                            )

                        # LIBRARY_PATH for linker (critical for finding static libraries)
                        current_lib = env.get("LIBRARY_PATH", "")
                        if cuda_lib not in current_lib:
                            env["LIBRARY_PATH"] = (
                                f"{cuda_lib}{os.pathsep}{current_lib}"
                                if current_lib
                                else cuda_lib
                            )

                    # Set LDFLAGS to explicitly include CUDA library directories
                    if cuda_lib_paths:
                        ldflags_parts = [
                            f"-L{cuda_lib}" for cuda_lib in cuda_lib_paths
                        ]

                        # Also add rpath for runtime library resolution
                        rpath_parts = [
                            f"-Wl,-rpath,{cuda_lib}"
                            for cuda_lib in cuda_lib_paths[:2]
                        ]  # Limit rpath

                        current_ldflags = env.get("LDFLAGS", "")
                        new_ldflags = " ".join(ldflags_parts + rpath_parts)
                        env["LDFLAGS"] = (
                            f"{new_ldflags} {current_ldflags}".strip()
                            if current_ldflags
                            else new_ldflags
                        )

                    # Set CUDA_LIB_PATH for some build systems
                    if cuda_lib_paths:
                        env["CUDA_LIB_PATH"] = os.pathsep.join(cuda_lib_paths)

                    # Set CPATH for CUDA headers
                    cuda_include = os.path.join(cuda_root, "include")
                    if os.path.exists(cuda_include):
                        current_cpath = env.get("CPATH", "")
                        if cuda_include not in current_cpath:
                            env["CPATH"] = (
                                f"{cuda_include}{os.pathsep}{current_cpath}"
                                if current_cpath
                                else cuda_include
                            )
                else:
                    # Windows: Add lib to PATH for DLLs
                    for lib_dir in ("lib/x64", "lib"):
                        cuda_lib = os.path.join(cuda_root, lib_dir)
                        if os.path.exists(cuda_lib) and cuda_lib not in env["PATH"]:
                            env["PATH"] = f"{cuda_lib}{os.pathsep}{env['PATH']}"
                            break

                # 4. Log the complete CUDA environment
                logger.info(
                    f"CUDA environment configured:\n"
                    f"  CUDA_PATH={env['CUDA_PATH']}\n"
                    f"  CUDACXX={env['CUDACXX']}\n"
                    f"  PATH includes: {cuda_bin}\n"
                    f"  LD_LIBRARY_PATH={env.get('LD_LIBRARY_PATH', 'not set')}"
                )

            # Log cmake arguments for debugging
            from backend.build_progress import apply_cmake_stage

            apply_cmake_stage(
                log_ctx,
                "configure",
                message="Configuring build with CMake...",
                base_message="Configuring build",
            )
            logger.info(f"CMake command: {' '.join(cmake_args)}")

            cmake_result = await _checked_streaming(
                host,
                [str(a) for a in cmake_args],
                cancel_event=cancel_event,
                checkout_path=clone_dir,
                cleanup_on_failure=False,
                cwd=build_dir,
                env=env,
                on_line=emit_line,
                merge_stderr=True,
                timeout=180.0,
            )
            await flush_logs(complete_stage=True)

            if cmake_result.returncode != 0:
                error_msg = "\n".join(cmake_result.lines[-40:]).strip()
                logger.warning(f"CMake configuration failed: {error_msg}")
                cleanup_partial_build(clone_dir)

                # Provide more helpful error messages for CUDA-related failures
                if build_config.enable_cuda and (
                    "CUDA" in error_msg.upper()
                    or "cuda" in error_msg.lower()
                    or "CUDA20" in error_msg
                ):
                    # Collect diagnostic information
                    diag_info = []

                    if validated_cuda_root:
                        nvcc_name = "nvcc.exe" if os.name == "nt" else "nvcc"
                        nvcc_path = os.path.join(
                            validated_cuda_root, "bin", nvcc_name
                        )

                        # Check toolkit completeness
                        is_complete, missing = host._verify_cuda_toolkit_complete(
                            validated_cuda_root
                        )
                        if not is_complete:
                            diag_info.append(
                                f"CUDA toolkit incomplete, missing: {', '.join(missing)}"
                            )

                        # Get versions
                        cmake_ver = host._get_cmake_version()
                        cuda_ver = (
                            host._get_cuda_version(nvcc_path)
                            if os.path.exists(nvcc_path)
                            else None
                        )

                        diag_info.append(f"CUDA_PATH: {validated_cuda_root}")
                        diag_info.append(
                            f"nvcc exists: {os.path.exists(nvcc_path)}"
                        )
                        diag_info.append(
                            f"include dir exists: {os.path.exists(os.path.join(validated_cuda_root, 'include'))}"
                        )
                        diag_info.append(
                            f"CMake version: {cmake_ver[0]}.{cmake_ver[1]}.{
                                cmake_ver[2] if cmake_ver else 'unknown'
                            }"
                        )
                        diag_info.append(
                            f"CUDA version: {cuda_ver[0]}.{cuda_ver[1] if cuda_ver else 'unknown'}"
                        )

                        # Check for CUDA20 specific error
                        if "CUDA20" in error_msg:
                            diag_info.append("")
                            diag_info.append(
                                "CUDA20 dialect error: This requires CMake 3.20+ and CUDA 11.2+"
                            )
                            if cmake_ver and (
                                cmake_ver[0] < 3
                                or (cmake_ver[0] == 3 and cmake_ver[1] < 20)
                            ):
                                diag_info.append(
                                    f"Your CMake ({cmake_ver[0]}.{cmake_ver[1]}) is too old for CUDA20"
                                )
                            if cuda_ver and (
                                cuda_ver[0] < 11
                                or (cuda_ver[0] == 11 and cuda_ver[1] < 2)
                            ):
                                diag_info.append(
                                    f"Your CUDA ({cuda_ver[0]}.{cuda_ver[1]}) is too old for CUDA20"
                                )

                    diagnostics = (
                        "\n".join(f"  - {d}" for d in diag_info)
                        if diag_info
                        else "  (no diagnostic info available)"
                    )

                    enhanced_error = (
                        f"CMake configuration failed with CUDA error:\n\n{error_msg}\n\n"
                        f"Diagnostic information:\n{diagnostics}\n\n"
                        "Possible solutions:\n"
                        "1. Upgrade CMake to 3.20 or newer\n"
                        "2. Upgrade CUDA Toolkit to 11.2 or newer\n"
                        "3. Ensure CUDA Toolkit is fully installed (not just runtime)\n"
                        "4. Disable CUDA in build configuration (set enable_cuda: false)"
                    )
                    raise Exception(enhanced_error)
                else:
                    raise Exception(f"CMake configuration failed: {error_msg}")

            logger.info("CMake configuration completed successfully")

            # List available targets for debugging (especially useful for ik_llama.cpp)
            try:
                targets_process = await _checked_command(
                    host,
                    cmake_exe,
                    "--build",
                    ".",
                    "--target",
                    "help",
                    cancel_event=cancel_event,
                    checkout_path=clone_dir,
                    cleanup_on_failure=False,
                    cwd=build_dir,
                    env=env,
                    timeout=30,
                )
                targets_stdout = targets_process.stdout or b""
                if targets_process.returncode == 0:
                    targets_output = targets_stdout.decode(
                        "utf-8", errors="replace"
                    )
                    # Extract target names (look for lines with "..." which indicate targets)
                    target_lines = [
                        line.strip()
                        for line in targets_output.split("\n")
                        if "..." in line
                        or "llama" in line.lower()
                        or "server" in line.lower()
                    ]
                    if target_lines:
                        logger.info(
                            f"Available CMake targets (sample): {target_lines[:10]}"
                        )
                        # Check if llama-server target exists
                        if not any(
                            "llama-server" in line.lower()
                            or "server" in line.lower()
                            for line in target_lines
                        ):
                            logger.warning(
                                f"llama-server target not found in available targets. Repository: {repo_source_name}"
                            )
                            if progress_manager and task_id:
                                await progress_manager.send_build_progress(
                                    task_id=task_id,
                                    stage="configure",
                                    progress=22,
                                    message="Warning: llama-server target not found, will try building all targets",
                                    log_lines=["Available targets (sample):"]
                                    + target_lines[:5],
                                )
            except Exception as targets_error:
                logger.debug(f"Could not list CMake targets: {targets_error}")
                # Non-critical, continue with build

        except asyncio.TimeoutError:
            raise Exception("CMake configuration timed out")

        # Stage 5: Build
        from backend.build_progress import apply_cmake_stage, cmake_stage_start

        apply_cmake_stage(
            log_ctx,
            "build",
            message="Building llama.cpp...",
            base_message="Building llama.cpp",
        )
        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="build",
                progress=cmake_stage_start("build"),
                message="Building llama.cpp...",
                log_lines=["Starting compilation..."],
            )

        # Build with optimal thread count
        try:
            thread_count = host.get_optimal_build_threads()
            logger.info(f"Building with {thread_count} threads")

            # CRITICAL: Reuse the same environment from CMake configuration
            # (env variable was already set up above with all CUDA paths)
            # Only update with user's env_vars if not already set
            if build_config.env_vars:
                for key, value in build_config.env_vars.items():
                    if key not in env:  # Don't override CUDA settings
                        env[key] = value

            # Log environment for debugging linker issues
            if build_config.enable_cuda:
                logger.info(
                    f"Build environment LIBRARY_PATH: {env.get('LIBRARY_PATH', 'not set')}"
                )
                logger.info(
                    f"Build environment LDFLAGS: {env.get('LDFLAGS', 'not set')}"
                )

            # Explicitly build llama-server target (stream full compiler output)
            build_result = await _checked_streaming(
                host,
                [
                    str(cmake_exe),
                    "--build",
                    ".",
                    "--target",
                    "llama-server",
                    "--parallel",
                    str(thread_count),
                ],
                cancel_event=cancel_event,
                checkout_path=clone_dir,
                cleanup_on_failure=False,
                cwd=build_dir,
                env=env,
                on_line=emit_line,
                merge_stderr=True,
                timeout=1800.0,
            )
            await flush_logs(complete_stage=True)

            build_output_lines = list(build_result.lines)
            build_output = "\n".join(build_output_lines)
            returncode = build_result.returncode

            if returncode != 0:
                logger.error(f"Build failed with return code {returncode}")
                logger.error(f"Build output:\n{build_output}")
                # Send error output via SSE if available
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="build",
                        progress=28,
                        message=f"Build failed (exit code {returncode})",
                        log_lines=build_output_lines[-120:]
                        if len(build_output_lines) > 120
                        else build_output_lines,
                    )
                cleanup_partial_build(clone_dir)
                raise Exception(
                    f"Build failed (exit code {returncode}). Check logs for details."
                )

            # Check if build output indicates actual success
            # Sometimes cmake returns 0 even if target wasn't built
            build_output_lower = build_output.lower()
            has_build_errors = any(
                keyword in build_output_lower
                for keyword in [
                    "error",
                    "failed",
                    "fatal",
                    "undefined reference",
                    "cannot find",
                    "no rule to make target",
                ]
            )

            # Check if llama-server was actually built (look for linking or building messages)
            # Also check for "up to date" which means target exists but wasn't rebuilt
            target_built = any(
                indicator in build_output_lower
                for indicator in [
                    "llama-server",
                    "linking",
                    "built target",
                    "server",
                    "up to date",
                ]
            )

            # Check if target was skipped or not found
            target_not_found = any(
                keyword in build_output_lower
                for keyword in [
                    "no rule to make target",
                    "target.*not found",
                    "unknown target",
                ]
            )

            if target_not_found:
                logger.warning(
                    "Build target 'llama-server' not found, trying 'server' target (for examples/server)..."
                )
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="build",
                        progress=28,
                        message="Trying 'server' target instead...",
                        log_lines=[
                            "Target 'llama-server' not found, trying 'server' target..."
                        ],
                    )

                # Try 'server' target (used when server is in examples/)
                logger.info("Attempting to build 'server' target...")
                server_target_result = await _checked_streaming(
                    host,
                    [
                        str(cmake_exe),
                        "--build",
                        ".",
                        "--target",
                        "server",
                        "--parallel",
                        str(thread_count),
                    ],
                    cancel_event=cancel_event,
                    checkout_path=clone_dir,
                    cleanup_on_failure=False,
                    cwd=build_dir,
                    env=env,
                    on_line=emit_line,
                    merge_stderr=True,
                    timeout=1800.0,
                )
                await flush_logs(complete_stage=True)

                server_target_output_lines = list(server_target_result.lines)
                server_target_output = "\n".join(server_target_output_lines)
                server_target_returncode = server_target_result.returncode

                if server_target_returncode == 0:
                    logger.info("Successfully built 'server' target")
                    build_output = server_target_output
                    build_output_lines = server_target_output_lines
                else:
                    logger.error(
                        "Build target 'server' also failed, trying all targets as last resort"
                    )
                    logger.error(
                        f"Server target build output:\n{server_target_output}"
                    )
                    if progress_manager and task_id:
                        await progress_manager.send_build_progress(
                            task_id=task_id,
                            stage="build",
                            progress=28,
                            message="Server target failed, building all targets...",
                            log_lines=[
                                "Target 'server' also not found, building all targets..."
                            ],
                        )
                    logger.info("Attempting to build all targets as fallback...")
                    all_targets_result = await _checked_streaming(
                        host,
                        [
                            str(cmake_exe),
                            "--build",
                            ".",
                            "--parallel",
                            str(thread_count),
                        ],
                        cancel_event=cancel_event,
                        checkout_path=clone_dir,
                        cleanup_on_failure=False,
                        cwd=build_dir,
                        env=env,
                        on_line=emit_line,
                        merge_stderr=True,
                        timeout=1800.0,
                    )
                    await flush_logs(complete_stage=True)

                    all_targets_output_lines = list(all_targets_result.lines)
                    all_targets_output = "\n".join(all_targets_output_lines)
                    all_targets_returncode = all_targets_result.returncode

                    if all_targets_returncode != 0:
                        logger.error(
                            f"Building all targets failed with return code {all_targets_returncode}"
                        )
                        logger.error(f"Build output:\n{all_targets_output}")
                        raise Exception(
                            f"Build target 'llama-server' not found and building all targets failed (exit code {all_targets_returncode})"
                        )

                    build_output_lines.extend(all_targets_output_lines)
                    build_output = "\n".join(build_output_lines)
                    logger.info(
                        "Building all targets completed, will search for binary"
                    )

            if has_build_errors and not target_built and not target_not_found:
                logger.error(
                    "Build completed with return code 0 but contains errors"
                )
                logger.error(f"Build output:\n{build_output}")
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="build",
                        progress=28,
                        message="Build completed but contains errors",
                        log_lines=build_output_lines[-200:]
                        if len(build_output_lines) > 200
                        else build_output_lines,
                    )
                raise Exception(
                    "Build completed but contains errors. Check logs for details."
                )

            logger.info("Build completed successfully")
            logger.info(
                "Build output (last 20 lines):\n"
                + "\n".join(build_output_lines[-20:])
            )
            if not target_built and not target_not_found:
                logger.warning(
                    "Build output doesn't clearly indicate llama-server was built - will verify binary exists"
                )

            # Immediately check if binary exists in common locations
            # Note: For ik_llama.cpp, binary is in clone_dir/bin/ (parent of build_dir)
            clone_dir = (
                os.path.dirname(build_dir)
                if os.path.basename(build_dir) == "build"
                else build_dir
            )
            quick_check_paths = [
                os.path.join(
                    clone_dir, "bin", "llama-server"
                ),  # Common location for ik_llama.cpp
                os.path.join(build_dir, "bin", "llama-server"),
                os.path.join(build_dir, "llama-server"),
            ]
            binary_found_quick = False
            for quick_path in quick_check_paths:
                if os.path.exists(quick_path):
                    logger.info(
                        f"Binary found immediately after build: {quick_path}"
                    )
                    binary_found_quick = True
                    break

            if not binary_found_quick:
                logger.warning(
                    "Binary not found in common locations immediately after build - will search more thoroughly"
                )
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="build",
                        progress=92,
                        message="Build completed, searching for binary...",
                        log_lines=[
                            "Binary not found in expected location, searching..."
                        ],
                    )

        except asyncio.TimeoutError:
            raise Exception("Build timed out")

        # Stage 6: Find executable
        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="verify",
                progress=92,
                message="Verifying build...",
                log_lines=["Searching for llama-server..."],
            )

        # Find llama-server executable
        final_server_path = None

        # Check common locations first
        # Note: For ik_llama.cpp and some builds, binary is in clone_dir/bin/ (parent of build_dir)
        clone_dir = (
            os.path.dirname(build_dir)
            if os.path.basename(build_dir) == "build"
            else build_dir
        )
        common_paths = [
            os.path.join(
                clone_dir, "bin", "llama-server"
            ),  # Common location for ik_llama.cpp
            os.path.join(build_dir, "bin", "llama-server"),
            os.path.join(build_dir, "llama-server"),
            os.path.join(build_dir, "server", "llama-server"),
        ]

        for path in common_paths:
            if os.path.exists(path):
                final_server_path = path
                logger.info(f"Found llama-server at: {final_server_path}")
                break

        # If not found in common locations, search recursively (look for both llama-server and server)
        if not final_server_path:
            logger.warning(
                "llama-server not found in common locations, searching recursively..."
            )
            for root, _, files in os.walk(build_dir):
                # Check for llama-server first (standard name)
                if "llama-server" in files:
                    final_server_path = os.path.join(root, "llama-server")
                    logger.info(f"Found llama-server at: {final_server_path}")
                    break
                # Also check for just "server" (used in examples/server for some forks)
                if "server" in files and os.path.isfile(
                    os.path.join(root, "server")
                ):
                    # Make sure it's executable and not a directory
                    server_path = os.path.join(root, "server")
                    if os.access(server_path, os.X_OK):
                        final_server_path = server_path
                        logger.info(f"Found server at: {final_server_path}")
                        break

        if not final_server_path or not os.path.exists(final_server_path):
            # List what was actually built
            logger.error(
                f"llama-server executable not found after build in {build_dir}"
            )
            logger.error(f"Repository source: {repo_source_name}")
            logger.error(f"Build directory: {build_dir}")
            logger.error("Searching for any executables in build directory...")

            # Also check the clone directory in case build structure is different
            executables_found = []
            search_dirs = [build_dir, clone_dir]

            for search_dir in search_dirs:
                if not os.path.exists(search_dir):
                    continue
                for root, _, files in os.walk(search_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Check if it's an executable (Unix) or has executable extension (Windows)
                        is_executable = (
                            os.access(file_path, os.X_OK)
                            if os.name != "nt"
                            else file_path.endswith((".exe", ".bat", ".cmd"))
                        )
                        if is_executable and os.path.isfile(file_path):
                            rel_path = os.path.relpath(file_path, build_dir)
                            executables_found.append(rel_path)

            # Also check for server-related binaries with different names
            server_variants = [
                "server",
                "llama_server",
                "llama-server.exe",
                "server.exe",
            ]
            for variant in server_variants:
                for search_dir in search_dirs:
                    if not os.path.exists(search_dir):
                        continue
                    for root, _, files in os.walk(search_dir):
                        if variant in files:
                            variant_path = os.path.join(root, variant)
                            if os.path.exists(variant_path):
                                logger.warning(
                                    f"Found server variant '{variant}' at: {variant_path}"
                                )
                                # Try to use this as the server path
                                final_server_path = variant_path
                                break
                    if final_server_path:
                        break
                if final_server_path:
                    break

            if not final_server_path:
                error_msg = (
                    f"llama-server executable not found after build in {build_dir}"
                )
                if executables_found:
                    logger.error(f"Found executables: {executables_found}")
                    error_msg += f"\n\nFound executables: {', '.join(executables_found[:20])}"
                    error_msg += "\n\nThis might indicate:\n"
                    error_msg += f"1. The build target name is different for {repo_source_name}\n"
                    error_msg += "2. The build structure is different\n"
                    error_msg += "3. The build failed silently\n\n"
                    error_msg += "Please check the build logs for errors."
                else:
                    error_msg += "\n\nNo executables found in build directory. This indicates the build likely failed silently."
                    error_msg += "\n\nPlease check:\n"
                    error_msg += "1. Build configuration is correct\n"
                    error_msg += "2. All dependencies are installed\n"
                    error_msg += "3. CMake configuration succeeded\n"
                    error_msg += "4. Build output for errors"

                # Send detailed error via SSE
                if progress_manager and task_id:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="error",
                        progress=0,
                        message="Build completed but binary not found",
                        log_lines=[error_msg]
                        + (executables_found[:10] if executables_found else []),
                    )

                raise Exception(error_msg)

        # Make executable
        os.chmod(final_server_path, 0o755)

        if workspace is not None:
            relative_binary = os.path.relpath(final_server_path, clone_dir)
            published = await asyncio.to_thread(
                workspace.publish_tree,
                clone_dir,
                os.path.join(version_dir, "llama.cpp"),
                [relative_binary],
            )
            version_server_path = os.path.join(published, relative_binary)
            if not os.path.isfile(version_server_path):
                raise RuntimeError(
                    "The published snapshot does not contain llama-server. "
                    "The previous engine version was left unchanged."
                )
        else:
            # Copy to version directory for easy access
            version_server_path = os.path.join(version_dir, "llama-server")
            shutil.copy2(final_server_path, version_server_path)
        os.chmod(version_server_path, 0o755)

        logger.info(f"Build completed, validating binary: {version_server_path}")

        # Validate the build
        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="validate",
                progress=92,
                message="Validating build...",
                log_lines=["Running validation tests..."],
            )

        is_valid = await host.validate_build(
            version_server_path, progress_manager, task_id
        )

        if not is_valid:
            logger.warning("Build validation failed")
            if progress_manager and task_id:
                await progress_manager.send_build_progress(
                    task_id=task_id,
                    stage="validate",
                    progress=92,
                    message="Build validation failed - binary may not work correctly",
                    log_lines=["Warning: Build validation failed"],
                )

        logger.info(f"Build completed successfully: {version_server_path}")

        await flush_logs()
        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id=task_id,
                stage="complete",
                progress=100,
                message="Build completed successfully!",
                log_lines=[
                    f"llama-server ready at: {version_server_path}",
                    f"Validation: {'Passed' if is_valid else 'Failed'}",
                ],
            )

        return version_server_path

    except TaskCancelledError:
        raise
    except Exception as e:
        logger.error(f"Build failed: {e}")
        if progress_manager and task_id:
            try:
                existing_task = progress_manager.get_task(task_id)
                existing_logs = (existing_task or {}).get("metadata", {}).get(
                    "log_lines"
                ) or []
                error_text = str(e)
                if error_text not in existing_logs:
                    await progress_manager.send_build_progress(
                        task_id=task_id,
                        stage="error",
                        progress=0,
                        message=f"Build failed: {error_text}",
                        log_lines=[f"Error: {error_text}"],
                    )
            except Exception as ws_error:
                logger.error(f"Failed to send error via SSE: {ws_error}")
        raise Exception(f"Failed to build from source {commit_sha}: {e}")
    finally:
        from backend.engines.build_workspace import release_held

        release_held(workspace)
        if task_id:
            unregister_task_cancel(task_id)

