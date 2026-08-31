"""Source-build and tool lifecycle for the native audio.cpp engine."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from backend.logging_config import get_logger
from backend.task_cancel_registry import (
    TaskCancelledError,
    is_task_cancel_requested,
    register_task_cancel,
    unregister_task_cancel,
)
from backend.utils.fs_ops import robust_rmtree


logger = get_logger(__name__)

AUDIO_CPP_REPOSITORY = "https://github.com/0xShug0/audio.cpp.git"
# Bootstrap fallback only when GitHub is unreachable; runtime tracking uses persisted settings.
AUDIO_CPP_DEFAULT_REF = "main"


@dataclass
class AudioCppBuildConfig:
    backend: str = "cpu"
    build_type: str = "RelWithDebInfo"
    cuda: bool = False
    hip: bool = False
    vulkan: bool = False
    metal: bool = False
    native_cpu: bool = True
    openmp: bool = True
    cuda_graphs: bool = True
    llamafile: bool = True
    cpu_all_variants: bool = False
    build_tests: bool = False
    build_examples: bool = False
    build_warmbench: bool = False
    deployment_build: bool = False
    native_model_manager: bool = True
    model_set: str = "full"
    models: str = ""
    jobs: int = 0
    custom_cmake_args: str = ""
    cflags: str = ""
    cxxflags: str = ""

    def normalized(self) -> "AudioCppBuildConfig":
        from backend.audio_build_options import settings_to_field_kwargs

        # Prefer explicit toggles; fall back to legacy backend string
        raw = asdict(self)
        if not any(raw.get(k) for k in ("cuda", "hip", "vulkan", "metal")):
            legacy = str(self.backend or "cpu").strip().lower()
            if legacy in {"cuda", "hip", "vulkan", "metal"}:
                raw[legacy] = True
        kwargs = settings_to_field_kwargs(raw)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.build_type not in {"Debug", "Release", "RelWithDebInfo", "MinSizeRel"}:
            self.build_type = "RelWithDebInfo"
        self.jobs = max(0, int(self.jobs or 0))
        return self


def _data_root() -> str:
    if os.path.isdir("/app/data"):
        return "/app/data"
    return os.path.abspath("data")


def _safe_slug(value: str, *, limit: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    slug = re.sub(r"-{2,}", "-", slug).strip("-._")
    return (slug or "source")[:limit]


def _valid_repository_url(url: str) -> bool:
    value = str(url or "").strip()
    return value.startswith(("https://", "http://", "git@", "ssh://"))


class AudioCppManager:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir or os.path.join(_data_root(), "audio-cpp"))
        self.builds_dir = os.path.join(self.root_dir, "builds")
        self.tools_dir = os.path.join(self.root_dir, "tools")
        self.models_dir = os.path.join(_data_root(), "models", "audio-cpp")
        self.server_configs_dir = os.path.join(
            _data_root(), "config", "audio-cpp", "servers"
        )
        for path in (
            self.root_dir,
            self.builds_dir,
            self.tools_dir,
            self.models_dir,
            self.server_configs_dir,
        ):
            os.makedirs(path, exist_ok=True)
        self._build_lock = asyncio.Lock()
        self._active_process: Optional[asyncio.subprocess.Process] = None

    @staticmethod
    def build_config_from_dict(raw: Optional[dict]) -> AudioCppBuildConfig:
        raw = raw if isinstance(raw, dict) else {}
        allowed = set(AudioCppBuildConfig.__dataclass_fields__)
        values = {key: value for key, value in raw.items() if key in allowed}
        try:
            return AudioCppBuildConfig(**values).normalized()
        except (TypeError, ValueError):
            return AudioCppBuildConfig().normalized()

    @staticmethod
    def supported_build_backends() -> List[str]:
        backends = ["cpu", "cuda", "hip", "vulkan"]
        if sys.platform == "darwin":
            backends.append("metal")
        return backends

    @classmethod
    def validate_build_config(cls, config: AudioCppBuildConfig) -> None:
        config = config.normalized()
        supported = set(cls.supported_build_backends())
        selected = [
            name
            for name in ("cuda", "hip", "vulkan", "metal")
            if getattr(config, name, False)
        ]
        unsupported = [name for name in selected if name not in supported]
        if unsupported:
            raise ValueError(
                f"audio.cpp backend '{unsupported[0]}' is not supported on "
                f"{sys.platform}; supported backends: "
                f"{', '.join(cls.supported_build_backends())}"
            )
        if config.cuda and config.hip:
            raise ValueError(
                "ENGINE_ENABLE_CUDA and ENGINE_ENABLE_HIP are mutually exclusive"
            )

    async def _emit(
        self,
        progress_manager: Any,
        task_id: Optional[str],
        stage: str,
        progress: int,
        message: str,
        *,
        new_lines: Optional[List[str]] = None,
        all_lines: Optional[List[str]] = None,
    ) -> None:
        if not progress_manager or not task_id:
            return
        # Keep a trailing window in task metadata for reconnect; stream only
        # newly produced lines on build_progress so the UI appends without
        # duplicating the whole buffer.
        history = list(all_lines or new_lines or [])[-100:]
        delta = list(new_lines or [])
        progress_manager.update_task(
            task_id,
            progress=float(progress),
            message=message,
            metadata_update={"stage": stage, "log_lines": history},
        )
        if delta:
            progress_manager.emit(
                "build_progress",
                {
                    "task_id": task_id,
                    "stage": stage,
                    "progress": progress,
                    "message": message,
                    "log_lines": delta,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

    def _raise_if_cancelled(self, task_id: Optional[str]) -> None:
        if is_task_cancel_requested(task_id):
            raise TaskCancelledError("audio.cpp build cancelled")

    async def _terminate_active_process(self) -> None:
        process = self._active_process
        if not process or process.returncode is not None:
            return
        try:
            if os.name != "nt":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except asyncio.TimeoutError:
            try:
                if os.name != "nt":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except (ProcessLookupError, PermissionError):
                pass
            await process.wait()

    async def _run_streaming(
        self,
        argv: List[str],
        *,
        cwd: Optional[str],
        task_id: Optional[str],
        progress_manager: Any,
        stage: str,
        progress: int,
        env: Optional[dict] = None,
    ) -> List[str]:
        from backend.build_progress import BuildProgressTracker, cmake_stage_window

        self._raise_if_cancelled(task_id)
        logger.info("audio.cpp command: %s", shlex.join(argv))
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=(os.name != "nt"),
        )
        self._active_process = process
        lines: List[str] = []
        pending: List[str] = []
        last_emit = time.monotonic()
        floor, ceil = cmake_stage_window(stage)
        # Prefer the shared cmake window; allow callers to start mid-stage.
        floor = min(floor, int(progress))
        tracker = BuildProgressTracker(
            floor=floor, ceil=ceil, progress=max(int(progress), floor)
        )
        base_message = {
            "clone": "Cloning audio.cpp",
            "checkout": "Checking out audio.cpp",
            "sync": "Syncing audio.cpp",
            "configure": "Configuring audio.cpp",
            "build": "Building audio.cpp",
            "validate": "Validating audio.cpp",
        }.get(stage, stage)
        try:
            assert process.stdout is not None

            async def flush_pending(message: str = "") -> None:
                nonlocal pending, last_emit
                if not pending:
                    return
                await self._emit(
                    progress_manager,
                    task_id,
                    stage,
                    tracker.progress,
                    message or pending[-1],
                    new_lines=list(pending),
                    all_lines=lines,
                )
                pending = []
                last_emit = time.monotonic()

            while True:
                if is_task_cancel_requested(task_id):
                    await self._terminate_active_process()
                    raise TaskCancelledError("audio.cpp build cancelled")
                try:
                    chunk = await asyncio.wait_for(process.stdout.readline(), timeout=0.25)
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    if pending and (time.monotonic() - last_emit) >= 0.4:
                        await flush_pending()
                    continue
                if not chunk:
                    break
                line = chunk.decode("utf-8", errors="replace").rstrip()
                if line:
                    lines.append(line)
                    pending.append(line)
                    step = tracker.apply_line(line)
                    if step:
                        _progress, suffix = step
                        await flush_pending(f"{base_message} {suffix}")
                    elif len(pending) >= 12 or (time.monotonic() - last_emit) >= 0.4:
                        await flush_pending(line)
            return_code = await process.wait()
            if return_code != 0:
                detail = "\n".join(lines[-40:]).strip()
                raise RuntimeError(
                    f"{stage} failed with exit code {return_code}"
                    + (f":\n{detail}" if detail else "")
                )
            tracker.complete()
            await self._emit(
                progress_manager,
                task_id,
                stage,
                tracker.progress,
                base_message,
                new_lines=[],
                all_lines=lines,
            )
            return lines
        finally:
            self._active_process = None

    @staticmethod
    async def _capture(argv: List[str], *, cwd: Optional[str] = None) -> str:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await process.communicate()
        text = (stdout or b"").decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(text.strip() or f"{argv[0]} exited {process.returncode}")
        return text.strip()

    def _cmake_args(
        self, source_dir: str, build_dir: str, config: AudioCppBuildConfig
    ) -> List[str]:
        from backend.build_progress import prefer_ninja_generator

        config = config.normalized()
        args = [
            "cmake",
            "-S",
            source_dir,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={config.build_type}",
            f"-DENGINE_ENABLE_NATIVE_CPU={'ON' if config.native_cpu else 'OFF'}",
            f"-DENGINE_ENABLE_OPENMP={'ON' if config.openmp else 'OFF'}",
            f"-DENGINE_ENABLE_LLAMAFILE={'ON' if config.llamafile else 'OFF'}",
            f"-DENGINE_ENABLE_CPU_ALL_VARIANTS={'ON' if config.cpu_all_variants else 'OFF'}",
            f"-DENGINE_BUILD_TESTS={'ON' if config.build_tests else 'OFF'}",
            f"-DENGINE_BUILD_EXAMPLES={'ON' if config.build_examples else 'OFF'}",
            f"-DENGINE_BUILD_WARMBENCH={'ON' if config.build_warmbench else 'OFF'}",
            f"-DENGINE_ENABLE_CUDA={'ON' if config.cuda else 'OFF'}",
            f"-DENGINE_ENABLE_HIP={'ON' if config.hip else 'OFF'}",
            f"-DENGINE_ENABLE_VULKAN={'ON' if config.vulkan else 'OFF'}",
            f"-DENGINE_ENABLE_METAL={'ON' if config.metal else 'OFF'}",
            f"-DENGINE_ENABLE_CUDA_GRAPHS={'ON' if config.cuda_graphs else 'OFF'}",
            f"-DAUDIOCPP_DEPLOYMENT_BUILD={'ON' if config.deployment_build else 'OFF'}",
            f"-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER={'ON' if config.native_model_manager else 'OFF'}",
            f"-DAUDIOCPP_MODEL_SET={config.model_set or 'full'}",
        ]
        if (config.model_set or "") == "custom" and str(config.models or "").strip():
            args.append(f"-DAUDIOCPP_MODELS={str(config.models).strip()}")
        if config.custom_cmake_args:
            args.extend(shlex.split(config.custom_cmake_args))
        return prefer_ninja_generator(args)

    @staticmethod
    def _binary_candidates(build_dir: str, name: str) -> List[str]:
        executable = f"{name}.exe" if os.name == "nt" else name
        return [
            os.path.join(build_dir, "bin", executable),
            os.path.join(build_dir, executable),
            os.path.join(build_dir, "Release", executable),
            os.path.join(build_dir, "bin", "Release", executable),
        ]

    def _find_binary(self, build_dir: str, name: str) -> str:
        for candidate in self._binary_candidates(build_dir, name):
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise RuntimeError(f"Built {name} binary was not found under {build_dir}")

    async def _sync_git_checkout(
        self,
        source_dir: str,
        branch: str,
        *,
        task_id: Optional[str],
        progress_manager: Any,
    ) -> None:
        from backend.build_progress import cmake_stage_start

        branch = str(branch or "").strip()
        if not branch or "\0" in branch:
            raise ValueError("A source branch is required for sync")
        if not os.path.isdir(os.path.join(source_dir, ".git")):
            raise ValueError(f"Existing source checkout not found: {source_dir}")

        await self._emit(
            progress_manager,
            task_id,
            "sync",
            cmake_stage_start("sync"),
            f"Fetching origin/{branch}",
        )
        await self._run_streaming(
            ["git", "fetch", "--prune", "origin", branch],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=cmake_stage_start("sync"),
        )
        self._raise_if_cancelled(task_id)
        await self._emit(
            progress_manager,
            task_id,
            "sync",
            8,
            f"Resetting checkout to origin/{branch}",
        )
        await self._run_streaming(
            ["git", "checkout", "-B", branch, "FETCH_HEAD"],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=8,
        )
        await self._run_streaming(
            ["git", "reset", "--hard", "FETCH_HEAD"],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=10,
        )
        await self._run_streaming(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=11,
        )

    async def _compile_tree(
        self,
        source_dir: str,
        build_dir: str,
        config: AudioCppBuildConfig,
        *,
        task_id: Optional[str],
        progress_manager: Any,
    ) -> Dict[str, str]:
        from backend.build_progress import cmake_stage_start

        await self._emit(
            progress_manager,
            task_id,
            "configure",
            cmake_stage_start("configure"),
            f"Configuring {config.backend} build",
        )
        env = os.environ.copy()
        if config.cflags:
            env["CFLAGS"] = config.cflags
        if config.cxxflags:
            env["CXXFLAGS"] = config.cxxflags
        from backend.build_progress import (
            apply_relocated_cuda_warning_flags,
            split_cmake_cli_warning_flags,
        )

        cmake_args, relocated_w_flags = split_cmake_cli_warning_flags(
            self._cmake_args(source_dir, build_dir, config)
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
        await self._run_streaming(
            cmake_args,
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="configure",
            progress=cmake_stage_start("configure"),
            env=env,
        )

        build_argv = [
            "cmake",
            "--build",
            build_dir,
            "--config",
            config.build_type,
            "--parallel",
        ]
        if config.jobs:
            build_argv.append(str(config.jobs))
        build_argv.extend(["--target", "audiocpp_cli", "audiocpp_server"])
        await self._emit(
            progress_manager,
            task_id,
            "build",
            cmake_stage_start("build"),
            "Building audio.cpp",
        )
        await self._run_streaming(
            build_argv,
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="build",
            progress=cmake_stage_start("build"),
        )

        server_binary = self._find_binary(build_dir, "audiocpp_server")
        cli_binary = self._find_binary(build_dir, "audiocpp_cli")
        for binary in (server_binary, cli_binary):
            os.chmod(binary, os.stat(binary).st_mode | 0o111)

        await self._emit(
            progress_manager,
            task_id,
            "validate",
            cmake_stage_start("validate"),
            "Validating audio.cpp binaries",
        )
        server_help, cli_help = await asyncio.gather(
            self._capture([server_binary, "--help"], cwd=os.path.dirname(server_binary)),
            self._capture([cli_binary, "--help"], cwd=os.path.dirname(cli_binary)),
        )
        if not server_help or not cli_help:
            raise RuntimeError("audio.cpp binaries returned empty help output")
        return {
            "server_binary_path": server_binary,
            "cli_binary_path": cli_binary,
        }

    async def build_source(
        self,
        *,
        source_ref: str,
        version_name: str,
        repository_url: str = AUDIO_CPP_REPOSITORY,
        build_config: Optional[AudioCppBuildConfig] = None,
        progress_manager: Any = None,
        task_id: Optional[str] = None,
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        config = (build_config or AudioCppBuildConfig()).normalized()
        self.validate_build_config(config)
        if not _valid_repository_url(repository_url):
            raise ValueError("repository_url must be a valid git clone URL")

        version_name = _safe_slug(version_name)
        version_dir = os.path.abspath(os.path.join(self.builds_dir, version_name))
        if os.path.commonpath([version_dir, self.builds_dir]) != self.builds_dir:
            raise ValueError("Invalid version name")
        if os.path.exists(version_dir):
            if replace_existing:
                robust_rmtree(version_dir)
            else:
                raise FileExistsError(f"audio.cpp version '{version_name}' already exists")

        source_dir = os.path.join(version_dir, "source")
        build_dir = os.path.join(version_dir, "build")
        async with self._build_lock:
            if task_id:
                register_task_cancel(task_id)
            try:
                from backend.build_progress import cmake_stage_start

                await self._emit(
                    progress_manager,
                    task_id,
                    "clone",
                    cmake_stage_start("clone"),
                    "Cloning audio.cpp",
                )
                os.makedirs(version_dir, exist_ok=False)
                await self._run_streaming(
                    ["git", "clone", "--recursive", repository_url, source_dir],
                    cwd=version_dir,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    stage="clone",
                    progress=cmake_stage_start("clone"),
                )
                self._raise_if_cancelled(task_id)
                await self._emit(
                    progress_manager,
                    task_id,
                    "checkout",
                    cmake_stage_start("checkout"),
                    f"Checking out {source_ref}",
                )
                await self._run_streaming(
                    ["git", "checkout", str(source_ref or AUDIO_CPP_DEFAULT_REF)],
                    cwd=source_dir,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    stage="checkout",
                    progress=cmake_stage_start("checkout"),
                )
                await self._run_streaming(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=source_dir,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    stage="checkout",
                    progress=14,
                )

                binaries = await self._compile_tree(
                    source_dir,
                    build_dir,
                    config,
                    task_id=task_id,
                    progress_manager=progress_manager,
                )
                source_commit = await self._capture(
                    ["git", "rev-parse", "HEAD"], cwd=source_dir
                )
                await self._emit(
                    progress_manager, task_id, "complete", 100, "audio.cpp built"
                )
                from backend.audio_cpp_model_managers import manager_paths_for_source

                return {
                    "version": version_name,
                    **binaries,
                    "source_path": source_dir,
                    **manager_paths_for_source(source_dir),
                    "source_commit": source_commit,
                    "source_ref": source_ref,
                    "source_repo": repository_url,
                    "build_config": asdict(config),
                }
            except BaseException:
                raise
            finally:
                if task_id:
                    unregister_task_cancel(task_id)

    def _local_src_dir(self) -> str:
        return os.path.realpath(os.path.join(self.root_dir, "src"))

    def _is_local_checkout(self, version_entry: Optional[Dict[str, Any]], source_dir: str) -> bool:
        install_type = str(
            (version_entry or {}).get("install_type")
            or (version_entry or {}).get("type")
            or ""
        ).strip().lower()
        return install_type == "local" and os.path.realpath(source_dir) == self._local_src_dir()

    def _allowed_sync_source_dir(
        self, source_dir: str, version_entry: Optional[Dict[str, Any]]
    ) -> bool:
        source_real = os.path.realpath(source_dir)
        builds = os.path.realpath(self.builds_dir)
        try:
            if os.path.commonpath([source_real, builds]) == builds:
                return True
        except ValueError:
            return False
        return self._is_local_checkout(version_entry, source_dir)

    def _resolve_sync_build_dir(
        self, version_entry: Optional[Dict[str, Any]], source_dir: str
    ) -> str:
        source_real = os.path.realpath(source_dir)
        for key in ("server_binary_path", "cli_binary_path"):
            binary = str((version_entry or {}).get(key) or "").strip()
            if not binary:
                continue
            current = os.path.dirname(os.path.abspath(binary))
            for _ in range(8):
                if os.path.isfile(os.path.join(current, "CMakeCache.txt")):
                    current_real = os.path.realpath(current)
                    try:
                        if os.path.commonpath([current_real, source_real]) == source_real:
                            return current
                    except ValueError:
                        break
                    break
                parent = os.path.dirname(current)
                if parent == current:
                    break
                current = parent
        if self._is_local_checkout(version_entry, source_dir):
            return os.path.join(source_dir, "build")
        return os.path.join(os.path.dirname(source_dir), "build")

    async def sync_source(
        self,
        *,
        version_entry: Dict[str, Any],
        branch: str,
        build_config: Optional[AudioCppBuildConfig] = None,
        progress_manager: Any = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Pull the tracked branch and rebuild an existing audio.cpp source install."""
        version_name = _safe_slug(str(version_entry.get("version") or "").strip())
        raw_source_path = str(version_entry.get("source_path") or "").strip()
        if not version_name or not raw_source_path:
            raise ValueError("audio.cpp source install metadata is incomplete")
        source_dir = os.path.abspath(raw_source_path)

        if not self._allowed_sync_source_dir(source_dir, version_entry):
            raise ValueError("Refusing to sync audio.cpp files outside the builds root")

        build_dir = self._resolve_sync_build_dir(version_entry, source_dir)
        config = (
            build_config
            or self.build_config_from_dict(version_entry.get("build_config"))
        ).normalized()
        self.validate_build_config(config)
        repository_url = str(
            version_entry.get("source_repo") or AUDIO_CPP_REPOSITORY
        ).strip()

        async with self._build_lock:
            if task_id:
                register_task_cancel(task_id)
            try:
                await self._sync_git_checkout(
                    source_dir,
                    branch,
                    task_id=task_id,
                    progress_manager=progress_manager,
                )
                binaries = await self._compile_tree(
                    source_dir,
                    build_dir,
                    config,
                    task_id=task_id,
                    progress_manager=progress_manager,
                )
                source_commit = await self._capture(
                    ["git", "rev-parse", "HEAD"], cwd=source_dir
                )
                await self._emit(
                    progress_manager, task_id, "complete", 100, "audio.cpp synced"
                )
                from backend.audio_cpp_model_managers import manager_paths_for_source

                return {
                    "version": version_name,
                    **binaries,
                    "source_path": source_dir,
                    **manager_paths_for_source(source_dir),
                    "source_commit": source_commit,
                    "source_ref": branch,
                    "source_ref_type": "branch",
                    "source_branch": branch,
                    "source_repo": repository_url,
                    "build_config": asdict(config),
                    "repository_source": "audio.cpp",
                }
            finally:
                if task_id:
                    unregister_task_cancel(task_id)

    def delete_version_files(self, version_row: Dict[str, Any]) -> None:
        raw_source = str(version_row.get("source_path") or "").strip()
        raw_install = str(version_row.get("install_dir") or "").strip()
        version_name = str(version_row.get("version") or "").strip()
        if raw_source:
            source_path = os.path.abspath(raw_source)
        elif raw_install:
            source_path = os.path.abspath(raw_install)
        elif version_name:
            source_path = os.path.abspath(os.path.join(self.builds_dir, version_name))
        else:
            return
        build_root = os.path.realpath(self.builds_dir)
        source_real = os.path.realpath(source_path)
        if os.path.commonpath([source_real, build_root]) != build_root:
            raise ValueError("Refusing to delete audio.cpp files outside the builds root")
        version_dir = Path(source_real)
        while version_dir.parent != Path(build_root) and version_dir != Path(build_root):
            version_dir = version_dir.parent
        if version_dir == Path(build_root):
            raise ValueError("Could not resolve audio.cpp version directory")
        robust_rmtree(str(version_dir))


_audio_cpp_manager: Optional[AudioCppManager] = None


def get_audio_cpp_manager() -> AudioCppManager:
    global _audio_cpp_manager
    if _audio_cpp_manager is None:
        _audio_cpp_manager = AudioCppManager()
    return _audio_cpp_manager


__all__ = [
    "AUDIO_CPP_DEFAULT_REF",
    "AUDIO_CPP_REPOSITORY",
    "AudioCppBuildConfig",
    "AudioCppManager",
    "get_audio_cpp_manager",
]
