"""Source-build and tool lifecycle for the native audio.cpp engine."""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
AUDIO_CPP_DEFAULT_REF = "release-0.2"
AUDIO_CPP_COMPATIBILITY_COMMIT = "88fe1fc217358d5ea84497b0b90161be63ff9fb8"


@dataclass
class AudioCppBuildConfig:
    backend: str = "cpu"
    build_type: str = "RelWithDebInfo"
    native_cpu: bool = True
    openmp: bool = True
    cuda_graphs: bool = True
    jobs: int = 0
    custom_cmake_args: str = ""

    def normalized(self) -> "AudioCppBuildConfig":
        backend = str(self.backend or "cpu").strip().lower()
        if backend not in {"cpu", "cuda", "vulkan", "metal"}:
            backend = "cpu"
        self.backend = backend
        if self.build_type not in {"Debug", "Release", "RelWithDebInfo"}:
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
        backends = ["cpu", "cuda", "vulkan"]
        if sys.platform == "darwin":
            backends.append("metal")
        return backends

    @classmethod
    def validate_build_config(cls, config: AudioCppBuildConfig) -> None:
        if config.backend not in cls.supported_build_backends():
            raise ValueError(
                f"audio.cpp backend '{config.backend}' is not supported on "
                f"{sys.platform}; supported backends: "
                f"{', '.join(cls.supported_build_backends())}"
            )

    async def _emit(
        self,
        progress_manager: Any,
        task_id: Optional[str],
        stage: str,
        progress: int,
        message: str,
        lines: Optional[List[str]] = None,
    ) -> None:
        if progress_manager and task_id:
            await progress_manager.send_build_progress(
                task_id,
                stage,
                progress,
                message=message,
                log_lines=(lines or [])[-40:],
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
        try:
            assert process.stdout is not None
            while True:
                if is_task_cancel_requested(task_id):
                    await self._terminate_active_process()
                    raise TaskCancelledError("audio.cpp build cancelled")
                try:
                    chunk = await asyncio.wait_for(process.stdout.readline(), timeout=0.25)
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    continue
                if not chunk:
                    break
                line = chunk.decode("utf-8", errors="replace").rstrip()
                if line:
                    lines.append(line)
                    if len(lines) % 20 == 0:
                        await self._emit(
                            progress_manager,
                            task_id,
                            stage,
                            progress,
                            line,
                            lines,
                        )
            return_code = await process.wait()
            if return_code != 0:
                detail = "\n".join(lines[-40:]).strip()
                raise RuntimeError(
                    f"{stage} failed with exit code {return_code}"
                    + (f":\n{detail}" if detail else "")
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
        args = [
            "cmake",
            "-S",
            source_dir,
            "-B",
            build_dir,
            f"-DCMAKE_BUILD_TYPE={config.build_type}",
            f"-DENGINE_ENABLE_NATIVE_CPU={'ON' if config.native_cpu else 'OFF'}",
            f"-DENGINE_ENABLE_OPENMP={'ON' if config.openmp else 'OFF'}",
            "-DENGINE_BUILD_TESTS=OFF",
            "-DENGINE_BUILD_EXAMPLES=OFF",
            "-DENGINE_BUILD_WARMBENCH=OFF",
            f"-DENGINE_ENABLE_CUDA={'ON' if config.backend == 'cuda' else 'OFF'}",
            f"-DENGINE_ENABLE_VULKAN={'ON' if config.backend == 'vulkan' else 'OFF'}",
            f"-DENGINE_ENABLE_METAL={'ON' if config.backend == 'metal' else 'OFF'}",
            f"-DENGINE_ENABLE_CUDA_GRAPHS={'ON' if config.cuda_graphs else 'OFF'}",
        ]
        if config.custom_cmake_args:
            args.extend(shlex.split(config.custom_cmake_args))
        return args

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
        branch = str(branch or "").strip()
        if not branch or "\0" in branch:
            raise ValueError("A source branch is required for sync")
        if not os.path.isdir(os.path.join(source_dir, ".git")):
            raise ValueError(f"Existing source checkout not found: {source_dir}")

        await self._emit(
            progress_manager,
            task_id,
            "sync",
            8,
            f"Fetching origin/{branch}",
        )
        await self._run_streaming(
            ["git", "fetch", "--prune", "origin", branch],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=12,
        )
        self._raise_if_cancelled(task_id)
        await self._emit(
            progress_manager,
            task_id,
            "sync",
            18,
            f"Resetting checkout to origin/{branch}",
        )
        await self._run_streaming(
            ["git", "checkout", "-B", branch, "FETCH_HEAD"],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=22,
        )
        await self._run_streaming(
            ["git", "reset", "--hard", "FETCH_HEAD"],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=25,
        )
        await self._run_streaming(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="sync",
            progress=28,
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
        await self._emit(
            progress_manager,
            task_id,
            "configure",
            30,
            f"Configuring {config.backend} build",
        )
        await self._run_streaming(
            self._cmake_args(source_dir, build_dir, config),
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="configure",
            progress=40,
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
            progress_manager, task_id, "build", 45, "Building audio.cpp"
        )
        await self._run_streaming(
            build_argv,
            cwd=source_dir,
            task_id=task_id,
            progress_manager=progress_manager,
            stage="build",
            progress=70,
        )

        server_binary = self._find_binary(build_dir, "audiocpp_server")
        cli_binary = self._find_binary(build_dir, "audiocpp_cli")
        for binary in (server_binary, cli_binary):
            os.chmod(binary, os.stat(binary).st_mode | 0o111)

        await self._emit(
            progress_manager,
            task_id,
            "validate",
            88,
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
            raise FileExistsError(f"audio.cpp version '{version_name}' already exists")

        source_dir = os.path.join(version_dir, "source")
        build_dir = os.path.join(version_dir, "build")
        async with self._build_lock:
            if task_id:
                register_task_cancel(task_id)
            try:
                await self._emit(
                    progress_manager, task_id, "clone", 2, "Cloning audio.cpp"
                )
                os.makedirs(version_dir, exist_ok=False)
                await self._run_streaming(
                    ["git", "clone", "--recursive", repository_url, source_dir],
                    cwd=version_dir,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    stage="clone",
                    progress=10,
                )
                self._raise_if_cancelled(task_id)
                await self._emit(
                    progress_manager,
                    task_id,
                    "checkout",
                    18,
                    f"Checking out {source_ref}",
                )
                await self._run_streaming(
                    ["git", "checkout", str(source_ref or AUDIO_CPP_DEFAULT_REF)],
                    cwd=source_dir,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    stage="checkout",
                    progress=20,
                )
                await self._run_streaming(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=source_dir,
                    task_id=task_id,
                    progress_manager=progress_manager,
                    stage="checkout",
                    progress=25,
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
                return {
                    "version": version_name,
                    **binaries,
                    "source_path": source_dir,
                    "model_manager_path": os.path.join(
                        source_dir, "tools", "model_manager.py"
                    ),
                    "source_commit": source_commit,
                    "source_ref": source_ref,
                    "source_repo": repository_url,
                    "build_config": asdict(config),
                }
            except BaseException:
                if os.path.isdir(version_dir):
                    robust_rmtree(version_dir)
                raise
            finally:
                if task_id:
                    unregister_task_cancel(task_id)

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
        source_dir = os.path.abspath(str(version_entry.get("source_path") or "").strip())
        if not version_name or not source_dir:
            raise ValueError("audio.cpp source install metadata is incomplete")

        build_root = os.path.realpath(self.builds_dir)
        source_real = os.path.realpath(source_dir)
        if os.path.commonpath([source_real, build_root]) != build_root:
            raise ValueError("Refusing to sync audio.cpp files outside the builds root")

        version_dir = os.path.dirname(source_dir)
        build_dir = os.path.join(version_dir, "build")
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
                return {
                    "version": version_name,
                    **binaries,
                    "source_path": source_dir,
                    "model_manager_path": os.path.join(
                        source_dir, "tools", "model_manager.py"
                    ),
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
        source_path = os.path.abspath(str(version_row.get("source_path") or ""))
        if not source_path:
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
    "AUDIO_CPP_COMPATIBILITY_COMMIT",
    "AUDIO_CPP_REPOSITORY",
    "AudioCppBuildConfig",
    "AudioCppManager",
    "get_audio_cpp_manager",
]

