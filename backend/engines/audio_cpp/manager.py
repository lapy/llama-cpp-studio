"""Build configuration, sync-target resolution and tooling for audio.cpp.

The install/release/source/sync FLOWS live on the definition-driven native host
(``backend/engines/audio_cpp/installer.py``); this module owns the engine's
build configuration (cmake argv/env, ccache/CUDA wiring), the recorded-layout
resolution a sync needs, and the singleton manager/git helpers.
"""

from __future__ import annotations

import asyncio
import os
import shlex
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from backend.git_https import git_argv
from backend.logging_config import get_logger
from backend.paths import studio_data_dir


logger = get_logger(__name__)

AUDIO_CPP_REPOSITORY = "https://github.com/0xShug0/audio.cpp.git"
# Bootstrap fallback only when GitHub is unreachable; runtime tracking uses persisted settings.
AUDIO_CPP_DEFAULT_REF = "main"

# Stage labels for the shared streaming runner's progress messages.
_STAGE_MESSAGES = {
    "clone": "Cloning audio.cpp",
    "checkout": "Checking out audio.cpp",
    "sync": "Syncing audio.cpp",
    "configure": "Configuring audio.cpp",
    "build": "Building audio.cpp",
    "validate": "Validating audio.cpp",
}


@dataclass
class AudioCppBuildConfig:
    backend: str = "cpu"
    build_type: str = "RelWithDebInfo"
    cuda: bool = False
    native_cpu: bool = True
    openmp: bool = True
    cuda_graphs: bool = True
    llamafile: bool = True
    cpu_all_variants: bool = False
    build_tests: bool = False
    build_extended_tests: bool = False
    build_model_tests: bool = False
    build_examples: bool = False
    build_warmbench: bool = False
    deployment_build: bool = False
    native_model_manager: bool = False
    use_system_openssl: bool = False
    build_server_frontends: bool = False
    build_c_api: bool = False
    static_espeak: bool = False
    model_set: str = "full"
    models: str = ""
    jobs: int = 0
    custom_cmake_args: str = ""
    cflags: str = ""
    cxxflags: str = ""

    def normalized(self) -> "AudioCppBuildConfig":
        from backend.engines.audio_cpp.build_options import settings_to_field_kwargs

        raw = asdict(self)
        kwargs = settings_to_field_kwargs(raw)
        for key, value in kwargs.items():
            setattr(self, key, value)
        if self.build_type not in {"Debug", "Release", "RelWithDebInfo", "MinSizeRel"}:
            self.build_type = "RelWithDebInfo"
        self.jobs = max(0, int(self.jobs or 0))
        self.backend = "cuda" if self.cuda else "cpu"
        return self


def _data_root() -> str:
    return studio_data_dir()


def cmake_args_for_host_cuda(args: List[str], sms: Sequence[int]) -> List[str]:
    """Pin nvcc to the host compute capabilities when the build did not.

    A blank list leaves the toolchain default. An explicit
    ``-DCMAKE_CUDA_ARCHITECTURES`` already in ``args`` is kept.
    """
    if not sms or any(str(arg).startswith("-DCMAKE_CUDA_ARCHITECTURES=") for arg in args):
        return list(args)
    flag = "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(str(int(sm)) for sm in sms)
    copied = list(args)
    if len(copied) >= 2 and copied[-2] == "-G":
        copied.insert(-2, flag)
        return copied
    copied.append(flag)
    return copied


def cmake_args_with_ccache_launchers(args: List[str], launchers: Sequence[str]) -> List[str]:
    """Insert ccache ``-D`` flags unless that launcher was already requested."""
    extra: List[str] = []
    for flag in launchers:
        text = str(flag).strip()
        key = text.split("=", 1)[0]
        if not key or any(str(item).startswith(key + "=") for item in args):
            continue
        extra.append(text)
    if not extra:
        return list(args)
    copied = list(args)
    if len(copied) >= 2 and copied[-2] == "-G":
        return copied[:-2] + extra + copied[-2:]
    return copied + extra


async def sync_git_checkout(
    *,
    source_dir: str,
    branch: str,
    run,
    emit,
    raise_if_cancelled=None,
) -> None:
    """Fetch/reset/submodule an existing checkout (shared by manager and host).

    ``run(argv, stage=, progress=)`` streams one command, ``emit(stage,
    progress, message)`` reports a stage transition, and ``raise_if_cancelled``
    is polled between commands. The argv sequence, stage labels and progress
    points are the legacy ones.
    """
    from backend.build_progress import cmake_stage_start

    branch = str(branch or "").strip()
    if not branch or "\0" in branch:
        raise ValueError("A source branch is required for sync")
    if not os.path.isdir(os.path.join(source_dir, ".git")):
        raise ValueError(f"Existing source checkout not found: {source_dir}")

    await emit("sync", cmake_stage_start("sync"), f"Fetching origin/{branch}")
    await run(
        git_argv("fetch", "--prune", "origin", branch),
        stage="sync",
        progress=cmake_stage_start("sync"),
    )
    if raise_if_cancelled is not None:
        raise_if_cancelled()
    await emit("sync", 8, f"Resetting checkout to origin/{branch}")
    await run(["git", "checkout", "-B", branch, "FETCH_HEAD"], stage="sync", progress=8)
    await run(["git", "reset", "--hard", "FETCH_HEAD"], stage="sync", progress=10)
    await run(
        git_argv("submodule", "update", "--init", "--recursive"),
        stage="sync",
        progress=11,
    )


class AudioCppManager:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = os.path.abspath(root_dir or os.path.join(_data_root(), "audio-cpp"))
        self.builds_dir = os.path.join(self.root_dir, "builds")
        self.tools_dir = os.path.join(self.root_dir, "tools")
        self.models_dir = os.path.join(_data_root(), "models", "audio-cpp")
        self.server_configs_dir = os.path.join(_data_root(), "config", "audio-cpp", "servers")
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
        return ["cpu", "cuda"]

    @classmethod
    def validate_build_config(cls, config: AudioCppBuildConfig) -> None:
        config = config.normalized()
        if config.backend not in cls.supported_build_backends():
            raise ValueError(
                f"audio.cpp backend '{config.backend}' is not supported; "
                f"supported backends: {', '.join(cls.supported_build_backends())}"
            )

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
            f"-DENGINE_BUILD_EXTENDED_TESTS={'ON' if config.build_extended_tests else 'OFF'}",
            f"-DENGINE_BUILD_MODEL_TESTS={'ON' if config.build_model_tests else 'OFF'}",
            f"-DENGINE_BUILD_EXAMPLES={'ON' if config.build_examples else 'OFF'}",
            f"-DENGINE_BUILD_WARMBENCH={'ON' if config.build_warmbench else 'OFF'}",
            f"-DAUDIOCPP_BUILD_C_API={'ON' if config.build_c_api else 'OFF'}",
            f"-DAUDIOCPP_STATIC_ESPEAK={'ON' if config.static_espeak else 'OFF'}",
            f"-DENGINE_ENABLE_CUDA={'ON' if config.cuda else 'OFF'}",
            f"-DENGINE_ENABLE_HIP=OFF",
            f"-DENGINE_ENABLE_VULKAN=OFF",
            f"-DENGINE_ENABLE_METAL=OFF",
            f"-DENGINE_ENABLE_CUDA_GRAPHS={'ON' if config.cuda and config.cuda_graphs else 'OFF'}",
            f"-DENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=OFF",
            f"-DAUDIOCPP_DEPLOYMENT_BUILD={'ON' if config.deployment_build else 'OFF'}",
            f"-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER={'ON' if config.native_model_manager else 'OFF'}",
            f"-DAUDIOCPP_USE_SYSTEM_OPENSSL={'ON' if config.use_system_openssl else 'OFF'}",
            f"-DAUDIOCPP_BUILD_SERVER_FRONTENDS={'ON' if config.build_server_frontends else 'OFF'}",
            f"-DAUDIOCPP_MODEL_SET={config.model_set or 'full'}",
        ]
        if (config.model_set or "") == "custom" and str(config.models or "").strip():
            args.append(f"-DAUDIOCPP_MODELS={str(config.models).strip()}")
        if config.custom_cmake_args:
            args.extend(shlex.split(config.custom_cmake_args))
        from backend.engines.build_workspace import origin_cmake_args

        args.extend(origin_cmake_args())
        return prefer_ninja_generator(args)

    async def _prepare_configure(
        self, source_dir: str, build_dir: str, config: AudioCppBuildConfig
    ) -> tuple[List[str], Dict[str, str]]:
        """Configure argv + build environment for one tree (shared with the host)."""
        env = os.environ.copy()
        if config.cflags:
            env["CFLAGS"] = config.cflags
        if config.cxxflags:
            env["CXXFLAGS"] = config.cxxflags
        from backend.engines.build_workspace import ccache_base_dir, ccache_environment

        env.update(
            ccache_environment(
                ccache_base_dir(source_dir),
                launchers=True,
                cuda=bool(config.cuda),
            )
        )
        from backend.build_progress import (
            apply_relocated_cuda_warning_flags,
            split_cmake_cli_warning_flags,
        )

        cmake_args, relocated_w_flags = split_cmake_cli_warning_flags(
            self._cmake_args(source_dir, build_dir, config)
        )
        if config.cuda:
            from backend.cuda_installer import get_cuda_installer
            from backend.engines.audio_cpp.prebuilt import detect_host_target

            # CMake's FindCUDAToolkit does not necessarily search PATH for
            # nvcc. Bind CUDA builds to the Studio-managed toolkit explicitly.
            cuda_env = get_cuda_installer().get_cuda_env()
            cuda_root = str(cuda_env.get("CUDA_HOME") or "").strip()
            if cuda_root:
                env.update(cuda_env)
                nvcc = os.path.join(cuda_root, "bin", "nvcc")
                if os.path.isfile(nvcc):
                    env["CUDACXX"] = nvcc
                for flag in (
                    f"-DCUDAToolkit_ROOT={cuda_root}",
                    f"-DCUDA_TOOLKIT_ROOT_DIR={cuda_root}",
                ):
                    if not any(
                        str(arg).startswith(flag.split("=", 1)[0] + "=") for arg in cmake_args
                    ):
                        cmake_args.append(flag)
                if os.path.isfile(nvcc) and not any(
                    str(arg).startswith("-DCMAKE_CUDA_COMPILER=") for arg in cmake_args
                ):
                    cmake_args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")

            host = await detect_host_target()
            cmake_args = cmake_args_for_host_cuda(cmake_args, host.sms)
        from backend.engines.build_workspace import ccache_launcher_cmake_args

        cmake_args = cmake_args_with_ccache_launchers(
            cmake_args,
            ccache_launcher_cmake_args(cuda=bool(config.cuda)),
        )
        extra_cmake = str(env.get("CMAKE_ARGS") or "").strip()
        if extra_cmake:
            extra_kept, extra_relocated = split_cmake_cli_warning_flags(shlex.split(extra_cmake))
            relocated_w_flags = list(relocated_w_flags) + extra_relocated
            if extra_relocated:
                env["CMAKE_ARGS"] = " ".join(extra_kept)
        if relocated_w_flags:
            env = apply_relocated_cuda_warning_flags(env, relocated_w_flags)
        return cmake_args, env

    @staticmethod
    def _build_argv(build_dir: str, config: AudioCppBuildConfig) -> List[str]:
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
        return build_argv

    def _local_src_dir(self) -> str:
        return os.path.realpath(os.path.join(self.root_dir, "src"))

    def _is_local_checkout(self, version_entry: Optional[Dict[str, Any]], source_dir: str) -> bool:
        install_type = (
            str(
                (version_entry or {}).get("install_type") or (version_entry or {}).get("type") or ""
            )
            .strip()
            .lower()
        )
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
