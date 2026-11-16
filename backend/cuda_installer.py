"""
CUDA Toolkit Installer

Handles downloading and installing CUDA Toolkit on Linux systems.
"""

import asyncio
import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Optional, Tuple
import aiohttp
import aiofiles

from backend import cuda_downloads, cuda_metadata
from backend.cuda_errors import (
    CudaInstallFailed,
    CudaInstallIncomplete,
    CudaOperationBusy,
    CudaUnsupportedPlatform,
    CudaVersionNotInstalled,
    CudaVersionUnsupported,
)
from backend.operations.cancellable import CancellableOperationManager
from backend.logging_config import get_logger
from backend.operations.progress import get_progress_manager
from backend.paths import studio_data_dir

logger = get_logger(__name__)

_installer_instance: Optional["CUDAInstaller"] = None


def get_cuda_installer() -> "CUDAInstaller":
    global _installer_instance
    if _installer_instance is None:
        _installer_instance = CUDAInstaller()
    return _installer_instance


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CUDAInstaller(CancellableOperationManager):
    """Install CUDA Toolkit on Linux systems."""

    MANAGER_NAME = "cuda"

    OPERATION_DESCRIPTIONS = {
        "install": "Install CUDA",
        "uninstall": "Uninstall CUDA",
        "activate": "Activate CUDA",
    }

    SUPPORTED_VERSIONS = cuda_metadata.SUPPORTED_CUDA_VERSIONS
    _MINIMUM_CUDA_VERSION = cuda_metadata.MINIMUM_CUDA_VERSION
    _SUPPORTED_CUDA_MAJORS = cuda_metadata.SUPPORTED_CUDA_MAJORS

    # cuDNN version mappings by CUDA major version
    CUDNN_VERSIONS = {
        "13": "9.5.1",  # cuDNN 9.x for CUDA 13.x
        "12": "9.5.1",  # cuDNN 9.x for CUDA 12.x
        "11": "8.9.7",  # cuDNN 8.x for CUDA 11.x
    }

    # TensorRT version mappings by CUDA major version
    TENSORRT_VERSIONS = {
        "13": "10.7.0",  # TensorRT 10.x for CUDA 13.x
        "12": "10.7.0",  # TensorRT 10.x for CUDA 12.x
        "11": "8.6.1",  # TensorRT 8.x for CUDA 11.x
    }

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        download_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._download_progress: Dict[str, Any] = {}

        data_root = studio_data_dir()

        log_path = log_path or os.path.join(data_root, "logs", "cuda_install.log")
        state_path = state_path or os.path.join(data_root, "config", "cuda_installer.json")
        download_dir = download_dir or os.path.join(data_root, "temp", "cuda_installers")
        self._cuda_install_dir = os.path.join(data_root, "cuda")

        self._log_path = os.path.abspath(log_path)
        self._state_path = os.path.abspath(state_path)
        self._download_dir = os.path.abspath(download_dir)
        self._metadata = cuda_metadata.CudaMetadataSource()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        os.makedirs(self._download_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        os.makedirs(self._cuda_install_dir, exist_ok=True)

    @staticmethod
    def _directory_size(path: str) -> int:
        """Total bytes of the regular files under ``path``.

        Unreadable or vanishing entries count as zero and symlinks are skipped
        (never followed), so status reporting survives a partially removed
        installation or installer cache.
        """
        total = 0
        try:
            for root, dirnames, filenames in os.walk(path, followlinks=False):
                dirnames[:] = [
                    name for name in dirnames if not os.path.islink(os.path.join(root, name))
                ]
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    try:
                        if os.path.islink(file_path):
                            continue
                        total += os.path.getsize(file_path)
                    except OSError:
                        continue
        except OSError:
            return total
        return total

    def _installer_cache_status(self) -> Dict[str, Any]:
        """Summarize the downloaded CUDA runfiles kept in the cache directory."""
        return cuda_downloads.installer_cache_status(self._download_dir)

    def _update_current_symlink(self, install_path: str) -> None:
        """Create or update the /app/data/cuda/current symlink to point to the active CUDA installation."""
        current_symlink = os.path.join(self._cuda_install_dir, "current")
        try:
            # Remove existing symlink if it exists
            if os.path.islink(current_symlink):
                os.remove(current_symlink)
            elif os.path.exists(current_symlink):
                # If it's not a symlink but exists, remove it (shouldn't happen, but be safe)
                os.remove(current_symlink)

            # Create new symlink pointing to the installation
            os.symlink(install_path, current_symlink)
            logger.info(f"Updated CUDA current symlink: {current_symlink} -> {install_path}")
        except OSError as e:
            logger.warning(f"Failed to update CUDA current symlink: {e}")

    def _remove_current_symlink(self) -> None:
        """Remove the current symlink and optionally re-point it to another installed version."""
        current_symlink = os.path.join(self._cuda_install_dir, "current")
        try:
            if os.path.islink(current_symlink) or os.path.exists(current_symlink):
                os.remove(current_symlink)

            # Try to find another installed version to point to
            state = self._load_state()
            installations = state.get("installations", {})

            # Find the most recently installed version that still exists
            latest_version = None
            latest_time = None
            for v, info in installations.items():
                install_path = info.get("path")
                if install_path and os.path.exists(install_path):
                    installed_at = info.get("installed_at", "")
                    if not latest_time or installed_at > latest_time:
                        latest_time = installed_at
                        latest_version = v

            # Re-point to the latest remaining installation
            if latest_version:
                install_path = installations[latest_version].get("path")
                if install_path and os.path.exists(install_path):
                    os.symlink(install_path, current_symlink)
                    logger.info(f"Re-pointed CUDA current symlink to: {install_path}")
        except OSError as e:
            logger.warning(f"Failed to update CUDA current symlink: {e}")

    def _get_platform(self) -> Tuple[str, str]:
        return cuda_metadata.get_platform()

    def _get_ubuntu_version(self) -> str:
        return cuda_metadata.get_ubuntu_version()

    def _get_archive_target_version(self) -> str:
        return cuda_metadata.get_archive_target_version(self._get_ubuntu_version())

    async def _get_repo_packages(self, ubuntu_version: str) -> list:
        return await self._metadata.get_repo_packages(ubuntu_version)

    @classmethod
    def _cuda_version_parts(cls, version: str) -> tuple[int, int, int]:
        return cuda_metadata.cuda_version_parts(version)

    @classmethod
    def _cuda_release_line(cls, version: str) -> str:
        return cuda_metadata.cuda_release_line(version)

    async def get_available_versions(self) -> list[str]:
        return cuda_metadata.resolve_available_versions(
            await self._get_repo_packages(self._get_ubuntu_version()), self.SUPPORTED_VERSIONS
        )

    def _version_key(self, version: str) -> tuple:
        return cuda_metadata.version_key(version)

    def _select_repo_package(
        self,
        packages: list,
        package_name: str,
        version_prefix: Optional[str] = None,
        version_contains: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        return cuda_metadata.select_repo_package(
            packages, package_name, version_prefix, version_contains, key=self._version_key
        )

    def _load_state(self) -> Dict[str, Any]:
        if not os.path.exists(self._state_path):
            return {}
        try:
            with open(self._state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception as exc:
            logger.warning(f"Failed to load CUDA installer state: {exc}")
            return {}

    def _save_state(self, state: Dict[str, Any]) -> None:
        tmp_path = f"{self._state_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, self._state_path)

    def _detect_installed_version(self) -> Optional[str]:
        """Detect installed CUDA version by checking nvcc or state."""
        state = self._load_state()
        installations = state.get("installations", {})

        # Activation records the selected version explicitly.  Prefer that
        # pointer over installation time so selecting an older toolkit remains
        # visible after a refresh/restart.
        active_version = str(state.get("installed_version") or "").strip()
        if active_version:
            active_info = installations.get(active_version, {})
            active_path = active_info.get("path") or state.get("cuda_path")
            if active_path and os.path.exists(active_path):
                return active_version

        # Older state files predate installed_version.  Resolve the current
        # link against their recorded installation paths before falling back
        # to the most recently installed entry.
        current_link = os.path.join(self._cuda_install_dir, "current")
        if os.path.lexists(current_link):
            try:
                current_path = os.path.normcase(os.path.realpath(current_link))
                for installed_version, info in installations.items():
                    install_path = info.get("path")
                    if (
                        install_path
                        and os.path.normcase(os.path.realpath(install_path)) == current_path
                    ):
                        return str(installed_version)
            except (OSError, ValueError):
                pass

        # First check state for installed versions
        if installations:
            # Return the most recently installed version
            latest_version = None
            latest_time = None
            for v, info in installations.items():
                installed_at = info.get("installed_at", "")
                if not latest_time or installed_at > latest_time:
                    latest_time = installed_at
                    latest_version = v
            if latest_version:
                install_path = installations[latest_version].get("path")
                if install_path and os.path.exists(install_path):
                    return latest_version

        # Fallback: try to detect via nvcc command
        try:
            # Get CUDA environment to find nvcc
            cuda_env = self.get_cuda_env()
            env = os.environ.copy()
            env.update(cuda_env)

            nvcc_path = shutil.which("nvcc", path=env.get("PATH", ""))
            if not nvcc_path:
                return None

            result = subprocess.run(
                [nvcc_path, "--version"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env,
            )
            if result.returncode == 0:
                # Parse version from output
                for line in result.stdout.split("\n"):
                    if "release" in line.lower():
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if "release" in part.lower() and i + 1 < len(parts):
                                version_str = parts[i + 1].rstrip(",")
                                # Extract major.minor
                                version_parts = version_str.split(".")
                                if len(version_parts) >= 2:
                                    return f"{version_parts[0]}.{version_parts[1]}"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return None

    async def activate(self, version: str) -> Dict[str, Any]:
        """Select an existing Studio-managed CUDA installation."""
        version = str(version or "").strip()
        if not version:
            raise CudaVersionUnsupported("CUDA version is required")

        async with self._lock:
            if self._operation:
                raise CudaOperationBusy("Another CUDA installer operation is already running")

            state = self._load_state()
            install_info = state.get("installations", {}).get(version)
            if not isinstance(install_info, dict):
                raise CudaVersionNotInstalled(f"CUDA {version} installation not found in state")
            install_path = str(install_info.get("path") or "").strip()
            if not install_path or not os.path.isfile(os.path.join(install_path, "bin", "nvcc")):
                raise CudaInstallIncomplete(f"CUDA {version} installation is missing or incomplete")

            self._update_current_symlink(install_path)
            selected_path = self._get_cuda_path()
            if not selected_path or os.path.normcase(
                os.path.realpath(selected_path)
            ) != os.path.normcase(os.path.realpath(install_path)):
                raise CudaInstallFailed(f"Failed to activate CUDA {version}")

            state["installed_version"] = version
            state["cuda_path"] = install_path
            self._save_state(state)

            cuda_env = self.get_cuda_env(version)
            if cuda_env:
                os.environ.update(cuda_env)

            try:
                from backend.proxy.control import get_proxy_controller

                await get_proxy_controller().restart_proxy()
            except Exception as restart_error:
                logger.warning(
                    "Failed to restart Studio proxy after activating CUDA %s: %s",
                    version,
                    restart_error,
                )

            return {
                "message": f"CUDA {version} activated",
                "version": version,
                "cuda_path": install_path,
            }

    def _get_cuda_path(self, version: Optional[str] = None) -> Optional[str]:
        """Get CUDA installation path."""
        state = self._load_state()
        installations = state.get("installations", {})

        # A caller asking for a specific toolkit must never silently receive
        # the currently active (but different) version.  Python engine builds
        # use this to bind reproducibly to their required Studio CUDA install.
        if version:
            requested = str(version).strip()
            for installed_version, info in installations.items():
                installed = str(installed_version).strip()
                if installed != requested and not installed.startswith(f"{requested}."):
                    continue
                install_path = info.get("path")
                if install_path and os.path.isfile(os.path.join(install_path, "bin", "nvcc")):
                    return install_path

            # Retain disk discovery for installations whose state file was
            # lost, but restrict it to the requested version.
            try:
                if os.path.isdir(self._cuda_install_dir):
                    for item in sorted(os.listdir(self._cuda_install_dir), reverse=True):
                        if item == "current":
                            continue
                        disk_version = item.removeprefix("cuda-")
                        if disk_version != requested and not disk_version.startswith(
                            f"{requested}."
                        ):
                            continue
                        full_path = os.path.join(self._cuda_install_dir, item)
                        if os.path.isfile(os.path.join(full_path, "bin", "nvcc")):
                            return full_path
            except OSError:
                pass
            return None

        # First, check the current symlink (most reliable for active installation)
        current_symlink = os.path.join(self._cuda_install_dir, "current")
        if os.path.islink(current_symlink) or os.path.exists(current_symlink):
            try:
                resolved_path = os.path.realpath(current_symlink)
                if os.path.exists(resolved_path):
                    nvcc_path = os.path.join(resolved_path, "bin", "nvcc")
                    if os.path.exists(nvcc_path):
                        return resolved_path
            except (OSError, ValueError):
                pass

        # Check for latest installed version in state
        if installations:
            # Get the most recently installed version
            latest_version = None
            latest_time = None
            for v, info in installations.items():
                installed_at = info.get("installed_at", "")
                if not latest_time or installed_at > latest_time:
                    latest_time = installed_at
                    latest_version = v

            if latest_version:
                install_path = installations[latest_version].get("path")
                if install_path and os.path.exists(install_path):
                    return install_path

        # Check environment variables (only accept paths under data directory)
        env_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if (
            env_path
            and os.path.exists(env_path)
            and os.path.abspath(env_path).startswith(self._cuda_install_dir)
        ):
            return env_path

        # Scan the data directory for CUDA installs as fallback
        try:
            if os.path.exists(self._cuda_install_dir):
                for item in sorted(os.listdir(self._cuda_install_dir), reverse=True):
                    # Skip the current symlink
                    if item == "current":
                        continue
                    full_path = os.path.join(self._cuda_install_dir, item)
                    if os.path.isdir(full_path):
                        nvcc_path = os.path.join(full_path, "bin", "nvcc")
                        if os.path.exists(nvcc_path):
                            return full_path
        except OSError:
            pass

        return None

    def get_cuda_env(self, version: Optional[str] = None) -> Dict[str, str]:
        """Get environment variables for CUDA installation."""
        cuda_path = self._get_cuda_path(version)
        if not cuda_path:
            return {}

        cuda_bin = os.path.join(cuda_path, "bin")
        cuda_lib = os.path.join(cuda_path, "lib64")

        env = {
            "CUDA_HOME": cuda_path,
            "CUDA_PATH": cuda_path,
        }

        def prepend(name: str, directory: str) -> None:
            if not os.path.isdir(directory):
                return
            current = os.environ.get(name, "")
            parts = [part for part in current.split(os.pathsep) if part]
            env[name] = current if directory in parts else os.pathsep.join([directory, *parts])

        # Add to PATH if bin directory exists
        prepend("PATH", cuda_bin)

        # Add to LD_LIBRARY_PATH if lib64 directory exists
        prepend("LD_LIBRARY_PATH", cuda_lib)

        cuda_include = os.path.join(cuda_path, "include")
        if os.path.isdir(cuda_include):
            env.update(
                {
                    "NCCL_ROOT": cuda_path,
                    "NCCL_HOME": cuda_path,
                    "NCCL_INCLUDE_DIR": cuda_include,
                }
            )
            prepend("CPATH", cuda_include)
        if os.path.isdir(cuda_lib):
            env["NCCL_LIB_DIR"] = cuda_lib
            prepend("LIBRARY_PATH", cuda_lib)
            prepend("CMAKE_LIBRARY_PATH", cuda_lib)
        prepend("CMAKE_PREFIX_PATH", cuda_path)

        # Add TensorRT path if TensorRT is installed
        tensorrt_version = self._detect_tensorrt_version(cuda_path)
        if tensorrt_version:
            env["TENSORRT_PATH"] = cuda_path
            env["TENSORRT_ROOT"] = cuda_path

        return env

    def _get_archive_url(self, version: str) -> str:
        return cuda_metadata.get_archive_url(version, self._get_archive_target_version())

    async def _fetch_download_url(self, version: str) -> str:
        return await self._metadata.fetch_download_url(
            version, self._get_archive_url(version), self._get_archive_target_version()
        )

    async def _broadcast_log_line(self, line: str) -> None:
        try:
            await self._append_task_log(line)
        except Exception as exc:
            logger.debug(f"Failed to broadcast CUDA log line: {exc}")

    async def _broadcast_progress(self, progress: Dict[str, Any]) -> None:
        """Record CUDA progress on the install task."""
        try:
            progress_value = float(progress.get("progress", 0))
            if self._progress_task_id:
                await self._update_progress_task(
                    progress_value,
                    progress.get("message") or "",
                    metadata_update={
                        k: v for k, v in progress.items() if k not in ("progress", "message")
                    },
                )
        except Exception as exc:
            logger.exception(f"Failed to broadcast CUDA progress: {exc}")

    async def _start_operation(
        self, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        description = self.OPERATION_DESCRIPTIONS.get(operation, "Install CUDA")
        return await self._begin_operation(operation, description, metadata)

    def _on_task_error(self, exc: Exception) -> None:
        logger.exception("CUDA installer task error")

    async def _download_installer(self, version: str, url: str, installer_path: str) -> None:
        """Download, verify and log the CUDA runfile via :mod:`backend.cuda_downloads`."""
        await cuda_downloads.download_installer(
            version,
            url,
            installer_path,
            log_path=self._log_path,
            utcnow=_utcnow,
            broadcast_log=self._broadcast_log_line,
            broadcast_progress=self._broadcast_progress,
        )

    def _is_docker_container(self) -> bool:
        """Check if running inside a Docker container."""
        # Check /.dockerenv
        if os.path.exists("/.dockerenv"):
            return True

        # Check /proc/self/cgroup for Docker
        try:
            if os.path.exists("/proc/self/cgroup"):
                with open("/proc/self/cgroup", "r") as f:
                    content = f.read()
                    if "docker" in content or "containerd" in content:
                        return True
        except OSError:
            pass

        return False

    async def _install_linux(
        self,
        installer_path: str,
        version: str,
        install_cudnn: bool = False,
        install_tensorrt: bool = False,
    ) -> str:
        """
        Install CUDA on Linux using runfile installer.

        Uses optimized installer options for custom location installation:
        - Silent installation with EULA acceptance
        - Toolkit-only installation (no driver)
        - Override installation checks for custom paths
        - Skip OpenGL libraries (not needed in Docker/headless environments)
        - Skip man pages to reduce installation size

        Args:
            installer_path: Path to the CUDA installer runfile
            version: CUDA version being installed
            install_cudnn: Whether to install cuDNN
            install_tensorrt: Whether to install TensorRT
        """
        await self._broadcast_log_line("Starting CUDA installation on Linux...")
        await self._broadcast_progress(
            {
                "stage": "install",
                "progress": 0,
                "message": "Installing CUDA Toolkit...",
            }
        )

        # Verify installer file exists and is not corrupted
        if not os.path.exists(installer_path):
            raise CudaInstallFailed(f"Installer file not found: {installer_path}")

        file_size = os.path.getsize(installer_path)
        if file_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious for CUDA installers
            raise CudaInstallFailed(
                f"Installer file appears to be corrupted or incomplete: {installer_path} "
                f"(size: {file_size / (1024 * 1024):.1f} MB, expected > 100 MB)"
            )

        # Verify the file starts with a shell script header (CUDA .run files are self-extracting)
        try:
            with open(installer_path, "rb") as f:
                header = f.read(100)
                if not header.startswith(b"#!/"):
                    raise CudaInstallFailed(
                        f"Installer file does not appear to be a valid shell script: {installer_path}"
                    )
        except CudaInstallFailed:
            raise
        except Exception as e:
            raise CudaInstallFailed(
                f"Failed to verify installer file: {installer_path}, error: {e}"
            ) from e

        await self._broadcast_log_line(
            f"Verifying installer file: {installer_path} ({file_size / (1024 * 1024):.1f} MB)"
        )

        # Make installer executable
        os.chmod(installer_path, 0o755)

        # Always install to the data directory for persistence
        install_path = os.path.join(self._cuda_install_dir, f"cuda-{version}")
        await self._broadcast_log_line(f"Installing to data directory: {install_path}")
        os.makedirs(install_path, exist_ok=True)

        # Build installer arguments with optimized options for custom location installation
        #
        # Selected options based on NVIDIA CUDA installer documentation:
        # - --silent: Required for silent installation, implies EULA acceptance
        # - --toolkit: Install toolkit only (not driver) - required for non-root installations
        # - --override: Override compiler, third-party library, and toolkit detection checks
        #   (essential for custom installation paths)
        # - --toolkitpath: Install to custom data directory path
        # - --no-opengl-libs: Skip OpenGL libraries (not needed in Docker/headless environments)
        # - --no-man-page: Skip man pages to reduce installation size
        #
        install_args = [
            "bash",
            installer_path,
            "--silent",  # Silent installation with EULA acceptance
            "--toolkit",  # Install toolkit only (not driver)
            "--override",  # Override installation checks for custom paths
            f"--toolkitpath={install_path}",  # Install to custom data directory
            "--no-opengl-libs",  # Skip OpenGL libraries (not needed in Docker)
            "--no-man-page",  # Skip man pages to reduce size
        ]

        # Skip 'bash' and installer_path
        await self._broadcast_log_line(f"Installer arguments: {' '.join(install_args[2:])}")

        # Set up environment to prevent /dev/tty access issues in Docker
        env = os.environ.copy()
        env["DEBIAN_FRONTEND"] = "noninteractive"
        # Disable interactive prompts
        env["PERL_BADLANG"] = "0"
        # Ensure we're in a non-interactive environment
        env["TERM"] = "dumb"
        # Prevent installer from trying to access /dev/tty
        env["CI"] = "true"  # Indicate we're in a CI/non-interactive environment

        process = await asyncio.create_subprocess_exec(
            *install_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,  # Redirect stdin to prevent /dev/tty access
            env=env,
        )
        self._active_process = process

        # Collect output for error analysis
        output_lines = []

        async def _stream_output():
            if process.stdout is None:
                return
            with open(self._log_path, "a", encoding="utf-8", buffering=1) as log_file:
                while True:
                    chunk = await process.stdout.readline()
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    output_lines.append(text)
                    log_file.write(text)
                    await self._broadcast_log_line(text.rstrip("\n"))

        await asyncio.gather(process.wait(), _stream_output())
        self._clear_active_process()

        if process.returncode != 0:
            # Check for specific error patterns
            output_text = "".join(output_lines)

            # Check for /dev/tty errors
            if "/dev/tty" in output_text.lower() or "cannot create /dev/tty" in output_text.lower():
                error_msg = (
                    f"CUDA installer failed due to /dev/tty access issue (common in Docker). "
                    f"This may indicate the installer file is corrupted or the environment is not properly configured. "
                    f"Exit code: {process.returncode}. "
                    f"Please check the installation logs for details. "
                    f"If the file appears corrupted, try deleting it and re-downloading."
                )
            # Check for gzip/corruption errors
            elif "gzip" in output_text.lower() and (
                "unexpected end" in output_text.lower() or "corrupt" in output_text.lower()
            ):
                error_msg = (
                    f"CUDA installer file appears to be corrupted (gzip error detected). "
                    f"Please delete the installer file at {installer_path} and try again. "
                    f"Exit code: {process.returncode}."
                )
            else:
                error_msg = (
                    f"CUDA installer exited with code {process.returncode}. "
                    "Please check the installation logs for details."
                )

            raise CudaInstallFailed(error_msg)

        # Verify installation and set up environment
        cuda_home = install_path
        cuda_bin = os.path.join(cuda_home, "bin")
        cuda_lib = os.path.join(cuda_home, "lib64")

        # Verify key directories exist
        if not os.path.exists(cuda_bin) or not os.path.exists(cuda_lib):
            raise RuntimeError(
                f"CUDA installation completed but expected directories not found. "
                f"Expected: {cuda_bin}, {cuda_lib}"
            )

        await self._broadcast_log_line(f"CUDA installed successfully to: {install_path}")
        await self._broadcast_log_line(f"CUDA_HOME={cuda_home}")
        await self._broadcast_log_line(f"Adding to PATH: {cuda_bin}")
        await self._broadcast_log_line(f"Adding to LD_LIBRARY_PATH: {cuda_lib}")

        # Install NCCL (required for multi-GPU and llama.cpp CUDA builds)
        await self._install_nccl_linux(version, install_path)

        # Install nvidia-smi (required for GPU monitoring)
        await self._install_nvidia_smi_linux(install_path)

        # Install cuDNN if requested
        if install_cudnn:
            await self._install_cudnn_linux(version, install_path)

        # Install TensorRT if requested
        if install_tensorrt:
            await self._install_tensorrt_linux(version, install_path)

        # Save installation path to state
        state = self._load_state()
        if "installations" not in state:
            state["installations"] = {}
        state["installations"][version] = {
            "path": install_path,
            "installed_at": _utcnow(),
            "is_system_install": False,
            "cudnn_installed": install_cudnn,
            "tensorrt_installed": install_tensorrt,
        }
        self._save_state(state)

        # Update the current symlink to point to this installation
        self._update_current_symlink(install_path)
        await self._broadcast_log_line(
            f"Updated CUDA current symlink: /app/data/cuda/current -> {install_path}"
        )

        components = ["CUDA", "NCCL", "nvidia-smi"]
        if install_cudnn:
            components.append("cuDNN")
        if install_tensorrt:
            components.append("TensorRT")

        await self._broadcast_progress(
            {
                "stage": "install",
                "progress": 100,
                "message": f"{', '.join(components)} installation completed",
            }
        )

        return install_path

    async def _install_nccl_linux(self, cuda_version: str, cuda_path: str) -> None:
        """Install NCCL library for multi-GPU support."""
        await self._broadcast_log_line(
            "Installing NCCL (NVIDIA Collective Communications Library)..."
        )
        await self._broadcast_progress(
            {
                "stage": "nccl",
                "progress": 0,
                "message": "Installing NCCL...",
            }
        )

        ubuntu_version = self._get_ubuntu_version()

        # Download NCCL from NVIDIA's repo package index
        await self._broadcast_log_line("Attempting manual NCCL installation...")

        try:
            cuda_major = cuda_version.split(".")[0]
            packages = await self._get_repo_packages(ubuntu_version)
            nccl_pkg = self._select_repo_package(
                packages,
                "libnccl2",
                version_prefix="2.",
                version_contains=f"+cuda{cuda_major}",
            )
            nccl_dev_pkg = self._select_repo_package(
                packages,
                "libnccl-dev",
                version_prefix="2.",
                version_contains=f"+cuda{cuda_major}",
            )

            if not nccl_pkg or not nccl_dev_pkg:
                await self._broadcast_log_line(
                    "NCCL packages not found in repository, skipping NCCL installation"
                )
                await self._broadcast_progress(
                    {
                        "stage": "nccl",
                        "progress": 100,
                        "message": "NCCL installation skipped (optional)",
                    }
                )
                return

            base_url = (
                f"https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64/"
            )
            nccl_url = base_url + nccl_pkg.get("Filename", "").lstrip("./")
            nccl_dev_url = base_url + nccl_dev_pkg.get("Filename", "").lstrip("./")

            nccl_path = os.path.join(self._download_dir, "libnccl2.deb")
            nccl_dev_path = os.path.join(self._download_dir, "libnccl-dev.deb")

            await self._broadcast_progress(
                {
                    "stage": "nccl",
                    "progress": 25,
                    "message": "Downloading NCCL packages...",
                }
            )

            # Download NCCL packages
            async with aiohttp.ClientSession() as session:
                for url, path, name in [
                    (nccl_url, nccl_path, "libnccl2"),
                    (nccl_dev_url, nccl_dev_path, "libnccl-dev"),
                ]:
                    try:
                        await self._broadcast_log_line(f"Downloading {name}...")
                        async with session.get(url) as response:
                            if response.status == 200:
                                async with aiofiles.open(path, "wb") as f:
                                    await f.write(await response.read())
                                await self._broadcast_log_line(f"Downloaded {name}")
                            else:
                                await self._broadcast_log_line(
                                    f"Failed to download {name}: HTTP {response.status}"
                                )
                                # Try alternative URL with different NCCL version
                                continue
                    except Exception as download_err:
                        await self._broadcast_log_line(f"Download error for {name}: {download_err}")
                        continue

            await self._broadcast_progress(
                {
                    "stage": "nccl",
                    "progress": 50,
                    "message": "Installing NCCL packages...",
                }
            )

            if os.path.exists(nccl_path):
                await self._broadcast_log_line("Extracting NCCL to CUDA directory...")

                # Extract .deb file (it's an ar archive containing data.tar)
                extract_dir = os.path.join(self._download_dir, "nccl_extract")
                os.makedirs(extract_dir, exist_ok=True)

                for deb_path in [nccl_path, nccl_dev_path]:
                    if os.path.exists(deb_path):
                        # Extract using ar and tar
                        extract_process = await asyncio.create_subprocess_exec(
                            "bash",
                            "-c",
                            f"cd {extract_dir} && ar x {deb_path} && tar xf data.tar.* 2>/dev/null || tar xf data.tar 2>/dev/null",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        await extract_process.wait()

                # Copy NCCL files to CUDA installation
                nccl_lib_src = os.path.join(extract_dir, "usr", "lib", "x86_64-linux-gnu")
                nccl_include_src = os.path.join(extract_dir, "usr", "include")

                cuda_lib_dst = os.path.join(cuda_path, "lib64")
                cuda_include_dst = os.path.join(cuda_path, "include")

                if os.path.exists(nccl_lib_src):
                    # First pass: collect files and symlinks, copy actual files first
                    files_to_copy = []
                    symlinks_to_create = []

                    for f in os.listdir(nccl_lib_src):
                        if "nccl" in f.lower():
                            src = os.path.join(nccl_lib_src, f)
                            dst = os.path.join(cuda_lib_dst, f)

                            if os.path.islink(src):
                                # Resolve symlink to find actual target
                                link_target = os.readlink(src)
                                # If relative symlink, resolve relative to source directory
                                if not os.path.isabs(link_target):
                                    link_target = os.path.normpath(
                                        os.path.join(os.path.dirname(src), link_target)
                                    )
                                # Find the actual target file name
                                actual_target = os.path.basename(link_target)
                                symlinks_to_create.append((f, actual_target, dst))
                            else:
                                files_to_copy.append((f, src, dst))

                    # Copy all actual files first
                    for f, src, dst in files_to_copy:
                        try:
                            shutil.copy2(src, dst)
                            await self._broadcast_log_line(f"Copied {f} to CUDA lib directory")
                        except Exception as copy_err:
                            await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                    # Then create symlinks pointing to the copied files
                    for link_name, target_name, dst in symlinks_to_create:
                        try:
                            if os.path.exists(dst):
                                os.remove(dst)
                            # Create symlink pointing to target in same directory
                            os.symlink(target_name, dst)
                            await self._broadcast_log_line(
                                f"Created symlink {link_name} -> {target_name} in CUDA lib directory"
                            )
                        except Exception as link_err:
                            await self._broadcast_log_line(
                                f"Failed to create symlink {link_name}: {link_err}"
                            )

                if os.path.exists(nccl_include_src):
                    for f in os.listdir(nccl_include_src):
                        if "nccl" in f.lower():
                            src = os.path.join(nccl_include_src, f)
                            dst = os.path.join(cuda_include_dst, f)
                            try:
                                if os.path.isdir(src):
                                    # Handle directories by copying recursively
                                    if os.path.exists(dst):
                                        shutil.rmtree(dst)
                                    shutil.copytree(src, dst)
                                    await self._broadcast_log_line(
                                        f"Copied directory {f} to CUDA include directory"
                                    )
                                else:
                                    # Handle regular files
                                    shutil.copy2(src, dst)
                                    await self._broadcast_log_line(
                                        f"Copied {f} to CUDA include directory"
                                    )
                            except Exception as copy_err:
                                await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                # Cleanup temporary extract directory only (keep .deb files)
                shutil.rmtree(extract_dir, ignore_errors=True)
                # Keep .deb files for future use
                logger.info(f"NCCL packages kept at: {nccl_path}, {nccl_dev_path}")

                await self._broadcast_log_line("NCCL extracted to CUDA directory")
                await self._broadcast_progress(
                    {
                        "stage": "nccl",
                        "progress": 100,
                        "message": "NCCL installed successfully",
                    }
                )
            else:
                await self._broadcast_log_line(
                    "NCCL packages not available, skipping NCCL installation"
                )
                await self._broadcast_log_line(
                    "Note: NCCL is optional but recommended for multi-GPU builds"
                )
                await self._broadcast_progress(
                    {
                        "stage": "nccl",
                        "progress": 100,
                        "message": "NCCL installation skipped (optional)",
                    }
                )

        except Exception as e:
            await self._broadcast_log_line(f"NCCL installation error: {e}")
            await self._broadcast_log_line(
                "Note: NCCL is optional. The build will continue without multi-GPU support."
            )
            await self._broadcast_progress(
                {
                    "stage": "nccl",
                    "progress": 100,
                    "message": "NCCL installation skipped (optional)",
                }
            )

    async def _install_nvidia_smi_linux(self, cuda_path: str) -> None:
        """Install nvidia-smi binary for GPU monitoring."""
        await self._broadcast_log_line(
            "Installing nvidia-smi (NVIDIA System Management Interface)..."
        )
        await self._broadcast_progress(
            {
                "stage": "nvidia-smi",
                "progress": 0,
                "message": "Installing nvidia-smi...",
            }
        )

        # Check if nvidia-smi already exists in CUDA installation
        cuda_bin = os.path.join(cuda_path, "bin")
        nvidia_smi_dst = os.path.join(cuda_bin, "nvidia-smi")
        if os.path.exists(nvidia_smi_dst):
            await self._broadcast_log_line(
                "nvidia-smi already exists in CUDA installation, skipping"
            )
            await self._broadcast_progress(
                {
                    "stage": "nvidia-smi",
                    "progress": 100,
                    "message": "nvidia-smi already installed",
                }
            )
            return

        ubuntu_version = self._get_ubuntu_version()

        try:
            # Try to find nvidia-utils package which contains nvidia-smi
            packages = await self._get_repo_packages(ubuntu_version)
            nvidia_utils_pkg = None

            # Try multiple package name patterns
            for pkg_name in ["nvidia-utils", "nvidia-driver-utils", "nvidia-utils-"]:
                nvidia_utils_pkg = self._select_repo_package(
                    packages,
                    pkg_name,
                )
                if nvidia_utils_pkg:
                    break

            if not nvidia_utils_pkg:
                await self._broadcast_log_line(
                    "nvidia-utils package not found in repository, skipping nvidia-smi installation"
                )
                await self._broadcast_log_line(
                    "Note: nvidia-smi will not be available. GPU monitoring may be limited."
                )
                await self._broadcast_progress(
                    {
                        "stage": "nvidia-smi",
                        "progress": 100,
                        "message": "nvidia-smi installation skipped (package not available)",
                    }
                )
                return

            base_url = (
                f"https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64/"
            )
            nvidia_utils_url = base_url + nvidia_utils_pkg.get("Filename", "").lstrip("./")

            nvidia_utils_path = os.path.join(self._download_dir, "nvidia-utils.deb")

            await self._broadcast_progress(
                {
                    "stage": "nvidia-smi",
                    "progress": 25,
                    "message": "Downloading nvidia-utils package...",
                }
            )

            # Download nvidia-utils package
            async with aiohttp.ClientSession() as session:
                try:
                    await self._broadcast_log_line("Downloading nvidia-utils...")
                    async with session.get(nvidia_utils_url) as response:
                        if response.status == 200:
                            async with aiofiles.open(nvidia_utils_path, "wb") as f:
                                await f.write(await response.read())
                            await self._broadcast_log_line("Downloaded nvidia-utils")
                        else:
                            await self._broadcast_log_line(
                                f"Failed to download nvidia-utils: HTTP {response.status}"
                            )
                            raise RuntimeError(
                                f"Failed to download nvidia-utils: HTTP {response.status}"
                            )
                except Exception as download_err:
                    await self._broadcast_log_line(
                        f"Download error for nvidia-utils: {download_err}"
                    )
                    raise

            await self._broadcast_progress(
                {
                    "stage": "nvidia-smi",
                    "progress": 50,
                    "message": "Extracting nvidia-smi...",
                }
            )

            if os.path.exists(nvidia_utils_path):
                await self._broadcast_log_line("Extracting nvidia-smi to CUDA directory...")

                # Extract .deb file
                extract_dir = os.path.join(self._download_dir, "nvidia_utils_extract")
                os.makedirs(extract_dir, exist_ok=True)

                # Extract using ar and tar
                extract_process = await asyncio.create_subprocess_exec(
                    "bash",
                    "-c",
                    f"cd {extract_dir} && ar x {nvidia_utils_path} && tar xf data.tar.* 2>/dev/null || tar xf data.tar 2>/dev/null",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                )
                await extract_process.wait()

                # Copy nvidia-smi binary to CUDA installation
                nvidia_smi_src = os.path.join(extract_dir, "usr", "bin", "nvidia-smi")
                cuda_bin_dst = os.path.join(cuda_path, "bin")
                nvidia_smi_dst = os.path.join(cuda_bin_dst, "nvidia-smi")

                if os.path.exists(nvidia_smi_src):
                    os.makedirs(cuda_bin_dst, exist_ok=True)
                    try:
                        shutil.copy2(nvidia_smi_src, nvidia_smi_dst)
                        os.chmod(nvidia_smi_dst, 0o755)
                        await self._broadcast_log_line("Copied nvidia-smi to CUDA bin directory")
                        await self._broadcast_progress(
                            {
                                "stage": "nvidia-smi",
                                "progress": 100,
                                "message": "nvidia-smi installed successfully",
                            }
                        )
                    except Exception as copy_err:
                        await self._broadcast_log_line(f"Failed to copy nvidia-smi: {copy_err}")
                        raise
                else:
                    await self._broadcast_log_line("nvidia-smi not found in extracted package")
                    await self._broadcast_progress(
                        {
                            "stage": "nvidia-smi",
                            "progress": 100,
                            "message": "nvidia-smi installation skipped (not in package)",
                        }
                    )

                # Cleanup temporary extract directory only (keep .deb file)
                shutil.rmtree(extract_dir, ignore_errors=True)
                # Keep .deb file for future use
                if os.path.exists(nvidia_utils_path):
                    logger.info(f"nvidia-utils package kept at: {nvidia_utils_path}")

            else:
                await self._broadcast_log_line(
                    "nvidia-utils package not available, skipping nvidia-smi installation"
                )
                await self._broadcast_progress(
                    {
                        "stage": "nvidia-smi",
                        "progress": 100,
                        "message": "nvidia-smi installation skipped (package not available)",
                    }
                )

        except Exception as e:
            await self._broadcast_log_line(f"nvidia-smi installation error: {e}")
            await self._broadcast_log_line(
                "Note: nvidia-smi installation failed. GPU monitoring may be limited."
            )
            await self._broadcast_progress(
                {
                    "stage": "nvidia-smi",
                    "progress": 100,
                    "message": "nvidia-smi installation skipped (error occurred)",
                }
            )

    async def _install_cudnn_linux(self, cuda_version: str, cuda_path: str) -> None:
        """Install cuDNN library for deep learning primitives."""
        await self._broadcast_log_line("Installing cuDNN (CUDA Deep Neural Network library)...")
        await self._broadcast_progress(
            {
                "stage": "cudnn",
                "progress": 0,
                "message": "Installing cuDNN...",
            }
        )

        try:
            # Determine CUDA major version for cuDNN compatibility
            cuda_major = cuda_version.split(".")[0]
            cudnn_version = self.CUDNN_VERSIONS.get(cuda_major)

            if not cudnn_version:
                await self._broadcast_log_line(
                    f"cuDNN version not available for CUDA {cuda_version}, skipping"
                )
                await self._broadcast_progress(
                    {
                        "stage": "cudnn",
                        "progress": 100,
                        "message": "cuDNN installation skipped (version not available)",
                    }
                )
                return

            ubuntu_version = self._get_ubuntu_version()

            # cuDNN package names vary by CUDA version
            # For CUDA 12.x: libcudnn9-cuda-12, libcudnn9-dev-cuda-12
            # For CUDA 11.x: libcudnn8-cuda-11, libcudnn8-dev-cuda-11
            if cuda_major == "12" or cuda_major == "13":
                cudnn_pkg = "libcudnn9"
                cudnn_cuda_suffix = f"cuda-{cuda_major}"
            else:
                cudnn_pkg = "libcudnn8"
                cudnn_cuda_suffix = f"cuda-{cuda_major}"

            # Manual cuDNN installation
            await self._broadcast_log_line("Installing cuDNN packages...")

            cudnn_package_name = f"{cudnn_pkg}-{cudnn_cuda_suffix}"
            cudnn_dev_package_name = f"{cudnn_pkg}-dev-{cudnn_cuda_suffix}"
            packages = await self._get_repo_packages(ubuntu_version)
            cudnn_pkg_entry = self._select_repo_package(
                packages, cudnn_package_name, version_prefix=cudnn_version
            )
            cudnn_dev_pkg_entry = self._select_repo_package(
                packages, cudnn_dev_package_name, version_prefix=cudnn_version
            )

            if not cudnn_pkg_entry or not cudnn_dev_pkg_entry:
                await self._broadcast_log_line(
                    "cuDNN packages not found in repository, skipping cuDNN installation"
                )
                await self._broadcast_progress(
                    {
                        "stage": "cudnn",
                        "progress": 100,
                        "message": "cuDNN installation skipped (optional)",
                    }
                )
                return

            base_url = (
                f"https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64/"
            )
            cudnn_url = base_url + cudnn_pkg_entry.get("Filename", "").lstrip("./")
            cudnn_dev_url = base_url + cudnn_dev_pkg_entry.get("Filename", "").lstrip("./")

            cudnn_path = os.path.join(self._download_dir, f"{cudnn_pkg}.deb")
            cudnn_dev_path = os.path.join(self._download_dir, f"{cudnn_pkg}-dev.deb")

            await self._broadcast_progress(
                {
                    "stage": "cudnn",
                    "progress": 25,
                    "message": "Downloading cuDNN packages...",
                }
            )

            # Download cuDNN packages
            async with aiohttp.ClientSession() as session:
                for url, path, name in [
                    (cudnn_url, cudnn_path, cudnn_pkg),
                    (cudnn_dev_url, cudnn_dev_path, f"{cudnn_pkg}-dev"),
                ]:
                    try:
                        await self._broadcast_log_line(f"Downloading {name}...")
                        async with session.get(url) as response:
                            if response.status == 200:
                                async with aiofiles.open(path, "wb") as f:
                                    await f.write(await response.read())
                                await self._broadcast_log_line(f"Downloaded {name}")
                            else:
                                await self._broadcast_log_line(
                                    f"Failed to download {name}: HTTP {response.status}"
                                )
                                # Try alternative URL pattern
                                continue
                    except Exception as download_err:
                        await self._broadcast_log_line(f"Download error for {name}: {download_err}")
                        continue

            await self._broadcast_progress(
                {
                    "stage": "cudnn",
                    "progress": 50,
                    "message": "Installing cuDNN packages...",
                }
            )

            if os.path.exists(cudnn_path):
                await self._broadcast_log_line("Extracting cuDNN to CUDA directory...")

                # Extract .deb file
                extract_dir = os.path.join(self._download_dir, "cudnn_extract")
                os.makedirs(extract_dir, exist_ok=True)

                for deb_path in [cudnn_path, cudnn_dev_path]:
                    if os.path.exists(deb_path):
                        # Extract using ar and tar
                        extract_process = await asyncio.create_subprocess_exec(
                            "bash",
                            "-c",
                            f"cd {extract_dir} && ar x {deb_path} && tar xf data.tar.* 2>/dev/null || tar xf data.tar 2>/dev/null",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        await extract_process.wait()

                # Copy cuDNN files to CUDA installation
                cudnn_lib_src = os.path.join(extract_dir, "usr", "lib", "x86_64-linux-gnu")
                cudnn_include_src = os.path.join(extract_dir, "usr", "include")

                cuda_lib_dst = os.path.join(cuda_path, "lib64")
                cuda_include_dst = os.path.join(cuda_path, "include")

                if os.path.exists(cudnn_lib_src):
                    for f in os.listdir(cudnn_lib_src):
                        if "cudnn" in f.lower():
                            src = os.path.join(cudnn_lib_src, f)
                            dst = os.path.join(cuda_lib_dst, f)
                            try:
                                if os.path.islink(src):
                                    linkto = os.readlink(src)
                                    if os.path.exists(dst):
                                        os.remove(dst)
                                    os.symlink(linkto, dst)
                                else:
                                    shutil.copy2(src, dst)
                                await self._broadcast_log_line(f"Copied {f} to CUDA lib directory")
                            except Exception as copy_err:
                                await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                if os.path.exists(cudnn_include_src):
                    for f in os.listdir(cudnn_include_src):
                        if "cudnn" in f.lower():
                            src = os.path.join(cudnn_include_src, f)
                            dst = os.path.join(cuda_include_dst, f)
                            try:
                                shutil.copy2(src, dst)
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA include directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                # Cleanup temporary extract directory only (keep .deb files)
                shutil.rmtree(extract_dir, ignore_errors=True)
                # Keep .deb files for future use
                logger.info(f"cuDNN packages kept at: {cudnn_path}, {cudnn_dev_path}")

                await self._broadcast_log_line("cuDNN extracted to CUDA directory")
                await self._broadcast_progress(
                    {
                        "stage": "cudnn",
                        "progress": 100,
                        "message": "cuDNN installed successfully",
                    }
                )
            else:
                await self._broadcast_log_line(
                    "cuDNN packages not available, skipping cuDNN installation"
                )
                await self._broadcast_progress(
                    {
                        "stage": "cudnn",
                        "progress": 100,
                        "message": "cuDNN installation skipped (optional)",
                    }
                )

        except Exception as e:
            await self._broadcast_log_line(f"cuDNN installation error: {e}")
            await self._broadcast_log_line(
                "Note: cuDNN is optional. The build will continue without cuDNN support."
            )
            await self._broadcast_progress(
                {
                    "stage": "cudnn",
                    "progress": 100,
                    "message": "cuDNN installation skipped (optional)",
                }
            )

    async def _install_tensorrt_linux(self, cuda_version: str, cuda_path: str) -> None:
        """Install TensorRT library for inference optimization."""
        await self._broadcast_log_line("Installing TensorRT (NVIDIA TensorRT inference library)...")
        await self._broadcast_progress(
            {
                "stage": "tensorrt",
                "progress": 0,
                "message": "Installing TensorRT...",
            }
        )

        try:
            # Determine CUDA major version for TensorRT compatibility
            cuda_major = cuda_version.split(".")[0]
            tensorrt_version = self.TENSORRT_VERSIONS.get(cuda_major)

            if not tensorrt_version:
                await self._broadcast_log_line(
                    f"TensorRT version not available for CUDA {cuda_version}, skipping"
                )
                await self._broadcast_progress(
                    {
                        "stage": "tensorrt",
                        "progress": 100,
                        "message": "TensorRT installation skipped (version not available)",
                    }
                )
                return

            ubuntu_version = self._get_ubuntu_version()

            # TensorRT package names
            # For CUDA 12.x/13.x: libnvinfer10, libnvinfer-dev, libnvinfer-plugin10, libnvinfer-plugin-dev
            # For CUDA 11.x: libnvinfer8, libnvinfer-dev, libnvinfer-plugin8, libnvinfer-plugin-dev
            if cuda_major == "12" or cuda_major == "13":
                tensorrt_pkg = "libnvinfer10"
                tensorrt_plugin_pkg = "libnvinfer-plugin10"
            else:
                tensorrt_pkg = "libnvinfer8"
                tensorrt_plugin_pkg = "libnvinfer-plugin8"

            # Manual TensorRT installation
            await self._broadcast_log_line("Installing TensorRT packages...")

            packages = await self._get_repo_packages(ubuntu_version)
            tensorrt_pkg_entry = self._select_repo_package(
                packages, tensorrt_pkg, version_prefix=tensorrt_version
            )
            tensorrt_dev_pkg_entry = self._select_repo_package(
                packages, f"{tensorrt_pkg}-dev", version_prefix=tensorrt_version
            )
            tensorrt_plugin_entry = self._select_repo_package(
                packages, tensorrt_plugin_pkg, version_prefix=tensorrt_version
            )
            tensorrt_plugin_dev_entry = self._select_repo_package(
                packages, f"{tensorrt_plugin_pkg}-dev", version_prefix=tensorrt_version
            )

            if not all(
                [
                    tensorrt_pkg_entry,
                    tensorrt_dev_pkg_entry,
                    tensorrt_plugin_entry,
                    tensorrt_plugin_dev_entry,
                ]
            ):
                await self._broadcast_log_line(
                    "TensorRT packages not found in repository, skipping TensorRT installation"
                )
                await self._broadcast_progress(
                    {
                        "stage": "tensorrt",
                        "progress": 100,
                        "message": "TensorRT installation skipped (optional)",
                    }
                )
                return

            base_url = (
                f"https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64/"
            )
            tensorrt_url = base_url + tensorrt_pkg_entry.get("Filename", "").lstrip("./")
            tensorrt_dev_url = base_url + tensorrt_dev_pkg_entry.get("Filename", "").lstrip("./")
            tensorrt_plugin_url = base_url + tensorrt_plugin_entry.get("Filename", "").lstrip("./")
            tensorrt_plugin_dev_url = base_url + tensorrt_plugin_dev_entry.get(
                "Filename", ""
            ).lstrip("./")

            tensorrt_path = os.path.join(self._download_dir, f"{tensorrt_pkg}.deb")
            tensorrt_dev_path = os.path.join(self._download_dir, f"{tensorrt_pkg}-dev.deb")
            tensorrt_plugin_path = os.path.join(self._download_dir, f"{tensorrt_plugin_pkg}.deb")
            tensorrt_plugin_dev_path = os.path.join(
                self._download_dir, f"{tensorrt_plugin_pkg}-dev.deb"
            )

            await self._broadcast_progress(
                {
                    "stage": "tensorrt",
                    "progress": 25,
                    "message": "Downloading TensorRT packages...",
                }
            )

            # Download TensorRT packages
            async with aiohttp.ClientSession() as session:
                for url, path, name in [
                    (tensorrt_url, tensorrt_path, tensorrt_pkg),
                    (tensorrt_dev_url, tensorrt_dev_path, f"{tensorrt_pkg}-dev"),
                    (tensorrt_plugin_url, tensorrt_plugin_path, tensorrt_plugin_pkg),
                    (
                        tensorrt_plugin_dev_url,
                        tensorrt_plugin_dev_path,
                        f"{tensorrt_plugin_pkg}-dev",
                    ),
                ]:
                    try:
                        await self._broadcast_log_line(f"Downloading {name}...")
                        async with session.get(url) as response:
                            if response.status == 200:
                                async with aiofiles.open(path, "wb") as f:
                                    await f.write(await response.read())
                                await self._broadcast_log_line(f"Downloaded {name}")
                            else:
                                await self._broadcast_log_line(
                                    f"Failed to download {name}: HTTP {response.status}"
                                )
                                continue
                    except Exception as download_err:
                        await self._broadcast_log_line(f"Download error for {name}: {download_err}")
                        continue

            await self._broadcast_progress(
                {
                    "stage": "tensorrt",
                    "progress": 50,
                    "message": "Installing TensorRT packages...",
                }
            )

            if os.path.exists(tensorrt_path):
                await self._broadcast_log_line("Extracting TensorRT to CUDA directory...")

                # Extract .deb file
                extract_dir = os.path.join(self._download_dir, "tensorrt_extract")
                os.makedirs(extract_dir, exist_ok=True)

                for deb_path in [
                    tensorrt_path,
                    tensorrt_dev_path,
                    tensorrt_plugin_path,
                    tensorrt_plugin_dev_path,
                ]:
                    if os.path.exists(deb_path):
                        # Extract using ar and tar
                        extract_process = await asyncio.create_subprocess_exec(
                            "bash",
                            "-c",
                            f"cd {extract_dir} && ar x {deb_path} && tar xf data.tar.* 2>/dev/null || tar xf data.tar 2>/dev/null",
                            stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.STDOUT,
                        )
                        await extract_process.wait()

                # Copy TensorRT files to CUDA installation
                tensorrt_lib_src = os.path.join(extract_dir, "usr", "lib", "x86_64-linux-gnu")
                tensorrt_include_src = os.path.join(extract_dir, "usr", "include")
                tensorrt_bin_src = os.path.join(extract_dir, "usr", "bin")

                cuda_lib_dst = os.path.join(cuda_path, "lib64")
                cuda_include_dst = os.path.join(cuda_path, "include")
                cuda_bin_dst = os.path.join(cuda_path, "bin")

                # Copy libraries
                if os.path.exists(tensorrt_lib_src):
                    for f in os.listdir(tensorrt_lib_src):
                        if "nvinfer" in f.lower() or "tensorrt" in f.lower():
                            src = os.path.join(tensorrt_lib_src, f)
                            dst = os.path.join(cuda_lib_dst, f)
                            try:
                                if os.path.islink(src):
                                    linkto = os.readlink(src)
                                    if os.path.exists(dst):
                                        os.remove(dst)
                                    os.symlink(linkto, dst)
                                else:
                                    shutil.copy2(src, dst)
                                await self._broadcast_log_line(f"Copied {f} to CUDA lib directory")
                            except Exception as copy_err:
                                await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                # Copy headers
                if os.path.exists(tensorrt_include_src):
                    for f in os.listdir(tensorrt_include_src):
                        if "nvinfer" in f.lower() or "tensorrt" in f.lower():
                            src = os.path.join(tensorrt_include_src, f)
                            dst = os.path.join(cuda_include_dst, f)
                            try:
                                if os.path.isdir(src):
                                    shutil.copytree(src, dst, dirs_exist_ok=True)
                                else:
                                    shutil.copy2(src, dst)
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA include directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                # Copy binaries (like trtexec)
                if os.path.exists(tensorrt_bin_src):
                    for f in os.listdir(tensorrt_bin_src):
                        if "trt" in f.lower() or "nvinfer" in f.lower():
                            src = os.path.join(tensorrt_bin_src, f)
                            dst = os.path.join(cuda_bin_dst, f)
                            try:
                                shutil.copy2(src, dst)
                                os.chmod(dst, 0o755)
                                await self._broadcast_log_line(f"Copied {f} to CUDA bin directory")
                            except Exception as copy_err:
                                await self._broadcast_log_line(f"Failed to copy {f}: {copy_err}")

                # Cleanup temporary extract directory only (keep .deb files)
                shutil.rmtree(extract_dir, ignore_errors=True)
                # Keep .deb files for future use
                logger.info(
                    f"TensorRT packages kept at: {tensorrt_path}, {tensorrt_dev_path}, "
                    f"{tensorrt_plugin_path}, {tensorrt_plugin_dev_path}"
                )

                await self._broadcast_log_line("TensorRT extracted to CUDA directory")
                await self._broadcast_progress(
                    {
                        "stage": "tensorrt",
                        "progress": 100,
                        "message": "TensorRT installed successfully",
                    }
                )
            else:
                await self._broadcast_log_line(
                    "TensorRT packages not available, skipping TensorRT installation"
                )
                await self._broadcast_progress(
                    {
                        "stage": "tensorrt",
                        "progress": 100,
                        "message": "TensorRT installation skipped (optional)",
                    }
                )

        except Exception as e:
            await self._broadcast_log_line(f"TensorRT installation error: {e}")
            await self._broadcast_log_line(
                "Note: TensorRT is optional. The build will continue without TensorRT support."
            )
            await self._broadcast_progress(
                {
                    "stage": "tensorrt",
                    "progress": 100,
                    "message": "TensorRT installation skipped (optional)",
                }
            )

    async def install(
        self,
        version: str = "12.6",
        install_cudnn: bool = False,
        install_tensorrt: bool = False,
    ) -> Dict[str, Any]:
        """Install CUDA Toolkit with optional cuDNN and TensorRT."""
        async with self._lock:
            version = str(version).strip()
            if self._operation:
                raise CudaOperationBusy("Another CUDA installer operation is already running")

            system, arch = self._get_platform()

            if system != "linux":
                raise CudaUnsupportedPlatform(
                    f"CUDA installation is only supported on Linux, not {system}"
                )

            try:
                self._cuda_version_parts(version)
            except ValueError as exc:
                raise CudaVersionUnsupported(str(exc)) from exc
            available_versions = await self.get_available_versions()
            permitted_versions = {*self.SUPPORTED_VERSIONS, *available_versions}
            if version not in permitted_versions:
                raise CudaVersionUnsupported(
                    f"Unsupported or unavailable CUDA version: {version}. Available versions: {', '.join(available_versions)}"
                )

            # Fetch the download URL dynamically
            await self._broadcast_log_line(f"Fetching download URL for CUDA {version}...")
            url = await self._fetch_download_url(version)
            installer_filename = os.path.basename(url)
            installer_path = os.path.join(self._download_dir, installer_filename)

            await self._start_operation("install", {"version": version})

            async def _runner():
                try:
                    # Download installer
                    await self._download_installer(version, url, installer_path)

                    # Install (Linux only) - returns the installation path
                    install_path = await self._install_linux(
                        installer_path, version, install_cudnn, install_tensorrt
                    )

                    # Update state (already saved in _install_linux, but update main fields)
                    state = self._load_state()
                    state["installed_version"] = version
                    state["installed_at"] = _utcnow()
                    state["cuda_path"] = install_path
                    if install_cudnn:
                        state["cudnn_installed"] = True
                    if install_tensorrt:
                        state["tensorrt_installed"] = True
                    self._save_state(state)

                    components = ["CUDA Toolkit"]
                    if install_cudnn:
                        components.append("cuDNN")
                    if install_tensorrt:
                        components.append("TensorRT")

                    await self._finish_operation(
                        True, f"{', '.join(components)} installed successfully"
                    )

                    # Update current process environment with CUDA paths
                    # This ensures the running application can use CUDA immediately
                    cuda_env = self.get_cuda_env(version)
                    if cuda_env:
                        os.environ.update(cuda_env)
                        logger.info(f"Updated process environment with CUDA {version} paths")

                    # Restart Studio proxy to pick up new CUDA environment variables
                    # Studio proxy needs to be restarted because subprocess environment
                    # variables are set at process creation time and can't be changed
                    try:
                        from backend.proxy.control import get_proxy_controller

                        native_proxy_manager = get_proxy_controller()
                        await native_proxy_manager.restart_proxy()
                        logger.info("Restarted Studio proxy to pick up new CUDA environment")
                    except Exception as restart_error:
                        # Don't fail the installation if restart fails
                        logger.warning(
                            f"Failed to restart Studio proxy after CUDA installation: {restart_error}. "
                            f"You may need to manually restart Studio proxy to use the new CUDA version."
                        )

                    # Keep installer file for future use (not deleting)
                    logger.info(f"Installer file kept at: {installer_path}")

                except Exception as exc:
                    self._last_error = str(exc)
                    await self._finish_operation(False, str(exc))
                    raise

            self._create_task(_runner())
            return self._started_response(f"CUDA {version} installation started")

    def _detect_cudnn_version(self, cuda_path: Optional[str]) -> Optional[str]:
        """Detect installed cuDNN version by checking library files."""
        if not cuda_path:
            return None

        lib_path = os.path.join(cuda_path, "lib64")
        if not os.path.exists(lib_path):
            return None

        try:
            for f in os.listdir(lib_path):
                if "libcudnn" in f and ".so" in f:
                    match = re.search(r"\.so(?:\.(\d+(?:\.\d+){0,2}))?", f)
                    if match and match.group(1):
                        return match.group(1)
        except Exception:
            pass

        return None

    def _detect_tensorrt_version(self, cuda_path: Optional[str]) -> Optional[str]:
        """Detect installed TensorRT version by checking library files."""
        if not cuda_path:
            return None

        lib_path = os.path.join(cuda_path, "lib64")
        if not os.path.exists(lib_path):
            return None

        try:
            for f in os.listdir(lib_path):
                if "libnvinfer" in f and ".so" in f and "plugin" not in f:
                    match = re.search(r"\.so(?:\.(\d+(?:\.\d+){0,2}))?", f)
                    if match and match.group(1):
                        return match.group(1)
        except Exception:
            pass

        return None

    def status(self) -> Dict[str, Any]:
        """Get CUDA installation status."""
        version = self._detect_installed_version()
        cuda_path = self._get_cuda_path()
        installed = version is not None and cuda_path is not None
        state = self._load_state()
        installations = state.get("installations", {})

        # Detect cuDNN and TensorRT
        cudnn_version = None
        tensorrt_version = None
        if cuda_path:
            cudnn_version = self._detect_cudnn_version(cuda_path)
            tensorrt_version = self._detect_tensorrt_version(cuda_path)

        # Get all installed versions with their details
        installed_versions = []
        for v, info in installations.items():
            install_path = info.get("path")
            if install_path and os.path.exists(install_path):
                installed_versions.append(
                    {
                        "version": v,
                        "path": install_path,
                        "size_bytes": self._directory_size(install_path),
                        "installed_at": info.get("installed_at"),
                        "is_system_install": info.get("is_system_install", False),
                        "is_current": v == version,
                        "cudnn_installed": info.get("cudnn_installed", False),
                        "tensorrt_installed": info.get("tensorrt_installed", False),
                    }
                )

        return {
            "installed": installed,
            "version": version,
            "cuda_path": cuda_path,
            "installed_at": state.get("installed_at"),
            "installed_versions": installed_versions,
            "installer_cache": self._installer_cache_status(),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "last_error": self._last_error,
            "log_path": self._log_path,
            "available_versions": self.SUPPORTED_VERSIONS,
            "platform": self._get_platform(),
            "cudnn": {
                "installed": cudnn_version is not None,
                "version": cudnn_version,
            },
            "tensorrt": {
                "installed": tensorrt_version is not None,
                "version": tensorrt_version,
            },
        }

    def read_log_tail(self, max_bytes: int = 8192) -> str:
        if not os.path.exists(self._log_path):
            return ""
        with open(self._log_path, "rb") as log_file:
            log_file.seek(0, os.SEEK_END)
            size = log_file.tell()
            log_file.seek(max(0, size - max_bytes))
            data = log_file.read().decode("utf-8", errors="replace")
            if size > max_bytes:
                data = data.split("\n", 1)[-1]
            return data.strip()

    async def uninstall(self, version: Optional[str] = None) -> Dict[str, Any]:
        """Uninstall CUDA Toolkit."""
        async with self._lock:
            if self._operation:
                raise CudaOperationBusy("Another CUDA installer operation is already running")

            # Determine which version to uninstall
            if not version:
                # Uninstall the currently detected version
                version = self._detect_installed_version()
                if not version:
                    raise CudaVersionNotInstalled("No CUDA installation found to uninstall")

            state = self._load_state()
            installations = state.get("installations", {})

            if version not in installations:
                raise CudaVersionNotInstalled(f"CUDA {version} installation not found in state")

            install_info = installations[version]
            install_path = install_info.get("path")

            if not install_path or not os.path.exists(install_path):
                # Path doesn't exist, just remove from state
                logger.warning(
                    f"CUDA installation path {install_path} does not exist, removing from state only"
                )
                installations.pop(version, None)
                if state.get("installed_version") == version:
                    state["installed_version"] = None
                    state["installed_at"] = None
                    state["cuda_path"] = None
                self._save_state(state)
                return {
                    "message": f"CUDA {version} removed from state (installation path not found)"
                }

            await self._start_operation("uninstall", {"version": version})

            async def _runner():
                try:
                    await self._broadcast_log_line(f"Starting uninstallation of CUDA {version}...")
                    await self._broadcast_progress(
                        {
                            "stage": "uninstall",
                            "progress": 0,
                            "message": f"Uninstalling CUDA {version}...",
                        }
                    )

                    # Remove the installation directory
                    if os.path.exists(install_path):
                        await self._broadcast_log_line(
                            f"Removing installation directory: {install_path}"
                        )
                        try:
                            shutil.rmtree(install_path)
                            await self._broadcast_log_line(f"Successfully removed {install_path}")
                        except Exception as e:
                            logger.error(f"Failed to remove CUDA installation directory: {e}")
                            raise RuntimeError(f"Failed to remove installation directory: {e}")

                    # Update state
                    installations.pop(version, None)
                    if state.get("installed_version") == version:
                        state["installed_version"] = None
                        state["installed_at"] = None
                        state["cuda_path"] = None
                    self._save_state(state)

                    # Update or remove the current symlink
                    self._remove_current_symlink()
                    await self._broadcast_log_line(
                        "Updated CUDA current symlink (removed or re-pointed to another version)"
                    )

                    await self._broadcast_progress(
                        {
                            "stage": "uninstall",
                            "progress": 100,
                            "message": "CUDA uninstallation completed",
                        }
                    )
                    await self._broadcast_log_line(f"CUDA {version} uninstalled successfully")
                    await self._finish_operation(True, f"CUDA {version} uninstalled successfully")

                except Exception as exc:
                    self._last_error = str(exc)
                    await self._finish_operation(False, str(exc))
                    raise

            self._create_task(_runner())
            return self._started_response(f"CUDA {version} uninstallation started")

    async def cleanup_installers(self) -> Dict[str, Any]:
        """Delete cached CUDA installer downloads, leaving installed toolkits alone.

        Only regular files under the download directory are removed. Directory
        entries and symlinks are skipped so cleanup can never follow a link out
        of the cache, and installed toolkits live under a different root.
        """
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another CUDA installer operation is already running")
            return cuda_downloads.cleanup_installers(self._download_dir)
