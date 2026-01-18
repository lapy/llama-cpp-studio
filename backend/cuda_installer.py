"""
CUDA Toolkit Installer

Handles downloading and installing CUDA Toolkit on Linux systems.
"""

import asyncio
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import gzip
from datetime import datetime, timezone
from typing import Any, Awaitable, Dict, Optional, Tuple
import aiohttp
import aiofiles

from backend.logging_config import get_logger
from backend.websocket_manager import websocket_manager

logger = get_logger(__name__)

_installer_instance: Optional["CUDAInstaller"] = None


def get_cuda_installer() -> "CUDAInstaller":
    global _installer_instance
    if _installer_instance is None:
        _installer_instance = CUDAInstaller()
    return _installer_instance


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class CUDAInstaller:
    """Install CUDA Toolkit on Linux systems."""

    # Supported CUDA versions - URLs are fetched dynamically from NVIDIA's archive
    # Format: version -> platform -> architecture (URLs fetched on demand)
    SUPPORTED_VERSIONS = [
        "13.0",
        "12.9",
        "12.8",
        "12.7",
        "12.6",
        "12.5",
        "12.4",
        "12.3",
        "12.2",
        "12.1",
        "12.0",
        "11.9",
        "11.8",
    ]

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
        "11": "8.6.1",   # TensorRT 8.x for CUDA 11.x
    }

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        download_dir: Optional[str] = None,
    ) -> None:
        self._lock = asyncio.Lock()
        self._operation: Optional[str] = None
        self._operation_started_at: Optional[str] = None
        self._current_task: Optional[asyncio.Task] = None
        self._last_error: Optional[str] = None
        self._download_progress: Dict[str, Any] = {}
        self._last_logged_percentage: int = -1

        # Determine data root - check Docker path first, then fallback to local
        if os.path.exists("/app/data"):
            data_root = "/app/data"
        else:
            data_root = os.path.abspath("data")

        log_path = log_path or os.path.join(data_root, "logs", "cuda_install.log")
        state_path = state_path or os.path.join(
            data_root, "configs", "cuda_installer.json"
        )
        download_dir = download_dir or os.path.join(
            data_root, "temp", "cuda_installers"
        )
        self._cuda_install_dir = os.path.join(data_root, "cuda")

        self._log_path = os.path.abspath(log_path)
        self._state_path = os.path.abspath(state_path)
        self._download_dir = os.path.abspath(download_dir)
        self._url_cache: Dict[str, str] = {}  # Cache for dynamically fetched URLs
        self._repo_cache: Dict[str, list] = {}  # Cache for NVIDIA repo packages
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        os.makedirs(self._download_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        os.makedirs(self._cuda_install_dir, exist_ok=True)

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
        """Get platform (os, arch) tuple."""
        system = platform.system().lower()
        machine = platform.machine().lower()

        if machine in ("x86_64", "amd64"):
            arch = "x86_64"
        else:
            arch = machine

        return system, arch

    def _get_ubuntu_version(self) -> str:
        """Get Ubuntu version for NVIDIA repository URLs."""
        # Try to detect Ubuntu version from /etc/os-release
        try:
            if os.path.exists("/etc/os-release"):
                with open("/etc/os-release", "r") as f:
                    for line in f:
                        if line.startswith("VERSION_ID="):
                            version = line.split("=")[1].strip().strip('"')
                            # Extract major.minor (e.g., "24.04" from "24.04.1")
                            parts = version.split(".")
                            if len(parts) >= 2:
                                major_minor = f"{parts[0]}{parts[1]}"
                                # Check if it's 24.04 or newer
                                if major_minor >= "2404":
                                    return "ubuntu2404"
                                else:
                                    return "ubuntu2204"
        except Exception:
            pass
        
        # Default to ubuntu2404 for Ubuntu 24.04 base image
        return "ubuntu2404"

    def _get_archive_target_version(self) -> str:
        """Get archive target version for CUDA runfile lookups."""
        ubuntu_version = self._get_ubuntu_version()
        if ubuntu_version == "ubuntu2404":
            return "24.04"
        return "22.04"

    async def _get_repo_packages(self, ubuntu_version: str) -> list:
        """Fetch and cache NVIDIA CUDA repo package metadata."""
        if ubuntu_version in self._repo_cache:
            return self._repo_cache[ubuntu_version]

        base_url = (
            f"https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64"
        )
        packages_url = f"{base_url}/Packages.gz"
        packages_plain_url = f"{base_url}/Packages"
        packages: list = []

        async with aiohttp.ClientSession() as session:
            data = None
            try:
                async with session.get(packages_url) as response:
                    if response.status == 200:
                        compressed = await response.read()
                        data = gzip.decompress(compressed)
            except Exception:
                data = None

            if data is None:
                try:
                    async with session.get(packages_plain_url) as response:
                        if response.status == 200:
                            data = await response.read()
                except Exception:
                    data = None

        if not data:
            self._repo_cache[ubuntu_version] = []
            return []

        text = data.decode("utf-8", errors="replace")
        current = {}
        for line in text.splitlines():
            if not line.strip():
                if current:
                    packages.append(current)
                    current = {}
                continue
            if line.startswith("Package:"):
                current["Package"] = line.split(":", 1)[1].strip()
            elif line.startswith("Version:"):
                current["Version"] = line.split(":", 1)[1].strip()
            elif line.startswith("Filename:"):
                current["Filename"] = line.split(":", 1)[1].strip()

        if current:
            packages.append(current)

        self._repo_cache[ubuntu_version] = packages
        return packages

    def _version_key(self, version: str) -> tuple:
        """Create a sortable key for package version strings."""
        tokens = re.split(r"[^\w]+", version)
        key = []
        for token in tokens:
            if token.isdigit():
                key.append(int(token))
            elif token:
                key.append(token)
        return tuple(key)

    def _select_repo_package(
        self,
        packages: list,
        package_name: str,
        version_prefix: Optional[str] = None,
        version_contains: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """Select the best matching package from repo metadata."""
        candidates = [
            pkg for pkg in packages if pkg.get("Package") == package_name
        ]
        if version_prefix:
            candidates = [
                pkg
                for pkg in candidates
                if pkg.get("Version", "").startswith(version_prefix)
            ]
        if version_contains:
            candidates = [
                pkg
                for pkg in candidates
                if version_contains in pkg.get("Version", "")
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda pkg: self._version_key(pkg.get("Version", "")))

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
        # First check state for installed versions
        state = self._load_state()
        installations = state.get("installations", {})
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

    def _get_cuda_path(self, version: Optional[str] = None) -> Optional[str]:
        """Get CUDA installation path."""
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

        # Check state for installed versions
        state = self._load_state()
        installations = state.get("installations", {})

        # If version specified, return that installation path
        if version and version in installations:
            install_path = installations[version].get("path")
            if install_path and os.path.exists(install_path):
                return install_path

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

        # Add to PATH if bin directory exists
        if os.path.exists(cuda_bin):
            current_path = os.environ.get("PATH", "")
            if cuda_bin not in current_path:
                env["PATH"] = f"{cuda_bin}:{current_path}" if current_path else cuda_bin

        # Add to LD_LIBRARY_PATH if lib64 directory exists
        if os.path.exists(cuda_lib):
            current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
            if cuda_lib not in current_ld_path:
                env["LD_LIBRARY_PATH"] = (
                    f"{cuda_lib}:{current_ld_path}" if current_ld_path else cuda_lib
                )

        # Add TensorRT path if TensorRT is installed
        tensorrt_version = self._detect_tensorrt_version(cuda_path)
        if tensorrt_version:
            env["TENSORRT_PATH"] = cuda_path
            env["TENSORRT_ROOT"] = cuda_path

        return env

    def _get_archive_url(self, version: str) -> str:
        """Get NVIDIA download archive URL for a CUDA version."""
        # Convert version like "12.8" to "12-8-0" for URL
        version_parts = version.split(".")
        major = version_parts[0]
        minor = version_parts[1] if len(version_parts) > 1 else "0"
        patch = version_parts[2] if len(version_parts) > 2 else "0"
        version_slug = f"{major}-{minor}-{patch}"
        target_version = self._get_archive_target_version()

        return (
            f"https://developer.nvidia.com/cuda-{version_slug}-download-archive"
            f"?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version={target_version}&target_type=runfile_local"
        )

    async def _fetch_download_url(self, version: str) -> str:
        """Fetch the actual download URL from NVIDIA's archive page."""
        # Check cache first
        cache_key = f"{version}_linux_x86_64"
        if cache_key in self._url_cache:
            return self._url_cache[cache_key]

        archive_url = self._get_archive_url(version)
        logger.info(f"Fetching CUDA {version} download URL from {archive_url}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    archive_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to fetch archive page: HTTP {response.status}"
                        )

                    html = await response.text()

                    # The page contains JSON data with download URLs
                    # The JSON structure has keys like "Linux/x86_64/Ubuntu/24.04/runfile_local"
                    # The URL is in the "details" field which contains HTML with href attributes
                    target_version = self._get_archive_target_version()
                    json_key = f"Linux/x86_64/Ubuntu/{target_version}/runfile_local"

                    # Pattern 1: Look for href in the details field (HTML may be escaped)
                    # Match: "Linux/x86_64/Ubuntu/<version>/runfile_local":{..."details":"...href=\"URL\"..."}
                    pattern1 = rf'"{re.escape(json_key)}"[^}}]*"details"[^"]*href[=:][\\"]*([^"\\s<>]+cuda_\d+\.\d+\.\d+_[^"\\s<>]+_linux\.run)'
                    matches = re.findall(pattern1, html, re.IGNORECASE | re.DOTALL)

                    if not matches:
                        # Pattern 2: Look for href with escaped quotes (\u0022 or \")
                        pattern2 = rf'"{re.escape(json_key)}"[^}}]*href[\\u0022=:]*([^"\\s<>]+cuda_\d+\.\d+\.\d+_[^"\\s<>]+_linux\.run)'
                        matches = re.findall(pattern2, html, re.IGNORECASE | re.DOTALL)

                    if not matches:
                        # Pattern 3: Look for the filename field and construct URL
                        pattern3 = rf'"{re.escape(json_key)}"[^}}]*"filename"[^"]*"([^"]+_linux\.run)"'
                        filename_matches = re.findall(pattern3, html, re.IGNORECASE)
                        if filename_matches:
                            filename = filename_matches[0]
                            version_full = f"{version}.0"
                            url = f"https://developer.download.nvidia.com/compute/cuda/{version_full}/local_installers/{filename}"
                            matches = [url]

                    if not matches:
                        # Pattern 4: Fallback - look for any URL matching the pattern
                        version_escaped = version.replace(".", r"\.")
                        pattern4 = rf'https://developer\.download\.nvidia\.com/compute/cuda/{version_escaped}\.0/local_installers/cuda_{version_escaped}\.0_[^"\'\s<>]+_linux\.run'
                        matches = re.findall(pattern4, html, re.IGNORECASE)

                    if matches:
                        url = matches[0]
                        # Cache it
                        self._url_cache[cache_key] = url
                        logger.info(f"Found CUDA {version} download URL: {url}")
                        return url
                    else:
                        raise RuntimeError(
                            f"Could not find download URL for CUDA {version} on archive page"
                        )

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Failed to fetch archive page: {e}")

    async def _broadcast_log_line(self, line: str) -> None:
        try:
            await websocket_manager.broadcast(
                {
                    "type": "cuda_install_log",
                    "line": line,
                    "timestamp": _utcnow(),
                }
            )
        except Exception as exc:
            logger.debug(f"Failed to broadcast CUDA log line: {exc}")

    async def _broadcast_progress(self, progress: Dict[str, Any]) -> None:
        try:
            await websocket_manager.broadcast(
                {
                    "type": "cuda_install_progress",
                    **progress,
                    "timestamp": _utcnow(),
                }
            )
        except Exception as exc:
            logger.debug(f"Failed to broadcast CUDA progress: {exc}")

    async def _set_operation(self, operation: str) -> None:
        self._operation = operation
        self._operation_started_at = _utcnow()
        self._last_error = None
        await websocket_manager.broadcast(
            {
                "type": "cuda_install_status",
                "status": operation,
                "started_at": self._operation_started_at,
            }
        )

    async def _finish_operation(self, success: bool, message: str = "") -> None:
        payload = {
            "type": "cuda_install_status",
            "status": "completed" if success else "failed",
            "operation": self._operation,
            "message": message,
            "ended_at": _utcnow(),
        }
        await websocket_manager.broadcast(payload)
        self._operation = None
        self._operation_started_at = None

    def _create_task(self, coro: Awaitable[Any]) -> None:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        self._current_task = task

        def _cleanup(fut: asyncio.Future) -> None:
            try:
                fut.result()
            except Exception as exc:
                logger.exception("CUDA installer task error")
            finally:
                self._current_task = None

        task.add_done_callback(_cleanup)

    async def _download_installer(
        self, version: str, url: str, installer_path: str
    ) -> None:
        """Download CUDA installer with progress tracking."""
        # Check if installer already exists
        if os.path.exists(installer_path):
            file_size = os.path.getsize(installer_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Verify existing file is not corrupted (should be at least 100MB for CUDA installers)
            if file_size < 100 * 1024 * 1024:
                await self._broadcast_log_line(
                    f"Existing installer file appears corrupted (too small: {file_size_mb:.1f} MB), re-downloading..."
                )
                try:
                    os.remove(installer_path)
                except OSError:
                    pass
            else:
                # Verify the file is actually valid and matches expected size from server
                try:
                    # First, check if it's a valid shell script
                    with open(installer_path, "rb") as f:
                        header = f.read(100)
                        if not header.startswith(b"#!/"):
                            await self._broadcast_log_line(
                                f"Existing installer file is not a valid shell script, re-downloading..."
                            )
                            try:
                                os.remove(installer_path)
                            except OSError:
                                pass
                        else:
                            # File appears valid, now verify size matches server expectation
                            # Fetch the expected file size from the server
                            try:
                                async with aiohttp.ClientSession() as session:
                                    async with session.head(url, allow_redirects=True) as head_response:
                                        expected_size = int(head_response.headers.get("Content-Length", 0))
                                        
                                        if expected_size > 0:
                                            # Verify file size matches (with small tolerance)
                                            size_diff = abs(file_size - expected_size)
                                            if size_diff > 1024:  # Allow 1KB tolerance
                                                await self._broadcast_log_line(
                                                    f"Existing installer file size mismatch: expected {expected_size / (1024*1024):.1f} MB, "
                                                    f"got {file_size_mb:.1f} MB (difference: {size_diff} bytes). Re-downloading..."
                                                )
                                                try:
                                                    os.remove(installer_path)
                                                except OSError:
                                                    pass
                                            else:
                                                # File size matches, verify it's stable (not currently being written)
                                                await asyncio.sleep(0.2)  # Brief pause to ensure file is fully written if being written
                                                new_size = os.path.getsize(installer_path)
                                                if new_size != file_size:
                                                    await self._broadcast_log_line(
                                                        f"File size changed during verification (was {file_size_mb:.1f} MB, now {new_size / (1024*1024):.1f} MB), "
                                                        f"file may still be downloading. Re-downloading..."
                                                    )
                                                    try:
                                                        os.remove(installer_path)
                                                    except OSError:
                                                        pass
                                                else:
                                                    await self._broadcast_log_line(
                                                        f"Installer file already exists and verified: {installer_path} ({file_size_mb:.1f} MB)"
                                                    )
                                                    await self._broadcast_progress(
                                                        {
                                                            "stage": "download",
                                                            "progress": 100,
                                                            "message": f"Using existing installer file ({file_size_mb:.1f} MB)",
                                                        }
                                                    )
                                                    return
                                        else:
                                            # Couldn't get expected size, but file looks valid - use it
                                            await self._broadcast_log_line(
                                                f"Installer file already exists: {installer_path} ({file_size_mb:.1f} MB). "
                                                f"Could not verify size from server, but file appears valid."
                                            )
                                            await self._broadcast_progress(
                                                {
                                                    "stage": "download",
                                                    "progress": 100,
                                                    "message": f"Using existing installer file ({file_size_mb:.1f} MB)",
                                                }
                                            )
                                            return
                            except Exception as size_check_error:
                                # If we can't verify size from server, but file looks valid, use it
                                await self._broadcast_log_line(
                                    f"Could not verify file size from server: {size_check_error}. "
                                    f"File appears valid, using existing file: {installer_path} ({file_size_mb:.1f} MB)"
                                )
                                await self._broadcast_progress(
                                    {
                                        "stage": "download",
                                        "progress": 100,
                                        "message": f"Using existing installer file ({file_size_mb:.1f} MB)",
                                    }
                                )
                                return
                except (OSError, IOError) as e:
                    await self._broadcast_log_line(
                        f"Failed to verify existing installer file: {e}, re-downloading..."
                    )
                    try:
                        os.remove(installer_path)
                    except OSError:
                        pass

        # Reset logging state for new download
        self._last_logged_percentage = -1

        log_header = f"[{_utcnow()}] Downloading CUDA {version} installer from {url}\n"
        with open(self._log_path, "w", encoding="utf-8") as log_file:
            log_file.write(log_header)

        await self._broadcast_log_line(
            f"Starting download of CUDA {version} installer..."
        )
        await self._broadcast_progress(
            {
                "stage": "download",
                "progress": 0,
                "message": f"Downloading CUDA {version} installer...",
            }
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0

                async with aiofiles.open(installer_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded += len(chunk)

                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            # Format sizes in MB
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            await self._broadcast_progress(
                                {
                                    "stage": "download",
                                    "progress": progress,
                                    "message": f"Downloading CUDA {version} installer... ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                                    "bytes_downloaded": downloaded,
                                    "total_bytes": total_size,
                                }
                            )

                        # Log progress only at key percentage milestones (10%, 25%, 50%, 75%, 90%, 100%)
                        # Only log when we cross a milestone, not when we're within it
                        should_log = False

                        # Check if we've crossed a key percentage milestone
                        if total_size > 0:
                            progress = int((downloaded / total_size) * 100)
                            if progress != self._last_logged_percentage and progress in [
                                10,
                                25,
                                50,
                                75,
                                90,
                                100,
                            ]:
                                should_log = True
                                self._last_logged_percentage = progress

                        if should_log:
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            log_line = f"Downloaded {downloaded_mb:.1f}/{total_mb:.1f} MB ({progress}%)\n"
                            with open(
                                self._log_path, "a", encoding="utf-8"
                            ) as log_file:
                                log_file.write(log_line)
                            await self._broadcast_log_line(
                                f"Downloaded {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({progress}%)"
                            )
                
                # File is automatically flushed when the context manager exits

        # Wait a brief moment to ensure file system has fully written the file
        # This helps ensure the file is completely written to disk before verification
        await asyncio.sleep(0.5)
        
        # Verify downloaded file exists and is complete
        if not os.path.exists(installer_path):
            raise RuntimeError(f"Downloaded file not found: {installer_path}")
        
        # Verify file size matches expected size (with a small tolerance for filesystem differences)
        actual_size = os.path.getsize(installer_path)
        if total_size > 0:
            size_diff = abs(actual_size - total_size)
            if size_diff > 1024:  # Allow 1KB tolerance for filesystem differences
                raise RuntimeError(
                    f"Downloaded file size mismatch: expected {total_size} bytes, "
                    f"got {actual_size} bytes (difference: {size_diff} bytes). File may be corrupted or incomplete."
                )
        
        if actual_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious
            raise RuntimeError(
                f"Downloaded file appears to be corrupted or incomplete: "
                f"{installer_path} (size: {actual_size} bytes)"
            )
        
        # Verify the file is a valid shell script (CUDA .run files are self-extracting)
        try:
            with open(installer_path, "rb") as verify_file:
                header = verify_file.read(100)
                if not header.startswith(b"#!/"):
                    raise RuntimeError(
                        f"Downloaded file does not appear to be a valid shell script: {installer_path}"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Failed to verify downloaded file integrity: {installer_path}, error: {e}"
            )
        
        await self._broadcast_log_line(
            f"Download completed and verified: {installer_path} ({actual_size / (1024*1024):.1f} MB)"
        )
        await self._broadcast_progress(
            {
                "stage": "download",
                "progress": 100,
                "message": "Download completed and verified",
            }
        )

    def _is_docker_container(self) -> bool:
        """Check if running inside a Docker container."""
        # Check for Docker-specific files
        docker_indicators = [
            "/.dockerenv",
            "/proc/self/cgroup",
        ]

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
        except (OSError, IOError):
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
            raise RuntimeError(f"Installer file not found: {installer_path}")
        
        file_size = os.path.getsize(installer_path)
        if file_size < 100 * 1024 * 1024:  # Less than 100MB is suspicious for CUDA installers
            raise RuntimeError(
                f"Installer file appears to be corrupted or incomplete: {installer_path} "
                f"(size: {file_size / (1024*1024):.1f} MB, expected > 100 MB)"
            )
        
        # Verify the file starts with a shell script header (CUDA .run files are self-extracting)
        try:
            with open(installer_path, "rb") as f:
                header = f.read(100)
                if not header.startswith(b"#!/"):
                    raise RuntimeError(
                        f"Installer file does not appear to be a valid shell script: {installer_path}"
                    )
        except Exception as e:
            raise RuntimeError(
                f"Failed to verify installer file: {installer_path}, error: {e}"
            )
        
        await self._broadcast_log_line(
            f"Verifying installer file: {installer_path} ({file_size / (1024*1024):.1f} MB)"
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
            "--silent",                    # Silent installation with EULA acceptance
            "--toolkit",                   # Install toolkit only (not driver)
            "--override",                  # Override installation checks for custom paths
            f"--toolkitpath={install_path}", # Install to custom data directory
            "--no-opengl-libs",            # Skip OpenGL libraries (not needed in Docker)
            "--no-man-page",               # Skip man pages to reduce size
        ]
        
        await self._broadcast_log_line(f"Installer arguments: {' '.join(install_args[2:])}")  # Skip 'bash' and installer_path

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
            elif "gzip" in output_text.lower() and ("unexpected end" in output_text.lower() or "corrupt" in output_text.lower()):
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
            
            raise RuntimeError(error_msg)

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

        await self._broadcast_log_line(
            f"CUDA installed successfully to: {install_path}"
        )
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
                        await self._broadcast_log_line(
                            f"Download error for {name}: {download_err}"
                        )
                        continue

            await self._broadcast_progress(
                {
                    "stage": "nccl",
                    "progress": 50,
                    "message": "Installing NCCL packages...",
                }
            )

            if os.path.exists(nccl_path):
                await self._broadcast_log_line(
                    "Extracting NCCL to CUDA directory..."
                )

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
                nccl_lib_src = os.path.join(
                    extract_dir, "usr", "lib", "x86_64-linux-gnu"
                )
                nccl_include_src = os.path.join(extract_dir, "usr", "include")

                cuda_lib_dst = os.path.join(cuda_path, "lib64")
                cuda_include_dst = os.path.join(cuda_path, "include")

                if os.path.exists(nccl_lib_src):
                    for f in os.listdir(nccl_lib_src):
                        if "nccl" in f.lower():
                            src = os.path.join(nccl_lib_src, f)
                            dst = os.path.join(cuda_lib_dst, f)
                            try:
                                if os.path.islink(src):
                                    linkto = os.readlink(src)
                                    if os.path.exists(dst):
                                        os.remove(dst)
                                    os.symlink(linkto, dst)
                                else:
                                    shutil.copy2(src, dst)
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA lib directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

                if os.path.exists(nccl_include_src):
                    for f in os.listdir(nccl_include_src):
                        if "nccl" in f.lower():
                            src = os.path.join(nccl_include_src, f)
                            dst = os.path.join(cuda_include_dst, f)
                            try:
                                shutil.copy2(src, dst)
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA include directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

                # Cleanup
                shutil.rmtree(extract_dir, ignore_errors=True)
                for deb_path in [nccl_path, nccl_dev_path]:
                    if os.path.exists(deb_path):
                        os.remove(deb_path)

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
                            raise RuntimeError(f"Failed to download nvidia-utils: HTTP {response.status}")
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
                await self._broadcast_log_line(
                    "Extracting nvidia-smi to CUDA directory..."
                )

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
                        await self._broadcast_log_line(
                            "Copied nvidia-smi to CUDA bin directory"
                        )
                        await self._broadcast_progress(
                            {
                                "stage": "nvidia-smi",
                                "progress": 100,
                                "message": "nvidia-smi installed successfully",
                            }
                        )
                    except Exception as copy_err:
                        await self._broadcast_log_line(
                            f"Failed to copy nvidia-smi: {copy_err}"
                        )
                        raise
                else:
                    await self._broadcast_log_line(
                        "nvidia-smi not found in extracted package"
                    )
                    await self._broadcast_progress(
                        {
                            "stage": "nvidia-smi",
                            "progress": 100,
                            "message": "nvidia-smi installation skipped (not in package)",
                        }
                    )

                # Cleanup
                shutil.rmtree(extract_dir, ignore_errors=True)
                if os.path.exists(nvidia_utils_path):
                    os.remove(nvidia_utils_path)

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
        await self._broadcast_log_line(
            "Installing cuDNN (CUDA Deep Neural Network library)..."
        )
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
                        await self._broadcast_log_line(
                            f"Download error for {name}: {download_err}"
                        )
                        continue

            await self._broadcast_progress(
                {
                    "stage": "cudnn",
                    "progress": 50,
                    "message": "Installing cuDNN packages...",
                }
            )

            if os.path.exists(cudnn_path):
                await self._broadcast_log_line(
                    "Extracting cuDNN to CUDA directory..."
                )

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
                cudnn_lib_src = os.path.join(
                    extract_dir, "usr", "lib", "x86_64-linux-gnu"
                )
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
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA lib directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

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
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

                # Cleanup
                shutil.rmtree(extract_dir, ignore_errors=True)
                for deb_path in [cudnn_path, cudnn_dev_path]:
                    if os.path.exists(deb_path):
                        os.remove(deb_path)

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
        await self._broadcast_log_line(
            "Installing TensorRT (NVIDIA TensorRT inference library)..."
        )
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
            tensorrt_plugin_dev_url = base_url + tensorrt_plugin_dev_entry.get("Filename", "").lstrip("./")

            tensorrt_path = os.path.join(self._download_dir, f"{tensorrt_pkg}.deb")
            tensorrt_dev_path = os.path.join(self._download_dir, f"{tensorrt_pkg}-dev.deb")
            tensorrt_plugin_path = os.path.join(self._download_dir, f"{tensorrt_plugin_pkg}.deb")
            tensorrt_plugin_dev_path = os.path.join(self._download_dir, f"{tensorrt_plugin_pkg}-dev.deb")

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
                    (tensorrt_plugin_dev_url, tensorrt_plugin_dev_path, f"{tensorrt_plugin_pkg}-dev"),
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
                        await self._broadcast_log_line(
                            f"Download error for {name}: {download_err}"
                        )
                        continue

            await self._broadcast_progress(
                {
                    "stage": "tensorrt",
                    "progress": 50,
                    "message": "Installing TensorRT packages...",
                }
            )

            if os.path.exists(tensorrt_path):
                await self._broadcast_log_line(
                    "Extracting TensorRT to CUDA directory..."
                )

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
                tensorrt_lib_src = os.path.join(
                    extract_dir, "usr", "lib", "x86_64-linux-gnu"
                )
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
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA lib directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

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
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

                # Copy binaries (like trtexec)
                if os.path.exists(tensorrt_bin_src):
                    for f in os.listdir(tensorrt_bin_src):
                        if "trt" in f.lower() or "nvinfer" in f.lower():
                            src = os.path.join(tensorrt_bin_src, f)
                            dst = os.path.join(cuda_bin_dst, f)
                            try:
                                shutil.copy2(src, dst)
                                os.chmod(dst, 0o755)
                                await self._broadcast_log_line(
                                    f"Copied {f} to CUDA bin directory"
                                )
                            except Exception as copy_err:
                                await self._broadcast_log_line(
                                    f"Failed to copy {f}: {copy_err}"
                                )

                # Cleanup
                shutil.rmtree(extract_dir, ignore_errors=True)
                for deb_path in [
                    tensorrt_path,
                    tensorrt_dev_path,
                    tensorrt_plugin_path,
                    tensorrt_plugin_dev_path,
                ]:
                    if os.path.exists(deb_path):
                        os.remove(deb_path)

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
            if self._operation:
                raise RuntimeError(
                    "Another CUDA installer operation is already running"
                )

            system, arch = self._get_platform()

            if system != "linux":
                raise RuntimeError(
                    f"CUDA installation is only supported on Linux, not {system}"
                )

            if version not in self.SUPPORTED_VERSIONS:
                raise ValueError(
                    f"Unsupported CUDA version: {version}. Supported versions: {', '.join(self.SUPPORTED_VERSIONS)}"
                )

            # Fetch the download URL dynamically
            await self._broadcast_log_line(
                f"Fetching download URL for CUDA {version}..."
            )
            url = await self._fetch_download_url(version)
            installer_filename = os.path.basename(url)
            installer_path = os.path.join(self._download_dir, installer_filename)

            await self._set_operation("install")

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
                        logger.info(
                            f"Updated process environment with CUDA {version} paths"
                        )

                    # Cleanup installer file
                    try:
                        if os.path.exists(installer_path):
                            os.remove(installer_path)
                    except Exception as exc:
                        logger.warning(f"Failed to cleanup installer: {exc}")

                except Exception as exc:
                    self._last_error = str(exc)
                    await self._finish_operation(False, str(exc))
                    raise

            self._create_task(_runner())
            return {"message": f"CUDA {version} installation started"}

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

    def is_operation_running(self) -> bool:
        return self._operation is not None

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
                raise RuntimeError(
                    "Another CUDA installer operation is already running"
                )

            # Determine which version to uninstall
            if not version:
                # Uninstall the currently detected version
                version = self._detect_installed_version()
                if not version:
                    raise RuntimeError("No CUDA installation found to uninstall")

            state = self._load_state()
            installations = state.get("installations", {})

            if version not in installations:
                raise RuntimeError(f"CUDA {version} installation not found in state")

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

            await self._set_operation("uninstall")

            async def _runner():
                try:
                    await self._broadcast_log_line(
                        f"Starting uninstallation of CUDA {version}..."
                    )
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
                            await self._broadcast_log_line(
                                f"Successfully removed {install_path}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to remove CUDA installation directory: {e}"
                            )
                            raise RuntimeError(
                                f"Failed to remove installation directory: {e}"
                            )

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
                    await self._broadcast_log_line(
                        f"CUDA {version} uninstalled successfully"
                    )
                    await self._finish_operation(
                        True, f"CUDA {version} uninstalled successfully"
                    )

                except Exception as exc:
                    self._last_error = str(exc)
                    await self._finish_operation(False, str(exc))
                    raise

            self._create_task(_runner())
            return {"message": f"CUDA {version} uninstallation started"}
