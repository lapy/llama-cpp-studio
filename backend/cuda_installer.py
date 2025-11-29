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
        "13.0", "12.9", "12.8", "12.7", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0",
        "11.9", "11.8"
    ]

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
        state_path = state_path or os.path.join(data_root, "configs", "cuda_installer.json")
        download_dir = download_dir or os.path.join(data_root, "temp", "cuda_installers")
        self._cuda_install_dir = os.path.join(data_root, "cuda")
        
        self._log_path = os.path.abspath(log_path)
        self._state_path = os.path.abspath(state_path)
        self._download_dir = os.path.abspath(download_dir)
        self._url_cache: Dict[str, str] = {}  # Cache for dynamically fetched URLs
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        os.makedirs(self._download_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)
        os.makedirs(self._cuda_install_dir, exist_ok=True)

    def _get_platform(self) -> Tuple[str, str]:
        """Get platform (os, arch) tuple."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        if machine in ("x86_64", "amd64"):
            arch = "x86_64"
        else:
            arch = machine
        
        return system, arch

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
                env=env
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
        
        # Check environment variables
        env_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Check common system installation paths (Linux only)
        common_paths = ["/usr/local/cuda"]
        
        for base_path in common_paths:
            if os.path.exists(base_path):
                # Check for versioned subdirectories
                try:
                    for item in os.listdir(base_path):
                        full_path = os.path.join(base_path, item)
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
                env["LD_LIBRARY_PATH"] = f"{cuda_lib}:{current_ld_path}" if current_ld_path else cuda_lib
        
        return env

    def _get_archive_url(self, version: str) -> str:
        """Get NVIDIA download archive URL for a CUDA version."""
        # Convert version like "12.8" to "12-8-0" for URL
        version_parts = version.split(".")
        major = version_parts[0]
        minor = version_parts[1] if len(version_parts) > 1 else "0"
        patch = version_parts[2] if len(version_parts) > 2 else "0"
        version_slug = f"{major}-{minor}-{patch}"
        
        return (
            f"https://developer.nvidia.com/cuda-{version_slug}-download-archive"
            f"?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local"
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
                async with session.get(archive_url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to fetch archive page: HTTP {response.status}")
                    
                    html = await response.text()
                    
                    # The page contains JSON data with download URLs
                    # The JSON structure has keys like "Linux/x86_64/Ubuntu/22.04/runfile_local"
                    # The URL is in the "details" field which contains HTML with href attributes
                    json_key = "Linux/x86_64/Ubuntu/22.04/runfile_local"
                    
                    # Pattern 1: Look for href in the details field (HTML may be escaped)
                    # Match: "Linux/x86_64/Ubuntu/22.04/runfile_local":{..."details":"...href=\"URL\"..."}
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
                        raise RuntimeError(f"Could not find download URL for CUDA {version} on archive page")
                        
            except aiohttp.ClientError as e:
                raise RuntimeError(f"Failed to fetch archive page: {e}")

    async def _broadcast_log_line(self, line: str) -> None:
        try:
            await websocket_manager.broadcast({
                "type": "cuda_install_log",
                "line": line,
                "timestamp": _utcnow(),
            })
        except Exception as exc:
            logger.debug(f"Failed to broadcast CUDA log line: {exc}")

    async def _broadcast_progress(self, progress: Dict[str, Any]) -> None:
        try:
            await websocket_manager.broadcast({
                "type": "cuda_install_progress",
                **progress,
                "timestamp": _utcnow(),
            })
        except Exception as exc:
            logger.debug(f"Failed to broadcast CUDA progress: {exc}")

    async def _set_operation(self, operation: str) -> None:
        self._operation = operation
        self._operation_started_at = _utcnow()
        self._last_error = None
        await websocket_manager.broadcast({
            "type": "cuda_install_status",
            "status": operation,
            "started_at": self._operation_started_at,
        })

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
                logger.error(f"CUDA installer task error: {exc}")
            finally:
                self._current_task = None

        task.add_done_callback(_cleanup)

    async def _download_installer(
        self, version: str, url: str, installer_path: str
    ) -> None:
        """Download CUDA installer with progress tracking."""
        # Reset logging state for new download
        self._last_logged_percentage = -1
        
        log_header = f"[{_utcnow()}] Downloading CUDA {version} installer from {url}\n"
        with open(self._log_path, "w", encoding="utf-8") as log_file:
            log_file.write(log_header)
        
        await self._broadcast_log_line(f"Starting download of CUDA {version} installer...")
        await self._broadcast_progress({
            "stage": "download",
            "progress": 0,
            "message": f"Downloading CUDA {version} installer...",
        })

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
                            await self._broadcast_progress({
                                "stage": "download",
                                "progress": progress,
                                "message": f"Downloading CUDA {version} installer... ({downloaded_mb:.1f}/{total_mb:.1f} MB)",
                                "bytes_downloaded": downloaded,
                                "total_bytes": total_size,
                            })
                        
                        # Log progress only at key percentage milestones (10%, 25%, 50%, 75%, 90%, 100%)
                        # Only log when we cross a milestone, not when we're within it
                        should_log = False
                        
                        # Check if we've crossed a key percentage milestone
                        if progress != self._last_logged_percentage and progress in [10, 25, 50, 75, 90, 100]:
                            should_log = True
                            self._last_logged_percentage = progress
                        
                        if should_log:
                            downloaded_mb = downloaded / (1024 * 1024)
                            total_mb = total_size / (1024 * 1024)
                            log_line = f"Downloaded {downloaded_mb:.1f}/{total_mb:.1f} MB ({progress}%)\n"
                            with open(self._log_path, "a", encoding="utf-8") as log_file:
                                log_file.write(log_line)
                            await self._broadcast_log_line(f"Downloaded {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({progress}%)")

        await self._broadcast_log_line(f"Download completed: {installer_path}")
        await self._broadcast_progress({
            "stage": "download",
            "progress": 100,
            "message": "Download completed",
        })

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

    async def _install_linux(self, installer_path: str, version: str) -> None:
        """Install CUDA on Linux using runfile installer."""
        await self._broadcast_log_line("Starting CUDA installation on Linux...")
        await self._broadcast_progress({
            "stage": "install",
            "progress": 0,
            "message": "Installing CUDA Toolkit...",
        })

        # Make installer executable
        os.chmod(installer_path, 0o755)

        # Determine installation path - use data directory if not root, otherwise use /usr/local
        is_root = os.geteuid() == 0 if hasattr(os, 'geteuid') else False
        
        if is_root:
            # Root can install to standard location
            install_path = f"/usr/local/cuda-{version}"
            await self._broadcast_log_line(f"Installing to system location: {install_path}")
        else:
            # Non-root: install to data directory
            install_path = os.path.join(self._cuda_install_dir, f"cuda-{version}")
            await self._broadcast_log_line(f"Installing to data directory: {install_path}")
            # Ensure the installation directory exists
            os.makedirs(install_path, exist_ok=True)

        # Linux runfile installer flags:
        # --silent = silent mode
        # --toolkit = install only toolkit (not driver)
        # --override = override existing installation
        # --toolkitpath = custom installation path (for non-root installs)
        install_args = [
            installer_path,
            "--silent",
            "--toolkit",
            "--override",
        ]
        
        # Add toolkitpath only for non-root installs (custom path)
        if not is_root:
            install_args.append(f"--toolkitpath={install_path}")

        # If not root, we don't need sudo since we're installing to a writable location
        if not is_root:
            await self._broadcast_log_line(f"Installing CUDA to user-writable directory: {install_path}")
        
        process = await asyncio.create_subprocess_exec(
            *install_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )

        async def _stream_output():
            if process.stdout is None:
                return
            with open(self._log_path, "a", encoding="utf-8", buffering=1) as log_file:
                while True:
                    chunk = await process.stdout.readline()
                    if not chunk:
                        break
                    text = chunk.decode("utf-8", errors="replace")
                    log_file.write(text)
                    await self._broadcast_log_line(text.rstrip("\n"))

        await asyncio.gather(process.wait(), _stream_output())
        
        if process.returncode != 0:
            # Installation failed
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
        
        await self._broadcast_log_line(f"CUDA installed successfully to: {install_path}")
        await self._broadcast_log_line(f"CUDA_HOME={cuda_home}")
        await self._broadcast_log_line(f"Adding to PATH: {cuda_bin}")
        await self._broadcast_log_line(f"Adding to LD_LIBRARY_PATH: {cuda_lib}")
        
        # Save installation path to state
        state = self._load_state()
        if "installations" not in state:
            state["installations"] = {}
        state["installations"][version] = {
            "path": install_path,
            "installed_at": _utcnow(),
            "is_system_install": is_root,
        }
        self._save_state(state)
        
        await self._broadcast_progress({
            "stage": "install",
            "progress": 100,
            "message": "CUDA installation completed",
        })
        
        return install_path

    async def install(self, version: str = "12.6") -> Dict[str, Any]:
        """Install CUDA Toolkit."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another CUDA installer operation is already running")
            
            system, arch = self._get_platform()
            
            if system != "linux":
                raise RuntimeError(f"CUDA installation is only supported on Linux, not {system}")
            
            if version not in self.SUPPORTED_VERSIONS:
                raise ValueError(f"Unsupported CUDA version: {version}. Supported versions: {', '.join(self.SUPPORTED_VERSIONS)}")
            
            # Fetch the download URL dynamically
            await self._broadcast_log_line(f"Fetching download URL for CUDA {version}...")
            url = await self._fetch_download_url(version)
            installer_filename = os.path.basename(url)
            installer_path = os.path.join(self._download_dir, installer_filename)
            
            await self._set_operation("install")

            async def _runner():
                try:
                    # Download installer
                    await self._download_installer(version, url, installer_path)
                    
                    # Install (Linux only) - returns the installation path
                    install_path = await self._install_linux(installer_path, version)
                    
                    # Update state (already saved in _install_linux, but update main fields)
                    state = self._load_state()
                    state["installed_version"] = version
                    state["installed_at"] = _utcnow()
                    state["cuda_path"] = install_path
                    self._save_state(state)
                    
                    await self._finish_operation(True, f"CUDA {version} installed successfully")
                    
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

    def status(self) -> Dict[str, Any]:
        """Get CUDA installation status."""
        version = self._detect_installed_version()
        cuda_path = self._get_cuda_path()
        installed = version is not None and cuda_path is not None
        state = self._load_state()
        
        return {
            "installed": installed,
            "version": version,
            "cuda_path": cuda_path,
            "installed_at": state.get("installed_at"),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "last_error": self._last_error,
            "log_path": self._log_path,
            "available_versions": self.SUPPORTED_VERSIONS,
            "platform": self._get_platform(),
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

