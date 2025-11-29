"""
CUDA Toolkit Installer

Handles downloading and installing CUDA Toolkit on Linux systems.
"""

import asyncio
import json
import os
import platform
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

    # CUDA download URLs - these point to NVIDIA's official download pages
    # We'll use the runfile installers for Linux
    CUDA_VERSIONS = {
        "13.0": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda_13.0.0_linux.run"
            }
        },
        "12.9": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_linux.run"
            }
        },
        "12.8": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_linux.run"
            }
        },
        "12.7": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.7.0/local_installers/cuda_12.7.0_linux.run"
            }
        },
        "12.6": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.70_linux.run"
            }
        },
        "12.5": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.5.0/local_installers/cuda_12.5.0_555.42_linux.run"
            }
        },
        "12.4": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run"
            }
        },
        "12.3": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.3.0/local_installers/cuda_12.3.0_549.85.05_linux.run"
            }
        },
        "12.2": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_537.13_linux.run"
            }
        },
        "12.1": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_531.14_linux.run"
            }
        },
        "12.0": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run"
            }
        },
        "11.9": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/11.9.0/local_installers/cuda_11.9.0_528.33_linux.run"
            }
        },
        "11.8": {
            "linux": {
                "x86_64": "https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_522.06_linux.run"
            }
        }
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
        
        data_root = os.path.abspath("data")
        log_path = log_path or os.path.join(data_root, "logs", "cuda_install.log")
        state_path = state_path or os.path.join(data_root, "configs", "cuda_installer.json")
        download_dir = download_dir or os.path.join(data_root, "temp", "cuda_installers")
        
        self._log_path = os.path.abspath(log_path)
        self._state_path = os.path.abspath(state_path)
        self._download_dir = os.path.abspath(download_dir)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        os.makedirs(self._download_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self._log_path), exist_ok=True)
        os.makedirs(os.path.dirname(self._state_path), exist_ok=True)

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
        """Detect installed CUDA version by checking nvcc."""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
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

    def _get_cuda_path(self) -> Optional[str]:
        """Get CUDA installation path."""
        env_path = os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME")
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Check common installation paths (Linux only)
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
                            await self._broadcast_progress({
                                "stage": "download",
                                "progress": progress,
                                "message": f"Downloading CUDA {version} installer... ({downloaded}/{total_size} bytes)",
                                "bytes_downloaded": downloaded,
                                "total_bytes": total_size,
                            })
                        
                        # Log progress every 10MB
                        if downloaded % (10 * 1024 * 1024) < 8192:
                            log_line = f"Downloaded {downloaded}/{total_size} bytes ({progress}%)\n"
                            with open(self._log_path, "a", encoding="utf-8") as log_file:
                                log_file.write(log_line)
                            await self._broadcast_log_line(f"Downloaded {downloaded // (1024*1024)} MB / {total_size // (1024*1024)} MB")

        await self._broadcast_log_line(f"Download completed: {installer_path}")
        await self._broadcast_progress({
            "stage": "download",
            "progress": 100,
            "message": "Download completed",
        })

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

        # Linux runfile installer flags:
        # --silent = silent mode
        # --toolkit = install only toolkit (not driver)
        # --override = override existing installation
        install_args = [
            installer_path,
            "--silent",
            "--toolkit",
            "--override",
        ]

        # Run installer with sudo if needed (user will need to provide password)
        # For now, we'll try without sudo first
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
            # If failed, might need sudo - inform user
            error_msg = (
                f"CUDA installer exited with code {process.returncode}. "
                "Linux installation may require sudo privileges. "
                "Please install CUDA manually or run this application with appropriate permissions."
            )
            raise RuntimeError(error_msg)

        await self._broadcast_progress({
            "stage": "install",
            "progress": 100,
            "message": "CUDA installation completed",
        })

    async def install(self, version: str = "12.6") -> Dict[str, Any]:
        """Install CUDA Toolkit."""
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another CUDA installer operation is already running")
            
            system, arch = self._get_platform()
            
            if system != "linux":
                raise RuntimeError(f"CUDA installation is only supported on Linux, not {system}")
            
            if version not in self.CUDA_VERSIONS:
                raise ValueError(f"Unsupported CUDA version: {version}")
            
            if arch not in self.CUDA_VERSIONS[version].get("linux", {}):
                raise ValueError(f"CUDA {version} is not available for Linux/{arch}")
            
            url = self.CUDA_VERSIONS[version]["linux"][arch]
            installer_filename = os.path.basename(url)
            installer_path = os.path.join(self._download_dir, installer_filename)
            
            await self._set_operation("install")

            async def _runner():
                try:
                    # Download installer
                    await self._download_installer(version, url, installer_path)
                    
                    # Install (Linux only)
                    await self._install_linux(installer_path, version)
                    
                    # Update state
                    state = self._load_state()
                    state["installed_version"] = version
                    state["installed_at"] = _utcnow()
                    state["cuda_path"] = self._get_cuda_path()
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
            "available_versions": list(self.CUDA_VERSIONS.keys()),
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

