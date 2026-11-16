"""CUDA Toolkit installer download pipeline and installer-cache accounting.

This module owns the large CUDA runfile download path (existing-file
verification, streaming download with progress updates, integrity checks) and
the cache inspection/cleanup helpers used by the installer's status and
maintenance APIs.

``CUDAInstaller`` keeps thin instance wrappers that supply its log path,
progress broadcasts and UTC clock, so callers and monkeypatch targets are
unchanged.  Logging messages, progress payloads, cancellation points, error
types and file handling are identical to the previous in-class implementation.
"""

import asyncio
import os
from typing import Any, Awaitable, Callable, Dict

import aiofiles
import aiohttp

from backend.logging_config import get_logger

logger = get_logger(__name__)


async def download_installer(
    version: str,
    url: str,
    installer_path: str,
    *,
    log_path: str,
    utcnow: Callable[[], str],
    broadcast_log: Callable[[str], Awaitable[None]],
    broadcast_progress: Callable[[Dict[str, Any]], Awaitable[None]],
) -> None:
    """Download CUDA installer with progress tracking."""
    # Check if installer already exists
    if os.path.exists(installer_path):
        file_size = os.path.getsize(installer_path)
        file_size_mb = file_size / (1024 * 1024)

        # Verify existing file is not corrupted (should be at least 100MB for CUDA installers)
        if file_size < 100 * 1024 * 1024:
            await broadcast_log(
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
                        await broadcast_log(
                            "Existing installer file is not a valid shell script, re-downloading..."
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
                                    expected_size = int(
                                        head_response.headers.get("Content-Length", 0)
                                    )

                                    if expected_size > 0:
                                        # Verify file size matches (with small tolerance)
                                        size_diff = abs(file_size - expected_size)
                                        if size_diff > 1024:  # Allow 1KB tolerance
                                            await broadcast_log(
                                                f"Existing installer file size mismatch: expected {expected_size / (1024 * 1024):.1f} MB, "
                                                f"got {file_size_mb:.1f} MB (difference: {size_diff} bytes). Re-downloading..."
                                            )
                                            try:
                                                os.remove(installer_path)
                                            except OSError:
                                                pass
                                        else:
                                            # File size matches, verify it's stable (not currently being written)
                                            # Brief pause to ensure file is fully written if being written
                                            await asyncio.sleep(0.2)
                                            new_size = os.path.getsize(installer_path)
                                            if new_size != file_size:
                                                await broadcast_log(
                                                    f"File size changed during verification (was {file_size_mb:.1f} MB, now {new_size / (1024 * 1024):.1f} MB), "
                                                    f"file may still be downloading. Re-downloading..."
                                                )
                                                try:
                                                    os.remove(installer_path)
                                                except OSError:
                                                    pass
                                            else:
                                                await broadcast_log(
                                                    f"Installer file already exists and verified: {installer_path} ({file_size_mb:.1f} MB)"
                                                )
                                                await broadcast_progress(
                                                    {
                                                        "stage": "download",
                                                        "progress": 100,
                                                        "message": f"Using existing installer file ({file_size_mb:.1f} MB)",
                                                    }
                                                )
                                                return
                                    else:
                                        # Couldn't get expected size, but file looks valid - use it
                                        await broadcast_log(
                                            f"Installer file already exists: {installer_path} ({file_size_mb:.1f} MB). "
                                            f"Could not verify size from server, but file appears valid."
                                        )
                                        await broadcast_progress(
                                            {
                                                "stage": "download",
                                                "progress": 100,
                                                "message": f"Using existing installer file ({file_size_mb:.1f} MB)",
                                            }
                                        )
                                        return
                        except Exception as size_check_error:
                            # If we can't verify size from server, but file looks valid, use it
                            await broadcast_log(
                                f"Could not verify file size from server: {size_check_error}. "
                                f"File appears valid, using existing file: {installer_path} ({file_size_mb:.1f} MB)"
                            )
                            await broadcast_progress(
                                {
                                    "stage": "download",
                                    "progress": 100,
                                    "message": f"Using existing installer file ({file_size_mb:.1f} MB)",
                                }
                            )
                            return
            except OSError as e:
                await broadcast_log(
                    f"Failed to verify existing installer file: {e}, re-downloading..."
                )
                try:
                    os.remove(installer_path)
                except OSError:
                    pass

    # Reset logging state for new download
    last_logged_percentage = -1

    log_header = f"[{utcnow()}] Downloading CUDA {version} installer from {url}\n"
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(log_header)

    await broadcast_log(f"Starting download of CUDA {version} installer...")
    await broadcast_progress(
        {
            "stage": "download",
            "progress": 0,
            "message": f"Downloading CUDA {version} installer...",
        }
    )

    # Configure timeout for large file downloads:
    # - total: 1 hour (3600s) for very large files and slow connections
    # - connect: 30s to establish connection
    # - sock_read: 5 minutes (300s) to allow for slow network during chunk reads
    timeout = aiohttp.ClientTimeout(
        total=3600,  # 1 hour total timeout
        connect=30,  # 30 seconds to connect
        sock_read=300,  # 5 minutes per read operation
    )

    downloaded = 0
    total_size = 0
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))

                async with aiofiles.open(installer_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)
                        downloaded += len(chunk)

                    if total_size > 0:
                        progress = int((downloaded / total_size) * 100)
                        # Format sizes in MB
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        await broadcast_progress(
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
                        if progress != last_logged_percentage and progress in [
                            10,
                            25,
                            50,
                            75,
                            90,
                            100,
                        ]:
                            should_log = True
                            last_logged_percentage = progress

                    if should_log:
                        downloaded_mb = downloaded / (1024 * 1024)
                        total_mb = total_size / (1024 * 1024)
                        log_line = (
                            f"Downloaded {downloaded_mb:.1f}/{total_mb:.1f} MB ({progress}%)\n"
                        )
                        with open(log_path, "a", encoding="utf-8") as log_file:
                            log_file.write(log_line)
                        await broadcast_log(
                            f"Downloaded {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({progress}%)"
                        )

            # File is automatically flushed when the context manager exits
    except asyncio.TimeoutError as e:
        # Clean up partial download on timeout
        if os.path.exists(installer_path):
            try:
                os.remove(installer_path)
            except OSError:
                pass
        downloaded_mb = downloaded / (1024 * 1024) if downloaded > 0 else 0
        total_mb = total_size / (1024 * 1024) if total_size > 0 else 0
        error_msg = (
            f"Download timeout: Failed to download CUDA {version} installer. "
            f"Downloaded {downloaded_mb:.1f} MB of {total_mb:.1f} MB. "
            f"This may be due to a slow network connection. Please try again."
        )
        await broadcast_log(error_msg)
        raise RuntimeError(error_msg) from e
    except aiohttp.ClientError as e:
        # Clean up partial download on client error
        if os.path.exists(installer_path):
            try:
                os.remove(installer_path)
            except OSError:
                pass
        error_msg = (
            f"Network error while downloading CUDA {version} installer: {e}. "
            f"Please check your network connection and try again."
        )
        await broadcast_log(error_msg)
        raise RuntimeError(error_msg) from e

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

    await broadcast_log(
        f"Download completed and verified: {installer_path} ({actual_size / (1024 * 1024):.1f} MB)"
    )
    await broadcast_progress(
        {
            "stage": "download",
            "progress": 100,
            "message": "Download completed and verified",
        }
    )


def installer_cache_status(download_dir: str) -> Dict[str, Any]:
    """Summarize the downloaded CUDA runfiles kept in the cache directory."""
    total_bytes = 0
    file_count = 0
    try:
        for root, _dirnames, filenames in os.walk(download_dir, followlinks=False):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                try:
                    if os.path.islink(file_path) or not os.path.isfile(file_path):
                        continue
                    total_bytes += os.path.getsize(file_path)
                except OSError:
                    continue
                file_count += 1
    except OSError:
        pass
    return {
        "bytes": total_bytes,
        "files": file_count,
        "directory": download_dir,
    }


def cleanup_installers(download_dir: str) -> Dict[str, Any]:
    """Delete cached CUDA installer downloads, leaving installed toolkits alone.

    Only regular files under the download directory are removed. Directory
    entries and symlinks are skipped so cleanup can never follow a link out
    of the cache, and installed toolkits live under a different root.
    """

    freed_bytes = 0
    removed = 0
    for root, _dirnames, filenames in os.walk(download_dir, followlinks=False):
        for filename in filenames:
            file_path = os.path.join(root, filename)
            try:
                if os.path.islink(file_path) or not os.path.isfile(file_path):
                    continue
                size = os.path.getsize(file_path)
                os.remove(file_path)
            except OSError as exc:
                logger.warning(f"Failed to remove CUDA installer {file_path}: {exc}")
                continue
            freed_bytes += size
            removed += 1

    # Best-effort prune of now-empty subdirectories; the cache root itself
    # stays so a later download does not need to recreate it.
    for root, dirnames, _filenames in os.walk(download_dir, topdown=False):
        for dirname in dirnames:
            dir_path = os.path.join(root, dirname)
            if os.path.islink(dir_path):
                continue
            try:
                os.rmdir(dir_path)
            except OSError:
                continue

    return {"freed_bytes": freed_bytes, "removed": removed}
