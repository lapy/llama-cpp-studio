"""NVIDIA CUDA metadata discovery and download-URL resolution.

This module owns the parts of the CUDA toolkit installer that read NVIDIA's apt
repository metadata and download-archive pages: Ubuntu target detection, the
cached ``Packages.gz`` stanza parser, release-line discovery, and archive URL
resolution.  :class:`CudaMetadataSource` holds the per-installer caches; the
pure helpers above it can be tested directly without a network or an installer
instance.

``CUDAInstaller`` delegates to these helpers while keeping thin instance
methods (for example ``_get_repo_packages``) so existing monkeypatch targets
and callers keep working.
"""

import gzip
import os
import platform
import re
from typing import Callable, Dict, Optional, Sequence, Set, Tuple

import aiohttp

from backend.logging_config import get_logger

logger = get_logger(__name__)

# Supported CUDA release lines.  The concrete patch for each line is
# resolved from NVIDIA's repository metadata so a new 12.8.x release does
# not require a Studio release.  These values are also the offline fallback.
SUPPORTED_CUDA_VERSIONS = [
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

_CUDA_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?$")
MINIMUM_CUDA_VERSION = (11, 8, 0)
SUPPORTED_CUDA_MAJORS = {"11", "12", "13"}


def get_platform() -> Tuple[str, str]:
    """Get platform (os, arch) tuple."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if machine in ("x86_64", "amd64"):
        arch = "x86_64"
    else:
        arch = machine

    return system, arch


def repo_for_ubuntu_version(version: str) -> str:
    """Map a distro version (``26.04.1``) to NVIDIA's repository codename."""
    parts = str(version or "").strip().split(".")
    if len(parts) < 2:
        return "ubuntu2204"
    major_minor = f"{parts[0]}{parts[1]}"
    if major_minor >= "2604":
        return "ubuntu2604"
    if major_minor >= "2404":
        return "ubuntu2404"
    return "ubuntu2204"


def get_ubuntu_version() -> str:
    """Get Ubuntu version for NVIDIA repository URLs."""
    # Try to detect Ubuntu version from /etc/os-release
    try:
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("VERSION_ID="):
                        version = line.split("=")[1].strip().strip('"')
                        return repo_for_ubuntu_version(version)
    except Exception:
        pass

    # Default to ubuntu2404; callers on 26.04 normally read /etc/os-release.
    return "ubuntu2404"


def get_archive_target_version(ubuntu_version: str) -> str:
    """Get archive target version for CUDA runfile lookups.

    CUDA 13.0 publishes no Ubuntu 26.04 runfile, while the 26.04 repository
    itself exists (used for cuDNN/NCCL/TensorRT). The 24.04 runfile is the
    newest archive that covers these supported versions; the installer already
    passes ``--override`` so a 26.04 host can run it.
    """
    if ubuntu_version in ("ubuntu2604", "ubuntu2404"):
        return "24.04"
    return "22.04"


def cuda_version_parts(version: str) -> tuple[int, int, int]:
    """Return a comparable CUDA version, accepting major.minor[.patch]."""
    match = _CUDA_VERSION_PATTERN.fullmatch(str(version).strip())
    if not match:
        raise ValueError(
            f"Invalid CUDA version: {version}. Expected major.minor or major.minor.patch"
        )
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch or 0)


def cuda_release_line(version: str) -> str:
    major, minor, _patch = cuda_version_parts(version)
    return f"{major}.{minor}"


def version_key(version: str) -> tuple:
    """Create a sortable key for package version strings."""
    tokens = re.split(r"[^\w]+", version)
    key = []
    for token in tokens:
        if token.isdigit():
            key.append(int(token))
        elif token:
            key.append(token)
    return tuple(key)


def parse_packages(text: str) -> list:
    """Parse apt ``Packages`` stanzas into package/version/filename mappings."""
    packages: list = []
    current: dict = {}
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

    return packages


def resolve_available_versions(
    packages: list,
    supported_versions: Sequence[str] = SUPPORTED_CUDA_VERSIONS,
    minimum_version: tuple[int, int, int] = MINIMUM_CUDA_VERSION,
    supported_majors: Set[str] = SUPPORTED_CUDA_MAJORS,
) -> list[str]:
    """Resolve the latest published patch for each supported release line.

    NVIDIA's apt metadata publishes packages such as ``cuda-toolkit-12-8``
    with versions such as ``12.8.2-1``.  Prefer the newest concrete toolkit
    version from that metadata and retain the major/minor fallback for release
    lines unavailable for this Ubuntu repository (or when the repository
    cannot be reached).
    """
    discovered: dict[str, str] = {}

    for package in packages:
        name = str(package.get("Package", ""))
        match = re.fullmatch(r"cuda-toolkit-(\d+)-(\d+)", name)
        if not match:
            continue
        release_line = f"{int(match.group(1))}.{int(match.group(2))}"
        major = match.group(1)
        if major not in supported_majors:
            continue

        raw_version = str(package.get("Version", ""))
        version_match = re.match(r"^(\d+\.\d+(?:\.\d+)?)", raw_version)
        if not version_match:
            continue
        version = version_match.group(1)
        if cuda_release_line(version) != release_line:
            continue
        if cuda_version_parts(version) < minimum_version:
            continue
        current = discovered.get(release_line)
        if current is None or cuda_version_parts(version) > cuda_version_parts(current):
            discovered[release_line] = version

    versions = [discovered.pop(line, line) for line in supported_versions]
    versions.extend(discovered.values())
    return sorted(versions, key=cuda_version_parts, reverse=True)


def select_repo_package(
    packages: list,
    package_name: str,
    version_prefix: Optional[str] = None,
    version_contains: Optional[str] = None,
    *,
    key: Callable[[str], tuple] = version_key,
) -> Optional[Dict[str, str]]:
    """Select the best matching package from repo metadata."""
    candidates = [pkg for pkg in packages if pkg.get("Package") == package_name]
    if version_prefix:
        candidates = [
            pkg for pkg in candidates if pkg.get("Version", "").startswith(version_prefix)
        ]
    if version_contains:
        candidates = [pkg for pkg in candidates if version_contains in pkg.get("Version", "")]
    if not candidates:
        return None
    return max(candidates, key=lambda pkg: key(pkg.get("Version", "")))


def get_archive_url(version: str, target_version: str) -> str:
    """Get NVIDIA download archive URL for a CUDA version."""
    # Convert version like "12.8" to "12-8-0" for URL
    major_number, minor_number, patch_number = cuda_version_parts(version)
    major = str(major_number)
    minor = str(minor_number)
    patch = str(patch_number)
    version_slug = f"{major}-{minor}-{patch}"

    return (
        f"https://developer.nvidia.com/cuda-{version_slug}-download-archive"
        f"?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version={target_version}&target_type=runfile_local"
    )


def extract_download_url(html: str, version: str, target_version: str) -> str:
    """Find the Linux runfile URL in an NVIDIA archive page body."""
    # The page contains JSON data with download URLs
    # The JSON structure has keys like "Linux/x86_64/Ubuntu/24.04/runfile_local"
    # The URL is in the "details" field which contains HTML with href attributes
    json_key = f"Linux/x86_64/Ubuntu/{target_version}/runfile_local"

    # Pattern 1: Look for href in the details field (HTML may be escaped)
    # Match: "Linux/x86_64/Ubuntu/<version>/runfile_local":{..."details":"...href=\"URL\"..."}
    pattern1 = rf'"{
        re.escape(json_key)
    }"[^}}]*"details"[^"]*href[=:][\\"]*([^"\\s<>]+cuda_\d+\.\d+\.\d+_[^"\\s<>]+_linux\.run)'
    matches = re.findall(pattern1, html, re.IGNORECASE | re.DOTALL)

    if not matches:
        # Pattern 2: Look for href with escaped quotes (\u0022 or \")
        pattern2 = rf'"{
            re.escape(json_key)
        }"[^}}]*href[\\u0022=:]*([^"\\s<>]+cuda_\d+\.\d+\.\d+_[^"\\s<>]+_linux\.run)'
        matches = re.findall(pattern2, html, re.IGNORECASE | re.DOTALL)

    if not matches:
        # Pattern 3: Look for the filename field and construct URL
        pattern3 = rf'"{re.escape(json_key)}"[^}}]*"filename"[^"]*"([^"]+_linux\.run)"'
        filename_matches = re.findall(pattern3, html, re.IGNORECASE)
        if filename_matches:
            filename = filename_matches[0]
            version_full = ".".join(str(part) for part in cuda_version_parts(version))
            url = f"https://developer.download.nvidia.com/compute/cuda/{version_full}/local_installers/{filename}"
            matches = [url]

    if not matches:
        # Pattern 4: Fallback - look for any URL matching the pattern
        version_full = ".".join(str(part) for part in cuda_version_parts(version))
        version_escaped = version_full.replace(".", r"\.")
        pattern4 = rf'https://developer\.download\.nvidia\.com/compute/cuda/{version_escaped}/local_installers/cuda_{version_escaped}_[^"\'\s<>]+_linux\.run'
        matches = re.findall(pattern4, html, re.IGNORECASE)

    if matches:
        return matches[0]
    raise RuntimeError(f"Could not find download URL for CUDA {version} on archive page")


class CudaMetadataSource:
    """Cache holder and network entry points for NVIDIA CUDA metadata.

    The installer owns one instance so its repository and URL caches live for
    the installer's lifetime as before; tests may inject a fresh source.
    """

    def __init__(self) -> None:
        self._repo_cache: Dict[str, list] = {}  # Cache for NVIDIA repo packages
        self._url_cache: Dict[str, str] = {}  # Cache for dynamically fetched URLs

    async def get_repo_packages(self, ubuntu_version: str) -> list:
        """Fetch and cache NVIDIA CUDA repo package metadata."""
        if ubuntu_version in self._repo_cache:
            return self._repo_cache[ubuntu_version]

        base_url = (
            f"https://developer.download.nvidia.com/compute/cuda/repos/{ubuntu_version}/x86_64"
        )
        packages_url = f"{base_url}/Packages.gz"
        packages_plain_url = f"{base_url}/Packages"
        data = None

        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
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

        packages = parse_packages(data.decode("utf-8", errors="replace"))
        self._repo_cache[ubuntu_version] = packages
        return packages

    async def fetch_download_url(self, version: str, archive_url: str, target_version: str) -> str:
        """Fetch the actual download URL from NVIDIA's archive page."""
        # Check cache first
        cache_key = f"{version}_linux_x86_64"
        if cache_key in self._url_cache:
            return self._url_cache[cache_key]

        logger.info(f"Fetching CUDA {version} download URL from {archive_url}")

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    archive_url, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to fetch archive page: HTTP {response.status}")

                    html = await response.text()
                    url = extract_download_url(html, version, target_version)
                    # Cache it
                    self._url_cache[cache_key] = url
                    logger.info(f"Found CUDA {version} download URL: {url}")
                    return url

            except aiohttp.ClientError as e:
                raise RuntimeError(f"Failed to fetch archive page: {e}")
