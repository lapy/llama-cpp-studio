"""Choose and unpack an official audio.cpp Linux x64 release archive.

A CUDA archive is installed only when the host CUDA version is the same or
newer than the package and every host GPU is in the archive's compute
capabilities. Anything else builds that release tag from source.
"""

from __future__ import annotations

import hashlib
import os
import platform
import re
import shutil
import sys
import tarfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import httpx

from backend.logging_config import get_logger

logger = get_logger(__name__)

RELEASES_API = "https://api.github.com/repos/0xShug0/audio.cpp/releases"
GITHUB_ACCEPT = "application/vnd.github+json"

# The Ubuntu CUDA 12.8 "colab" archive is built with these architectures.
_KNOWN_CUDA_ARCHITECTURES = {
    "colab": (75, 80, 86, 89, 90),
}

_UBUNTU_ASSET = re.compile(
    r"^audio-.+-bin-ubuntu-x64-"
    r"(?P<kind>cpu-portable|vulkan-portable|cpu|vulkan|"
    r"cuda(?P<cuda>\d+\.\d+)(?:-(?P<profile>[A-Za-z0-9]+))?)"
    r"\.tar\.gz$"
)


@dataclass(frozen=True)
class HostTarget:
    cuda: Optional[Tuple[int, int]] = None
    sms: Tuple[int, ...] = ()
    has_nvidia: bool = False
    avx512: bool = False
    linux_x64: bool = False


@dataclass(frozen=True)
class PrebuiltPlan:
    asset_name: str
    url: str
    backend: str
    package_cuda: Optional[str]
    host_cuda: Optional[str]
    architectures: Tuple[int, ...]
    portable: bool
    reason: str
    cli_asset_name: str = ""
    cli_url: str = ""


def parse_cuda_version(value: Any) -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` from ``12.8`` or ``12.8.1``."""
    text = str(value or "").strip()
    if not text:
        return None
    match = re.match(r"^(\d+)\.(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def format_cuda_version(version: Optional[Tuple[int, int]]) -> str:
    if not version:
        return ""
    return f"{version[0]}.{version[1]}"


def parse_sm(value: Any) -> Optional[int]:
    text = str(value or "").strip().lower()
    if not text or text in {"unknown", "none"}:
        return None
    if "." in text:
        major, _, minor = text.partition(".")
        if major.isdigit() and minor[:1].isdigit():
            return int(major) * 10 + int(minor[0])
        return None
    if text.isdigit():
        return int(text)
    return None


def host_is_linux_x64() -> bool:
    machine = platform.machine().lower()
    return sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}


def host_has_avx512() -> bool:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if line.lower().startswith("flags") and "avx512f" in line.lower():
                    return True
    except OSError:
        return False
    return False


def _headers() -> Dict[str, str]:
    return {"Accept": GITHUB_ACCEPT, "User-Agent": "llama-cpp-studio"}


def _parse_asset(asset: Dict[str, Any]) -> Optional[dict]:
    name = str(asset.get("name") or "").strip()
    url = str(asset.get("browser_download_url") or "").strip()
    match = _UBUNTU_ASSET.match(name)
    if not match or not url:
        return None
    kind = match.group("kind")
    cuda = parse_cuda_version(match.group("cuda")) if match.group("cuda") else None
    profile = str(match.group("profile") or "").strip().lower()
    if kind.startswith("cuda"):
        backend = "cuda"
        portable = False
        architectures = _KNOWN_CUDA_ARCHITECTURES.get(profile, ())
    elif kind.startswith("vulkan"):
        backend = "vulkan"
        portable = kind.endswith("portable")
        architectures = ()
    else:
        backend = "cpu"
        portable = kind.endswith("portable")
        architectures = ()
    return {
        "name": name,
        "url": url,
        "backend": backend,
        "cuda": cuda,
        "profile": profile,
        "portable": portable,
        "architectures": architectures,
        "includes_cli": profile != "colab",
    }


def _covers_sms(architectures: Sequence[int], sms: Sequence[int]) -> bool:
    if not sms or not architectures:
        return True
    supported = set(architectures)
    return all(sm in supported for sm in sms)


def _cpu_cli_archive(parsed: Sequence[dict], host: HostTarget) -> Optional[dict]:
    """CPU archive from the same release, used when the CUDA bundle omits audiocpp_cli."""
    cpus = [item for item in parsed if item["backend"] == "cpu" and item["includes_cli"]]
    if not cpus:
        return None
    if host.avx512:
        native = [item for item in cpus if not item["portable"]]
        if native:
            return native[0]
    portable = [item for item in cpus if item["portable"]]
    return portable[0] if portable else cpus[0]


def _plan_from_asset(
    asset: dict,
    host: HostTarget,
    reason: str,
    *,
    cli_asset: Optional[dict] = None,
) -> PrebuiltPlan:
    return PrebuiltPlan(
        asset_name=asset["name"],
        url=asset["url"],
        backend=asset["backend"],
        package_cuda=format_cuda_version(asset["cuda"]) or None,
        host_cuda=format_cuda_version(host.cuda) or None,
        architectures=tuple(asset["architectures"]),
        portable=bool(asset["portable"]),
        reason=reason,
        cli_asset_name=str((cli_asset or {}).get("name") or ""),
        cli_url=str((cli_asset or {}).get("url") or ""),
    )


def select_prebuilt(assets: Iterable[Any], host: HostTarget) -> Optional[PrebuiltPlan]:
    """Pick a Linux x64 archive for this host, or None when source build is required."""
    if not host.linux_x64:
        return None
    parsed = [
        item
        for item in (_parse_asset(asset) for asset in assets if isinstance(asset, dict))
        if item
    ]
    wants_cuda = host.has_nvidia or host.cuda is not None
    if wants_cuda:
        if host.cuda is None:
            return None
        fitting = []
        for item in parsed:
            if item["backend"] != "cuda" or item["cuda"] is None:
                continue
            if host.cuda < item["cuda"]:
                continue
            if not _covers_sms(item["architectures"], host.sms):
                continue
            fitting.append(item)
        if not fitting:
            return None
        chosen = max(
            fitting,
            key=lambda item: (item["cuda"], -len(item["architectures"]), item["name"]),
        )
        host_text = format_cuda_version(host.cuda)
        package_text = format_cuda_version(chosen["cuda"])
        if host.cuda == chosen["cuda"]:
            reason = f"host CUDA {host_text} matches package CUDA {package_text}"
        else:
            reason = (
                f"host CUDA {host_text} is newer than package CUDA {package_text}"
            )
        cli_asset = None
        if not chosen["includes_cli"]:
            cli_asset = _cpu_cli_archive(parsed, host)
            if cli_asset is None:
                return None
            reason = f"{reason}; CLI from {cli_asset['name']}"
        return _plan_from_asset(chosen, host, reason, cli_asset=cli_asset)

    cpus = [item for item in parsed if item["backend"] == "cpu"]
    if not cpus:
        return None
    if host.avx512:
        native = [item for item in cpus if not item["portable"]]
        if native:
            return _plan_from_asset(native[0], host, "cpu prebuilt (AVX-512)")
    portable = [item for item in cpus if item["portable"]]
    chosen = portable[0] if portable else cpus[0]
    reason = "cpu-portable prebuilt" if chosen["portable"] else "cpu prebuilt"
    return _plan_from_asset(chosen, host, reason)


def prebuilt_skip_reason(assets: Iterable[Any], host: HostTarget) -> str:
    """Why ``select_prebuilt`` returned None."""
    if not host.linux_x64:
        return "Prebuilts are published for Linux x64, so this release will be built from source."
    parsed = [
        item
        for item in (_parse_asset(asset) for asset in assets if isinstance(asset, dict))
        if item
    ]
    cuda_assets = [item for item in parsed if item["backend"] == "cuda" and item["cuda"]]
    if host.has_nvidia or host.cuda is not None:
        if host.cuda is None:
            return "CUDA is not installed, so this release will be built from source."
        if not cuda_assets:
            return "This release has no Linux CUDA prebuilt, so it will be built from source."
        newer_host = [item for item in cuda_assets if host.cuda >= item["cuda"]]
        if not newer_host:
            package = max(item["cuda"] for item in cuda_assets)
            return (
                f"Host CUDA {format_cuda_version(host.cuda)} is older than "
                f"package CUDA {format_cuda_version(package)}, so this release "
                "will be built from source."
            )
        covered = [
            item for item in newer_host if _covers_sms(item["architectures"], host.sms)
        ]
        if not covered:
            return (
                "No published CUDA prebuilt covers this GPU, so this release "
                "will be built from source."
            )
        if all(not item["includes_cli"] for item in covered) and not _cpu_cli_archive(
            parsed, host
        ):
            return (
                "The CUDA prebuilt does not include audiocpp_cli and this "
                "release has no CPU archive to supply one, so it will be "
                "built from source."
            )
        return (
            "No published CUDA prebuilt covers this GPU, so this release "
            "will be built from source."
        )
    return "This release has no Linux CPU prebuilt, so it will be built from source."


async def detect_host_target() -> HostTarget:
    cuda = None
    try:
        from backend.cuda_installer import get_cuda_installer

        status = get_cuda_installer().status()
        if status.get("installed"):
            cuda = parse_cuda_version(status.get("version"))
    except Exception as exc:
        logger.debug("CUDA status unavailable for audio.cpp prebuilt pick: %s", exc)

    sms: List[int] = []
    has_nvidia = False
    try:
        from backend.gpu_detector import get_gpu_info

        info = await get_gpu_info()
    except Exception as exc:
        logger.debug("GPU info unavailable for audio.cpp prebuilt pick: %s", exc)
        info = None
    if isinstance(info, dict):
        vendor = str(info.get("vendor") or "").lower()
        has_nvidia = vendor == "nvidia" and int(info.get("device_count") or 0) > 0
        for gpu in info.get("gpus") or []:
            if not isinstance(gpu, dict):
                continue
            sm = parse_sm(gpu.get("compute_capability"))
            if sm:
                sms.append(sm)
    return HostTarget(
        cuda=cuda,
        sms=tuple(sorted(set(sms))),
        has_nvidia=has_nvidia or bool(sms),
        avx512=host_has_avx512(),
        linux_x64=host_is_linux_x64(),
    )


async def fetch_release(tag: str) -> Dict[str, Any]:
    url = f"{RELEASES_API}/tags/{tag}"
    async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
        response = await client.get(url, headers=_headers())
        response.raise_for_status()
        body = response.json() or {}
    if not isinstance(body, dict):
        raise RuntimeError(f"audio.cpp release {tag} was not a JSON object")
    return body


async def choose_release_prebuilt(tag: str) -> Tuple[Optional[PrebuiltPlan], str]:
    """Return the plan for ``tag``, and a skip reason when the plan is None."""
    host = await detect_host_target()
    if not host.linux_x64:
        return None, prebuilt_skip_reason([], host)
    release = await fetch_release(tag)
    assets = release.get("assets") or []
    plan = select_prebuilt(assets, host)
    if plan is None:
        return None, prebuilt_skip_reason(assets, host)
    return plan, ""


def _safe_extract_tar(archive: tarfile.TarFile, dest: str) -> None:
    dest_real = os.path.realpath(dest)
    for member in archive.getmembers():
        target = os.path.realpath(os.path.join(dest, member.name))
        if target != dest_real and not target.startswith(dest_real + os.sep):
            raise RuntimeError(f"Refusing to extract unsafe path {member.name}")
    archive.extractall(dest)


def find_named_binary(root: str, name: str) -> str:
    for dirpath, _, files in os.walk(root):
        if name in files:
            return os.path.join(dirpath, name)
    return ""


def _verify_checksums(root: str) -> None:
    checksum_path = ""
    for dirpath, _, files in os.walk(root):
        if "SHA256SUMS" in files:
            checksum_path = os.path.join(dirpath, "SHA256SUMS")
            break
    if not checksum_path:
        return
    base = os.path.dirname(checksum_path)
    with open(checksum_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            digest, _, filename = text.partition(" ")
            filename = filename.strip().lstrip("*")
            if not digest or not filename:
                continue
            target = os.path.join(base, filename)
            if not os.path.isfile(target):
                continue
            hasher = hashlib.sha256()
            with open(target, "rb") as payload:
                for chunk in iter(lambda: payload.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest().lower() != digest.lower():
                raise RuntimeError(f"Checksum mismatch for {filename}")


def extract_archive(archive_path: str, dest_dir: str) -> Dict[str, str]:
    os.makedirs(dest_dir, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        _safe_extract_tar(archive, dest_dir)
    _verify_checksums(dest_dir)
    server = find_named_binary(dest_dir, "audiocpp_server")
    cli = find_named_binary(dest_dir, "audiocpp_cli")
    if not server:
        raise RuntimeError("audiocpp_server was not found in the audio.cpp archive")
    for binary in (server, cli):
        if binary:
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
    return {"server_binary_path": server, "cli_binary_path": cli}


async def download_and_extract(url: str, dest_dir: str) -> Dict[str, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError("Refusing download from a non-HTTP URL")
    os.makedirs(dest_dir, exist_ok=True)
    archive_path = os.path.join(dest_dir, os.path.basename(parsed.path) or "audio.cpp.tar.gz")
    async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
        async with client.stream("GET", url, headers=_headers()) as response:
            response.raise_for_status()
            with open(archive_path, "wb") as handle:
                async for chunk in response.aiter_bytes():
                    handle.write(chunk)
    try:
        return extract_archive(archive_path, dest_dir)
    finally:
        if os.path.isfile(archive_path):
            os.remove(archive_path)


async def materialize_cli(binaries: Dict[str, str], plan: PrebuiltPlan, install_dir: str) -> str:
    """Return an audiocpp_cli path, taking it from the CPU archive when needed.

    The Ubuntu CUDA prebuilt ships audiocpp_server only. ``--inspect`` and
    ``--list-loaders`` exist on audiocpp_cli, so the same release's CPU archive
    supplies that binary and it is copied next to the CUDA server.
    """
    cli = str(binaries.get("cli_binary_path") or "")
    if cli and os.path.isfile(cli):
        return cli
    if not plan.cli_url:
        raise RuntimeError("audiocpp_cli was not in the audio.cpp archive")
    companion_dir = os.path.join(install_dir, ".cli-archive")
    try:
        companion = await download_and_extract(plan.cli_url, companion_dir)
        source = str(companion.get("cli_binary_path") or "")
        if not source or not os.path.isfile(source):
            raise RuntimeError(
                "audiocpp_cli was not in the CPU archive for this release"
            )
        dest = os.path.join(install_dir, "audiocpp_cli")
        shutil.copy2(source, dest)
        os.chmod(dest, os.stat(dest).st_mode | 0o111)
        return dest
    finally:
        from backend.utils.fs_ops import robust_rmtree

        if os.path.isdir(companion_dir):
            robust_rmtree(companion_dir)
