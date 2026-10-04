"""Install Unsloth's published llama-server mix builds.

These are upstream llama.cpp nightlies plus Unsloth's pinned patch set, shipped
as ready ``llama-server`` archives. Fork ``master`` is not the mix; only the
release assets are.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from backend.data_store import get_store, studio_data_dir
from backend.engines.unsloth_llama.prebuilt import (
    CHECKSUM_ASSET_NAME,
    MANIFEST_ASSET_NAME,
    MAX_RELEASE_WALKBACK,
    HostTarget,
    ReleasePlan,
    find_named_asset,
    host_target_from_gpu_info,
    linux_x64_asset_name,
    parse_cuda_major,
    select_asset_flavor,
    select_release_asset,
)
from backend.logging_config import get_logger
from backend.operations.cancellable import CancellableOperationManager
from backend.proxy.llama_swap.manager import mark_swap_config_stale


logger = get_logger(__name__)

ENGINE_ID = "unsloth_llama"
REPOSITORY_SOURCE = "Unsloth llama.cpp"
UNSLOTH_LLAMA_REPO = "https://github.com/unslothai/llama.cpp.git"
RELEASES_API = "https://api.github.com/repos/unslothai/llama.cpp/releases"
GITHUB_ACCEPT = "application/vnd.github+json"
GITHUB_TIMEOUT = 30.0

_manager_instance: Optional["UnslothLlamaInstaller"] = None


def get_unsloth_llama_manager() -> "UnslothLlamaInstaller":
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = UnslothLlamaInstaller()
    return _manager_instance


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_checksum_asset(assets: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    return find_named_asset(assets, CHECKSUM_ASSET_NAME)


def _digest_from_entry(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        digest = value.get("sha256") or value.get("sha256sum") or value.get("hash")
        if digest:
            return str(digest).strip()
    return None


def expected_sha256(manifest: Any, asset_name: str) -> str:
    name = str(asset_name or "").strip()
    if not name:
        raise ValueError("Asset name is required for checksum lookup")
    if isinstance(manifest, dict):
        digest = _digest_from_entry(manifest.get(name))
        if digest:
            return digest
        artifacts = manifest.get("artifacts")
        if isinstance(artifacts, dict):
            digest = _digest_from_entry(artifacts.get(name))
            if digest:
                return digest
            nested = expected_sha256_from_list(artifacts.get("files"), name)
            if nested:
                return nested
        for key in ("files", "assets", "checksums"):
            nested = expected_sha256_from_list(manifest.get(key), name)
            if nested:
                return nested
    if isinstance(manifest, list):
        nested = expected_sha256_from_list(manifest, name)
        if nested:
            return nested
    raise ValueError(f"No sha256 entry for {name} in {CHECKSUM_ASSET_NAME}")


def expected_sha256_from_list(items: Any, asset_name: str) -> Optional[str]:
    if not isinstance(items, list):
        return None
    for item in items:
        if not isinstance(item, dict):
            continue
        if str(item.get("name") or item.get("filename") or "") != asset_name:
            continue
        digest = item.get("sha256") or item.get("sha256sum") or item.get("hash")
        if digest:
            return str(digest).strip()
    return None


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: str, expected: str) -> None:
    actual = sha256_file(path)
    want = str(expected or "").strip().lower()
    if actual.lower() != want:
        raise RuntimeError(
            f"Checksum mismatch for {os.path.basename(path)}: expected {want}, got {actual}"
        )


def find_llama_server(root: str) -> Optional[str]:
    for dirpath, _, filenames in os.walk(root):
        if "llama-server" in filenames:
            return os.path.join(dirpath, "llama-server")
    return None


def host_is_linux_x64() -> bool:
    machine = platform.machine().lower()
    return sys.platform.startswith("linux") and machine in {"x86_64", "amd64"}


def _safe_extract_tar(archive: tarfile.TarFile, dest: str) -> None:
    dest_real = os.path.realpath(dest)
    for member in archive.getmembers():
        target = os.path.realpath(os.path.join(dest, member.name))
        if target != dest_real and not target.startswith(dest_real + os.sep):
            raise RuntimeError(f"Refusing to extract unsafe path {member.name}")
    archive.extractall(dest)


class UnslothLlamaInstaller(CancellableOperationManager):
    MANAGER_NAME = "unsloth_llama"

    def __init__(self, *, base_dir: Optional[str] = None) -> None:
        super().__init__()
        self._root_dir = os.path.abspath(base_dir or os.path.join(studio_data_dir(), "unsloth-llama"))
        os.makedirs(self._root_dir, exist_ok=True)

    @property
    def root_dir(self) -> str:
        return self._root_dir

    def _cuda_major(self) -> Optional[int]:
        try:
            from backend.cuda_installer import get_cuda_installer

            status = get_cuda_installer().status()
        except Exception as exc:
            logger.debug("CUDA status unavailable for Unsloth asset pick: %s", exc)
            return None
        if not status.get("installed"):
            return None
        return parse_cuda_major(status.get("version"))

    async def _host_target(self) -> HostTarget:
        gpu_info: Optional[Dict[str, Any]] = None
        try:
            from backend.gpu_detector import get_gpu_info

            gpu_info = await get_gpu_info()
        except Exception as exc:
            logger.debug("GPU info unavailable for Unsloth asset pick: %s", exc)
        return host_target_from_gpu_info(gpu_info, cuda_major=self._cuda_major())

    def status(self) -> Dict[str, Any]:
        active = get_store().get_active_engine_version(ENGINE_ID) or {}
        binary = str(active.get("binary_path") or "")
        return {
            "engine": ENGINE_ID,
            "installed": bool(binary and os.path.isfile(binary)),
            "version": active.get("version"),
            "binary_path": binary or None,
            "install_dir": active.get("install_dir"),
            "install_type": active.get("install_type") or active.get("type"),
            "asset_name": (active.get("build_config") or {}).get("asset_name"),
            "flavor": (active.get("build_config") or {}).get("flavor"),
            "coverage_class": (active.get("build_config") or {}).get("coverage_class"),
            "upstream_tag": (active.get("build_config") or {}).get("upstream_tag"),
            "cuda_version": active.get("cuda_version"),
            "operation": self._operation,
            "operation_started_at": self._operation_started_at,
            "progress_task_id": self._progress_task_id,
            "last_error": self._last_error,
        }

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": GITHUB_ACCEPT,
            "User-Agent": "llama-cpp-studio",
        }

    async def list_published_releases(
        self, *, max_items: int = MAX_RELEASE_WALKBACK
    ) -> List[Dict[str, Any]]:
        from backend.engines.unsloth_llama.prebuilt import sort_releases_by_publish_time

        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(
                RELEASES_API,
                headers=self._headers(),
                params={"per_page": 30},
            )
            response.raise_for_status()
            payload = response.json() or []
        return sort_releases_by_publish_time(payload)[: max(1, max_items)]

    async def fetch_release(self, tag_name: Optional[str] = None) -> Dict[str, Any]:
        tag = str(tag_name or "").strip()
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True) as client:
            if tag:
                response = await client.get(
                    f"{RELEASES_API}/tags/{tag}", headers=self._headers()
                )
                response.raise_for_status()
                payload = response.json()
            else:
                listing = await self.list_published_releases()
                payload = listing[0] if listing else None
        if not isinstance(payload, dict) or not payload.get("tag_name"):
            raise RuntimeError("Unsloth llama.cpp release metadata was empty")
        return payload

    async def _fetch_json(self, url: str) -> Any:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError("Refusing download from a non-HTTP URL")
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()

    async def latest_published_tag(self) -> Optional[str]:
        releases = await self.list_published_releases(max_items=1)
        if not releases:
            return None
        return str(releases[0].get("tag_name") or "").strip() or None

    async def _resolve_plan(
        self, tag_name: Optional[str], host: HostTarget
    ) -> Tuple[Dict[str, Any], ReleasePlan, Any]:
        from backend.engines.unsloth_llama.prebuilt import (
            fallback_plan_from_filenames,
            plan_from_manifest,
        )

        pinned = bool(str(tag_name or "").strip())
        if pinned:
            releases = [await self.fetch_release(tag_name)]
        else:
            releases = await self.list_published_releases()
        if not releases:
            raise RuntimeError("No published Unsloth llama.cpp releases were found")
        skipped: List[str] = []
        last_error = "No published Unsloth llama.cpp releases were usable"
        for release in releases:
            tag = str(release.get("tag_name") or "").strip()
            assets = [item for item in (release.get("assets") or []) if isinstance(item, dict)]
            manifest = None
            manifest_asset = find_named_asset(assets, MANIFEST_ASSET_NAME)
            if manifest_asset:
                try:
                    manifest = await self._fetch_json(
                        str(manifest_asset["browser_download_url"])
                    )
                except Exception as exc:
                    logger.warning("Unsloth prebuilt manifest for %s: %s", tag, exc)
            plan = None
            if manifest is not None:
                plan = plan_from_manifest(
                    tag=tag,
                    assets=assets,
                    manifest=manifest,
                    host=host,
                    skipped_newer=skipped,
                )
            if plan is None:
                plan = fallback_plan_from_filenames(
                    tag=tag,
                    assets=assets,
                    host=host,
                    skipped_newer=skipped,
                )
            if plan is not None:
                return release, plan, manifest
            last_error = (
                f"No Linux x64 bundle in {tag} covers this host "
                f"(CUDA {host.cuda_major or 'none'}, SM {','.join(str(sm) for sm in host.sms) or 'unknown'})"
            )
            if pinned:
                raise RuntimeError(last_error)
            skipped.append(tag)
        raise RuntimeError(last_error)

    async def _download_file(self, url: str, dest: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise RuntimeError("Refusing download from a non-HTTP URL")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            async with client.stream("GET", url, headers=self._headers()) as response:
                response.raise_for_status()
                with open(dest, "wb") as handle:
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)

    def _unique_version_name(self, tag: str) -> str:
        existing = {
            str(row.get("version") or "")
            for row in get_store().get_engine_versions(ENGINE_ID) or []
        }
        if tag not in existing:
            return tag
        return f"{tag}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    def _register_pending(self, version_name: str, extra: Dict[str, Any]) -> None:
        from backend.engines.lifecycle import mark_engine_version_building

        mark_engine_version_building(
            get_store(),
            ENGINE_ID,
            {
                "version": version_name,
                "type": "release",
                "install_type": "release",
                "repository_source": REPOSITORY_SOURCE,
                "source_repo": UNSLOTH_LLAMA_REPO,
                "installed_at": utcnow(),
                **extra,
            },
            task_id=self._progress_task_id,
        )

    def _fail_pending(self, version_name: str, error: str, extra: Optional[Dict[str, Any]] = None) -> None:
        from backend.engines.lifecycle import mark_engine_version_failed

        mark_engine_version_failed(
            get_store(),
            ENGINE_ID,
            version_name,
            error=str(error),
            extra=extra or {},
        )

    async def install_release(self, tag_name: Optional[str] = None) -> Dict[str, Any]:
        async with self._lock:
            if self._operation:
                raise RuntimeError("Another Unsloth llama.cpp operation is already running")
            if not host_is_linux_x64():
                raise RuntimeError(
                    "Unsloth llama.cpp prebuilts are installed for Linux x64 only"
                )
            await self._start_operation("install", {"engine": ENGINE_ID})
            pending = self._unique_version_name(str(tag_name or "pending").strip() or "pending")
            install_dir = os.path.join(self._root_dir, pending)
            extra = {
                "install_dir": install_dir,
                "repository_source": REPOSITORY_SOURCE,
            }
            self._register_pending(pending, extra)

            async def runner() -> None:
                current = pending
                current_extra = dict(extra)
                try:
                    await self._update_progress_task(8, "Fetching Unsloth llama.cpp release")
                    host = await self._host_target()
                    release, plan, published_manifest = await self._resolve_plan(tag_name, host)
                    tag = plan.tag
                    flavor = plan.flavor
                    cuda_major = host.cuda_major
                    assets = [
                        asset
                        for asset in (release.get("assets") or [])
                        if isinstance(asset, dict)
                    ]
                    asset = find_named_asset(assets, plan.asset_name)
                    if not asset:
                        raise RuntimeError(
                            f"No Linux x64 {flavor} archive in Unsloth release {tag}"
                        )
                    checksum_asset = find_checksum_asset(assets)
                    if not checksum_asset:
                        raise RuntimeError(
                            f"{CHECKSUM_ASSET_NAME} is missing from Unsloth release {tag}"
                        )
                    if plan.skipped_newer:
                        logger.info(
                            "Unsloth release walk-back: skipped %s, using %s (%s)",
                            ", ".join(plan.skipped_newer),
                            tag,
                            plan.reason,
                        )
                    store = get_store()
                    if current == tag or current.startswith(f"{tag}-"):
                        final_name = current
                    else:
                        final_name = self._unique_version_name(tag)
                    if final_name != current:
                        store.delete_engine_version(ENGINE_ID, current)
                        current = final_name
                        current_extra["install_dir"] = os.path.join(self._root_dir, current)
                        self._register_pending(current, current_extra)
                    dest_dir = current_extra["install_dir"]
                    if os.path.exists(dest_dir):
                        shutil.rmtree(dest_dir)
                    os.makedirs(dest_dir, exist_ok=True)

                    checksum_path = os.path.join(dest_dir, CHECKSUM_ASSET_NAME)
                    archive_path = os.path.join(dest_dir, str(asset["name"]))
                    await self._update_progress_task(20, "Downloading checksum manifest")
                    await self._download_file(
                        str(checksum_asset["browser_download_url"]), checksum_path
                    )
                    if isinstance(published_manifest, dict):
                        with open(
                            os.path.join(dest_dir, MANIFEST_ASSET_NAME),
                            "w",
                            encoding="utf-8",
                        ) as handle:
                            json.dump(published_manifest, handle)
                    with open(checksum_path, "r", encoding="utf-8") as handle:
                        manifest = json.load(handle)
                    expected = expected_sha256(manifest, str(asset["name"]))
                    await self._update_progress_task(40, f"Downloading {asset['name']}")
                    await self._download_file(str(asset["browser_download_url"]), archive_path)
                    await self._update_progress_task(72, "Verifying archive checksum")
                    verify_sha256(archive_path, expected)
                    await self._update_progress_task(82, "Extracting llama-server")
                    with tarfile.open(archive_path, "r:gz") as archive:
                        _safe_extract_tar(archive, dest_dir)
                    os.remove(archive_path)
                    binary = find_llama_server(dest_dir)
                    if not binary:
                        raise RuntimeError("llama-server was not found in the Unsloth archive")
                    os.chmod(binary, 0o755)
                    from backend.engines.lifecycle import mark_engine_version_ready

                    payload = {
                        "version": current,
                        "type": "release",
                        "install_type": "release",
                        "binary_path": binary,
                        "install_dir": dest_dir,
                        "repository_source": REPOSITORY_SOURCE,
                        "source_repo": UNSLOTH_LLAMA_REPO,
                        "source_ref": tag,
                        "source_ref_type": "tag",
                        "cuda_version": None if cuda_major is None else str(cuda_major),
                        "installed_at": utcnow(),
                        "build_config": {
                            "tag_name": tag,
                            "asset_name": asset.get("name"),
                            "flavor": flavor,
                            "coverage_class": plan.artifact.coverage_class,
                            "runtime_line": plan.artifact.runtime_line,
                            "upstream_tag": plan.upstream_tag,
                            "selection_reason": plan.reason,
                            "host_sms": list(host.sms),
                            "skipped_newer": list(plan.skipped_newer),
                            "release_url": release.get("html_url"),
                        },
                    }
                    mark_engine_version_ready(store, ENGINE_ID, payload)
                    store.set_active_engine_version(ENGINE_ID, current)
                    try:
                        from backend.engines.scan.scanner import scan_engine_version

                        await asyncio.to_thread(scan_engine_version, store, ENGINE_ID, payload)
                    except Exception as scan_err:
                        logger.warning(
                            "CLI param scan after Unsloth install %s: %s", current, scan_err
                        )
                    mark_swap_config_stale()
                    await self._finish_operation(True, f"Installed Unsloth llama.cpp {current}")
                except asyncio.CancelledError:
                    self._fail_pending(current, "Operation cancelled by user", current_extra)
                    raise
                except Exception as exc:
                    self._last_error = str(exc)
                    self._fail_pending(current, str(exc), current_extra)
                    await self._finish_operation(False, str(exc))

            self._create_task(runner())
            return self._started_response("Unsloth llama.cpp installation started")

    async def _start_operation(self, operation: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        return await self._begin_operation(
            operation,
            "Install Unsloth llama.cpp",
            {"engine": ENGINE_ID, **(metadata or {})},
        )
