"""Generic GitHub-release native host.

Selections and asset policy come from the engine definition's
``release_assets`` block; the manifest algorithms remain shared helpers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import sys
import tarfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

import httpx

from backend.data_store import get_store
from backend.engines.python.native import GenericNativeInstaller
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

logger = get_logger(__name__)

GITHUB_ACCEPT = "application/vnd.github+json"
GITHUB_TIMEOUT = 30.0

def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_checksum_asset(
    assets: List[Dict[str, Any]], name: str = CHECKSUM_ASSET_NAME
) -> Optional[Dict[str, Any]]:
    return find_named_asset(assets, name)


def _digest_from_entry(value: Any) -> Optional[str]:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        digest = value.get("sha256") or value.get("sha256sum") or value.get("hash")
        if digest:
            return str(digest).strip()
    return None


def expected_sha256(
    manifest: Any, asset_name: str, checksum_name: str = CHECKSUM_ASSET_NAME
) -> str:
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
    raise ValueError(f"No sha256 entry for {name} in {checksum_name}")


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


class GenericReleaseInstaller(GenericNativeInstaller):
    """Release-archive host configured by ``definition.release_assets``."""

    def __init__(self, definition, *, base_dir: Optional[str] = None) -> None:
        self.engine_id = definition.id
        self._repository_source = definition.repository_source
        policy = dict(definition.release_assets)
        super().__init__(
            definition,
            install_root=base_dir or None,
        )
        self._repository = str(policy.get("repository") or "")
        self._repository_url = str(policy.get("repository_url") or "")
        self._manifest_asset = str(policy.get("manifest_asset") or MANIFEST_ASSET_NAME)
        self._checksum_asset = str(policy.get("checksum_asset") or CHECKSUM_ASSET_NAME)
        self._max_walkback = int(policy.get("max_walkback") or MAX_RELEASE_WALKBACK)
        self._releases_api = f"https://api.github.com/repos/{self._repository}/releases"
        self._cached_resolution: Optional[Tuple[Any, Any, Any, Any]] = None
        self._last_resolution: Dict[str, Any] = {}

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
        active = get_store().get_active_engine_version(self.engine_id) or {}
        binary = str(active.get("binary_path") or "")
        return {
            "engine": self.engine_id,
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
        self, *, max_items: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        from backend.engines.unsloth_llama.prebuilt import sort_releases_by_publish_time

        limit = self._max_walkback if max_items is None else max_items
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True) as client:
            response = await client.get(
                self._releases_api,
                headers=self._headers(),
                params={"per_page": 30},
            )
            response.raise_for_status()
            payload = response.json() or []
        return sort_releases_by_publish_time(payload)[: max(1, limit)]

    async def fetch_release(self, tag_name: Optional[str] = None) -> Dict[str, Any]:
        tag = str(tag_name or "").strip()
        async with httpx.AsyncClient(timeout=GITHUB_TIMEOUT, follow_redirects=True) as client:
            if tag:
                response = await client.get(
                    f"{self._releases_api}/tags/{tag}", headers=self._headers()
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
            manifest_asset = find_named_asset(assets, self._manifest_asset)
            if manifest_asset:
                try:
                    manifest = await self._fetch_json(str(manifest_asset["browser_download_url"]))
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
            for row in get_store().get_engine_versions(self.engine_id) or []
        }
        if tag not in existing:
            return tag
        return f"{tag}-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    def _register_pending(self, version_name: str, extra: Dict[str, Any]) -> None:
        from backend.engines.lifecycle import mark_engine_version_building

        mark_engine_version_building(
            get_store(),
            self.engine_id,
            {
                "version": version_name,
                "type": "release",
                "install_type": "release",
                "repository_source": self._repository_source,
                "source_repo": self._repository_url,
                "installed_at": utcnow(),
                **extra,
            },
            task_id=self._progress_task_id,
        )

    def _fail_pending(
        self, version_name: str, error: str, extra: Optional[Dict[str, Any]] = None
    ) -> None:
        from backend.engines.lifecycle import mark_engine_version_failed

        mark_engine_version_failed(
            get_store(),
            self.engine_id,
            version_name,
            error=str(error),
            extra=extra or {},
        )

    def _register_pending_install(self, version_name: str, install_dir: str, task_id: str) -> None:
        self._register_pending(
            version_name,
            {"install_dir": install_dir, "repository_source": self._repository_source},
        )

    def _install_failed(self, version_name: str, error: str, install_dir: str) -> None:
        self._fail_pending(version_name, error, {"install_dir": install_dir})

    async def _resolve_latest_tag(self) -> Optional[str]:
        host = await self._host_target()
        release, plan_res, manifest = await self._resolve_plan(None, host)
        self._cached_resolution = (release, plan_res, manifest, host)
        return plan_res.tag

    async def install_release(self, tag_name: Optional[str] = None) -> Dict[str, Any]:
        if not host_is_linux_x64():
            raise RuntimeError("Unsloth llama.cpp prebuilts are installed for Linux x64 only")
        requested = str(tag_name or "").strip() or None
        return await self.start_release_install(
            version=requested,
            version_resolver=None if requested else self._resolve_latest_tag,
            extra_context={"input.tag_name": requested},
        )

    def _extra_plan_handlers(self) -> Dict[str, Any]:
        return {
            "prereqs.resolve_release_assets": self._resolve_release_assets,
            "artifact.read_checksum": self._read_checksum,
        }

    async def _resolve_release_assets(self, step, requirements, context):
        requested = requirements.get("input.tag_name")
        await self._update_progress_task(8, "Fetching Unsloth llama.cpp release")
        cached = self._cached_resolution
        self._cached_resolution = None
        if cached is not None and not str(requested or "").strip():
            release, plan_res, published_manifest, host = cached
        else:
            host = await self._host_target()
            release, plan_res, published_manifest = await self._resolve_plan(
                str(requested or "").strip() or None, host
            )
        tag = plan_res.tag
        assets = [asset for asset in (release.get("assets") or []) if isinstance(asset, dict)]
        asset = find_named_asset(assets, plan_res.asset_name)
        if not asset:
            raise RuntimeError(f"No Linux x64 {plan_res.flavor} archive in Unsloth release {tag}")
        checksum_asset = find_checksum_asset(assets, self._checksum_asset)
        if not checksum_asset:
            raise RuntimeError(f"{self._checksum_asset} is missing from Unsloth release {tag}")
        if plan_res.skipped_newer:
            logger.info(
                "Unsloth release walk-back: skipped %s, using %s (%s)",
                ", ".join(plan_res.skipped_newer),
                tag,
                plan_res.reason,
            )
        current = str(context.get("input.version_name") or "")
        install_dir = str(context.get("input.install_dir") or "")
        resolution = {
            "release": release,
            "plan": plan_res,
            "manifest": published_manifest,
            "host": host,
            "asset_name": str(asset["name"]),
        }
        self._last_resolution = resolution
        return {
            "resolution": resolution,
            "current": current,
            "install_dir": install_dir,
            "checksum_path": os.path.join(install_dir, self._checksum_asset),
            "checksum_url": str(checksum_asset["browser_download_url"]),
            "archive_path": os.path.join(install_dir, str(asset["name"])),
            "archive_url": str(asset["browser_download_url"]),
            "archive_download_label": f"Downloading {asset['name']}",
            "input.version_name": current,
            "input.install_dir": install_dir,
            "input.install_type": "release",
            "input.requested_version": str(requested or ""),
            "input.source_repo": self._repository_url,
            "input.source_ref": plan_res.tag,
            "input.source_ref_type": "tag",
            "input.auto_activate": True,
            "unsloth.resolution": resolution,
        }

    async def _read_checksum(self, step, requirements, context):
        resolution = requirements.get("resolution") or {}
        checksum_path = str(requirements.get("checksum_path") or "")
        dest_dir = os.path.dirname(checksum_path)
        published_manifest = resolution.get("manifest")
        if isinstance(published_manifest, dict):
            with open(
                os.path.join(dest_dir, self._manifest_asset), "w", encoding="utf-8"
            ) as handle:
                json.dump(published_manifest, handle)
        with open(checksum_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        expected = expected_sha256(
            manifest, str(resolution.get("asset_name") or ""), self._checksum_asset
        )
        return {"expected_sha256": expected}

    def _row_extras(self, context: Mapping[str, Any]) -> Dict[str, Any]:
        resolution = context.get("unsloth.resolution") or self._last_resolution or {}
        plan_res = resolution.get("plan")
        host = resolution.get("host")
        if plan_res is None:
            return {}
        cuda_major = getattr(host, "cuda_major", None)
        return {
            "source_ref": plan_res.tag,
            "source_ref_type": "tag",
            "cuda_version": None if cuda_major is None else str(cuda_major),
            "build_config": {
                "tag_name": plan_res.tag,
                "asset_name": resolution.get("asset_name"),
                "flavor": plan_res.flavor,
                "coverage_class": plan_res.artifact.coverage_class,
                "runtime_line": plan_res.artifact.runtime_line,
                "upstream_tag": plan_res.upstream_tag,
                "selection_reason": plan_res.reason,
                "host_sms": list(host.sms),
                "skipped_newer": list(plan_res.skipped_newer),
                "release_url": (resolution.get("release") or {}).get("html_url"),
            },
        }

    def _record_published_snapshot(self, version_name: str, resolved: Dict[str, Any]) -> None:
        """Record the version-associated install snapshot (best-effort)."""
        resolution = self._last_resolution or {}
        plan_res = resolution.get("plan")
        if plan_res is not None:
            resolved = {
                **resolved,
                "release_tag": plan_res.tag,
                "asset_name": resolution.get("asset_name"),
            }
        try:
            from backend.engines.definitions.loader import load_default_engine_definitions

            definition = load_default_engine_definitions().get(self.engine_id)
            digest = str(getattr(definition, "definition_digest", "") or "")
            if not digest:
                return
            get_store().record_install_snapshot(
                self.engine_id,
                definition_digest=digest,
                recipe_version=f"schema-v{getattr(definition, 'schema_version', 2)}",
                resolved_inputs=resolved,
                version=version_name,
                phase="published",
            )
        except Exception:  # noqa: BLE001 — bookkeeping must not fail installs
            logger.debug("install snapshot publish failed for %s", self.engine_id, exc_info=True)

    async def _start_operation(
        self, operation: str, metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        return await self._begin_operation(
            operation,
            f"Install {self.label}",
            {"engine": self.engine_id, **(metadata or {})},
        )


_RELEASE_INSTALLERS: Dict[str, GenericReleaseInstaller] = {}


def get_release_installer(engine_id: str) -> GenericReleaseInstaller:
    """Singleton generic release host for a definition declaring release_assets."""
    from backend.engines.definitions.loader import load_default_engine_definitions

    key = str(engine_id or "")
    installer = _RELEASE_INSTALLERS.get(key)
    if installer is None:
        definition = load_default_engine_definitions().definitions.get(key)
        if definition is None or not definition.release_assets:
            raise RuntimeError(f"{key!r} has no release_assets definition")
        installer = GenericReleaseInstaller(definition)
        _RELEASE_INSTALLERS[key] = installer
    return installer


def reset_release_installers() -> None:
    _RELEASE_INSTALLERS.clear()
