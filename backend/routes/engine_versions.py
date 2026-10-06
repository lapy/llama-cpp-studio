"""Shared HTTP surface for Python-engine installers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException

from backend.data_store import get_store
from backend.engines.protocol import EngineInstaller
from backend.logging_config import get_logger
from backend.venv_install_settings import coerce_install_settings


logger = get_logger(__name__)


@dataclass(frozen=True)
class UpdateSource:
    kind: str
    package: str = ""
    repo: str = ""
    branch: str = "main"
    url: str = ""
    timeout: float = 15.0
    error_status: int = 502


def pypi_updates(
    package: str, *, url: str = "", timeout: float = 15.0, error_status: int = 502
) -> UpdateSource:
    return UpdateSource(
        kind="pypi",
        package=package,
        url=url or f"https://pypi.org/project/{package}/",
        timeout=timeout,
        error_status=error_status,
    )


def github_commit_updates(repo: str, *, branch: str = "main") -> UpdateSource:
    return UpdateSource(
        kind="github_commit",
        repo=repo,
        branch=branch,
        url=f"https://github.com/{repo}/commits/{branch}",
    )


def github_release_updates(repo: str) -> UpdateSource:
    return UpdateSource(kind="github_releases", repo=repo, timeout=10.0, error_status=500)


async def _check_updates(engine_id: str, source: UpdateSource) -> Dict[str, Any]:
    active = get_store().get_active_engine_version(engine_id) or {}
    try:
        if source.kind == "pypi":
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"https://pypi.org/pypi/{source.package}/json",
                    timeout=source.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            latest = (payload.get("info") or {}).get("version")
            current = active.get("package_version") or active.get("version")
            return {
                "latest_version": latest,
                "current_version": current,
                "update_available": bool(latest and latest != current),
                "releases": list((payload.get("releases") or {}).keys()),
                "url": source.url,
            }
        if source.kind == "github_commit":
            async with httpx.AsyncClient(
                headers={"Accept": "application/vnd.github+json"}
            ) as client:
                response = await client.get(
                    f"https://api.github.com/repos/{source.repo}/commits/{source.branch}",
                    timeout=source.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            latest = str(payload.get("sha") or "")
            current = str(active.get("source_commit") or "")
            return {
                "latest_version": latest[:12] or None,
                "latest_commit": latest or None,
                "current_version": current[:12] or None,
                "update_available": bool(latest and latest != current),
                "url": payload.get("html_url") or source.url,
            }
        if source.kind == "github_releases":
            async with httpx.AsyncClient(
                headers={"Accept": "application/vnd.github+json"},
                timeout=source.timeout,
            ) as client:
                response = await client.get(
                    f"https://api.github.com/repos/{source.repo}/releases",
                    timeout=source.timeout,
                )
                response.raise_for_status()
                releases = response.json() or []
            tags = [rel.get("tag_name") for rel in releases if rel.get("tag_name")]
            latest = None
            for rel in releases:
                if rel.get("prerelease"):
                    continue
                latest = rel.get("tag_name")
                break
            if latest is None and releases:
                latest = releases[0].get("tag_name")
            return {
                "latest_version": (latest or "").lstrip("v") or None,
                "releases": [str(tag).lstrip("v") for tag in tags],
            }
        raise ValueError(f"Unknown update source {source.kind!r}")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=source.error_status,
            detail=f"Failed to check {engine_id} updates: {exc}",
        ) from exc


def python_engine_router(
    *,
    engine_id: str,
    url_prefix: str,
    get_installer: Callable[[], EngineInstaller],
    update_source: UpdateSource,
    prefer_source_install: bool = False,
    include_logs: bool = False,
    status_fallback: Optional[Dict[str, Any]] = None,
    version_setting_key: str = "pip_version",
) -> APIRouter:
    router = APIRouter()
    prefix = url_prefix.rstrip("/")
    op = engine_id.replace("-", "_")

    def _saved() -> Dict[str, Any]:
        return coerce_install_settings(
            engine_id, get_store().get_engine_build_settings(engine_id) or {}
        )

    @router.get(f"{prefix}/check-updates", operation_id=f"{op}_check_updates")
    async def check_updates() -> Dict:
        return await _check_updates(engine_id, update_source)

    @router.get(f"{prefix}/status", operation_id=f"{op}_status")
    async def status() -> Dict:
        try:
            return get_installer().status()
        except Exception as exc:
            logger.warning("%s/status: %s", prefix, exc)
            fallback = {
                "engine": engine_id,
                "installed": False,
                "version": None,
                "binary_path": None,
                "venv_path": None,
                "operation": None,
                "operation_started_at": None,
                "progress_task_id": None,
                "last_error": str(exc),
                "install_type": None,
                "source_repo": None,
                "source_branch": None,
            }
            if status_fallback:
                fallback.update(status_fallback)
            return fallback

    @router.get(f"{prefix}/build-settings", operation_id=f"{op}_get_build_settings")
    async def get_build_settings() -> Dict:
        return _saved()

    @router.put(f"{prefix}/build-settings", operation_id=f"{op}_save_build_settings")
    async def save_build_settings(settings: Optional[Dict] = Body(None)) -> Dict:
        if settings is not None and not isinstance(settings, dict):
            raise HTTPException(status_code=400, detail="settings must be an object")
        coerced = coerce_install_settings(engine_id, settings or {})
        get_store().replace_engine_build_settings(engine_id, coerced)
        return coerced

    @router.post(f"{prefix}/install", operation_id=f"{op}_install")
    async def install(request: Optional[Dict[str, Any]] = None) -> Dict:
        payload = request or {}
        from backend.operations.action_recovery import (
            ActionAdmissionError,
            bind_action_confirmation,
        )

        bind_action_confirmation(payload)
        saved = _saved()
        try:
            if prefer_source_install:
                return await get_installer().install_from_source(
                    repo_url=str(payload.get("repo_url") or saved["source_repo"]),
                    branch=str(payload.get("branch") or saved["source_branch"]),
                )
            version = payload.get("version")
            if version is None or not str(version).strip():
                version = saved.get(version_setting_key) or None
            return await get_installer().install_release(
                version=str(version) if version else None,
                force_reinstall=bool(payload.get("force_reinstall")),
            )
        except ActionAdmissionError as exc:
            raise HTTPException(status_code=409, detail=exc.detail) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post(f"{prefix}/install-source", operation_id=f"{op}_install_source")
    async def install_source(request: Optional[Dict[str, Any]] = None) -> Dict:
        payload = request or {}
        from backend.operations.action_recovery import (
            ActionAdmissionError,
            bind_action_confirmation,
        )

        bind_action_confirmation(payload)
        saved = _saved()
        try:
            return await get_installer().install_from_source(
                repo_url=str(payload.get("repo_url") or saved.get("source_repo") or ""),
                branch=str(payload.get("branch") or saved.get("source_branch") or "main"),
            )
        except ActionAdmissionError as exc:
            raise HTTPException(status_code=409, detail=exc.detail) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post(f"{prefix}/remove", operation_id=f"{op}_remove")
    async def remove() -> Dict:
        try:
            return await get_installer().remove()
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @router.post(f"{prefix}/cancel", operation_id=f"{op}_cancel")
    async def cancel(payload: dict = Body(...)) -> Dict:
        task_id = (payload or {}).get("task_id")
        if not task_id:
            raise HTTPException(status_code=400, detail="task_id is required")
        return get_installer().cancel_task(str(task_id))

    if include_logs:
        @router.get(f"{prefix}/logs", operation_id=f"{op}_logs")
        async def logs() -> Dict:
            installer = get_installer()
            return {"log": installer.read_log_tail(), "path": installer.log_path}

    return router
