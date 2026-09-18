"""Install/status endpoints for upstream SGLang and SGLang-V100."""

from __future__ import annotations

from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Body, HTTPException

from backend.data_store import get_store
from backend.logging_config import get_logger
from backend.sglang_manager import SGLANG_REPOSITORIES, get_sglang_manager
from backend.venv_install_settings import coerce_install_settings


router = APIRouter()
logger = get_logger(__name__)

_SLUG_TO_ENGINE = {
    "sglang": "sglang",
    "sglang-v100": "sglang_v100",
}


def _engine_for_slug(slug: str) -> str:
    engine = _SLUG_TO_ENGINE.get(str(slug or ""))
    if not engine:
        raise HTTPException(status_code=404, detail="Unknown SGLang variant")
    return engine


@router.get("/{slug}/check-updates")
async def sglang_check_updates(slug: str) -> Dict:
    engine = _engine_for_slug(slug)
    active = get_store().get_active_engine_version(engine) or {}
    try:
        async with httpx.AsyncClient(
            headers={"Accept": "application/vnd.github+json"}
        ) as client:
            if engine == "sglang":
                response = await client.get(
                    "https://pypi.org/pypi/sglang/json", timeout=15.0
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
                    "url": "https://pypi.org/project/sglang/",
                }

            response = await client.get(
                "https://api.github.com/repos/haohervchb/sglang-V100/commits/main",
                timeout=15.0,
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
                "url": payload.get("html_url")
                or "https://github.com/haohervchb/sglang-V100/commits/main",
            }
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Failed to check {engine} updates: {exc}"
        ) from exc


@router.get("/{slug}/status")
async def sglang_status(slug: str) -> Dict:
    engine = _engine_for_slug(slug)
    try:
        return get_sglang_manager(engine).status()
    except Exception as exc:
        logger.warning("%s/status: %s", slug, exc)
        return {
            "engine": engine,
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


@router.get("/{slug}/build-settings")
async def sglang_get_build_settings(slug: str) -> Dict:
    engine = _engine_for_slug(slug)
    saved = get_store().get_engine_build_settings(engine) or {}
    return coerce_install_settings(engine, saved)


@router.put("/{slug}/build-settings")
async def sglang_save_build_settings(
    slug: str, settings: Optional[Dict] = Body(None)
) -> Dict:
    engine = _engine_for_slug(slug)
    if settings is not None and not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    coerced = coerce_install_settings(engine, settings or {})
    get_store().replace_engine_build_settings(engine, coerced)
    return coerced


@router.post("/{slug}/install")
async def sglang_install(
    slug: str, request: Optional[Dict[str, object]] = None
) -> Dict:
    engine = _engine_for_slug(slug)
    manager = get_sglang_manager(engine)
    payload = request or {}
    saved = coerce_install_settings(
        engine, get_store().get_engine_build_settings(engine) or {}
    )
    try:
        if engine == "sglang_v100":
            return await manager.install_from_source(
                repo_url=str(payload.get("repo_url") or saved["source_repo"]),
                branch=str(payload.get("branch") or saved["source_branch"]),
            )
        version = payload.get("version")
        if version is None or not str(version).strip():
            version = saved.get("pip_version") or None
        return await manager.install_release(
            version=str(version) if version else None,
            force_reinstall=bool(payload.get("force_reinstall")),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{slug}/install-source")
async def sglang_install_source(
    slug: str, request: Optional[Dict[str, object]] = None
) -> Dict:
    engine = _engine_for_slug(slug)
    payload = request or {}
    saved = coerce_install_settings(
        engine, get_store().get_engine_build_settings(engine) or {}
    )
    try:
        return await get_sglang_manager(engine).install_from_source(
            repo_url=str(
                payload.get("repo_url")
                or saved.get("source_repo")
                or SGLANG_REPOSITORIES[engine]
            ),
            branch=str(payload.get("branch") or saved.get("source_branch") or "main"),
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{slug}/remove")
async def sglang_remove(slug: str) -> Dict:
    engine = _engine_for_slug(slug)
    try:
        return await get_sglang_manager(engine).remove()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/{slug}/cancel")
async def sglang_cancel(slug: str, payload: dict = Body(...)) -> Dict:
    engine = _engine_for_slug(slug)
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return get_sglang_manager(engine).cancel_task(str(task_id))


@router.get("/{slug}/logs")
async def sglang_logs(slug: str) -> Dict:
    engine = _engine_for_slug(slug)
    manager = get_sglang_manager(engine)
    return {"log": manager.read_log_tail(), "path": manager._log_path}
