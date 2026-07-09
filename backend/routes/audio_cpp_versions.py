"""Install and manage native audio.cpp engine versions."""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests
from fastapi import APIRouter, Body, Depends, HTTPException

from backend.audio_cpp_manager import (
    AUDIO_CPP_COMPATIBILITY_COMMIT,
    AUDIO_CPP_DEFAULT_REF,
    AUDIO_CPP_REPOSITORY,
    AudioCppBuildConfig,
    get_audio_cpp_manager,
)
from backend.build_task_manager import BuildTaskManager
from backend.data_store import get_store
from backend.feature_flags import audio_cpp_enabled
from backend.logging_config import get_logger
from backend.progress_manager import get_progress_manager


logger = get_logger(__name__)


def _require_audio_cpp_enabled() -> None:
    if not audio_cpp_enabled():
        raise HTTPException(
            status_code=404,
            detail="The experimental audio.cpp integration is disabled by AUDIO_CPP_ENABLED",
        )


router = APIRouter(dependencies=[Depends(_require_audio_cpp_enabled)])


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ref_kind(value: str) -> str:
    ref = str(value or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return "commit"
    if re.match(r"^v?\d+(?:\.\d+)+(?:[-+].*)?$", ref):
        return "release"
    return "branch"


def _version_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return re.sub(r"-{2,}", "-", slug).strip("-._")[:32] or "source"


async def _latest_upstream(ref: str = AUDIO_CPP_DEFAULT_REF) -> Dict[str, Any]:
    url = f"https://api.github.com/repos/0xShug0/audio.cpp/commits/{ref}"

    def _request() -> Dict[str, Any]:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        body = response.json()
        commit = body.get("commit") if isinstance(body, dict) else {}
        return {
            "sha": body.get("sha"),
            "message": (commit or {}).get("message"),
            "commit_date": ((commit or {}).get("committer") or {}).get("date"),
            "html_url": body.get("html_url"),
            "ref": ref,
        }

    try:
        return await asyncio.to_thread(_request)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 500
        if status == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded")
        if status == 404:
            raise HTTPException(status_code=404, detail=f"audio.cpp ref '{ref}' not found")
        raise HTTPException(status_code=502, detail=f"GitHub API error: {exc}")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}")


async def _activate(version: str) -> dict:
    store = get_store()
    row = next(
        (
            item
            for item in store.get_engine_versions("audio_cpp")
            if str(item.get("version")) == str(version)
        ),
        None,
    )
    if not row:
        raise HTTPException(status_code=404, detail="audio.cpp version not found")
    missing = [
        key
        for key in ("server_binary_path", "cli_binary_path")
        if not row.get(key) or not os.path.isfile(str(row[key]))
    ]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"audio.cpp version is missing: {', '.join(missing)}",
        )
    store.set_active_engine_version("audio_cpp", str(row["version"]))

    try:
        from backend.engine_param_scanner import scan_engine_version

        await asyncio.to_thread(scan_engine_version, store, "audio_cpp", row)
    except Exception as exc:
        logger.warning("audio.cpp parameter scan failed after activation: %s", exc)

    try:
        from backend.llama_swap_manager import get_llama_swap_manager, mark_swap_config_stale

        mark_swap_config_stale()
        await get_llama_swap_manager().start_proxy()
    except Exception as exc:
        # Engine activation remains valid even when the proxy cannot yet start.
        logger.warning("Could not start llama-swap after audio.cpp activation: %s", exc)
    return {"message": f"Activated audio.cpp version {row['version']}"}


async def _build_task(
    *,
    task_id: str,
    version_name: str,
    source_ref: str,
    source_ref_type: str,
    repository_url: str,
    build_config: AudioCppBuildConfig,
    auto_activate: bool,
) -> None:
    manager = get_audio_cpp_manager()
    store = get_store()
    pm = get_progress_manager()
    try:
        result = await manager.build_source(
            source_ref=source_ref,
            version_name=version_name,
            repository_url=repository_url,
            build_config=build_config,
            progress_manager=pm,
            task_id=task_id,
        )
        row = {
            **result,
            "type": "source",
            "install_type": "source",
            "repository_source": "audio.cpp",
            "source_ref_type": source_ref_type,
            "source_branch": source_ref if source_ref_type == "branch" else None,
            "installed_at": _utcnow(),
        }
        store.add_engine_version("audio_cpp", row)
        if auto_activate:
            await _activate(version_name)
        else:
            try:
                from backend.llama_swap_manager import mark_swap_config_stale

                mark_swap_config_stale()
            except Exception:
                pass
        pm.complete_task(task_id, f"Installed audio.cpp {version_name}")
        await pm.send_notification(
            title="audio.cpp installed",
            message=f"Built audio.cpp {version_name} ({build_config.backend})",
            type="success",
            task_id=task_id,
        )
    except asyncio.CancelledError:
        pm.fail_task(task_id, "audio.cpp build cancelled")
        raise
    except Exception as exc:
        logger.exception("audio.cpp source build failed")
        pm.fail_task(task_id, str(exc))
        await pm.send_notification(
            title="audio.cpp build failed",
            message=str(exc),
            type="error",
            task_id=task_id,
        )


async def _sync_task(
    *,
    task_id: str,
    version_name: str,
    branch: str,
    build_config: AudioCppBuildConfig,
) -> None:
    manager = get_audio_cpp_manager()
    store = get_store()
    pm = get_progress_manager()
    version_row = next(
        (
            item
            for item in store.get_engine_versions("audio_cpp")
            if str(item.get("version")) == str(version_name)
        ),
        None,
    )
    if not version_row:
        pm.fail_task(task_id, f"audio.cpp version '{version_name}' not found")
        return
    try:
        result = await manager.sync_source(
            version_entry=version_row,
            branch=branch,
            build_config=build_config,
            progress_manager=pm,
            task_id=task_id,
        )
        updated = store.update_engine_version("audio_cpp", version_name, {
            **result,
            "updated_at": _utcnow(),
        })
        if not updated:
            raise RuntimeError(f"Version '{version_name}' disappeared during sync")
        try:
            from backend.engine_param_scanner import scan_engine_version

            scan_engine_version(store, "audio_cpp", updated)
        except Exception as exc:
            logger.warning("audio.cpp parameter scan failed after sync: %s", exc)
        try:
            from backend.llama_swap_manager import mark_swap_config_stale

            mark_swap_config_stale()
        except Exception:
            pass
        pm.complete_task(task_id, f"Synced audio.cpp {version_name}")
        await pm.send_notification(
            title="audio.cpp sync complete",
            message=f"Rebuilt audio.cpp {version_name} from {branch}",
            type="success",
            task_id=task_id,
        )
    except asyncio.CancelledError:
        pm.fail_task(task_id, "audio.cpp sync cancelled")
        raise
    except Exception as exc:
        logger.exception("audio.cpp source sync failed")
        pm.fail_task(task_id, str(exc))
        await pm.send_notification(
            title="audio.cpp sync failed",
            message=str(exc),
            type="error",
            task_id=task_id,
        )


def schedule_audio_cpp_sync(version_entry: dict, branch: str, build_config: AudioCppBuildConfig) -> dict:
    version_name = str(version_entry.get("version") or "").strip()
    if not version_name:
        raise HTTPException(status_code=400, detail="Version metadata is missing a name")

    branch = str(branch or "").strip()
    task_id = f"build_sync_{_version_slug(version_name)}_{int(time.time())}"
    pm = get_progress_manager()
    pm.create_task(
        "build",
        f"Sync audio.cpp {branch}",
        {
            "engine": "audio_cpp",
            "version_name": version_name,
            "repository_source": "audio.cpp",
            "source_ref": branch,
            "source_ref_type": "branch",
            "sync": True,
        },
        task_id=task_id,
    )
    asyncio.create_task(
        _sync_task(
            task_id=task_id,
            version_name=version_name,
            branch=branch,
            build_config=build_config,
        )
    )
    return {
        "message": f"Syncing audio.cpp {version_name} from {branch}",
        "task_id": task_id,
        "status": "started",
        "progress": 0,
        "version_name": version_name,
        "repository_source": "audio.cpp",
        "source_ref": branch,
        "source_ref_type": "branch",
    }


def _schedule_build(payload: dict) -> dict:
    source_ref = str(payload.get("source_ref") or payload.get("commit_sha") or AUDIO_CPP_DEFAULT_REF).strip()
    repository_url = str(payload.get("repository_url") or AUDIO_CPP_REPOSITORY).strip()
    source_ref_type = str(payload.get("source_ref_type") or _ref_kind(source_ref))
    manager = get_audio_cpp_manager()
    build_config = manager.build_config_from_dict(payload.get("build_config"))
    try:
        manager.validate_build_config(build_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    suffix = str(payload.get("version_suffix") or int(time.time())).strip()
    version_name = f"source-{_version_slug(source_ref)}-{_version_slug(suffix)}"
    store = get_store()
    if any(
        str(row.get("version")) == version_name
        for row in store.get_engine_versions("audio_cpp")
    ):
        raise HTTPException(status_code=409, detail=f"Version '{version_name}' already exists")

    task_id = f"build_audio_cpp_{_version_slug(version_name)}_{int(time.time())}"
    pm = get_progress_manager()
    pm.create_task(
        "build",
        f"Build audio.cpp {source_ref}",
        {
            "engine": "audio_cpp",
            "version_name": version_name,
            "repository_source": "audio.cpp",
            "source_ref": source_ref,
            "source_ref_type": source_ref_type,
            "backend": build_config.backend,
            "auto_activate": bool(payload.get("auto_activate", True)),
        },
        task_id=task_id,
    )
    asyncio.create_task(
        _build_task(
            task_id=task_id,
            version_name=version_name,
            source_ref=source_ref,
            source_ref_type=source_ref_type,
            repository_url=repository_url,
            build_config=build_config,
            auto_activate=bool(payload.get("auto_activate", True)),
        )
    )
    return {
        "message": f"Building audio.cpp {source_ref}",
        "task_id": task_id,
        "status": "started",
        "version_name": version_name,
        "source_ref": source_ref,
        "source_ref_type": source_ref_type,
    }


@router.get("")
@router.get("/")
async def list_versions():
    store = get_store()
    active = store.get_active_engine_version("audio_cpp")
    active_version = active.get("version") if active else None
    return [
        {
            **row,
            "id": f"audio_cpp:{row.get('version')}",
            "is_active": str(row.get("version")) == str(active_version),
        }
        for row in store.get_engine_versions("audio_cpp")
    ]


@router.get("/status")
async def status():
    store = get_store()
    active = store.get_active_engine_version("audio_cpp")
    return {
        "installed": bool(store.get_engine_versions("audio_cpp")),
        "active": active,
        "runnable": bool(
            active
            and all(
                active.get(key) and os.path.isfile(str(active[key]))
                for key in ("server_binary_path", "cli_binary_path")
            )
        ),
        "models_root": get_audio_cpp_manager().models_dir,
        "model_manager_ready": bool(
            active
            and active.get("model_manager_path")
            and os.path.isfile(str(active["model_manager_path"]))
        ),
        "compatibility_ref": AUDIO_CPP_DEFAULT_REF,
        "compatibility_commit": AUDIO_CPP_COMPATIBILITY_COMMIT,
        "compatibility_verified": bool(
            active
            and str(active.get("source_commit") or "")
            == AUDIO_CPP_COMPATIBILITY_COMMIT
        ),
        "supported_build_backends": get_audio_cpp_manager().supported_build_backends(),
    }


@router.get("/build-settings")
async def get_build_settings():
    store = get_store()
    raw = store.get_engine_build_settings("audio_cpp")
    return get_audio_cpp_manager().build_config_from_dict(raw).__dict__


@router.put("/build-settings")
async def save_build_settings(payload: dict = Body(default_factory=dict)):
    config = get_audio_cpp_manager().build_config_from_dict(payload)
    return get_store().update_engine_build_settings("audio_cpp", config.__dict__)


@router.post("/build-source")
async def build_source(payload: dict = Body(default_factory=dict)):
    return _schedule_build(payload or {})


@router.post("/update")
async def update(payload: dict = Body(default_factory=dict)):
    ref = str((payload or {}).get("source_ref") or AUDIO_CPP_DEFAULT_REF)
    latest = await _latest_upstream(ref)
    return _schedule_build(
        {
            **(payload or {}),
            "source_ref": latest["sha"],
            "source_ref_type": "commit",
            "version_suffix": (latest["sha"] or "")[:8],
            "auto_activate": True,
        }
    )


@router.get("/check-updates")
async def check_updates(ref: str = AUDIO_CPP_DEFAULT_REF):
    latest = await _latest_upstream(ref)
    active = get_store().get_active_engine_version("audio_cpp")
    current = (active or {}).get("source_commit")
    return {
        "current_version": current,
        "latest_version": latest.get("sha"),
        "update_available": bool(current and latest.get("sha") and current != latest["sha"]),
        "latest_commit": latest,
    }


@router.post("/cancel")
async def cancel(payload: dict = Body(default_factory=dict)):
    task_id = str((payload or {}).get("task_id") or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return BuildTaskManager.cancel(task_id)


@router.post("/versions/activate")
async def activate(payload: dict = Body(default_factory=dict)):
    version_id = str((payload or {}).get("version_id") or "").strip()
    if version_id.startswith("audio_cpp:"):
        version_id = version_id.split(":", 1)[1]
    if not version_id:
        raise HTTPException(status_code=400, detail="version_id is required")
    return await _activate(version_id)


@router.delete("/versions/{version}")
async def delete_version(version: str):
    store = get_store()
    row = next(
        (
            item
            for item in store.get_engine_versions("audio_cpp")
            if str(item.get("version")) == str(version)
        ),
        None,
    )
    if not row:
        raise HTTPException(status_code=404, detail="audio.cpp version not found")
    active = store.get_active_engine_version("audio_cpp")
    if active and str(active.get("version")) == str(version):
        raise HTTPException(status_code=409, detail="Cannot delete the active audio.cpp version")
    try:
        get_audio_cpp_manager().delete_version_files(row)
        store.delete_engine_version("audio_cpp", str(version))
        try:
            from backend.llama_swap_manager import mark_swap_config_stale

            mark_swap_config_stale()
        except Exception:
            pass
        return {"message": f"Deleted audio.cpp version {version}"}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Failed deleting audio.cpp version")
        raise HTTPException(status_code=500, detail=str(exc))

