"""Install and manage native audio.cpp engine versions."""

from __future__ import annotations

import asyncio
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, Body, Depends, HTTPException

from backend.audio_cpp_manager import (
    AUDIO_CPP_DEFAULT_REF,
    AUDIO_CPP_REPOSITORY,
    AudioCppBuildConfig,
    get_audio_cpp_manager,
)
from backend.audio_cpp_tracking import (
    ensure_tracking_settings,
    merge_settings,
    split_settings,
)
from backend.build_task_manager import BuildTaskManager
from backend.data_store import get_store
from backend.engine_param_catalog import get_version_entry
from backend.feature_flags import audio_cpp_enabled
from backend.logging_config import get_logger
from backend.progress_manager import get_progress_manager


logger = get_logger(__name__)


def _require_audio_cpp_enabled() -> None:
    if not audio_cpp_enabled():
        raise HTTPException(
            status_code=404,
            detail="The audio.cpp integration is disabled by AUDIO_CPP_ENABLED",
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


async def _latest_upstream(ref: str) -> Dict[str, Any]:
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


def _capability_delta(previous: Optional[dict], current: Optional[dict]) -> Dict[str, Any]:
    from backend.engine_param_scanner import compute_audio_cpp_capability_delta

    return compute_audio_cpp_capability_delta(previous, current)


def _audio_models_affected_by_delta(
    store, delta: Optional[dict], *, contract_changed: bool = False
) -> List[dict]:
    """Return lightweight model rows whose saved family/task intersect *delta*."""
    delta = delta or {}
    families = {
        str(item).strip().lower()
        for item in [
            *(delta.get("added_families") or []),
            *(delta.get("removed_families") or []),
        ]
        if str(item).strip()
    }
    tasks = {
        str(item).strip().lower()
        for item in [
            *(delta.get("added_tasks") or []),
            *(delta.get("removed_tasks") or []),
        ]
        if str(item).strip()
    }
    affected: List[dict] = []
    for model in store.list_models() or []:
        if not isinstance(model, dict):
            continue
        config = model.get("config") if isinstance(model.get("config"), dict) else {}
        engine = str(config.get("engine") or model.get("engine") or "").strip()
        engines = config.get("engines") if isinstance(config.get("engines"), dict) else {}
        audio_cfg = engines.get("audio_cpp") if isinstance(engines.get("audio_cpp"), dict) else {}
        if engine != "audio_cpp" and not audio_cfg:
            continue
        family = str(
            audio_cfg.get("family") or model.get("family") or ""
        ).strip().lower()
        task = str(audio_cfg.get("task") or "").strip().lower()
        intersects = (family and family in families) or (task and task in tasks)
        if intersects or (contract_changed and not families and not tasks):
            affected.append(
                {
                    "id": model.get("id"),
                    "name": model.get("name") or model.get("display_name") or model.get("id"),
                    "family": family or None,
                    "task": task or None,
                    "last_reviewed_fingerprint": audio_cfg.get(
                        "last_reviewed_fingerprint"
                    ),
                }
            )
    return affected


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
    previous_entry = get_version_entry(store, "audio_cpp", str(row.get("version") or ""))
    store.set_active_engine_version("audio_cpp", str(row["version"]))

    scan_entry = None
    try:
        from backend.engine_param_scanner import scan_engine_version

        scan_entry = await asyncio.to_thread(scan_engine_version, store, "audio_cpp", row)
    except Exception as exc:
        logger.warning("audio.cpp parameter scan failed after activation: %s", exc)

    try:
        from backend.llama_swap_manager import get_llama_swap_manager, mark_swap_config_stale

        mark_swap_config_stale()
        await get_llama_swap_manager().start_proxy()
    except Exception as exc:
        # Engine activation remains valid even when the proxy cannot yet start.
        logger.warning("Could not start llama-swap after audio.cpp activation: %s", exc)

    delta = (
        (scan_entry or {}).get("capability_delta")
        if isinstance(scan_entry, dict) and (scan_entry or {}).get("capability_delta")
        else _capability_delta(
            previous_entry, scan_entry if isinstance(scan_entry, dict) else None
        )
    )
    return {
        "message": f"Activated audio.cpp version {row['version']}",
        "capability_delta": delta,
        "contract_fingerprint": (scan_entry or {}).get("contract_fingerprint")
        if isinstance(scan_entry, dict)
        else None,
        "contract_changed": bool(
            isinstance(scan_entry, dict) and scan_entry.get("contract_changed")
        ),
        "affected_models": _audio_models_affected_by_delta(
            store,
            delta,
            contract_changed=bool(
                isinstance(scan_entry, dict) and scan_entry.get("contract_changed")
            ),
        ),
    }


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
        # Keep synced branch install active when it already was, or activate it
        active = store.get_active_engine_version("audio_cpp")
        if not active or str(active.get("version")) == str(version_name):
            await _activate(version_name)
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
        "sync": True,
    }


def _schedule_build(payload: dict) -> dict:
    store = get_store()
    tracking, _cmake = split_settings(store.get_engine_build_settings("audio_cpp"))
    default_ref = tracking.get("tracking_ref") or AUDIO_CPP_DEFAULT_REF
    default_repo = tracking.get("repository_url") or AUDIO_CPP_REPOSITORY
    source_ref = str(payload.get("source_ref") or payload.get("commit_sha") or default_ref).strip()
    repository_url = str(payload.get("repository_url") or default_repo).strip()
    source_ref_type = str(payload.get("source_ref_type") or _ref_kind(source_ref))
    manager = get_audio_cpp_manager()
    build_config = manager.build_config_from_dict(payload.get("build_config"))
    try:
        manager.validate_build_config(build_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    suffix = str(payload.get("version_suffix") or int(time.time())).strip()
    version_name = f"source-{_version_slug(source_ref)}-{_version_slug(suffix)}"
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
    # Persist tracking when the user builds from an explicit branch/tag
    if source_ref_type in {"branch", "release"}:
        store.update_engine_build_settings(
            "audio_cpp",
            merge_settings(
                tracking_ref=source_ref,
                repository_url=repository_url,
                build_config=build_config.__dict__,
                existing=store.get_engine_build_settings("audio_cpp"),
            ),
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
    settings = await ensure_tracking_settings(store)
    tracking, _cmake = split_settings(settings)
    active = store.get_active_engine_version("audio_cpp")
    entry = (
        get_version_entry(store, "audio_cpp", str(active.get("version") or ""))
        if active
        else None
    )
    caps = (entry or {}).get("capabilities") or {}
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
        "tracking_ref": tracking.get("tracking_ref"),
        "repository_url": tracking.get("repository_url") or AUDIO_CPP_REPOSITORY,
        "contract_fingerprint": (entry or {}).get("contract_fingerprint"),
        "contract_changed": bool((entry or {}).get("contract_changed")),
        "previous_contract_fingerprint": (entry or {}).get("previous_contract_fingerprint"),
        "capability_delta": (entry or {}).get("capability_delta") or {
            "added_families": [],
            "removed_families": [],
            "added_tasks": [],
            "removed_tasks": [],
        },
        "families": list(caps.get("families") or []),
        "tasks": list(caps.get("tasks") or []),
        "discovery_source": caps.get("discovery_source"),
        "affected_models": _audio_models_affected_by_delta(
            store,
            (entry or {}).get("capability_delta"),
            contract_changed=bool((entry or {}).get("contract_changed")),
        ),
        "supported_build_backends": get_audio_cpp_manager().supported_build_backends(),
    }


@router.get("/build-settings")
async def get_build_settings():
    store = get_store()
    settings = await ensure_tracking_settings(store)
    tracking, cmake = split_settings(settings)
    return {**cmake, **tracking}


@router.put("/build-settings")
async def save_build_settings(payload: dict = Body(default_factory=dict)):
    store = get_store()
    existing = store.get_engine_build_settings("audio_cpp") or {}
    payload = payload or {}
    # Accept either flat envelope or nested build_config
    build_config = payload.get("build_config")
    if not isinstance(build_config, dict):
        build_config = {
            key: value
            for key, value in payload.items()
            if key not in {"tracking_ref", "repository_url", "build_config"}
        }
    merged = merge_settings(
        tracking_ref=payload.get("tracking_ref"),
        repository_url=payload.get("repository_url"),
        build_config=build_config,
        existing=existing,
    )
    stored = store.update_engine_build_settings("audio_cpp", merged)
    tracking, cmake = split_settings(stored)
    return {**cmake, **tracking}


@router.post("/build-source")
async def build_source(payload: dict = Body(default_factory=dict)):
    await ensure_tracking_settings()
    return _schedule_build(payload or {})


@router.post("/update")
async def update(payload: dict = Body(default_factory=dict)):
    store = get_store()
    settings = await ensure_tracking_settings(store)
    tracking, cmake = split_settings(settings)
    payload = payload or {}
    ref = str(payload.get("source_ref") or tracking.get("tracking_ref") or AUDIO_CPP_DEFAULT_REF).strip()
    repository_url = str(
        payload.get("repository_url") or tracking.get("repository_url") or AUDIO_CPP_REPOSITORY
    ).strip()
    build_config = get_audio_cpp_manager().build_config_from_dict(
        payload.get("build_config") or cmake
    )
    # Persist the tracking ref the user is updating against
    store.update_engine_build_settings(
        "audio_cpp",
        merge_settings(
            tracking_ref=ref,
            repository_url=repository_url,
            build_config=build_config.__dict__,
            existing=settings,
        ),
    )

    latest = await _latest_upstream(ref)
    active = store.get_active_engine_version("audio_cpp")
    active_branch = str((active or {}).get("source_branch") or "").strip()
    ref_kind = _ref_kind(ref)

    # Prefer in-place sync when the active install already tracks this branch/tag as a branch
    if (
        active
        and active_branch
        and active_branch == ref
        and ref_kind in {"branch", "release"}
        and active.get("source_path")
    ):
        return schedule_audio_cpp_sync(active, ref, build_config)

    # Rebuild as a syncable branch/tag install (not a detached tip SHA)
    return _schedule_build(
        {
            **payload,
            "source_ref": ref,
            "source_ref_type": ref_kind if ref_kind != "commit" else "branch",
            "repository_url": repository_url,
            "build_config": build_config.__dict__,
            "version_suffix": (latest.get("sha") or "")[:8] or str(int(time.time())),
            "auto_activate": True,
        }
    )


@router.get("/check-updates")
async def check_updates(ref: Optional[str] = None):
    store = get_store()
    settings = await ensure_tracking_settings(store)
    tracking, _cmake = split_settings(settings)
    track_ref = str(ref or tracking.get("tracking_ref") or AUDIO_CPP_DEFAULT_REF).strip()
    latest = await _latest_upstream(track_ref)
    active = store.get_active_engine_version("audio_cpp")
    current = (active or {}).get("source_commit")
    entry = (
        get_version_entry(store, "audio_cpp", str(active.get("version") or ""))
        if active
        else None
    )
    return {
        "current_version": current,
        "latest_version": latest.get("sha"),
        "update_available": bool(current and latest.get("sha") and current != latest["sha"]),
        "latest_commit": latest,
        "tracking_ref": track_ref,
        "contract_fingerprint": (entry or {}).get("contract_fingerprint"),
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


@router.post("/migrate-defaults")
async def migrate_defaults(payload: dict = Body(default_factory=dict)):
    """Batch-migrate audio.cpp request defaults after endpoint / contract drift."""
    from backend.audio_defaults_migration import migrate_audio_models_defaults

    store = get_store()
    payload = payload or {}
    model_ids = payload.get("model_ids")
    if model_ids is not None and not isinstance(model_ids, list):
        raise HTTPException(status_code=400, detail="model_ids must be a list")
    mark_reviewed = payload.get("mark_reviewed", True)
    if not isinstance(mark_reviewed, bool):
        mark_reviewed = bool(mark_reviewed)
    result = migrate_audio_models_defaults(
        store,
        model_ids=model_ids,
        mark_reviewed=mark_reviewed,
    )
    return result


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
