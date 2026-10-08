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

from backend.engines.audio_cpp.activation import (
    models_affected_by_delta as _audio_models_affected_by_delta,
)
from backend.engines.audio_cpp.manager import (
    AUDIO_CPP_DEFAULT_REF,
    AUDIO_CPP_REPOSITORY,
    AudioCppBuildConfig,
    get_audio_cpp_manager,
)
from backend.engines.audio_cpp.tracking import (
    ensure_tracking_settings,
    is_audio_release_tag,
    merge_settings,
    resolve_latest_github_release,
    resolve_latest_release_tag,
    split_settings,
)
from backend.engines.audio_cpp.build_options import catalog_for_ui, coerce_build_settings
from backend.repo_identity import source_build_type_labels_for_engine
from backend.build_task_manager import BuildTaskManager
from backend.data_store import get_store
from backend.engines.params import get_version_entry
from backend.engines.registry import required_active_path_fields
from backend.feature_flags import audio_cpp_enabled
from backend.logging_config import get_logger
from backend.operations.action_recovery import (
    CUDA_TOOLKIT_KEY,
    bind_action_confirmation,
    engine_installation_key,
)
from backend.operations.supervisor import ResourceBusyError, get_supervisor
from backend.operations.progress import get_progress_manager


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
    """Classify a git ref: commit SHA, GitHub release tag, or branch."""
    ref = str(value or "").strip()
    if re.fullmatch(r"[0-9a-fA-F]{40}", ref):
        return "commit"
    # audio.cpp: v0.7.0 (current); also release-0.5.1 and v0.2.0-windows-prebuilt
    if is_audio_release_tag(ref):
        return "release"
    return "branch"


def _version_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value or "").strip())
    return re.sub(r"-{2,}", "-", slug).strip("-._")[:32] or "source"


def _github_api_repo_slug(repository_url: str) -> Optional[str]:
    """Return ``owner/repo`` for GitHub clone URLs, else ``None``."""
    value = str(repository_url or "").strip().rstrip("/")
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", value, re.I)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


async def _latest_upstream(
    ref: str, repository_url: Optional[str] = None
) -> Dict[str, Any]:
    repo_url = str(repository_url or AUDIO_CPP_REPOSITORY).strip() or AUDIO_CPP_REPOSITORY
    slug = _github_api_repo_slug(repo_url)
    if not slug:
        raise HTTPException(
            status_code=400,
            detail=(
                "Update checks require a GitHub repository URL "
                f"(got {repo_url!r}). Build/sync still work with any git remote."
            ),
        )
    url = f"https://api.github.com/repos/{slug}/commits/{ref}"

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
            "repository": slug,
        }

    try:
        return await asyncio.to_thread(_request)
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else 500
        if status == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limit exceeded")
        if status == 404:
            raise HTTPException(
                status_code=404,
                detail=f"audio.cpp ref '{ref}' not found on {slug}",
            )
        raise HTTPException(status_code=502, detail=f"GitHub API error: {exc}")
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"GitHub request failed: {exc}")


async def _activate(
    version: str,
    payload: Optional[dict] = None,
    *,
    covered_by: Optional[str] = None,
) -> dict:
    from backend.engines.activation import activate_engine_version

    return await activate_engine_version(
        "audio_cpp",
        version,
        payload=payload,
        covered_by=covered_by,
    )


async def _build_task(
    *,
    task_id: str,
    version_name: str,
    source_ref: str,
    source_ref_type: str,
    repository_url: str,
    build_config: AudioCppBuildConfig,
    auto_activate: bool,
    replace_existing: bool = False,
    use_workspace: bool = True,
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
            replace_existing=replace_existing,
            use_workspace=use_workspace,
        )
        type_labels = source_build_type_labels_for_engine("audio_cpp", repository_url)
        builds_dir = getattr(manager, "builds_dir", "") or ""
        row = {
            **result,
            "type": type_labels["type"],
            "install_type": type_labels["install_type"],
            "is_fork": type_labels["is_fork"],
            "repository_source": "audio.cpp",
            "source_ref_type": source_ref_type,
            "source_branch": source_ref if source_ref_type in {"branch", "release"} else None,
            "install_dir": result.get("install_dir")
            or (os.path.join(builds_dir, version_name) if builds_dir else None),
            "installed_at": _utcnow(),
        }
        from backend.engines.lifecycle import mark_engine_version_ready

        mark_engine_version_ready(store, "audio_cpp", row)
        if auto_activate:
            await _activate(version_name, covered_by=task_id)
        else:
            try:
                from backend.proxy.llama_swap.manager import mark_swap_config_stale

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
        from backend.engines.lifecycle import mark_engine_version_failed

        mark_engine_version_failed(
            store,
            "audio_cpp",
            version_name,
            error="audio.cpp build cancelled",
            cancelled=True,
            extra={
                "source_ref": source_ref,
                "source_repo": repository_url,
                "repository_source": "audio.cpp",
                "install_dir": os.path.join(getattr(manager, "builds_dir", "") or "", version_name)
                if getattr(manager, "builds_dir", None)
                else None,
            },
        )
        pm.fail_task(task_id, "audio.cpp build cancelled")
        raise
    except Exception as exc:
        from backend.engines.lifecycle import mark_engine_version_failed

        logger.exception("audio.cpp source build failed")
        mark_engine_version_failed(
            store,
            "audio_cpp",
            version_name,
            error=str(exc),
            extra={
                "source_ref": source_ref,
                "source_repo": repository_url,
                "repository_source": "audio.cpp",
                "install_dir": os.path.join(getattr(manager, "builds_dir", "") or "", version_name)
                if getattr(manager, "builds_dir", None)
                else None,
            },
        )
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
        repo_url = str(
            result.get("source_repo")
            or version_row.get("source_repo")
            or ""
        ).strip()
        existing_kind = str(
            version_row.get("install_type") or version_row.get("type") or ""
        ).strip().lower()
        if existing_kind == "local":
            type_labels = {"type": "local", "install_type": "local", "is_fork": False}
        else:
            type_labels = source_build_type_labels_for_engine("audio_cpp", repo_url)
        updated = store.update_engine_version("audio_cpp", version_name, {
            **result,
            "type": type_labels["type"],
            "install_type": type_labels["install_type"],
            "is_fork": type_labels["is_fork"],
            "updated_at": _utcnow(),
        })
        if not updated:
            raise RuntimeError(f"Version '{version_name}' disappeared during sync")
        try:
            from backend.proxy.llama_swap.manager import mark_swap_config_stale

            mark_swap_config_stale()
        except Exception:
            pass
        # Keep a synced branch install active when it already was, or activate
        # it when nothing else is. ``_activate`` already runs the parameter
        # scan, so skip the extra scan on that path (it opened a second tray card).
        active = store.get_active_engine_version("audio_cpp")
        will_activate = not active or str(active.get("version")) == str(version_name)
        if not will_activate:
            try:
                from backend.engines.scan.scanner import scan_engine_version

                scan_engine_version(store, "audio_cpp", updated)
            except Exception as exc:
                logger.warning("audio.cpp parameter scan failed after sync: %s", exc)
        if will_activate:
            await _activate(version_name, covered_by=task_id)
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


async def _fence_build(task_id: str) -> None:
    from backend.store_io import StoreDurabilityError

    try:
        await get_supervisor().fence_effect_started(task_id)
    except StoreDurabilityError as exc:
        raise HTTPException(
            status_code=503 if exc.committed is False else 500,
            detail={
                "code": "ACTION_NOT_SENT",
                "committed": exc.committed,
                "message": (
                    "The build had not started. It was not launched."
                    if exc.committed is False
                    else "Whether this build started could not be established. It was not launched."
                ),
            },
        ) from exc


def _guard_build_admission(func):
    from backend.operations.action_recovery import ActionAdmissionError

    try:
        return func()
    except ActionAdmissionError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except ResourceBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def schedule_audio_cpp_sync(version_entry: dict, branch: str, build_config: AudioCppBuildConfig) -> dict:
    version_name = str(version_entry.get("version") or "").strip()
    if not version_name:
        raise HTTPException(status_code=400, detail="Version metadata is missing a name")

    branch = str(branch or "").strip()
    task_id = f"build_sync_{_version_slug(version_name)}_{int(time.time())}"
    pm = get_progress_manager()
    install_dir = str(
        version_entry.get("install_dir")
        or os.path.join(get_audio_cpp_manager().builds_dir, version_name)
    )
    _guard_build_admission(
        lambda: pm.create_task(
        "sync_source",
        f"Sync audio.cpp {branch}",
        {
            "engine": "audio_cpp",
            "version_name": version_name,
            "repository_source": "audio.cpp",
            "source_ref": branch,
            "source_ref_type": "branch",
            "sync": True,
            "resource_key": engine_installation_key("audio_cpp", install_dir),
            "depends_on": CUDA_TOOLKIT_KEY,
        },
        task_id=task_id,
        )
    )
    await _fence_build(task_id)
    get_supervisor().spawn(
        task_id,
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


async def schedule_audio_cpp_retry(version_entry: dict) -> dict:
    """Rebuild a failed/broken audio.cpp version in place."""
    store = get_store()
    manager = get_audio_cpp_manager()
    version_name = str(version_entry.get("version") or "").strip()
    source_ref = str(
        version_entry.get("source_ref")
        or version_entry.get("source_branch")
        or version_entry.get("source_commit")
        or ""
    ).strip()
    repository_url = str(
        version_entry.get("source_repo") or AUDIO_CPP_REPOSITORY
    ).strip()
    if not version_name or not source_ref:
        raise HTTPException(
            status_code=400,
            detail="This audio.cpp version does not have enough metadata to retry",
        )
    source_ref_type = str(version_entry.get("source_ref_type") or _ref_kind(source_ref))
    raw_config = version_entry.get("build_config")
    if isinstance(raw_config, dict):
        build_config = manager.build_config_from_dict(raw_config)
    else:
        build_config = manager.build_config_from_dict(
            store.get_engine_build_settings("audio_cpp")
        )
    try:
        manager.validate_build_config(build_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    task_id = f"build_retry_{_version_slug(version_name)}_{int(time.time())}"
    from backend.engines.lifecycle import mark_engine_version_building

    type_labels = source_build_type_labels_for_engine("audio_cpp", repository_url)
    install_dir = str(
        version_entry.get("install_dir")
        or os.path.join(manager.builds_dir, version_name)
    )
    pm = get_progress_manager()
    _guard_build_admission(
        lambda: pm.create_task(
            "update",
            f"Retry audio.cpp {version_name}",
            {
                "engine": "audio_cpp",
                "version_name": version_name,
                "repository_source": "audio.cpp",
                "source_ref": source_ref,
                "source_ref_type": source_ref_type,
                "retry": True,
                "resource_key": engine_installation_key("audio_cpp", install_dir),
                "depends_on": CUDA_TOOLKIT_KEY,
            },
            task_id=task_id,
        )
    )
    await _fence_build(task_id)
    mark_engine_version_building(
        store,
        "audio_cpp",
        {
            **{k: v for k, v in version_entry.items() if k != "orphan"},
            "version": version_name,
            "type": version_entry.get("type") or type_labels["type"],
            "install_type": version_entry.get("install_type") or type_labels["install_type"],
            "source_ref": source_ref,
            "source_ref_type": source_ref_type,
            "source_repo": repository_url,
            "repository_source": "audio.cpp",
            "install_dir": install_dir,
            "server_binary_path": None,
            "cli_binary_path": None,
        },
        task_id=task_id,
    )
    get_supervisor().spawn(
        task_id,
        _build_task(
            task_id=task_id,
            version_name=version_name,
            source_ref=source_ref,
            source_ref_type=source_ref_type,
            repository_url=repository_url,
            build_config=build_config,
            auto_activate=False,
            replace_existing=True,
            use_workspace=False,
        )
    )
    return {
        "message": f"Retrying audio.cpp {version_name}",
        "task_id": task_id,
        "status": "started",
        "version_name": version_name,
        "source_ref": source_ref,
        "source_ref_type": source_ref_type,
        "retry": True,
    }


async def _schedule_build(payload: dict) -> dict:
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
    from backend.engines.lifecycle import mark_engine_version_building
    from backend.repo_identity import source_build_type_labels_for_engine as _labels

    type_labels = _labels("audio_cpp", repository_url)
    pm = get_progress_manager()
    install_dir = os.path.join(manager.builds_dir, version_name)
    _guard_build_admission(
        lambda: pm.create_task(
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
            "resource_key": engine_installation_key("audio_cpp", install_dir),
            "depends_on": CUDA_TOOLKIT_KEY,
            },
            task_id=task_id,
        )
    )
    await _fence_build(task_id)
    mark_engine_version_building(
        store,
        "audio_cpp",
        {
            "version": version_name,
            "type": type_labels["type"],
            "install_type": type_labels["install_type"],
            "is_fork": type_labels["is_fork"],
            "source_ref": source_ref,
            "source_ref_type": source_ref_type,
            "source_branch": source_ref if source_ref_type in {"branch", "release"} else None,
            "source_repo": repository_url,
            "build_config": build_config.__dict__,
            "repository_source": "audio.cpp",
            "install_dir": install_dir,
            "installed_at": _utcnow(),
        },
        task_id=task_id,
    )
    get_supervisor().spawn(
        task_id,
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


async def _prebuilt_task(
    *,
    task_id: str,
    version_name: str,
    source_ref: str,
    repository_url: str,
    plan,
    auto_activate: bool,
) -> None:
    from backend.engines.audio_cpp.prebuilt import download_and_extract, materialize_cli

    store = get_store()
    pm = get_progress_manager()
    install_dir = os.path.join(get_audio_cpp_manager().builds_dir, version_name)
    try:
        pm.update_task(task_id, progress=15, message=f"Downloading {plan.asset_name}")
        binaries = await download_and_extract(plan.url, install_dir)
        if not binaries.get("cli_binary_path"):
            pm.update_task(
                task_id,
                progress=55,
                message=f"Downloading audiocpp_cli from {plan.cli_asset_name}",
            )
        cli_path = await materialize_cli(binaries, plan, install_dir)
        row = {
            "version": version_name,
            "type": "prebuilt",
            "install_type": "prebuilt",
            "is_fork": False,
            "repository_source": "audio.cpp",
            "source_ref": source_ref,
            "source_ref_type": "release",
            "source_branch": source_ref,
            "source_repo": repository_url,
            "install_dir": install_dir,
            "installed_at": _utcnow(),
            "server_binary_path": binaries["server_binary_path"],
            "cli_binary_path": cli_path,
            "cuda_version": plan.host_cuda,
            "build_config": {
                "prebuilt": True,
                "backend": plan.backend,
                "asset_name": plan.asset_name,
                "cli_asset_name": plan.cli_asset_name,
                "package_cuda": plan.package_cuda,
                "host_cuda": plan.host_cuda,
                "selection_reason": plan.reason,
                "architectures": list(plan.architectures),
            },
        }
        from backend.engines.lifecycle import mark_engine_version_ready

        mark_engine_version_ready(store, "audio_cpp", row)
        if auto_activate:
            await _activate(version_name, covered_by=task_id)
        pm.complete_task(task_id, f"Installed audio.cpp {version_name}")
        await pm.send_notification(
            title="audio.cpp installed",
            message=f"Installed audio.cpp {version_name} ({plan.reason})",
            type="success",
            task_id=task_id,
        )
    except asyncio.CancelledError:
        from backend.engines.lifecycle import mark_engine_version_failed

        mark_engine_version_failed(
            store,
            "audio_cpp",
            version_name,
            error="audio.cpp install cancelled",
            cancelled=True,
            extra={"source_ref": source_ref, "install_dir": install_dir},
        )
        pm.fail_task(task_id, "audio.cpp install cancelled")
        raise
    except Exception as exc:
        from backend.engines.lifecycle import mark_engine_version_failed

        logger.exception("audio.cpp prebuilt install failed")
        mark_engine_version_failed(
            store,
            "audio_cpp",
            version_name,
            error=str(exc),
            extra={"source_ref": source_ref, "install_dir": install_dir},
        )
        pm.fail_task(task_id, str(exc))
        await pm.send_notification(
            title="audio.cpp install failed",
            message=str(exc),
            type="error",
            task_id=task_id,
        )


async def _schedule_prebuilt(payload: dict, plan) -> dict:
    store = get_store()
    source_ref = str(payload.get("source_ref") or "").strip()
    repository_url = str(payload.get("repository_url") or AUDIO_CPP_REPOSITORY).strip()
    version_name = f"prebuilt-{_version_slug(source_ref)}"
    if any(
        str(row.get("version")) == version_name
        for row in store.get_engine_versions("audio_cpp")
    ):
        version_name = f"{version_name}-{_version_slug(str(int(time.time())))}"
    task_id = f"install_audio_cpp_{_version_slug(version_name)}_{int(time.time())}"
    from backend.engines.lifecycle import mark_engine_version_building

    pm = get_progress_manager()
    install_dir = os.path.join(get_audio_cpp_manager().builds_dir, version_name)
    _guard_build_admission(
        lambda: pm.create_task(
            "install",
            f"Install audio.cpp {source_ref}",
            {
                "engine": "audio_cpp",
                "version_name": version_name,
                "repository_source": "audio.cpp",
                "source_ref": source_ref,
                "source_ref_type": "release",
                "asset_name": plan.asset_name,
                "auto_activate": bool(payload.get("auto_activate", True)),
                "resource_key": engine_installation_key("audio_cpp", install_dir),
                "depends_on": CUDA_TOOLKIT_KEY,
            },
            task_id=task_id,
        )
    )
    await _fence_build(task_id)
    build_config = payload.get("build_config") if isinstance(payload.get("build_config"), dict) else {}
    mark_engine_version_building(
        store,
        "audio_cpp",
        {
            "version": version_name,
            "type": "prebuilt",
            "install_type": "prebuilt",
            "source_ref": source_ref,
            "source_ref_type": "release",
            "source_branch": source_ref,
            "source_repo": repository_url,
            "repository_source": "audio.cpp",
            "install_dir": install_dir,
            "build_config": {
                **build_config,
                "prebuilt": True,
                "asset_name": plan.asset_name,
                "package_cuda": plan.package_cuda,
                "host_cuda": plan.host_cuda,
                "selection_reason": plan.reason,
            },
            "installed_at": _utcnow(),
        },
        task_id=task_id,
    )
    get_supervisor().spawn(
        task_id,
        _prebuilt_task(
            task_id=task_id,
            version_name=version_name,
            source_ref=source_ref,
            repository_url=repository_url,
            plan=plan,
            auto_activate=bool(payload.get("auto_activate", True)),
        ),
    )
    store.update_engine_build_settings(
        "audio_cpp",
        merge_settings(
            tracking_ref=source_ref,
            repository_url=repository_url,
            build_config=build_config,
            existing=store.get_engine_build_settings("audio_cpp"),
        ),
    )
    return {
        "message": f"Installing audio.cpp {source_ref}",
        "task_id": task_id,
        "status": "started",
        "version_name": version_name,
        "source_ref": source_ref,
        "source_ref_type": "release",
        "prebuilt": True,
        "asset_name": plan.asset_name,
        "prebuilt_reason": plan.reason,
    }


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
    from backend.engines.audio_cpp.model_managers import resolve_model_manager_path

    return {
        "installed": bool(store.get_engine_versions("audio_cpp")),
        "active": active,
        "runnable": bool(
            active
            and all(
                active.get(key) and os.path.isfile(str(active[key]))
                for key in required_active_path_fields("audio_cpp", active)
            )
        ),
        "models_root": get_audio_cpp_manager().models_dir,
        "model_manager_ready": bool(
            active and resolve_model_manager_path(version_row=active)
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
            "contract_grade": caps.get("contract_grade"),
            "families_without_tasks": [],
            "warnings": [],
        },
        "families": list(caps.get("families") or []),
        "tasks": list(caps.get("tasks") or []),
        "discovery_source": caps.get("discovery_source"),
        "catalog_source": caps.get("catalog_source"),
        "contract_grade": caps.get("contract_grade"),
        "contract_warnings": list(caps.get("contract_warnings") or []),
        "affected_models": _audio_models_affected_by_delta(
            store,
            (entry or {}).get("capability_delta"),
            contract_changed=bool((entry or {}).get("contract_changed")),
        ),
        "supported_build_backends": get_audio_cpp_manager().supported_build_backends(),
    }


@router.get("/build-options")
async def get_build_options():
    """Return the audio.cpp CMake build-option catalog for the settings UI."""
    return catalog_for_ui()


@router.get("/build-settings")
async def get_build_settings():
    store = get_store()
    settings = await ensure_tracking_settings(store)
    tracking, cmake = split_settings(settings)
    return {**coerce_build_settings(cmake), **tracking}


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
    build_config = coerce_build_settings(build_config)
    merged = merge_settings(
        tracking_ref=payload.get("tracking_ref"),
        repository_url=payload.get("repository_url"),
        build_config=build_config,
        existing=existing,
    )
    stored = store.update_engine_build_settings("audio_cpp", merged)
    tracking, cmake = split_settings(stored)
    return {**coerce_build_settings(cmake), **tracking}


@router.post("/build-source")
async def build_source(payload: dict = Body(default_factory=dict)):
    await ensure_tracking_settings()
    bind_action_confirmation(payload)
    return await _schedule_build(payload or {})


@router.post("/update")
async def update(payload: dict = Body(default_factory=dict)):
    store = get_store()
    settings = await ensure_tracking_settings(store)
    tracking, cmake = split_settings(settings)
    payload = payload or {}
    repository_url = str(
        payload.get("repository_url") or tracking.get("repository_url") or AUDIO_CPP_REPOSITORY
    ).strip()

    # Mirror llama.cpp "From release": build the latest GitHub release tag.
    # audio.cpp tags are ``release-X.Y(.Z)`` (not plain ``v*``). When the
    # tracking/source ref is already a release tag, Update also advances to
    # the newest published release (tags are immutable; following an old tag
    # tip never yields a newer version).
    want_latest_release = bool(
        payload.get("from_release") or payload.get("use_latest_release")
    )
    candidate = str(
        payload.get("source_ref") or tracking.get("tracking_ref") or AUDIO_CPP_DEFAULT_REF
    ).strip()
    if want_latest_release or is_audio_release_tag(candidate):
        tag = await asyncio.to_thread(resolve_latest_release_tag)
        if not tag:
            if want_latest_release:
                raise HTTPException(
                    status_code=404,
                    detail="No GitHub release found for audio.cpp",
                )
            ref = candidate
            ref_kind = _ref_kind(ref)
        else:
            ref = tag
            ref_kind = "release"
    else:
        ref = candidate
        ref_kind = _ref_kind(ref)

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

    latest = await _latest_upstream(ref, repository_url)
    active = store.get_active_engine_version("audio_cpp")
    active_branch = str((active or {}).get("source_branch") or "").strip()

    # Prefer in-place sync when the active install already tracks this branch/tag
    if (
        not want_latest_release
        and active
        and active_branch
        and active_branch == ref
        and ref_kind in {"branch", "release"}
        and active.get("source_path")
    ):
        bind_action_confirmation(payload)
        return await schedule_audio_cpp_sync(active, ref, build_config)

    bind_action_confirmation(payload)
    if ref_kind == "release":
        prebuilt_plan = None
        skip_reason = ""
        try:
            from backend.engines.audio_cpp.prebuilt import choose_release_prebuilt

            prebuilt_plan, skip_reason = await choose_release_prebuilt(ref)
        except Exception as exc:
            logger.info(
                "audio.cpp prebuilt lookup for %s failed, building from source: %s",
                ref,
                exc,
            )
            skip_reason = "The prebuilt release could not be read, so this release will be built from source."
        if prebuilt_plan is not None:
            return await _schedule_prebuilt(
                {
                    **payload,
                    "source_ref": ref,
                    "repository_url": repository_url,
                    "build_config": build_config.__dict__,
                    "auto_activate": True,
                },
                prebuilt_plan,
            )
        if skip_reason:
            logger.info("audio.cpp %s: %s", ref, skip_reason)

    # Rebuild as a syncable branch/tag install (not a detached tip SHA)
    scheduled = await _schedule_build(
        {
            **payload,
            "source_ref": ref,
            "source_ref_type": "release" if ref_kind == "release" else (
                ref_kind if ref_kind != "commit" else "branch"
            ),
            "repository_url": repository_url,
            "build_config": build_config.__dict__,
            "version_suffix": (latest.get("sha") or "")[:8] or str(int(time.time())),
            "auto_activate": True,
        }
    )
    if ref_kind == "release" and skip_reason:
        scheduled["prebuilt"] = False
        scheduled["prebuilt_skipped"] = skip_reason
    return scheduled


@router.get("/check-updates")
async def check_updates(ref: Optional[str] = None):
    store = get_store()
    settings = await ensure_tracking_settings(store)
    tracking, _cmake = split_settings(settings)
    track_ref = str(ref or tracking.get("tracking_ref") or AUDIO_CPP_DEFAULT_REF).strip()
    repository_url = str(
        tracking.get("repository_url") or AUDIO_CPP_REPOSITORY
    ).strip()
    active = store.get_active_engine_version("audio_cpp")
    current = (active or {}).get("source_commit")
    active_ref = str((active or {}).get("source_ref") or "").strip()
    entry = (
        get_version_entry(store, "audio_cpp", str(active.get("version") or ""))
        if active
        else None
    )

    # The configured (or explicitly requested) ref defines the channel. An
    # old active release must not override an explicit branch check.
    release_channel = is_audio_release_tag(track_ref)
    latest_release = (
        await asyncio.to_thread(resolve_latest_github_release)
        if release_channel
        else None
    )

    if release_channel and latest_release and latest_release.get("tag_name"):
        release_tag = str(latest_release["tag_name"])
        tip = await _latest_upstream(release_tag, repository_url)
        installed_ref = active_ref or track_ref
        newer_tag = bool(installed_ref and installed_ref != release_tag)
        tip_moved = bool(current and tip.get("sha") and current != tip["sha"])
        update_available = newer_tag or tip_moved
        latest_version = release_tag
    else:
        tip = await _latest_upstream(track_ref, repository_url)
        update_available = bool(
            current and tip.get("sha") and current != tip["sha"]
        )
        latest_version = tip.get("sha")

    return {
        "current_version": current,
        "latest_version": latest_version,
        "update_available": update_available,
        "latest_commit": tip,
        "latest_release": latest_release if release_channel else None,
        "update_channel": "release" if release_channel else "branch",
        "tracking_ref": track_ref,
        "repository_url": repository_url,
        "contract_fingerprint": (entry or {}).get("contract_fingerprint"),
    }


@router.post("/cancel")
async def cancel(payload: dict = Body(default_factory=dict)):
    task_id = str((payload or {}).get("task_id") or "").strip()
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return BuildTaskManager.cancel(task_id)

