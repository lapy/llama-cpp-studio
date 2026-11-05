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
from backend.engines.audio_cpp.identity import default_audio_engine_id
from backend.engines.definitions.extensions import assert_extensions_served
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

# Engine identity comes from the definition (see audio_cpp/identity.py). The
# router is mounted per declared audio engine; requests set this context so the
# same extension surface can serve more than one definition.
from contextvars import ContextVar

_DEFAULT_ENGINE_ID = default_audio_engine_id()
_CURRENT_ENGINE: ContextVar[str] = ContextVar("audio_engine_id", default=_DEFAULT_ENGINE_ID)


def current_engine_id() -> str:
    return _CURRENT_ENGINE.get()


def set_engine_context(engine_id: str):
    return _CURRENT_ENGINE.set(str(engine_id or _DEFAULT_ENGINE_ID))


def reset_engine_context(token) -> None:
    _CURRENT_ENGINE.reset(token)


def segment_for_engine(engine_id: str) -> str:
    return str(engine_id or "").replace("_", "-")


def engine_for_path(path: str) -> str:
    from backend.engines.audio_cpp.identity import audio_engine_ids

    requested = str(path or "")
    for engine_id in audio_engine_ids():
        if requested.startswith(f"/api/{segment_for_engine(engine_id)}/"):
            return engine_id
    return _DEFAULT_ENGINE_ID


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


def release_version_name(tag: str) -> str:
    """Version id shared by a release archive and a source build of that tag."""
    return _version_slug(tag)


def _release_type_labels() -> Dict[str, Any]:
    return {"type": "release", "install_type": "release", "is_fork": False}


def _claim_version_name(store, base: str) -> str:
    """Return ``base``, or ``base-<time>`` when that release version already exists."""
    taken = {
        str(row.get("version") or "") for row in store.get_engine_versions(current_engine_id())
    }
    if base not in taken:
        return base
    return f"{base}-{_version_slug(str(int(time.time())))}"


def _github_api_repo_slug(repository_url: str) -> Optional[str]:
    """Return ``owner/repo`` for GitHub clone URLs, else ``None``."""
    value = str(repository_url or "").strip().rstrip("/")
    match = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?$", value, re.I)
    if not match:
        return None
    return f"{match.group(1)}/{match.group(2)}"


async def _latest_upstream(ref: str, repository_url: Optional[str] = None) -> Dict[str, Any]:
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
        current_engine_id(),
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
    from dataclasses import asdict

    from backend.engines.audio_cpp.installer import AudioCppInstaller

    manager = get_audio_cpp_manager()
    store = get_store()
    pm = get_progress_manager()
    builds_dir = getattr(manager, "builds_dir", "") or ""
    version_dir = os.path.join(builds_dir, version_name) if builds_dir else ""
    if source_ref_type == "release":
        type_labels = _release_type_labels()
    else:
        type_labels = source_build_type_labels_for_engine(current_engine_id(), repository_url)
    # Phase 7 cutover: the definition's `source` recipe runs on the native host
    # (AudioCppInstaller owns the workspace, cmake and publication steps); the
    # route keeps its admission, task, notifications and failure bookkeeping.
    installer = AudioCppInstaller(install_root=builds_dir or None)
    installer._record_pending_snapshot(
        version_name,
        {
            "install_type": type_labels["install_type"],
            "requested_version": source_ref,
            "source_ref": source_ref,
            "source_repo": repository_url,
            "install_dir": version_dir,
        },
    )
    try:
        execution = await installer.execute_operation(
            "source",
            task_id=task_id,
            install_dir=version_dir or None,
            create_install_dir=False,
            context={
                "input.install_dir": version_dir,
                "input.version_name": version_name,
                "input.install_type": type_labels["install_type"],
                "input.type_labels": type_labels,
                "input.source_ref": source_ref,
                "input.source_ref_type": source_ref_type,
                "input.repository_url": repository_url,
                "input.source_repo": repository_url,
                "input.build_config": asdict(build_config.normalized()),
                "input.replace_existing": bool(replace_existing),
                "input.use_workspace": bool(use_workspace),
                "input.auto_activate": bool(auto_activate),
            },
            store=store,
        )
        if execution.status != "completed":
            failure = execution.failed_step
            raise RuntimeError(
                failure.error
                if failure is not None and failure.error
                else "audio.cpp source build did not complete"
            )
        if not auto_activate:
            try:
                from backend.proxy.control import mark_runtime_changed

                mark_runtime_changed()
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
            current_engine_id(),
            version_name,
            error="audio.cpp build cancelled",
            cancelled=True,
            extra={
                "source_ref": source_ref,
                "source_repo": repository_url,
                "repository_source": "audio.cpp",
                "install_dir": version_dir or None,
            },
        )
        pm.fail_task(task_id, "audio.cpp build cancelled")
        raise
    except Exception as exc:
        from backend.engines.lifecycle import mark_engine_version_failed

        logger.exception("audio.cpp source build failed")
        mark_engine_version_failed(
            store,
            current_engine_id(),
            version_name,
            error=str(exc),
            extra={
                "source_ref": source_ref,
                "source_repo": repository_url,
                "repository_source": "audio.cpp",
                "install_dir": version_dir or None,
            },
        )
        pm.fail_task(task_id, str(exc))
        await pm.send_notification(
            title="audio.cpp build failed",
            message=str(exc),
            type="error",
            task_id=task_id,
        )
    finally:
        installer._release_active_workspace()


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
            for item in store.get_engine_versions(current_engine_id())
            if str(item.get("version")) == str(version_name)
        ),
        None,
    )
    if not version_row:
        pm.fail_task(task_id, f"audio.cpp version '{version_name}' not found")
        return
    builds_dir = getattr(manager, "builds_dir", "") or ""
    repo_url = str(version_row.get("source_repo") or "").strip()
    existing_kind = (
        str(version_row.get("install_type") or version_row.get("type") or "").strip().lower()
    )
    if existing_kind == "local":
        type_labels = {"type": "local", "install_type": "local", "is_fork": False}
    elif existing_kind == "release":
        type_labels = _release_type_labels()
    else:
        type_labels = source_build_type_labels_for_engine(current_engine_id(), repo_url)
    from dataclasses import asdict

    from backend.engines.audio_cpp.installer import AudioCppInstaller

    installer = AudioCppInstaller(install_root=builds_dir or None)
    installer._record_pending_snapshot(
        version_name,
        {
            "install_type": type_labels["install_type"],
            "requested_version": branch,
            "source_ref": branch,
            "source_repo": repo_url,
            "install_dir": str(version_row.get("install_dir") or ""),
        },
    )
    try:
        # Phase 7 cutover: the definition's `sync` recipe runs on the native
        # host; the route keeps its scan/activation policy and messaging.
        execution = await installer.execute_operation(
            "sync",
            task_id=task_id,
            install_dir=str(version_row.get("install_dir") or "") or None,
            context={
                "input.version_entry": dict(version_row),
                "input.version_name": version_name,
                "input.install_dir": str(version_row.get("install_dir") or ""),
                "input.type_labels": type_labels,
                "input.source_ref": branch,
                "input.source_ref_type": "branch",
                "input.repository_url": repo_url,
                "input.source_repo": repo_url,
                "input.branch": branch,
                "input.build_config": asdict(build_config.normalized()),
                "input.auto_activate": False,
            },
            store=store,
        )
        if execution.status != "completed":
            failure = execution.failed_step
            raise RuntimeError(
                failure.error
                if failure is not None and failure.error
                else "audio.cpp source sync did not complete"
            )
        updated = next(
            (
                item
                for item in store.get_engine_versions(current_engine_id())
                if str(item.get("version")) == str(version_name)
            ),
            None,
        )
        if not updated:
            raise RuntimeError(f"Version '{version_name}' disappeared during sync")
        try:
            from backend.proxy.control import mark_runtime_changed

            mark_runtime_changed()
        except Exception:
            pass
        # Keep a synced branch install active when it already was, or activate
        # it when nothing else is. ``_activate`` already runs the parameter
        # scan, so skip the extra scan on that path (it opened a second tray card).
        active = store.get_active_engine_version(current_engine_id())
        will_activate = not active or str(active.get("version")) == str(version_name)
        if not will_activate:
            try:
                from backend.engines.scan.scanner import scan_engine_version

                scan_engine_version(store, current_engine_id(), updated)
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
    finally:
        installer._release_active_workspace()


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


async def schedule_audio_cpp_sync(
    version_entry: dict, branch: str, build_config: AudioCppBuildConfig
) -> dict:
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
                "engine": current_engine_id(),
                "version_name": version_name,
                "repository_source": "audio.cpp",
                "source_ref": branch,
                "source_ref_type": "branch",
                "sync": True,
                "resource_key": engine_installation_key(current_engine_id(), install_dir),
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
        ),
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
    # Snapshot-backed retry: refuse a changed definition and prefer the frozen
    # source identity over the live row.
    from backend.engines.definitions.loader import load_default_engine_definitions
    from backend.engines.install_snapshots import retry_resolved_inputs

    definition = load_default_engine_definitions().get(manager.engine_id)
    try:
        frozen = retry_resolved_inputs(
            store,
            manager.engine_id,
            version_name,
            current_digest=str(getattr(definition, "definition_digest", "") or ""),
            label=str(getattr(definition, "label", manager.engine_id) or manager.engine_id),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    source_ref = str(
        version_entry.get("source_ref")
        or version_entry.get("source_branch")
        or version_entry.get("source_commit")
        or frozen.get("source_ref")
        or frozen.get("branch")
        or ""
    ).strip()
    repository_url = str(
        version_entry.get("source_repo") or frozen.get("source_repo") or AUDIO_CPP_REPOSITORY
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
            store.get_engine_build_settings(current_engine_id())
        )
    try:
        manager.validate_build_config(build_config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    task_id = f"build_retry_{_version_slug(version_name)}_{int(time.time())}"
    from backend.engines.lifecycle import mark_engine_version_building

    type_labels = source_build_type_labels_for_engine(current_engine_id(), repository_url)
    install_dir = str(
        version_entry.get("install_dir") or os.path.join(manager.builds_dir, version_name)
    )
    pm = get_progress_manager()
    _guard_build_admission(
        lambda: pm.create_task(
            "update",
            f"Retry audio.cpp {version_name}",
            {
                "engine": current_engine_id(),
                "version_name": version_name,
                "repository_source": "audio.cpp",
                "source_ref": source_ref,
                "source_ref_type": source_ref_type,
                "retry": True,
                "resource_key": engine_installation_key(current_engine_id(), install_dir),
                "depends_on": CUDA_TOOLKIT_KEY,
            },
            task_id=task_id,
        )
    )
    await _fence_build(task_id)
    mark_engine_version_building(
        store,
        current_engine_id(),
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
        ),
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
    tracking, _cmake = split_settings(store.get_engine_build_settings(current_engine_id()))
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
    from backend.engines.lifecycle import mark_engine_version_building
    from backend.repo_identity import source_build_type_labels_for_engine as _labels

    if source_ref_type == "release":
        type_labels = _release_type_labels()
        version_name = _claim_version_name(store, release_version_name(source_ref))
    else:
        type_labels = _labels(current_engine_id(), repository_url)
        suffix = str(payload.get("version_suffix") or int(time.time())).strip()
        version_name = f"source-{_version_slug(source_ref)}-{_version_slug(suffix)}"
        if any(
            str(row.get("version")) == version_name
            for row in store.get_engine_versions(current_engine_id())
        ):
            raise HTTPException(status_code=409, detail=f"Version '{version_name}' already exists")

    task_id = f"build_audio_cpp_{_version_slug(version_name)}_{int(time.time())}"
    pm = get_progress_manager()
    install_dir = os.path.join(manager.builds_dir, version_name)
    _guard_build_admission(
        lambda: pm.create_task(
            "build",
            f"Build audio.cpp {source_ref}",
            {
                "engine": current_engine_id(),
                "version_name": version_name,
                "repository_source": "audio.cpp",
                "source_ref": source_ref,
                "source_ref_type": source_ref_type,
                "backend": build_config.backend,
                "auto_activate": bool(payload.get("auto_activate", True)),
                "resource_key": engine_installation_key(current_engine_id(), install_dir),
                "depends_on": CUDA_TOOLKIT_KEY,
            },
            task_id=task_id,
        )
    )
    await _fence_build(task_id)
    mark_engine_version_building(
        store,
        current_engine_id(),
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
        ),
    )
    # Persist tracking when the user builds from an explicit branch/tag
    if source_ref_type in {"branch", "release"}:
        store.update_engine_build_settings(
            current_engine_id(),
            merge_settings(
                tracking_ref=source_ref,
                repository_url=repository_url,
                build_config=build_config.__dict__,
                existing=store.get_engine_build_settings(current_engine_id()),
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
    from urllib.parse import urlparse

    from backend.engines.audio_cpp.installer import AudioCppInstaller

    store = get_store()
    pm = get_progress_manager()
    manager = get_audio_cpp_manager()
    install_dir = os.path.join(manager.builds_dir, version_name)
    archive_path = os.path.join(
        install_dir, os.path.basename(urlparse(str(plan.url)).path) or "audio.cpp.tar.gz"
    )
    installer = AudioCppInstaller(install_root=manager.builds_dir)
    # Phase 7 cutover: the prebuilt branch executes the `prebuilt` recipe
    # through the definition-driven native host (AudioCppInstaller). The route
    # keeps its task/admission/notification plumbing around it; the archive
    # deletion stays at the legacy point (a plan step, right after the binaries
    # are located) and this finally mirrors the legacy download cleanup when
    # cancellation strikes mid-download.
    try:
        execution = await installer.execute_operation(
            "prebuilt",
            task_id=task_id,
            install_dir=install_dir,
            context={
                "input.plan": plan,
                "input.plan_url": str(plan.url),
                "input.download_label": f"Downloading {plan.asset_name}",
                "input.archive_path": archive_path,
                "input.install_dir": install_dir,
                "input.install_type": "release",
                "input.version_name": version_name,
                "input.source_ref": source_ref,
                "input.source_ref_type": "release",
                "input.source_repo": repository_url,
                "input.repository_url": repository_url,
                "input.auto_activate": bool(auto_activate),
            },
        )
        if execution.status != "completed":
            failure = execution.failed_step
            raise RuntimeError(
                failure.error
                if failure is not None and failure.error
                else "audio.cpp prebuilt install did not complete"
            )
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
            current_engine_id(),
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
            current_engine_id(),
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
    finally:
        if os.path.isfile(archive_path):
            os.remove(archive_path)


async def _schedule_prebuilt(payload: dict, plan) -> dict:
    store = get_store()
    source_ref = str(payload.get("source_ref") or "").strip()
    repository_url = str(payload.get("repository_url") or AUDIO_CPP_REPOSITORY).strip()
    version_name = _claim_version_name(store, release_version_name(source_ref))
    task_id = f"install_audio_cpp_{_version_slug(version_name)}_{int(time.time())}"
    from backend.engines.lifecycle import mark_engine_version_building

    pm = get_progress_manager()
    install_dir = os.path.join(get_audio_cpp_manager().builds_dir, version_name)
    _guard_build_admission(
        lambda: pm.create_task(
            "install",
            f"Install audio.cpp {source_ref}",
            {
                "engine": current_engine_id(),
                "version_name": version_name,
                "repository_source": "audio.cpp",
                "source_ref": source_ref,
                "source_ref_type": "release",
                "asset_name": plan.asset_name,
                "auto_activate": bool(payload.get("auto_activate", True)),
                "resource_key": engine_installation_key(current_engine_id(), install_dir),
                "depends_on": CUDA_TOOLKIT_KEY,
            },
            task_id=task_id,
        )
    )
    await _fence_build(task_id)
    build_config = (
        payload.get("build_config") if isinstance(payload.get("build_config"), dict) else {}
    )
    mark_engine_version_building(
        store,
        current_engine_id(),
        {
            "version": version_name,
            **_release_type_labels(),
            "source_ref": source_ref,
            "source_ref_type": "release",
            "source_branch": source_ref,
            "source_repo": repository_url,
            "repository_source": "audio.cpp",
            "install_dir": install_dir,
            "build_config": {
                **build_config,
                "prebuilt": True,
                "release_tag": source_ref,
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
        current_engine_id(),
        merge_settings(
            tracking_ref=source_ref,
            repository_url=repository_url,
            build_config=build_config,
            existing=store.get_engine_build_settings(current_engine_id()),
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
    active = store.get_active_engine_version(current_engine_id())
    entry = (
        get_version_entry(store, current_engine_id(), str(active.get("version") or ""))
        if active
        else None
    )
    caps = (entry or {}).get("capabilities") or {}
    from backend.engines.audio_cpp.model_managers import resolve_model_manager_path

    return {
        "installed": bool(store.get_engine_versions(current_engine_id())),
        "active": active,
        "runnable": bool(
            active
            and all(
                active.get(key) and os.path.isfile(str(active[key]))
                for key in required_active_path_fields(current_engine_id(), active)
            )
        ),
        "models_root": get_audio_cpp_manager().models_dir,
        "model_manager_ready": bool(active and resolve_model_manager_path(version_row=active)),
        "tracking_ref": tracking.get("tracking_ref"),
        "repository_url": tracking.get("repository_url") or AUDIO_CPP_REPOSITORY,
        "contract_fingerprint": (entry or {}).get("contract_fingerprint"),
        "contract_changed": bool((entry or {}).get("contract_changed")),
        "previous_contract_fingerprint": (entry or {}).get("previous_contract_fingerprint"),
        "capability_delta": (entry or {}).get("capability_delta")
        or {
            "added_families": [],
            "removed_families": [],
            "added_tasks": [],
            "removed_tasks": [],
            "families_without_tasks": [],
            "warnings": [],
        },
        "families": list(caps.get("families") or []),
        "tasks": list(caps.get("tasks") or []),
        "discovery_source": caps.get("discovery_source"),
        "catalog_source": caps.get("catalog_source"),
        "contract_schema": caps.get("contract_schema"),
        "drift_warnings": list((entry or {}).get("contract", {}).get("drift_warnings") or [])
        if isinstance((entry or {}).get("contract"), dict)
        else [],
        "affected_models": _audio_models_affected_by_delta(
            store,
            (entry or {}).get("capability_delta"),
            contract_changed=bool((entry or {}).get("contract_changed")),
        ),
        "supported_build_backends": get_audio_cpp_manager().supported_build_backends(),
    }


@router.get("/contract")
async def get_integration_contract():
    """Composed audio.cpp integration contract for the active version.

    The same descriptor the backend consumers read (operations, task routing,
    family routing, defaults keys, drift warnings). Per-model composition rides
    the param-registry payload; the frontend renders forms from these two.
    """
    store = get_store()
    active = store.get_active_engine_version(current_engine_id()) or {}
    entry = (
        get_version_entry(store, current_engine_id(), str(active.get("version") or ""))
        if active
        else None
    )
    descriptor = (entry or {}).get("contract")
    if not isinstance(descriptor, dict):
        raise HTTPException(
            status_code=409,
            detail=(
                "No composed audio.cpp contract is available. Activate a "
                "v0.9.1+ audio.cpp version (JSON loader listing required) and "
                "let its parameter scan finish."
            ),
        )
    return {
        **descriptor,
        "engine_version": str(active.get("version") or ""),
        "contract_fingerprint": (entry or {}).get("contract_fingerprint"),
        "contract_changed": bool((entry or {}).get("contract_changed")),
        "scanned_at": (entry or {}).get("scanned_at"),
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
    existing = store.get_engine_build_settings(current_engine_id()) or {}
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
    stored = store.update_engine_build_settings(current_engine_id(), merged)
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
    want_latest_release = bool(payload.get("from_release") or payload.get("use_latest_release"))
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
        current_engine_id(),
        merge_settings(
            tracking_ref=ref,
            repository_url=repository_url,
            build_config=build_config.__dict__,
            existing=settings,
        ),
    )

    latest = await _latest_upstream(ref, repository_url)
    active = store.get_active_engine_version(current_engine_id())
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
            skip_reason = (
                "The prebuilt release could not be read, so this release will be built from source."
            )
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
        from backend.engines.audio_cpp.prebuilt import detect_host_target

        host = await detect_host_target()
        if host.has_nvidia and host.cuda is not None:
            build_config.cuda = True
            build_config.normalized()

    # A release tag is registered like the prebuilt. A branch stays a source build.
    scheduled = await _schedule_build(
        {
            **payload,
            "source_ref": ref,
            "source_ref_type": "release"
            if ref_kind == "release"
            else (ref_kind if ref_kind != "commit" else "branch"),
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
    repository_url = str(tracking.get("repository_url") or AUDIO_CPP_REPOSITORY).strip()
    active = store.get_active_engine_version(current_engine_id())
    current = (active or {}).get("source_commit")
    active_ref = str((active or {}).get("source_ref") or "").strip()
    entry = (
        get_version_entry(store, current_engine_id(), str(active.get("version") or ""))
        if active
        else None
    )

    # The configured (or explicitly requested) ref defines the channel. An
    # old active release must not override an explicit branch check.
    release_channel = is_audio_release_tag(track_ref)
    latest_release = (
        await asyncio.to_thread(resolve_latest_github_release) if release_channel else None
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
        update_available = bool(current and tip.get("sha") and current != tip["sha"])
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


@router.get("/community-voices")
async def get_community_voices():
    """List the active checkout's demo voices and whether they are installed."""
    from backend.audio.community_voices import library_status

    active = get_store().get_active_engine_version(current_engine_id()) or {}
    return library_status(source_root=str(active.get("source_path") or ""))


@router.post("/community-voices/install")
async def install_community_voices():
    """Install demo voices from the active checkout, or the pinned GitHub commit."""
    from backend.audio.community_voices import CommunityVoiceError, install_community_voices
    from backend.proxy.control import mark_runtime_changed

    active = get_store().get_active_engine_version(current_engine_id()) or {}
    source_root = str(active.get("source_path") or "")
    try:
        payload = await install_community_voices(source_root=source_root)
    except CommunityVoiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not store community preset voices: {exc}",
        ) from exc
    try:
        mark_runtime_changed()
    except Exception as exc:
        logger.debug("mark_runtime_changed: %s", exc)
    return payload


@router.delete("/community-voices")
async def remove_community_voices():
    """Remove the installed community preset voice library from the data root."""
    from backend.audio.community_voices import CommunityVoiceError
    from backend.audio.community_voices import remove_community_voices as remove_library
    from backend.proxy.control import mark_runtime_changed

    active = get_store().get_active_engine_version(current_engine_id()) or {}
    source_root = str(active.get("source_path") or "")
    try:
        payload = await asyncio.to_thread(remove_library, source_root=source_root)
    except CommunityVoiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not remove community preset voices: {exc}",
        ) from exc
    try:
        mark_runtime_changed()
    except Exception as exc:
        logger.debug("mark_runtime_changed: %s", exc)
    return payload


# The engine's declared extension surfaces (schema-v2 ``extensions``) must be
# served by THIS router: a drifted declaration refuses import/startup instead of
# silently describing endpoints that no longer exist. (Single implementation:
# `assert_extensions_served`, also used by the surface gate test.)
assert_extensions_served(current_engine_id(), router.routes)
