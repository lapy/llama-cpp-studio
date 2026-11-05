"""Shared HTTP surface for Python-engine installers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
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


def github_commit_updates(
    repo: str,
    *,
    branch: str = "main",
    timeout: float = 15.0,
    error_status: int = 502,
) -> UpdateSource:
    return UpdateSource(
        kind="github_commit",
        repo=repo,
        branch=branch,
        url=f"https://github.com/{repo}/commits/{branch}",
        timeout=timeout,
        error_status=error_status,
    )


def github_release_updates(
    repo: str, *, timeout: float = 10.0, error_status: int = 500
) -> UpdateSource:
    return UpdateSource(
        kind="github_releases", repo=repo, timeout=timeout, error_status=error_status
    )


@dataclass(frozen=True)
class EngineLifecycle:
    """Everything the shared lifecycle needs about one engine.

    ``definition_compat_router`` and the native lifecycle registrations create
    one record per engine; the generic ``/api/engines/{engine_id}`` router
    (``routes/engine_lifecycle.py``) resolves against the same records, so
    the legacy per-engine prefixes and the generic surface run ONE
    implementation — bodies and error contracts are identical by
    construction (pinned by tests/test_generic_engine_router.py).

    Native (llama-family) engines share the generic route SPACE but not the
    venv installer machinery: they register a ``verbs`` override map whose
    callables delegate to their own route functions, so both surfaces still
    run the same implementation byte-for-byte. A verb that is neither
    overridden nor backed by an installer answers 405 — e.g. native
    engines have no per-engine ``status`` (versions ride the shared
    ``/api/llama-versions`` list) and no engine-wide ``remove`` (deletion
    is version-scoped). The overrides are exercised by
    tests/test_native_engine_lifecycle.py."""

    engine_id: str
    url_prefix: str
    get_installer: Optional[Callable[[], EngineInstaller]] = None
    update_source: Optional[UpdateSource] = None
    prefer_source_install: bool = False
    status_fallback: Optional[Dict[str, Any]] = None
    version_setting_key: str = "pip_version"
    verbs: Dict[str, Callable[..., Any]] = field(default_factory=dict)


# Populated by the compatibility routers, native route registrations and
# register_definition_lifecycles at import/startup; read per-request by both
# router surfaces.
LIFECYCLE: Dict[str, EngineLifecycle] = {}


def _verb_unavailable(lc: EngineLifecycle, verb: str) -> HTTPException:
    return HTTPException(
        status_code=405, detail=f"{verb} is not available for engine '{lc.engine_id}'"
    )


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def _saved(lc: EngineLifecycle) -> Dict[str, Any]:
    return coerce_install_settings(
        lc.engine_id, get_store().get_engine_build_settings(lc.engine_id) or {}
    )


def _source_target(payload: Dict[str, Any], saved: Dict[str, Any]) -> tuple[str, str]:
    """Resolve an explicit source target before falling back to saved defaults.

    ``repo_url`` is the original install API field, while build settings use
    ``source_repo``.  Accept both (and the common ``repository_url`` alias)
    so a caller cannot save or submit a fork URL only for the install route
    to silently choose the maintained default repository.
    """
    repo = next(
        (
            payload.get(key)
            for key in ("repo_url", "source_repo", "repository_url")
            if str(payload.get(key) or "").strip()
        ),
        saved.get("source_repo") or "",
    )
    branch = next(
        (
            payload.get(key)
            for key in ("branch", "source_branch", "ref")
            if str(payload.get(key) or "").strip()
        ),
        saved.get("source_branch") or "main",
    )
    return str(repo).strip(), str(branch).strip()


async def lifecycle_check_updates(lc: EngineLifecycle) -> Dict[str, Any]:
    fn = lc.verbs.get("check_updates")
    if fn is not None:
        return await _maybe_await(fn())
    if lc.update_source is None:
        raise _verb_unavailable(lc, "check-updates")
    return await _check_updates(lc.engine_id, lc.update_source)


async def lifecycle_status(lc: EngineLifecycle) -> Dict[str, Any]:
    fn = lc.verbs.get("status")
    if fn is not None:
        return await _maybe_await(fn())
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "status")
    try:
        return lc.get_installer().status()
    except Exception as exc:
        logger.warning("%s/status: %s", lc.url_prefix.rstrip("/"), exc)
        fallback = {
            "engine": lc.engine_id,
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
        if lc.status_fallback:
            fallback.update(lc.status_fallback)
        return fallback


async def lifecycle_get_build_settings(lc: EngineLifecycle) -> Dict[str, Any]:
    fn = lc.verbs.get("get_build_settings")
    if fn is not None:
        return await _maybe_await(fn())
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "build-settings")
    return _saved(lc)


async def lifecycle_save_build_settings(
    lc: EngineLifecycle, settings: Optional[Dict]
) -> Dict[str, Any]:
    fn = lc.verbs.get("save_build_settings")
    if fn is not None:
        return await _maybe_await(fn(settings))
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "build-settings")
    if settings is not None and not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    coerced = coerce_install_settings(lc.engine_id, settings or {})
    get_store().replace_engine_build_settings(lc.engine_id, coerced)
    return coerced


async def lifecycle_install(
    lc: EngineLifecycle, request: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    fn = lc.verbs.get("install")
    if fn is not None:
        result = await _maybe_await(fn(request))
        _record_install_snapshot(lc, request)
        return result
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "install")
    payload = request or {}
    from backend.operations.action_recovery import (
        ActionAdmissionError,
        bind_action_confirmation,
    )

    bind_action_confirmation(payload)
    saved = _saved(lc)
    try:
        if lc.prefer_source_install:
            repo_url, branch = _source_target(payload, saved)
            result = await lc.get_installer().install_from_source(
                repo_url=repo_url,
                branch=branch,
            )
            _record_install_snapshot(
                lc,
                payload,
                resolved_inputs={
                    "install_type": "source",
                    "repo_url": repo_url,
                    "branch": branch,
                },
            )
            return result
        version = payload.get("version")
        if version is None or not str(version).strip():
            version = saved.get(lc.version_setting_key) or None
        result = await lc.get_installer().install_release(
            version=str(version) if version else None,
            force_reinstall=bool(payload.get("force_reinstall")),
        )
        _record_install_snapshot(
            lc,
            payload,
            resolved_inputs={
                "install_type": "pip",
                "version": str(version) if version else "",
            },
        )
        return result
    except ActionAdmissionError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def lifecycle_install_source(
    lc: EngineLifecycle, request: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    fn = lc.verbs.get("install_source")
    if fn is not None:
        result = await _maybe_await(fn(request))
        _record_install_snapshot(lc, request)
        return result
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "install-source")
    payload = request or {}
    from backend.operations.action_recovery import (
        ActionAdmissionError,
        bind_action_confirmation,
    )

    bind_action_confirmation(payload)
    saved = _saved(lc)
    try:
        repo_url, branch = _source_target(payload, saved)
        result = await lc.get_installer().install_from_source(
            repo_url=repo_url,
            branch=branch,
        )
        _record_install_snapshot(
            lc,
            payload,
            resolved_inputs={
                "install_type": "source",
                "repo_url": repo_url,
                "branch": branch,
            },
        )
        return result
    except ActionAdmissionError as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _record_install_snapshot(
    lc: EngineLifecycle,
    request: Optional[Dict[str, Any]],
    *,
    resolved_inputs: Optional[Dict[str, Any]] = None,
) -> None:
    """YAML campaign S1: record WHICH engine definition/recipe was in
    effect for every install scheduled through the generic router (the
    plan's "Immutable definitions": digest + recipe version + resolved
    inputs). Deliberately best-effort — a bookkeeping failure must never
    change the install response (byte-preservation gates own the verbs);
    later slices read this ledger instead of reinterpreting old rows.

    ``resolved_inputs`` are the values the dispatch ACTUALLY used (version
    resolved from saved settings, repo/branch after defaults), not just the
    submitted request — otherwise a retry cannot reproduce the install.
    """
    try:
        from backend.engines.definitions.loader import load_default_engine_definitions

        definition = load_default_engine_definitions().definitions.get(lc.engine_id)
        if definition is None:
            return
        effective = coerce_install_settings(lc.engine_id, request or {})
        # Explicit request fields that are not install-settings defaults still
        # belong to the frozen resolution (version/repo/branch), including for
        # engines whose install verb is supplied directly.
        payload = request if isinstance(request, dict) else {}
        for key in ("version", "repo_url", "branch", "force_reinstall"):
            if payload.get(key) not in (None, ""):
                effective.setdefault(key, payload[key])
        if resolved_inputs:
            effective.update({str(k): v for k, v in resolved_inputs.items() if v is not None})
        get_store().record_install_snapshot(
            lc.engine_id,
            definition_digest=definition.definition_digest,
            recipe_version=f"schema-v{definition.schema_version}",
            resolved_inputs=effective,
        )
    except Exception:  # noqa: BLE001 — inert to the response by design
        logger.warning("install snapshot bookkeeping failed for %s", lc.engine_id, exc_info=True)


async def lifecycle_remove(lc: EngineLifecycle, payload: Optional[dict]) -> Dict[str, Any]:
    fn = lc.verbs.get("remove")
    if fn is not None:
        return await _maybe_await(fn(payload))
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "remove")
    from backend.utils.fs_ops import FilesystemRefusal

    confirmed = bool((payload or {}).get("retire_launch_references"))
    try:
        return await lc.get_installer().remove(retire_references=confirmed)
    except FilesystemRefusal as exc:
        raise HTTPException(status_code=409, detail=exc.detail) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


async def lifecycle_cancel(lc: EngineLifecycle, payload: dict) -> Dict[str, Any]:
    fn = lc.verbs.get("cancel")
    if fn is not None:
        return await _maybe_await(fn(payload))
    if lc.get_installer is None:
        raise _verb_unavailable(lc, "cancel")
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return lc.get_installer().cancel_task(str(task_id))


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


def mount_lifecycle_router(lc: EngineLifecycle) -> APIRouter:
    """Mount one lifecycle record's verbs under its declared URL prefix.

    The historic per-engine prefixes and the generic
    ``/api/engines/{engine_id}/…`` surface both resolve the SAME
    ``EngineLifecycle`` records, so there is exactly one implementation per
    verb regardless of which URL is called.
    """
    LIFECYCLE[lc.engine_id] = lc
    engine_id = lc.engine_id
    router = APIRouter()
    prefix = lc.url_prefix.rstrip("/")
    op = engine_id.replace("-", "_")

    def _current() -> EngineLifecycle:
        """Resolve the LIVE record per request.

        Both surfaces must observe the same record even if a test (or a future
        re-registration) replaces it after mount time.
        """
        return LIFECYCLE[engine_id]

    @router.get(f"{prefix}/check-updates", operation_id=f"{op}_check_updates")
    async def check_updates() -> Dict:
        return await lifecycle_check_updates(_current())

    @router.get(f"{prefix}/status", operation_id=f"{op}_status")
    async def status() -> Dict:
        return await lifecycle_status(_current())

    @router.get(f"{prefix}/build-settings", operation_id=f"{op}_get_build_settings")
    async def get_build_settings() -> Dict:
        return await lifecycle_get_build_settings(_current())

    @router.put(f"{prefix}/build-settings", operation_id=f"{op}_save_build_settings")
    async def save_build_settings(settings: Optional[Dict] = Body(None)) -> Dict:
        return await lifecycle_save_build_settings(_current(), settings)

    @router.post(f"{prefix}/install", operation_id=f"{op}_install")
    async def install(request: Optional[Dict[str, Any]] = None) -> Dict:
        return await lifecycle_install(_current(), request)

    @router.post(f"{prefix}/install-source", operation_id=f"{op}_install_source")
    async def install_source(request: Optional[Dict[str, Any]] = None) -> Dict:
        return await lifecycle_install_source(_current(), request)

    @router.post(f"{prefix}/remove", operation_id=f"{op}_remove")
    async def remove(payload: Optional[dict] = Body(default=None)) -> Dict:
        return await lifecycle_remove(_current(), payload)

    @router.post(f"{prefix}/cancel", operation_id=f"{op}_cancel")
    async def cancel(payload: dict = Body(...)) -> Dict:
        return await lifecycle_cancel(_current(), payload)

    return router


def definition_compat_router(engine_id: str, url_prefix: str) -> APIRouter:
    """Mount a definition-owned engine's legacy compatibility prefix.

    Every fact (update source/quirks, status fallback, version setting key,
    installer recipes, binary identity) comes from the authoritative
    definition; the module that owns the old URL supplies only the prefix.
    """
    from dataclasses import replace

    from backend.engines.definitions.loader import load_default_engine_definitions

    definition = load_default_engine_definitions().definitions.get(engine_id)
    if definition is None:
        raise RuntimeError(f"{engine_id} engine definition is unavailable")
    lc = build_definition_lifecycle(definition)
    return mount_lifecycle_router(replace(lc, url_prefix=url_prefix))


# --- Definition-driven lifecycle registration ---------------------------------------
#
# Every non-generated engine gets its generic-surface lifecycle from its
# definition's ``lifecycle`` section: converted shipped engines through the
# definition-derived compatibility router they mount at import, YAML-only
# engines through ``register_definition_lifecycles`` at startup. The
# exporter-generated engines keep their Python registrations byte-identically
# until their own conversion; there is never a second record for one engine.


def update_source_from_lifecycle(spec: Dict[str, Any]) -> UpdateSource:
    kind = str(spec.get("kind") or "")
    extra: Dict[str, Any] = {}
    if spec.get("timeout") is not None:
        extra["timeout"] = float(spec["timeout"])
    if spec.get("error_status") is not None:
        extra["error_status"] = int(spec["error_status"])
    if kind == "pypi":
        return pypi_updates(str(spec["package"]), **extra)
    if kind == "github_commit":
        return github_commit_updates(
            str(spec["repo"]), branch=str(spec.get("branch") or "main"), **extra
        )
    if kind == "github_releases":
        return github_release_updates(str(spec["repo"]), **extra)
    raise ValueError(f"unsupported lifecycle update_source kind {kind!r}")


def build_definition_lifecycle(definition) -> EngineLifecycle:
    """Materialize the generic lifecycle record from a definition.

    Used for definition-owned engines and for the compatibility prefix a
    converted engine keeps (``definition_compat_router``), so both surfaces
    serve one record built from the YAML."""
    lifecycle = dict(definition.lifecycle)
    from backend.engines.python.generic import GenericVenvInstaller

    installer_holder: Dict[str, EngineInstaller] = {}

    def _installer() -> EngineInstaller:
        if not installer_holder:
            installer_holder["installer"] = GenericVenvInstaller(definition)
        return installer_holder["installer"]

    update_source = lifecycle.get("update_source")
    fallback = lifecycle.get("status_fallback")
    return EngineLifecycle(
        engine_id=definition.id,
        url_prefix=str(lifecycle.get("url_prefix") or f"/{definition.id}"),
        get_installer=_installer,
        update_source=(
            update_source_from_lifecycle(dict(update_source)) if update_source else None
        ),
        prefer_source_install=bool(lifecycle.get("prefer_source_install")),
        status_fallback=dict(fallback) if isinstance(fallback, dict) else None,
        version_setting_key=str(lifecycle.get("version_setting_key") or "pip_version"),
    )


def generic_verb_names(lc: EngineLifecycle) -> list[str]:
    """Which generic-surface verbs this lifecycle actually serves — mirrors
    the dispatch conditions in the lifecycle_* helpers (verb override or
    installer backing). YAML campaign S5: surfaced on /api/engines for
    AUTHORITATIVE engines so the frontend can poll/drive them without a
    local registry entry (generated engines keep the legacy descriptor)."""
    verbs: list[str] = []
    if lc.update_source is not None or "check_updates" in lc.verbs:
        verbs.append("check-updates")
    if lc.get_installer is not None or "status" in lc.verbs:
        verbs.append("status")
    if lc.get_installer is not None or "get_build_settings" in lc.verbs:
        verbs.append("build-settings")
    if lc.get_installer is not None or "install" in lc.verbs:
        verbs.append("install")
    if lc.get_installer is not None or "install_source" in lc.verbs:
        verbs.append("install-source")
    if lc.get_installer is not None or "remove" in lc.verbs:
        verbs.append("remove")
    if lc.get_installer is not None or "cancel" in lc.verbs:
        verbs.append("cancel")
    return verbs


def definition_install_defaults(definition) -> dict:
    """Install-settings defaults for an authoritative definition.

    The definition's ``settings_defaults`` is the DECLARATION of record; the
    ``lifecycle`` fields are only a fallback for keys it does not pin. (Before
    this merge the lifecycle values overwrote a declared branch or version,
    which could install the wrong branch/version.)

    A definition that declares ANY settings defaults keeps exactly those keys
    (plus the source repo/branch fallbacks): a version key it does not declare
    is not silently added to its settings surface. A lifecycle-only definition
    gets the fully derived map so it can onboard with no settings block.
    """
    lifecycle = dict(definition.lifecycle)
    version_key = str(lifecycle.get("version_setting_key") or "pip_version")
    declared = {str(key): value for key, value in dict(definition.settings_defaults).items()}
    derived = {
        "source_repo": str(lifecycle.get("default_repo_url") or ""),
        "source_branch": str(lifecycle.get("default_branch") or "main"),
    }
    if not declared:
        derived[version_key] = ""
    derived.update(declared)
    return derived


def register_definition_lifecycles() -> list[str]:
    """Finish definition-driven startup wiring for every non-generated engine.

    Two independent responsibilities:

    1. Register install-settings defaults for every definition that declares a
       ``lifecycle`` block (so the settings verbs work even when a
       compatibility router already owns the engine's URLs).
    2. Create the generic lifecycle record when the engine has no Python
       registration yet. A converted engine whose legacy prefix module mounted
       a definition-derived record at import keeps that single record (the
       warning-free skip), so there is no duplicate registration.

    Called once at startup after definition verification; returns the ids for
    which a NEW lifecycle record was created.
    """
    from backend.engines.definitions.loader import load_default_engine_definitions
    from backend.venv_install_settings import register_engine_install_settings

    registered = []
    for engine_id, definition in sorted(load_default_engine_definitions().definitions.items()):
        if not definition.lifecycle:
            continue
        register_engine_install_settings(engine_id, definition_install_defaults(definition))
        # A definition-derived compatibility router may already have registered
        # the SAME engine's record at import; registering again would replace
        # it needlessly. (An unrelated Python lifecycle for a converted engine
        # would be caught by the descriptor-equality startup gate instead.)
        if engine_id in LIFECYCLE:
            continue
        LIFECYCLE[engine_id] = build_definition_lifecycle(definition)
        registered.append(engine_id)
    return registered
