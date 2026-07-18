from fastapi import APIRouter, HTTPException, Body
from typing import List, Optional
from dataclasses import asdict, fields
import asyncio
import os
import requests
import time
import re
from datetime import datetime

from backend.data_store import get_store
from backend.engine_registry import VALID_ENGINE_IDS
from backend.llama_manager import LlamaManager, BuildConfig
from backend.progress_manager import get_progress_manager
from backend.logging_config import get_logger
from backend.build_cancel_registry import BuildCancelledError, request_build_cancel
from backend.build_task_manager import BuildTaskManager
from backend.gpu_detector import detect_build_capabilities
from backend.cuda_installer import get_cuda_installer
from backend.llama_swap_manager import mark_swap_config_stale
from backend.llama_github_refs import (
    fetch_ik_llama_main_tip_commit,
    fetch_latest_release_for_repository_source,
)
from backend.utils.fs_ops import robust_rmtree

router = APIRouter()
llama_manager = LlamaManager()
logger = get_logger(__name__)

_BUILD_CONFIG_FIELD_NAMES = {f.name for f in fields(BuildConfig)}
_COMMIT_REF_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")
_LIKELY_RELEASE_TAG_RE = re.compile(
    r"^(?:v?\d+(?:\.\d+){1,}(?:[-+][0-9A-Za-z._-]+)?|b\d+)$"
)


def _utcnow() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _looks_like_commit_ref(value: str) -> bool:
    return bool(_COMMIT_REF_RE.fullmatch(str(value or "").strip()))


def _looks_like_release_tag(value: str) -> bool:
    return bool(_LIKELY_RELEASE_TAG_RE.fullmatch(str(value or "").strip()))


def _infer_source_ref_type(source_ref: str, explicit: Optional[str] = None) -> str:
    explicit = str(explicit or "").strip().lower()
    if explicit in ("branch", "commit", "release"):
        return explicit
    ref = str(source_ref or "").strip()
    if _looks_like_commit_ref(ref):
        return "commit"
    if _looks_like_release_tag(ref):
        return "release"
    return "branch"


def _branch_for_source_entry(version_entry: dict) -> Optional[str]:
    """Return a syncable branch name for source-version metadata, if any."""
    if not isinstance(version_entry, dict):
        return None
    install_type = version_entry.get("install_type") or version_entry.get("type")
    if install_type != "source":
        return None
    branch = str(version_entry.get("source_branch") or "").strip()
    if branch:
        return branch
    ref = str(version_entry.get("source_ref") or "").strip()
    ref_type = str(version_entry.get("source_ref_type") or "").strip().lower()
    if ref_type == "branch" and ref:
        return ref
    if ref_type in ("commit", "release"):
        return None
    if ref and not _looks_like_commit_ref(ref) and not _looks_like_release_tag(ref):
        return ref
    return None


@router.post("/scan-engine-params")
async def scan_engine_params_route(payload: dict = Body(default_factory=dict)):
    """Re-run --help parsing for the active (or specified) engine version into ``engine_params_catalog.yaml``."""
    from backend.engine_param_scanner import (
        resolve_version_row,
        scan_audio_cpp_model_profile,
        scan_engine_version,
    )

    engine = (payload or {}).get("engine")
    version = (payload or {}).get("version")
    if engine not in VALID_ENGINE_IDS:
        raise HTTPException(
            status_code=400,
            detail=f"engine must be one of: {', '.join(sorted(VALID_ENGINE_IDS))}",
        )
    store = get_store()
    row = resolve_version_row(store, engine, version)
    if not row:
        raise HTTPException(
            status_code=404,
            detail="No matching engine version (set active or pass version).",
        )
    entry = await asyncio.to_thread(scan_engine_version, store, engine, row)
    profile = None
    model_id = (payload or {}).get("model_id")
    if engine == "audio_cpp" and model_id:
        model = store.get_model(str(model_id))
        if not model:
            raise HTTPException(status_code=404, detail="Model context not found")
        profile = await asyncio.to_thread(
            scan_audio_cpp_model_profile, store, row, model, force=True
        )
    n_params = sum(len(s.get("params") or []) for s in entry.get("sections") or [])
    return {
        "ok": not entry.get("scan_error"),
        "engine": engine,
        "version": row.get("version"),
        "scan_error": entry.get("scan_error"),
        "scanned_at": entry.get("scanned_at"),
        "param_count": n_params,
        "profile_fingerprint": (profile or {}).get("fingerprint"),
        "profile_scan_error": (profile or {}).get("scan_error"),
    }


@router.get("")
@router.get("/")
async def list_llama_versions():
    """List all installed llama.cpp, ik_llama, and LMDeploy versions."""
    store = get_store()
    result = []
    for engine, repo_label in [
        ("llama_cpp", "llama.cpp"),
        ("ik_llama", "ik_llama.cpp"),
    ]:
        active = store.get_active_engine_version(engine)
        active_version = active.get("version") if active else None
        for i, v in enumerate(store.get_engine_versions(engine)):
            version_str = v.get("version")
            result.append(
                {
                    "id": f"{engine}:{version_str}",
                    "version": version_str,
                    "type": v.get("type", "source"),
                    "install_type": v.get("type", "source"),
                    "binary_path": v.get("binary_path"),
                    "source_commit": v.get("source_commit"),
                    "source_ref": v.get("source_ref"),
                    "source_ref_type": v.get("source_ref_type"),
                    "source_repo": v.get("source_repo"),
                    "source_branch": v.get("source_branch")
                    or _branch_for_source_entry(v),
                    "patches": [],  # No longer storing patches in YAML
                    "installed_at": v.get("installed_at"),
                    "updated_at": v.get("updated_at"),
                    "is_active": v.get("version") == active_version,
                    "build_config": v.get("build_config"),
                    "repository_source": v.get("repository_source") or repo_label,
                }
            )
    for engine, repo_label, default_inst in [
        ("lmdeploy", "LMDeploy", "pip"),
        ("1cat_vllm", "1Cat-vLLM", "release"),
    ]:
        active_venv = store.get_active_engine_version(engine)
        active_venv_version = active_venv.get("version") if active_venv else None
        for v in store.get_engine_versions(engine):
            version_str = v.get("version")
            inst_type = v.get("install_type") or v.get("type") or default_inst
            result.append(
                {
                    "id": f"{engine}:{version_str}",
                    "version": version_str,
                    "type": inst_type,
                    "install_type": inst_type,
                    "venv_path": v.get("venv_path"),
                    "source_repo": v.get("source_repo"),
                    "source_branch": v.get("source_branch"),
                    "source_commit": v.get("source_commit"),
                    "release_tag": v.get("release_tag"),
                    "installed_at": v.get("installed_at"),
                    "updated_at": v.get("updated_at"),
                    "is_active": v.get("version") == active_venv_version,
                    "repository_source": repo_label,
                }
            )
    active_audio = store.get_active_engine_version("audio_cpp")
    active_audio_version = active_audio.get("version") if active_audio else None
    for v in store.get_engine_versions("audio_cpp"):
        version_str = v.get("version")
        result.append(
            {
                **v,
                "id": f"audio_cpp:{version_str}",
                "version": version_str,
                "type": v.get("type", "source"),
                "install_type": v.get("install_type", "source"),
                "is_active": v.get("version") == active_audio_version,
                "repository_source": "audio.cpp",
            }
        )
    return result


def _default_build_settings() -> dict:
    """Default build-settings payload for engines when nothing is saved yet.
    Covers all BuildConfig fields so backend and frontend stay in sync.
    """
    return {
        "build_type": "Release",
        "cuda": False,
        "openblas": False,
        "flash_attention": False,
        "build_common": True,
        "build_tests": True,
        "build_tools": True,
        "build_examples": True,
        "build_server": True,
        "install_tools": True,
        "backend_dl": False,
        "cpu_all_variants": False,
        "lto": False,
        "native": True,
        "custom_cmake_args": "",
        "cuda_architectures": "",
        "cflags": "",
        "cxxflags": "",
    }


def _coerce_build_settings(settings: Optional[dict]) -> dict:
    base = _default_build_settings()
    if not isinstance(settings, dict):
        return base

    def _bool(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("1", "true", "yes", "on")
        return bool(v)

    def _str(v, default=""):
        return str(v).strip() if v is not None else default

    build_type = _str(settings.get("build_type"), base["build_type"])
    if build_type not in ("Debug", "Release", "RelWithDebInfo", "MinSizeRel"):
        build_type = base["build_type"]

    return {
        "build_type": build_type,
        "cuda": _bool(settings.get("cuda", base["cuda"])),
        "openblas": _bool(settings.get("openblas", base["openblas"])),
        "flash_attention": _bool(
            settings.get("flash_attention", base["flash_attention"])
        ),
        "build_common": _bool(settings.get("build_common", base["build_common"])),
        "build_tests": _bool(settings.get("build_tests", base["build_tests"])),
        "build_tools": _bool(settings.get("build_tools", base["build_tools"])),
        "build_examples": _bool(settings.get("build_examples", base["build_examples"])),
        "build_server": _bool(settings.get("build_server", base["build_server"])),
        "install_tools": _bool(settings.get("install_tools", base["install_tools"])),
        "backend_dl": _bool(settings.get("backend_dl", base["backend_dl"])),
        "cpu_all_variants": _bool(
            settings.get("cpu_all_variants", base["cpu_all_variants"])
        ),
        "lto": _bool(settings.get("lto", base["lto"])),
        "native": _bool(settings.get("native", base["native"])),
        "custom_cmake_args": _str(
            settings.get("custom_cmake_args"), base["custom_cmake_args"]
        ),
        "cuda_architectures": _str(
            settings.get("cuda_architectures"), base["cuda_architectures"]
        ),
        "cflags": _str(settings.get("cflags"), base["cflags"]),
        "cxxflags": _str(settings.get("cxxflags"), base["cxxflags"]),
    }


def _build_config_from_settings(settings: Optional[dict]) -> BuildConfig:
    normalized = _coerce_build_settings(settings)
    return BuildConfig(
        build_type=normalized["build_type"],
        enable_cuda=normalized["cuda"],
        enable_openblas=normalized["openblas"],
        enable_flash_attention=normalized["flash_attention"],
        build_common=normalized["build_common"],
        build_tests=normalized["build_tests"],
        build_tools=normalized["build_tools"],
        build_examples=normalized["build_examples"],
        build_server=normalized["build_server"],
        install_tools=normalized["install_tools"],
        enable_backend_dl=normalized["backend_dl"],
        enable_cpu_all_variants=normalized["cpu_all_variants"],
        enable_lto=normalized["lto"],
        enable_native=normalized["native"],
        custom_cmake_args=normalized["custom_cmake_args"],
        cuda_architectures=normalized["cuda_architectures"],
        cflags=normalized["cflags"],
        cxxflags=normalized["cxxflags"],
    )


def _build_config_from_any(config: Optional[dict]) -> BuildConfig:
    """Accept either frontend build settings or stored BuildConfig-shaped metadata."""
    if not isinstance(config, dict):
        return BuildConfig()
    if any(k in config for k in ("enable_cuda", "enable_openblas", "enable_native")):
        filtered = {k: v for k, v in config.items() if k in _BUILD_CONFIG_FIELD_NAMES}
        try:
            return BuildConfig(**filtered)
        except (TypeError, ValueError) as exc:
            logger.warning("Stored BuildConfig is invalid (%s), using defaults", exc)
            return BuildConfig()
    return _build_config_from_settings(config)


def _build_config_for_source_sync(
    engine: str, version_entry: dict, store
) -> BuildConfig:
    raw = (version_entry or {}).get("build_config")
    if isinstance(raw, dict):
        config = _build_config_from_any(raw)
    else:
        config = _build_config_from_settings(
            store.get_engine_build_settings(engine) or {}
        )
    if engine == "ik_llama":
        config.build_examples = True
    return config


def _source_ref_slug(source_ref: str) -> str:
    value = str(source_ref or "").strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "-", value)
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    return value[:32] or "source"


def _resolve_engine_build_target(engine: str) -> tuple[str, str]:
    if engine == "ik_llama":
        repository_source = "ik_llama.cpp"
    elif engine == "llama_cpp":
        repository_source = "llama.cpp"
    else:
        raise HTTPException(
            status_code=400, detail="engine must be 'llama_cpp' or 'ik_llama'"
        )

    repository_url = llama_manager.REPOSITORY_SOURCES.get(repository_source)
    if not repository_url:
        raise HTTPException(
            status_code=400, detail=f"Unknown repository source: {repository_source}"
        )
    return repository_source, repository_url


def _apply_engine_specific_build_defaults(engine: str, settings: dict) -> dict:
    """Apply engine-specific build defaults. ik_llama.cpp requires LLAMA_BUILD_EXAMPLES=ON (server in examples)."""
    out = dict(settings)
    if engine == "ik_llama":
        out["build_examples"] = True
    return out


@router.get("/build-settings")
async def get_build_settings(engine: str = "llama_cpp"):
    """Get persisted build settings for an engine ('llama_cpp' or 'ik_llama')."""
    if engine not in ("llama_cpp", "ik_llama"):
        raise HTTPException(
            status_code=400, detail="engine must be 'llama_cpp' or 'ik_llama'"
        )
    store = get_store()
    settings = store.get_engine_build_settings(engine) or {}
    # Always return a full shape so the frontend can rely on defaults.
    base = _default_build_settings()
    base.update({k: v for k, v in settings.items() if k in base})
    return _apply_engine_specific_build_defaults(engine, base)


@router.put("/build-settings")
async def update_build_settings(engine: str = "llama_cpp", settings: dict = Body(...)):
    """Persist build settings for an engine ('llama_cpp' or 'ik_llama')."""
    if engine not in ("llama_cpp", "ik_llama"):
        raise HTTPException(
            status_code=400, detail="engine must be 'llama_cpp' or 'ik_llama'"
        )
    if not isinstance(settings, dict):
        raise HTTPException(status_code=400, detail="settings must be an object")
    store = get_store()
    # Only persist known build keys; ignore extras.
    allowed = _default_build_settings().keys()
    filtered = {k: v for k, v in settings.items() if k in allowed}
    filtered = _apply_engine_specific_build_defaults(engine, filtered)
    stored = store.update_engine_build_settings(engine, filtered)
    base = _default_build_settings()
    base.update({k: v for k, v in stored.items() if k in base})
    return _apply_engine_specific_build_defaults(engine, base)


@router.post("/update")
async def update_engine(request: dict):
    """Build latest upstream: llama.cpp = newest GitHub release tag; ik_llama.cpp = ``main`` tip commit."""
    engine = (request or {}).get("engine", "llama_cpp")
    version_suffix = (request or {}).get("version_suffix")
    repository_source, repository_url = _resolve_engine_build_target(engine)
    store = get_store()
    settings = store.get_engine_build_settings(engine) or {}
    build_config = _build_config_from_settings(settings)

    try:
        if engine == "ik_llama":
            tip = fetch_ik_llama_main_tip_commit()
            if not tip or not tip.get("sha"):
                raise HTTPException(
                    status_code=404,
                    detail="Could not resolve latest commit on ik_llama.cpp main",
                )
            source_ref = tip["sha"]
            ref_type = "ref"
        else:
            latest_release = fetch_latest_release_for_repository_source(
                repository_source
            )
            if not latest_release or not latest_release.get("tag_name"):
                raise HTTPException(
                    status_code=404, detail="No release found for this engine"
                )
            source_ref = latest_release["tag_name"]
            ref_type = "release"
    except HTTPException:
        raise
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise HTTPException(
                status_code=429,
                detail="GitHub API rate limit exceeded. Please try again later.",
            )
        if e.response.status_code == 404:
            raise HTTPException(
                status_code=404, detail="GitHub repository or release not found"
            )
        raise HTTPException(status_code=500, detail=f"GitHub API error: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")

    return _schedule_source_build(
        source_ref=source_ref,
        patches=[],
        build_config=build_config,
        repository_source=repository_source,
        repository_url=repository_url,
        version_suffix=version_suffix,
        auto_activate=True,
        source_ref_type=ref_type,
    )


@router.get("/check-updates")
async def check_updates(source: str | None = None):
    """Check upstream versions: llama.cpp = releases + default-branch tip; ik_llama.cpp = ``main`` tip only."""
    try:
        is_ik = source == "ik_llama"
        if is_ik:
            tip = fetch_ik_llama_main_tip_commit()
            return {
                "latest_release": None,
                "latest_commit": (
                    {
                        "sha": tip["sha"],
                        "commit_date": tip.get("commit_date"),
                        "message": tip.get("message"),
                    }
                    if tip
                    else None
                ),
            }

        repository_source = "llama.cpp"
        commits_url = (
            "https://api.github.com/repos/ggerganov/llama.cpp/commits?per_page=1"
        )

        latest_release = fetch_latest_release_for_repository_source(repository_source)

        commits_response = requests.get(commits_url, allow_redirects=True)
        commits_response.raise_for_status()
        raw_commits = commits_response.json()
        commits = raw_commits if isinstance(raw_commits, list) else []
        tip = commits[0] if commits else None

        return {
            "latest_release": (
                {
                    "tag_name": latest_release["tag_name"],
                    "published_at": latest_release.get("published_at"),
                    "html_url": latest_release.get("html_url"),
                }
                if latest_release
                else None
            ),
            "latest_commit": (
                {
                    "sha": tip["sha"],
                    "commit_date": tip["commit"]["committer"]["date"],
                    "message": tip["commit"]["message"],
                }
                if tip
                else None
            ),
        }
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 403:
            raise HTTPException(
                status_code=429,
                detail="GitHub API rate limit exceeded. Please try again later.",
            )
        elif e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="GitHub repository not found")
        else:
            raise HTTPException(status_code=500, detail=f"GitHub API error: {str(e)}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Update check failed: {str(e)}")


@router.get("/releases/{tag_name}/assets")
async def get_release_assets(tag_name: str):
    raise HTTPException(
        status_code=410,
        detail="Prebuilt llama.cpp release installation has been removed. Build from source instead.",
    )


@router.get("/build-capabilities")
async def get_build_capabilities_endpoint():
    """Get build capabilities (CUDA, OpenBLAS)."""
    try:
        return await detect_build_capabilities()
    except Exception as e:
        logger.error(f"Error detecting build capabilities: {e}")
        # Return safe defaults
        return {
            "cuda": {
                "available": False,
                "recommended": False,
                "reason": f"Error: {str(e)}",
            },
            "openblas": {
                "available": False,
                "recommended": False,
                "reason": f"Error: {str(e)}",
            },
        }


@router.post("/install-release")
async def install_release(request: dict):
    raise HTTPException(
        status_code=410,
        detail="Prebuilt llama.cpp release installation has been removed. Build from source instead.",
    )


async def install_release_task(
    tag_name: str,
    progress_manager=None,
    task_id: str = None,
    asset_id: Optional[int] = None,
):
    """Background task to install release with SSE progress updates"""
    store = get_store()
    try:
        install_result = await llama_manager.install_release(
            tag_name, progress_manager, task_id, asset_id
        )
        binary_path = install_result.get("binary_path")
        asset_info = install_result.get("asset")
        version_name = install_result.get("version_name") or tag_name

        if not binary_path:
            raise Exception("Installation completed without returning a binary path.")

        version_data = {
            "version": version_name,
            "type": "release",
            "binary_path": binary_path,
            "installed_at": datetime.utcnow().isoformat() + "Z",
            "build_config": (
                {"release_asset": asset_info, "tag_name": tag_name}
                if asset_info
                else None
            ),
            "repository_source": "llama.cpp",
        }
        store.add_engine_version("llama_cpp", version_data)

        mark_swap_config_stale()

        if progress_manager:
            asset_label = ""
            if asset_info and asset_info.get("name"):
                asset_label = f" ({asset_info['name']})"
            progress_manager.complete_task(task_id, f"Installed {version_name}")
            await progress_manager.send_notification(
                title="Installation Complete",
                message=f"Successfully installed llama.cpp release {version_name}{asset_label}",
                type="success",
            )

    except Exception as e:
        logger.error(f"Release installation failed: {e}")
        if progress_manager and task_id:
            progress_manager.fail_task(task_id, str(e))
        if progress_manager:
            await progress_manager.send_notification(
                title="Installation Failed",
                message=f"Failed to install llama.cpp release: {str(e)}",
                type="error",
            )


@router.post("/build-source")
async def build_source(request: dict):
    """Build llama.cpp from source with optional patches"""
    try:
        commit_sha = request.get("commit_sha")
        patches = request.get("patches", [])
        build_config_dict = request.get("build_config")
        repository_source = request.get("repository_source", "llama.cpp")
        version_suffix = request.get("version_suffix")
        auto_activate = bool(request.get("auto_activate"))

        if not commit_sha:
            raise HTTPException(status_code=400, detail="commit_sha is required")
        source_ref_type = _infer_source_ref_type(
            commit_sha, request.get("source_ref_type")
        )

        if repository_source == "ik_llama.cpp":
            _, repository_url = _resolve_engine_build_target("ik_llama")
        elif repository_source == "llama.cpp":
            _, repository_url = _resolve_engine_build_target("llama_cpp")
            # Optional fork / mirror (same as LMDeploy-style "install from source").
            override = request.get("repository_url") or request.get("repo_url")
            if override and str(override).strip():
                u = str(override).strip()
                if not (
                    u.startswith("https://")
                    or u.startswith("http://")
                    or u.startswith("git@")
                    or u.startswith("ssh://")
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="repository_url must be a valid git clone URL (https, http, git@, or ssh)",
                    )
                repository_url = u
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown repository source: {repository_source}",
            )

        build_config = _build_config_from_any(build_config_dict)

        return _schedule_source_build(
            source_ref=commit_sha,
            patches=patches,
            build_config=build_config,
            repository_source=repository_source,
            repository_url=repository_url,
            version_suffix=version_suffix,
            auto_activate=auto_activate,
            source_ref_type=source_ref_type,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/build-cancel")
async def build_cancel(payload: dict = Body(...)):
    """Request cancellation of an in-flight llama.cpp / ik_llama source build."""
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return BuildTaskManager.cancel(str(task_id))


@router.post("/cuda/cancel")
async def cuda_cancel(payload: dict = Body(...)):
    """Request cancellation of an in-flight CUDA install/uninstall."""
    task_id = (payload or {}).get("task_id")
    if not task_id:
        raise HTTPException(status_code=400, detail="task_id is required")
    return get_cuda_installer().cancel_task(str(task_id))


async def _llama_checkout_commit(version_name: str) -> Optional[str]:
    clone_dir = os.path.join(llama_manager.llama_dir, version_name, "llama.cpp")
    if not os.path.isdir(clone_dir):
        return None
    try:
        proc = await llama_manager._run_command(
            "git", "rev-parse", "HEAD", cwd=clone_dir, timeout=15
        )
        if proc.returncode != 0:
            return None
        return (proc.stdout or b"").decode("utf-8", errors="replace").strip() or None
    except Exception as exc:
        logger.debug("Could not read source commit for %s: %s", version_name, exc)
        return None


@router.post("/versions/sync")
async def sync_version_body(payload: dict = Body(...)):
    """Sync a branch-based source install in-place and rebuild incrementally."""
    version_id = (payload or {}).get("version_id")
    if not version_id:
        raise HTTPException(status_code=400, detail="version_id required")

    store = get_store()
    version_entry, engine = _find_version_entry(store, str(version_id))
    if not version_entry or not engine:
        raise HTTPException(status_code=404, detail="Version not found")

    branch = _branch_for_source_entry(version_entry)
    if not branch:
        raise HTTPException(
            status_code=400,
            detail="Only source installs that track a branch can be synced.",
        )

    if engine in ("llama_cpp", "ik_llama"):
        repository_source = (
            version_entry.get("repository_source")
            or ("ik_llama.cpp" if engine == "ik_llama" else "llama.cpp")
        )
        if repository_source not in ("llama.cpp", "ik_llama.cpp"):
            raise HTTPException(status_code=400, detail="Unsupported source engine")
        repository_url = version_entry.get("source_repo") or llama_manager.REPOSITORY_SOURCES[
            repository_source
        ]
        build_config = _build_config_for_source_sync(engine, version_entry, store)
        return _schedule_source_sync(
            version_entry=version_entry,
            engine=engine,
            branch=branch,
            build_config=build_config,
            repository_source=repository_source,
            repository_url=repository_url,
        )

    if engine == "lmdeploy":
        from backend.lmdeploy_manager import get_lmdeploy_manager

        try:
            return await get_lmdeploy_manager().sync_source_version(version_entry)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    if engine == "1cat_vllm":
        from backend.onecat_vllm_manager import get_onecat_vllm_manager

        try:
            return await get_onecat_vllm_manager().sync_source_version(version_entry)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    if engine == "audio_cpp":
        from backend.audio_cpp_manager import get_audio_cpp_manager
        from backend.routes.audio_cpp_versions import schedule_audio_cpp_sync

        manager = get_audio_cpp_manager()
        raw_config = (version_entry or {}).get("build_config")
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
        return schedule_audio_cpp_sync(version_entry, branch, build_config)

    raise HTTPException(status_code=400, detail="Unsupported engine")


async def build_source_task(
    commit_sha: str,
    patches: List[str],
    build_config: BuildConfig,
    version_name: str,
    repository_source: str,
    repository_url: str,
    progress_manager=None,
    task_id: str = None,
    auto_activate: bool = False,
    source_ref_type: str = "ref",
):
    """Background task to build from source with SSE progress"""
    logger.info(
        "Build task started: version_name=%s, repository_source=%s, commit_sha=%s",
        version_name,
        repository_source,
        commit_sha[:8] if commit_sha else "",
    )
    try:
        store = get_store()
        engine = "ik_llama" if repository_source == "ik_llama.cpp" else "llama_cpp"

        binary_path = await llama_manager.build_source(
            commit_sha,
            patches,
            build_config,
            progress_manager,
            task_id,
            repository_url=repository_url,
            version_name=version_name,
        )

        build_config_dict = None
        if build_config:
            build_config_dict = asdict(build_config)
            build_config_dict["repository_source"] = repository_source

        actual_commit = await _llama_checkout_commit(version_name)
        source_branch = commit_sha if source_ref_type == "branch" else None
        version_data = {
            "version": version_name,
            "type": "patched" if patches else "source",
            "binary_path": binary_path,
            "source_commit": actual_commit or commit_sha,
            "source_ref": commit_sha,
            "source_ref_type": source_ref_type,
            "source_branch": source_branch,
            "source_repo": repository_url,
            "build_config": build_config_dict,
            "repository_source": repository_source,
            "installed_at": _utcnow(),
        }
        store.add_engine_version(engine, version_data)

        mark_swap_config_stale()

        if auto_activate:
            try:
                # Reuse the existing activation flow (includes llama-swap handling).
                await _do_activate_version(f"{engine}:{version_name}")
            except HTTPException as e:
                logger.error(
                    "Auto-activation failed for %s:%s: %s",
                    engine,
                    version_name,
                    e.detail,
                )
            except Exception as e:
                logger.exception(
                    "Auto-activation failed for %s:%s: %s", engine, version_name, e
                )

        if progress_manager:
            if task_id:
                progress_manager.complete_task(task_id, f"Built {version_name}")
            await progress_manager.send_notification(
                title="Build Complete",
                message=f"Successfully built {repository_source} from source {commit_sha[:8]}",
                type="success",
            )

    except BuildCancelledError:
        logger.info("Source build cancelled: task_id=%s", task_id)
        if progress_manager and task_id:
            progress_manager.fail_task(task_id, "Build cancelled by user")
            try:
                await progress_manager.send_notification(
                    title="Build cancelled",
                    message="The source build was stopped before completion.",
                    type="warn",
                )
            except Exception as notify_err:
                logger.debug("build cancel notification: %s", notify_err)

    except Exception as e:
        logger.exception("Source build failed: %s", e)
        if progress_manager:
            try:
                if task_id:
                    progress_manager.fail_task(task_id, str(e))
                await progress_manager.send_notification(
                    title="Build Failed",
                    message=f"Failed to build llama.cpp from source: {str(e)}",
                    type="error",
                )
            except Exception as ws_error:
                logger.error(f"Failed to send build failure notification: {ws_error}")


async def sync_source_build_task(
    branch: str,
    build_config: BuildConfig,
    version_name: str,
    engine: str,
    repository_source: str,
    repository_url: str,
    progress_manager=None,
    task_id: str = None,
):
    """Background task: sync an existing source checkout and rebuild it in place."""
    logger.info(
        "Source sync task started: engine=%s version=%s branch=%s",
        engine,
        version_name,
        branch,
    )
    try:
        store = get_store()
        binary_path = await llama_manager.build_source(
            branch,
            [],
            build_config,
            progress_manager,
            task_id,
            repository_url=repository_url,
            version_name=version_name,
            reuse_existing_checkout=True,
            source_branch=branch,
        )
        actual_commit = await _llama_checkout_commit(version_name)
        build_config_dict = asdict(build_config) if build_config else None
        if build_config_dict is not None:
            build_config_dict["repository_source"] = repository_source

        updated = store.update_engine_version(
            engine,
            version_name,
            {
                "binary_path": binary_path,
                "source_commit": actual_commit or branch,
                "source_ref": branch,
                "source_ref_type": "branch",
                "source_branch": branch,
                "source_repo": repository_url,
                "build_config": build_config_dict,
                "repository_source": repository_source,
                "updated_at": _utcnow(),
            },
        )
        if not updated:
            raise RuntimeError(f"Version '{version_name}' disappeared during sync")

        try:
            from backend.engine_param_scanner import scan_engine_version

            scan_engine_version(store, engine, updated)
        except Exception as scan_e:
            logger.warning("Param scan after source sync failed: %s", scan_e)

        mark_swap_config_stale()

        if progress_manager:
            if task_id:
                progress_manager.complete_task(task_id, f"Synced {version_name}")
            await progress_manager.send_notification(
                title="Sync Complete",
                message=f"Rebuilt {repository_source} from {branch}",
                type="success",
            )

    except BuildCancelledError:
        logger.info("Source sync cancelled: task_id=%s", task_id)
        if progress_manager and task_id:
            progress_manager.fail_task(task_id, "Build cancelled by user")
            try:
                await progress_manager.send_notification(
                    title="Sync cancelled",
                    message="The source sync was stopped before completion.",
                    type="warn",
                )
            except Exception as notify_err:
                logger.debug("sync cancel notification: %s", notify_err)

    except Exception as e:
        logger.exception("Source sync failed: %s", e)
        if progress_manager:
            try:
                if task_id:
                    progress_manager.fail_task(task_id, str(e))
                await progress_manager.send_notification(
                    title="Sync Failed",
                    message=f"Failed to sync {repository_source}: {str(e)}",
                    type="error",
                )
            except Exception as ws_error:
                logger.error(f"Failed to send sync failure notification: {ws_error}")


@router.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    return {
        "task_id": task_id,
        "status": "running",
        "message": "Task is running. Subscribe to GET /api/events for real-time SSE progress updates.",
    }


@router.get("/verify/{version}")
async def verify_version(version: str):
    """Verify that all required llama.cpp commands are available for a version"""
    try:
        verification = llama_manager.verify_installation(version)
        commands = llama_manager.get_all_commands(version)

        return {
            "version": version,
            "verification": verification,
            "commands": commands,
            "all_available": all(verification.values()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/commands/{version}")
async def get_version_commands(version: str):
    """Get all available commands for a specific version"""
    try:
        commands = llama_manager.get_all_commands(version)
        return {"version": version, "commands": commands}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _abs_venv_path(venv: str) -> str:
    if not venv:
        return ""
    if not os.path.isabs(venv):
        if os.path.exists("/app/data"):
            venv = os.path.normpath(os.path.join("/app", venv))
        else:
            venv = os.path.normpath(os.path.join(os.getcwd(), venv))
    return venv


def _lmdeploy_binary_for_entry(version_entry: dict) -> str:
    """Absolute path to lmdeploy executable for a version entry (venv_path required)."""
    venv = _abs_venv_path((version_entry or {}).get("venv_path") or "")
    if not venv:
        return ""
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "lmdeploy.exe" if os.name == "nt" else "lmdeploy"
    return os.path.join(venv, sub, exe)


def _onecat_vllm_binary_for_entry(version_entry: dict) -> str:
    """1Cat-vLLM is served via ``python -m vllm``; validate the venv python exists."""
    venv = _abs_venv_path((version_entry or {}).get("venv_path") or "")
    if not venv:
        return ""
    sub = "Scripts" if os.name == "nt" else "bin"
    exe = "python.exe" if os.name == "nt" else "python"
    return os.path.join(venv, sub, exe)


def _resolve_binary_path(binary_path: str) -> str:
    if not binary_path:
        return ""
    if os.path.isabs(binary_path):
        return binary_path
    # Docker: paths relative to /app; local: relative to project root
    if os.path.exists("/app/data"):
        return os.path.normpath(os.path.join("/app", binary_path))
    cwd = os.getcwd()
    resolved = os.path.normpath(os.path.join(cwd, binary_path))
    if os.path.exists(resolved):
        return resolved
    # When run with --app-dir backend, cwd may be backend/; project root is parent
    parent = os.path.dirname(cwd)
    return os.path.normpath(os.path.join(parent, binary_path))


def _find_version_entry(store, version_id: str):
    """Resolve version_id ('engine:version' or plain version) to (version_entry, engine). Returns (None, None) if not found."""
    version_entry = None
    engine = None
    if ":" in version_id:
        parts = version_id.split(":", 1)
        eng, version_str = parts[0], parts[1]
        if eng in VALID_ENGINE_IDS:
            version_entry = next(
                (
                    v
                    for v in store.get_engine_versions(eng)
                    if str(v.get("version")) == version_str
                ),
                None,
            )
            if version_entry:
                engine = eng
    if not version_entry:
        for eng in VALID_ENGINE_IDS:
            versions = store.get_engine_versions(eng)
            version_entry = next(
                (v for v in versions if str(v.get("version")) == str(version_id)), None
            )
            if version_entry:
                engine = eng
                break
    return version_entry, engine


def _schedule_source_build(
    source_ref: str,
    patches: List[str],
    build_config: BuildConfig,
    repository_source: str,
    repository_url: str,
    version_suffix: Optional[str] = None,
    auto_activate: bool = False,
    source_ref_type: str = "ref",
):
    store = get_store()
    engine = "ik_llama" if repository_source == "ik_llama.cpp" else "llama_cpp"
    ref_slug = _source_ref_slug(source_ref)
    if version_suffix:
        version_name = f"source-{ref_slug}-{version_suffix}"
    else:
        timestamp = int(time.time())
        version_name = f"source-{ref_slug}-{timestamp}"

    existing_versions = store.get_engine_versions(engine)
    existing = next(
        (v for v in existing_versions if v.get("version") == version_name), None
    )
    if existing:
        raise HTTPException(
            status_code=400, detail=f"Version '{version_name}' already installed"
        )

    task_id = f"build_{version_name}_{int(time.time())}"
    pm = get_progress_manager()
    pm.create_task(
        "build",
        f"Build {repository_source} {ref_slug}",
        {
            "version_name": version_name,
            "engine": engine,
            "repository_source": repository_source,
            "auto_activate": auto_activate,
            "source_ref": source_ref,
            "source_ref_type": source_ref_type,
        },
        task_id=task_id,
    )
    asyncio.create_task(
        build_source_task(
            source_ref,
            patches,
            build_config or BuildConfig(),
            version_name,
            repository_source,
            repository_url,
            pm,
            task_id,
            auto_activate=auto_activate,
            source_ref_type=source_ref_type,
        )
    )
    return {
        "message": f"Building from source {ref_slug}",
        "task_id": task_id,
        "status": "started",
        "progress": 0,
        "version_name": version_name,
        "repository_source": repository_source,
        "source_ref": source_ref,
        "source_ref_type": source_ref_type,
    }


def _schedule_source_sync(
    version_entry: dict,
    engine: str,
    branch: str,
    build_config: BuildConfig,
    repository_source: str,
    repository_url: str,
):
    version_name = str(version_entry.get("version") or "").strip()
    if not version_name:
        raise HTTPException(status_code=400, detail="Version metadata is missing a name")

    branch_slug = _source_ref_slug(branch)
    task_id = f"build_sync_{_source_ref_slug(version_name)}_{int(time.time())}"
    pm = get_progress_manager()
    pm.create_task(
        "build",
        f"Sync {repository_source} {branch_slug}",
        {
            "version_name": version_name,
            "engine": engine,
            "repository_source": repository_source,
            "source_ref": branch,
            "source_ref_type": "branch",
            "sync": True,
        },
        task_id=task_id,
    )
    asyncio.create_task(
        sync_source_build_task(
            branch=branch,
            build_config=build_config or BuildConfig(),
            version_name=version_name,
            engine=engine,
            repository_source=repository_source,
            repository_url=repository_url,
            progress_manager=pm,
            task_id=task_id,
        )
    )
    return {
        "message": f"Syncing {version_name} from {branch}",
        "task_id": task_id,
        "status": "started",
        "progress": 0,
        "version_name": version_name,
        "repository_source": repository_source,
        "source_ref": branch,
        "source_ref_type": "branch",
    }


@router.post("/versions/activate")
async def activate_version_body(payload: dict = Body(...)):
    """Activate a version; body: { \"version_id\": \"llama_cpp:version\" or \"version\" }."""
    version_id = (payload or {}).get("version_id")
    if not version_id:
        raise HTTPException(status_code=400, detail="version_id required")
    return await _do_activate_version(version_id)


async def _do_activate_version(version_id: str):
    store = get_store()
    version_entry, engine = _find_version_entry(store, version_id)
    if not version_entry or not engine:
        logger.warning(
            "activate_version: version not found, version_id=%r, llama_cpp versions=%s",
            version_id,
            [v.get("version") for v in store.get_engine_versions("llama_cpp")],
        )
        raise HTTPException(status_code=404, detail="Version not found")
    version_str = str(version_entry.get("version"))
    # audio.cpp has its own activate path (always scan + capability delta).
    if engine == "audio_cpp":
        from backend.routes.audio_cpp_versions import _activate

        return await _activate(version_str)
    if engine == "lmdeploy":
        bin_path = _lmdeploy_binary_for_entry(version_entry)
        if not bin_path or not os.path.exists(bin_path):
            raise HTTPException(
                status_code=400,
                detail="LMDeploy binary not found for this version",
            )
    elif engine == "1cat_vllm":
        bin_path = _onecat_vllm_binary_for_entry(version_entry)
        if not bin_path or not os.path.exists(bin_path):
            raise HTTPException(
                status_code=400,
                detail="1Cat-vLLM environment not found for this version",
            )
    else:
        binary_path = _resolve_binary_path(version_entry.get("binary_path"))
        if not binary_path or not os.path.exists(binary_path):
            raise HTTPException(status_code=400, detail="Binary file does not exist")
    store.set_active_engine_version(engine, version_str)

    from backend.engine_param_catalog import get_version_entry
    from backend.engine_param_scanner import scan_engine_version

    catalog_entry = get_version_entry(store, engine, version_str)
    if catalog_entry is None:
        try:
            await asyncio.to_thread(
                scan_engine_version, store, engine, version_entry
            )
        except Exception as scan_err:
            logger.warning(
                "CLI param scan after activating %s:%s: %s",
                engine,
                version_str,
                scan_err,
            )

    if engine == "llama_cpp":
        try:
            from backend.llama_swap_manager import get_llama_swap_manager

            llama_swap_manager = get_llama_swap_manager()
            await llama_swap_manager._ensure_correct_binary_path()
            try:
                await llama_swap_manager.start_proxy()
            except Exception as e:
                logger.warning(
                    "Failed to start llama-swap after version activation: %s", e
                )
        except Exception as e:
            logger.error("Failed to start llama-swap after activation: %s", e)
    elif engine in ("lmdeploy", "1cat_vllm"):
        try:
            from backend.llama_swap_manager import get_llama_swap_manager

            llama_swap_manager = get_llama_swap_manager()
            await llama_swap_manager.sync_running_models()
            try:
                await llama_swap_manager.start_proxy()
            except Exception as e:
                logger.warning(
                    "Failed to start llama-swap after %s activation: %s", engine, e
                )
        except Exception as e:
            logger.error("Failed after %s activation: %s", engine, e)
    mark_swap_config_stale()
    logger.info("Activated %s version: %s", engine, version_str)
    return {"message": f"Activated {engine} version {version_str}"}


@router.delete("/{version_id}")
async def delete_version(version_id: str):
    """Delete an engine version (version_id is 'engine:version' or a unique version string)."""
    store = get_store()
    version_entry = None
    if ":" in version_id:
        parts = version_id.split(":", 1)
        engine, version_str = parts[0], parts[1]
        if engine in VALID_ENGINE_IDS:
            version_entry = next(
                (
                    v
                    for v in store.get_engine_versions(engine)
                    if str(v.get("version")) == version_str
                ),
                None,
            )
            if version_entry:
                version_entry["_engine"] = engine
    if not version_entry:
        for engine in VALID_ENGINE_IDS:
            versions = store.get_engine_versions(engine)
            version_entry = next(
                (v for v in versions if str(v.get("version")) == str(version_id)), None
            )
            if version_entry:
                version_entry["_engine"] = engine
                break
    if not version_entry:
        raise HTTPException(status_code=404, detail="Version not found")
    engine = version_entry.get("_engine", "llama_cpp")
    version_str = str(version_entry.get("version"))
    active = store.get_active_engine_version(engine)
    if active and str(active.get("version")) == version_str:
        raise HTTPException(status_code=400, detail="Cannot delete active version")
    if engine in ("lmdeploy", "1cat_vllm"):
        try:
            venv_path = version_entry.get("venv_path") or ""
            if venv_path:
                if not os.path.isabs(venv_path):
                    if os.path.exists("/app/data"):
                        venv_path = os.path.normpath(os.path.join("/app", venv_path))
                    else:
                        venv_path = os.path.normpath(
                            os.path.join(os.getcwd(), venv_path)
                        )
                version_root = os.path.dirname(venv_path)
                if version_root and os.path.isdir(version_root):
                    robust_rmtree(version_root)
            store.delete_engine_version(engine, version_str)
            logger.info("Deleted %s version: %s", engine, version_str)
            mark_swap_config_stale()
            return {"message": f"Deleted version {version_str}"}
        except Exception as e:
            logger.error(f"Failed to delete {engine} version {version_str}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to delete version: {e}"
            )
    if engine == "audio_cpp":
        try:
            from backend.audio_cpp_manager import get_audio_cpp_manager

            get_audio_cpp_manager().delete_version_files(version_entry)
            store.delete_engine_version(engine, version_str)
            mark_swap_config_stale()
            return {"message": f"Deleted version {version_str}"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("Failed to delete audio.cpp version %s: %s", version_str, e)
            raise HTTPException(
                status_code=500, detail=f"Failed to delete version: {e}"
            )
    try:
        binary_path = _resolve_binary_path(version_entry.get("binary_path") or "")
        if binary_path and os.path.exists(binary_path):
            # Safely resolve the on-disk version directory without ever deleting the
            # entire llama-cpp root. Versions are stored as subdirectories of
            # llama_manager.llama_dir (e.g. <llama_dir>/<version>/.../llama-server).
            try:
                llama_root = os.path.realpath(llama_manager.llama_dir)
                binary_real = os.path.realpath(binary_path)
            except Exception:
                llama_root = llama_manager.llama_dir
                binary_real = binary_path

            version_dir = None

            # If the binary lives under the llama root, treat the first path
            # component under that root as the version directory.
            try:
                if os.path.commonpath([binary_real, llama_root]) == llama_root:
                    rel = os.path.relpath(binary_real, llama_root)
                    first_component = rel.split(os.sep)[0]
                    if first_component and first_component not in (".", ""):
                        candidate = os.path.join(llama_root, first_component)
                        if os.path.isdir(candidate):
                            version_dir = candidate
            except Exception:
                # Fall back to parent-directory logic below if commonpath/relpath fail
                version_dir = None

            # Fallback: use the binary's parent directory, but never delete the
            # llama root itself.
            if not version_dir:
                candidate = os.path.dirname(binary_real)
                if (
                    candidate
                    and os.path.isdir(candidate)
                    and os.path.commonpath([candidate, llama_root]) == llama_root
                    and os.path.abspath(candidate) != os.path.abspath(llama_root)
                ):
                    version_dir = candidate

            if version_dir and os.path.exists(version_dir):
                robust_rmtree(version_dir)
            else:
                # As a last resort, remove just the binary to avoid leaving a
                # completely broken entry on disk.
                try:
                    os.remove(binary_real)
                except OSError:
                    pass
        store.delete_engine_version(engine, version_str)
        logger.info(f"Deleted version: {version_str}")
        mark_swap_config_stale()
        return {"message": f"Deleted version {version_str}"}
    except Exception as e:
        logger.error(f"Failed to delete version {version_str}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete version: {e}")


# CUDA Installer endpoints
@router.get("/cuda-status")
async def get_cuda_status():
    """Get CUDA installation status"""
    try:
        installer = get_cuda_installer()
        status = installer.status()
        return status
    except Exception as e:
        logger.error(f"Failed to get CUDA status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cuda-install")
async def install_cuda(request: dict):
    """Install CUDA Toolkit with optional cuDNN and TensorRT"""
    try:
        version = request.get("version", "12.6")
        install_cudnn = request.get("install_cudnn", False)
        install_tensorrt = request.get("install_tensorrt", False)
        installer = get_cuda_installer()

        if installer.is_operation_running():
            raise HTTPException(
                status_code=400,
                detail="A CUDA installation operation is already running",
            )

        result = await installer.install(
            version=version,
            install_cudnn=install_cudnn,
            install_tensorrt=install_tensorrt,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start CUDA installation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cuda-logs")
async def get_cuda_logs():
    """Get CUDA installation logs"""
    try:
        installer = get_cuda_installer()
        logs = installer.read_log_tail()
        return {"logs": logs}
    except Exception as e:
        logger.error(f"Failed to get CUDA logs: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cuda-uninstall")
async def uninstall_cuda(request: dict):
    """Uninstall CUDA Toolkit"""
    try:
        version = request.get(
            "version"
        )  # Optional - if not provided, uninstalls current
        installer = get_cuda_installer()

        if installer.is_operation_running():
            raise HTTPException(
                status_code=400,
                detail="A CUDA installation operation is already running",
            )

        result = await installer.uninstall(version)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start CUDA uninstallation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
