"""Engine version build status, registration, orphan discovery, and retryability.

Failed builds are first-class rows in ``engines.yaml`` so leftover install
directories are visible and can be deleted or retried. Unregistered folders on
disk are surfaced as synthetic ``broken`` rows until the user cleans them up.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from backend.engine_registry import ENGINE_REGISTRY, get_engine_spec
from backend.logging_config import get_logger
from backend.repo_identity import github_owner_repo


logger = get_logger(__name__)

BUILD_STATUS_READY = "ready"
BUILD_STATUS_BUILDING = "building"
BUILD_STATUS_FAILED = "failed"
BUILD_STATUS_CANCELLED = "cancelled"
BUILD_STATUS_BROKEN = "broken"

KNOWN_BUILD_STATUSES = frozenset(
    {
        BUILD_STATUS_READY,
        BUILD_STATUS_BUILDING,
        BUILD_STATUS_FAILED,
        BUILD_STATUS_CANCELLED,
        BUILD_STATUS_BROKEN,
    }
)
RETRYABLE_STATUSES = frozenset(
    {
        BUILD_STATUS_FAILED,
        BUILD_STATUS_CANCELLED,
        BUILD_STATUS_BROKEN,
    }
)

ENGINE_REPO_LABELS = {
    "llama_cpp": "llama.cpp",
    "ik_llama": "ik_llama.cpp",
    "lmdeploy": "LMDeploy",
    "1cat_vllm": "1Cat-vLLM",
    "audio_cpp": "audio.cpp",
}

_NATIVE_SHARED_ROOT_ENGINES = ("llama_cpp", "ik_llama")
_SKIP_DIR_NAMES = frozenset({".git", "__pycache__"})


def upsert_engine_version(store, engine: str, version_data: dict) -> dict:
    """Insert or merge an engine version row, keyed by ``version``."""
    payload = dict(version_data or {})
    version = str(payload.get("version") or "").strip()
    if not version:
        raise ValueError("engine version data requires a version name")
    payload["version"] = version
    updated = store.update_engine_version(engine, version, payload)
    if updated is not None:
        return updated
    store.add_engine_version(engine, payload)
    return payload


def mark_engine_version_building(
    store, engine: str, version_data: dict, *, task_id: Optional[str] = None
) -> dict:
    payload = dict(version_data or {})
    payload["build_status"] = BUILD_STATUS_BUILDING
    payload["build_error"] = None
    if task_id:
        payload["build_task_id"] = task_id
    return upsert_engine_version(store, engine, payload)


def mark_engine_version_ready(store, engine: str, version_data: dict) -> dict:
    payload = dict(version_data or {})
    payload["build_status"] = BUILD_STATUS_READY
    payload["build_error"] = None
    return upsert_engine_version(store, engine, payload)


def mark_engine_version_failed(
    store,
    engine: str,
    version: str,
    *,
    error: str,
    cancelled: bool = False,
    extra: Optional[dict] = None,
) -> dict:
    payload = dict(extra or {})
    payload["version"] = version
    payload["build_status"] = (
        BUILD_STATUS_CANCELLED if cancelled else BUILD_STATUS_FAILED
    )
    payload["build_error"] = str(error or "").strip() or (
        "Build cancelled by user" if cancelled else "Build failed"
    )
    return upsert_engine_version(store, engine, payload)


def engine_row_has_required_paths(engine: str, row: Optional[dict]) -> bool:
    spec = get_engine_spec(engine)
    if not spec or not isinstance(row, dict):
        return False
    return all(bool(str(row.get(field) or "").strip()) for field in spec.active_path_fields)


def normalize_engine_version_status(
    engine: str,
    row: Optional[dict],
    *,
    task_running: Optional[bool] = None,
) -> str:
    """Return the effective build status for a stored or synthetic version row."""
    if not isinstance(row, dict):
        return BUILD_STATUS_BROKEN
    explicit = str(row.get("build_status") or "").strip().lower()
    if explicit == BUILD_STATUS_BUILDING:
        if task_running is False:
            return BUILD_STATUS_BROKEN
        return BUILD_STATUS_BUILDING
    if explicit in KNOWN_BUILD_STATUSES:
        if explicit == BUILD_STATUS_READY and not engine_row_has_required_paths(
            engine, row
        ):
            return BUILD_STATUS_BROKEN
        return explicit
    if not engine_row_has_required_paths(engine, row):
        return BUILD_STATUS_BROKEN
    return BUILD_STATUS_READY


def engine_version_is_retryable(
    engine: str,
    row: Optional[dict],
    *,
    status: Optional[str] = None,
) -> bool:
    if not isinstance(row, dict):
        return False
    resolved = status or normalize_engine_version_status(engine, row)
    if resolved not in RETRYABLE_STATUSES:
        return False
    if engine in ("llama_cpp", "ik_llama", "audio_cpp"):
        return bool(
            str(
                row.get("source_ref")
                or row.get("source_branch")
                or row.get("source_commit")
                or ""
            ).strip()
        )
    if engine in ("lmdeploy", "1cat_vllm"):
        kind = str(row.get("install_type") or row.get("type") or "").strip().lower()
        if kind in {"source", "fork", "patched", "local"}:
            return bool(
                str(row.get("source_repo") or "").strip()
                and str(row.get("source_branch") or row.get("source_ref") or "").strip()
            )
        return kind in {"pip", "release"} or bool(str(row.get("venv_path") or "").strip())
    return False


def repair_stale_building_versions(store, *, get_task=None) -> int:
    """Persist ``building`` rows whose task is gone as ``broken`` so they stay visible."""
    repaired = 0
    for engine in ENGINE_REGISTRY:
        for row in list(store.get_engine_versions(engine) or []):
            if str(row.get("build_status") or "").strip().lower() != BUILD_STATUS_BUILDING:
                continue
            task_id = str(row.get("build_task_id") or "").strip()
            running = False
            if task_id and callable(get_task):
                task = get_task(task_id) or {}
                running = str(task.get("status") or "") == "running"
            if running:
                continue
            version = str(row.get("version") or "").strip()
            if not version:
                continue
            store.update_engine_version(
                engine,
                version,
                {
                    "build_status": BUILD_STATUS_BROKEN,
                    "build_error": row.get("build_error")
                    or "Build was interrupted before it finished",
                },
            )
            repaired += 1
    return repaired


def _list_subdirs(root: str) -> List[str]:
    if not root or not os.path.isdir(root):
        return []
    names = []
    try:
        entries = os.listdir(root)
    except OSError as exc:
        logger.debug("Could not list engine root %s: %s", root, exc)
        return []
    for name in entries:
        if name in _SKIP_DIR_NAMES or name.startswith("."):
            continue
        path = os.path.join(root, name)
        if os.path.isdir(path):
            names.append(name)
    return names


def _realpath(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:
        return os.path.abspath(path)


def _safe_under(root: str, path: str) -> bool:
    if not root or not path:
        return False
    root_real = _realpath(root)
    path_real = _realpath(path)
    try:
        return os.path.commonpath([root_real, path_real]) == root_real
    except ValueError:
        return False


def read_git_origin(repo_dir: str) -> Optional[str]:
    git_dir = _resolve_git_dir(repo_dir)
    if not git_dir:
        return None
    config_path = os.path.join(git_dir, "config")
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            in_origin = False
            for raw in handle:
                line = raw.strip()
                if line.startswith("[") and line.endswith("]"):
                    in_origin = line.lower() in {"[remote \"origin\"]", "[remote 'origin']"}
                    continue
                if in_origin and line.lower().startswith("url"):
                    _, value = line.split("=", 1)
                    url = value.strip()
                    return url or None
    except OSError:
        return None
    return None


def read_git_head_ref(repo_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(ref, ref_type)`` from HEAD (branch name, or commit SHA)."""
    git_dir = _resolve_git_dir(repo_dir)
    if not git_dir:
        return None, None
    head_path = os.path.join(git_dir, "HEAD")
    try:
        with open(head_path, "r", encoding="utf-8") as handle:
            value = handle.read().strip()
    except OSError:
        return None, None
    match = re.match(r"ref:\s*refs/heads/(.+)$", value)
    if match:
        return match.group(1).strip() or None, "branch"
    if re.fullmatch(r"[0-9a-fA-F]{7,40}", value or ""):
        return value, "commit"
    return None, None


def _resolve_git_dir(repo_dir: str) -> Optional[str]:
    git_path = os.path.join(repo_dir, ".git")
    if os.path.isdir(git_path):
        return git_path
    if os.path.isfile(git_path):
        try:
            with open(git_path, "r", encoding="utf-8") as handle:
                line = handle.readline().strip()
        except OSError:
            return None
        if line.lower().startswith("gitdir:"):
            target = line.split(":", 1)[1].strip()
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(repo_dir, target))
            return target if os.path.isdir(target) else None
    return None


def infer_native_engine_from_dir(version_dir: str) -> str:
    clone_dir = os.path.join(version_dir, "llama.cpp")
    origin = read_git_origin(clone_dir) or ""
    parsed = github_owner_repo(origin)
    if parsed:
        _owner, repo = parsed
        if "ik_llama" in repo.lower():
            return "ik_llama"
        if "llama.cpp" in repo.lower() or repo.lower() == "llama.cpp":
            return "llama_cpp"
    lowered = origin.lower()
    if "ik_llama" in lowered:
        return "ik_llama"
    return "llama_cpp"


def infer_checkout_metadata(clone_dir: str) -> Dict[str, Any]:
    origin = read_git_origin(clone_dir) or ""
    ref, ref_type = read_git_head_ref(clone_dir)
    meta: Dict[str, Any] = {}
    if origin:
        meta["source_repo"] = origin
    if ref:
        meta["source_ref"] = ref
        meta["source_ref_type"] = ref_type
        if ref_type == "branch":
            meta["source_branch"] = ref
        if ref_type == "commit":
            meta["source_commit"] = ref
    return meta


def claimed_install_paths(engine: str, row: dict, roots: Dict[str, str]) -> Set[str]:
    """Return realpaths of directories this version row already owns."""
    claimed: Set[str] = set()
    if not isinstance(row, dict):
        return claimed
    version = str(row.get("version") or "").strip()
    root = roots.get(engine) or ""
    for key in ("install_dir", "source_path"):
        path = str(row.get(key) or "").strip()
        if path:
            claimed.add(_realpath(path if os.path.isdir(path) else os.path.dirname(path)))
    venv = str(row.get("venv_path") or "").strip()
    if venv:
        claimed.add(_realpath(os.path.dirname(venv) if os.path.basename(venv) == "venv" else venv))
    binary = str(row.get("binary_path") or row.get("server_binary_path") or "").strip()
    if binary and root and _safe_under(root, binary):
        rel = os.path.relpath(_realpath(binary), _realpath(root))
        first = rel.split(os.sep)[0]
        if first and first not in {".", ".."}:
            claimed.add(_realpath(os.path.join(root, first)))
    if version and root:
        claimed.add(_realpath(os.path.join(root, version)))
    return claimed


def _orphan_error_message() -> str:
    return "On-disk engine folder has no configuration entry"


def _native_clone_dir(version_dir: str, engine: str) -> str:
    if engine == "audio_cpp":
        return os.path.join(version_dir, "source")
    return os.path.join(version_dir, "llama.cpp")


def collect_orphan_engine_rows(
    store,
    roots: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Return synthetic broken rows for install dirs not owned by ``engines.yaml``."""
    roots = roots or discover_engine_install_roots()
    claimed: Set[str] = set()
    registered_names: Dict[str, Set[str]] = {
        engine: {
            str(row.get("version") or "")
            for row in store.get_engine_versions(engine) or []
            if row.get("version")
        }
        for engine in ENGINE_REGISTRY
    }
    for engine in ENGINE_REGISTRY:
        for row in store.get_engine_versions(engine) or []:
            claimed |= claimed_install_paths(engine, row, roots)

    orphans: List[dict] = []
    llama_root = roots.get("llama_cpp") or roots.get("ik_llama") or ""
    if llama_root:
        for name in _list_subdirs(llama_root):
            version_dir = os.path.join(llama_root, name)
            real = _realpath(version_dir)
            if real in claimed or name in registered_names["llama_cpp"] or name in registered_names["ik_llama"]:
                continue
            engine = infer_native_engine_from_dir(version_dir)
            meta = infer_checkout_metadata(_native_clone_dir(version_dir, engine))
            orphans.append(
                _synthetic_orphan_row(
                    engine,
                    name,
                    version_dir,
                    extra=meta,
                )
            )

    audio_root = roots.get("audio_cpp") or ""
    if audio_root:
        for name in _list_subdirs(audio_root):
            version_dir = os.path.join(audio_root, name)
            real = _realpath(version_dir)
            if real in claimed or name in registered_names.get("audio_cpp", set()):
                continue
            meta = infer_checkout_metadata(_native_clone_dir(version_dir, "audio_cpp"))
            orphans.append(
                _synthetic_orphan_row("audio_cpp", name, version_dir, extra=meta)
            )

    for engine in ("lmdeploy", "1cat_vllm"):
        root = roots.get(engine) or ""
        if not root:
            continue
        for name in _list_subdirs(root):
            version_dir = os.path.join(root, name)
            real = _realpath(version_dir)
            if real in claimed or name in registered_names.get(engine, set()):
                continue
            meta = infer_checkout_metadata(os.path.join(version_dir, "source"))
            extra = dict(meta)
            venv_path = os.path.join(version_dir, "venv")
            if os.path.isdir(venv_path):
                extra["venv_path"] = venv_path
            orphans.append(_synthetic_orphan_row(engine, name, version_dir, extra=extra))
    return orphans


def _synthetic_orphan_row(
    engine: str, version: str, install_dir: str, extra: Optional[dict] = None
) -> dict:
    labels = extra or {}
    row = {
        "engine": engine,
        "version": version,
        "type": "broken",
        "install_type": str(labels.get("install_type") or "source"),
        "is_fork": bool(labels.get("is_fork")),
        "build_status": BUILD_STATUS_BROKEN,
        "build_error": _orphan_error_message(),
        "install_dir": install_dir,
        "orphan": True,
        "repository_source": labels.get("repository_source") or ENGINE_REPO_LABELS.get(engine),
        "source_repo": labels.get("source_repo"),
        "source_ref": labels.get("source_ref"),
        "source_ref_type": labels.get("source_ref_type"),
        "source_branch": labels.get("source_branch"),
        "source_commit": labels.get("source_commit"),
        "venv_path": labels.get("venv_path"),
        "installed_at": None,
    }
    row["retryable"] = engine_version_is_retryable(engine, row)
    return row


def discover_engine_install_roots() -> Dict[str, str]:
    """Resolve on-disk roots for each engine's versioned install folders."""
    roots: Dict[str, str] = {}
    try:
        from backend.llama_manager import LlamaManager

        llama_dir = LlamaManager().llama_dir
        roots["llama_cpp"] = llama_dir
        roots["ik_llama"] = llama_dir
    except Exception as exc:
        logger.debug("Could not resolve llama.cpp install root: %s", exc)
    try:
        from backend.audio_cpp_manager import get_audio_cpp_manager

        roots["audio_cpp"] = get_audio_cpp_manager().builds_dir
    except Exception as exc:
        logger.debug("Could not resolve audio.cpp install root: %s", exc)
    try:
        from backend.lmdeploy_manager import get_lmdeploy_manager

        roots["lmdeploy"] = get_lmdeploy_manager()._root_dir
    except Exception as exc:
        logger.debug("Could not resolve LMDeploy install root: %s", exc)
    try:
        from backend.onecat_vllm_manager import get_onecat_vllm_manager

        roots["1cat_vllm"] = get_onecat_vllm_manager()._root_dir
    except Exception as exc:
        logger.debug("Could not resolve 1Cat-vLLM install root: %s", exc)
    return roots


def annotate_version_row(engine: str, row: dict, *, task_running: Optional[bool] = None) -> dict:
    """Copy a store row and attach normalized status + retryable flag."""
    annotated = dict(row or {})
    status = normalize_engine_version_status(
        engine, annotated, task_running=task_running
    )
    annotated["build_status"] = status
    annotated["retryable"] = engine_version_is_retryable(
        engine, annotated, status=status
    )
    if not annotated.get("repository_source"):
        annotated["repository_source"] = ENGINE_REPO_LABELS.get(engine)
    return annotated


def is_safe_install_dir(root: str, path: str) -> bool:
    """True when ``path`` is a direct child directory of ``root``."""
    if not root or not path:
        return False
    root_real = _realpath(root)
    path_real = _realpath(path)
    if not os.path.isdir(path_real):
        return False
    if not _safe_under(root, path_real):
        return False
    parent = os.path.dirname(path_real)
    return parent == root_real and path_real != root_real


def resolve_install_dir(
    engine: str, row: dict, roots: Optional[Dict[str, str]] = None
) -> Optional[str]:
    roots = roots or discover_engine_install_roots()
    root = roots.get(engine) or ""
    if engine in _NATIVE_SHARED_ROOT_ENGINES:
        root = roots.get("llama_cpp") or roots.get("ik_llama") or root
    explicit = str(row.get("install_dir") or "").strip()
    if explicit and is_safe_install_dir(root, explicit):
        return _realpath(explicit)
    version = str(row.get("version") or "").strip()
    if version and root:
        candidate = os.path.join(root, version)
        if is_safe_install_dir(root, candidate):
            return _realpath(candidate)
    for key in ("source_path", "venv_path", "binary_path", "server_binary_path"):
        raw = str(row.get(key) or "").strip()
        if not raw:
            continue
        path = raw if os.path.isdir(raw) else os.path.dirname(raw)
        if key == "venv_path" and os.path.basename(path) == "venv":
            path = os.path.dirname(path)
        if is_safe_install_dir(root, path):
            return _realpath(path)
        if root and _safe_under(root, path):
            rel = os.path.relpath(_realpath(path), _realpath(root))
            first = rel.split(os.sep)[0]
            candidate = os.path.join(root, first)
            if is_safe_install_dir(root, candidate):
                return _realpath(candidate)
    return None


def find_disk_version(
    version_id: str, store, roots: Optional[Dict[str, str]] = None
) -> Tuple[Optional[dict], Optional[str]]:
    """Resolve ``engine:version`` to a store row or a synthetic orphan row."""
    engine = None
    version = str(version_id or "").strip()
    if ":" in version:
        engine, version = version.split(":", 1)
    if engine and engine not in ENGINE_REGISTRY:
        return None, None
    engines: Sequence[str] = (engine,) if engine else tuple(ENGINE_REGISTRY)
    for eng in engines:
        for row in store.get_engine_versions(eng) or []:
            if str(row.get("version")) == version:
                return dict(row), eng
    roots = roots or discover_engine_install_roots()
    search_engines = list(engines)
    if engine in _NATIVE_SHARED_ROOT_ENGINES or engine is None:
        # Shared llama.cpp root: try both native engines when looking at disk.
        for extra in _NATIVE_SHARED_ROOT_ENGINES:
            if extra not in search_engines:
                search_engines.append(extra)
    for eng in search_engines:
        root = roots.get(eng) or ""
        if eng in _NATIVE_SHARED_ROOT_ENGINES:
            root = roots.get("llama_cpp") or roots.get("ik_llama") or root
        if not version or not root:
            continue
        path = os.path.join(root, version)
        if not is_safe_install_dir(root, path):
            continue
        inferred = eng
        extra: Dict[str, Any] = {}
        if eng in _NATIVE_SHARED_ROOT_ENGINES:
            inferred = infer_native_engine_from_dir(path)
            extra = infer_checkout_metadata(_native_clone_dir(path, inferred))
        elif eng == "audio_cpp":
            extra = infer_checkout_metadata(_native_clone_dir(path, eng))
        else:
            extra = infer_checkout_metadata(os.path.join(path, "source"))
            venv_path = os.path.join(path, "venv")
            if os.path.isdir(venv_path):
                extra["venv_path"] = venv_path
        return _synthetic_orphan_row(inferred, version, path, extra=extra), inferred
    return None, None
