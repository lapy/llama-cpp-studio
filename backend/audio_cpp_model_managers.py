"""Resolve audio.cpp model-manager scripts (v2 preferred, legacy fallback).

Upstream (audio.cpp main) made ``tools/model_manager_v2.py`` the supported
download path: packages come from ``model_specs/*.json`` (GGUF-first). The older
hardcoded catalog lives as ``tools/model_manager_deprecated.py`` (still named
``model_manager.py`` on older checkouts) for composite/converter leftovers.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


MANAGER_V2_BASENAME = "model_manager_v2.py"
MANAGER_LEGACY_BASENAMES = (
    "model_manager_deprecated.py",
    "model_manager.py",
)

_GGUF_PACKAGE_SIDECARS = ("tokenizer.model", "config.yaml")


def gguf_snapshot_sidecar_prefixes(files: Optional[Sequence[Any]] = None) -> List[str]:
    """HF prefixes for files audio.cpp expects next to a GGUF weight.

    ``model_specs`` GGUF packages often list only the ``.gguf``. PocketTTS still
    loads voices from ``embeddings/<id>.safetensors`` beside that file.
    """
    prefixes: List[str] = []
    seen = set()

    def add(value: str) -> None:
        text = str(value or "").replace("\\", "/").lstrip("/")
        if not text or text in seen:
            return
        seen.add(text)
        prefixes.append(text)

    for item in files or []:
        path = str(item or "").replace("\\", "/").lstrip("/")
        if not path.lower().endswith(".gguf"):
            continue
        parent = path.rsplit("/", 1)[0] if "/" in path else ""
        if parent:
            add(f"{parent}/embeddings/")
            for name in _GGUF_PACKAGE_SIDECARS:
                add(f"{parent}/{name}")
        else:
            add("embeddings/")
            for name in _GGUF_PACKAGE_SIDECARS:
                add(name)
    return prefixes


def _tools_dir(source_path: str) -> str:
    return os.path.join(str(source_path or "").rstrip(os.sep), "tools")


def resolve_model_manager_v2_path(
    source_path: str = "",
    *,
    version_row: Optional[dict] = None,
) -> str:
    """Return path to ``model_manager_v2.py`` when present."""
    row = version_row if isinstance(version_row, dict) else {}
    explicit = str(row.get("model_manager_v2_path") or "").strip()
    if explicit and os.path.isfile(explicit):
        return explicit
    source = str(row.get("source_path") or source_path or "").strip()
    if not source:
        return ""
    candidate = os.path.join(_tools_dir(source), MANAGER_V2_BASENAME)
    return candidate if os.path.isfile(candidate) else ""


def resolve_model_manager_legacy_path(
    source_path: str = "",
    *,
    version_row: Optional[dict] = None,
) -> str:
    """Return path to deprecated/legacy ``model_manager*.py`` when present."""
    row = version_row if isinstance(version_row, dict) else {}
    for key in ("model_manager_legacy_path", "model_manager_deprecated_path"):
        explicit = str(row.get(key) or "").strip()
        if explicit and os.path.isfile(explicit):
            return explicit
    # Older Studio rows pointed model_manager_path at the legacy script.
    primary = str(row.get("model_manager_path") or "").strip()
    if primary and os.path.isfile(primary):
        base = os.path.basename(primary)
        if base in MANAGER_LEGACY_BASENAMES:
            return primary
    source = str(row.get("source_path") or source_path or "").strip()
    if not source:
        return ""
    tools = _tools_dir(source)
    for name in MANAGER_LEGACY_BASENAMES:
        candidate = os.path.join(tools, name)
        if os.path.isfile(candidate):
            return candidate
    return ""


def resolve_model_manager_path(
    source_path: str = "",
    *,
    version_row: Optional[dict] = None,
) -> str:
    """Primary manager for catalog listing: prefer v2, else legacy."""
    row = version_row if isinstance(version_row, dict) else {}
    v2 = resolve_model_manager_v2_path(source_path, version_row=row)
    if v2:
        return v2
    legacy = resolve_model_manager_legacy_path(source_path, version_row=row)
    if legacy:
        return legacy
    primary = str(row.get("model_manager_path") or "").strip()
    if primary and os.path.isfile(primary):
        return primary
    return ""


def manager_paths_for_source(source_path: str) -> Dict[str, str]:
    """Build manager path fields for a freshly built/synced audio.cpp tree."""
    source = str(source_path or "").strip()
    v2 = resolve_model_manager_v2_path(source)
    legacy = resolve_model_manager_legacy_path(source)
    primary = v2 or legacy
    out: Dict[str, str] = {
        "model_manager_path": primary,
        "model_manager_v2_path": v2,
        "model_manager_legacy_path": legacy,
    }
    return out


def manager_script_kind(path: str) -> str:
    """Classify a manager script path as ``v2``, ``legacy``, or ``unknown``."""
    base = os.path.basename(str(path or ""))
    if base == MANAGER_V2_BASENAME:
        return "v2"
    if base in MANAGER_LEGACY_BASENAMES:
        return "legacy"
    return "unknown"


def catalog_json_has_identity(row: dict) -> bool:
    """True when a ``list --json`` row carries enough identity for contract grading."""
    if not isinstance(row, dict):
        return False
    # Legacy model_manager list --json
    if all(key in row for key in ("family", "standalone", "tasks", "gated")):
        return True
    # model_manager_v2 list --json
    if all(key in row for key in ("family", "id", "target_directory", "repo")):
        return True
    return False


def normalize_v2_catalog_packages(rows: Sequence[dict]) -> List[Dict[str, Any]]:
    """Map ``model_manager_v2 list --json`` rows into Studio package dicts."""
    packages: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        package_id = str(row.get("id") or "").strip()
        if not package_id:
            continue
        repo = str(row.get("repo") or "").strip()
        family = str(row.get("family") or package_id).strip() or package_id
        format_name = str(row.get("format") or "").strip()
        precision = str(row.get("precision") or "").strip()
        bits = [part for part in (format_name, precision) if part]
        description = " ".join(bits)
        if row.get("default"):
            description = (description + " (default)").strip()
        declared_files = list(row.get("files") or row.get("required_files") or [])
        include_prefixes = list(row.get("include_prefixes") or declared_files)
        target_directory = str(
            row.get("target_directory") or package_id
        ).strip() or package_id
        strip_prefix = str(row.get("strip_prefix") or "")
        for extra in gguf_snapshot_sidecar_prefixes(declared_files or include_prefixes):
            if extra not in include_prefixes:
                include_prefixes.append(extra)
        if str(format_name).lower() == "gguf":
            remote_dir = str(strip_prefix or target_directory).replace("\\", "/").strip("/")
            if remote_dir and remote_dir != ".":
                for extra in (
                    f"{remote_dir}/embeddings/",
                    f"{remote_dir}/tokenizer.model",
                    f"{remote_dir}/config.yaml",
                ):
                    if extra not in include_prefixes:
                        include_prefixes.append(extra)
        packages.append(
            {
                "id": package_id,
                "display_name": str(row.get("display_name") or package_id).strip()
                or package_id,
                "target_directory": target_directory,
                "description": description,
                "required_files": declared_files,
                "family": family,
                "standalone": True,
                "format": format_name,
                "precision": precision,
                "default": bool(row.get("default")),
                "source": {
                    "kind": "huggingface_snapshot",
                    "repo_id": repo,
                    "revision": str(
                        (row.get("download") or {}).get("revision")
                        if isinstance(row.get("download"), dict)
                        else row.get("revision") or "main"
                    ),
                    "include_prefixes": include_prefixes,
                    "exclude_prefixes": [],
                    "strip_prefix": str(row.get("strip_prefix") or ""),
                },
                "installable": bool(repo),
                "install_kind": "snapshot",
                "manager_backend": "v2",
                "usage_examples": [],
            }
        )
    return packages


def merge_catalog_packages(
    preferred: Sequence[dict],
    extra: Sequence[dict],
) -> List[dict]:
    """Prefer packages from the first list; append extras with new ids only."""
    out: List[dict] = []
    seen = set()
    for package in list(preferred) + list(extra):
        if not isinstance(package, dict):
            continue
        package_id = str(package.get("id") or "").strip()
        if not package_id or package_id in seen:
            continue
        seen.add(package_id)
        out.append(package)
    return out
