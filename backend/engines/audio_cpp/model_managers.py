"""Resolve the audio.cpp v2 model manager.

Packages come from ``tools/model_manager_v2.py list --json``, backed by
``model_specs/*.json``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


MANAGER_V2_BASENAME = "model_manager_v2.py"

_GGUF_PACKAGE_SIDECARS = ("tokenizer.model", "config.yaml")


def gguf_snapshot_sidecar_prefixes(files: Optional[Sequence[Any]] = None) -> List[str]:
    """HF prefixes for files audio.cpp expects next to a GGUF weight.

    ``model_specs`` GGUF packages often list only the ``.gguf``. Some packages still
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


def resolve_model_manager_path(
    source_path: str = "",
    *,
    version_row: Optional[dict] = None,
) -> str:
    """Return ``model_manager_v2.py`` when the active tree has it."""
    return resolve_model_manager_v2_path(source_path, version_row=version_row)


def manager_paths_for_source(source_path: str) -> Dict[str, str]:
    """Build manager path fields for a freshly built/synced audio.cpp tree."""
    source = str(source_path or "").strip()
    v2 = resolve_model_manager_v2_path(source)
    return {
        "model_manager_path": v2,
        "model_manager_v2_path": v2,
    }


def manager_script_kind(path: str) -> str:
    """Classify a manager script path as ``v2`` or ``unknown``."""
    base = os.path.basename(str(path or ""))
    if base == MANAGER_V2_BASENAME:
        return "v2"
    return "unknown"


def catalog_json_has_identity(row: dict) -> bool:
    """True when a ``list --json`` row carries enough identity for contract grading."""
    if not isinstance(row, dict):
        return False
    return all(key in row for key in ("family", "id", "target_directory", "repo"))


def normalize_v2_catalog_packages(
    rows: Sequence[dict], *, source_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Map ``model_manager_v2 list --json`` rows into Studio package dicts."""
    from backend.engines.audio_cpp.contracts import load_family_contracts

    declarations: Dict[str, List[dict]] = {}
    for contract in load_family_contracts(source_path).values() if source_path else []:
        defaults = contract.get("package_defaults") or {}
        for package in contract.get("packages") or []:
            if not isinstance(package, dict) or not package.get("id"):
                continue
            declarations.setdefault(package["id"], []).append({
                **package,
                "family": contract["family"],
                "download": {**(defaults.get("download") or {}), **(package.get("download") or {})},
            })

    packages: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        package_id = str(row.get("id") or "").strip()
        if not package_id:
            continue
        matches = [item for item in declarations.get(package_id, [])
                   if not row.get("family") or item["family"] == row["family"]]
        if len(matches) == 1:
            # The manager's compact listing omits these fields. Resolve them
            # from the exact engine declaration, never package-name heuristics.
            row = {**matches[0], **row,
                   "download": {**matches[0]["download"], **(row.get("download") or {})}}
        download = row.get("download") if isinstance(row.get("download"), dict) else {}
        authoritative_files = "files" in row or "required_files" in row
        repo = str(download.get("repo") or row.get("repo") or "").strip()
        family = str(row.get("family") or "").strip()
        format_name = str(row.get("format") or "").strip()
        precision = str(row.get("precision") or "").strip()
        bits = [part for part in (format_name, precision) if part]
        description = str(row.get("description") or " ".join(bits))
        if row.get("default"):
            description = (description + " (default)").strip()
        declared_files = list(row.get("files") or row.get("required_files") or [])
        include_prefixes = list(row.get("include_prefixes") or declared_files)
        target_directory = str(
            row.get("target_directory") or package_id
        ).strip() or package_id
        strip_prefix = str(row.get("strip_prefix") or "")
        # ``list --json`` omits strip_prefix; v2 still installs into
        # target_directory after stripping that HF prefix from remote paths.
        if not authoritative_files and not strip_prefix and str(format_name).lower() == "gguf":
            candidate = str(target_directory or "").replace("\\", "/").strip("/")
            if candidate and candidate != ".":
                strip_prefix = candidate
        for extra in ([] if authoritative_files else gguf_snapshot_sidecar_prefixes(declared_files or include_prefixes)):
            if extra not in include_prefixes:
                include_prefixes.append(extra)
        if not authoritative_files and str(format_name).lower() == "gguf":
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
                "required_files": [
                    str(path).removeprefix(strip_prefix.rstrip("/") + "/")
                    if strip_prefix and "files" in row else str(path)
                    for path in declared_files
                ],
                **({"files": declared_files, "strip_prefix": strip_prefix, "layout_source": "engine"}
                   if "files" in row else {}),
                "family": family,
                "standalone": True,
                "format": format_name,
                "precision": precision,
                "default": bool(row.get("default")),
                **({"gated": row["gated"]} if isinstance(row.get("gated"), bool) else {}),
                "source": {
                    **({"gated": row["download"]["gated"]}
                       if isinstance(row.get("download"), dict)
                       and isinstance(row["download"].get("gated"), bool) else {}),
                    "kind": str(download.get("kind") or "huggingface_snapshot"),
                    "repo_id": repo,
                    "revision": str(download.get("revision") or row.get("revision") or "main"),
                    "include_prefixes": include_prefixes,
                    "exclude_prefixes": [],
                    "strip_prefix": strip_prefix,
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
