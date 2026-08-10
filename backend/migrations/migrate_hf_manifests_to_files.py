"""TEMPORARY: migrate legacy HF sidecars into model ``files`` ledgers.

Remove this module and its DataStore startup hook in a future release after
existing installations have had time to run the migration.
"""

from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Dict, Iterable, Tuple

from backend.model_files import infer_file_role, normalize_model_files

MIGRATION_MARKER = "hf_manifest_migration"
MIGRATION_VERSION = 1


def _safe_repo_name(huggingface_id: str) -> str:
    return str(huggingface_id or "unknown").replace("/", "_") or "unknown"


def _manifest_path(data_root: str, model_format: str, huggingface_id: str) -> str:
    return os.path.join(
        data_root,
        "models",
        model_format,
        _safe_repo_name(huggingface_id),
        "manifest.json",
    )


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _quantization(filename: str) -> str:
    match = re.search(
        r"(?:^|[-_.])((?:i\d+-)?(?:q|iq|tq|fp|bf|f)\d+(?:_[a-z0-9]+)*)",
        str(filename or ""),
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else ""


def _legacy_files_for_model(model: Dict[str, Any], manifest: Any) -> list[Dict[str, Any]]:
    fmt = str(model.get("format") or model.get("model_format") or "").lower()
    if fmt == "safetensors":
        raw_files = manifest.get("files") if isinstance(manifest, dict) else []
    elif fmt == "gguf":
        rows = manifest if isinstance(manifest, list) else []
        model_id = model.get("id")
        quant = str(model.get("quantization") or "").lower()
        raw_files = [
            row
            for row in rows
            if isinstance(row, dict)
            and (
                row.get("model_id") == model_id
                or (
                    not row.get("model_id")
                    and quant
                    and _quantization(row.get("filename", "")) == quant
                )
            )
        ]
    else:
        raw_files = []

    return normalize_model_files(
        [
            {
                "filename": row.get("filename"),
                "role": infer_file_role(row.get("filename", "")),
                "size": row.get("file_size", row.get("size")),
                "etag": row.get("etag"),
                "sha256": row.get("sha256"),
                "downloaded_at": row.get("downloaded_at"),
            }
            for row in raw_files or []
            if isinstance(row, dict)
        ]
    )


def _hub_cache_root() -> str:
    explicit = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
    if explicit:
        return explicit
    hf_home = os.getenv("HF_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache", "huggingface"
    )
    return os.path.join(hf_home, "hub")


def _cached_repo_files(huggingface_id: str) -> Iterable[Tuple[str, int]]:
    snapshots = os.path.join(
        _hub_cache_root(),
        "models--" + huggingface_id.replace("/", "--"),
        "snapshots",
    )
    if not os.path.isdir(snapshots):
        return []
    found: Dict[str, int] = {}
    for revision in os.listdir(snapshots):
        root = os.path.join(snapshots, revision)
        if not os.path.isdir(root):
            continue
        for directory, _subdirs, filenames in os.walk(root):
            for filename in filenames:
                path = os.path.join(directory, filename)
                relative = os.path.relpath(path, root).replace(os.sep, "/")
                try:
                    found[relative] = os.path.getsize(os.path.realpath(path))
                except OSError:
                    continue
    return found.items()


def _cached_files_for_model(model: Dict[str, Any]) -> list[Dict[str, Any]]:
    huggingface_id = model.get("huggingface_id")
    fmt = str(model.get("format") or model.get("model_format") or "").lower()
    quant = str(model.get("quantization") or "").lower()
    if not huggingface_id:
        return []
    entries = []
    for filename, size in _cached_repo_files(huggingface_id):
        lower = filename.lower()
        if fmt == "safetensors" and not lower.endswith(".safetensors"):
            continue
        if fmt == "gguf":
            if not re.search(r"\.gguf(?:\.|$)", lower):
                continue
            role = infer_file_role(filename)
            selected_companions = {
                model.get("mmproj_filename"),
                model.get("mtp_filename"),
                model.get("dflash_filename"),
            }
            if role in {"mmproj", "mtp", "dflash"}:
                if filename not in selected_companions:
                    continue
            elif quant and _quantization(filename) != quant:
                continue
        entries.append(
            {
                "filename": filename,
                "role": infer_file_role(filename),
                "size": size,
            }
        )
    return normalize_model_files(entries)


def _promote_limits(model: Dict[str, Any], manifest: Any) -> None:
    if model.get("max_context_length") and model.get("layer_count"):
        return
    fmt = str(model.get("format") or model.get("model_format") or "").lower()
    if fmt == "gguf" and isinstance(manifest, list):
        relevant = _legacy_files_for_model(model, manifest)
        names = {entry["filename"] for entry in relevant}
        row = next(
            (
                item
                for item in manifest
                if isinstance(item, dict) and item.get("filename") in names
            ),
            {},
        )
        max_ctx = row.get("max_context_length")
        layer_count = (row.get("gguf_layer_info") or {}).get("layer_count")
    elif fmt == "safetensors" and isinstance(manifest, dict):
        metadata = manifest.get("metadata") or {}
        config = metadata.get("config") or {}
        max_ctx = manifest.get("max_context_length") or metadata.get(
            "max_context_length"
        )
        layer_count = metadata.get("layer_count")
        if not layer_count:
            for key in (
                "num_hidden_layers",
                "n_layer",
                "num_layers",
                "n_layers",
                "decoder_layers",
                "encoder_layers",
            ):
                value = config.get(key)
                if isinstance(value, (int, float)) and value > 0:
                    layer_count = int(value) + 1
                    break
    else:
        return
    if not model.get("max_context_length") and isinstance(max_ctx, (int, float)):
        model["max_context_length"] = int(max_ctx)
    if not model.get("layer_count") and isinstance(layer_count, (int, float)):
        model["layer_count"] = int(layer_count)


def migrate_document(
    document: Dict[str, Any], data_root: str
) -> tuple[Dict[str, Any], bool, list[str]]:
    """Return migrated document, change flag, and sidecars safe to delete."""
    root = copy.deepcopy(document if isinstance(document, dict) else {})
    if root.get(MIGRATION_MARKER) == MIGRATION_VERSION:
        return root, False, []

    changed = False
    complete = True
    for model in root.get("models") or []:
        if not isinstance(model, dict):
            continue
        fmt = str(model.get("format") or model.get("model_format") or "").lower()
        huggingface_id = model.get("huggingface_id")
        if fmt not in {"gguf", "safetensors"} or not huggingface_id:
            continue
        existing_files = normalize_model_files(model.get("files"))
        if any(
            entry.get("role") in {"weight", "shard"} for entry in existing_files
        ):
            continue
        path = _manifest_path(data_root, fmt, huggingface_id)
        manifest = _load_json(path) if os.path.isfile(path) else None
        files = _legacy_files_for_model(model, manifest) if manifest is not None else []
        if not files:
            files = _cached_files_for_model(model)
        if not files:
            complete = False
            continue
        model["files"] = normalize_model_files([*existing_files, *files])
        if manifest is not None:
            _promote_limits(model, manifest)
        changed = True

    if complete:
        root[MIGRATION_MARKER] = MIGRATION_VERSION
        changed = True
    cleanup = []
    for fmt in ("gguf", "safetensors"):
        base = os.path.join(data_root, "models", fmt)
        if not os.path.isdir(base):
            continue
        for directory, _subdirs, filenames in os.walk(base):
            if "manifest.json" in filenames:
                cleanup.append(os.path.join(directory, "manifest.json"))
    return root, changed, cleanup if complete else []


def cleanup_legacy_sidecars(paths: Iterable[str], data_root: str) -> None:
    """Delete migrated sidecars and empty HF-only model directories."""
    for path in paths:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
    models_root = os.path.join(data_root, "models")
    for fmt in ("gguf", "safetensors"):
        base = os.path.join(models_root, fmt)
        if not os.path.isdir(base):
            continue
        for directory, _subdirs, _files in os.walk(base, topdown=False):
            try:
                os.rmdir(directory)
            except OSError:
                pass
