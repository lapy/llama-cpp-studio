"""Lean file-ledger helpers for Hugging Face model records."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, Optional

MODEL_FILE_ROLES = frozenset({"weight", "shard", "mmproj", "mtp", "dflash"})


def infer_file_role(filename: str) -> str:
    """Infer a model-file role without depending on the Hugging Face module."""
    name = str(filename or "").lower()
    if "mmproj" in name:
        return "mmproj"
    if "dflash" in name:
        return "dflash"
    if re.search(r"(?:^|[/\\\-_.])mtp(?:[/\\\-_.]|$)", name):
        return "mtp"
    if re.search(r"-\d{5}-of-\d{5}(?=\.gguf$|\.safetensors$)", name) or re.search(
        r"\.(?:gguf|safetensors)\.part\d+of\d+$", name
    ):
        return "shard"
    return "weight"


def normalize_model_file(entry: Any) -> Optional[Dict[str, Any]]:
    """Return a compact, normalized ledger entry or ``None``."""
    if not isinstance(entry, dict):
        return None
    filename = str(entry.get("filename") or "").strip()
    if not filename:
        return None

    role = str(entry.get("role") or infer_file_role(filename)).lower()
    if role not in MODEL_FILE_ROLES:
        role = infer_file_role(filename)

    normalized: Dict[str, Any] = {"filename": filename, "role": role}
    size = entry.get("size")
    if size is None:
        size = entry.get("file_size")
    if isinstance(size, (int, float)) and size >= 0:
        normalized["size"] = int(size)

    for key in ("etag", "sha256", "downloaded_at"):
        value = entry.get(key)
        if value not in (None, ""):
            normalized[key] = str(value)
    return normalized


def normalize_model_files(value: Any) -> list[Dict[str, Any]]:
    """Normalize and de-duplicate a model's file ledger by filename."""
    if not isinstance(value, (list, tuple)):
        return []
    by_filename: Dict[str, Dict[str, Any]] = {}
    for raw in value:
        entry = normalize_model_file(raw)
        if not entry:
            continue
        filename = entry["filename"]
        existing = by_filename.get(filename, {})
        by_filename[filename] = {**existing, **entry}
    return list(by_filename.values())


def iter_model_files(
    model: Dict[str, Any], roles: Optional[Iterable[str]] = None
) -> Iterator[Dict[str, Any]]:
    """Iterate normalized entries, optionally restricted to file roles."""
    allowed = {str(role).lower() for role in roles} if roles is not None else None
    for entry in normalize_model_files(model.get("files")):
        if allowed is None or entry.get("role") in allowed:
            yield entry


def shard_sort_key(entry: Dict[str, Any]) -> tuple[int, int, str]:
    """Sort single weights first, followed by numbered shards."""
    filename = str(entry.get("filename") or "")
    lower = filename.lower()
    match = re.search(r"-(\d{5})-of-\d{5}(?=\.gguf$|\.safetensors$)", lower)
    if not match:
        match = re.search(r"\.(?:gguf|safetensors)\.part(\d+)of\d+$", lower)
    if match:
        return (1, int(match.group(1)), lower)
    return (0, 0, lower)


def upsert_model_file(
    store,
    model_id: str,
    entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Merge one file entry into a stored model, preserving omitted identity fields."""
    incoming = normalize_model_file(entry)
    if not incoming:
        raise ValueError("Model file entry requires a filename")
    model = store.get_model(model_id)
    if not model:
        return None

    files = normalize_model_files(model.get("files"))
    existing_index = next(
        (
            index
            for index, current in enumerate(files)
            if current.get("filename") == incoming["filename"]
        ),
        None,
    )
    if existing_index is None:
        if "downloaded_at" not in incoming:
            incoming["downloaded_at"] = datetime.now(timezone.utc).isoformat()
        files.append(incoming)
    else:
        files[existing_index] = {**files[existing_index], **incoming}
    return store.update_model(model_id, {"files": files})


def remove_model_files(
    store,
    model_id: str,
    *,
    filenames: Optional[Iterable[str]] = None,
    roles: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Remove ledger entries matching filenames or roles."""
    model = store.get_model(model_id)
    if not model:
        return None
    names = {str(value) for value in filenames or [] if value}
    role_set = {str(value).lower() for value in roles or [] if value}
    files = [
        entry
        for entry in normalize_model_files(model.get("files"))
        if entry.get("filename") not in names and entry.get("role") not in role_set
    ]
    return store.update_model(model_id, {"files": files})
