"""Manage reference audio files for audio.cpp models."""

from __future__ import annotations

import os
import re
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

REFERENCE_AUDIO_SUBDIR = "refs"
REFERENCE_AUDIO_DATA_SUBDIR = os.path.join("models", "audio-cpp", "reference-audio")
ALLOWED_EXTENSIONS = frozenset({".wav"})
MAX_REFERENCE_AUDIO_BYTES = 60 * 1024 * 1024


def _is_wav_content(content: bytes) -> bool:
    return (
        len(content) >= 12
        and content[:4] == b"RIFF"
        and content[8:12] == b"WAVE"
    )


def get_audio_model_bundle_root(model: dict) -> str:
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    raw = str(
        artifact.get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    ).strip()
    if not raw:
        raise HTTPException(
            status_code=400,
            detail="Model has no installed bundle path",
        )
    return os.path.realpath(raw)


def _data_root() -> str:
    if os.path.isdir("/app/data"):
        return "/app/data"
    return os.path.abspath("data")


def _safe_storage_key(value: str) -> str:
    raw = str(value or "").strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("-._")[:64]
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{slug or 'audio-model'}-{digest}"


def reference_audio_storage_root(
    bundle_root: str,
    *,
    storage_key: Optional[str] = None,
) -> str:
    key = _safe_storage_key(storage_key or os.path.realpath(bundle_root))
    return os.path.join(_data_root(), REFERENCE_AUDIO_DATA_SUBDIR, key)


def reference_audio_dir(bundle_root: str, *, storage_key: Optional[str] = None) -> str:
    return os.path.join(
        reference_audio_storage_root(bundle_root, storage_key=storage_key),
        REFERENCE_AUDIO_SUBDIR,
    )


def legacy_reference_audio_dir(bundle_root: str) -> str:
    return os.path.join(os.path.realpath(bundle_root), REFERENCE_AUDIO_SUBDIR)


def relative_reference_path(filename: str) -> str:
    return f"{REFERENCE_AUDIO_SUBDIR}/{filename}"


def _ensure_within_root(root: str, target: str) -> None:
    root_real = os.path.realpath(root)
    target_real = os.path.realpath(target)
    try:
        common = os.path.commonpath([root_real, target_real])
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid path") from exc
    if common != root_real:
        raise HTTPException(status_code=400, detail="Path escapes model bundle")


def sanitize_reference_filename(name: str) -> str:
    base = os.path.basename(str(name or "").strip())
    if not base:
        raise HTTPException(status_code=400, detail="Filename is required")
    stem, ext = os.path.splitext(base)
    ext = ext.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(sorted(ALLOWED_EXTENSIONS))} files are supported",
        )
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    if not safe_stem:
        raise HTTPException(status_code=400, detail="Invalid filename")
    return f"{safe_stem}{ext}"


def _unique_filename(directory: str, filename: str) -> str:
    candidate = filename
    stem, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(os.path.join(directory, candidate)):
        candidate = f"{stem}_{counter}{ext}"
        counter += 1
    return candidate


def _format_entry(
    bundle_root: str,
    filename: str,
    *,
    storage_key: Optional[str] = None,
    legacy: bool = False,
) -> Dict[str, Any]:
    refs_dir = (
        legacy_reference_audio_dir(bundle_root)
        if legacy
        else reference_audio_dir(bundle_root, storage_key=storage_key)
    )
    path = os.path.join(refs_dir, filename)
    stat = os.stat(path)
    rel = relative_reference_path(filename)
    return {
        "filename": filename,
        "path": os.path.realpath(path),
        "relative_path": rel,
        "display_path": rel,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "storage": "legacy_bundle" if legacy else "data",
    }


def _list_reference_audio_dir(
    bundle_root: str,
    refs_dir: str,
    *,
    storage_key: Optional[str] = None,
    legacy: bool = False,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not os.path.isdir(refs_dir):
        return entries
    for name in sorted(os.listdir(refs_dir)):
        path = os.path.join(refs_dir, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            continue
        entries.append(
            _format_entry(
                bundle_root,
                name,
                storage_key=storage_key,
                legacy=legacy,
            )
        )
    return entries


def list_reference_audio(
    bundle_root: str,
    *,
    storage_key: Optional[str] = None,
    include_legacy: bool = True,
) -> List[Dict[str, Any]]:
    entries = _list_reference_audio_dir(
        bundle_root,
        reference_audio_dir(bundle_root, storage_key=storage_key),
        storage_key=storage_key,
    )
    if include_legacy:
        seen = {entry["filename"] for entry in entries}
        for entry in _list_reference_audio_dir(
            bundle_root,
            legacy_reference_audio_dir(bundle_root),
            legacy=True,
        ):
            if entry["filename"] not in seen:
                entries.append(entry)
    return entries


def _iter_config_string_values(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_config_string_values(child, nested)
    elif isinstance(value, str):
        yield prefix, value.replace("\\", "/")


def reference_path_candidates(path: str) -> set[str]:
    normalized = str(path or "").replace("\\", "/")
    candidates = {normalized} if normalized else set()
    if normalized:
        candidates.add(relative_reference_path(os.path.basename(normalized)))
    return candidates


def find_config_references(effective_config: dict, reference_path: str) -> List[str]:
    candidates = reference_path_candidates(reference_path)
    matches: List[str] = []
    for key in (
        "voice_presets",
        "speech_defaults",
        "default_voice_preset",
        "transcription_defaults",
        "task_defaults",
    ):
        for location, value in _iter_config_string_values(key, effective_config.get(key)):
            if value in candidates:
                matches.append(location)
    return matches


def save_reference_audio(
    bundle_root: str,
    *,
    filename: str,
    content: bytes,
    storage_key: Optional[str] = None,
) -> Dict[str, Any]:
    if len(content) > MAX_REFERENCE_AUDIO_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File exceeds maximum size of {MAX_REFERENCE_AUDIO_BYTES // (1024 * 1024)} MB",
        )
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if not _is_wav_content(content):
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid WAV",
        )

    safe_name = sanitize_reference_filename(filename)
    storage_root = reference_audio_storage_root(bundle_root, storage_key=storage_key)
    refs_dir = reference_audio_dir(bundle_root, storage_key=storage_key)
    os.makedirs(refs_dir, exist_ok=True)
    _ensure_within_root(storage_root, refs_dir)

    final_name = _unique_filename(refs_dir, safe_name)
    dest = os.path.join(refs_dir, final_name)
    _ensure_within_root(storage_root, dest)

    with open(dest, "wb") as handle:
        handle.write(content)

    return _format_entry(bundle_root, final_name, storage_key=storage_key)


def delete_reference_audio(
    bundle_root: str,
    *,
    filename: str,
    storage_key: Optional[str] = None,
    effective_config: Optional[dict] = None,
) -> None:
    safe_name = sanitize_reference_filename(filename)
    storage_root = reference_audio_storage_root(bundle_root, storage_key=storage_key)
    refs_dir = reference_audio_dir(bundle_root, storage_key=storage_key)
    target = os.path.join(refs_dir, safe_name)
    legacy = False
    _ensure_within_root(storage_root, target)
    if not os.path.isfile(target):
        legacy_refs_dir = legacy_reference_audio_dir(bundle_root)
        legacy_target = os.path.join(legacy_refs_dir, safe_name)
        _ensure_within_root(bundle_root, legacy_target)
        if not os.path.isfile(legacy_target):
            raise HTTPException(status_code=404, detail="Reference audio not found")
        target = legacy_target
        legacy = True

    entry = _format_entry(
        bundle_root,
        safe_name,
        storage_key=storage_key,
        legacy=legacy,
    )
    if effective_config:
        refs = find_config_references(effective_config, entry["path"])
        if refs:
            joined = ", ".join(refs)
            raise HTTPException(
                status_code=409,
                detail=f"Reference audio is still used in configuration: {joined}",
            )

    os.remove(target)
