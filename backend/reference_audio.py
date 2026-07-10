"""Manage reference audio files stored under an audio.cpp model bundle."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

REFERENCE_AUDIO_SUBDIR = "refs"
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


def reference_audio_dir(bundle_root: str) -> str:
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


def _format_entry(bundle_root: str, filename: str) -> Dict[str, Any]:
    path = os.path.join(reference_audio_dir(bundle_root), filename)
    stat = os.stat(path)
    rel = relative_reference_path(filename)
    return {
        "filename": filename,
        "path": rel,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def list_reference_audio(bundle_root: str) -> List[Dict[str, Any]]:
    refs_dir = reference_audio_dir(bundle_root)
    if not os.path.isdir(refs_dir):
        return []
    entries: List[Dict[str, Any]] = []
    for name in sorted(os.listdir(refs_dir)):
        path = os.path.join(refs_dir, name)
        if not os.path.isfile(path):
            continue
        _, ext = os.path.splitext(name)
        if ext.lower() not in ALLOWED_EXTENSIONS:
            continue
        entries.append(_format_entry(bundle_root, name))
    return entries


def _iter_config_string_values(prefix: str, value: Any):
    if isinstance(value, dict):
        for key, nested in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_config_string_values(child, nested)
    elif isinstance(value, str):
        yield prefix, value.replace("\\", "/")


def find_config_references(effective_config: dict, relative_path: str) -> List[str]:
    rel = relative_path.replace("\\", "/")
    matches: List[str] = []
    for key in (
        "voice_presets",
        "speech_defaults",
        "default_voice_preset",
        "transcription_defaults",
        "task_defaults",
    ):
        for location, value in _iter_config_string_values(key, effective_config.get(key)):
            if value == rel:
                matches.append(location)
    return matches


def save_reference_audio(
    bundle_root: str,
    *,
    filename: str,
    content: bytes,
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
    refs_dir = reference_audio_dir(bundle_root)
    os.makedirs(refs_dir, exist_ok=True)
    _ensure_within_root(bundle_root, refs_dir)

    final_name = _unique_filename(refs_dir, safe_name)
    dest = os.path.join(refs_dir, final_name)
    _ensure_within_root(bundle_root, dest)

    with open(dest, "wb") as handle:
        handle.write(content)

    return _format_entry(bundle_root, final_name)


def delete_reference_audio(
    bundle_root: str,
    *,
    filename: str,
    effective_config: Optional[dict] = None,
) -> None:
    safe_name = sanitize_reference_filename(filename)
    refs_dir = reference_audio_dir(bundle_root)
    target = os.path.join(refs_dir, safe_name)
    _ensure_within_root(bundle_root, target)
    if not os.path.isfile(target):
        raise HTTPException(status_code=404, detail="Reference audio not found")

    rel = relative_reference_path(safe_name)
    if effective_config:
        refs = find_config_references(effective_config, rel)
        if refs:
            joined = ", ".join(refs)
            raise HTTPException(
                status_code=409,
                detail=f"Reference audio is still used in configuration: {joined}",
            )

    os.remove(target)
