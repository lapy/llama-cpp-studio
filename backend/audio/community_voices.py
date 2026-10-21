"""Official audio.cpp demo voices used as a shared ``voice_dir`` library.

audio.cpp does not ship a separate Hugging Face preset pack. The embedded WebUI
demo clips live in the engine checkout at ``webui/native/demo_voices`` together
with a ``prompt_text`` transcript file (``<id>|<transcript>`` per line). A
``voice_dir`` of those WAVs plus ``prompt_text`` lets a speech request use
``"voice": "demo_1_man"`` as a clone reference, or any of the files as a custom
``voice_ref``.

The catalog is never hardcoded here: it is derived from the demo-voice directory
of the *active* engine checkout, and Studio keeps a verified copy under
``data/models/audio-cpp/community-voices`` so launches do not depend on the
checkout staying in place.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

COMMUNITY_VOICE_REPOSITORY = "https://github.com/0xShug0/audio.cpp"
COMMUNITY_VOICE_PATH = "webui/native/demo_voices"
PROMPT_TEXT_NAME = "prompt_text"


class CommunityVoiceError(Exception):
    """The active checkout's demo-voice pack could not be read or copied."""


def _data_root() -> str:
    from backend import reference_audio

    return reference_audio._data_root()


def community_voice_dir() -> str:
    return os.path.join(_data_root(), "models", "audio-cpp", "community-voices")


def _demo_voice_dir(source_root: str) -> str:
    root = str(source_root or "").strip()
    if not root:
        return ""
    return os.path.join(root, *COMMUNITY_VOICE_PATH.split("/"))


def _label_for(voice_id: str) -> str:
    return voice_id.replace("_", " ").strip().title() or voice_id


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ── Memoization ──────────────────────────────────────────────────────────────
# ``library_status``/``voice_catalog`` full-read + sha256 every WAV in the pack.
# They run once per audio model on every llama-swap rebuild and per reference
# listing, so a catalog of N models hashed ~5 MB × N per rebuild. Results are
# memoized under a cheap (name, mtime_ns, size) directory fingerprint, which
# invalidates exactly when a pack file changes.
_CATALOG_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}
_STATUS_CACHE: Dict[tuple, Dict[str, Any]] = {}
_CACHE_MAX = 16


def _directory_fingerprint(directory: str) -> Optional[tuple]:
    try:
        entries = sorted(os.listdir(directory))
    except OSError:
        return None
    parts: List[tuple] = []
    for name in entries:
        try:
            stat = os.stat(os.path.join(directory, name))
        except OSError:
            return None
        if not os.path.isfile(os.path.join(directory, name)):
            return None
        parts.append((name, stat.st_mtime_ns, stat.st_size))
    return tuple(parts)


def _cache_store(cache: Dict[tuple, Any], key: tuple, value: Any) -> None:
    if len(cache) >= _CACHE_MAX:
        cache.clear()
    cache[key] = value


def _clear_voice_caches() -> None:
    _CATALOG_CACHE.clear()
    _STATUS_CACHE.clear()


def _read_bytes(path: str) -> Optional[bytes]:
    try:
        with open(path, "rb") as handle:
            return handle.read()
    except OSError:
        return None


def _read_prompt_texts(directory: str) -> Dict[str, str]:
    raw = _read_bytes(os.path.join(directory, PROMPT_TEXT_NAME))
    texts: Dict[str, str] = {}
    for line in (raw or b"").decode("utf-8", errors="replace").splitlines():
        voice_id, separator, text = line.partition("|")
        if separator and voice_id.strip():
            texts[voice_id.strip()] = text.strip()
    return texts


def _catalog_from(directory: str) -> List[Dict[str, Any]]:
    """Derive the voice catalog from a demo-voice directory (WAVs + prompt_text)."""
    if not directory or not os.path.isdir(directory):
        return []
    fingerprint = _directory_fingerprint(directory)
    if fingerprint is not None:
        cached = _CATALOG_CACHE.get(fingerprint)
        if cached is not None:
            return [dict(row) for row in cached]
    texts = _read_prompt_texts(directory)
    catalog: List[Dict[str, Any]] = []
    for filename in sorted(os.listdir(directory)):
        if not filename.lower().endswith(".wav") or os.path.isdir(os.path.join(directory, filename)):
            continue
        data = _read_bytes(os.path.join(directory, filename)) or b""
        voice_id = os.path.splitext(filename)[0]
        catalog.append(
            {
                "id": voice_id,
                "filename": filename,
                "label": _label_for(voice_id),
                "reference_text": texts.get(voice_id, ""),
                "size_bytes": len(data),
                "sha256": _sha256(data),
            }
        )
    if fingerprint is not None and catalog:
        _cache_store(_CATALOG_CACHE, fingerprint, [dict(row) for row in catalog])
    return catalog


def voice_catalog(source_root: str = "") -> List[Dict[str, Any]]:
    """Catalog from the active checkout; falls back to the installed copy."""
    catalog = _catalog_from(_demo_voice_dir(source_root))
    if catalog:
        return catalog
    return _catalog_from(community_voice_dir())


def prompt_text_bytes(source_root: str = "") -> bytes:
    raw = _read_bytes(os.path.join(_demo_voice_dir(source_root) or "", PROMPT_TEXT_NAME))
    if raw is None:
        raw = _read_bytes(os.path.join(community_voice_dir(), PROMPT_TEXT_NAME))
    if raw is None:
        return b""
    return raw if raw.endswith(b"\n") else raw + b"\n"


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _write_staged(directory: str, filename: str, data: bytes) -> None:
    if os.path.basename(filename) != filename or filename in {".", ".."}:
        raise CommunityVoiceError(f"Invalid community voice filename: {filename}")
    os.makedirs(directory, exist_ok=True)
    dest = os.path.join(directory, filename)
    temporary = dest + ".partial"
    with open(temporary, "wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, dest)


def library_status(source_root: str = "") -> Dict[str, Any]:
    """Installed state of the active checkout's demo-voice pack.

    A voice counts as installed when the Studio copy byte-matches the checkout.
    With no checkout on disk, the installed copy is reported as-is (launches
    keep working after a source directory is pruned).
    """
    source_dir = _demo_voice_dir(source_root)
    checkout_dir_fp = _directory_fingerprint(source_dir) if source_dir else None
    root = community_voice_dir()
    root_fp = _directory_fingerprint(root)
    cache_key: Optional[tuple] = (
        (source_dir, checkout_dir_fp, root_fp)
        if checkout_dir_fp is not None or root_fp is not None
        else None
    )
    if cache_key is not None:
        cached = _STATUS_CACHE.get(cache_key)
        if cached is not None:
            return cached
    checkout_catalog = _catalog_from(source_dir)
    from_checkout = bool(checkout_catalog)
    catalog = checkout_catalog or _catalog_from(root)
    items: List[Dict[str, Any]] = []
    ready_voices = 0
    for spec in catalog:
        path = os.path.join(root, spec["filename"])
        data = _read_bytes(path)
        installed = data is not None and (
            not from_checkout
            or (_sha256(data) == spec["sha256"] and len(data) == spec["size_bytes"])
        )
        if installed:
            ready_voices += 1
            size_bytes = len(data or b"")
            stat = os.stat(path)
        else:
            size_bytes = int(spec["size_bytes"])
            stat = None
        items.append(
            {
                "id": spec["id"],
                "filename": spec["filename"],
                "label": spec["label"],
                "reference_text": spec["reference_text"],
                "size_bytes": size_bytes,
                "installed": installed,
                "path": os.path.realpath(path) if installed else "",
                "voice_id": spec["id"],
                "storage": "community",
                "modified_at": (
                    datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
                    if stat is not None
                    else None
                ),
            }
        )
    installed = bool(catalog) and ready_voices == len(catalog) and os.path.isfile(
        os.path.join(root, PROMPT_TEXT_NAME)
    )
    status = {
        "source": {
            "repository": COMMUNITY_VOICE_REPOSITORY,
            "path": COMMUNITY_VOICE_PATH,
            "checkout": os.path.realpath(source_dir) if from_checkout else "",
        },
        "installed": installed,
        "voice_dir": os.path.realpath(root) if installed else None,
        "items": items,
    }
    if cache_key is not None:
        _cache_store(_STATUS_CACHE, cache_key, status)
    return status


def installed_voice_dir() -> str:
    """Absolute ``voice_dir`` when the pack is complete, else empty."""
    voice_dir = library_status().get("voice_dir")
    return str(voice_dir or "")


def list_reference_entries() -> List[Dict[str, Any]]:
    """Installed demo clips in the same shape as uploaded reference audio."""
    status = library_status()
    if not status.get("installed"):
        return []
    entries: List[Dict[str, Any]] = []
    for item in status["items"]:
        if not item.get("installed") or not item.get("path"):
            continue
        filename = str(item["filename"])
        entries.append(
            {
                "filename": filename,
                "path": item["path"],
                "relative_path": f"community/{filename}",
                "display_path": f"community/{filename}",
                "size_bytes": item["size_bytes"],
                "modified_at": item.get("modified_at"),
                "storage": "community",
                "voice_id": item["id"],
                "reference_text": item["reference_text"],
                "label": item["label"],
            }
        )
    return entries


def remove_community_voices(*, source_root: str = "") -> Dict[str, Any]:
    """Delete the installed community-voice library from the Studio data root.

    Idempotent: a missing directory is a successful no-op. Returns the same
    ``library_status`` payload as the GET route so the UI can refresh in one
    round-trip. Voice presets that reference community clips need a proxy
    apply + model restart after removal, exactly like install.
    """
    root = community_voice_dir()
    # Refuse to rmtree anything but the Studio-managed library directory.
    if os.path.basename(os.path.normpath(root)) != "community-voices":
        raise CommunityVoiceError(f"Refusing to remove unexpected voice library path: {root}")
    if os.path.isdir(root):
        shutil.rmtree(root)
    _clear_voice_caches()
    return library_status(source_root=source_root)


async def install_community_voices(*, source_root: str = "") -> Dict[str, Any]:
    """Copy the active checkout's demo-voice pack into the Studio data root."""
    source_dir = _demo_voice_dir(source_root)
    catalog = _catalog_from(source_dir)
    if not catalog:
        raise CommunityVoiceError(
            f"No demo voices found in the active audio.cpp checkout "
            f"({COMMUNITY_VOICE_PATH}). Update or reinstall audio.cpp."
        )
    prompt = _read_bytes(os.path.join(source_dir, PROMPT_TEXT_NAME))
    if prompt is None:
        raise CommunityVoiceError(
            f"{COMMUNITY_VOICE_PATH}/{PROMPT_TEXT_NAME} is missing from the checkout"
        )

    payloads: List[tuple[str, bytes]] = []
    for spec in catalog:
        data = _read_bytes(os.path.join(source_dir, spec["filename"]))
        if data is None or not _is_wav(data):
            raise CommunityVoiceError(f"{spec['filename']} is missing or not a WAV file")
        payloads.append((spec["filename"], data))
    payloads.append((PROMPT_TEXT_NAME, prompt))

    root = community_voice_dir()
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="community-voices-", dir=parent)
    try:
        for filename, data in payloads:
            _write_staged(staging, filename, data)
        os.makedirs(root, exist_ok=True)
        for filename, _data in payloads:
            os.replace(os.path.join(staging, filename), os.path.join(root, filename))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return library_status(source_root=source_root)
