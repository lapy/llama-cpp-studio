"""Official audio.cpp demo voices used as a shared ``voice_dir`` library.

audio.cpp does not ship a separate Hugging Face preset pack. The embedded WebUI
demo clips live in the engine repository at ``webui/native/demo_voices`` and are
compiled into ``audiocpp_server``. A ``voice_dir`` of those WAVs plus
``prompt_text`` lets a speech request use ``"voice": "demo_1_man"`` as a clone
reference, or any of the files as a custom ``voice_ref``.

Pinned to audio.cpp commit ``c7f5743f037d588049c63aa75e9b4fdb279cfe01``.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

COMMUNITY_VOICE_REPOSITORY = "https://github.com/0xShug0/audio.cpp"
COMMUNITY_VOICE_COMMIT = "c7f5743f037d588049c63aa75e9b4fdb279cfe01"
COMMUNITY_VOICE_PATH = "webui/native/demo_voices"
PROMPT_TEXT_NAME = "prompt_text"
PROMPT_TEXT_SHA256 = "52693d0d896cbd0991777db15e0a242d68edec37c04c5f1913dd18b2955f5cc4"

_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "demo_1_man",
        "filename": "demo_1_man.wav",
        "label": "Demo 1 (man)",
        "reference_text": "okay,I'm Cemo and what you just heard wasn't a human voice.",
        "size_bytes": 905294,
        "sha256": "3e5321095813b09c6b108e9b12dfd988155020b35602ef074498aa0fa771fd39",
    },
    {
        "id": "demo_2_man",
        "filename": "demo_2_man.wav",
        "label": "Demo 2 (man)",
        "reference_text": "它的目标是模拟、延伸和扩展人的智能，让机器能够胜任通常需要人类智慧才能完成的任务",
        "size_bytes": 1462350,
        "sha256": "771a74e34ef7bb690bc95fa4122eaf518c35788828d8226e6527a6cc772e5e04",
    },
    {
        "id": "demo_3_woman",
        "filename": "demo_3_woman.wav",
        "label": "Demo 3 (woman)",
        "reference_text": (
            "以前我对这句话一知半解，现在好像有点懂了。因为你我开始留意很多以前不曾关心的事，"
            "开始对这个世界有了更多的好奇和善意。"
        ),
        "size_bytes": 1896526,
        "sha256": "b82c84722499444a79ae5831cd99cad085be20eceac156edec0f0dd263be3f1c",
    },
    {
        "id": "demo_4_woman",
        "filename": "demo_4_woman.wav",
        "label": "Demo 4 (woman)",
        "reference_text": "这都不会啊，麻将牌九掷色子，四色牌你总会一样吧",
        "size_bytes": 917582,
        "sha256": "357582df68a5cec668a8e77f3aaa8b2bf383d83d1f31a31a8f9b0972e47efd39",
    },
)


class CommunityVoiceError(Exception):
    """The pinned demo-voice pack could not be copied or downloaded."""


def _data_root() -> str:
    from backend import reference_audio

    return reference_audio._data_root()


def voice_catalog() -> List[Dict[str, Any]]:
    return [dict(item) for item in _CATALOG]


def prompt_text_bytes() -> bytes:
    lines = [f"{item['id']}|{item['reference_text']}" for item in _CATALOG]
    return ("\n".join(lines) + "\n").encode("utf-8")


def community_voice_dir() -> str:
    return os.path.join(_data_root(), "models", "audio-cpp", "community-voices")


def _source_payload() -> Dict[str, str]:
    return {
        "repository": COMMUNITY_VOICE_REPOSITORY,
        "commit": COMMUNITY_VOICE_COMMIT,
        "path": COMMUNITY_VOICE_PATH,
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _matches(data: bytes, *, size_bytes: int, sha256: str) -> bool:
    return len(data) == int(size_bytes) and _sha256(data) == sha256


def _is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _download_url(filename: str) -> str:
    return (
        "https://raw.githubusercontent.com/0xShug0/audio.cpp/"
        f"{COMMUNITY_VOICE_COMMIT}/{COMMUNITY_VOICE_PATH}/{filename}"
    )


def _local_demo_dir(source_root: str) -> str:
    root = str(source_root or "").strip()
    if not root:
        return ""
    return os.path.join(root, *COMMUNITY_VOICE_PATH.split("/"))


def _read_matching_local(directory: str, filename: str, *, size_bytes: int, sha256: str) -> Optional[bytes]:
    if not directory:
        return None
    path = os.path.join(directory, filename)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return None
    if _matches(data, size_bytes=size_bytes, sha256=sha256):
        return data
    return None


async def _fetch_bytes(url: str) -> bytes:
    import httpx

    from backend.http_client import get_http_client

    client = get_http_client()
    timeout = httpx.Timeout(connect=5.0, read=60.0, write=10.0, pool=5.0)
    last_error: Optional[BaseException] = None
    for _attempt in range(3):
        try:
            response = await client.get(url, timeout=timeout)
        except httpx.TransportError as exc:
            last_error = exc
            continue
        if response.status_code != 200:
            raise CommunityVoiceError(
                f"Download failed ({response.status_code}) for {url}"
            )
        return response.content
    raise CommunityVoiceError(f"Download failed for {url}: {last_error}")


def _write_verified(directory: str, filename: str, data: bytes) -> None:
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


def _file_status(path: str, *, size_bytes: int, sha256: str) -> bool:
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError:
        return False
    return _matches(data, size_bytes=size_bytes, sha256=sha256)


def library_status() -> Dict[str, Any]:
    """Return the pinned catalog and whether each file is installed locally."""
    root = community_voice_dir()
    prompt = prompt_text_bytes()
    items: List[Dict[str, Any]] = []
    ready_voices = 0
    for spec in _CATALOG:
        path = os.path.join(root, spec["filename"])
        installed = _file_status(
            path,
            size_bytes=int(spec["size_bytes"]),
            sha256=str(spec["sha256"]),
        )
        if installed:
            ready_voices += 1
        stat = os.stat(path) if installed else None
        items.append(
            {
                "id": spec["id"],
                "filename": spec["filename"],
                "label": spec["label"],
                "reference_text": spec["reference_text"],
                "size_bytes": int(spec["size_bytes"]),
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
    prompt_ready = _file_status(
        os.path.join(root, PROMPT_TEXT_NAME),
        size_bytes=len(prompt),
        sha256=PROMPT_TEXT_SHA256,
    )
    installed = prompt_ready and ready_voices == len(_CATALOG)
    return {
        "source": _source_payload(),
        "installed": installed,
        "voice_dir": os.path.realpath(root) if installed else None,
        "items": items,
    }


def installed_voice_dir() -> str:
    """Absolute ``voice_dir`` when the pinned pack is complete, else empty."""
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


async def _load_named(
    filename: str,
    *,
    size_bytes: int,
    sha256: str,
    source_dir: str,
    require_wav: bool,
) -> bytes:
    local = _read_matching_local(
        source_dir,
        filename,
        size_bytes=size_bytes,
        sha256=sha256,
    )
    data = local if local is not None else await _fetch_bytes(_download_url(filename))
    if require_wav and not _is_wav(data):
        raise CommunityVoiceError(f"{filename} is not a WAV file")
    if not _matches(data, size_bytes=size_bytes, sha256=sha256):
        raise CommunityVoiceError(
            f"{filename} does not match the pinned audio.cpp demo voice"
        )
    return data


async def install_community_voices(*, source_root: str = "") -> Dict[str, Any]:
    """Copy a matching local checkout pack, otherwise download the pinned commit."""
    source_dir = _local_demo_dir(source_root)
    prompt = prompt_text_bytes()
    if _sha256(prompt) != PROMPT_TEXT_SHA256:
        raise CommunityVoiceError("Pinned demo voice transcripts do not match prompt_text")

    payloads: List[tuple[str, bytes]] = []
    for spec in _CATALOG:
        data = await _load_named(
            str(spec["filename"]),
            size_bytes=int(spec["size_bytes"]),
            sha256=str(spec["sha256"]),
            source_dir=source_dir,
            require_wav=True,
        )
        payloads.append((str(spec["filename"]), data))
    prompt_data = await _load_named(
        PROMPT_TEXT_NAME,
        size_bytes=len(prompt),
        sha256=PROMPT_TEXT_SHA256,
        source_dir=source_dir,
        require_wav=False,
    )
    payloads.append((PROMPT_TEXT_NAME, prompt_data))

    root = community_voice_dir()
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    staging = tempfile.mkdtemp(prefix="community-voices-", dir=parent)
    try:
        for filename, data in payloads:
            _write_verified(staging, filename, data)
        os.makedirs(root, exist_ok=True)
        for filename, _data in payloads:
            os.replace(os.path.join(staging, filename), os.path.join(root, filename))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return library_status()
