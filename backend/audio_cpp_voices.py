"""Builtin TTS voice ids from audio.cpp package layouts and model_specs.

Sources, matching the engine rather than a generic folder walk:

1. ``GET /v1/audio/voices`` packaged listing: ``<model_root>/embeddings/*.safetensors``
   (PocketTTS; other families simply have no such directory).
2. ``model_specs/<family>.json`` voice resources:
   - Supertonic ``voice_style_<id>`` → ``voice_styles/<id>.json``
   - NeuTTS ``speaker_text_<id>`` → ``samples/<id>.txt``
   - ``options.request`` ``voice_id`` / ``speaker`` enum values when sidecars
     are missing (GGUF-embedded prompts).
3. Qwen3 CustomVoice ``config.json`` ``talker_config.spk_id`` keys
   (``src/models/qwen3_tts/assets.cpp``).

GGUF packages keep those sidecars next to the weights. When ``model_path`` is a
``.gguf`` file, the package root is its parent (``roots.model = "."``).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_VOICE_OPTION_KEYS = frozenset({"voice_id", "speaker"})
_SPEC_VOICE_KEY_RE = re.compile(
    r"^(?:voice_style_|speaker_text_)(.+)$"
)
# Same relative dirs as model_specs when the JSON is not available locally.
_FAMILY_VOICE_FILE_DIRS = {
    "supertonic": ("voice_styles", ".json"),
    "neutts": ("samples", ".txt"),
}


def merge_voice_ids(*groups: Any) -> List[str]:
    """Unique voice ids, sorted the same way as audio.cpp handle_voices."""
    seen = set()
    out: List[str] = []
    for group in groups:
        if group is None:
            continue
        if isinstance(group, (str, bytes)):
            items: Iterable[Any] = [group]
        elif isinstance(group, dict):
            items = group.values()
        else:
            try:
                items = list(group)
            except TypeError:
                items = [group]
        for item in items:
            if isinstance(item, dict):
                value = str(
                    item.get("value") or item.get("id") or item.get("name") or ""
                ).strip()
            else:
                value = str(item or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            out.append(value)
    out.sort()
    return out


def discover_packaged_voices(
    model_path: Optional[str],
    family: Optional[str] = None,
    source_path: Optional[str] = None,
) -> List[str]:
    """Return builtin voice/speaker ids shipped with this audio.cpp package."""
    package_root = _package_root(model_path)
    if not package_root:
        return []
    family_key = str(family or "").strip().lower()
    spec = _load_family_spec(source_path, family_key)
    search_roots = _package_search_roots(package_root, spec)

    found: List[str] = []
    for root in search_roots:
        found.extend(_embeddings_stems(root))

    spec_ids, spec_files = _spec_voice_resources(spec)
    spec_ids = list(spec_ids)
    spec_ids.extend(_spec_request_enum_ids(spec))
    existing = [
        voice_id
        for voice_id, rel in spec_files
        if os.path.isfile(os.path.join(package_root, rel))
    ]
    disk_ids = existing or _family_voice_dir_stems(package_root, family_key)
    if disk_ids:
        found.extend(disk_ids)
    else:
        found.extend(spec_ids)

    if family_key == "qwen3_tts":
        found.extend(_qwen3_custom_speakers(package_root))

    return merge_voice_ids(found)


def attach_packaged_voices(
    inspection: Optional[Dict[str, Any]],
    model_path: Optional[str],
    family: Optional[str] = None,
    source_path: Optional[str] = None,
) -> List[str]:
    """Store discovered builtin ids on ``inspection['packaged_voices']``."""
    family_name = str(family or "").strip()
    if not family_name and isinstance(inspection, dict):
        family_name = str(inspection.get("family") or "").strip()
    voices = discover_packaged_voices(
        model_path, family=family_name, source_path=source_path
    )
    if isinstance(inspection, dict) and voices:
        inspection["packaged_voices"] = voices
    return voices


def apply_packaged_voice_field_options(
    groups: Optional[Sequence[Dict[str, Any]]],
    voices: Optional[Sequence[str]],
) -> List[Dict[str, Any]]:
    """Copy packaged ids onto ``voice_id`` and ``speaker`` request fields."""
    source = [dict(group) for group in (groups or [])]
    ids = merge_voice_ids(voices)
    if not ids:
        return source
    out: List[Dict[str, Any]] = []
    for group in source:
        fields = []
        for field in group.get("fields") or []:
            spec = dict(field)
            if spec.get("key") in _VOICE_OPTION_KEYS:
                spec["options"] = [{"value": voice, "label": voice} for voice in ids]
                if not spec.get("placeholder"):
                    spec["placeholder"] = ids[0]
            fields.append(spec)
        copied = dict(group)
        copied["fields"] = fields
        out.append(copied)
    return out


def _package_root(model_path: Optional[str]) -> str:
    path = os.path.abspath(str(model_path or "").strip())
    if os.path.isdir(path):
        return path
    if os.path.isfile(path):
        return os.path.dirname(path)
    return ""


def _package_search_roots(package_root: str, spec: Optional[dict]) -> List[str]:
    roots = [package_root]
    if not isinstance(spec, dict):
        return roots
    seen = {os.path.realpath(package_root)}
    for source in spec.get("sources") or []:
        if not isinstance(source, dict):
            continue
        raw_roots = source.get("roots")
        if not isinstance(raw_roots, dict):
            continue
        for rel in raw_roots.values():
            if not isinstance(rel, str) or rel in {".", "$gguf"}:
                continue
            nested = os.path.join(package_root, rel)
            real = os.path.realpath(nested) if os.path.isdir(nested) else ""
            if real and real not in seen:
                seen.add(real)
                roots.append(nested)
    return roots


def _embeddings_stems(root: str) -> List[str]:
    embeddings_dir = os.path.join(root, "embeddings")
    return _stems_in_dir(embeddings_dir, ".safetensors")


def _family_voice_dir_stems(package_root: str, family: str) -> List[str]:
    layout = _FAMILY_VOICE_FILE_DIRS.get(family)
    if not layout:
        return []
    relative, ext = layout
    return _stems_in_dir(os.path.join(package_root, relative), ext)


def _stems_in_dir(directory: str, extension: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    try:
        names = os.listdir(directory)
    except OSError:
        return []
    stems: List[str] = []
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.isfile(path):
            continue
        stem, ext = os.path.splitext(name)
        if ext != extension or not stem:
            continue
        stems.append(stem)
    return stems


def _load_family_spec(source_path: Optional[str], family: str) -> Optional[dict]:
    if not source_path or not family:
        return None
    path = os.path.join(str(source_path), "model_specs", f"{family}.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _spec_voice_resources(spec: Optional[dict]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Return (ids, [(id, relative_path)]) from model_specs voice resource files."""
    if not isinstance(spec, dict):
        return [], []
    ids: List[str] = []
    files: List[Tuple[str, str]] = []
    seen = set()
    for source in spec.get("sources") or []:
        if not isinstance(source, dict):
            continue
        mapping = source.get("files")
        if not isinstance(mapping, dict):
            continue
        for raw_key, raw_path in mapping.items():
            match = _SPEC_VOICE_KEY_RE.match(str(raw_key or ""))
            if not match:
                continue
            voice_id = match.group(1).strip()
            if not voice_id or voice_id in seen:
                continue
            seen.add(voice_id)
            ids.append(voice_id)
            rel = _spec_relative_path(raw_path)
            if rel:
                files.append((voice_id, rel))
    return ids, files


def _spec_request_enum_ids(spec: Optional[dict]) -> List[str]:
    """Builtin ids declared on model_specs ``options.request`` voice_id/speaker enums."""
    if not isinstance(spec, dict):
        return []
    options = spec.get("options")
    if not isinstance(options, dict):
        return []
    ids: List[str] = []
    seen = set()
    for item in options.get("request") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if name not in _VOICE_OPTION_KEYS:
            continue
        for raw in item.get("values") or []:
            value = str(raw or "").strip()
            if not value or value in seen:
                continue
            seen.add(value)
            ids.append(value)
    return ids


def _spec_relative_path(raw: Any) -> str:
    if not isinstance(raw, str):
        return ""
    text = raw.strip()
    if not text or text.startswith("$"):
        return ""
    if ":" in text:
        _, _, rest = text.partition(":")
        text = rest.strip()
    return text.lstrip("/\\")


def _qwen3_custom_speakers(package_root: str) -> List[str]:
    path = os.path.join(package_root, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(payload, dict):
        return []
    talker = payload.get("talker_config")
    if not isinstance(talker, dict):
        return []
    spk_id = talker.get("spk_id")
    if not isinstance(spk_id, dict):
        return []
    return [str(name).strip() for name in spk_id if str(name).strip()]
