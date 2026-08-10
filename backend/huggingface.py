from huggingface_hub import HfApi, hf_hub_download
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
import os
import threading
import time
import re
from collections import deque
from datetime import datetime
from backend.logging_config import get_logger
from backend.model_files import iter_model_files, shard_sort_key

try:
    # Optional import available in newer huggingface_hub versions
    from huggingface_hub import get_safetensors_metadata as hf_get_safetensors_metadata
except ImportError:  # pragma: no cover - fallback if function missing
    hf_get_safetensors_metadata = None

logger = get_logger(__name__)

# Initialize HF API - will be updated with token if provided
hf_api = HfApi()

# Check for environment variable on module initialization
_env_token = os.getenv("HUGGINGFACE_API_KEY")
if _env_token:
    hf_api = HfApi(token=_env_token)
    logger.debug("HuggingFace API key loaded from environment variable")

# Simple cache for search results
_search_cache: Dict[str, Tuple[List[Dict], float]] = {}
_cache_timeout = 300  # 5 minutes

# Cache for safetensors metadata (per repo)
_safetensors_metadata_cache: Dict[str, Tuple[Dict, float]] = {}
_safetensors_metadata_ttl = 600  # 10 minutes


def get_accurate_file_sizes(repo_id: str, paths: List[str]) -> Dict[str, Optional[int]]:
    """Fetch accurate file sizes from HuggingFace API via get_paths_info."""
    info = get_remote_file_info(repo_id, paths)
    return {path: meta.get("size") for path, meta in info.items()}


def _path_basename(path: str) -> str:
    return str(path or "").replace("\\", "/").rstrip("/").split("/")[-1]


def _paths_info_to_meta(paths_info: Any) -> Dict[str, Dict[str, Any]]:
    """Convert Hugging Face ``get_paths_info`` rows into path -> metadata."""
    result: Dict[str, Dict[str, Any]] = {}
    for pi in paths_info or []:
        path = getattr(pi, "path", None) or getattr(pi, "rfilename", None)
        if not path:
            continue
        etag = getattr(pi, "etag", None) or getattr(pi, "lfs", None)
        if isinstance(etag, dict):
            etag = etag.get("sha256") or etag.get("oid") or etag.get("etag")
        sha256 = None
        lfs = getattr(pi, "lfs", None)
        if isinstance(lfs, dict):
            sha256 = lfs.get("sha256") or lfs.get("oid")
            if not etag:
                etag = sha256
        result[str(path)] = {
            "size": getattr(pi, "size", None),
            "etag": str(etag) if etag else None,
            "sha256": str(sha256) if sha256 else None,
        }
    return result


def _resolve_repo_path_alias(
    requested: str, repo_files: List[str]
) -> Optional[str]:
    """Map a stored path to an actual repo-relative path when folders differ.

    Common case: ledger/companion field kept ``mtp-….gguf`` while Hugging Face
    hosts it under ``MTP/mtp-….gguf``.
    """
    if not requested or not repo_files:
        return None
    req = str(requested).replace("\\", "/")
    if req in repo_files:
        return req
    lower_map = {f.replace("\\", "/").lower(): f.replace("\\", "/") for f in repo_files}
    if req.lower() in lower_map:
        return lower_map[req.lower()]

    req_base = _path_basename(req)
    if not req_base:
        return None
    candidates = [
        f.replace("\\", "/")
        for f in repo_files
        if _path_basename(f) == req_base
    ]
    if not candidates:
        req_base_l = req_base.lower()
        candidates = [
            f.replace("\\", "/")
            for f in repo_files
            if _path_basename(f).lower() == req_base_l
        ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    def _top_dir(path: str) -> str:
        parts = path.split("/")
        return parts[0].lower() if len(parts) > 1 else ""

    preferred_dirs: List[str] = []
    if is_mtp_filename(req) or req_base.lower().startswith(("mtp-", "mtp_")):
        preferred_dirs.append("mtp")
    if is_dflash_filename(req) or "dflash" in req_base.lower():
        preferred_dirs.append("dflash")
    if is_mmproj_filename(req) or "mmproj" in req_base.lower():
        preferred_dirs.append("mmproj")

    for directory in preferred_dirs:
        narrowed = [c for c in candidates if _top_dir(c) == directory]
        if len(narrowed) == 1:
            return narrowed[0]
        if narrowed:
            candidates = narrowed

    # Prefer a nested path when the stored name was basename-only.
    if "/" not in req:
        nested = [c for c in candidates if "/" in c]
        if len(nested) == 1:
            return nested[0]
        if nested:
            candidates = nested
    return candidates[0] if len(candidates) == 1 else None


def get_remote_file_info(
    repo_id: str, paths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Fetch remote size/etag (and sha when available) via get_paths_info.

    Keys are the *requested* paths. When a file only exists under a different
    repo-relative path (e.g. ``MTP/<basename>``), the entry includes
    ``resolved_path`` with the actual remote path.
    """
    if not paths or not repo_id:
        return {}

    requested: List[str] = []
    seen: set = set()
    for path in paths:
        if not path:
            continue
        try:
            safe = _sanitize_filename(path)
        except ValueError:
            continue
        if safe in seen:
            continue
        seen.add(safe)
        requested.append(safe)
    if not requested:
        return {}

    try:
        paths_info = hf_api.get_paths_info(repo_id=repo_id, paths=requested)
    except Exception as e:
        logger.debug("get_paths_info failed for %s: %s", repo_id, e)
        paths_info = []

    by_remote = _paths_info_to_meta(paths_info)
    lower_remote = {p.lower(): p for p in by_remote}

    result: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for req in requested:
        if req in by_remote:
            result[req] = dict(by_remote[req])
            continue
        alt = lower_remote.get(req.lower())
        if alt:
            meta = dict(by_remote[alt])
            if alt != req:
                meta["resolved_path"] = alt
            result[req] = meta
            continue
        missing.append(req)

    if not missing:
        return result

    try:
        repo_files = [
            f.replace("\\", "/")
            for f in hf_api.list_repo_files(repo_id=repo_id)
            if isinstance(f, str)
        ]
    except Exception as e:
        logger.debug("list_repo_files failed for %s: %s", repo_id, e)
        return result

    alias_targets: List[str] = []
    req_to_alias: Dict[str, str] = {}
    for req in missing:
        alias = _resolve_repo_path_alias(req, repo_files)
        if not alias or alias == req:
            continue
        req_to_alias[req] = alias
        if alias not in by_remote:
            alias_targets.append(alias)

    if alias_targets:
        try:
            more_info = hf_api.get_paths_info(repo_id=repo_id, paths=alias_targets)
        except Exception as e:
            logger.debug("get_paths_info alias lookup failed for %s: %s", repo_id, e)
            more_info = []
        by_remote.update(_paths_info_to_meta(more_info))

    for req, alias in req_to_alias.items():
        remote_meta = by_remote.get(alias)
        if not remote_meta:
            continue
        meta = dict(remote_meta)
        meta["resolved_path"] = alias
        result[req] = meta
        logger.debug(
            "Resolved HF path alias for %s: %s -> %s", repo_id, req, alias
        )
    return result


def detect_hf_file_changes(
    huggingface_id: str,
    filenames: List[str],
    local_entries: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Compare remote HF file metadata against local model-ledger/cache entries.

    ``local_entries`` maps filename -> {file_size, etag, sha256, file_path}.
    Returns ``changed``, ``unchanged``, and ``removed_remote`` lists of
    ``{filename, size, etag, sha256, reason}``.
    """
    unique_names: List[str] = []
    seen: set = set()
    for name in filenames:
        if not name:
            continue
        try:
            safe = _sanitize_filename(name)
        except ValueError:
            continue
        if safe in seen:
            continue
        seen.add(safe)
        unique_names.append(safe)

    remote = get_remote_file_info(huggingface_id, unique_names)
    changed: List[Dict[str, Any]] = []
    unchanged: List[Dict[str, Any]] = []
    removed_remote: List[Dict[str, Any]] = []
    seen_remote_paths: set = set()

    for filename in unique_names:
        local = local_entries.get(filename) or {}
        # local_entries may still be keyed by the pre-sanitize spelling.
        if not local and filename not in local_entries:
            local = local_entries.get(_path_basename(filename)) or {}
        remote_meta = remote.get(filename)
        if not remote_meta:
            removed_remote.append(
                {
                    "filename": filename,
                    "size": local.get("file_size"),
                    "etag": local.get("etag"),
                    "sha256": local.get("sha256"),
                    "reason": "missing_remote",
                }
            )
            continue

        resolved_path = remote_meta.get("resolved_path") or filename
        # Deduplicate when both basename and folder-qualified paths are checked.
        remote_key = str(resolved_path).replace("\\", "/").lower()
        if remote_key in seen_remote_paths:
            continue
        seen_remote_paths.add(remote_key)

        remote_size = remote_meta.get("size")
        remote_etag = remote_meta.get("etag")
        remote_sha = remote_meta.get("sha256")
        local_size = local.get("file_size")
        local_etag = local.get("etag")
        local_sha = local.get("sha256")
        file_path = local.get("file_path")

        cached_path = None
        if file_path and os.path.lexists(file_path):
            cached_path = file_path
        else:
            cached_path = resolve_cached_model_path(huggingface_id, resolved_path)
            if not cached_path and resolved_path != filename:
                cached_path = resolve_cached_model_path(huggingface_id, filename)

        missing_local = not cached_path or not os.path.lexists(cached_path)
        reason = None
        if missing_local:
            reason = "missing_local"
        elif local_sha and remote_sha and str(local_sha) != str(remote_sha):
            reason = "sha_mismatch"
        elif local_etag and remote_etag and str(local_etag) != str(remote_etag):
            reason = "etag_mismatch"
        elif (
            remote_size is not None
            and local_size is not None
            and int(remote_size) != int(local_size)
        ):
            reason = "size_mismatch"
        elif local_size is None and remote_size is not None and not missing_local:
            # Have a file on disk but no recorded size — compare disk size.
            try:
                disk_size = os.path.getsize(os.path.realpath(cached_path))
                if int(remote_size) != int(disk_size):
                    reason = "size_mismatch"
            except OSError:
                reason = "missing_local"

        entry = {
            # Prefer the real repo path so downloads/heals use MTP/… etc.
            "filename": resolved_path,
            "size": remote_size,
            "etag": remote_etag,
            "sha256": remote_sha,
            "reason": reason,
        }
        if resolved_path != filename:
            entry["previous_filename"] = filename
            entry["resolved_path"] = resolved_path
        if reason:
            changed.append(entry)
        else:
            unchanged.append(entry)

    return {
        "changed": changed,
        "unchanged": unchanged,
        "removed_remote": removed_remote,
    }


def list_repo_companion_files(huggingface_id: str) -> Dict[str, List[Dict[str, Any]]]:
    """List mmproj / MTP / DFlash files available in an HF repo."""
    empty = {"mmproj_files": [], "mtp_files": [], "dflash_files": []}
    if not huggingface_id:
        return empty
    try:
        files = list(hf_api.list_repo_files(repo_id=huggingface_id))
    except Exception as exc:
        logger.debug("list_repo_files failed for %s: %s", huggingface_id, exc)
        return empty

    mmproj_files: List[Dict[str, Any]] = []
    mtp_files: List[Dict[str, Any]] = []
    dflash_files: List[Dict[str, Any]] = []
    for filename in files:
        if not isinstance(filename, str):
            continue
        if is_mmproj_filename(filename):
            mmproj_files.append({"filename": filename, "size": 0})
        elif is_mtp_filename(filename):
            mtp_files.append(
                {
                    "filename": filename,
                    "size": 0,
                    "label": mtp_option_label(filename),
                }
            )
        elif is_dflash_filename(filename):
            dflash_files.append(
                {
                    "filename": filename,
                    "size": 0,
                    "label": dflash_option_label(filename),
                }
            )

    all_paths = [
        f["filename"] for f in (mmproj_files + mtp_files + dflash_files)
    ]
    sizes = get_accurate_file_sizes(huggingface_id, all_paths)
    for group in (mmproj_files, mtp_files, dflash_files):
        for entry in group:
            size = sizes.get(entry["filename"])
            if size is not None:
                entry["size"] = size
    return {
        "mmproj_files": mmproj_files,
        "mtp_files": mtp_files,
        "dflash_files": dflash_files,
    }


def _list_remote_gguf_weight_files(
    huggingface_id: str, quantization: Optional[str]
) -> List[str]:
    """Return remote GGUF weight filenames matching a library quantization."""
    if not huggingface_id or not quantization:
        return []
    try:
        files = list(hf_api.list_repo_files(repo_id=huggingface_id))
    except Exception as exc:
        logger.debug("list_repo_files failed for %s: %s", huggingface_id, exc)
        return []

    quant_lower = str(quantization).lower()
    matched: List[str] = []
    for filename in files:
        if not isinstance(filename, str) or not re.search(r"\.gguf(\.|$)", filename):
            continue
        if is_auxiliary_gguf_filename(filename):
            continue
        if _extract_quantization(filename).lower() == quant_lower:
            matched.append(filename)
    return matched


def _list_remote_safetensors_files(huggingface_id: str) -> List[str]:
    if not huggingface_id:
        return []
    try:
        files = list(hf_api.list_repo_files(repo_id=huggingface_id))
    except Exception as exc:
        logger.debug("list_repo_files failed for %s: %s", huggingface_id, exc)
        return []
    return [f for f in files if isinstance(f, str) and f.endswith(".safetensors")]


def collect_model_refresh_plan(model: Dict[str, Any]) -> Dict[str, Any]:
    """Build the refresh check plan for a library model (weights + companions)."""
    huggingface_id = model.get("huggingface_id")
    if not huggingface_id:
        raise ValueError("Model has no huggingface_id")

    fmt = (model.get("format") or model.get("model_format") or "gguf").lower()
    filenames: List[str] = []
    local_entries: Dict[str, Dict[str, Any]] = {}
    companion_filenames: List[str] = []

    if fmt == "gguf":
        quantization = model.get("quantization")
        for entry in iter_model_files(model):
            fn = entry.get("filename")
            if not fn:
                continue
            filenames.append(fn)
            local_entries[fn] = {
                "file_size": entry.get("size"),
                "etag": entry.get("etag"),
                "sha256": entry.get("sha256"),
            }
        for fn in _list_remote_gguf_weight_files(huggingface_id, quantization):
            if fn not in local_entries:
                filenames.append(fn)
                local_entries.setdefault(fn, {})

        for field in ("mmproj_filename", "mtp_filename", "dflash_filename"):
            companion = model.get(field)
            if not companion:
                continue
            companion_filenames.append(companion)
            if companion not in filenames:
                filenames.append(companion)
            if companion not in local_entries:
                cached = resolve_cached_model_path(huggingface_id, companion)
                size = None
                if cached and os.path.exists(cached):
                    try:
                        size = os.path.getsize(os.path.realpath(cached))
                    except OSError:
                        size = None
                local_entries[companion] = {
                    "file_size": size,
                    "etag": None,
                    "sha256": None,
                }
    elif fmt == "safetensors":
        for entry in iter_model_files(model):
            fn = entry.get("filename")
            if not fn:
                continue
            filenames.append(fn)
            local_entries[fn] = {
                "file_size": entry.get("size"),
                "etag": entry.get("etag"),
                "sha256": entry.get("sha256"),
            }
        for fn in _list_remote_safetensors_files(huggingface_id):
            if fn not in local_entries:
                filenames.append(fn)
                local_entries.setdefault(fn, {})
    else:
        raise ValueError(f"Refresh is not supported for format '{fmt}'")

    detection = detect_hf_file_changes(huggingface_id, filenames, local_entries)
    path_corrections: List[Dict[str, str]] = []
    seen_corrections: set = set()
    for group_name in ("changed", "unchanged"):
        for entry in detection.get(group_name) or []:
            previous = entry.get("previous_filename")
            resolved = entry.get("resolved_path") or entry.get("filename")
            if not previous or not resolved or previous == resolved:
                continue
            key = (previous, resolved)
            if key in seen_corrections:
                continue
            seen_corrections.add(key)
            path_corrections.append({"from": previous, "to": resolved})
    return {
        "huggingface_id": huggingface_id,
        "format": fmt,
        "filenames": filenames,
        "companion_filenames": companion_filenames,
        "path_corrections": path_corrections,
        **detection,
    }


def get_mmproj_f16_filename(repo_id: str) -> Optional[str]:
    """
    If the repo contains vision projector (mmproj) GGUF files, return the F16 one to download.
    Prefers mmproj-F16.gguf, then any *mmproj*F16*.gguf, then first mmproj*.gguf.
    Returns None if no mmproj files or on API error.
    """
    try:
        files = list(hf_api.list_repo_files(repo_id=repo_id))
    except Exception as e:
        logger.debug(f"list_repo_files failed for {repo_id}: {e}")
        return None
    mmproj = [f for f in files if is_mmproj_filename(f)]
    if not mmproj:
        return None
    # Prefer exact mmproj-F16.gguf, then any filename containing F16, then first mmproj
    for name in mmproj:
        if name == "mmproj-F16.gguf" or name.lower().endswith("/mmproj-f16.gguf"):
            return name
    for name in mmproj:
        if "f16" in name.lower():
            return name
    return mmproj[0]


def is_mmproj_filename(filename: Optional[str]) -> bool:
    """True for vision projector GGUF companions (never main-model quants)."""
    name = str(filename or "").replace("\\", "/").strip()
    lower = name.lower()
    return bool(lower) and "mmproj" in lower and lower.endswith(".gguf")


def is_mtp_filename(filename: Optional[str]) -> bool:
    """True for Multi-Token Prediction *draft companion* GGUFs.

    Matches Unsloth-style companions only:
    - files under an ``MTP/`` directory
    - basenames prefixed with ``mtp-`` / ``mtp_``
    - ``…-<quant>-MTP.gguf`` (MTP after a quantization token)

    Does **not** treat model names that merely contain ``MTP`` as companions
    (e.g. ``Qwen3.6-27B-MTP-Q8_0.gguf`` from ``…/Qwen3.6-27B-MTP-GGUF``).
    """
    name = str(filename or "").replace("\\", "/").strip()
    lower = name.lower()
    if not lower.endswith(".gguf"):
        return False
    parts = lower.split("/")
    basename = parts[-1]
    # Dedicated companion directory (e.g. MTP/mtp-….gguf, MTP/….Q8_0-MTP.gguf)
    if len(parts) > 1 and parts[0] == "mtp":
        return True
    # Explicit companion prefix at the start of the file name
    if basename.startswith("mtp-") or basename.startswith("mtp_"):
        return True
    # Companion suffix only when MTP follows a quantization token, not when MTP
    # is part of the model family name (…-27B-MTP-Q8_0.gguf).
    if re.search(
        r"-(?:q\d+(?:_[a-z0-9]+)?|iq\d+(?:_[a-z0-9]+)?|ud-[a-z0-9_]+|bf16|f16|f32)-mtp\.gguf$",
        lower,
    ):
        return True
    return False


def is_dflash_filename(filename: Optional[str]) -> bool:
    """True for DFlash speculative-decoding *draft companion* GGUFs.

    Matches Poolside-style companions such as ``laguna-s-2.1-DFlash-BF16.gguf``,
    files under a ``DFlash/`` directory, or basenames prefixed with ``dflash-``.
    """
    name = str(filename or "").replace("\\", "/").strip()
    lower = name.lower()
    if not lower.endswith(".gguf"):
        return False
    parts = lower.split("/")
    basename = parts[-1]
    if len(parts) > 1 and parts[0] == "dflash":
        return True
    if basename.startswith("dflash-") or basename.startswith("dflash_"):
        return True
    # Embedded token (e.g. laguna-s-2.1-DFlash-BF16.gguf)
    if "dflash" in basename:
        return True
    return False


def is_auxiliary_gguf_filename(filename: Optional[str]) -> bool:
    """mmproj / MTP / DFlash companions that are not main-model quantization shards."""
    return (
        is_mmproj_filename(filename)
        or is_mtp_filename(filename)
        or is_dflash_filename(filename)
    )


def mtp_option_label(filename: str) -> str:
    """Human label for an MTP draft file (quant if present, else Default)."""
    quant = _extract_quantization(filename)
    if quant and quant != "unknown":
        return quant
    return "Default"


def dflash_option_label(filename: str) -> str:
    """Human label for a DFlash draft file (precision/quant if present, else Default)."""
    quant = _extract_quantization(filename)
    if quant and quant != "unknown":
        return quant
    return "Default"


def _download_repo_json(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    try:
        path = hf_hub_download(repo_id, filename, local_dir_use_symlinks=False)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug(f"Unable to download {filename} for {repo_id}: {exc}")
        return None


def _hf_int_metric(obj: Any, attr: str, default: int = 0) -> int:
    """Coerce HF hub metrics. getattr(obj, attr, 0) returns None when the attribute exists but is null."""
    v = getattr(obj, attr, None)
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _hf_datetime_iso(model: Any, *attr_names: str) -> Optional[str]:
    """ModelInfo uses created_at / last_modified (snake_case); older code expected camelCase."""
    for name in attr_names:
        v = getattr(model, name, None)
        if v is None:
            continue
        if hasattr(v, "isoformat"):
            try:
                return v.isoformat()
            except Exception:
                continue
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _hf_gated_flag(raw: Any) -> bool:
    """HF gated may be False, or 'manual' / 'auto'."""
    if raw is False or raw is None:
        return False
    if raw is True:
        return True
    if isinstance(raw, str):
        return raw.strip().lower() in ("manual", "auto", "true", "1")
    return bool(raw)


def _model_card_to_dict(model: Any) -> Dict[str, Any]:
    """
    HuggingFace returns ModelCardData (has to_dict()), not a plain dict.
    Attribute name is card_data (snake) on current huggingface_hub.
    """
    raw = getattr(model, "card_data", None) or getattr(model, "cardData", None)
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if hasattr(raw, "to_dict"):
        try:
            return dict(raw.to_dict())
        except Exception:
            return {}
    return {}


def _normalize_card_scalar(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list) and val:
        return str(val[0]).strip()
    return str(val).strip()


def _language_hints_from_tags(tags: Optional[List[str]]) -> List[str]:
    """Infer language codes from repo tags (en, zh, multilingual, …)."""
    if not tags:
        return []
    lowered = [t.lower() for t in tags if isinstance(t, str)]
    if "multilingual" in lowered or "multi-lingual" in lowered:
        return ["multilingual"]
    known = frozenset(
        {
            "en",
            "zh",
            "ja",
            "ko",
            "de",
            "fr",
            "es",
            "it",
            "pt",
            "ru",
            "ar",
            "hi",
            "vi",
            "th",
            "id",
            "tr",
            "pl",
            "nl",
        }
    )
    out: List[str] = []
    seen: set = set()
    for t in tags:
        if not isinstance(t, str):
            continue
        tl = t.lower().strip()
        if tl in known and tl not in seen:
            seen.add(tl)
            out.append(tl)
        if len(out) >= 8:
            break
    return out


def _get_tokenizer_config(repo_id: str) -> Optional[Dict[str, Any]]:
    return _download_repo_json(repo_id, "tokenizer_config.json")


def get_tokenizer_config(repo_id: str) -> Optional[Dict[str, Any]]:
    """Public API for tokenizer_config.json fetch (used by services)."""
    return _get_tokenizer_config(repo_id)


MODEL_FORMATS = ("gguf", "safetensors")


def _hf_repo_folder_name(huggingface_id: str) -> str:
    """Return the HF cache folder name for a model repo (e.g. models--Org--Repo)."""
    return "models--" + huggingface_id.replace("/", "--")


def _hf_hub_cache_root() -> str:
    """Return the active Hugging Face hub cache directory."""
    # Prefer process env so deletes follow the same overrides as downloads
    # (Docker sets HUGGINGFACE_HUB_CACHE=/app/data/hf-cache/hub).
    env_cache = os.getenv("HUGGINGFACE_HUB_CACHE") or os.getenv("HF_HUB_CACHE")
    if env_cache:
        return env_cache
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        return str(HF_HUB_CACHE)
    except Exception:
        hf_home = os.getenv("HF_HOME") or os.path.join(
            os.path.expanduser("~"), ".cache", "huggingface"
        )
        return os.path.join(hf_home, "hub")


def _hf_repo_cache_dir(huggingface_id: str) -> str:
    return os.path.join(_hf_hub_cache_root(), _hf_repo_folder_name(huggingface_id))


def _repo_relative_filename_from_cache_path(file_path: str) -> Optional[str]:
    """Extract repo-relative filename from an HF hub snapshot path."""
    if not file_path:
        return None
    normalized = os.path.normpath(file_path).replace("\\", "/")
    marker = "/snapshots/"
    idx = normalized.find(marker)
    if idx < 0:
        return None
    rest = normalized[idx + len(marker) :]
    parts = rest.split("/", 1)
    if len(parts) < 2 or not parts[1]:
        return None
    try:
        return _sanitize_filename(parts[1])
    except ValueError:
        return None


def _find_cached_snapshot_path(huggingface_id: str, filename: str) -> Optional[str]:
    """Locate a file under hub cache snapshots/ without calling the hub API."""
    if not huggingface_id or not filename:
        return None
    try:
        safe_name = _sanitize_filename(filename)
    except ValueError:
        return None
    snapshots_dir = os.path.join(_hf_repo_cache_dir(huggingface_id), "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None
    for rev in os.listdir(snapshots_dir):
        candidate = os.path.join(snapshots_dir, rev, safe_name)
        if os.path.lexists(candidate):
            return candidate
    return None


def _path_is_under_hub_blobs(path: str) -> bool:
    if not path:
        return False
    try:
        hub_root = os.path.realpath(_hf_hub_cache_root())
        real = os.path.realpath(path)
        if os.path.commonpath([hub_root, real]) != hub_root:
            return False
    except ValueError:
        return False
    return f"{os.sep}blobs{os.sep}" in real or real.endswith(f"{os.sep}blobs")


def _delete_cache_path_and_blob(path: str) -> bool:
    """Delete a hub snapshot path and its underlying blob (symlink or hardlink)."""
    if not path or not os.path.lexists(path):
        return False
    try:
        if os.path.islink(path):
            blob_path = os.path.realpath(path)
            os.unlink(path)
            if blob_path and os.path.isfile(blob_path):
                os.remove(blob_path)
            return True

        real_path = os.path.realpath(path)
        os.remove(path)
        # Hardlinks / non-symlink layouts: also drop the blobs/ object if it remains.
        if (
            real_path
            and os.path.isfile(real_path)
            and os.path.abspath(real_path) != os.path.abspath(path)
            and _path_is_under_hub_blobs(real_path)
        ):
            os.remove(real_path)
        return True
    except OSError as exc:
        logger.warning(f"Could not remove cached path {path}: {exc}")
        return False


def resolve_cached_model_path(huggingface_id: str, filename: str) -> Optional[str]:
    """Return the local path for a cached HF model file without triggering a download.

    Returns None if the file is not in the HF cache.
    """
    try:
        return hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            local_files_only=True,
        )
    except Exception:
        return _find_cached_snapshot_path(huggingface_id, filename)


def delete_cached_model_file(
    huggingface_id: str,
    filename: str,
    file_path: Optional[str] = None,
) -> bool:
    """Delete a specific model file from the HuggingFace cache.

    Removes both the snapshot symlink/hardlink and the underlying content blob.
    ``file_path`` from stored model metadata is used when hub resolution fails.
    Returns True if at least one cache path was deleted.
    """
    try:
        safe_filename = _sanitize_filename(filename) if filename else ""
    except ValueError:
        safe_filename = ""

    candidates: List[str] = []
    if safe_filename:
        try:
            cached_path = hf_hub_download(
                repo_id=huggingface_id,
                filename=safe_filename,
                local_files_only=True,
            )
            if cached_path:
                candidates.append(cached_path)
        except Exception:
            pass
        scanned = _find_cached_snapshot_path(huggingface_id, safe_filename)
        if scanned:
            candidates.append(scanned)

    if file_path:
        candidates.append(file_path)
        rel_from_path = _repo_relative_filename_from_cache_path(file_path)
        if rel_from_path and rel_from_path != safe_filename:
            scanned = _find_cached_snapshot_path(huggingface_id, rel_from_path)
            if scanned:
                candidates.append(scanned)
            try:
                cached_path = hf_hub_download(
                    repo_id=huggingface_id,
                    filename=rel_from_path,
                    local_files_only=True,
                )
                if cached_path:
                    candidates.append(cached_path)
            except Exception:
                pass

    deleted = False
    seen: set = set()
    for path in candidates:
        if not path:
            continue
        abs_path = os.path.abspath(path)
        if abs_path in seen:
            continue
        seen.add(abs_path)
        if _delete_cache_path_and_blob(abs_path):
            deleted = True

    if deleted:
        logger.info(f"Deleted cached model file: {huggingface_id}/{filename}")
    else:
        logger.warning(
            f"delete_cached_model_file: {huggingface_id}/{filename} not found in HF cache"
        )
    return deleted


def purge_hf_repo_cache(huggingface_id: str) -> bool:
    """Remove the entire HF hub cache directory for a repo (blobs/snapshots/refs)."""
    if not huggingface_id:
        return False
    repo_dir = _hf_repo_cache_dir(huggingface_id)
    if not os.path.isdir(repo_dir):
        return False
    from backend.utils.fs_ops import robust_rmtree

    try:
        robust_rmtree(repo_dir)
        logger.info(f"Purged HF hub cache for {huggingface_id}: {repo_dir}")
        return True
    except Exception as exc:
        logger.warning(f"Failed to purge HF hub cache for {huggingface_id}: {exc}")
        return False


def resolve_gguf_model_path(model: Dict[str, Any]) -> Optional[str]:
    """Resolve the first local GGUF weight/shard recorded on a model row."""
    huggingface_id = model.get("huggingface_id")
    if not huggingface_id:
        return None
    entries = sorted(
        iter_model_files(model, roles={"weight", "shard"}), key=shard_sort_key
    )
    for entry in entries:
        filename = entry.get("filename")
        if not filename:
            continue
        path = resolve_cached_model_path(huggingface_id, filename)
        if path and os.path.exists(path):
            return path
    return None


def clear_search_cache():
    """Clear the search cache to force fresh results"""
    global _search_cache
    _search_cache = {}
    global _safetensors_metadata_cache
    _safetensors_metadata_cache = {}


# Rate limiting
_last_request_time = 0
_min_request_interval = 0.5  # Reduced to 0.5 seconds since we're making fewer calls


def _sanitize_filename(filename: str) -> str:
    """Ensure filename is a safe relative path without traversal."""
    if not filename or filename.strip() == "":
        raise ValueError("filename is required")
    normalized = os.path.normpath(filename).replace("\\", "/")
    if (
        normalized.startswith("../")
        or normalized.startswith("..\\")
        or normalized.startswith("/")
    ):
        raise ValueError("invalid filename")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        normalized = "/".join(part for part in parts if part not in ("", ".", ".."))
    if ".." in normalized.split("/"):
        raise ValueError("invalid filename")
    return normalized or os.path.basename(filename)


# Compiled regex patterns for better performance
# Order matters: more specific/longer patterns first, including optional
# variant markers like "iQ3_K_S" before plain "Q3_K_S".
QUANTIZATION_PATTERNS = [
    # Mixed-precision and exotic formats (MXFP4, FP8, etc.)
    re.compile(r"MXFP\d+_MOE"),  # MXFP4_MOE style (mixed-precision MoE)
    re.compile(r"MXFP\d+"),  # MXFP4, MXFP8 style
    re.compile(r"FP\d+"),  # FP8, FP16, FP32 style
    re.compile(r"BF16"),  # BF16 (Brain Float 16)
    re.compile(r"F16"),  # F16 (alias for FP16)
    re.compile(r"F32"),  # F32 (alias for FP32)
    # Standard integer quantization patterns
    re.compile(r"iQ\d+_K_[A-Z]+"),  # iQ3_K_S style
    re.compile(r"iQ\d+_\d+"),  # iQ4_0 style
    re.compile(r"iQ\d+_K"),  # iQ6_K style
    re.compile(r"iQ\d+"),  # iQ3 style (fallback)
    re.compile(r"IQ\d+_[A-Z]+"),  # IQ1_S, IQ2_M, etc.
    re.compile(r"Q\d+_K_[A-Z]+"),  # Q4_K_M, Q5_K_S, etc.
    re.compile(r"Q\d+_\d+"),  # Q4_0, Q5_1, etc.
    re.compile(r"Q\d+_K"),  # Q2_K, Q6_K, etc.
    re.compile(r"Q\d+"),  # Q3, Q4, etc. (fallback)
]

# Model size extraction pattern


def set_huggingface_token(token: str):
    """Set HuggingFace API token for authenticated requests"""
    global hf_api
    if token:
        hf_api = HfApi(token=token)
        logger.info("HuggingFace API token set - using authenticated requests")
    else:
        hf_api = HfApi()
        logger.info("HuggingFace API token cleared - using unauthenticated requests")


def get_huggingface_token() -> Optional[str]:
    """Get current HuggingFace API token"""
    return getattr(hf_api, "token", None)


async def _rate_limit():
    """Async rate limiting to avoid hitting HuggingFace limits"""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    if time_since_last < _min_request_interval:
        sleep_time = _min_request_interval - time_since_last
        logger.debug("HF rate limit delay: %.2f seconds", sleep_time)
        await asyncio.sleep(sleep_time)
    _last_request_time = time.time()


async def search_models(
    query: str, limit: int = 20, model_format: str = "gguf"
) -> List[Dict]:
    """Search HuggingFace for GGUF or safetensors models."""
    try:
        model_format = (model_format or "gguf").lower()
        if model_format not in MODEL_FORMATS:
            raise ValueError(
                f"Unsupported model format '{model_format}'. Must be one of {MODEL_FORMATS}"
            )

        # Check cache first
        cache_key = f"{model_format}:{query.lower()}_{limit}"
        current_time = time.time()

        if cache_key in _search_cache:
            cached_data, cache_time = _search_cache[cache_key]
            if current_time - cache_time < _cache_timeout:
                logger.debug("Returning cached HF results for %r", query)
                return cached_data[:limit]  # Return only requested limit

        logger.debug(
            "Searching HF models: query=%r limit=%s format=%s",
            query,
            limit,
            model_format,
        )
        # Always attempt API search; authentication will be used automatically if a token is set
        return await _search_with_api(query, limit, model_format)

    except Exception as e:
        logger.error(f"Search error: {e}")
        raise Exception(f"Failed to search models: {e}")


async def _search_with_api(query: str, limit: int, model_format: str) -> List[Dict]:
    """Search using HuggingFace Hub API (authenticated if token is configured)."""
    try:
        # Apply rate limiting
        await _rate_limit()

        # Use the configured API client so auth tokens are honored. `full=True`
        # keeps likes populated; the partial `expand=[...]` query shape returns
        # `likes=None` on current Hugging Face responses.
        filter_value = "gguf" if model_format == "gguf" else "safetensors"

        models_generator = hf_api.list_models(
            search=query,
            limit=min(limit * 2, 50),  # Get more models to filter from
            sort="downloads",
            filter=filter_value,
            full=True,
        )

        # Convert generator to list
        models = list(models_generator)
        logger.debug("HF search returned %s candidate models", len(models))

        # Process models in parallel for better performance
        results = await _process_models_parallel(models, limit, model_format)

        # Cache the results
        cache_key = f"{model_format}:{query.lower()}_{limit}"
        _search_cache[cache_key] = (results, time.time())

        logger.debug("Returning %s processed HF results", len(results))
        return results

    except Exception as e:
        logger.warning("HF API search failed: %s", e)
        # Return empty results if API fails
        return []


async def _process_models_parallel(
    models: List, limit: int, model_format: str, max_concurrent: int = 5
) -> List[Dict]:
    """Process models in parallel with semaphore for concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_model(model):
        async with semaphore:
            return await _process_single_model(model, model_format)

    # Create tasks for all models
    tasks = [process_model(model) for model in models[: limit * 2]]

    # Execute in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and None results
    valid_results = []
    for result in results:
        if isinstance(result, Exception):
            logger.debug("HF model processing failed: %s", result)
            continue
        if result is not None:
            valid_results.append(result)

    if model_format == "gguf":
        def _gguf_sort_key(item: Dict[str, Any]):
            quantizations = item.get("quantizations") or {}
            size_candidates = [
                q.get("total_size") or 0
                for q in quantizations.values()
                if isinstance(q, dict)
            ]
            positive_sizes = [size for size in size_candidates if size > 0]
            min_size = min(positive_sizes) if positive_sizes else float("inf")
            return (min_size, -(item.get("downloads") or 0), item.get("id") or "")

        valid_results.sort(key=_gguf_sort_key)

    return valid_results[:limit]


async def _process_single_model(model, model_format: str) -> Optional[Dict]:
    """Process a single model and extract all metadata"""
    try:
        logger.debug("Processing HF model: %s", model.id)

        quantizations: Dict[str, Dict] = {}
        mmproj_files: List[Dict[str, Any]] = []
        mtp_files: List[Dict[str, Any]] = []
        dflash_files: List[Dict[str, Any]] = []
        safetensors_files: List[Dict] = []
        repo_files: List[Dict[str, Any]] = []

        if hasattr(model, "siblings") and model.siblings:
            if model_format == "gguf":
                # Group GGUF files by logical quantization, handling multi-part shards.
                gguf_siblings = [
                    s
                    for s in model.siblings
                    if isinstance(getattr(s, "rfilename", None), str)
                    and re.search(r"\.gguf(\.|$)", s.rfilename)
                ]
                logger.debug(
                    "HF model %s has %s GGUF files", model.id, len(gguf_siblings)
                )
                if not gguf_siblings:
                    return None

                for sibling in gguf_siblings:
                    filename = sibling.rfilename
                    size_bytes = getattr(sibling, "size", 0) or 0
                    if is_mmproj_filename(filename):
                        mmproj_files.append(
                            {
                                "filename": filename,
                                "size": size_bytes,
                            }
                        )
                        continue
                    if is_mtp_filename(filename):
                        mtp_files.append(
                            {
                                "filename": filename,
                                "size": size_bytes,
                                "label": mtp_option_label(filename),
                            }
                        )
                        continue
                    if is_dflash_filename(filename):
                        dflash_files.append(
                            {
                                "filename": filename,
                                "size": size_bytes,
                                "label": dflash_option_label(filename),
                            }
                        )
                        continue
                    # Normalize filename by stripping shard suffix patterns like:
                    #   -00001-of-00002.gguf (TheBloke-style)
                    #   .gguf.part1of2 (Hugging Face-style multi-part)
                    base_for_quant = re.sub(r"-\d{5}-of-\d{5}(?=\.gguf$)", "", filename)
                    base_for_quant = re.sub(
                        r"\.gguf\.part\d+of\d+$", ".gguf", base_for_quant
                    )
                    quantization = _extract_quantization(base_for_quant)
                    if quantization == "unknown":
                        continue

                    # Detect optional variant prefix immediately before the quantization (e.g. "i1-" in "i1-IQ3_M")
                    variant_prefix = ""
                    try:
                        prefix_match = re.search(
                            rf"(i\d+)-{re.escape(quantization)}", base_for_quant
                        )
                        if prefix_match:
                            variant_prefix = prefix_match.group(1)
                    except Exception:
                        variant_prefix = ""

                    # Use full variant-aware key so that different variants (e.g. "i1-Q4_K_M"
                    # vs "Q4_K_M") are treated as distinct quantizations everywhere.
                    quant_key = (
                        f"{variant_prefix}-{quantization}"
                        if variant_prefix
                        else quantization
                    )

                    entry = quantizations.setdefault(
                        quant_key,
                        {
                            # Store both the raw quantization and any variant prefix for clients
                            # that want to render them separately.
                            "quantization": quantization,
                            "files": [],
                            "total_size": 0,
                            "size_mb": 0.0,
                            "variant_prefix": variant_prefix or "",
                        },
                    )
                    if variant_prefix and not entry.get("variant_prefix"):
                        entry["variant_prefix"] = variant_prefix
                    entry["files"].append(
                        {
                            "filename": filename,
                            "size": size_bytes,
                        }
                    )
                    entry["total_size"] += size_bytes
                    entry["size_mb"] = (
                        round(entry["total_size"] / (1024 * 1024), 2)
                        if entry["total_size"]
                        else 0.0
                    )

                # Search should stay to a single HF API call. Accurate file sizes are lazy-loaded on expand.
                # If no downloadable GGUF entries were detected after grouping, skip this model.
                if not quantizations and not mmproj_files and not mtp_files and not dflash_files:
                    return None
            else:
                safetensors_files = []
                for sibling in model.siblings:
                    filename = sibling.rfilename
                    size_bytes = getattr(sibling, "size", 0) or 0
                    repo_files.append(
                        {
                            "filename": filename,
                            "is_safetensors": filename.endswith(".safetensors"),
                        }
                    )
                    if not filename.endswith(".safetensors"):
                        continue
                    safetensors_files.append({"filename": filename})

                logger.debug(
                    "HF model %s has %s safetensors files",
                    model.id,
                    len(safetensors_files),
                )
                if not safetensors_files:
                    return None
        else:
            return None

        # Extract rich metadata from model and cardData
        metadata = _extract_model_metadata(model)

        result = {
            "id": model.id,
            "name": getattr(
                model, "modelId", model.id
            ),  # Use modelId if available, fallback to id
            "author": getattr(model, "author", ""),
            "downloads": _hf_int_metric(model, "downloads", 0),
            "likes": _hf_int_metric(model, "likes", 0),
            "tags": model.tags or [],
            # Canonical single field for "what type is this HF result"
            "format": model_format,
            "quantizations": quantizations if model_format == "gguf" else {},
            "mmproj_files": mmproj_files if model_format == "gguf" else [],
            "mtp_files": mtp_files if model_format == "gguf" else [],
            "dflash_files": dflash_files if model_format == "gguf" else [],
            "safetensors_files": (
                safetensors_files if model_format == "safetensors" else []
            ),
            "repo_files": repo_files if model_format == "safetensors" else [],
            **metadata,  # Include all extracted metadata
        }

        logger.debug("Added HF model %s to results", model.id)
        return result

    except Exception as e:
        logger.debug("Error processing HF model %s: %s", model.id, e)
        return None


def _extract_model_metadata(model) -> Dict:
    """Extract rich metadata from ModelInfo and model card (ModelCardData or dict)."""
    pipeline = getattr(model, "pipeline_tag", None) or ""
    library = getattr(model, "library_name", None) or ""

    metadata = {
        "description": "",
        "license": "",
        "pipeline_tag": pipeline,
        "library_name": library,
        "language": [],
        "base_model": "",
        "architecture": "",
        "parameters": "",
        "context_length": None,
        "gated": _hf_gated_flag(getattr(model, "gated", False)),
        "private": bool(getattr(model, "private", False)),
        "readme_url": f"https://huggingface.co/{model.id}",
        "created_at": _hf_datetime_iso(model, "created_at", "createdAt"),
        "updated_at": _hf_datetime_iso(model, "last_modified", "lastModified"),
        "safetensors": {},
    }

    card = _model_card_to_dict(model)
    if card:
        lic = card.get("license")
        if lic is not None and lic != "":
            metadata["license"] = _normalize_card_scalar(lic)

        bm = card.get("base_model")
        if isinstance(bm, list) and bm:
            metadata["base_model"] = str(bm[0]).strip()
        elif isinstance(bm, str) and bm.strip():
            metadata["base_model"] = bm.strip()

        language_data = card.get("language")
        if isinstance(language_data, list) and language_data:
            metadata["language"] = [str(x) for x in language_data if x is not None]
        elif isinstance(language_data, str) and language_data.strip():
            metadata["language"] = [language_data.strip()]

        if not metadata.get("pipeline_tag") and card.get("pipeline_tag"):
            metadata["pipeline_tag"] = str(card["pipeline_tag"]).strip()

        model_index = card.get("model-index") or card.get("model_index") or []
        if isinstance(model_index, list):
            for item in model_index:
                if not isinstance(item, dict):
                    continue
                if not metadata["architecture"] and item.get("name"):
                    metadata["architecture"] = str(item["name"])
                if not metadata["parameters"] and item.get("params") is not None:
                    metadata["parameters"] = str(item["params"])
                if metadata["context_length"] is None and item.get("context_length") is not None:
                    metadata["context_length"] = item["context_length"]

    # Merge repo tags + card tags for language inference
    all_tags: List[str] = list(model.tags or [])
    if card:
        ct = card.get("tags")
        if isinstance(ct, list):
            all_tags.extend(str(t) for t in ct if t is not None)
    if not metadata["language"]:
        metadata["language"] = _language_hints_from_tags(all_tags)

    # Parameter size hint from repo id when card has no model-index
    if not metadata["parameters"]:
        model_id = getattr(model, "modelId", model.id)
        size_match = re.search(r"(\d+(?:\.\d+)?)[Bb]", str(model_id))
        if size_match:
            metadata["parameters"] = f"{size_match.group(1)}B"

    if hasattr(model, "siblings") and model.siblings:
        metadata["safetensors"] = _extract_safetensors_metadata(model.siblings)

    return metadata


def _extract_quantization(filename: str) -> str:
    """Extract quantization from filename using compiled regex patterns"""
    for pattern in QUANTIZATION_PATTERNS:
        match = pattern.search(filename)
        if match:
            return match.group()
    return "unknown"


def extract_quantization(filename: str) -> str:
    """Public API for quantization parsing (used by routes/services)."""
    return _extract_quantization(filename)


def _extract_safetensors_metadata(siblings) -> Dict:
    """Extract safetensors metadata from siblings if available"""
    safetensors_info = {
        "has_safetensors": False,
        "safetensors_files": [],
        "total_tensors": 0,
        "total_size": 0,
    }

    if not siblings:
        return safetensors_info

    safetensors_files = []
    total_size = 0

    for sibling in siblings:
        if sibling.rfilename.endswith(".safetensors"):
            safetensors_files.append({"filename": sibling.rfilename})
            total_size += sibling.size or 0

    if safetensors_files:
        safetensors_info.update(
            {
                "has_safetensors": True,
                "safetensors_files": safetensors_files,
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
            }
        )

    return safetensors_info


async def get_safetensors_metadata_summary(model_id: str) -> Dict:
    """Fetch safetensors metadata on demand with caching and aggregation."""
    if not model_id:
        raise ValueError("model_id is required")

    cache_key = model_id
    current_time = time.time()
    cached_entry = _safetensors_metadata_cache.get(cache_key)
    if cached_entry:
        cached_data, cached_time = cached_entry
        if current_time - cached_time < _safetensors_metadata_ttl:
            return cached_data

    if not hf_get_safetensors_metadata and not hasattr(
        hf_api, "get_safetensors_metadata"
    ):
        raise RuntimeError(
            "Safetensors metadata is not supported by the installed huggingface_hub version"
        )

    await _rate_limit()
    loop = asyncio.get_running_loop()

    def _fetch_metadata():
        if hasattr(hf_api, "get_safetensors_metadata"):
            return hf_api.get_safetensors_metadata(repo_id=model_id)
        return hf_get_safetensors_metadata(model_id)

    try:
        metadata = await loop.run_in_executor(None, _fetch_metadata)
    except Exception as err:
        error_msg = str(err)
        # Handle hf_transfer missing error gracefully
        if (
            "hf_transfer" in error_msg.lower()
            or "HF_HUB_ENABLE_HF_TRANSFER" in error_msg
        ):
            logger.debug(
                "hf_transfer unavailable for %s; using standard download: %s",
                model_id,
                err,
            )
            # Temporarily disable HF_TRANSFER and retry
            original_env = os.environ.get("HF_HUB_ENABLE_HF_TRANSFER")
            try:
                os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
                metadata = await loop.run_in_executor(None, _fetch_metadata)
                # Restore original env if it existed
                if original_env:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = original_env
            except Exception as retry_err:
                # Restore original env if it existed
                if original_env:
                    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = original_env
                logger.error(
                    f"Failed to fetch safetensors metadata for {model_id} even after disabling hf_transfer: {retry_err}"
                )
                raise RuntimeError(
                    f"Safetensors metadata is not available: {retry_err}"
                )
        else:
            logger.error(f"Failed to fetch safetensors metadata for {model_id}: {err}")
            raise

    files_summary = []
    dtype_totals: Dict[str, int] = {}
    total_tensors = 0

    # Handle both dict and object responses from HuggingFace API
    def _get_attr_or_key(obj, key, default=None):
        """Get attribute or key from object or dict"""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    files_metadata = _get_attr_or_key(metadata, "files_metadata", {}) or {}
    if not files_metadata:
        # If files_metadata is empty, try to extract from the metadata structure
        # Some versions return metadata directly as a dict
        if isinstance(metadata, dict):
            files_metadata = metadata
        else:
            logger.warning(
                f"No files_metadata found in safetensors metadata for {model_id}"
            )
            return {
                "repo_id": model_id,
                "total_files": 0,
                "total_tensors": 0,
                "dtype_totals": {},
                "files": [],
                "cached_at": datetime.utcnow().isoformat(),
                "error": "No safetensors files found",
            }

    for filename, file_meta in files_metadata.items():
        if not isinstance(file_meta, (dict, object)):
            continue

        tensors = _get_attr_or_key(file_meta, "tensors", {}) or {}
        parameter_count = _get_attr_or_key(file_meta, "parameter_count", {}) or {}

        tensor_details = []
        for tensor_name, tensor_info in tensors.items():
            if not tensor_info:
                continue
            tensor_details.append(
                {
                    "name": tensor_name,
                    "dtype": _get_attr_or_key(tensor_info, "dtype", "unknown"),
                    "shape": _get_attr_or_key(tensor_info, "shape", []),
                }
            )

        dtype_counts = {}
        if isinstance(parameter_count, dict):
            for dtype, count in parameter_count.items():
                dtype_counts[dtype] = count
                dtype_totals[dtype] = dtype_totals.get(dtype, 0) + count
        elif hasattr(parameter_count, "items"):
            # Handle object with items() method
            for dtype, count in parameter_count.items():
                dtype_counts[dtype] = count
                dtype_totals[dtype] = dtype_totals.get(dtype, 0) + count

        total_tensors += len(tensor_details)
        files_summary.append(
            {
                "filename": filename,
                "tensor_count": len(tensor_details),
                "dtype_counts": dtype_counts,
                "tensors": tensor_details,
            }
        )

    summary = {
        "repo_id": model_id,
        "total_files": len(files_summary),
        "total_tensors": total_tensors,
        "dtype_totals": dtype_totals,
        "files": files_summary,
        "cached_at": datetime.utcnow().isoformat(),
    }

    _safetensors_metadata_cache[cache_key] = (summary, current_time)
    return summary


def _get_model_details_blocking(model_id: str) -> Dict:
    """Blocking Hugging Face API + config.json fetch (run via asyncio.to_thread)."""
    try:
        # Get model info with expanded data
        model_info = hf_api.model_info(model_id, expand=["cardData", "siblings"])

        # Extract basic metadata
        metadata = _extract_model_metadata(model_info)

        # Add additional details
        details = {
            "id": model_info.id,
            "name": getattr(
                model_info, "modelId", model_info.id
            ),  # Use modelId if available, fallback to id
            "author": getattr(model_info, "author", ""),
            "downloads": _hf_int_metric(model_info, "downloads", 0),
            "likes": _hf_int_metric(model_info, "likes", 0),
            "tags": model_info.tags or [],
            **metadata,
        }

        # Try to get config.json for architecture details
        try:
            config_files = [
                s for s in model_info.siblings if s.rfilename == "config.json"
            ]
            if config_files:
                # Download and parse config.json
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename="config.json",
                    local_dir="data/hf-cache",
                    local_dir_use_symlinks=False,
                )

                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)

                # Store full config for downstream consumers (e.g. safetensors metadata extraction)
                details["config"] = config or {}

                # Clean up temp file
                os.remove(config_path)

        except Exception as e:
            logger.warning(f"Could not fetch config.json for {model_id}: {e}")
            details["config"] = {}

        return details

    except Exception as e:
        logger.error(f"Error getting model details for {model_id}: {e}")
        raise Exception(f"Failed to get model details: {e}")


async def get_model_details(model_id: str) -> Dict:
    """Get detailed model information including config and README."""
    return await asyncio.to_thread(_get_model_details_blocking, model_id)


async def download_model(
    huggingface_id: str,
    filename: str,
    model_format: str = "gguf",
    *,
    force_download: bool = False,
) -> tuple[str, int]:
    """Download model from HuggingFace to the native HF cache."""
    try:
        filename = _sanitize_filename(filename)
        if force_download:
            delete_cached_model_file(huggingface_id, filename)

        file_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            force_download=force_download,
        )

        # Use realpath so getsize works even when file_path is a symlink
        real_path = os.path.realpath(file_path)
        file_size = os.path.getsize(real_path if os.path.exists(real_path) else file_path)

        return file_path, file_size

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


def _download_speed_mbps(
    samples: deque,
    *,
    now: float,
    bytes_downloaded: int,
    start_time: float,
    last_speed: float = 0.0,
    window_s: float = 3.0,
    min_window_s: float = 1.0,
) -> float:
    """Stable download rate from a rolling byte/time window.

    Instantaneous poll-interval rates spike to hundreds of MB/s when HF/tqdm
    or ``.incomplete`` blob sizes jump in large chunks. Prefer a multi-second
    window, and ignore discontinuous catch-up jumps for the rate (keep the bar
    bytes accurate; only the MB/s display is smoothed).
    """
    bytes_downloaded = max(0, int(bytes_downloaded))
    if samples:
        t_prev, b_prev = samples[-1]
        dt = max(now - t_prev, 1e-6)
        jump_mbps = max(0, bytes_downloaded - b_prev) / dt / (1024 * 1024)
        # Adaptive ceiling: real transfers rarely leap several× in one poll.
        plausible = max(64.0, float(last_speed) * 3.0) if last_speed > 1 else 128.0
        if jump_mbps > plausible:
            samples.clear()
            samples.append((now, bytes_downloaded))
            return max(0.0, float(last_speed))

    samples.append((now, bytes_downloaded))
    while len(samples) > 1 and (now - samples[0][0]) > window_s:
        samples.popleft()

    if len(samples) < 2:
        elapsed_total = max(now - start_time, 1e-6)
        return max(0.0, bytes_downloaded / elapsed_total / (1024 * 1024))

    t0, b0 = samples[0]
    t1, b1 = samples[-1]
    dt = t1 - t0
    if dt < min_window_s:
        elapsed_total = max(now - start_time, 1e-6)
        return max(0.0, bytes_downloaded / elapsed_total / (1024 * 1024))
    return max(0.0, (b1 - b0) / dt / (1024 * 1024))


class _HubDownloadProgressState:
    """Thread-safe byte progress shared between hf_hub_download's tqdm and the SSE loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.n = 0
        self.total = 0

    def set(self, *, n: Optional[int] = None, total: Optional[int] = None) -> None:
        with self._lock:
            if n is not None:
                self.n = max(0, int(n))
            if total is not None and int(total) > 0:
                self.total = int(total)

    def snapshot(self) -> Tuple[int, int]:
        with self._lock:
            return self.n, self.total


def _make_hub_download_tqdm(state: _HubDownloadProgressState):
    """Build a tqdm subclass that mirrors download bytes into ``state``."""
    from tqdm.auto import tqdm as base_tqdm

    class ReportingTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs)
            # Quiet console output. Track bytes ourselves because disable=True
            # freezes tqdm's internal ``n`` counter.
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self._reported_n = int(getattr(self, "n", 0) or 0)
            self._reported_total = int(getattr(self, "total", 0) or 0)
            state.set(n=self._reported_n, total=self._reported_total)

        def update(self, n=1):
            self._reported_n += int(n or 0)
            total = getattr(self, "total", None)
            if total:
                self._reported_total = int(total)
            state.set(n=self._reported_n, total=self._reported_total)
            return None

        def reset(self, total=None):
            if total is not None:
                self._reported_total = int(total)
                try:
                    self.total = total
                except Exception:
                    pass
            self._reported_n = 0
            state.set(n=0, total=self._reported_total)
            return None

    return ReportingTqdm


def _incomplete_blob_bytes(blobs_dir: str) -> int:
    """Best-effort fallback when tqdm progress is unavailable."""
    if not os.path.isdir(blobs_dir):
        return 0
    incomplete_bytes = 0
    try:
        for fname in os.listdir(blobs_dir):
            if not fname.endswith(".incomplete"):
                continue
            try:
                incomplete_bytes = max(
                    incomplete_bytes,
                    os.path.getsize(os.path.join(blobs_dir, fname)),
                )
            except OSError:
                pass
    except OSError:
        return 0
    return incomplete_bytes


async def download_model_with_progress(
    huggingface_id: str,
    filename: str,
    progress_manager,
    task_id: str,
    total_bytes: int = 0,
    model_format: str = "gguf",
    huggingface_id_for_progress: str = None,
    revision: str = None,
    token: str = None,
    force_download: bool = False,
):
    """Download model to the HF native cache with SSE progress updates.

    Primary progress comes from Hugging Face's ``tqdm_class`` hook (works for both
    HTTP and Xet transfers). Polling ``*.incomplete`` blobs remains a fallback for
    older cache write paths that do not drive tqdm.
    """
    from huggingface_hub.constants import HF_HUB_CACHE

    from backend.task_cancel_registry import TaskCancelledError, is_task_cancel_requested

    filename = _sanitize_filename(filename)
    progress_hf_id = huggingface_id_for_progress or huggingface_id
    if force_download:
        delete_cached_model_file(huggingface_id, filename)

    logger.info(
        f"Starting HF-cache download: {huggingface_id}/{filename} task={task_id}"
        f"{' force' if force_download else ''}"
    )

    # Resolve total size if not provided
    if total_bytes == 0:
        try:
            file_info = HfApi().repo_file_info(repo_id=huggingface_id, filename=filename)
            total_bytes = file_info.size or 0
            logger.debug(
                "HF file size for %s/%s: %s",
                huggingface_id,
                filename,
                total_bytes,
            )
        except Exception as e:
            logger.debug(
                "Could not get HF file size for %s/%s: %s",
                huggingface_id,
                filename,
                e,
            )

    await progress_manager.send_download_progress(
        task_id=task_id,
        progress=0,
        message=f"Starting download of {filename}",
        bytes_downloaded=0,
        total_bytes=total_bytes,
        speed_mbps=0,
        eta_seconds=0,
        filename=filename,
        model_format=model_format,
        huggingface_id=progress_hf_id,
    )

    repo_folder = _hf_repo_folder_name(huggingface_id)
    blobs_dir = os.path.join(HF_HUB_CACHE, repo_folder, "blobs")

    download_result: dict = {"file_path": None, "error": None, "done": False}
    progress_state = _HubDownloadProgressState()
    if total_bytes > 0:
        progress_state.set(total=total_bytes)
    tqdm_class = _make_hub_download_tqdm(progress_state)

    def _do_download():
        try:
            download_result["file_path"] = hf_hub_download(
                repo_id=huggingface_id,
                filename=filename,
                revision=revision,
                token=token,
                tqdm_class=tqdm_class,
                force_download=force_download,
            )
        except Exception as exc:
            download_result["error"] = exc
        finally:
            download_result["done"] = True

    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()

    start_time = time.time()
    last_emitted_bytes = 0
    last_emit_time = start_time
    speed_samples: deque = deque()
    speed_mbps = 0.0
    bytes_high_water = 0
    # Poll often for cancel checks; emit more sparsely so the UI stays smooth.
    poll_interval_s = 0.25
    emit_interval_s = 0.5
    min_emit_bytes = 256 * 1024

    while not download_result["done"]:
        if is_task_cancel_requested(task_id):
            raise TaskCancelledError("Download cancelled by user")
        await asyncio.sleep(poll_interval_s)

        tqdm_n, tqdm_total = progress_state.snapshot()
        incomplete_bytes = _incomplete_blob_bytes(blobs_dir)
        bytes_high_water = max(bytes_high_water, tqdm_n, incomplete_bytes, 0)
        bytes_downloaded = bytes_high_water
        known_total = tqdm_total or total_bytes or 0
        if known_total and total_bytes <= 0:
            total_bytes = known_total

        now = time.time()
        speed_mbps = _download_speed_mbps(
            speed_samples,
            now=now,
            bytes_downloaded=bytes_downloaded,
            start_time=start_time,
            last_speed=speed_mbps,
        )

        # Skip unchanged snapshots so we do not flood SSE.
        if bytes_downloaded == last_emitted_bytes:
            continue

        emit_delta = bytes_downloaded - last_emitted_bytes
        time_since_emit = now - last_emit_time
        significant = emit_delta >= max(
            min_emit_bytes,
            int(known_total * 0.002) if known_total > 0 else min_emit_bytes,
        )
        if time_since_emit < emit_interval_s and not significant:
            continue

        elapsed_total = max(now - start_time, 1e-6)
        progress = (
            min(99, int(bytes_downloaded / known_total * 100))
            if known_total > 0
            else (1 if bytes_downloaded > 0 else 0)
        )
        eta = 0
        if known_total > bytes_downloaded and bytes_downloaded > 0:
            rate = bytes_downloaded / elapsed_total
            if rate > 0:
                eta = int((known_total - bytes_downloaded) / rate)

        size_hint = ""
        if known_total > 0:
            size_hint = (
                f" ({bytes_downloaded / (1024 * 1024):.1f}/"
                f"{known_total / (1024 * 1024):.1f} MB"
            )
            if speed_mbps > 0.01:
                size_hint += f", {speed_mbps:.1f} MB/s"
            size_hint += ")"
        await progress_manager.send_download_progress(
            task_id=task_id,
            progress=progress,
            message=f"Downloading {filename}{size_hint}",
            bytes_downloaded=bytes_downloaded,
            total_bytes=known_total,
            speed_mbps=round(speed_mbps, 2),
            eta_seconds=eta,
            filename=filename,
            model_format=model_format,
            huggingface_id=progress_hf_id,
        )
        last_emitted_bytes = bytes_downloaded
        last_emit_time = now

    if download_result["error"]:
        err = download_result["error"]
        await progress_manager.send_download_progress(
            task_id=task_id,
            progress=0,
            message=f"Download failed: {err}",
            bytes_downloaded=0,
            total_bytes=total_bytes,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename,
            model_format=model_format,
            huggingface_id=progress_hf_id,
        )
        raise err

    # Success: get final path and size
    file_path = download_result["file_path"]
    real_path = os.path.realpath(file_path) if file_path else file_path
    file_size = os.path.getsize(real_path if os.path.exists(real_path) else file_path)

    await progress_manager.send_download_progress(
        task_id=task_id,
        progress=100,
        message=f"Download completed: {filename}",
        bytes_downloaded=file_size,
        total_bytes=file_size,
        speed_mbps=0,
        eta_seconds=0,
        filename=filename,
        model_format=model_format,
        huggingface_id=progress_hf_id,
    )

    return file_path, file_size


async def get_quantization_sizes_from_hf(
    huggingface_id: str, quantizations: Dict[str, Dict]
) -> Dict[str, Dict]:
    """Return actual file sizes for provided quantizations using Hugging Face Hub API.
    Uses the shared hf_api instance and mirrors logic used elsewhere in this module.
    """
    try:
        # Prefer fetching only required files to reduce payload.
        all_filenames: List[str] = []
        quant_to_files: Dict[str, List[str]] = {}

        for quant_name, quant_data in (quantizations or {}).items():
            if not isinstance(quant_data, dict):
                continue
            files = quant_data.get("files")
            if isinstance(files, list) and files:
                paths = [
                    f.get("filename")
                    for f in files
                    if isinstance(f, dict) and f.get("filename")
                ]
            else:
                single = quant_data.get("filename")
                paths = [single] if single else []

            paths = [p for p in paths if p]
            if not paths:
                continue
            quant_to_files[quant_name] = paths
            all_filenames.extend(paths)

        updated: Dict[str, Dict] = {}

        if all_filenames:
            file_sizes = get_accurate_file_sizes(huggingface_id, all_filenames)
            if not file_sizes:
                # Fallback: fetch full metadata once
                try:
                    model_info = hf_api.model_info(
                        repo_id=huggingface_id, files_metadata=True
                    )
                    if hasattr(model_info, "siblings") and model_info.siblings:
                        for sibling in model_info.siblings:
                            key = getattr(sibling, "path", getattr(sibling, "rfilename", ""))
                            if key:
                                file_sizes[key] = getattr(sibling, "size", None)
                except Exception as fallback_err:
                    logger.warning(
                        f"model_info fallback failed for {huggingface_id}: {fallback_err}"
                    )
                    file_sizes = {}

            for quant_name, filenames in quant_to_files.items():
                files_with_sizes = []
                total_size = 0
                for filename in filenames:
                    actual_size = file_sizes.get(filename)
                    if not actual_size or actual_size <= 0:
                        try:
                            file_info = hf_api.repo_file_info(
                                repo_id=huggingface_id, path=filename
                            )
                            actual_size = getattr(file_info, "size", None)
                        except Exception as file_err:
                            logger.warning(
                                f"repo_file_info failed for {huggingface_id}/{filename}: {file_err}"
                            )
                            actual_size = None
                    if actual_size and actual_size > 0:
                        total_size += actual_size
                        size_value = actual_size
                    else:
                        logger.warning(
                            f"Unable to determine size for {huggingface_id}/{filename}"
                        )
                        size_value = 0
                    files_with_sizes.append(
                        {
                            "filename": filename,
                            "size": size_value,
                        }
                    )

                if files_with_sizes:
                    updated[quant_name] = {
                        "files": files_with_sizes,
                        "total_size": total_size,
                        "size_mb": (
                            round(total_size / (1024 * 1024), 2) if total_size else 0.0
                        ),
                    }

        return updated
    except Exception as e:
        logger.error(f"Failed to fetch quantization sizes for {huggingface_id}: {e}")
        return {}
