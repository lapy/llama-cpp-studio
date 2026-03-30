from huggingface_hub import HfApi, hf_hub_download
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import json
import os
import threading
import time
import re
import traceback
from datetime import datetime
from backend.logging_config import get_logger
from backend.gguf_reader import get_model_layer_info

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
    logger.info("HuggingFace API key loaded from environment variable")

# Simple cache for search results
_search_cache: Dict[str, Tuple[List[Dict], float]] = {}
_cache_timeout = 300  # 5 minutes

# Cache for safetensors metadata (per repo)
_safetensors_metadata_cache: Dict[str, Tuple[Dict, float]] = {}
_safetensors_metadata_ttl = 600  # 10 minutes


def get_accurate_file_sizes(repo_id: str, paths: List[str]) -> Dict[str, Optional[int]]:
    """Fetch accurate file sizes from HuggingFace API via get_paths_info."""
    if not paths:
        return {}
    try:
        paths_info = hf_api.get_paths_info(repo_id=repo_id, paths=paths)
        return {
            getattr(pi, "path", getattr(pi, "rfilename", "")): getattr(pi, "size", None)
            for pi in paths_info
        }
    except Exception as e:
        logger.warning(f"get_paths_info failed for {repo_id}: {e}")
        return {}


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
    mmproj = [f for f in files if "mmproj" in f.lower() and f.lower().endswith(".gguf")]
    if not mmproj:
        return None
    # Prefer exact mmproj-F16.gguf, then any filename containing F16, then first mmproj
    for name in mmproj:
        if name == "mmproj-F16.gguf":
            return name
    for name in mmproj:
        if "f16" in name.lower():
            return name
    return mmproj[0]


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


MODEL_FORMATS = ("gguf", "safetensors")
MODEL_BASE_DIR = os.path.join("data", "models")
FORMAT_SUBDIRS = {
    "gguf": os.path.join(MODEL_BASE_DIR, "gguf"),
    "safetensors": os.path.join(MODEL_BASE_DIR, "safetensors"),
}
REPO_MANIFEST_FORMATS = {"gguf", "safetensors"}
_format_manifest_locks = {
    fmt: threading.Lock() for fmt in FORMAT_SUBDIRS if fmt not in REPO_MANIFEST_FORMATS
}
_repo_manifest_locks = {fmt: {} for fmt in REPO_MANIFEST_FORMATS}
SAFETENSORS_DIR = FORMAT_SUBDIRS["safetensors"]
GGUF_DIR = FORMAT_SUBDIRS["gguf"]


def _safe_repo_name(huggingface_id: Optional[str]) -> str:
    safe_name = (huggingface_id or "unknown").replace("/", "_")
    return safe_name or "unknown"


def _get_repo_dir(model_format: str, huggingface_id: str) -> str:
    if model_format in ("safetensors", "gguf"):
        safe_repo = _safe_repo_name(huggingface_id)
        base_dir = FORMAT_SUBDIRS[model_format]
        path = os.path.join(base_dir, safe_repo)
    else:
        path = MODEL_BASE_DIR
    os.makedirs(path, exist_ok=True)
    return path


def _get_download_directory(model_format: str, huggingface_id: str) -> str:
    """Return the directory where files for the given format should be stored."""
    if model_format in ("safetensors", "gguf"):
        return _get_repo_dir(model_format, huggingface_id)
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    return MODEL_BASE_DIR


def _hf_repo_folder_name(huggingface_id: str) -> str:
    """Return the HF cache folder name for a model repo (e.g. models--Org--Repo)."""
    return "models--" + huggingface_id.replace("/", "--")


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
        return None


def delete_cached_model_file(huggingface_id: str, filename: str) -> bool:
    """Delete a specific model file from the HuggingFace cache.

    Removes both the snapshot symlink and the underlying content blob.
    Returns True if the file was found and deleted, False otherwise.
    """
    try:
        cached_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            local_files_only=True,
        )
    except Exception:
        logger.warning(
            f"delete_cached_model_file: {huggingface_id}/{filename} not found in HF cache"
        )
        return False

    if os.path.islink(cached_path):
        blob_path = os.path.realpath(cached_path)
        try:
            os.unlink(cached_path)
        except OSError as e:
            logger.warning(f"Could not remove symlink {cached_path}: {e}")
        if os.path.exists(blob_path):
            try:
                os.remove(blob_path)
            except OSError as e:
                logger.warning(f"Could not remove blob {blob_path}: {e}")
    elif os.path.exists(cached_path):
        try:
            os.remove(cached_path)
        except OSError as e:
            logger.warning(f"Could not remove file {cached_path}: {e}")

    logger.info(f"Deleted cached model file: {huggingface_id}/{filename}")
    return True


def _gguf_entry_matches_store_model(
    entry: Dict[str, Any],
    store_model_id: str,
    quantization: Optional[str],
) -> bool:
    """Whether a GGUF manifest row belongs to a given library model (excludes mmproj rows)."""
    fn = entry.get("filename") or ""
    lower = fn.lower()
    if "mmproj" in lower and lower.endswith(".gguf"):
        return False
    if entry.get("model_id") == store_model_id:
        return True
    if entry.get("model_id"):
        return False
    if not quantization:
        return False
    return _extract_quantization(fn).lower() == str(quantization).lower()


def purge_gguf_store_model(
    huggingface_id: str,
    store_model_id: str,
    quantization: Optional[str],
) -> int:
    """
    Remove GGUF manifest entries for this library model and delete the files from the HF hub cache.
    Returns the number of manifest rows removed.
    """
    removed = 0
    manifest_lock = _get_manifest_lock("gguf", huggingface_id)
    with manifest_lock:
        manifest = _load_manifest("gguf", huggingface_id)
        kept: List[Dict[str, Any]] = []
        for entry in manifest:
            if entry.get("huggingface_id") != huggingface_id:
                kept.append(entry)
                continue
            if not _gguf_entry_matches_store_model(
                entry, store_model_id, quantization
            ):
                kept.append(entry)
                continue
            fn = entry.get("filename")
            if fn:
                try:
                    delete_cached_model_file(
                        huggingface_id, _sanitize_filename(fn)
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to delete cached GGUF file {huggingface_id}/{fn}: {exc}"
                    )
            removed += 1
        if kept:
            _save_manifest("gguf", kept, huggingface_id)
        else:
            manifest_path = _get_manifest_path("gguf", huggingface_id)
            try:
                if os.path.exists(manifest_path):
                    os.remove(manifest_path)
            except OSError as exc:
                logger.warning(
                    f"Failed to remove empty GGUF manifest {manifest_path}: {exc}"
                )
            repo_dir = os.path.join(GGUF_DIR, _safe_repo_name(huggingface_id))
            try:
                if os.path.isdir(repo_dir) and not os.listdir(repo_dir):
                    os.rmdir(repo_dir)
            except OSError:
                pass
    return removed


def purge_safetensors_repo_completely(huggingface_id: str) -> None:
    """Delete all safetensors files for a repo, then remove per-repo manifest and empty dirs."""
    # Load manifest without holding a second lock (see _load_repo_safetensors_manifest).
    manifest = _load_repo_safetensors_manifest(huggingface_id)
    for file_entry in list(manifest.get("files") or []):
        fp = file_entry.get("file_path")
        if not fp:
            continue
        if os.path.exists(fp):
            try:
                os.remove(fp)
                parent_dir = os.path.dirname(fp)
                if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                    os.rmdir(parent_dir)
            except OSError as exc:
                logger.warning(f"Failed to delete safetensors file {fp}: {exc}")
    manifest_lock = _get_manifest_lock("safetensors", huggingface_id)
    with manifest_lock:
        manifest_path = _get_manifest_path("safetensors", huggingface_id)
        try:
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
        except OSError as exc:
            logger.warning(
                f"Failed to remove safetensors manifest {manifest_path}: {exc}"
            )
        repo_dir = os.path.join(SAFETENSORS_DIR, _safe_repo_name(huggingface_id))
        try:
            if os.path.isdir(repo_dir) and not os.listdir(repo_dir):
                os.rmdir(repo_dir)
        except OSError:
            pass


def resolve_model_path(
    huggingface_id: str,
    filename: Optional[str] = None,
    model_format: str = "gguf",
) -> Optional[str]:
    """
    Resolve a model's local path from current storage (data/models/...).
    For GGUF: returns path to the specific file if filename is given.
    For safetensors: returns the repo directory (filename ignored).
    Returns None if the path does not exist. Does not create directories.
    """
    if not huggingface_id:
        return None
    safe_repo = _safe_repo_name(huggingface_id)
    base_dir = FORMAT_SUBDIRS.get(model_format, MODEL_BASE_DIR)
    repo_dir = os.path.join(base_dir, safe_repo)
    for prefix in ("", "/app"):
        candidate = repo_dir if not prefix else os.path.join(prefix, repo_dir)
        if not os.path.exists(candidate):
            continue
        if model_format == "gguf" and filename:
            path = os.path.join(candidate, filename)
            if os.path.isfile(path):
                return path
            continue
        if model_format == "safetensors" or not filename:
            if os.path.isdir(candidate):
                return candidate
    return None


def get_model_disk_size(
    huggingface_id: str,
    filename: Optional[str] = None,
    model_format: str = "gguf",
) -> int:
    """
    Compute actual disk usage in bytes for a model in current storage.
    For GGUF: size of the given file. For safetensors: sum of all files in repo dir.
    """
    path = resolve_model_path(huggingface_id, filename, model_format)
    if not path:
        return 0
    if os.path.isfile(path):
        try:
            return os.path.getsize(path)
        except OSError:
            return 0
    if os.path.isdir(path):
        total = 0
        try:
            for _dirpath, _dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(_dirpath, f)
                    if os.path.isfile(fp):
                        total += os.path.getsize(fp)
        except OSError:
            pass
        return total
    return 0


def _get_manifest_lock(
    model_format: str, huggingface_id: Optional[str] = None
) -> threading.Lock:
    if model_format in REPO_MANIFEST_FORMATS:
        if not huggingface_id:
            raise ValueError("huggingface_id is required for repo manifests")
        safe_repo = _safe_repo_name(huggingface_id)
        repo_locks = _repo_manifest_locks[model_format]
        if safe_repo not in repo_locks:
            repo_locks[safe_repo] = threading.Lock()
        return repo_locks[safe_repo]
    return _format_manifest_locks[model_format]


def _get_manifest_path(model_format: str, huggingface_id: Optional[str] = None) -> str:
    if model_format in REPO_MANIFEST_FORMATS:
        if not huggingface_id:
            raise ValueError("huggingface_id is required for repo manifests")
        directory = _get_repo_dir(model_format, huggingface_id)
    else:
        directory = FORMAT_SUBDIRS[model_format]
        os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, "manifest.json")


def _load_manifest(
    model_format: str, huggingface_id: Optional[str] = None
) -> List[Dict]:
    manifest_path = _get_manifest_path(model_format, huggingface_id)
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(
            f"Failed to load {model_format} manifest ({manifest_path}): {exc}"
        )
        return []


def _save_manifest(
    model_format: str, entries: List[Dict], huggingface_id: Optional[str] = None
):
    manifest_path = _get_manifest_path(model_format, huggingface_id)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp_path, manifest_path)


def _load_repo_safetensors_manifest(huggingface_id: str) -> Dict[str, Any]:
    """Load unified safetensors manifest (single object per repo, not per-shard list)."""
    manifest_lock = _get_manifest_lock("safetensors", huggingface_id)
    with manifest_lock:
        manifest_path = _get_manifest_path("safetensors", huggingface_id)
        if not os.path.exists(manifest_path):
            return {
                "huggingface_id": huggingface_id,
                "files": [],
                "metadata": {},
                "max_context_length": None,
            }
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "files" in data:
                    return data
                # Old list format or invalid, return empty
                logger.warning(
                    f"Invalid manifest format for {huggingface_id}, resetting"
                )
                return {
                    "huggingface_id": huggingface_id,
                    "files": [],
                    "metadata": {},
                    "max_context_length": None,
                }
        except Exception as exc:
            logger.warning(
                f"Failed to load safetensors manifest ({manifest_path}): {exc}"
            )
            return {
                "huggingface_id": huggingface_id,
                "files": [],
                "metadata": {},
                "max_context_length": None,
            }


def _save_repo_safetensors_manifest(huggingface_id: str, manifest: Dict[str, Any]):
    """Save unified safetensors manifest (single object per repo)."""
    manifest_lock = _get_manifest_lock("safetensors", huggingface_id)
    with manifest_lock:
        manifest_path = _get_manifest_path("safetensors", huggingface_id)
        tmp_path = f"{manifest_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)


MAX_ROPE_SCALING_FACTOR = 4.0


def get_safetensors_manifest_entries(huggingface_id: str) -> Dict[str, Any]:
    """Get unified safetensors manifest for a repo."""
    return _load_repo_safetensors_manifest(huggingface_id)


def get_safetensors_limits_from_manifest(
    huggingface_id: str,
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (max_context_length, layer_count) from the safetensors manifest for the given
    huggingface_id. layer_count is read from manifest metadata config (num_hidden_layers,
    n_layer, num_layers). Returns (None, None) if manifest is empty or missing.
    """
    manifest = get_safetensors_manifest_entries(huggingface_id)
    if not manifest or not manifest.get("files"):
        return None, None
    max_ctx = manifest.get("max_context_length")
    if isinstance(max_ctx, (int, float)) and max_ctx > 0:
        max_ctx = int(max_ctx)
    else:
        max_ctx = None
    config = (manifest.get("metadata") or {}).get("config")
    if not isinstance(config, dict):
        return max_ctx, None
    layer_count = None
    for key in ("num_hidden_layers", "n_layer", "num_layers"):
        val = config.get(key)
        if isinstance(val, (int, float)) and val > 0:
            # Config reports hidden block count; +1 for output head matches llama-server layer count.
            layer_count = int(val) + 1
            break
    return max_ctx, layer_count


def save_safetensors_manifest_entries(huggingface_id: str, manifest: Dict[str, Any]):
    """Save unified safetensors manifest for a repo."""
    _save_repo_safetensors_manifest(huggingface_id, manifest)


def record_safetensors_download(
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    tensor_summary: Optional[Dict[str, Any]] = None,
    model_id: Optional[int] = None,
):
    """Record safetensors download metadata in unified manifest."""
    metadata = metadata or {}
    tensor_summary = tensor_summary or {}

    manifest = _load_repo_safetensors_manifest(huggingface_id)

    # Update repo-level fields
    if model_id:
        manifest["model_id"] = model_id
    if metadata:
        manifest["metadata"] = metadata
    max_ctx = metadata.get("max_context_length")
    if max_ctx:
        existing_max = manifest.get("max_context_length")
        if existing_max is None or max_ctx > existing_max:
            manifest["max_context_length"] = max_ctx

    # Add/update file entry
    manifest.setdefault("files", [])
    file_entry = {
        "filename": filename,
        "file_path": file_path,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else 0,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "tensor_summary": tensor_summary,
    }

    # Remove existing entry for this filename if present
    manifest["files"] = [f for f in manifest["files"] if f.get("filename") != filename]
    manifest["files"].append(file_entry)

    _save_repo_safetensors_manifest(huggingface_id, manifest)


def list_safetensors_downloads() -> List[Dict]:
    """Return unified safetensors manifests (one per repo), pruning missing files."""
    results: List[Dict] = []
    if not os.path.exists(SAFETENSORS_DIR):
        return results

    for repo_dir in os.scandir(SAFETENSORS_DIR):
        if not repo_dir.is_dir():
            continue
        manifest_path = os.path.join(repo_dir.path, "manifest.json")
        if not os.path.exists(manifest_path):
            continue

        # Extract huggingface_id from directory name
        repo_name = repo_dir.name
        huggingface_id = repo_name.replace("_", "/")

        try:
            manifest = _load_repo_safetensors_manifest(huggingface_id)
        except Exception as exc:
            logger.warning(
                f"Failed to load safetensors manifest at {manifest_path}: {exc}"
            )
            continue

        if not isinstance(manifest, dict) or "files" not in manifest:
            continue

        # Prune missing files and update sizes
        changed = False
        valid_files = []
        for file_entry in manifest.get("files", []):
            file_path = file_entry.get("file_path")
            if file_path and os.path.exists(file_path):
                size = os.path.getsize(file_path)
                file_size_mb = round(size / (1024 * 1024), 2)
                if size != file_entry.get("file_size"):
                    file_entry["file_size"] = size
                    file_entry["file_size_mb"] = file_size_mb
                    changed = True
                valid_files.append(file_entry)
            else:
                changed = True
                logger.debug(f"Pruning missing safetensors file: {file_path}")

        if changed:
            manifest["files"] = valid_files
            _save_repo_safetensors_manifest(huggingface_id, manifest)

        # Only include manifests with at least one valid file
        if valid_files:
            results.append(manifest)

    return results


def list_grouped_safetensors_downloads() -> List[Dict]:
    """Return unified safetensors manifests formatted for UI consumption."""
    manifests = list_safetensors_downloads()

    # Format each unified manifest for UI
    formatted = []
    for manifest in manifests:
        files = manifest.get("files", [])
        total_size = sum(f.get("file_size", 0) for f in files)

        # Find latest download time
        latest_downloaded_at = None
        for file_entry in files:
            downloaded_at = file_entry.get("downloaded_at")
            if downloaded_at:
                if latest_downloaded_at is None or downloaded_at > latest_downloaded_at:
                    latest_downloaded_at = downloaded_at

        formatted.append(
            {
                "huggingface_id": manifest.get("huggingface_id"),
                "files": files,
                "file_count": len(files),
                "total_size": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "latest_downloaded_at": latest_downloaded_at,
                "metadata": manifest.get("metadata", {}),
                "max_context_length": manifest.get("max_context_length"),
                "model_id": manifest.get("model_id"),
            }
        )

    # Sort by latest download time (descending)
    formatted.sort(
        key=lambda g: g.get("latest_downloaded_at") or "",
        reverse=True,
    )
    return formatted


async def collect_gguf_runtime_metadata(
    huggingface_id: Optional[str], file_path: str
) -> Tuple[Dict[str, Any], Dict[str, Any], Optional[int]]:
    """Gather Hugging Face model card metadata and GGUF layer info for manifest entries."""
    metadata: Dict[str, Any] = {}
    layer_info: Dict[str, Any] = {}
    max_context_length: Optional[int] = None

    async def _fetch_and_merge(repo_id: Optional[str]):
        nonlocal metadata, max_context_length
        if not repo_id:
            return
        try:
            details = await get_model_details(repo_id)
        except Exception as exc:
            logger.warning(
                f"Failed to collect model card metadata for {repo_id}: {exc}"
            )
            return
        if not details:
            return
        context_from_card = details.get("context_length")
        candidate = {
            "architecture": details.get("architecture"),
            "base_model": details.get("base_model"),
            "pipeline_tag": details.get("pipeline_tag"),
            "parameters": details.get("parameters"),
            "context_length": context_from_card,
            "language": details.get("language"),
            "license": details.get("license"),
        }
        for key, value in candidate.items():
            if value and (metadata.get(key) in (None, "", [], {})):
                metadata[key] = value
        if context_from_card and not max_context_length:
            metadata["max_context_length"] = context_from_card
            max_context_length = context_from_card
        tokenizer_config = _get_tokenizer_config(repo_id)
        if tokenizer_config and "tokenizer_config" not in metadata:
            metadata["tokenizer_config"] = tokenizer_config
        tokenizer_max = None
        if tokenizer_config:
            tokenizer_max = next(
                (
                    tokenizer_config.get(key)
                    for key in ("model_max_length", "max_len", "max_length")
                    if isinstance(tokenizer_config.get(key), int)
                    and tokenizer_config.get(key) > 0
                ),
                None,
            )
        if tokenizer_max and not max_context_length:
            metadata.setdefault("context_length", tokenizer_max)
            metadata["max_context_length"] = tokenizer_max
            max_context_length = tokenizer_max

        config_json = _download_repo_json(repo_id, "config.json")
        if config_json and "config" not in metadata:
            metadata["config"] = config_json

        generation_config = _download_repo_json(repo_id, "generation_config.json")
        if generation_config and "generation_config" not in metadata:
            metadata["generation_config"] = generation_config
        gen_ctx = None
        if generation_config:
            gen_ctx = next(
                (
                    generation_config.get(key)
                    for key in (
                        "max_length",
                        "max_position_embeddings",
                        "max_tokens",
                        "max_new_tokens",
                    )
                    if isinstance(generation_config.get(key), int)
                    and generation_config.get(key) > 0
                ),
                None,
            )
        if gen_ctx and not max_context_length:
            metadata.setdefault("context_length", gen_ctx)
            metadata["max_context_length"] = gen_ctx
            max_context_length = gen_ctx

        special_tokens_map = _download_repo_json(repo_id, "special_tokens_map.json")
        if special_tokens_map and "special_tokens_map" not in metadata:
            metadata["special_tokens_map"] = special_tokens_map

        tokenizer_json = _download_repo_json(repo_id, "tokenizer.json")
        if tokenizer_json and "tokenizer" not in metadata:
            metadata["tokenizer"] = tokenizer_json

    await _fetch_and_merge(huggingface_id)

    try:
        layer_info = get_model_layer_info(file_path) or {}
        layer_context = layer_info.get("context_length")
        if layer_context:
            max_context_length = layer_context
            metadata.setdefault("max_context_length", layer_context)
    except Exception as exc:
        logger.warning(f"Failed to read GGUF layer info for {file_path}: {exc}")

    return metadata or {}, layer_info or {}, max_context_length


def record_gguf_download(
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    layer_info: Optional[Dict[str, Any]] = None,
    max_context_length: Optional[int] = None,
    model_id: Optional[int] = None,
):
    """Record GGUF download metadata in manifest."""
    metadata = metadata or {}
    layer_info = layer_info or {}
    entry = {
        "huggingface_id": huggingface_id,
        "filename": filename,
        "file_path": file_path,
        "file_size": file_size,
        "file_size_mb": round(file_size / (1024 * 1024), 2) if file_size else 0,
        "downloaded_at": datetime.utcnow().isoformat() + "Z",
        "model_id": model_id,
        "metadata": metadata,
        "gguf_layer_info": layer_info,
        "max_context_length": max_context_length,
    }
    manifest_lock = _get_manifest_lock("gguf", huggingface_id)
    with manifest_lock:
        manifest = _load_manifest("gguf", huggingface_id)
        manifest = [
            e
            for e in manifest
            if not (
                e.get("huggingface_id") == huggingface_id
                and e.get("filename") == filename
            )
        ]
        manifest.append(entry)
        _save_manifest("gguf", manifest, huggingface_id)
    return entry


def get_gguf_manifest_entry(
    huggingface_id: str, filename: str
) -> Optional[Dict[str, Any]]:
    safe_filename = _sanitize_filename(filename)
    manifest_lock = _get_manifest_lock("gguf", huggingface_id)
    with manifest_lock:
        manifest = _load_manifest("gguf", huggingface_id)
        for entry in manifest:
            if (
                entry.get("huggingface_id") == huggingface_id
                and entry.get("filename") == safe_filename
            ):
                return entry
    return None


def resolve_gguf_model_path_for_quant(
    huggingface_id: str, quantization: str
) -> Optional[str]:
    """
    Return the on-disk path for the main GGUF file (or first shard) for the given
    huggingface_id and quantization, from the app's GGUF manifest. Excludes mmproj.
    Returns None if not found or file missing (caller can fall back to --hf-repo).
    """
    if not huggingface_id or not quantization:
        return None
    quant_lower = str(quantization).lower()
    manifest_lock = _get_manifest_lock("gguf", huggingface_id)
    with manifest_lock:
        manifest = _load_manifest("gguf", huggingface_id)
    matching = []
    for entry in manifest:
        fn = entry.get("filename") or ""
        if "mmproj" in fn.lower():
            continue
        entry_quant = _extract_quantization(fn)
        if entry_quant.lower() != quant_lower:
            continue
        file_path = entry.get("file_path")
        if file_path:
            matching.append(entry)
    if not matching:
        return None
    # Sort so the first shard is chosen: no -shard, then -shard1, then by name
    def shard_order(e: Dict[str, Any]) -> tuple:
        fn = (e.get("filename") or "").lower()
        if "-shard" not in fn:
            return (0, 0, fn)
        m = re.search(r"-shard(\d+)", fn)
        return (1, int(m.group(1)) if m else 999, fn)
    matching.sort(key=shard_order)
    first_path = matching[0].get("file_path")
    if first_path and os.path.exists(first_path):
        return first_path
    return None


def get_gguf_limits_from_manifest(
    huggingface_id: str, quantization: str
) -> Tuple[Optional[int], Optional[int]]:
    """
    Return (max_context_length, layer_count) from the GGUF manifest for the given
    huggingface_id and quantization. Uses the first matching main-model entry (excludes mmproj).
    Returns (None, None) if no matching entry is found.
    """
    if not huggingface_id or not quantization:
        return None, None
    quant_lower = str(quantization).lower()
    manifest_lock = _get_manifest_lock("gguf", huggingface_id)
    with manifest_lock:
        manifest = _load_manifest("gguf", huggingface_id)
    matching = []
    for entry in manifest:
        fn = entry.get("filename") or ""
        if "mmproj" in fn.lower():
            continue
        entry_quant = _extract_quantization(fn)
        if entry_quant.lower() != quant_lower:
            continue
        matching.append(entry)
    if not matching:
        return None, None
    def shard_order(e: Dict[str, Any]) -> tuple:
        fn = (e.get("filename") or "").lower()
        if "-shard" not in fn:
            return (0, 0, fn)
        m = re.search(r"-shard(\d+)", fn)
        return (1, int(m.group(1)) if m else 999, fn)
    matching.sort(key=shard_order)
    entry = matching[0]
    max_ctx = entry.get("max_context_length")
    if isinstance(max_ctx, (int, float)) and max_ctx > 0:
        max_ctx = int(max_ctx)
    else:
        max_ctx = None
    layer_info = entry.get("gguf_layer_info") or {}
    layer_count = layer_info.get("layer_count")
    if isinstance(layer_count, (int, float)) and layer_count > 0:
        layer_count = int(layer_count)
    else:
        layer_count = None
    return max_ctx, layer_count


def list_gguf_downloads() -> List[Dict]:
    """Return GGUF downloads from manifest, pruning missing files."""
    base_dir = FORMAT_SUBDIRS["gguf"]
    if not os.path.exists(base_dir):
        return []

    result = []
    for repo_name in os.listdir(base_dir):
        repo_dir = os.path.join(base_dir, repo_name)
        if not os.path.isdir(repo_dir):
            continue
        manifest_lock = _get_manifest_lock("gguf", repo_name)
        with manifest_lock:
            manifest = _load_manifest("gguf", repo_name)
            updated_manifest = []
            changed = False
            for entry in manifest:
                file_path = entry.get("file_path")
                if file_path and os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    entry["file_size"] = size
                    entry["file_size_mb"] = round(size / (1024 * 1024), 2)
                    result.append(entry)
                    updated_manifest.append(entry)
                else:
                    changed = True
                    logger.debug(f"Pruning missing GGUF file: {file_path}")
            if changed:
                _save_manifest("gguf", updated_manifest, repo_name)
    return result


async def create_gguf_manifest_entry(
    huggingface_id: Optional[str],
    file_path: str,
    file_size: int,
    *,
    model_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Collect metadata and persist a GGUF manifest entry."""
    metadata, layer_info, max_context = await collect_gguf_runtime_metadata(
        huggingface_id, file_path
    )
    filename = os.path.basename(file_path)
    return record_gguf_download(
        huggingface_id=huggingface_id or "unknown",
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        metadata=metadata,
        layer_info=layer_info,
        max_context_length=max_context,
        model_id=model_id,
    )


def delete_safetensors_download(huggingface_id: str, filename: str) -> None:
    """Delete a safetensors file and remove it from unified manifest."""
    manifest = _load_repo_safetensors_manifest(huggingface_id)

    if not isinstance(manifest, dict) or "files" not in manifest:
        return

    files = manifest.get("files", [])
    remaining_files = []
    file_deleted = False

    for file_entry in files:
        if file_entry.get("filename") == filename:
            file_path = file_entry.get("file_path")
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    parent_dir = os.path.dirname(file_path)
                    if os.path.isdir(parent_dir) and not os.listdir(parent_dir):
                        os.rmdir(parent_dir)
                file_deleted = True
            except Exception as exc:
                logger.warning(f"Failed to delete safetensors file {file_path}: {exc}")
        else:
            remaining_files.append(file_entry)

    if file_deleted:
        manifest["files"] = remaining_files
        _save_repo_safetensors_manifest(huggingface_id, manifest)


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
        logger.warning(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
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
                logger.info(f"Returning cached results for '{query}'")
                return cached_data[:limit]  # Return only requested limit

        logger.info(
            f"Searching for models with query: '{query}', limit: {limit}, format: {model_format}"
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
        logger.info(
            f"Found {len(models)} models from HuggingFace API with expanded metadata"
        )

        # Process models in parallel for better performance
        results = await _process_models_parallel(models, limit, model_format)

        # Cache the results
        cache_key = f"{model_format}:{query.lower()}_{limit}"
        _search_cache[cache_key] = (results, time.time())

        logger.info(f"Returning {len(results)} results from API")
        return results

    except Exception as e:
        logger.error(f"API search error: {e}")
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
            logger.error(f"Model processing error: {result}")
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
        logger.info(f"Processing model: {model.id}")

        quantizations: Dict[str, Dict] = {}
        mmproj_files: List[Dict[str, Any]] = []
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
                logger.info(f"Model {model.id}: {len(gguf_siblings)} GGUF files found")
                if not gguf_siblings:
                    return None

                for sibling in gguf_siblings:
                    filename = sibling.rfilename
                    if "mmproj" in filename.lower():
                        mmproj_files.append(
                            {
                                "filename": filename,
                                "size": getattr(sibling, "size", 0) or 0,
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
                    size_bytes = getattr(sibling, "size", 0) or 0
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
                if not quantizations and not mmproj_files:
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

                logger.info(
                    f"Model {model.id}: {len(safetensors_files)} safetensors files found"
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
            "safetensors_files": (
                safetensors_files if model_format == "safetensors" else []
            ),
            "repo_files": repo_files if model_format == "safetensors" else [],
            **metadata,  # Include all extracted metadata
        }

        logger.info(f"Added model {model.id} to results")
        return result

    except Exception as e:
        logger.error(f"Error processing model {model.id}: {e}")
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
            logger.warning(
                f"hf_transfer not available for {model_id}, falling back to standard download. Error: {err}"
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


async def get_model_details(model_id: str) -> Dict:
    """Get detailed model information including config and README"""
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


async def download_model(
    huggingface_id: str, filename: str, model_format: str = "gguf"
) -> tuple[str, int]:
    """Download model from HuggingFace to the native HF cache."""
    try:
        filename = _sanitize_filename(filename)

        file_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
        )

        # Use realpath so getsize works even when file_path is a symlink
        real_path = os.path.realpath(file_path)
        file_size = os.path.getsize(real_path if os.path.exists(real_path) else file_path)

        return file_path, file_size

    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


async def download_model_with_progress(
    huggingface_id: str,
    filename: str,
    progress_manager,
    task_id: str,
    total_bytes: int = 0,
    model_format: str = "gguf",
    huggingface_id_for_progress: str = None,
):
    """Download model to the HF native cache with SSE progress updates.

    Progress is tracked by monitoring the .incomplete blob file that hf_hub_download
    writes to the HF cache during the download.
    """
    import threading
    import time
    from huggingface_hub.constants import HF_HUB_CACHE

    filename = _sanitize_filename(filename)
    progress_hf_id = huggingface_id_for_progress or huggingface_id

    logger.info(f"Starting HF-cache download: {huggingface_id}/{filename} task={task_id}")

    # Resolve total size if not provided
    if total_bytes == 0:
        try:
            file_info = HfApi().repo_file_info(repo_id=huggingface_id, filename=filename)
            total_bytes = file_info.size or 0
            logger.info(f"Got file size from HuggingFace API: {total_bytes}")
        except Exception as e:
            logger.warning(f"Could not get file size: {e}")

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

    # Run the blocking hf_hub_download in a background thread
    repo_folder = _hf_repo_folder_name(huggingface_id)
    blobs_dir = os.path.join(HF_HUB_CACHE, repo_folder, "blobs")

    download_result: dict = {"file_path": None, "error": None, "done": False}

    def _do_download():
        try:
            download_result["file_path"] = hf_hub_download(
                repo_id=huggingface_id,
                filename=filename,
            )
        except Exception as exc:
            download_result["error"] = exc
        finally:
            download_result["done"] = True

    thread = threading.Thread(target=_do_download, daemon=True)
    thread.start()

    # Poll the .incomplete blob for progress
    start_time = time.time()
    last_bytes = 0
    last_poll = start_time

    while not download_result["done"]:
        await asyncio.sleep(0.5)

        incomplete_bytes = 0
        if os.path.isdir(blobs_dir):
            for fname in os.listdir(blobs_dir):
                if fname.endswith(".incomplete"):
                    try:
                        incomplete_bytes = max(
                            incomplete_bytes,
                            os.path.getsize(os.path.join(blobs_dir, fname)),
                        )
                    except OSError:
                        pass

        if incomplete_bytes > 0:
            now = time.time()
            elapsed_total = now - start_time
            elapsed_poll = now - last_poll
            delta = incomplete_bytes - last_bytes
            speed_mbps = (delta / elapsed_poll / (1024 * 1024)) if elapsed_poll > 0 else 0
            progress = min(99, int(incomplete_bytes / total_bytes * 100)) if total_bytes else 0
            eta = (
                int((total_bytes - incomplete_bytes) / (incomplete_bytes / elapsed_total))
                if elapsed_total > 0 and incomplete_bytes > 0 and total_bytes > incomplete_bytes
                else 0
            )
            await progress_manager.send_download_progress(
                task_id=task_id,
                progress=progress,
                message=f"Downloading {filename}",
                bytes_downloaded=incomplete_bytes,
                total_bytes=total_bytes,
                speed_mbps=round(speed_mbps, 2),
                eta_seconds=eta,
                filename=filename,
                model_format=model_format,
                huggingface_id=progress_hf_id,
            )
            last_bytes = incomplete_bytes
            last_poll = now

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
