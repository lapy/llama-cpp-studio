from huggingface_hub import HfApi, hf_hub_download, list_models
from typing import List, Dict, Optional, Tuple, Any
import asyncio
import aiohttp
import json
import os
import threading
from tqdm import tqdm
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
_env_token = os.getenv('HUGGINGFACE_API_KEY')
if _env_token:
    hf_api = HfApi(token=_env_token)
    logger.info("HuggingFace API key loaded from environment variable")

# Simple cache for search results
_search_cache: Dict[str, Tuple[List[Dict], float]] = {}
_cache_timeout = 300  # 5 minutes

# Cache for safetensors metadata (per repo)
_safetensors_metadata_cache: Dict[str, Tuple[Dict, float]] = {}
_safetensors_metadata_ttl = 600  # 10 minutes


def _download_repo_json(repo_id: str, filename: str) -> Optional[Dict[str, Any]]:
    try:
        path = hf_hub_download(repo_id, filename, local_dir_use_symlinks=False)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.debug(f"Unable to download {filename} for {repo_id}: {exc}")
        return None


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
    fmt: threading.Lock()
    for fmt in FORMAT_SUBDIRS
    if fmt not in REPO_MANIFEST_FORMATS
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


def _get_manifest_lock(model_format: str, huggingface_id: Optional[str] = None) -> threading.Lock:
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


def _load_manifest(model_format: str, huggingface_id: Optional[str] = None) -> List[Dict]:
    manifest_path = _get_manifest_path(model_format, huggingface_id)
    if not os.path.exists(manifest_path):
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception as exc:
        logger.warning(f"Failed to load {model_format} manifest ({manifest_path}): {exc}")
        return []


def _save_manifest(model_format: str, entries: List[Dict], huggingface_id: Optional[str] = None):
    manifest_path = _get_manifest_path(model_format, huggingface_id)
    tmp_path = f"{manifest_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp_path, manifest_path)


LEGACY_SAFETENSORS_MANIFEST = os.path.join(SAFETENSORS_DIR, "manifest.json")
_legacy_manifest_migrated = False

def _migrate_legacy_safetensors_manifest():
    global _legacy_manifest_migrated
    if _legacy_manifest_migrated:
        return
    _legacy_manifest_migrated = True
    if not os.path.exists(LEGACY_SAFETENSORS_MANIFEST):
        return
    try:
        with open(LEGACY_SAFETENSORS_MANIFEST, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to read legacy safetensors manifest: {exc}")
        return
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries if isinstance(entries, list) else []:
        repo_id = entry.get("huggingface_id")
        if not repo_id:
            continue
        grouped.setdefault(repo_id, []).append(entry)
    for repo_id, repo_entries in grouped.items():
        _save_repo_safetensors_manifest(repo_id, repo_entries)
    try:
        os.remove(LEGACY_SAFETENSORS_MANIFEST)
        logger.info("Migrated legacy safetensors manifest to per-repo manifests")
    except Exception as exc:
        logger.warning(f"Failed to remove legacy safetensors manifest: {exc}")

def _load_repo_safetensors_manifest(huggingface_id: str) -> Dict[str, Any]:
    """Load unified safetensors manifest (single object per repo, not per-shard list)."""
    _migrate_legacy_safetensors_manifest()
    manifest_lock = _get_manifest_lock("safetensors", huggingface_id)
    with manifest_lock:
        manifest_path = _get_manifest_path("safetensors", huggingface_id)
        if not os.path.exists(manifest_path):
            return {
                "huggingface_id": huggingface_id,
                "files": [],
                "metadata": {},
                "max_context_length": None,
                "lmdeploy": {
                    "config": get_default_lmdeploy_config(),
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                },
            }
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Migrate old list format to new unified format
                if isinstance(data, list):
                    return _migrate_manifest_list_to_unified(huggingface_id, data)
                # Already unified format
                if isinstance(data, dict) and "files" in data:
                    return data
                # Invalid format, return empty
                logger.warning(f"Invalid manifest format for {huggingface_id}, resetting")
                return {
                    "huggingface_id": huggingface_id,
                    "files": [],
                    "metadata": {},
                    "max_context_length": None,
                    "lmdeploy": {
                        "config": get_default_lmdeploy_config(),
                        "updated_at": datetime.utcnow().isoformat() + "Z",
                    },
                }
        except Exception as exc:
            logger.warning(f"Failed to load safetensors manifest ({manifest_path}): {exc}")
            return {
                "huggingface_id": huggingface_id,
                "files": [],
                "metadata": {},
                "max_context_length": None,
                "lmdeploy": {
                    "config": get_default_lmdeploy_config(),
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                },
            }

def _migrate_manifest_list_to_unified(huggingface_id: str, entries: List[Dict]) -> Dict[str, Any]:
    """Migrate old per-shard list format to unified structure."""
    if not entries:
        return {
            "huggingface_id": huggingface_id,
            "files": [],
            "metadata": {},
            "max_context_length": None,
            "lmdeploy": {
                "config": get_default_lmdeploy_config(),
                "updated_at": datetime.utcnow().isoformat() + "Z",
            },
        }
    
    # Aggregate metadata from all entries
    unified_metadata = {}
    max_context_length = None
    model_id = None
    latest_lmdeploy = None
    latest_lmdeploy_updated = None
    
    files = []
    for entry in entries:
        # Extract file-level info
        file_info = {
            "filename": entry.get("filename"),
            "file_path": entry.get("file_path"),
            "file_size": entry.get("file_size", 0),
            "file_size_mb": entry.get("file_size_mb", 0),
            "downloaded_at": entry.get("downloaded_at"),
            "tensor_summary": entry.get("tensor_summary", {}),
        }
        files.append(file_info)
        
        # Aggregate repo-level metadata
        if entry.get("model_id") and not model_id:
            model_id = entry.get("model_id")
        
        entry_metadata = entry.get("metadata") or {}
        if entry_metadata and not unified_metadata:
            unified_metadata = entry_metadata
        
        entry_max_ctx = entry.get("max_context_length")
        if entry_max_ctx:
            if max_context_length is None or entry_max_ctx > max_context_length:
                max_context_length = entry_max_ctx
        
        # Use most recent LMDeploy config
        entry_lmdeploy = entry.get("lmdeploy", {})
        entry_updated = entry_lmdeploy.get("updated_at")
        if entry_updated:
            if latest_lmdeploy_updated is None or entry_updated > latest_lmdeploy_updated:
                latest_lmdeploy = entry_lmdeploy
                latest_lmdeploy_updated = entry_updated
    
    result = {
        "huggingface_id": huggingface_id,
        "files": files,
        "metadata": unified_metadata,
        "max_context_length": max_context_length,
        "lmdeploy": latest_lmdeploy or {
            "config": get_default_lmdeploy_config(max_context_length),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        },
    }
    if model_id:
        result["model_id"] = model_id
    
    return result

def _save_repo_safetensors_manifest(huggingface_id: str, manifest: Dict[str, Any]):
    """Save unified safetensors manifest (single object per repo)."""
    manifest_lock = _get_manifest_lock("safetensors", huggingface_id)
    with manifest_lock:
        manifest_path = _get_manifest_path("safetensors", huggingface_id)
        tmp_path = f"{manifest_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        os.replace(tmp_path, manifest_path)

DEFAULT_LMDEPLOY_CONTEXT = 4096
MAX_LMDEPLOY_CONTEXT = 256000
MAX_ROPE_SCALING_FACTOR = 4.0


def get_safetensors_manifest_entries(huggingface_id: str) -> Dict[str, Any]:
    """Get unified safetensors manifest for a repo."""
    return _load_repo_safetensors_manifest(huggingface_id)


def save_safetensors_manifest_entries(huggingface_id: str, manifest: Dict[str, Any]):
    """Save unified safetensors manifest for a repo."""
    _save_repo_safetensors_manifest(huggingface_id, manifest)


def get_default_lmdeploy_config(max_context_length: Optional[int] = None) -> Dict[str, Any]:
    """Return default LMDeploy runtime configuration."""
    context_len = max_context_length or DEFAULT_LMDEPLOY_CONTEXT
    context_len = max(1024, min(context_len, MAX_LMDEPLOY_CONTEXT))
    return {
        "session_len": context_len,
        "max_context_token_num": context_len,
        "max_prefill_token_num": context_len * 2,
        "tensor_parallel": 1,
        "tensor_split": [],
        "max_batch_size": 4,
        "dtype": "auto",
        "cache_max_entry_count": 0.8,
        "cache_block_seq_len": 64,
        "enable_prefix_caching": False,
        "quant_policy": 0,
        "model_format": "",
        "hf_overrides": {},
        "enable_metrics": False,
        "rope_scaling_mode": "disabled",
        "rope_scaling_factor": 1.0,
        "num_tokens_per_iter": 0,
        "max_prefill_iters": 1,
        "communicator": "nccl",
        "additional_args": "",
        "effective_session_len": context_len,
    }


def record_safetensors_download(
    huggingface_id: str,
    filename: str,
    file_path: str,
    file_size: int,
    *,
    metadata: Optional[Dict[str, Any]] = None,
    tensor_summary: Optional[Dict[str, Any]] = None,
    lmdeploy_config: Optional[Dict[str, Any]] = None,
    model_id: Optional[int] = None,
):
    """Record safetensors download metadata in unified manifest."""
    metadata = metadata or {}
    tensor_summary = tensor_summary or {}
    lmdeploy_config = lmdeploy_config or get_default_lmdeploy_config(metadata.get("max_context_length"))
    
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
    
    # Update LMDeploy config if provided
    if lmdeploy_config:
        manifest.setdefault("lmdeploy", {})
        manifest["lmdeploy"]["config"] = lmdeploy_config
        manifest["lmdeploy"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
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


def update_lmdeploy_config(huggingface_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Update the stored LMDeploy config for a safetensors repo (unified manifest).
    
    Args:
        huggingface_id: The Hugging Face repository ID
        config: The LMDeploy configuration to store
    """
    manifest = _load_repo_safetensors_manifest(huggingface_id)
    
    # Update repo-level LMDeploy config
    manifest.setdefault("lmdeploy", {})
    manifest["lmdeploy"]["config"] = config
    manifest["lmdeploy"]["updated_at"] = datetime.utcnow().isoformat() + "Z"
    
    # Update max_context_length if rope_scaling provides original_max_position_embeddings
    rope_overrides = config.get("hf_overrides") if isinstance(config.get("hf_overrides"), dict) else {}
    rope_scaling = rope_overrides.get("rope_scaling") if isinstance(rope_overrides.get("rope_scaling"), dict) else {}
    base_override = (
        rope_scaling.get("original_max_position_embeddings")
        or rope_scaling.get("original_max_position_embedding")
    )
    if base_override:
        try:
            manifest["max_context_length"] = int(base_override)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid original_max_position_embeddings override for %s",
                huggingface_id,
            )
    
    _save_repo_safetensors_manifest(huggingface_id, manifest)
    
    return manifest


def list_safetensors_downloads() -> List[Dict]:
    """Return unified safetensors manifests (one per repo), pruning missing files."""
    _migrate_legacy_safetensors_manifest()
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
            logger.warning(f"Failed to load safetensors manifest at {manifest_path}: {exc}")
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
        
        formatted.append({
            "huggingface_id": manifest.get("huggingface_id"),
            "files": files,
            "file_count": len(files),
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "latest_downloaded_at": latest_downloaded_at,
            "metadata": manifest.get("metadata", {}),
            "max_context_length": manifest.get("max_context_length"),
            "model_id": manifest.get("model_id"),
            "lmdeploy": manifest.get("lmdeploy", {}),
        })
    
    # Sort by latest download time (descending)
    formatted.sort(
        key=lambda g: g.get("latest_downloaded_at") or "",
        reverse=True,
    )
    return formatted


async def collect_gguf_runtime_metadata(
    huggingface_id: Optional[str],
    file_path: str
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
            logger.warning(f"Failed to collect model card metadata for {repo_id}: {exc}")
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
                    if isinstance(tokenizer_config.get(key), int) and tokenizer_config.get(key) > 0
                ),
                None
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
                    for key in ("max_length", "max_position_embeddings", "max_tokens", "max_new_tokens")
                    if isinstance(generation_config.get(key), int) and generation_config.get(key) > 0
                ),
                None
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
    if huggingface_id and huggingface_id.lower().endswith("-gguf"):
        base_repo = huggingface_id[:-5]
        await _fetch_and_merge(base_repo)

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
            e for e in manifest
            if not (e.get("huggingface_id") == huggingface_id and e.get("filename") == filename)
        ]
        manifest.append(entry)
        _save_manifest("gguf", manifest, huggingface_id)
    return entry


def get_gguf_manifest_entry(huggingface_id: str, filename: str) -> Optional[Dict[str, Any]]:
    safe_filename = _sanitize_filename(filename)
    manifest_lock = _get_manifest_lock("gguf", huggingface_id)
    with manifest_lock:
        manifest = _load_manifest("gguf", huggingface_id)
        for entry in manifest:
            if entry.get("huggingface_id") == huggingface_id and entry.get("filename") == safe_filename:
                return entry
    return None


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
    model_id: Optional[int] = None
) -> Dict[str, Any]:
    """Collect metadata and persist a GGUF manifest entry."""
    metadata, layer_info, max_context = await collect_gguf_runtime_metadata(huggingface_id, file_path)
    filename = os.path.basename(file_path)
    return record_gguf_download(
        huggingface_id=huggingface_id or "unknown",
        filename=filename,
        file_path=file_path,
        file_size=file_size,
        metadata=metadata,
        layer_info=layer_info,
        max_context_length=max_context,
        model_id=model_id
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
    if normalized.startswith("../") or normalized.startswith("..\\") or normalized.startswith("/"):
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
    re.compile(r'iQ\d+_K_[A-Z]+'),  # iQ3_K_S style
    re.compile(r'iQ\d+_\d+'),       # iQ4_0 style
    re.compile(r'iQ\d+_K'),         # iQ6_K style
    re.compile(r'iQ\d+'),           # iQ3 style (fallback)
    re.compile(r'IQ\d+_[A-Z]+'),    # IQ1_S, IQ2_M, etc.
    re.compile(r'Q\d+_K_[A-Z]+'),   # Q4_K_M, Q5_K_S, etc.
    re.compile(r'Q\d+_\d+'),        # Q4_0, Q5_1, etc.
    re.compile(r'Q\d+_K'),          # Q2_K, Q6_K, etc.
    re.compile(r'Q\d+'),            # Q3, Q4, etc. (fallback)
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
    return getattr(hf_api, 'token', None)


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


async def search_models(query: str, limit: int = 20, model_format: str = "gguf") -> List[Dict]:
    """Search HuggingFace for GGUF or safetensors models."""
    try:
        model_format = (model_format or "gguf").lower()
        if model_format not in MODEL_FORMATS:
            raise ValueError(f"Unsupported model format '{model_format}'. Must be one of {MODEL_FORMATS}")

        # Check cache first
        cache_key = f"{model_format}:{query.lower()}_{limit}"
        current_time = time.time()
        
        if cache_key in _search_cache:
            cached_data, cache_time = _search_cache[cache_key]
            if current_time - cache_time < _cache_timeout:
                logger.info(f"Returning cached results for '{query}'")
                return cached_data[:limit]  # Return only requested limit
        
        logger.info(f"Searching for models with query: '{query}', limit: {limit}, format: {model_format}")
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
        
        # Use real HuggingFace API search with expand parameter for rich metadata
        filter_value = "gguf" if model_format == "gguf" else "safetensors"

        models_generator = list_models(
            search=query,
            limit=min(limit * 2, 50),  # Get more models to filter from
            sort="downloads",
            direction=-1,
            filter=filter_value,
            expand=["author", "cardData", "siblings"]  # Ensure author is present, plus metadata
        )
        
        # Convert generator to list
        models = list(models_generator)
        logger.info(f"Found {len(models)} models from HuggingFace API with expanded metadata")
        
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


async def _process_models_parallel(models: List, limit: int, model_format: str, max_concurrent: int = 5) -> List[Dict]:
    """Process models in parallel with semaphore for concurrency control"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_model(model):
        async with semaphore:
            return await _process_single_model(model, model_format)
    
    # Create tasks for all models
    tasks = [process_model(model) for model in models[:limit * 2]]
    
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
    
    return valid_results[:limit]


async def _process_single_model(model, model_format: str) -> Optional[Dict]:
    """Process a single model and extract all metadata"""
    try:
        logger.info(f"Processing model: {model.id}")
        
        quantizations: Dict[str, Dict] = {}
        safetensors_files: List[Dict] = []
        repo_files: List[Dict[str, Any]] = []

        if hasattr(model, 'siblings') and model.siblings:
            if model_format == "gguf":
                # Group GGUF files by logical quantization, handling multi-part shards
                # Accept both plain `.gguf` and multi-part patterns like `.gguf.part1of2`
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
                    # Normalize filename by stripping shard suffix patterns like:
                    #   -00001-of-00002.gguf (TheBloke-style)
                    #   .gguf.part1of2 (Hugging Face-style multi-part)
                    base_for_quant = re.sub(r'-\d{5}-of-\d{5}(?=\.gguf$)', '', filename)
                    base_for_quant = re.sub(r'\.gguf\.part\d+of\d+$', '.gguf', base_for_quant)
                    quantization = _extract_quantization(base_for_quant)
                    if quantization == "unknown":
                        continue

                    # Detect optional variant prefix immediately before the quantization (e.g. "i1-" in "i1-IQ3_M")
                    variant_prefix = ""
                    try:
                        prefix_match = re.search(rf"(i\d+)-{re.escape(quantization)}", base_for_quant)
                        if prefix_match:
                            variant_prefix = prefix_match.group(1)
                    except Exception:
                        variant_prefix = ""

                    # Use full variant-aware key so that different variants (e.g. "i1-Q4_K_M"
                    # vs "Q4_K_M") are treated as distinct quantizations everywhere.
                    quant_key = f"{variant_prefix}-{quantization}" if variant_prefix else quantization

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
                    entry["size_mb"] = round(entry["total_size"] / (1024 * 1024), 2) if entry["total_size"] else 0.0

                # If no quantizations were detected after grouping, skip this model
                if not quantizations:
                    return None
            else:
                safetensors_files = []
                for sibling in model.siblings:
                    filename = sibling.rfilename
                    size_bytes = getattr(sibling, 'size', 0) or 0
                    repo_files.append({
                        "filename": filename,
                        "is_safetensors": filename.endswith('.safetensors')
                    })
                    if not filename.endswith('.safetensors'):
                        continue
                    safetensors_files.append({
                        "filename": filename
                    })

                logger.info(f"Model {model.id}: {len(safetensors_files)} safetensors files found")
                if not safetensors_files:
                    return None
        else:
            return None

        # Extract rich metadata from model and cardData
        metadata = _extract_model_metadata(model)
        
        result = {
            "id": model.id,
            "name": getattr(model, 'modelId', model.id),  # Use modelId if available, fallback to id
            "author": getattr(model, 'author', ''),
            "downloads": model.downloads,
            "likes": getattr(model, 'likes', 0),
            "tags": model.tags or [],
            "model_format": model_format,
            "quantizations": quantizations if model_format == "gguf" else {},
            "safetensors_files": safetensors_files if model_format == "safetensors" else [],
            "repo_files": repo_files if model_format == "safetensors" else [],
            **metadata  # Include all extracted metadata
        }
        
        logger.info(f"Added model {model.id} to results")
        return result
        
    except Exception as e:
        logger.error(f"Error processing model {model.id}: {e}")
        return None


def _extract_model_metadata(model) -> Dict:
    """Extract rich metadata from ModelInfo and cardData"""
    metadata = {
        "description": "",
        "license": "",
        "pipeline_tag": getattr(model, 'pipeline_tag', ''),
        "library_name": getattr(model, 'library_name', ''),
        "language": [],
        "base_model": "",
        "architecture": "",
        "parameters": "",
        "context_length": None,
        "gated": getattr(model, 'gated', False),
        "private": getattr(model, 'private', False),
        "readme_url": f"https://huggingface.co/{model.id}",
        "created_at": getattr(model, 'createdAt', None),
        "updated_at": getattr(model, 'lastModified', None),
        "safetensors": {}
    }
    
    # Extract from cardData if available
    if hasattr(model, 'cardData') and model.cardData:
        card_data = model.cardData
        
        # Ensure card_data is not None and is a dict
        if card_data and isinstance(card_data, dict):
            # Extract basic info
            metadata["license"] = card_data.get('license', '')
            language_data = card_data.get('language', [])
            # Ensure language is always an array
            metadata["language"] = language_data if isinstance(language_data, list) else []
            metadata["base_model"] = card_data.get('base_model', '')
            
            # Extract from model_index if available
            model_index = card_data.get('model-index', [])
            if model_index:
                for item in model_index:
                    if isinstance(item, dict):
                        # Extract architecture
                        if 'name' in item:
                            metadata["architecture"] = item['name']
                        
                        # Extract parameters
                        if 'params' in item:
                            metadata["parameters"] = str(item['params'])
                        
                        # Extract context length
                        if 'context_length' in item:
                            metadata["context_length"] = item['context_length']
    
    # Extract model size from filename if not found in cardData
    if not metadata["parameters"]:
        # Try to extract model size from modelId using regex
        import re
        model_id = getattr(model, 'modelId', model.id)
        size_match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_id)
        if size_match:
            metadata["parameters"] = f"{size_match.group(1)}B"
    
    # Extract safetensors metadata from siblings
    if hasattr(model, 'siblings') and model.siblings:
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
        "total_size": 0
    }
    
    if not siblings:
        return safetensors_info
    
    safetensors_files = []
    total_size = 0
    
    for sibling in siblings:
        if sibling.rfilename.endswith('.safetensors'):
            safetensors_files.append({
                "filename": sibling.rfilename
            })
            total_size += sibling.size or 0
    
    if safetensors_files:
        safetensors_info.update({
            "has_safetensors": True,
            "safetensors_files": safetensors_files,
            "total_size": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2)
        })
    
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
    
    if not hf_get_safetensors_metadata and not hasattr(hf_api, "get_safetensors_metadata"):
        raise RuntimeError("Safetensors metadata is not supported by the installed huggingface_hub version")
    
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
        if "hf_transfer" in error_msg.lower() or "HF_HUB_ENABLE_HF_TRANSFER" in error_msg:
            logger.warning(f"hf_transfer not available for {model_id}, falling back to standard download. Error: {err}")
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
                logger.error(f"Failed to fetch safetensors metadata for {model_id} even after disabling hf_transfer: {retry_err}")
                raise RuntimeError(f"Safetensors metadata is not available: {retry_err}")
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
            logger.warning(f"No files_metadata found in safetensors metadata for {model_id}")
            return {
                "repo_id": model_id,
                "total_files": 0,
                "total_tensors": 0,
                "dtype_totals": {},
                "files": [],
                "cached_at": datetime.utcnow().isoformat(),
                "error": "No safetensors files found"
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
            tensor_details.append({
                "name": tensor_name,
                "dtype": _get_attr_or_key(tensor_info, "dtype", "unknown"),
                "shape": _get_attr_or_key(tensor_info, "shape", []),
            })
        
        dtype_counts = {}
        if isinstance(parameter_count, dict):
            for dtype, count in parameter_count.items():
                dtype_counts[dtype] = count
                dtype_totals[dtype] = dtype_totals.get(dtype, 0) + count
        elif hasattr(parameter_count, 'items'):
            # Handle object with items() method
            for dtype, count in parameter_count.items():
                dtype_counts[dtype] = count
                dtype_totals[dtype] = dtype_totals.get(dtype, 0) + count
        
        total_tensors += len(tensor_details)
        files_summary.append({
            "filename": filename,
            "tensor_count": len(tensor_details),
            "dtype_counts": dtype_counts,
            "tensors": tensor_details
        })
    
    summary = {
        "repo_id": model_id,
        "total_files": len(files_summary),
        "total_tensors": total_tensors,
        "dtype_totals": dtype_totals,
        "files": files_summary,
        "cached_at": datetime.utcnow().isoformat()
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
            "name": getattr(model_info, 'modelId', model_info.id),  # Use modelId if available, fallback to id
            "author": getattr(model_info, 'author', ''),
            "downloads": model_info.downloads,
            "likes": getattr(model_info, 'likes', 0),
            "tags": model_info.tags or [],
            **metadata
        }
        
        # Try to get config.json for architecture details
        try:
            config_files = [s for s in model_info.siblings if s.rfilename == 'config.json']
            if config_files:
                # Download and parse config.json
                config_path = hf_hub_download(
                    repo_id=model_id,
                    filename='config.json',
                    local_dir="data/temp",
                    local_dir_use_symlinks=False
                )
                
                with open(config_path, 'r', encoding='utf-8') as f:
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


async def download_model(huggingface_id: str, filename: str, model_format: str = "gguf") -> tuple[str, int]:
    """Download model from HuggingFace"""
    try:
        models_dir = _get_download_directory(model_format, huggingface_id)
        
        # Sanitize filename
        filename = _sanitize_filename(filename)

        # Download the file
        file_path = hf_hub_download(
            repo_id=huggingface_id,
            filename=filename,
            local_dir=models_dir,
            local_dir_use_symlinks=False
        )
        
        # Get file size
        file_size = os.path.getsize(file_path)
        
        return file_path, file_size
        
    except Exception as e:
        logger.error(f"Failed to download model: {e}")
        raise


async def download_model_with_websocket_progress(
    huggingface_id: str,
    filename: str,
    websocket_manager,
    task_id: str,
    total_bytes: int = 0,
    model_format: str = "gguf",
    huggingface_id_for_progress: str = None
):
    """Download model with WebSocket progress updates by tracking filesystem size"""
    import asyncio
    import time
    
    logger.info(f"=== DOWNLOAD PROGRESS START ===")
    logger.info(f"Download task: {task_id}")
    logger.info(f"HuggingFace ID: {huggingface_id}")
    logger.info(f"Filename: {filename}")
    logger.info(f"Total bytes from search: {total_bytes}")
    logger.info(f"WebSocket manager: {websocket_manager}")
    logger.info(f"Active connections: {len(websocket_manager.active_connections)}")
    
    try:
        models_dir = _get_download_directory(model_format, huggingface_id)
        
        # Sanitize filename and build path
        filename = _sanitize_filename(filename)
        file_path = os.path.join(models_dir, filename)
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        
        # Send initial progress
        logger.info(f"Sending initial progress message...")
        progress_hf_id = huggingface_id_for_progress or huggingface_id
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=0,
            message=f"Starting download of {filename}",
            bytes_downloaded=0,
            total_bytes=total_bytes,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename,
            model_format=model_format,
            huggingface_id=progress_hf_id
        )
        logger.info(f"Initial progress message sent")
        
        # Get file size from HuggingFace API if not provided
        if total_bytes == 0:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                file_info = api.repo_file_info(repo_id=huggingface_id, path=filename)
                total_bytes = file_info.size
                logger.info(f"Got file size from HuggingFace API: {total_bytes}")
            except Exception as e:
                logger.warning(f"Could not get file size from HuggingFace API: {e}")
                # If we can't get the size, we'll estimate it
                total_bytes = 0
        
        # Send total size update
        if total_bytes > 0:
            await websocket_manager.send_download_progress(
                task_id=task_id,
                progress=0,
                message=f"Downloading {filename}",
                bytes_downloaded=0,
                total_bytes=total_bytes,
                speed_mbps=0,
                eta_seconds=0,
                filename=filename,
                model_format=model_format,
                huggingface_id=progress_hf_id
            )
        
        # Start the download with built-in progress tracking
        logger.info(f"🚀 Starting download with built-in progress tracking...")
        
        file_path, file_size = await download_with_progress_tracking(
            huggingface_id, filename, file_path, models_dir,
            websocket_manager, task_id, total_bytes, model_format, progress_hf_id
        )
        
        # Send final completion
        await websocket_manager.send_download_progress(
            task_id=task_id,
            progress=100,
            message=f"Download completed: {filename}",
            bytes_downloaded=file_size,
            total_bytes=file_size,
            speed_mbps=0,
            eta_seconds=0,
            filename=filename,
            model_format=model_format,
            huggingface_id=progress_hf_id
        )
        
        return file_path, file_size
        
    except Exception as e:
        # Send error notification
        if websocket_manager and task_id:
            progress_hf_id = huggingface_id_for_progress or huggingface_id
            await websocket_manager.send_download_progress(
                task_id=task_id,
                progress=0,
                message=f"Download failed: {str(e)}",
                bytes_downloaded=0,
                total_bytes=0,
                speed_mbps=0,
                eta_seconds=0,
                filename=filename,
                model_format=model_format,
                huggingface_id=progress_hf_id
            )
            await websocket_manager.send_notification(
                "error", "Download Failed", f"Failed to download {filename}: {str(e)}", task_id
            )
        raise


async def download_with_progress_tracking(
    huggingface_id: str,
    filename: str,
    file_path: str,
    models_dir: str,
    websocket_manager,
    task_id: str,
    total_bytes: int,
    model_format: str,
    huggingface_id_for_progress: str = None
):
    """Download the file using custom http_get method with progress tracking"""
    try:
        import aiofiles
        
        logger.info(f"📁 Starting download of {filename} ({total_bytes} bytes) [{model_format}]")
        
        # Use the standard HuggingFace resolve URL (this is the default/preferred method)
        safe_filename = _sanitize_filename(filename)
        download_url = f"https://huggingface.co/{huggingface_id}/resolve/main/{safe_filename}"
        actual_file_size = total_bytes  # Start with the provided size
        
        # Optionally get exact file size from HuggingFace API
        try:
            api = HfApi()
            file_info = api.repo_file_info(repo_id=huggingface_id, filename=safe_filename)
            if hasattr(file_info, 'size') and file_info.size:
                actual_file_size = file_info.size
                logger.info(f"📊 Got file size from HuggingFace API: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
        except Exception as e:
            logger.debug(f"Could not get file size from API: {e}, using provided size: {total_bytes}")
        
        logger.info(f"📁 Download URL: {download_url}")
        
        # Build headers manually
        hf_headers = {
            "User-Agent": "llama-cpp-studio/1.0.0",
            "Accept": "*/*",
            "Accept-Encoding": "gzip, deflate",
        }
        
        # Create final destination path
        final_path = os.path.join(models_dir, safe_filename)
        final_dir = os.path.dirname(final_path)
        if final_dir and not os.path.exists(final_dir):
            os.makedirs(final_dir, exist_ok=True)
        
        # Custom progress bar that sends WebSocket updates
        progress_hf_id = huggingface_id_for_progress or huggingface_id
        class WebSocketProgressBar(tqdm):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.websocket_manager = websocket_manager
                self.task_id = task_id
                self.filename = filename
                self.huggingface_id = progress_hf_id
                self.start_time = time.time()
                self.last_update_time = self.start_time
            
            def update(self, n=1):
                super().update(n)
                # Send WebSocket update with current progress
                current_time = time.time()
                if current_time - self.last_update_time >= 0.5:  # Update every 0.5 seconds
                    if self.total > 0:
                        progress = int((self.n / self.total) * 100)
                        current_bytes = int(self.n)
                        
                        # Calculate speed and ETA
                        elapsed_time = current_time - self.start_time
                        speed_bytes_per_sec = current_bytes / elapsed_time if elapsed_time > 0 else 0
                        speed_mbps = speed_bytes_per_sec / (1024 * 1024)
                        
                        remaining_bytes = self.total - self.n
                        eta_seconds = int(remaining_bytes / speed_bytes_per_sec) if speed_bytes_per_sec > 0 else 0
                        
                        logger.debug(f"📊 Progress: {progress}% ({current_bytes}/{self.total} bytes) - {speed_mbps:.1f} MB/s")
                        
                        # Send WebSocket update
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                asyncio.create_task(self.websocket_manager.send_download_progress(
                                    task_id=self.task_id,
                                    progress=progress,
                                    message=f"Downloading {self.filename}",
                                    bytes_downloaded=current_bytes,
                                    total_bytes=self.total,
                                    speed_mbps=speed_mbps,
                                    eta_seconds=eta_seconds,
                                    filename=self.filename,
                                    model_format=model_format,
                                    huggingface_id=self.huggingface_id
                                ))
                        except Exception as e:
                            logger.error(f"Error sending progress update: {e}")
                        
                        self.last_update_time = current_time
        
        # Create our custom progress bar
        custom_progress_bar = WebSocketProgressBar(
            desc=safe_filename,
            total=actual_file_size,  # Use the actual file size
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            disable=False
        )
        
        # Download using aiohttp with timeout and our custom progress bar
        timeout = aiohttp.ClientTimeout(total=3600, connect=30)  # 1 hour total, 30s connect
        async with aiohttp.ClientSession(headers=hf_headers, timeout=timeout) as session:
            async with session.get(download_url) as response:
                if response.status != 200:
                    raise Exception(f"Failed to download: HTTP {response.status}")
                
                # Get actual file size from response headers
                content_length = response.headers.get('content-length')
                if content_length:
                    response_size = int(content_length)
                    if response_size != actual_file_size:
                        logger.debug(f"📏 Size difference: API said {actual_file_size}, response says {response_size} (diff: {abs(response_size - actual_file_size)} bytes)")
                        # Use the response size as it's more accurate
                        actual_file_size = response_size
                        custom_progress_bar.total = actual_file_size
                        logger.info(f"📊 Using response size: {actual_file_size} bytes ({actual_file_size / (1024*1024):.2f} MB)")
                
                # Download with progress tracking
                # Use 64KB chunks for better performance with large files
                chunk_size = 65536
                downloaded_bytes = 0
                async with aiofiles.open(final_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(chunk_size):
                        await f.write(chunk)
                        downloaded_bytes += len(chunk)
                        custom_progress_bar.update(len(chunk))
        
        # Close the progress bar
        custom_progress_bar.close()
        
        logger.info(f"📁 Downloaded to: {final_path}")
        
        # Validate downloaded file size
        file_size = os.path.getsize(final_path)
        if actual_file_size and actual_file_size > 0 and file_size != actual_file_size:
            logger.warning(f"⚠️ Download size mismatch: expected {actual_file_size}, got {file_size}")
            # Allow small differences (like metadata)
            if abs(file_size - actual_file_size) > 1024:  # More than 1KB difference
                raise Exception(f"Download incomplete: expected {actual_file_size} bytes, got {file_size} bytes")
        
        return final_path, file_size
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Traceback:\n{traceback.format_exc()}")
        raise


async def get_quantization_sizes_from_hf(huggingface_id: str, quantizations: Dict[str, Dict]) -> Dict[str, Dict]:
    """Return actual file sizes for provided quantizations using Hugging Face Hub API.
    Uses the shared hf_api instance and mirrors logic used elsewhere in this module.
    """
    try:
        # Prefer fetching only required files to reduce payload.
        # Support both legacy single-file structure and new multi-file bundles.
        all_filenames: List[str] = []
        quant_to_files: Dict[str, List[str]] = {}

        for quant_name, quant_data in (quantizations or {}).items():
            if not isinstance(quant_data, dict):
                continue
            files = quant_data.get("files")
            if isinstance(files, list) and files:
                paths = [f.get("filename") for f in files if isinstance(f, dict) and f.get("filename")]
            else:
                # Backward compatibility: single filename field
                single = quant_data.get("filename")
                paths = [single] if single else []

            paths = [p for p in paths if p]
            if not paths:
                continue
            quant_to_files[quant_name] = paths
            all_filenames.extend(paths)

        updated: Dict[str, Dict] = {}

        if all_filenames:
            try:
                # Newer API: batch query specific paths for metadata
                paths_info = hf_api.get_paths_info(repo_id=huggingface_id, paths=all_filenames)
                # Build lookup
                file_sizes: Dict[str, Optional[int]] = {pi.path: getattr(pi, "size", None) for pi in paths_info}
            except Exception as batch_err:
                logger.warning(f"get_paths_info failed for {huggingface_id}: {batch_err}")
                # Fallback: fetch full metadata once
                model_info = hf_api.model_info(repo_id=huggingface_id, files_metadata=True)
                file_sizes = {}
                if hasattr(model_info, "siblings") and model_info.siblings:
                    for sibling in model_info.siblings:
                        file_sizes[sibling.rfilename] = getattr(sibling, "size", None)

            for quant_name, filenames in quant_to_files.items():
                files_with_sizes = []
                total_size = 0
                for filename in filenames:
                    actual_size = file_sizes.get(filename)
                    if not actual_size or actual_size <= 0:
                        try:
                            file_info = hf_api.repo_file_info(repo_id=huggingface_id, path=filename)
                            actual_size = getattr(file_info, "size", None)
                        except Exception as file_err:
                            logger.warning(f"repo_file_info failed for {huggingface_id}/{filename}: {file_err}")
                            actual_size = None
                    if actual_size and actual_size > 0:
                        total_size += actual_size
                        size_value = actual_size
                    else:
                        logger.warning(f"Unable to determine size for {huggingface_id}/{filename}")
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
                        "size_mb": round(total_size / (1024 * 1024), 2) if total_size else 0.0,
                    }

        return updated
    except Exception as e:
        logger.error(f"Failed to fetch quantization sizes for {huggingface_id}: {e}")
        return {}