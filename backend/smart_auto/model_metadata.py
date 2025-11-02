from typing import Dict, Any
import os
from functools import lru_cache

from backend.logging_config import get_logger
from backend.gguf_reader import get_model_layer_info
from .architecture_config import resolve_architecture, get_architecture_default_context
from .models import ModelMetadata

logger = get_logger(__name__)


@lru_cache(maxsize=256)
def _get_layer_info_from_file(file_path: str, mtime: float) -> Dict[str, Any]:
    """
    Get layer info from GGUF file with LRU caching.
    Uses mtime as part of cache key for invalidation.
    """
    try:
        return get_model_layer_info(file_path) or {}
    except Exception as e:
        logger.warning(f"Failed to read layer info from {file_path}: {e}")
        return {}


@lru_cache(maxsize=64)
def _estimate_layer_count_cached(model_name: str) -> int:
    """Cached version of layer count estimation from model name."""
    if "7b" in model_name or "7B" in model_name:
        return 32
    elif "3b" in model_name or "3B" in model_name:
        return 28
    elif "1b" in model_name or "1B" in model_name:
        return 22
    elif "13b" in model_name or "13B" in model_name:
        return 40
    elif "30b" in model_name or "30B" in model_name:
        return 60
    elif "65b" in model_name or "65B" in model_name:
        return 80
    else:
        return 32  # Default fallback


def get_model_metadata(model) -> ModelMetadata:
    """
    Get comprehensive model metadata with caching.
    
    Uses LRU cache with mtime-based invalidation to prevent redundant file I/O.
    This is the single source of truth for model layer information.
    """
    # Default metadata structure
    meta: Dict[str, Any] = {
        "layer_count": 32,
        "architecture": "unknown",
        "context_length": 0,
        "vocab_size": 0,
        "embedding_length": 0,
        "attention_head_count": 0,
        "attention_head_count_kv": 0,
        "block_count": 0,
        "is_moe": False,
        "expert_count": 0,
        "experts_used_count": 0,
    }

    try:
        if model.file_path and os.path.exists(model.file_path):
            # Use LRU cache with mtime-based invalidation
            mtime = os.path.getmtime(model.file_path)
            layer_info = _get_layer_info_from_file(model.file_path, mtime)
            if layer_info:
                meta.update(layer_info)
            
            # Resolve architecture from GGUF metadata
            raw_architecture = meta.get("architecture", "")
            normalized = resolve_architecture(raw_architecture)
            meta["architecture"] = normalized
            
            if normalized not in ("unknown", "generic") and raw_architecture != normalized:
                logger.debug(f"Resolved architecture: '{raw_architecture}' -> '{normalized}'")
    except Exception as e:
        logger.warning(f"Failed to read GGUF metadata for model {getattr(model, 'id', 'unknown')}: {e}")

    # Fallback to name-based detection if architecture is still unknown
    current_arch = meta.get("architecture", "").strip()
    if not current_arch or current_arch == "unknown":
        detected = resolve_architecture(getattr(model, "name", ""))
        meta["architecture"] = detected
        if detected not in ("unknown", "generic"):
            logger.debug(f"Detected architecture from model name: '{detected}'")
    
    # Fix context_length if it's 0 by using architecture default
    if meta.get("context_length", 0) == 0 and meta["architecture"] != "unknown":
        meta["context_length"] = get_architecture_default_context(meta["architecture"])
    
    # Fallback to name-based layer count estimation if needed
    if meta.get("layer_count", 0) == 32 and current_arch == "unknown":
        model_name = getattr(model, "name", "").lower()
        meta["layer_count"] = _estimate_layer_count_cached(model_name)

    # Return as ModelMetadata dataclass
    return ModelMetadata.from_dict(meta)


