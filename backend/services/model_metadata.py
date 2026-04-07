"""Model metadata, embedding heuristics, and GGUF refresh (no HTTP)."""

from __future__ import annotations

import os
import re
import time
from typing import Any, Dict, Optional

from backend.gguf_reader import get_model_layer_info
from backend.gpu_detector import get_gpu_info
from backend.logging_config import get_logger
from backend.model_config import effective_model_config_from_raw

logger = get_logger(__name__)

# Lightweight cache for GPU info to avoid repeated NVML calls during rapid estimate requests
_gpu_info_cache: Dict[str, Any] = {"data": None, "timestamp": 0.0}
GPU_INFO_CACHE_TTL = 2.0  # seconds


async def get_cached_gpu_info() -> Dict[str, Any]:
    """Return cached GPU info when available to reduce NVML overhead."""
    now = time.monotonic()
    cached = _gpu_info_cache["data"]
    if cached is not None and now - _gpu_info_cache["timestamp"] < GPU_INFO_CACHE_TTL:
        return cached

    data = await get_gpu_info()
    _gpu_info_cache["data"] = data
    _gpu_info_cache["timestamp"] = now
    return data


EMBEDDING_PIPELINE_TAGS = frozenset(
    {
        "text-embedding",
        "feature-extraction",
        "sentence-similarity",
    }
)
EMBEDDING_KEYWORDS = [
    "embedding",
    "embed-",
    "text-embedding",
    "feature-extraction",
    "nomic",
    "gte-",
    "e5-",
    "bge-",
    "snowflake-arctic-embed",
    "minilm",
]


def looks_like_embedding_model(
    pipeline_tag: Optional[str], *name_parts: Optional[str]
) -> bool:
    """Detect embedding-capable models based on pipeline metadata or name heuristics."""
    pipeline = (pipeline_tag or "").lower()
    if pipeline in EMBEDDING_PIPELINE_TAGS:
        return True
    combined = " ".join(part for part in name_parts if part).lower()
    if not combined:
        return False
    return any(keyword in combined for keyword in EMBEDDING_KEYWORDS)


def model_is_embedding(model: dict) -> bool:
    """Determine if a stored model should run in embedding mode."""
    config = effective_model_config_from_raw(model.get("config"))
    if config.get("embedding"):
        return True
    return looks_like_embedding_model(
        model.get("pipeline_tag"),
        model.get("huggingface_id"),
        model.get("display_name") or model.get("name"),
        model.get("base_model_name"),
    )


def normalize_architecture(raw_architecture: str) -> str:
    """Normalize GGUF architecture string (stub after smart_auto removal)."""
    if not raw_architecture or not isinstance(raw_architecture, str):
        return "unknown"
    return raw_architecture.strip() or "unknown"


def detect_architecture_from_name(name: str) -> str:
    """Infer architecture from model name (stub after smart_auto removal)."""
    if not name or not isinstance(name, str):
        return "unknown"
    name_lower = name.lower()
    if "llama" in name_lower:
        return "llama"
    if "qwen" in name_lower:
        return "qwen2"
    if "mistral" in name_lower:
        return "mistral"
    if "phi" in name_lower:
        return "phi-2"
    return "unknown"


def extract_model_type(filename: str) -> str:
    """Extract model type from filename."""
    filename_lower = filename.lower()
    if "llama" in filename_lower:
        return "llama"
    if "mistral" in filename_lower:
        return "mistral"
    if "codellama" in filename_lower:
        return "codellama"
    if "gemma" in filename_lower:
        return "gemma"
    if "qwen" in filename_lower:
        return "qwen"
    return "unknown"


def extract_base_model_name(filename: str) -> str:
    """Extract base model name from filename by removing quantization."""
    name = filename.replace(".gguf", "").replace(".safetensors", "")
    quantization_patterns = [
        r"IQ\d+_[A-Z]+",
        r"Q\d+_K_[A-Z]+",
        r"Q\d+_[A-Z]+",
        r"Q\d+[K_]?[A-Z]*",
        r"Q\d+",
    ]
    for pattern in quantization_patterns:
        name = re.sub(pattern, "", name)
    name = name.rstrip("._")
    return name if name else filename


def refresh_gguf_model_metadata(model: dict, store, gguf_path: str) -> Dict[str, Any]:
    """
    Re-read GGUF metadata from a concrete on-disk file (HF cache path) and update the store.
    """
    normalized_path = gguf_path.replace("\\", "/")
    if not normalized_path or not os.path.exists(normalized_path):
        raise FileNotFoundError("Model file not found on disk")

    layer_info = get_model_layer_info(normalized_path)
    if not layer_info:
        raise ValueError("Failed to read model metadata from file")

    raw_architecture = layer_info.get("architecture", "")
    normalized_architecture = normalize_architecture(raw_architecture)
    if not normalized_architecture or normalized_architecture == "unknown":
        normalized_architecture = detect_architecture_from_name(
            model.get("display_name")
            or model.get("name")
            or model.get("huggingface_id")
            or ""
        )

    update_fields: Dict[str, Any] = {}
    if (
        normalized_architecture
        and normalized_architecture != "unknown"
        and normalized_architecture != model.get("model_type")
    ):
        update_fields["model_type"] = normalized_architecture

    if update_fields:
        store.update_model(model["id"], update_fields)

    return {
        "updated_fields": update_fields,
        "metadata": {
            "architecture": normalized_architecture,
            "layer_count": layer_info.get("layer_count", 0),
            "context_length": layer_info.get("context_length", 0),
            "parameter_count": layer_info.get("parameter_count"),
            "vocab_size": layer_info.get("vocab_size", 0),
            "embedding_length": layer_info.get("embedding_length", 0),
            "attention_head_count": layer_info.get("attention_head_count", 0),
            "attention_head_count_kv": layer_info.get("attention_head_count_kv", 0),
            "block_count": layer_info.get("block_count", 0),
            "is_moe": layer_info.get("is_moe", False),
            "expert_count": layer_info.get("expert_count", 0),
            "experts_used_count": layer_info.get("experts_used_count", 0),
        },
    }
