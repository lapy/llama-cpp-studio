"""Model metadata, embedding heuristics, and GGUF refresh (no HTTP)."""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any, Dict, Optional

from backend.gguf_reader import get_model_layer_info
from backend.gpu_detector import get_gpu_info, probe_gpu_list
from backend.logging_config import get_logger
from backend.model_config import effective_model_config_from_raw

logger = get_logger(__name__)

# Lightweight cache for GPU info to avoid repeated NVML calls during rapid estimate requests
_gpu_info_cache: Dict[str, Any] = {
    "data": None,
    "timestamp": 0.0,
    "refresh_task": None,
}
_gpu_info_lock: Optional[asyncio.Lock] = None
GPU_INFO_CACHE_TTL = 30.0  # seconds — shared by /api/gpu-info and VRAM estimates
GPU_INFO_INITIAL_WAIT_SECONDS = float(
    os.environ.get("GPU_INFO_INITIAL_WAIT_SECONDS", "0.25")
)

# Startup cache for GET /api/gpu-list (index + name only; no per-request probing)
_gpu_list_cache: Optional[Dict[str, Any]] = None


def _gpu_info_fallback(
    *, detecting: bool = False, error: Optional[str] = None
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "vendor": None,
        "cuda_version": "Unknown",
        "device_count": 0,
        "gpus": [],
        "total_vram": 0,
        "available_vram": 0,
        "cpu_only_mode": True,
    }
    if detecting:
        payload["detecting"] = True
    if error:
        payload["error"] = error
    return payload


def _get_gpu_info_lock() -> asyncio.Lock:
    global _gpu_info_lock
    if _gpu_info_lock is None:
        _gpu_info_lock = asyncio.Lock()
    return _gpu_info_lock


async def _refresh_gpu_info_cache() -> Dict[str, Any]:
    try:
        data = await get_gpu_info()
    except Exception as exc:
        logger.warning("GPU detection refresh failed: %s", exc)
        data = _gpu_info_fallback(error=str(exc))
    _gpu_info_cache["data"] = data
    _gpu_info_cache["timestamp"] = time.monotonic()
    return data


async def warm_gpu_list_cache() -> Dict[str, Any]:
    """Probe GPUs once at startup and cache the lightweight list."""
    global _gpu_list_cache
    try:
        _gpu_list_cache = await probe_gpu_list()
    except Exception as exc:
        logger.warning("Startup GPU list probe failed: %s", exc)
        _gpu_list_cache = {
            "vendor": None,
            "device_count": 0,
            "gpus": [],
            "cpu_only_mode": True,
            "error": str(exc),
        }
    logger.info(
        "GPU list cache warmed: vendor=%s device_count=%s",
        _gpu_list_cache.get("vendor"),
        _gpu_list_cache.get("device_count"),
    )
    return _gpu_list_cache


def get_startup_gpu_list() -> Dict[str, Any]:
    """Return the startup GPU list cache (instant; no hardware probe)."""
    if _gpu_list_cache is not None:
        return _gpu_list_cache
    return {
        "vendor": None,
        "device_count": 0,
        "gpus": [],
        "cpu_only_mode": True,
    }


async def get_cached_gpu_info(
    *, initial_wait_seconds: float = GPU_INFO_INITIAL_WAIT_SECONDS
) -> Dict[str, Any]:
    """Return GPU info without letting slow hardware probes block page load."""
    now = time.monotonic()
    cached = _gpu_info_cache["data"]
    if cached is not None and now - _gpu_info_cache["timestamp"] < GPU_INFO_CACHE_TTL:
        return cached

    async with _get_gpu_info_lock():
        now = time.monotonic()
        cached = _gpu_info_cache["data"]
        if (
            cached is not None
            and now - _gpu_info_cache["timestamp"] < GPU_INFO_CACHE_TTL
        ):
            return cached

        task = _gpu_info_cache.get("refresh_task")
        if task is None or task.done():
            task = asyncio.create_task(_refresh_gpu_info_cache())
            _gpu_info_cache["refresh_task"] = task

    if cached is not None:
        return cached

    try:
        return await asyncio.wait_for(
            asyncio.shield(task), timeout=max(0.0, initial_wait_seconds)
        )
    except asyncio.TimeoutError:
        return _gpu_info_fallback(detecting=True)


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

    context_length = layer_info.get("context_length")
    if isinstance(context_length, (int, float)) and context_length > 0:
        if int(context_length) != model.get("max_context_length"):
            update_fields["max_context_length"] = int(context_length)

    layer_count = layer_info.get("layer_count")
    if isinstance(layer_count, (int, float)) and layer_count > 0:
        if int(layer_count) != model.get("layer_count"):
            update_fields["layer_count"] = int(layer_count)

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
