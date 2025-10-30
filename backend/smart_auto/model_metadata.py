from typing import Dict, Any
import os

from backend.logging_config import get_logger
from backend.gguf_reader import get_model_layer_info

logger = get_logger(__name__)


def detect_architecture_from_name(model_name: str) -> str:
    name = (model_name or "").lower()
    if "qwen3" in name or "qwen-3" in name or "qwen" in name:
        return "qwen3"
    if "gemma-3" in name or "gemma3" in name or "gemma" in name:
        return "gemma3"
    if "deepseek" in name:
        return "deepseek"
    if "glm" in name:
        return "glm"
    if "llama" in name:
        return "llama"
    if "mistral" in name:
        return "mistral"
    return "unknown"


def get_model_metadata(model) -> Dict[str, Any]:
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
            li = get_model_layer_info(model.file_path)
            if li:
                meta.update(li)
    except Exception as e:
        logger.warning(f"Failed to read GGUF metadata: {e}")

    if meta.get("architecture", "unknown") == "unknown":
        meta["architecture"] = detect_architecture_from_name(getattr(model, "name", ""))

    return meta


