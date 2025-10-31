from typing import Dict, Any
import os

from backend.logging_config import get_logger
from backend.gguf_reader import get_model_layer_info

logger = get_logger(__name__)


def normalize_architecture(architecture: str) -> str:
    """
    Normalize architecture string from GGUF metadata to recognized architecture names.
    Examples: "qwen2.5" -> "qwen2", "qwen3-moe" -> "qwen3", "llama-3" -> "llama3"
    """
    if not architecture or not architecture.strip():
        return "unknown"
    
    arch_lower = architecture.lower().strip()
    
    # Qwen architectures
    if "qwen" in arch_lower:
        if "qwen3" in arch_lower or "qwen-3" in arch_lower:
            return "qwen3"
        elif "qwen2" in arch_lower or "qwen-2" in arch_lower:
            return "qwen2"
        return "qwen"
    
    # Llama architectures
    if "llama" in arch_lower:
        if "codellama" in arch_lower:
            return "codellama"
        elif "llama3" in arch_lower or "llama-3" in arch_lower:
            return "llama3"
        elif "llama2" in arch_lower or "llama-2" in arch_lower:
            return "llama2"
        return "llama"
    
    # Gemma architectures
    if "gemma" in arch_lower:
        if "gemma3" in arch_lower or "gemma-3" in arch_lower:
            return "gemma3"
        return "gemma"
    
    # GLM architectures
    if "glm" in arch_lower or "chatglm" in arch_lower:
        if "glm-4" in arch_lower or "glm4" in arch_lower:
            return "glm4"
        return "glm"
    
    # DeepSeek architectures
    if "deepseek" in arch_lower:
        if "v3" in arch_lower or "v3.1" in arch_lower:
            return "deepseek-v3"
        return "deepseek"
    
    # Other architectures
    if "mistral" in arch_lower:
        return "mistral"
    if "phi" in arch_lower:
        return "phi"
    
    # If architecture contains something but not recognized, return as-is for now
    # but log a warning
    if arch_lower and arch_lower != "unknown":
        logger.debug(f"Unrecognized architecture format: {architecture}, returning as-is")
        return arch_lower
    
    return "unknown"


def detect_architecture_from_name(model_name: str) -> str:
    """Detect model architecture from model name"""
    name = (model_name or "").lower()
    
    if "llama" in name:
        if "codellama" in name:
            return "codellama"
        elif "llama3" in name or "llama-3" in name:
            return "llama3"
        elif "llama2" in name or "llama-2" in name:
            return "llama2"
        return "llama"
    elif "mistral" in name:
        return "mistral"
    elif "phi" in name:
        return "phi"
    elif "glm" in name or "chatglm" in name:
        if "glm-4" in name or "glm4" in name:
            return "glm4"
        return "glm"
    elif "deepseek" in name:
        if "v3" in name or "v3.1" in name:
            return "deepseek-v3"
        return "deepseek"
    elif "qwen" in name:
        if "qwen3" in name or "qwen-3" in name:
            return "qwen3"
        elif "qwen2" in name or "qwen-2" in name:
            return "qwen2"
        return "qwen"
    elif "gemma" in name:
        if "gemma3" in name or "gemma-3" in name:
            return "gemma3"
        return "gemma"
    
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
                
                # Normalize architecture from GGUF metadata
                raw_architecture = meta.get("architecture", "")
                normalized = normalize_architecture(raw_architecture)
                meta["architecture"] = normalized
                
                if normalized != "unknown" and raw_architecture != normalized:
                    logger.debug(f"Normalized architecture: '{raw_architecture}' -> '{normalized}'")
    except Exception as e:
        logger.warning(f"Failed to read GGUF metadata: {e}")

    # Fallback to name-based detection if architecture is still unknown or empty
    current_arch = meta.get("architecture", "").strip()
    if not current_arch or current_arch == "unknown":
        detected = detect_architecture_from_name(getattr(model, "name", ""))
        meta["architecture"] = detected
        if detected != "unknown":
            logger.debug(f"Detected architecture from model name: '{detected}'")

    return meta


