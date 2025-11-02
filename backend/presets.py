from typing import Dict, Any, Tuple
import os

from backend.gguf_reader import get_model_layer_info
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _detect_architecture_from_name(model_name: str) -> str:
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


def get_architecture_and_presets(model) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Source of truth for presets. Returns (architecture, presets dict).
    Presets include keys like temp, top_p, top_k, repeat_penalty.
    """
    # Import normalize_architecture from model_metadata to ensure consistency
    from backend.smart_auto.architecture_config import normalize_architecture, detect_architecture_from_name
    
    # Try GGUF metadata
    architecture = "unknown"
    try:
        if model.file_path and os.path.exists(model.file_path):
            layer_info = get_model_layer_info(model.file_path)
            if layer_info:
                raw_architecture = layer_info.get("architecture", "")
                architecture = normalize_architecture(raw_architecture)
                if architecture != "unknown" and raw_architecture != architecture:
                    logger.debug(f"Normalized architecture for presets: '{raw_architecture}' -> '{architecture}'")
    except Exception as e:
        logger.warning(f"Failed to get layer info for presets: {e}")

    # Fallback to name-based detection if architecture is still unknown or empty
    if not architecture or architecture == "unknown":
        architecture = detect_architecture_from_name(model.name)
        if architecture != "unknown":
            logger.debug(f"Detected architecture from name for presets: '{architecture}'")

    # Defaults
    presets: Dict[str, Dict[str, Any]] = {
        "coding": {},
        "conversational": {},
    }

    model_lower = (model.name or "").lower()
    is_coding_model = "code" in model_lower or architecture in ["codellama", "deepseek"]

    if architecture in ["glm", "glm4"]:
        presets["coding"] = {"temp": 1.0, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05}
        presets["conversational"] = {"temp": 1.0, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1}
    elif architecture in ["deepseek", "deepseek-v3"]:
        presets["coding"] = {"temp": 1.0, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05}
        presets["conversational"] = {"temp": 0.9, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1}
    elif architecture in ["qwen", "qwen2", "qwen3"]:
        presets["coding"] = {"temp": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
        presets["conversational"] = {"temp": 0.7, "top_p": 0.8, "top_k": 20, "repeat_penalty": 1.05}
    elif architecture in ["gemma", "gemma3"]:
        presets["coding"] = {"temp": 0.9, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05}
        presets["conversational"] = {"temp": 0.9, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1}
    elif is_coding_model:
        presets["coding"] = {"temp": 0.1, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.05}
        presets["conversational"] = {"temp": 0.7, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1}
    else:
        presets["coding"] = {"temp": 0.7, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1}
        presets["conversational"] = {"temp": 0.8, "top_p": 0.95, "top_k": 40, "repeat_penalty": 1.1}

    return architecture, presets


