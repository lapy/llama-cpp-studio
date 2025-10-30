from typing import Dict, Any, Tuple
import os

from backend.gguf_reader import get_model_layer_info
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _detect_architecture_from_name(model_name: str) -> str:
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


def get_architecture_and_presets(model) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    Source of truth for presets. Returns (architecture, presets dict).
    Presets include keys like temp, top_p, top_k, repeat_penalty.
    """
    # Try GGUF metadata
    architecture = "unknown"
    try:
        if model.file_path and os.path.exists(model.file_path):
            layer_info = get_model_layer_info(model.file_path)
            if layer_info:
                architecture = layer_info.get("architecture", "unknown")
    except Exception as e:
        logger.warning(f"Failed to get layer info for presets: {e}")

    if architecture == "unknown":
        architecture = _detect_architecture_from_name(model.name)

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


