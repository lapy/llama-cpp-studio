"""
Architecture configuration and detection module.
Consolidates architecture detection and default configuration values.
"""
from functools import lru_cache
from .constants import ARCHITECTURE_CONTEXT_DEFAULTS, DEFAULT_CONTEXT_LENGTH


@lru_cache(maxsize=128)
def resolve_architecture(architecture_or_name: str) -> str:
    """
    Unified function to resolve architecture from either a model name or architecture string.
    
    Handles both detection from model names and normalization of GGUF metadata.
    This replaces the separate detect_architecture_from_name() and normalize_architecture() functions.
    
    Args:
        architecture_or_name: Either a model name or architecture string from GGUF metadata
        
    Returns:
        Normalized architecture name (e.g., "llama3", "qwen3", etc.)
    """
    if not architecture_or_name or not architecture_or_name.strip():
        return "unknown"
    
    # Normalize input
    text = architecture_or_name.lower().strip()
    
    # Check architectures in order of specificity (most specific first)
    
    # Qwen architectures
    if "qwen" in text:
        if "qwen3" in text or "qwen-3" in text:
            return "qwen3"
        if "qwen2" in text or "qwen-2" in text:
            return "qwen2"
        return "qwen"
    
    # Llama architectures (CodeLlama before other Llama variants)
    if "codellama" in text:
        return "codellama"
    if "llama3" in text or "llama-3" in text:
        return "llama3"
    if "llama2" in text or "llama-2" in text:
        return "llama2"
    if "llama" in text:
        return "llama"
    
    # Gemma architectures
    if "gemma3" in text or "gemma-3" in text:
        return "gemma3"
    if "gemma" in text:
        return "gemma"
    
    # GLM architectures
    if "glm-4" in text or "glm4" in text:
        return "glm4"
    if "glm" in text or "chatglm" in text:
        return "glm"
    
    # DeepSeek architectures
    if "deepseek" in text:
        if "v3" in text or "v3.1" in text:
            return "deepseek-v3"
        return "deepseek"
    
    # Other architectures
    if "mistral" in text:
        return "mistral"
    if "phi" in text:
        return "phi"
    
    # If text contains something but not recognized, return as generic for model names
    # or unknown for invalid architecture strings
    if text and text not in ["unknown", "generic"]:
        return "generic"
    
    return "unknown"


def get_architecture_default_context(architecture: str) -> int:
    """
    Get default context length for an architecture.
    
    Args:
        architecture: Normalized architecture name
        
    Returns:
        Default context length in tokens
    """
    return ARCHITECTURE_CONTEXT_DEFAULTS.get(architecture, DEFAULT_CONTEXT_LENGTH)


# Backward compatibility aliases
def detect_architecture_from_name(model_name: str) -> str:
    """Deprecated: Use resolve_architecture() instead."""
    return resolve_architecture(model_name)


def normalize_architecture(architecture: str) -> str:
    """Deprecated: Use resolve_architecture() instead."""
    return resolve_architecture(architecture)

