"""
Configuration builder module.
Handles configuration sanitization, server parameters, and preset tuning.
"""
from typing import Dict, Any, Optional
from backend.logging_config import get_logger

logger = get_logger(__name__)


def clamp_int(name: str, val: Any, lo: int, hi: int, default: int) -> int:
    """Helper to clamp integer values."""
    try:
        iv = int(val)
    except (ValueError, TypeError):
        iv = default
    return max(lo, min(hi, iv))


def generate_server_params() -> Dict[str, Any]:
    """Generate server-specific parameters"""
    return {
        "host": "0.0.0.0",  # Allow external connections
        "timeout": 300  # 5 minutes timeout
    }


def sanitize_config(config: Dict[str, Any], gpu_count: int) -> Dict[str, Any]:
    """Clamp and sanitize final config values to enforce invariants and avoid edge-case crashes."""
    sanitized = dict(config)
    
    # Clamp integer values
    sanitized["ctx_size"] = clamp_int("ctx_size", sanitized.get("ctx_size", 4096), 512, 262144, 4096)
    sanitized["batch_size"] = clamp_int("batch_size", sanitized.get("batch_size", 512), 1, 4096, 512)
    sanitized["ubatch_size"] = clamp_int("ubatch_size", sanitized.get("ubatch_size"), 1, sanitized.get("batch_size", 512), max(1, sanitized.get("batch_size", 512)//2))
    sanitized["parallel"] = clamp_int("parallel", sanitized.get("parallel", 1), 1, max(1, gpu_count if gpu_count > 0 else 1), 1)
    
    # Ensure boolean fields are properly typed
    boolean_fields = ["no_mmap", "mlock", "low_vram", "logits_all", "flash_attn"]
    sanitized.update({b: bool(sanitized[b]) for b in boolean_fields if b in sanitized})
    
    return sanitized


def apply_preset_tuning(config: Dict[str, Any], preset_name: str) -> None:
    """
    Apply preset-specific tuning to configuration parameters.
    
    Consolidates both generation parameter adjustments and config factor tuning
    into a single clear function.
    """
    if preset_name == "coding":
        config["temperature"] = 0.7
        config["repeat_penalty"] = 1.05
        if "batch_size" in config:
            config["batch_size"] = max(1, int(config["batch_size"] * 0.8))
        if "ubatch_size" in config:
            config["ubatch_size"] = max(1, int(config["ubatch_size"] * 0.8))
        if "parallel" in config:
            config["parallel"] = max(1, int(config["parallel"] * 1.2))
        logger.debug("Applied preset 'coding' tuning")
    # conversational preset has no changes (factors = 1.0), so skip

