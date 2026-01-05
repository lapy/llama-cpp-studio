"""
MoE (Mixture of Experts) handling module.
Handles MoE model offloading patterns and architecture-specific flags.
"""
from typing import Dict, Any, Tuple
from backend.logging_config import get_logger
from .constants import VRAM_RATIO_VERY_TIGHT, VRAM_RATIO_TIGHT, VRAM_RATIO_MODERATE

logger = get_logger(__name__)


# MoE offload strategies: (vram_ratio_threshold, pattern)
MOE_OFFLOAD_STRATEGIES: list[Tuple[float, str]] = [
    (VRAM_RATIO_VERY_TIGHT, ".ffn_.*_exps.=CPU"),      # Very tight: all MoE offloaded
    (VRAM_RATIO_TIGHT, ".ffn_(up|down)_exps.=CPU"),    # Tight: up/down offloaded
    (VRAM_RATIO_MODERATE, ".ffn_(up)_exps.=CPU"),      # Moderate: only up offloaded
    (float('inf'), "")                                  # Ample: no offloading
]


def generate_moe_offload_pattern(architecture: str, available_vram_gb: float, 
                                 model_size_mb: float, is_moe: bool = False, 
                                 expert_count: int = 0) -> str:
    """Generate optimal MoE offloading pattern based on VRAM availability
    
    Returns regex pattern for the -ot (offload type) parameter to control MoE layer placement
    """
    if not is_moe or expert_count == 0:
        return ""  # No MoE offloading for non-MoE models
    
    model_size_gb = model_size_mb / 1024
    
    # Calculate VRAM pressure
    vram_ratio = available_vram_gb / model_size_gb if model_size_gb > 0 else 1.0
    
    # Find the appropriate strategy based on VRAM ratio
    for threshold, pattern in MOE_OFFLOAD_STRATEGIES:
        if vram_ratio < threshold:
            return pattern
    
    return ""  # Fallback: no offloading needed


def needs_jinja_template(architecture: str, layer_info: Dict[str, Any]) -> bool:
    """Determine if architecture requires jinja template."""
    # GLM architectures always need jinja
    if architecture in ["glm", "glm4"]:
        return True
    # Qwen3 coder variants need jinja
    if architecture == "qwen3":
        arch_str = layer_info.get('architecture', '').lower()
        if "coder" in arch_str:
            return True
    return False


def get_architecture_specific_flags(architecture: str, layer_info: Dict[str, Any]) -> Dict[str, Any]:
    """Get architecture-specific flags and settings.
    
    Returns dict with flags like jinja, moe_offload_custom, etc.
    """
    flags = {"jinja": False, "moe_offload_custom": ""}
    
    # Check jinja requirement
    if needs_jinja_template(architecture, layer_info):
        flags["jinja"] = True
        logger.info(f"{architecture} architecture detected - enabling jinja template")
    
    # Generate MoE offloading pattern if applicable
    is_moe = layer_info.get('is_moe', False)
    available_vram_gb = layer_info.get('available_vram_gb', 0)
    
    if not is_moe or available_vram_gb == 0:
        if is_moe and available_vram_gb == 0:
            logger.debug("MoE model detected but no GPU available - MoE layers will run on CPU")
        return flags
    
    # Generate MoE offload pattern for GPU mode
    expert_count = layer_info.get('expert_count', 0)
    model_size_mb = layer_info.get('model_size_mb', 0)
    moe_pattern = generate_moe_offload_pattern(architecture, available_vram_gb, model_size_mb, is_moe, expert_count)
    
    if moe_pattern:
        flags["moe_offload_custom"] = moe_pattern
        logger.debug(f"Generated MoE offload pattern: {moe_pattern}")
    
    return flags

