"""
Constants used across the smart_auto module.
Centralizes magic numbers and configuration limits.
"""

from typing import Dict, Any

# ============================================================================
# Memory optimization factors
# ============================================================================

KV_CACHE_OPTIMIZATION_FACTOR = 1.0  # Use actual memory (no optimization factor) - memory mapping doesn't reduce peak usage

# Usage mode factors for KV cache estimation
# Based on theoretical model: single_user accumulates context (peak), multi_user clears context (typical usage)
KV_CACHE_SINGLE_USER_FACTOR = 1.0  # Peak estimate (full context window)
KV_CACHE_MULTI_USER_FACTOR = (
    0.4  # Typical usage (context cleared between requests, ~40% of peak)
)

MOE_OFFLOAD_ALL_RATIO = 0.3  # 30% of model for all MoE offloaded
MOE_OFFLOAD_UP_DOWN_RATIO = 0.2  # 20% of model for up/down MoE offloaded
MOE_OFFLOAD_UP_RATIO = 0.1  # 10% of model for up MoE offloaded

LLAMA_CPP_OVERHEAD_MB = 256  # 256MB overhead for llama.cpp

# Compute buffer constants (M_compute)
COMPUTE_FIXED_OVERHEAD_MB = (
    550  # Fixed CUDA overhead (~550MB for CUDA context, cuBLAS workspace, etc.)
)
COMPUTE_SCRATCH_PER_UBATCH_MB = (
    0.5  # Variable scratch buffer per ubatch size (rough estimate)
)

# VRAM pressure thresholds for MoE offloading
VRAM_RATIO_VERY_TIGHT = 1.2  # Very tight VRAM - offload all MoE
VRAM_RATIO_TIGHT = 1.5  # Tight VRAM - offload up/down projections
VRAM_RATIO_MODERATE = 2.0  # Moderate VRAM - offload only up projection

# ============================================================================
# KV Cache quantization factors
# ============================================================================

KV_CACHE_QUANT_FACTORS: Dict[str, float] = {
    "f32": 1.0,  # Full precision (no reduction)
    "f16": 0.5,  # Half precision
    "bf16": 0.5,  # Bfloat16
    "q8_0": 0.25,  # 8-bit quant
    "q5_1": 0.156,  # 5-bit high quality
    "q5_0": 0.156,  # 5-bit
    "q4_1": 0.125,  # 4-bit high quality
    "q4_0": 0.125,  # 4-bit
    "iq4_nl": 0.125,  # 4-bit non-linear
}

QUANTIZATION_AVERAGE_FACTOR = 0.5  # Average of K and V cache quantization factors

# ============================================================================
# Architecture context defaults
# ============================================================================

ARCHITECTURE_CONTEXT_DEFAULTS: Dict[str, int] = {
    "llama2": 4096,
    "llama3": 8192,
    "llama": 4096,
    "codellama": 16384,
    "mistral": 32768,
    "phi": 2048,
    "glm": 8192,
    "glm4": 204800,  # 200K context for GLM-4.6
    "deepseek": 32768,
    "deepseek-v3": 32768,
    "qwen": 32768,  # 32K context
    "qwen2": 32768,  # 32K context
    "qwen3": 131072,  # 128K context for Qwen3
    "gemma": 8192,
    "gemma3": 8192,
    "generic": 4096,
}

DEFAULT_CONTEXT_LENGTH = 4096

# ============================================================================
# Memory calculation defaults
# ============================================================================

DEFAULT_BYTES_PER_ELEMENT = 2  # Assume fp16 for activations
BATCH_INTERMEDIATE_FACTOR = 0.08  # 8% factor for intermediate activations
BATCH_QKV_FACTOR = 0.04  # 4% factor for QKV projections
BATCH_COMPUTATION_OVERHEAD_KB = 400  # ~400KB per batch item
BATCH_FALLBACK_MB = 1.5  # 1.5MB per batch item fallback

BATCH_VRAM_OVERHEAD_RATIO = 0.1  # 10% of KV cache VRAM for batch overhead
BATCH_RAM_OVERHEAD_RATIO = 0.1  # 10% of KV cache RAM for batch overhead

# Layer estimation defaults
FALLBACK_LAYER_COUNT = 32
FALLBACK_EMBEDDING_LENGTH = 4096
FALLBACK_KV_CACHE_PER_TOKEN_BYTES = 60 * (4096 * 2 + 4096 * 2)  # ~960 KB per token

# ============================================================================
# Context size limits
# ============================================================================

MIN_CONTEXT_SIZE = 512
MAX_CONTEXT_SIZE = 262144  # 256K
MAX_CPU_CONTEXT_SIZE = 8192  # Conservative limit for CPU mode

# ============================================================================
# Batch size limits
# ============================================================================

MIN_BATCH_SIZE = 1
MAX_BATCH_SIZE = 4096

# ============================================================================
# GPU/VRAM calculation constants
# ============================================================================

# VRAM safety margins
VRAM_SAFETY_MARGIN = 0.9  # 90% of available VRAM
VRAM_FRAGMENTATION_MARGIN = 0.7  # 70% for batch size calculations
CONTEXT_SAFETY_MARGIN = 0.8  # 80% for context size calculations

# GPU layer estimation
LAYERS_PER_GB_SMALL_MODEL = 8  # Models < 1GB
LAYERS_PER_GB_LARGE_MODEL = 4  # Models >= 1GB
GPU_LAYER_BUFFER = 0.8  # Leave 20% buffer

# ============================================================================
# CPU calculation constants
# ============================================================================

# RAM reservation overhead
MODEL_RAM_OVERHEAD_GB = 2.0  # Overhead for model loading
CONTEXT_RAM_OVERHEAD_GB = 1.0  # Additional overhead for context

# ============================================================================
# Architecture-specific configuration profiles
# ============================================================================

# CPU architecture optimization profiles
# Maps architecture to dict of optimization settings
ARCHITECTURE_CPU_PROFILES: Dict[str, Dict[str, Any]] = {
    "mistral": {
        "use_mmap": True,
    },
    "llama3": {
        "use_mmap": "dynamic",  # Special flag for conditional mmap
    },
    "llama2": {
        "use_mmap": "dynamic",  # Special flag for conditional mmap
    },
    "codellama": {
        "use_mmap": True,
        "logits_all": False,
    },
    "phi": {
        "use_mmap": True,
    },
}

# CPU batch size limits per architecture
ARCHITECTURE_CPU_BATCH_LIMITS: Dict[str, Dict[str, int]] = {
    "mistral": {
        "max_batch": 2048,
        "max_ubatch": 1024,
        "min_batch": 64,
        "min_ubatch": 32,
    },
    "llama3": {"max_batch": 1536, "max_ubatch": 768, "min_batch": 64, "min_ubatch": 32},
    "codellama": {
        "max_batch": 1536,
        "max_ubatch": 768,
        "min_batch": 64,
        "min_ubatch": 32,
    },
    "default": {
        "max_batch": 1024,
        "max_ubatch": 512,
        "min_batch": 32,
        "min_ubatch": 16,
    },
}
