"""
Calculation utilities for smart_auto module.
Pure functions for batch size, context size, and GPU layer calculations.
"""
from functools import lru_cache
from typing import Tuple, Optional
from .constants import (
    VRAM_FRAGMENTATION_MARGIN,
    CONTEXT_SAFETY_MARGIN,
    LAYERS_PER_GB_SMALL_MODEL,
    LAYERS_PER_GB_LARGE_MODEL,
    GPU_LAYER_BUFFER,
    CONTEXT_RAM_OVERHEAD_GB,
    MIN_CONTEXT_SIZE,
    MAX_CONTEXT_SIZE,
    MIN_BATCH_SIZE,
    ARCHITECTURE_CPU_BATCH_LIMITS,
)


def calculate_ubatch_size(batch_size: int) -> int:
    """
    Calculate optimal ubatch_size from batch_size.
    
    Unified helper to derive ubatch_size consistently across GPU and CPU modes.
    """
    return max(1, min(batch_size, max(1, batch_size // 2)))


def calculate_optimal_batch_size_gpu(
    available_vram_gb: float,
    model_size_mb: float,
    context_size: int,
    embedding_length: int,
    layer_count: int,
    cache_type_k: Optional[str] = None,
    cache_type_v: Optional[str] = None
) -> int:
    """
    Calculate optimal batch size for GPU based on memory requirements.
    
    Uses data-driven approach when possible, falls back to VRAM-based estimation.
    """
    # Memory requirements per batch item
    model_memory_gb = model_size_mb / 1024
    
    # KV cache memory per batch item
    # Note: KV cache is shared across batch items in continuous batching, but we estimate
    # as if each item needs its own context window (conservative estimate)
    if embedding_length > 0 and layer_count > 0:
        # Use actual quantization bytes-per-value if provided, otherwise default to fp16
        from .constants import KV_CACHE_QUANT_FACTORS
        quant_factor_k = KV_CACHE_QUANT_FACTORS.get(cache_type_k or "f16", 0.5)  # f16 = 0.5
        quant_factor_v = KV_CACHE_QUANT_FACTORS.get(cache_type_v or cache_type_k or "f16", quant_factor_k)
        bytes_per_k = quant_factor_k * 4  # Convert factor to bytes (f32=4, f16=2, etc.)
        bytes_per_v = quant_factor_v * 4
        bytes_per_element = (bytes_per_k + bytes_per_v) / 2  # Average for K+V
        # Conservative estimate: use embedding_length directly (overestimates slightly for GQA)
        # This is a simplified calculation for batch sizing - precise GQA calculation done in memory_estimator
        kv_cache_per_item_gb = (context_size * embedding_length * layer_count * bytes_per_element) / (1024**3)
    else:
        # Conservative estimate: 64 bytes per token
        kv_cache_per_item_gb = context_size * 64 / (1024**3)
    
    total_per_item_gb = model_memory_gb + kv_cache_per_item_gb
    
    if total_per_item_gb <= 0:
        return MIN_BATCH_SIZE
    
    # Calculate max batch size based on available memory
    max_batch_size = int(available_vram_gb * VRAM_FRAGMENTATION_MARGIN / total_per_item_gb)
    
    # Apply reasonable limits based on model size
    if embedding_length > 2048:  # Large models (7B+)
        max_batch_size = min(max_batch_size, 512)
    elif embedding_length > 1024:  # Medium models (3B-7B)
        max_batch_size = min(max_batch_size, 1024)
    else:  # Small models (<3B)
        max_batch_size = min(max_batch_size, 2048)
    
    return max(MIN_BATCH_SIZE, max_batch_size)


def calculate_optimal_batch_size_cpu(
    available_ram_gb: float,
    model_size_mb: float,
    context_size: int,
    architecture: str
) -> Tuple[int, int]:
    """
    Calculate optimal batch sizes for CPU mode using dict-based architecture profiles.
    
    Returns:
        Tuple of (batch_size, ubatch_size)
    """
    model_ram_gb = model_size_mb / 1024
    
    # Calculate available RAM for batching after model and context
    reserved_ram_gb = model_ram_gb + (context_size / 1000) + CONTEXT_RAM_OVERHEAD_GB
    available_for_batch = max(0, available_ram_gb - reserved_ram_gb)
    
    # Estimate batch memory usage (rough: 1MB per batch item)
    max_batch_size = int(available_for_batch * 1000)  # 1GB = ~1000 batch items
    
    # Get architecture-specific limits or use defaults
    limits = ARCHITECTURE_CPU_BATCH_LIMITS.get(architecture, ARCHITECTURE_CPU_BATCH_LIMITS["default"])
    
    batch_size = min(limits["max_batch"], max(limits["min_batch"], max_batch_size))
    ubatch_size = min(limits["max_ubatch"], max(limits["min_ubatch"], batch_size // 2))
    
    return batch_size, ubatch_size


@lru_cache(maxsize=128)
def calculate_max_context_size_gpu(
    available_vram_gb: float,
    model_size_mb: float,
    layer_count: int,
    embedding_length: int,
    attention_head_count: int,
    attention_head_count_kv: int,
    cache_type_k: Optional[str] = None,
    cache_type_v: Optional[str] = None,
    usage_mode: str = "single_user"
) -> int:
    """
    Calculate maximum context size for GPU based on memory requirements.
    
    Cached with LRU to avoid redundant calculations for same parameters.
    
    Returns:
        Maximum context size in tokens
    """
    # Reserve memory for model
    model_memory_gb = model_size_mb / 1024
    reserved_memory_gb = model_memory_gb + 1.0  # Model + 1GB overhead
    available_for_context_gb = max(0, available_vram_gb - reserved_memory_gb)
    
    if available_for_context_gb <= 0:
        return MIN_CONTEXT_SIZE
    
    # Calculate KV cache memory per token based on transformer architecture
    # GQA-aware formula: M_kv = n_ctx × N_layers × N_head_kv × d_head × (p_a_k + p_a_v)
    # where d_head = N_embd / N_head
    # Use actual quantization bytes-per-value instead of hardcoded fp16
    if embedding_length > 0 and layer_count > 0:
        # Get actual quantization bytes-per-value
        from .constants import KV_CACHE_QUANT_FACTORS
        quant_factor_k = KV_CACHE_QUANT_FACTORS.get(cache_type_k or "f16", 0.5)  # f16 = 0.5
        quant_factor_v = KV_CACHE_QUANT_FACTORS.get(cache_type_v or cache_type_k or "f16", quant_factor_k)
        bytes_per_k = quant_factor_k * 4  # Convert factor to bytes (f32=4, f16=2, etc.)
        bytes_per_v = quant_factor_v * 4
        
        if attention_head_count_kv > 0 and attention_head_count > 0:
            # GQA-aware calculation
            d_head = embedding_length / attention_head_count
            # KV cache per token: K and V cache per layer, each storing N_head_kv heads
            kv_cache_per_layer_k = attention_head_count_kv * d_head * bytes_per_k
            kv_cache_per_layer_v = attention_head_count_kv * d_head * bytes_per_v
            kv_cache_per_token_bytes = (kv_cache_per_layer_k + kv_cache_per_layer_v) * layer_count
        else:
            # Fallback for non-GQA models (MHA: N_head_kv = N_head)
            kv_cache_per_token_bytes = layer_count * embedding_length * (bytes_per_k + bytes_per_v)
        
        # Apply usage mode factor for multi_user (allows larger context since KV cache is lower)
        # For max context calculation: n_ctx = available_vram / (kv_cache_per_token * usage_factor)
        # So: tokens_per_gb = 1GB / (kv_cache_per_token * usage_factor)
        from .constants import KV_CACHE_SINGLE_USER_FACTOR, KV_CACHE_MULTI_USER_FACTOR
        if usage_mode == "multi_user":
            # In multi_user mode, KV cache usage is lower (typical usage), so we can fit more context
            usage_factor = KV_CACHE_MULTI_USER_FACTOR
            # Calculate tokens per GB: divide by (bytes_per_token * usage_factor)
            # This gives more tokens since usage_factor < 1.0
            tokens_per_gb = (1024**3) / (kv_cache_per_token_bytes * usage_factor) if kv_cache_per_token_bytes > 0 else 0
        else:
            # Single user mode: full KV cache (peak usage), standard calculation
            usage_factor = KV_CACHE_SINGLE_USER_FACTOR
            tokens_per_gb = (1024**3) / (kv_cache_per_token_bytes * usage_factor) if kv_cache_per_token_bytes > 0 else 0
        
        # Calculate max context size with safety margin
        if tokens_per_gb > 0:
            max_context_tokens = int(available_for_context_gb * tokens_per_gb * CONTEXT_SAFETY_MARGIN)
            # Ensure minimum context size
            return max(MIN_CONTEXT_SIZE, max_context_tokens)
        else:
            # Fallback if calculation fails (e.g., kv_cache_per_token_bytes is 0)
            return MIN_CONTEXT_SIZE
    else:
        # Fallback to conservative estimate: ~1000 tokens per GB
        estimated = int(available_for_context_gb * 1000)
        return max(MIN_CONTEXT_SIZE, min(MAX_CONTEXT_SIZE, estimated))


def calculate_optimal_context_size_gpu(
    architecture: str,
    available_vram: int,
    model_size_mb: float = 0,
    layer_count: int = 32,
    embedding_length: int = 0,
    attention_head_count: int = 0,
    attention_head_count_kv: int = 0,
    base_context: Optional[int] = None,
    cache_type_k: Optional[str] = None,
    cache_type_v: Optional[str] = None,
    usage_mode: str = "single_user"
) -> int:
    """
    Calculate optimal context size for GPU based on VRAM and architecture defaults.
    
    Returns:
        Optimal context size in tokens
    """
    from .architecture_config import get_architecture_default_context
    
    base_ctx = base_context or get_architecture_default_context(architecture)
    
    if available_vram == 0:
        # CPU mode - conservative context
        return max(MIN_CONTEXT_SIZE, min(base_ctx, 2048))
    
    # Use data-driven calculation if we have model parameters
    if model_size_mb > 0 and layer_count > 0 and embedding_length > 0:
        vram_gb = available_vram / (1024**3)
        calculated_max = calculate_max_context_size_gpu(
            vram_gb, model_size_mb, layer_count, embedding_length,
            attention_head_count, attention_head_count_kv,
            cache_type_k=cache_type_k, cache_type_v=cache_type_v,
            usage_mode=usage_mode
        )
        result = min(base_ctx, calculated_max) if calculated_max > 0 else base_ctx
        return max(MIN_CONTEXT_SIZE, min(result, MAX_CONTEXT_SIZE))
    
    # Fallback to architecture-based limits if no model data
    vram_gb = available_vram / (1024**3)
    
    # Conservative scaling based on VRAM capacity
    if vram_gb >= 24:    # High-end GPU
        return max(MIN_CONTEXT_SIZE, min(base_ctx, MAX_CONTEXT_SIZE))
    elif vram_gb >= 12:   # Mid-range GPU
        return max(MIN_CONTEXT_SIZE, min(base_ctx, int(base_ctx * 0.75)))
    elif vram_gb >= 8:    # Lower-end GPU
        return max(MIN_CONTEXT_SIZE, min(base_ctx, int(base_ctx * 0.5)))
    else:                 # Very limited VRAM
        return max(MIN_CONTEXT_SIZE, min(base_ctx, 2048))


def calculate_optimal_gpu_layers(
    free_vram_gb: float,
    model_size_mb: float,
    total_layers: int,
    context_size: int = 4096,
    cache_type_k: Optional[str] = None,
    cache_type_v: Optional[str] = None,
    ubatch_size: int = 512,
    attention_head_count: int = 0,
    attention_head_count_kv: int = 0,
    embedding_length: int = 0,
    layer_count: int = 0,
    usage_mode: str = "single_user"
) -> int:
    """
    Calculate optimal number of layers to offload to GPU.
    
    Uses exact M_kv and M_compute calculations according to theoretical model:
    n_ngl_max = floor((VRAM_available - M_kv - M_compute) / (M_weights_total / N_layers))
    
    Args:
        free_vram_gb: Available VRAM in GB
        model_size_mb: Model size in MB (GGUF file size)
        total_layers: Total number of layers in model
        context_size: Context size in tokens (default: 4096)
        cache_type_k: K cache quantization type (default: f16)
        cache_type_v: V cache quantization type (default: same as cache_type_k)
        ubatch_size: Micro-batch size (default: 512)
        attention_head_count: Number of attention heads (for GQA calculation)
        attention_head_count_kv: Number of KV attention heads (for GQA calculation)
        embedding_length: Embedding dimension (for GQA calculation)
        layer_count: Layer count (alias for total_layers, for compatibility)
    
    Returns:
        Number of GPU layers
    """
    # Use total_layers if provided, otherwise layer_count
    actual_layer_count = total_layers if total_layers > 0 else (layer_count if layer_count > 0 else 0)
    
    if actual_layer_count == 0:
        # Fallback to old heuristic if layer count unknown
        estimated_layers_per_gb = LAYERS_PER_GB_SMALL_MODEL if model_size_mb < 1000 else LAYERS_PER_GB_LARGE_MODEL
        max_layers = int(free_vram_gb * estimated_layers_per_gb * GPU_LAYER_BUFFER)
        return max_layers
    
    # Calculate exact M_kv and M_compute
    free_vram_bytes = free_vram_gb * (1024**3)
    model_size_bytes = model_size_mb * (1024**2)
    
    # Calculate M_kv using exact formula
    from .memory_estimator import calculate_kv_cache_size
    # Use default values if not provided
    cache_type_k_actual = cache_type_k or "f16"
    cache_type_v_actual = cache_type_v or cache_type_k_actual
    
    # If we have architecture parameters, use precise calculation
    if embedding_length > 0 and attention_head_count > 0:
        kv_cache_bytes = calculate_kv_cache_size(
            context_size, 1,  # parallel=1 for layer calculation
            actual_layer_count, embedding_length,
            attention_head_count, attention_head_count_kv or attention_head_count,
            cache_type_k_actual, cache_type_v_actual if cache_type_v else None,
            usage_mode=usage_mode
        )
    else:
        # Fallback: estimate KV cache size (conservative)
        # Assume fp16, use embedding_length if available, otherwise estimate
        if embedding_length > 0:
            # Simplified estimate: assume MHA (not GQA)
            bytes_per_token = actual_layer_count * embedding_length * 4  # K+V at fp16
            kv_cache_bytes = context_size * bytes_per_token
        else:
            # Very conservative fallback: ~64 bytes per token per layer
            kv_cache_bytes = context_size * actual_layer_count * 64
    
    # Calculate M_compute: Fixed overhead + variable scratch buffer
    from .constants import COMPUTE_FIXED_OVERHEAD_MB, COMPUTE_SCRATCH_PER_UBATCH_MB
    compute_overhead_mb = COMPUTE_FIXED_OVERHEAD_MB + (ubatch_size * COMPUTE_SCRATCH_PER_UBATCH_MB)
    compute_overhead_bytes = int(compute_overhead_mb * (1024**2))
    
    # Formula from theoretical model:
    # n_ngl_max = floor((VRAM_available - M_kv - M_compute) / (M_weights_total / N_layers))
    available_for_weights_bytes = free_vram_bytes - kv_cache_bytes - compute_overhead_bytes
    
    if available_for_weights_bytes <= 0:
        # Not enough VRAM even for M_kv and M_compute
        return 0
    
    mb_per_layer = model_size_bytes / actual_layer_count if actual_layer_count > 0 else 0
    if mb_per_layer <= 0:
        # Fallback if calculation fails
        estimated_layers_per_gb = LAYERS_PER_GB_SMALL_MODEL if model_size_mb < 1000 else LAYERS_PER_GB_LARGE_MODEL
        max_layers = int(free_vram_gb * estimated_layers_per_gb * GPU_LAYER_BUFFER)
        return min(max_layers, actual_layer_count)
    
    max_layers = int(available_for_weights_bytes / mb_per_layer) if mb_per_layer > 0 else 0
    
    return min(max_layers, actual_layer_count)

