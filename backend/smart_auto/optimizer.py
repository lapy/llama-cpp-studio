"""
Joint optimization algorithm for llama.cpp configuration.

Implements the theoretical model algorithm from Section III-B that jointly
optimizes (n_ngl, n_ctx, ubatch_size) given VRAM and RAM constraints.

Prioritizes "Full Offload" (Max_Speed) regime before falling back to "Hybrid" mode.
"""
from typing import Dict, Any, Optional, Tuple
from .memory_estimator import calculate_kv_cache_size
from .constants import (
    COMPUTE_FIXED_OVERHEAD_MB,
    COMPUTE_SCRATCH_PER_UBATCH_MB,
    MIN_CONTEXT_SIZE,
    MIN_BATCH_SIZE
)


def find_optimal_config(
    model_size_bytes: int,
    total_layers: int,
    embedding_length: int,
    attention_head_count: int,
    attention_head_count_kv: int,
    available_vram_bytes: int,
    available_ram_bytes: int,
    cache_type_k: str = "f16",
    cache_type_v: Optional[str] = None,
    ubatch_size: int = 512,
    desired_performance: str = "Max_Speed",
    min_context_size: int = MIN_CONTEXT_SIZE
) -> Dict[str, Any]:
    """
    Find optimal configuration using joint optimization algorithm.
    
    Implements the theoretical model algorithm that:
    1. Prioritizes "Full Offload" (Max_Speed) regime first
    2. Falls back to "Hybrid" mode if full offload fails
    3. Maximizes context length (n_ctx) given VRAM constraint
    
    Args:
        model_size_bytes: Total model size in bytes (GGUF file size)
        total_layers: Total number of layers (N_layers)
        embedding_length: Hidden embedding dimension (N_embd)
        attention_head_count: Number of attention heads (N_head)
        attention_head_count_kv: Number of KV attention heads (N_head_kv)
        available_vram_bytes: Available VRAM in bytes
        available_ram_bytes: Available RAM in bytes
        cache_type_k: K cache quantization type
        cache_type_v: V cache quantization type (default: same as cache_type_k)
        ubatch_size: Micro-batch size for M_compute calculation
        desired_performance: 'Max_Speed' or 'Max_Context'
        min_context_size: Minimum acceptable context size
    
    Returns:
        Dictionary with:
            - mode: "Full_Offload", "Hybrid_Mode", "Full_Offload_Failed", or "Insufficient_Memory"
            - n_ngl_best: Optimal number of GPU layers
            - n_ctx_best: Optimal context size
            - cache_type_k: Selected K cache quantization
            - cache_type_v: Selected V cache quantization
            - ubatch_size: Optimal micro-batch size
    """
    # Step 1: Calculate model constants
    N_layers = total_layers
    M_weights_total = model_size_bytes
    
    # Calculate KV cache cost per token (C_kv_per_token)
    # Using GQA-aware formula: M_kv = n_ctx × N_layers × N_head_kv × d_head × (p_a_k + p_a_v)
    if embedding_length > 0 and attention_head_count > 0:
        from .constants import KV_CACHE_QUANT_FACTORS
        quant_factor_k = KV_CACHE_QUANT_FACTORS.get(cache_type_k, 0.5)
        quant_factor_v = KV_CACHE_QUANT_FACTORS.get(cache_type_v or cache_type_k, quant_factor_k)
        bytes_per_k = quant_factor_k * 4
        bytes_per_v = quant_factor_v * 4
        
        if attention_head_count_kv > 0:
            d_head = embedding_length / attention_head_count
            kv_cache_per_layer_k = attention_head_count_kv * d_head * bytes_per_k
            kv_cache_per_layer_v = attention_head_count_kv * d_head * bytes_per_v
        else:
            # Fallback for non-GQA models
            kv_cache_per_layer_k = embedding_length * bytes_per_k
            kv_cache_per_layer_v = embedding_length * bytes_per_v
        
        C_kv_per_token = (kv_cache_per_layer_k + kv_cache_per_layer_v) * N_layers
    else:
        # Fallback: conservative estimate
        C_kv_per_token = 1024  # ~1KB per token fallback
    
    # Calculate M_compute
    M_compute_bytes = int((COMPUTE_FIXED_OVERHEAD_MB + (ubatch_size * COMPUTE_SCRATCH_PER_UBATCH_MB)) * (1024**2))
    
    # Step 2: Try "Full Offload" (Max_Speed) regime first
    M_weights_vram_full = M_weights_total  # n_ngl = N_layers
    M_weights_ram_full = 0
    
    VRAM_fixed_cost_full = M_weights_vram_full + M_compute_bytes
    
    if VRAM_fixed_cost_full < available_vram_bytes and M_weights_ram_full < available_ram_bytes:
        # Model fits in VRAM. Calculate max context size.
        VRAM_remaining_for_kv = available_vram_bytes - VRAM_fixed_cost_full
        
        if C_kv_per_token > 0:
            n_ctx_candidate = VRAM_remaining_for_kv // C_kv_per_token
        else:
            n_ctx_candidate = 0
        
        if n_ctx_candidate >= min_context_size:
            return {
                "mode": "Full_Offload (Max_Speed)",
                "n_ngl_best": N_layers,
                "n_ctx_best": n_ctx_candidate,
                "cache_type_k": cache_type_k,
                "cache_type_v": cache_type_v,
                "ubatch_size": ubatch_size
            }
    
    # If full offload failed and user wants Max_Speed only, return failure
    if desired_performance == "Max_Speed":
        return {
            "mode": "Full_Offload_Failed",
            "n_ngl_best": 0,
            "n_ctx_best": 0,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "ubatch_size": ubatch_size
        }
    
    # Step 3: Try "Hybrid" (Max_Context) regime
    # Find minimum n_ngl required to fit remaining weights in RAM
    if M_weights_total > available_ram_bytes:
        n_ngl_min = max(1, int((M_weights_total - available_ram_bytes) * N_layers / M_weights_total))
    else:
        n_ngl_min = 0
    
    n_ngl_best = 0
    n_ctx_best = 0
    
    # Iterate from full offload down to minimum
    for n_ngl_candidate in range(N_layers, n_ngl_min - 1, -1):
        if n_ngl_candidate <= 0:
            continue
        
        M_weights_vram_hybrid = (n_ngl_candidate / N_layers) * M_weights_total
        M_weights_ram_hybrid = M_weights_total - M_weights_vram_hybrid
        VRAM_fixed_cost_hybrid = M_weights_vram_hybrid + M_compute_bytes
        
        # Check if this n_ngl is possible (weights + compute must fit in VRAM)
        if VRAM_fixed_cost_hybrid >= available_vram_bytes:
            continue  # This n_ngl is too high
        
        # Check if remaining weights fit in RAM
        if M_weights_ram_hybrid > available_ram_bytes:
            continue  # This n_ngl requires too much RAM
        
        # Calculate max context for this n_ngl
        VRAM_remaining_for_kv = available_vram_bytes - VRAM_fixed_cost_hybrid
        
        if C_kv_per_token > 0:
            n_ctx_candidate = VRAM_remaining_for_kv // C_kv_per_token
        else:
            n_ctx_candidate = 0
        
        # We're looking for the combination that yields the highest n_ctx
        if n_ctx_candidate > n_ctx_best:
            n_ctx_best = n_ctx_candidate
            n_ngl_best = n_ngl_candidate
    
    if n_ctx_best >= min_context_size:
        return {
            "mode": "Hybrid_Mode (Max_Context)",
            "n_ngl_best": n_ngl_best,
            "n_ctx_best": n_ctx_best,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "ubatch_size": ubatch_size
        }
    else:
        return {
            "mode": "Insufficient_Memory",
            "n_ngl_best": 0,
            "n_ctx_best": 0,
            "cache_type_k": cache_type_k,
            "cache_type_v": cache_type_v,
            "ubatch_size": ubatch_size
        }

