from typing import Dict, Any


def get_optimal_kv_cache_quant(available_vram_gb: float, context_length: int,
                                architecture: str, flash_attn_available: bool = False) -> Dict[str, Any]:
    """Determine optimal KV cache quantization to balance memory usage and quality."""
    if context_length > 32768:
        cache_type_k = "q5_1" if available_vram_gb > 40 else "q4_1"
        cache_type_v = cache_type_k if flash_attn_available else None
        return {"cache_type_k": cache_type_k, "cache_type_v": cache_type_v}

    if context_length > 8192:
        cache_type_k = "q8_0" if available_vram_gb > 24 else "q4_1"
        cache_type_v = cache_type_k if flash_attn_available else None
        return {"cache_type_k": cache_type_k, "cache_type_v": cache_type_v}

    if available_vram_gb > 16:
        return {"cache_type_k": "f16", "cache_type_v": "f16" if flash_attn_available else None}

    return {"cache_type_k": "q8_0", "cache_type_v": "q8_0" if flash_attn_available else None}


