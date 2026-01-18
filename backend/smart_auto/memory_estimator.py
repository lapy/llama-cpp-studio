"""
Memory estimation module.
Consolidates RAM and VRAM estimation with shared KV cache calculation logic.
Also provides CPU memory utilities.
"""

from typing import Dict, Any, Tuple, Optional
from functools import lru_cache
import psutil

from backend.database import Model
from backend.logging_config import get_logger
from .model_metadata import get_model_metadata
from .models import ModelMetadata
from .constants import (
    KV_CACHE_QUANT_FACTORS,
    KV_CACHE_OPTIMIZATION_FACTOR,
    KV_CACHE_SINGLE_USER_FACTOR,
    KV_CACHE_MULTI_USER_FACTOR,
    MODEL_RAM_OVERHEAD_GB,
    FALLBACK_LAYER_COUNT,
    FALLBACK_EMBEDDING_LENGTH,
    FALLBACK_KV_CACHE_PER_TOKEN_BYTES,
    BATCH_VRAM_OVERHEAD_RATIO,
    BATCH_RAM_OVERHEAD_RATIO,
    VRAM_SAFETY_MARGIN,
    MOE_OFFLOAD_ALL_RATIO,
    MOE_OFFLOAD_UP_DOWN_RATIO,
    MOE_OFFLOAD_UP_RATIO,
    LLAMA_CPP_OVERHEAD_MB,
    DEFAULT_BYTES_PER_ELEMENT,
    BATCH_INTERMEDIATE_FACTOR,
    BATCH_QKV_FACTOR,
    BATCH_COMPUTATION_OVERHEAD_KB,
    BATCH_FALLBACK_MB,
    QUANTIZATION_AVERAGE_FACTOR,
    COMPUTE_FIXED_OVERHEAD_MB,
    COMPUTE_SCRATCH_PER_UBATCH_MB,
)

logger = get_logger(__name__)


# CPU memory utilities
def get_cpu_memory_gb() -> Tuple[float, float, float]:
    """Return (total_gb, used_gb, available_gb) where available = total - used.
    Uses actual values, no 60% approximations.
    """
    mem = psutil.virtual_memory()
    total = mem.total / (1024**3)
    used = mem.used / (1024**3)
    available = max(0.0, total - used)
    return total, used, available


@lru_cache(maxsize=64)
def tokens_per_gb_by_model_size(model_size_gb: float) -> int:
    """Heuristic tokens per GB for KV budget by model size."""
    if model_size_gb < 2:
        return 3000
    if model_size_gb < 6:
        return 2000
    if model_size_gb < 12:
        return 1300
    return 400


def ctx_tokens_budget_greedy(
    model_size_gb: float, available_cpu_ram_gb: float, reserve_overhead_gb: float = None
) -> int:
    """Compute context token budget from CPU RAM after reserving model + overhead.
    Returns total tokens budget (not divided by batch/parallel).
    """
    if reserve_overhead_gb is None:
        reserve_overhead_gb = MODEL_RAM_OVERHEAD_GB
    reserved = model_size_gb + max(0.0, reserve_overhead_gb)
    for_ctx = max(0.0, available_cpu_ram_gb - reserved)
    tpg = tokens_per_gb_by_model_size(model_size_gb)
    return max(0, int(for_ctx * tpg))


def get_kv_cache_quant_factor(cache_type: str) -> float:
    """Get memory reduction factor for KV cache quantization."""
    return KV_CACHE_QUANT_FACTORS.get(cache_type, 1.0)


@lru_cache(maxsize=512)
def calculate_kv_cache_size(
    ctx_size: int,
    parallel: int,
    layer_count: int,
    embedding_length: int,
    attention_head_count: int,
    attention_head_count_kv: int,
    cache_type_k: str,
    cache_type_v: Optional[str] = None,
    usage_mode: str = "single_user",
) -> int:
    """
    Calculate KV cache size in bytes for memory estimation.

    Cached with LRU to avoid redundant calculations for same parameters.
    Cache size increased to 512 for production workloads.

    Returns:
        Total KV cache bytes
    """
    # Get quantization factors
    quant_factor_k = get_kv_cache_quant_factor(cache_type_k)
    quant_factor_v = (
        get_kv_cache_quant_factor(cache_type_v) if cache_type_v else quant_factor_k
    )

    if embedding_length > 0 and layer_count > 0:
        # Calculate bytes per element for K and V cache
        bytes_per_k = (
            quant_factor_k * 4
        )  # Convert factor to actual bytes (f32=4, f16=2, etc.)
        bytes_per_v = quant_factor_v * 4 if cache_type_v else bytes_per_k

        # GQA-aware KV cache calculation (correct formula from theoretical model)
        # M_kv = n_ctx × N_layers × N_head_kv × d_head × (p_a_k + p_a_v)
        # where d_head = N_embd / N_head
        if attention_head_count_kv > 0 and attention_head_count > 0:
            # Dimension per head
            d_head = embedding_length / attention_head_count
            # KV cache stores N_head_kv heads per layer, each of size d_head
            kv_cache_per_layer_k = attention_head_count_kv * d_head * bytes_per_k
            kv_cache_per_layer_v = attention_head_count_kv * d_head * bytes_per_v
        else:
            # Fallback for non-GQA models (MHA: N_head_kv = N_head)
            # In this case, use full embedding dimension
            kv_cache_per_layer_k = embedding_length * bytes_per_k
            kv_cache_per_layer_v = embedding_length * bytes_per_v

        # Total per token: (Key + Value) * layers
        kv_cache_per_token = (kv_cache_per_layer_k + kv_cache_per_layer_v) * layer_count
    else:
        # Fallback using constants
        kv_cache_per_token = FALLBACK_KV_CACHE_PER_TOKEN_BYTES

    # KV cache: ctx_size tokens, each with kv_cache_per_token bytes
    # Parallel might create multiple context copies, so multiply by parallel
    # Use actual memory (optimization factor is 1.0 - memory mapping doesn't reduce peak usage)
    base_kv_cache_bytes = int(
        ctx_size * kv_cache_per_token * parallel * KV_CACHE_OPTIMIZATION_FACTOR
    )

    # Apply usage mode factor based on theoretical model:
    # - single_user: Peak estimate (full context accumulates, full KV cache)
    # - multi_user: Typical usage (context cleared between requests, lower estimate)
    if usage_mode == "multi_user":
        usage_factor = KV_CACHE_MULTI_USER_FACTOR
    else:  # single_user or default
        usage_factor = KV_CACHE_SINGLE_USER_FACTOR

    kv_cache_bytes = int(base_kv_cache_bytes * usage_factor)

    return kv_cache_bytes


def estimate_vram_usage(
    model: Model,
    config: Dict[str, Any],
    gpu_info: Dict[str, Any],
    metadata: Optional[ModelMetadata] = None,
    usage_mode: str = "single_user",
) -> Dict[str, Any]:
    """Estimate VRAM usage for given configuration using comprehensive model metadata

    Args:
        model: The model to estimate for
        config: Configuration dictionary
        gpu_info: GPU information dictionary
        metadata: Optional pre-computed ModelMetadata to avoid redundant calls
    """
    try:
        model_size = model.file_size if model.file_size else 0

        # Extract frequently accessed config values early to avoid repeated dict lookups
        n_gpu_layers = int(config.get("n_gpu_layers", 0) or 0)
        ctx_size = int(config.get("ctx_size", 4096) or 4096)
        parallel = max(1, int(config.get("parallel", 1) or 1))
        cache_type_k = config.get("cache_type_k", "f16")
        cache_type_v = config.get("cache_type_v")

        # Use provided metadata or fetch it (cached internally)
        layer_info = metadata if metadata is not None else get_model_metadata(model)
        total_layers = max(1, layer_info.layer_count or FALLBACK_LAYER_COUNT)
        embedding_length = layer_info.embedding_length or 0
        attention_head_count = layer_info.attention_head_count or 0
        attention_head_count_kv = layer_info.attention_head_count_kv or 0

        # Layer split between GPU and CPU
        layer_ratio = min(
            1.0, max(0.0, (n_gpu_layers / total_layers) if total_layers > 0 else 0.0)
        )
        model_vram = int(model_size * layer_ratio)
        model_ram = max(0, int(model_size - model_vram))

        # Use shared KV cache calculation
        kv_cache_bytes = calculate_kv_cache_size(
            ctx_size,
            parallel,
            total_layers,
            embedding_length,
            attention_head_count,
            attention_head_count_kv,
            cache_type_k,
            cache_type_v,
            usage_mode=usage_mode,
        )

        # Determine if KV cache goes to VRAM or RAM
        # According to theoretical model: when n_gpu_layers > 0, M_kv goes to VRAM by default
        # The "VRAM Trap": in hybrid mode, M_kv and M_compute both go to VRAM
        if n_gpu_layers > 0:
            # In GPU mode (including hybrid), KV cache goes to VRAM
            kv_cache_vram = kv_cache_bytes
            kv_cache_ram = 0
        else:
            # CPU-only mode: KV cache goes to RAM
            kv_cache_vram = 0
            kv_cache_ram = kv_cache_bytes

        # M_compute: Fixed overhead + variable scratch buffer
        # According to theoretical model: M_compute = M_overhead_fixed + M_scratch_variable(n_ubatch)
        ubatch_size = config.get("ubatch_size", 512)
        compute_overhead_mb = COMPUTE_FIXED_OVERHEAD_MB + (
            ubatch_size * COMPUTE_SCRATCH_PER_UBATCH_MB
        )
        compute_overhead_bytes = int(compute_overhead_mb * 1024 * 1024)

        # Allocate M_compute to VRAM if GPU layers > 0 (VRAM Trap)
        if n_gpu_layers > 0:
            batch_vram = compute_overhead_bytes
            batch_ram = 0
        else:
            batch_vram = 0
            batch_ram = compute_overhead_bytes

        estimated_vram = model_vram + kv_cache_vram + batch_vram
        estimated_ram = model_ram + kv_cache_ram + batch_ram

        # System RAM usage snapshot
        try:
            vm = psutil.virtual_memory()
            system_ram_used = int(vm.used)
            system_ram_total = int(vm.total)
        except Exception:
            system_ram_used = 0
            system_ram_total = 0

        # VRAM headroom check
        # Extract gpus list once to avoid repeated dict lookup
        gpus = gpu_info.get("gpus", [])
        total_free_vram = sum(g.get("memory", {}).get("free", 0) for g in gpus)
        fits_in_gpu = (n_gpu_layers == 0) or (
            estimated_vram <= max(0, total_free_vram * VRAM_SAFETY_MARGIN)
        )

        memory_mode = "ram_only"
        if n_gpu_layers > 0:
            if estimated_ram > 0:
                memory_mode = "mixed"
            else:
                memory_mode = "vram_only"

        return {
            "memory_mode": memory_mode,
            # VRAM
            "estimated_vram": estimated_vram,
            "model_vram": model_vram,
            "kv_cache_vram": kv_cache_vram,
            "batch_vram": batch_vram,
            # RAM
            "estimated_ram": estimated_ram,
            "model_ram": model_ram,
            "kv_cache_ram": kv_cache_ram,
            "batch_ram": batch_ram,
            # System RAM snapshot
            "system_ram_used": system_ram_used,
            "system_ram_total": system_ram_total,
            # Fit flag
            "fits_in_gpu": fits_in_gpu,
        }
    except Exception:
        try:
            vm = psutil.virtual_memory()
            system_ram_used = int(vm.used)
            system_ram_total = int(vm.total)
        except Exception:
            system_ram_used = 0
            system_ram_total = 0
        return {
            "memory_mode": "unknown",
            "estimated_vram": 0,
            "model_vram": 0,
            "kv_cache_vram": 0,
            "batch_vram": 0,
            "estimated_ram": 0,
            "model_ram": 0,
            "kv_cache_ram": 0,
            "batch_ram": 0,
            "system_ram_used": system_ram_used,
            "system_ram_total": system_ram_total,
            "fits_in_gpu": True,
        }


def estimate_ram_usage(
    model: Model,
    config: Dict[str, Any],
    metadata: Optional[ModelMetadata] = None,
    usage_mode: str = "single_user",
) -> Dict[str, Any]:
    """Estimate RAM usage for given configuration

    Args:
        model: The model to estimate for
        config: Configuration dictionary
        metadata: Optional pre-computed ModelMetadata to avoid redundant calls
    """
    try:
        model_size = model.file_size if model.file_size else 0

        # Extract frequently accessed config values early to avoid repeated dict lookups
        n_gpu_layers = config.get("n_gpu_layers", 0)
        ctx_size = config.get("ctx_size", 4096)
        batch_size = config.get("batch_size", 512)
        parallel = config.get("parallel", 1)
        cache_type_k = config.get("cache_type_k", "f16")
        cache_type_v = config.get("cache_type_v")

        # Get system RAM info (extract once)
        vm = psutil.virtual_memory()
        total_memory = vm.total
        available_memory = vm.available

        # Use provided metadata or fetch it (cached internally)
        # Use ModelMetadata dataclass attributes directly
        layer_info = metadata if metadata is not None else get_model_metadata(model)
        total_layers = layer_info.layer_count or FALLBACK_LAYER_COUNT
        embedding_length = layer_info.embedding_length or 0
        attention_head_count = layer_info.attention_head_count or 0
        attention_head_count_kv = layer_info.attention_head_count_kv or 0
        is_moe = layer_info.is_moe

        cpu_layers = total_layers - n_gpu_layers if n_gpu_layers > 0 else total_layers

        if n_gpu_layers > 0:
            # GPU layers: full model loaded in RAM for GPU transfer
            model_ram = model_size
        else:
            # CPU-only: only CPU layers in RAM
            layer_ratio = cpu_layers / total_layers if cpu_layers > 0 else 1
            model_ram = int(model_size * layer_ratio)

        # Enhanced KV cache estimation using model architecture
        # Use shared KV cache calculation
        kv_cache_ram = calculate_kv_cache_size(
            ctx_size,
            parallel,
            total_layers,
            embedding_length,
            attention_head_count,
            attention_head_count_kv,
            cache_type_k,
            cache_type_v,
            usage_mode=usage_mode,
        )

        # MoE models with CPU offloading use RAM for offloaded layers
        moe_cpu_ram = 0
        if is_moe and n_gpu_layers > 0:
            moe_pattern = config.get("moe_offload_custom", "")
            if moe_pattern:
                # Estimate RAM usage for offloaded MoE layers
                if ".*_exps" in moe_pattern:
                    # All MoE offloaded
                    moe_cpu_ram = int(model_size * MOE_OFFLOAD_ALL_RATIO)
                elif "up|down" in moe_pattern:
                    # Up/Down offloaded
                    moe_cpu_ram = int(model_size * MOE_OFFLOAD_UP_DOWN_RATIO)
                elif "_up_" in moe_pattern:
                    # Only Up offloaded
                    moe_cpu_ram = int(model_size * MOE_OFFLOAD_UP_RATIO)

        # Batch processing overhead
        if embedding_length > 0:
            bytes_per_element = DEFAULT_BYTES_PER_ELEMENT

            # Intermediate activations: batch_size tokens * embedding_length
            intermediate_ram = int(
                batch_size
                * embedding_length
                * bytes_per_element
                * BATCH_INTERMEDIATE_FACTOR
            )

            # QKV projections are also temporary and reused
            qkv_ram = int(
                batch_size * 3 * embedding_length * bytes_per_element * BATCH_QKV_FACTOR
            )

            # Additional buffers are minimal and reused
            computation_overhead = batch_size * BATCH_COMPUTATION_OVERHEAD_KB * 1024

            batch_ram = intermediate_ram + qkv_ram + computation_overhead
        else:
            # Fallback: reduced estimate based on actual usage
            batch_ram = batch_size * int(BATCH_FALLBACK_MB * 1024 * 1024)

        # Additional overhead for llama.cpp
        llama_overhead = LLAMA_CPP_OVERHEAD_MB * 1024 * 1024

        total_ram = model_ram + kv_cache_ram + batch_ram + llama_overhead + moe_cpu_ram

        # Check if fits in available RAM
        fits_in_ram = total_ram <= available_memory

        # Calculate quantization savings
        quant_factor_k = get_kv_cache_quant_factor(cache_type_k)
        quant_factor_v = (
            get_kv_cache_quant_factor(cache_type_v) if cache_type_v else quant_factor_k
        )

        # Calculate raw KV cache size (for savings calculation) using correct GQA-aware formula
        if embedding_length > 0 and total_layers > 0:
            bytes_per_k = quant_factor_k * 4
            bytes_per_v = quant_factor_v * 4 if cache_type_v else bytes_per_k
            if attention_head_count_kv > 0 and attention_head_count > 0:
                # GQA-aware calculation
                d_head = embedding_length / attention_head_count
                kv_cache_per_layer_k = attention_head_count_kv * d_head * bytes_per_k
                kv_cache_per_layer_v = attention_head_count_kv * d_head * bytes_per_v
            else:
                # Fallback for non-GQA models
                kv_cache_per_layer_k = embedding_length * bytes_per_k
                kv_cache_per_layer_v = embedding_length * bytes_per_v
            kv_cache_per_token = (
                kv_cache_per_layer_k + kv_cache_per_layer_v
            ) * total_layers
        else:
            kv_cache_per_token = FALLBACK_KV_CACHE_PER_TOKEN_BYTES

        # Calculate savings (difference between f32 and current quantization)
        kv_cache_savings = int(
            ctx_size
            * parallel
            * kv_cache_per_token
            * (
                1
                - (
                    QUANTIZATION_AVERAGE_FACTOR * quant_factor_k
                    + QUANTIZATION_AVERAGE_FACTOR * quant_factor_v
                )
            )
        )

        return {
            "estimated_ram": total_ram,
            "model_ram": model_ram,
            "kv_cache_ram": kv_cache_ram,
            "batch_ram": batch_ram,
            "moe_cpu_ram": moe_cpu_ram,
            "llama_overhead": llama_overhead,
            "fits_in_ram": fits_in_ram,
            "available_ram": available_memory,
            "total_ram": total_memory,
            "utilization_percent": (
                (total_ram / total_memory * 100) if total_memory > 0 else 0
            ),
            "kv_cache_savings": kv_cache_savings,
        }

    except Exception as e:
        return {"error": str(e), "estimated_ram": 0, "fits_in_ram": False}
