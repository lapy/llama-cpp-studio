"""
GPU configuration module.
Handles all GPU-specific configuration logic including single GPU, multi-GPU, and NVLink topologies.
"""
from typing import Dict, Any, Optional, List
import psutil

from .architecture_config import get_architecture_default_context
from .calculators import (
    calculate_optimal_batch_size_gpu,
    calculate_max_context_size_gpu,
    calculate_optimal_context_size_gpu,
    calculate_optimal_gpu_layers,
    calculate_ubatch_size
)
from .constants import VRAM_FRAGMENTATION_MARGIN, VRAM_SAFETY_MARGIN, MIN_CONTEXT_SIZE, MAX_CONTEXT_SIZE, MIN_BATCH_SIZE, MAX_BATCH_SIZE


def parse_compute_capability(value: str) -> float:
    """Parse compute capability like '8.0', '7.5' to a float safely."""
    try:
        parts = str(value).split('.')
        major = int(parts[0]) if parts and parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return major + minor / 10.0
    except Exception:
        return 0.0


def calculate_optimal_batch_size(available_vram_gb: float, model_size_mb: float, context_size: int, 
                                  embedding_length: int, layer_count: int,
                                  cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None) -> int:
    """Calculate optimal batch size based on memory and throughput analysis."""
    return calculate_optimal_batch_size_gpu(
        available_vram_gb, model_size_mb, context_size, embedding_length, layer_count,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v
    )


def calculate_max_context_size(available_vram_gb: float, model_size_mb: float, layer_count: int, 
                              embedding_length: int, attention_head_count: int, attention_head_count_kv: int) -> int:
    """Calculate maximum context size based on actual memory requirements."""
    return calculate_max_context_size_gpu(
        available_vram_gb, model_size_mb, layer_count, embedding_length, attention_head_count, attention_head_count_kv
    )


def get_optimal_context_size(architecture: str, available_vram: int, model_size_mb: float = 0, 
                            layer_count: int = 32, embedding_length: int = 0, 
                            attention_head_count: int = 0, attention_head_count_kv: int = 0,
                            cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
                            usage_mode: str = "single_user") -> int:
    """Calculate optimal context size based on actual memory requirements and architecture."""
    base_context = get_architecture_default_context(architecture)
    return calculate_optimal_context_size_gpu(
        architecture, available_vram, model_size_mb, layer_count, 
        embedding_length, attention_head_count, attention_head_count_kv, base_context,
        cache_type_k=cache_type_k, cache_type_v=cache_type_v,
        usage_mode=usage_mode
    )


def single_gpu_config(model_size_mb: float, architecture: str, gpu: Dict, layer_count: int = 32, 
                     embedding_length: int = 0, attention_head_count: int = 0, 
                     attention_head_count_kv: int = 0,
                     compute_capability: float = 0.0, context_length: int = 4096,
                     cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
                     usage_mode: str = "single_user") -> Dict[str, Any]:
    """Configuration for single GPU.
    
    Args:
        compute_capability: Pre-parsed compute capability (e.g., 8.0, 7.5). 
                          Use 0.0 if not available.
    """
    # Extract frequently accessed values to avoid repeated dict lookups
    gpu_memory = gpu.get("memory", {})
    vram_gb = gpu_memory.get("total", 0) / (1024**3)
    free_vram_gb = gpu_memory.get("free", 0) / (1024**3)
    gpu_index = gpu.get("index", 0)
    
    # Use calculator for GPU layer estimation
    # Use exact M_kv/M_compute calculation with estimated context/batch values
    # These will be refined later, but using estimates here gives better initial calculation
    n_gpu_layers = calculate_optimal_gpu_layers(
        free_vram_gb, model_size_mb, layer_count,
        context_size=context_length,  # Use architecture default context
        cache_type_k=cache_type_k,
        cache_type_v=cache_type_v,
        ubatch_size=512,  # Reasonable estimate for initial calculation
        attention_head_count=attention_head_count,
        attention_head_count_kv=attention_head_count_kv,
        embedding_length=embedding_length,
        usage_mode=usage_mode
    )
    
    config = {
        "n_gpu_layers": n_gpu_layers,
        "main_gpu": gpu_index,
        "threads": max(1, (psutil.cpu_count(logical=False) or 2) - 2),
        "threads_batch": max(1, (psutil.cpu_count(logical=False) or 2) - 2)
    }
    
    # Calculate optimal batch sizes based on actual memory requirements
    # Note: Use architecture default context_length (will be refined later in generate_gpu_config)
    # Use selected KV cache quantization if provided
    if embedding_length > 0 and layer_count > 0:
        # Use data-driven calculation with architecture default context length
        optimal_batch_size = calculate_optimal_batch_size(free_vram_gb, model_size_mb, context_length, 
                                                          embedding_length, layer_count,
                                                          cache_type_k=cache_type_k, cache_type_v=cache_type_v)
        config["batch_size"] = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, optimal_batch_size))
        config["ubatch_size"] = calculate_ubatch_size(config["batch_size"])
    else:
        # Fallback to VRAM-based estimation
        if vram_gb >= 24:    # High-end GPU
            config["batch_size"] = min(1024, max(256, int(vram_gb * 30)))
            config["ubatch_size"] = min(512, max(128, int(vram_gb * 15)))
        elif vram_gb >= 12:   # Mid-range GPU
            config["batch_size"] = min(512, max(128, int(vram_gb * 25)))
            config["ubatch_size"] = min(256, max(64, int(vram_gb * 12)))
        elif vram_gb >= 8:    # Lower-end GPU
            config["batch_size"] = min(256, max(64, int(vram_gb * 20)))
            config["ubatch_size"] = min(128, max(32, int(vram_gb * 10)))
        else:                 # Very limited VRAM
            config["batch_size"] = min(128, max(32, int(vram_gb * 15)))
            config["ubatch_size"] = min(64, max(16, int(vram_gb * 7)))
    
    # Enable flash attention for supported GPUs (Ampere and newer: >= 8.0)
    if compute_capability >= 8.0:
        config["flash_attn"] = True
    
    return config


def multi_gpu_config(model_size_mb: float, architecture: str, gpus: list, nvlink_topology: Dict, 
                    layer_count: int = 32, compute_capabilities: Optional[List[float]] = None) -> Dict[str, Any]:
    """Configuration for multiple GPUs with NVLink awareness.
    
    Args:
        compute_capabilities: Pre-parsed compute capabilities list. If None, will parse from gpus.
    """
    config = {
        "main_gpu": 0,
        "n_gpu_layers": -1,  # Use all layers
        "threads": max(1, psutil.cpu_count(logical=False) - 2),
        "threads_batch": max(1, psutil.cpu_count(logical=False) - 2)
    }
    
    # Enable flash attention if all GPUs support it (Ampere and newer: >= 8.0)
    if compute_capabilities:
        # Use pre-parsed compute capabilities
        if all(cc >= 8.0 for cc in compute_capabilities):
            config["flash_attn"] = True
    else:
        # Fallback: parse from gpus if not provided
        if all(parse_compute_capability(gpu.get("compute_capability", "0.0")) >= 8.0 for gpu in gpus):
            config["flash_attn"] = True
    
    # Configure based on NVLink topology
    strategy = nvlink_topology.get("recommended_strategy", "pcie_only")
    
    if strategy == "nvlink_unified":
        # All GPUs connected via NVLink - use unified memory approach
        config.update(nvlink_unified_config(gpus, nvlink_topology))
    elif strategy == "nvlink_clustered":
        # Multiple NVLink clusters - optimize per cluster
        config.update(nvlink_clustered_config(gpus, nvlink_topology))
    elif strategy == "nvlink_partial":
        # Partial NVLink connectivity - hybrid approach
        config.update(nvlink_partial_config(gpus, nvlink_topology))
    else:
        # PCIe only - traditional tensor splitting
        config.update(pcie_only_config(gpus))
    
    return config


def nvlink_unified_config(gpus: list, nvlink_topology: Dict) -> Dict[str, Any]:
    """Configuration for unified NVLink cluster."""
    # With NVLink, we can use more aggressive tensor splitting
    # Extract memory values once to avoid repeated dict lookups
    vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in gpus]
    total_vram = sum(vram_sizes)
    total_vram_gb = total_vram / (1024**3)
    
    # Pre-calculate ratios as floats, format only at the end
    tensor_split = [f"{vram / total_vram:.3f}" if total_vram > 0 else "0.000" for vram in vram_sizes]
    
    return {
        "tensor_split": ",".join(tensor_split),
        "parallel": min(8, len(gpus) * 2),  # Higher parallelism with NVLink
        "batch_size": min(4096, max(512, int(total_vram_gb * 150))),  # Larger batches for high VRAM
        "ubatch_size": min(2048, max(256, int(total_vram_gb * 75)))
    }


def nvlink_clustered_config(gpus: list, nvlink_topology: Dict) -> Dict[str, Any]:
    """Configuration for multiple NVLink clusters."""
    # Extract clusters once to avoid repeated dict lookup
    clusters = nvlink_topology.get("clusters", [])
    
    if not clusters:
        return pcie_only_config(gpus)
    
    # Use the largest cluster for primary processing
    largest_cluster = max(clusters, key=lambda c: len(c["gpus"]))
    cluster_gpu_indices = set(largest_cluster["gpus"])
    
    # Configure tensor split for the largest cluster
    # Pre-extract all GPU memory values once to avoid repeated dict lookups
    gpu_memories = [gpu.get("memory", {}) for gpu in gpus]
    cluster_vram_sizes = [gpu_memories[i].get("total", 0) for i in cluster_gpu_indices]
    total_vram = sum(cluster_vram_sizes)
    total_vram_gb = total_vram / (1024**3)
    
    # Pre-calculate ratios as floats, format only at the end
    tensor_split_ratios = []
    for i, gpu_memory in enumerate(gpu_memories):
        if i in cluster_gpu_indices:
            ratio = gpu_memory.get("total", 0) / total_vram if total_vram > 0 else 0.0
            tensor_split_ratios.append(ratio)
        else:
            tensor_split_ratios.append(0.0)
    
    # Format all ratios in a single pass
    tensor_split = [f"{ratio:.3f}" for ratio in tensor_split_ratios]
    
    return {
        "tensor_split": ",".join(tensor_split),
        "parallel": min(6, len(largest_cluster["gpus"]) * 2),
        "batch_size": min(3072, max(384, int(total_vram_gb * 120))),
        "ubatch_size": min(1536, max(192, int(total_vram_gb * 60)))
    }


def nvlink_partial_config(gpus: list, nvlink_topology: Dict) -> Dict[str, Any]:
    """Configuration for partial NVLink connectivity."""
    # Use conservative approach for partial NVLink
    vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in gpus]
    total_vram = sum(vram_sizes)
    total_vram_gb = total_vram / (1024**3)
    
    # Pre-calculate ratios as floats, format only at the end
    tensor_split = [f"{vram / total_vram:.2f}" if total_vram > 0 else "0.00" for vram in vram_sizes]
    
    return {
        "tensor_split": ",".join(tensor_split),
        "parallel": min(4, len(gpus)),
        "batch_size": min(2048, max(256, int(total_vram_gb * 100))),
        "ubatch_size": min(1024, max(128, int(total_vram_gb * 50)))
    }


def pcie_only_config(gpus: list) -> Dict[str, Any]:
    """Configuration for PCIe-only multi-GPU setup."""
    # Calculate tensor split based on VRAM
    vram_sizes = [gpu.get("memory", {}).get("total", 0) for gpu in gpus]
    total_vram = sum(vram_sizes)
    total_vram_gb = total_vram / (1024**3)
    
    # Pre-calculate ratios as floats, format only at the end
    tensor_split = [f"{vram / total_vram:.2f}" if total_vram > 0 else "0.00" for vram in vram_sizes]
    
    return {
        "tensor_split": ",".join(tensor_split),
        "parallel": min(2, len(gpus)),  # Conservative parallelism for PCIe
        "batch_size": min(1024, max(128, int(total_vram_gb * 80))),
        "ubatch_size": min(512, max(64, int(total_vram_gb * 40)))
    }


def generate_gpu_config(model_size_mb: float, architecture: str, gpus: list, total_vram: int, 
                       gpu_count: int, nvlink_topology: Dict, layer_count: int = 32, 
                       context_length: int = 4096, vocab_size: int = 0, embedding_length: int = 0, 
                       attention_head_count: int = 0, attention_head_count_kv: int = 0,
                       compute_capabilities: Optional[List[float]] = None,
                       cache_type_k: Optional[str] = None, cache_type_v: Optional[str] = None,
                       usage_mode: str = "single_user",
                       debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate GPU-optimized configuration.
    
    Args:
        compute_capabilities: Pre-parsed compute capabilities list from SystemResources.
    """
    config = {}
    
    # Calculate optimal GPU layers
    available_vram = sum(gpu.get("memory", {}).get("free", 0) for gpu in gpus)
    available_vram_gb = available_vram / (1024**3)
    
    if gpu_count == 1:
        # Use pre-parsed compute capability for single GPU
        gpu_cc = compute_capabilities[0] if compute_capabilities and len(compute_capabilities) > 0 else 0.0
        config.update(single_gpu_config(model_size_mb, architecture, gpus[0], layer_count, 
                                       embedding_length, attention_head_count, attention_head_count_kv,
                                       gpu_cc, context_length,
                                       cache_type_k=cache_type_k, cache_type_v=cache_type_v,
                                       usage_mode=usage_mode))
    else:
        config.update(multi_gpu_config(model_size_mb, architecture, gpus, nvlink_topology, 
                                      layer_count, compute_capabilities))
    
    # Context size based on available VRAM and model parameters
    # Use selected KV cache quantization if provided
    ctx_size = get_optimal_context_size(architecture, available_vram, model_size_mb, layer_count, 
                                        embedding_length, attention_head_count, attention_head_count_kv,
                                        cache_type_k=cache_type_k, cache_type_v=cache_type_v,
                                        usage_mode=usage_mode)
    # Clamp GPU ctx size to sane bounds
    config["ctx_size"] = max(MIN_CONTEXT_SIZE, min(ctx_size, MAX_CONTEXT_SIZE))
    if debug is not None:
        debug.update({
            "gpu_available_vram_bytes": int(available_vram),
            "gpu_ctx_size": config["ctx_size"],
        })
    
    # Batch sizes based on actual memory requirements
    # Use selected KV cache quantization if provided
    if embedding_length > 0 and layer_count > 0:
        optimal_batch_size = calculate_optimal_batch_size(available_vram_gb, model_size_mb, config["ctx_size"], 
                                                          embedding_length, layer_count,
                                                          cache_type_k=cache_type_k, cache_type_v=cache_type_v)
        config["batch_size"] = max(MIN_BATCH_SIZE, min(MAX_BATCH_SIZE, optimal_batch_size))
        config["ubatch_size"] = calculate_ubatch_size(config["batch_size"])
    else:
        # Fallback to size-based estimation
        config["batch_size"] = min(1024, max(64, int(model_size_mb / 50)))
        config["ubatch_size"] = min(config["batch_size"], max(16, int(model_size_mb / 100)))
    
    # Parallel sequences (conservative for multi-GPU)
    if gpu_count > 1:
        config["parallel"] = max(1, min(4, gpu_count))
    else:
        config["parallel"] = 1
    
    return config

