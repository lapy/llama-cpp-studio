"""
CPU configuration module.
Handles all CPU-specific configuration logic for model inference.
"""
from typing import Dict, Any, Optional, Tuple
import psutil

from .architecture_config import get_architecture_default_context
from .memory_estimator import get_cpu_memory_gb, tokens_per_gb_by_model_size, ctx_tokens_budget_greedy
from .calculators import calculate_optimal_batch_size_cpu, calculate_ubatch_size
from .constants import (
    MODEL_RAM_OVERHEAD_GB, CONTEXT_RAM_OVERHEAD_GB, MAX_CPU_CONTEXT_SIZE, MIN_CONTEXT_SIZE,
    ARCHITECTURE_CPU_PROFILES, ARCHITECTURE_CPU_BATCH_LIMITS
)


def get_optimal_cpu_context_size(architecture: str, available_ram_gb: float, model_size_mb: float) -> int:
    """Calculate optimal context size for CPU-only mode based on available RAM."""
    base_context = get_architecture_default_context(architecture)
    
    # Calculate how much RAM we can allocate for context
    # Reserve space for model + overhead
    model_ram_gb = model_size_mb / 1024
    reserved_ram_gb = model_ram_gb + MODEL_RAM_OVERHEAD_GB
    available_for_context = max(0, available_ram_gb - reserved_ram_gb)
    
    # Estimate context memory usage (rough: 1MB per 1000 tokens)
    max_context_tokens = int(available_for_context * 1000)  # 1GB = ~1000 tokens
    
    # Apply architecture-specific limits
    if architecture == "mistral":
        # Mistral can handle very large contexts
        optimal_context = min(base_context, max_context_tokens)
    elif architecture in ["llama3", "codellama"]:
        # Llama3 and CodeLlama have good context handling
        optimal_context = min(base_context, max_context_tokens)
    else:
        # Conservative for other architectures
        optimal_context = min(base_context, max_context_tokens, MAX_CPU_CONTEXT_SIZE)
    
    # Ensure minimum context size
    return max(MIN_CONTEXT_SIZE, optimal_context)


def calculate_optimal_batch_sizes(available_ram_gb: float, model_size_mb: float, 
                                  ctx_size: int, architecture: str) -> Tuple[int, int]:
    """Calculate optimal batch sizes for CPU mode."""
    return calculate_optimal_batch_size_cpu(
        available_ram_gb, model_size_mb, ctx_size, architecture
    )


def get_optimal_parallel_cpu(available_ram_gb: float, model_size_mb: float) -> int:
    """Calculate optimal parallel sequences for CPU mode."""
    model_ram_gb = model_size_mb / 1024
    
    # Calculate how many parallel sequences we can run
    # Each parallel sequence needs roughly 1GB of RAM
    max_parallel = int(available_ram_gb / (model_ram_gb + 1.0))
    
    # Apply reasonable limits
    if available_ram_gb >= 32:  # High RAM system
        return min(8, max(1, max_parallel))
    elif available_ram_gb >= 16:  # Mid RAM system
        return min(4, max(1, max_parallel))
    else:  # Low RAM system
        return min(2, max(1, max_parallel))


def get_cpu_architecture_optimizations(architecture: str, available_ram_gb: float) -> Dict[str, Any]:
    """Get architecture-specific optimizations for CPU mode using dict-based profiles."""
    # Get architecture-specific profile, or empty dict if not found
    profile = ARCHITECTURE_CPU_PROFILES.get(architecture, {})
    optimizations = dict(profile)  # Copy to avoid mutating the original
    
    # Handle dynamic mmap setting for llama architectures
    if optimizations.get("use_mmap") == "dynamic":
        optimizations["use_mmap"] = available_ram_gb < 16
    
    # Common CPU optimizations applied to all architectures
    optimizations.update({
        "embedding": False,  # Disable embedding mode for inference
        "cont_batching": True,  # Enable continuous batching for efficiency
        "no_kv_offload": True,  # Don't offload KV cache (CPU mode)
    })
    
    return optimizations


def generate_cpu_config(model_size_mb: float, architecture: str, layer_count: int = 32, 
                        context_length: int = 4096, vocab_size: int = 0, 
                        embedding_length: int = 0, attention_head_count: int = 0, 
                        debug: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Generate CPU-only configuration optimized for available RAM."""
    # Get system memory info (from centralized helper)
    total_ram_gb, used_ram_gb, available_ram_gb = get_cpu_memory_gb()
    if debug is not None:
        debug.update({
            "cpu_total_ram_gb": total_ram_gb,
            "cpu_available_ram_gb": available_ram_gb,
        })
    
    # Estimate CPU threads (leave some cores free for system)
    cpu_count_phys = psutil.cpu_count(logical=False) or 1
    logical_cpu_count = psutil.cpu_count(logical=True) or cpu_count_phys
    threads = max(1, cpu_count_phys - 1)  # Leave 1 core for system
    threads_batch = max(1, min(threads, max(1, logical_cpu_count - 2)))  # Guard negatives
    
    # Calculate optimal context size based on model's max and available RAM (no hard cap)
    base_ctx = max(512, context_length or 4096)
    model_gb = max(0.001, model_size_mb / 1024.0)
    # Tokens per GB heuristic (centralized)
    tokens_per_gb = tokens_per_gb_by_model_size(model_gb)
    # Reserve RAM for model + overhead using actual available RAM
    reserved_ram_gb = model_gb + 2.0
    available_for_ctx_gb = max(0.0, available_ram_gb - reserved_ram_gb)
    # Provide a small minimum window so we don't quantize to zero
    if available_for_ctx_gb <= 0:
        available_for_ctx_gb = max(0.25, available_ram_gb * 0.1)
    if debug is not None:
        debug.update({
            "model_gb": model_gb,
            "tokens_per_gb": tokens_per_gb,
            "reserved_ram_gb": reserved_ram_gb,
            "available_for_ctx_gb": available_for_ctx_gb,
        })
    # Initial cap ignoring batch/parallel
    max_tokens_by_ram = ctx_tokens_budget_greedy(model_gb, available_ram_gb, reserve_overhead_gb=2.0)
    optimal_ctx_size = max(512, min(base_ctx, max_tokens_by_ram))
    
    # Calculate optimal batch sizes using centralized function
    batch_size, ubatch_size = calculate_optimal_batch_size_cpu(available_ram_gb, model_size_mb, optimal_ctx_size, architecture)
    
    # Adjust ctx_size to account for batch and parallel (ctx * batch * parallel <= tokens_budget)
    parallel = 1
    tokens_budget = int(tokens_per_gb * available_for_ctx_gb)
    if tokens_budget > 0:
        # Budget ctx tokens directly from available RAM; batch is handled separately
        safe_ctx = int(tokens_budget)
        optimal_ctx_size = max(512, min(optimal_ctx_size, safe_ctx))
    if debug is not None:
        debug.update({
            "tokens_budget": tokens_budget,
            "batch_size": batch_size,
            "ubatch_size": ubatch_size,
            "parallel": parallel,
            "optimal_ctx_size": optimal_ctx_size,
        })

    config = {
        "threads": threads,
        "threads_batch": threads_batch,
        "ctx_size": optimal_ctx_size,
        "batch_size": batch_size,
        "ubatch_size": ubatch_size,
        "parallel": parallel,
        "no_mmap": False,
        "mlock": False,
        "low_vram": False,
        "logits_all": False,  # Don't compute all logits to save memory
    }
    
    # Add architecture-specific optimizations
    config.update(get_cpu_architecture_optimizations(architecture, available_ram_gb))
    
    return config

