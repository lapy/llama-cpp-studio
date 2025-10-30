from typing import Tuple
import psutil


def get_cpu_memory_gb() -> Tuple[float, float, float]:
    """Return (total_gb, used_gb, available_gb) where available = total - used.
    Uses actual values, no 60% approximations.
    """
    mem = psutil.virtual_memory()
    total = mem.total / (1024 ** 3)
    used = mem.used / (1024 ** 3)
    available = max(0.0, total - used)
    return total, used, available


def tokens_per_gb_by_model_size(model_size_gb: float) -> int:
    """Heuristic tokens per GB for KV budget by model size."""
    if model_size_gb < 2:
        return 3000
    if model_size_gb < 6:
        return 2000
    if model_size_gb < 12:
        return 1300
    return 400


def ctx_tokens_budget_greedy(model_size_gb: float, available_cpu_ram_gb: float, reserve_overhead_gb: float = 2.0) -> int:
    """Compute context token budget from CPU RAM after reserving model + overhead.
    Returns total tokens budget (not divided by batch/parallel).
    """
    reserved = model_size_gb + max(0.0, reserve_overhead_gb)
    for_ctx = max(0.0, available_cpu_ram_gb - reserved)
    tpg = tokens_per_gb_by_model_size(model_size_gb)
    return max(0, int(for_ctx * tpg))


