"""
Registry of model config parameters for llama.cpp (and optionally LMDeploy).
Used by the frontend to render basic vs advanced settings from a single source of truth.
"""

import copy
from typing import Any, Dict, List

# Param entry: key, label, type ("int"|"float"|"bool"|"string"), default, min, max (optional), description (optional), options (optional)
# options: list of {"value": ..., "label": "..."} for select/dropdown; when present, frontend renders a select instead of text/number input
ParamDef = Dict[str, Any]

# Basic params shown by default (most common for chat/embedding)
# Host and port are not included: they are managed by llama-swap (--port ${PORT}, host default 0.0.0.0)
LLAMA_CPP_BASIC: List[ParamDef] = [
    {"key": "model_alias", "label": "Model alias", "type": "string", "default": "", "description": "Expose this model under a custom runtime ID instead of the default Hugging Face-derived name"},
    {"key": "ctx_size", "label": "Context size", "type": "int", "default": 2048, "min": 512, "max": 1_000_000, "description": "Maximum context length in tokens"},
    {"key": "n_gpu_layers", "label": "GPU layers", "type": "int", "default": -1, "min": -1, "max": 1000, "description": "Number of layers to offload to GPU (-1 = all)"},
    {"key": "tensor_split", "label": "Tensor split", "type": "string", "default": "", "description": "Comma-separated GPU allocation (e.g. 40,30,30)"},
    {"key": "main_gpu", "label": "Main GPU", "type": "int", "default": 0, "min": 0, "description": "Index of the main GPU"},
    {"key": "batch_size", "label": "Batch size", "type": "int", "default": 512, "min": 1, "max": 2048, "description": "Batch size for prompt processing"},
    {"key": "threads", "label": "Threads", "type": "int", "default": 4, "min": 1, "max": 64, "description": "Number of threads"},
    {"key": "embedding", "label": "Embedding mode", "type": "bool", "default": False, "description": "Enable embedding-only mode"},
]

# Advanced params (shown in expandable "Advanced" section)
LLAMA_CPP_ADVANCED: List[ParamDef] = [
    {"key": "n_predict", "label": "Max tokens to predict", "type": "int", "default": -1, "min": -1, "max": 100_000, "description": "Maximum tokens to generate (-1 = no limit)"},
    {"key": "ubatch_size", "label": "Ubatch size", "type": "int", "default": 512, "min": 1, "max": 2048, "description": "Micro-batch size for prompt processing"},
    {"key": "temp", "label": "Temperature", "type": "float", "default": 0.8, "min": 0, "max": 2, "description": "Sampling temperature (higher = more random)"},
    {"key": "top_k", "label": "Top K", "type": "int", "default": 40, "min": 0, "max": 1000, "description": "Limit sampling to top K tokens (0 = disabled)"},
    {"key": "top_p", "label": "Top P", "type": "float", "default": 0.9, "min": 0, "max": 1, "description": "Nucleus sampling: keep smallest set with cumulative probability"},
    {"key": "min_p", "label": "Min P", "type": "float", "default": 0.0, "min": 0, "max": 1, "description": "Minimum token probability for sampling"},
    {"key": "typical_p", "label": "Typical P", "type": "float", "default": 1.0, "min": 0, "max": 1, "description": "Typical sampling (1 = disabled)"},
    {"key": "repeat_penalty", "label": "Repeat penalty", "type": "float", "default": 1.1, "min": 1, "max": 2, "description": "Penalize repeated tokens"},
    {"key": "presence_penalty", "label": "Presence penalty", "type": "float", "default": 0, "min": -2, "max": 2, "description": "Penalize tokens that have appeared"},
    {"key": "frequency_penalty", "label": "Frequency penalty", "type": "float", "default": 0, "min": -2, "max": 2, "description": "Penalize by token frequency"},
    {"key": "seed", "label": "Seed", "type": "int", "default": -1, "min": -1, "max": 2**31 - 1, "description": "Random seed for reproducibility (-1 = random)"},
    {"key": "threads_batch", "label": "Threads (batch)", "type": "int", "default": -1, "min": -1, "max": 64, "description": "Threads for batch inference (-1 = use threads)"},
    {"key": "parallel", "label": "Parallel", "type": "int", "default": 1, "min": 1, "max": 64, "description": "Number of parallel sequences"},
    {"key": "rope_freq_base", "label": "RoPE freq base", "type": "float", "default": 0, "min": 0, "description": "RoPE base frequency (0 = from model)"},
    {"key": "rope_freq_scale", "label": "RoPE freq scale", "type": "float", "default": 0, "min": 0, "description": "RoPE frequency scaling (0 = from model)"},
    {"key": "flash_attn", "label": "Flash attention", "type": "bool", "default": False, "description": "Use Flash Attention when available"},
    {"key": "yarn_ext_factor", "label": "YaRN ext factor", "type": "float", "default": -1, "min": -1, "description": "YaRN extension factor (-1 = disabled)"},
    {"key": "yarn_attn_factor", "label": "YaRN attn factor", "type": "float", "default": 1, "min": 0, "description": "YaRN attention factor"},
    {"key": "no_mmap", "label": "No mmap", "type": "bool", "default": False, "description": "Disable memory-mapped model loading"},
    {"key": "mlock", "label": "MLock", "type": "bool", "default": False, "description": "Lock model in RAM to prevent swapping"},
    {"key": "low_vram", "label": "Low VRAM", "type": "bool", "default": False, "description": "Reduce VRAM use (slower inference)"},
    {"key": "logits_all", "label": "Logits all", "type": "bool", "default": False, "description": "Return logits for all tokens (embedding mode)"},
    {"key": "cont_batching", "label": "Continuous batching", "type": "bool", "default": True, "description": "Use continuous batching for higher throughput"},
    {"key": "no_kv_offload", "label": "No KV offload", "type": "bool", "default": False, "description": "Disable KV cache offload to CPU"},
    {"key": "split_mode", "label": "Split mode", "type": "string", "default": "", "description": "Tensor split mode (e.g. layer, row)", "options": [{"value": "", "label": "Default"}, {"value": "layer", "label": "Layer"}, {"value": "row", "label": "Row"}]},
    {"key": "cache_type_k", "label": "Cache type K", "type": "string", "default": "", "description": "K cache backend (e.g. f16, q8_0)", "options": [{"value": "", "label": "Default"}, {"value": "f16", "label": "f16"}, {"value": "q8_0", "label": "q8_0"}, {"value": "q4_0", "label": "q4_0"}, {"value": "q4_1", "label": "q4_1"}]},
    {"key": "cache_type_v", "label": "Cache type V", "type": "string", "default": "", "description": "V cache backend (e.g. f16, q8_0)", "options": [{"value": "", "label": "Default"}, {"value": "f16", "label": "f16"}, {"value": "q8_0", "label": "q8_0"}, {"value": "q4_0", "label": "q4_0"}, {"value": "q4_1", "label": "q4_1"}]},
    {"key": "grammar", "label": "Grammar", "type": "string", "default": "", "description": "BNF grammar for constrained output"},
    {"key": "json_schema", "label": "JSON schema", "type": "string", "default": "", "description": "JSON schema for structured output"},
    {"key": "cpu_moe", "label": "CPU MoE", "type": "bool", "default": False, "description": "Run MoE experts on CPU"},
    {"key": "n_cpu_moe", "label": "N CPU MoE", "type": "int", "default": 0, "min": 0, "description": "Number of MoE experts to run on CPU"},
    {"key": "override_tensor", "label": "Override tensor", "type": "string", "default": "", "description": "Override tensor layout (e.g. for debugging)"},
    {"key": "rope_scaling", "label": "RoPE scaling", "type": "string", "default": "", "description": "RoPE scaling type (e.g. yarn, linear)", "options": [{"value": "", "label": "Default"}, {"value": "yarn", "label": "YaRN"}, {"value": "linear", "label": "Linear"}, {"value": "none", "label": "None"}]},
    {"key": "mirostat", "label": "Mirostat", "type": "int", "default": 0, "min": 0, "max": 2, "description": "Mirostat mode (0=off, 1=mirostat, 2=mirostat 2.0)", "options": [{"value": 0, "label": "Off"}, {"value": 1, "label": "Mirostat 1"}, {"value": 2, "label": "Mirostat 2.0"}]},
    {"key": "mirostat_tau", "label": "Mirostat tau", "type": "float", "default": 5.0, "min": 0, "description": "Mirostat target perplexity"},
    {"key": "mirostat_eta", "label": "Mirostat eta", "type": "float", "default": 0.1, "min": 0, "description": "Mirostat learning rate"},
]

# ik_llama.cpp: same as llama_cpp plus these extras (and different mirostat flag names)
IK_LLAMA_EXTRA: List[ParamDef] = [
    {"key": "mla_attn", "label": "MLA attention", "type": "bool", "default": False, "description": "Enable MLA attention"},
    {"key": "attn_max_batch", "label": "Attention max batch", "type": "int", "default": 0, "min": 0, "description": "Max attention batch size"},
    {"key": "fused_moe", "label": "Fused MoE", "type": "bool", "default": True, "description": "Enable fused MoE"},
    {"key": "smart_expert_reduction", "label": "Smart expert reduction", "type": "bool", "default": False, "description": "Enable smart expert reduction"},
]

# LMDeploy (safetensors / TurboMind)
LMDEPLOY_BASIC: List[ParamDef] = [
    {"key": "model_alias", "label": "Model alias", "type": "string", "default": "", "description": "Expose this model under a custom runtime ID instead of the default Hugging Face-derived name"},
    {"key": "session_len", "label": "Session length", "type": "int", "default": 2048, "min": 512, "max": 1_000_000, "description": "Maximum session length"},
    {"key": "max_batch_size", "label": "Max batch size", "type": "int", "default": 128, "min": 1, "max": 1024, "description": "Maximum batch size"},
    {"key": "tensor_parallel", "label": "Tensor parallel", "type": "int", "default": 1, "min": 1, "max": 8, "description": "Tensor parallelism degree"},
]
LMDEPLOY_ADVANCED: List[ParamDef] = [
    {"key": "dtype", "label": "Dtype", "type": "string", "default": "auto", "description": "Model dtype (auto, float16, bfloat16)", "options": [{"value": "auto", "label": "Auto"}, {"value": "float16", "label": "float16"}, {"value": "bfloat16", "label": "bfloat16"}]},
    {"key": "quant_policy", "label": "Quantization policy", "type": "int", "default": 0, "min": 0, "max": 8, "description": "KV cache quantization (0=off, 4=4bit, 8=8bit)", "options": [{"value": 0, "label": "Off"}, {"value": 4, "label": "4-bit"}, {"value": 8, "label": "8-bit"}]},
    {"key": "enable_prefix_caching", "label": "Prefix caching", "type": "bool", "default": False, "description": "Enable prefix caching"},
    {"key": "chat_template", "label": "Chat template", "type": "string", "default": "", "description": "Override chat template"},
]


def get_llama_cpp_param_registry() -> Dict[str, List[ParamDef]]:
    """Return basic and advanced param definitions for llama.cpp config forms."""
    return {
        "basic": LLAMA_CPP_BASIC,
        "advanced": LLAMA_CPP_ADVANCED,
    }


def get_ik_llama_param_registry() -> Dict[str, List[ParamDef]]:
    """Return param definitions for ik_llama.cpp (llama_cpp params plus ik_llama extras)."""
    basic = copy.deepcopy(LLAMA_CPP_BASIC)
    advanced = copy.deepcopy(LLAMA_CPP_ADVANCED) + copy.deepcopy(IK_LLAMA_EXTRA)
    return {"basic": basic, "advanced": advanced}


def get_lmdeploy_param_registry() -> Dict[str, List[ParamDef]]:
    """Return param definitions for LMDeploy (safetensors / TurboMind)."""
    return {
        "basic": LMDEPLOY_BASIC,
        "advanced": LMDEPLOY_ADVANCED,
    }


def get_param_registry(engine: str = "llama_cpp") -> Dict[str, List[ParamDef]]:
    """Return param registry for the given engine."""
    if engine == "llama_cpp":
        return get_llama_cpp_param_registry()
    if engine == "ik_llama":
        return get_ik_llama_param_registry()
    if engine == "lmdeploy":
        return get_lmdeploy_param_registry()
    return {"basic": [], "advanced": []}
