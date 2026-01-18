"""
Data models for smart_auto module.
Provides type-safe data classes to replace dictionary passing throughout the module.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple


@dataclass
class ModelMetadata:
    """Comprehensive model metadata extracted from GGUF file or name."""

    layer_count: int
    architecture: str
    context_length: int
    vocab_size: int
    embedding_length: int
    attention_head_count: int
    attention_head_count_kv: int
    block_count: int = 0
    is_moe: bool = False
    expert_count: int = 0
    experts_used_count: int = 0
    parameter_count: Optional[str] = None  # Formatted as "32B", "36B", etc.

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create ModelMetadata from a dictionary (e.g., from get_model_metadata result)."""
        return cls(
            layer_count=data.get("layer_count", 32),
            architecture=data.get("architecture", "unknown"),
            context_length=data.get("context_length", 0),
            vocab_size=data.get("vocab_size", 0),
            embedding_length=data.get("embedding_length", 0),
            attention_head_count=data.get("attention_head_count", 0),
            attention_head_count_kv=data.get("attention_head_count_kv", 0),
            block_count=data.get("block_count", 0),
            is_moe=data.get("is_moe", False),
            expert_count=data.get("expert_count", 0),
            experts_used_count=data.get("experts_used_count", 0),
            parameter_count=data.get("parameter_count"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {
            "layer_count": self.layer_count,
            "architecture": self.architecture,
            "context_length": self.context_length,
            "vocab_size": self.vocab_size,
            "embedding_length": self.embedding_length,
            "attention_head_count": self.attention_head_count,
            "attention_head_count_kv": self.attention_head_count_kv,
            "block_count": self.block_count,
            "is_moe": self.is_moe,
            "expert_count": self.expert_count,
            "experts_used_count": self.experts_used_count,
            "parameter_count": self.parameter_count,
        }


@dataclass
class SystemResources:
    """System resources information."""

    gpus: List[Dict[str, Any]]
    total_vram: int
    available_vram_gb: float
    gpu_count: int
    nvlink_topology: Dict[str, Any]
    cpu_cores: int
    cpu_memory_gb: Tuple[float, float, float]  # total, used, available
    flash_attn_available: bool = False
    compute_capabilities: List[float] = field(
        default_factory=list
    )  # Pre-parsed compute capabilities

    @classmethod
    def from_gpu_info(
        cls,
        gpu_info: Dict[str, Any],
        cpu_memory: Tuple[float, float, float],
        cpu_cores: int,
        flash_attn_available: bool = False,
    ) -> "SystemResources":
        """Create SystemResources from gpu_info and system data."""
        gpus = gpu_info.get("gpus", [])

        # Pre-parse compute capabilities to avoid repeated string parsing
        compute_capabilities = []
        for gpu in gpus:
            cc_str = gpu.get("compute_capability", "0.0")
            try:
                parts = str(cc_str).split(".")
                major = int(parts[0]) if parts and parts[0].isdigit() else 0
                minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
                compute_capabilities.append(major + minor / 10.0)
            except Exception:
                compute_capabilities.append(0.0)

        return cls(
            gpus=gpus,
            total_vram=gpu_info.get("total_vram", 0),
            available_vram_gb=(
                sum(gpu.get("memory", {}).get("free", 0) for gpu in gpus) / (1024**3)
                if gpus
                else 0.0
            ),
            gpu_count=gpu_info.get("device_count", 0),
            nvlink_topology=gpu_info.get("nvlink_topology", {}),
            cpu_cores=cpu_cores,
            cpu_memory_gb=cpu_memory,
            flash_attn_available=flash_attn_available,
            compute_capabilities=compute_capabilities,
        )


@dataclass
class GenerationConfig:
    """Complete generation configuration with type-safe fields."""

    # GPU configuration
    n_gpu_layers: int = 0
    main_gpu: int = 0
    tensor_split: str = ""
    flash_attn: bool = False

    # Memory and context
    ctx_size: int = 4096
    batch_size: int = 512
    ubatch_size: int = 256
    parallel: int = 1

    # CPU configuration
    threads: int = 4
    threads_batch: int = 4

    # Memory optimization
    no_mmap: bool = False
    mlock: bool = False
    low_vram: bool = False
    logits_all: bool = False
    cont_batching: bool = True
    no_kv_offload: bool = False

    # Generation parameters
    temperature: float = 0.8
    temp: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    typical_p: float = 1.0
    min_p: float = 0.0
    tfs_z: float = 1.0
    repeat_penalty: float = 1.1
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    mirostat: int = 0
    mirostat_tau: float = 5.0
    mirostat_eta: float = 0.1
    n_predict: int = -1
    stop: List[str] = field(default_factory=list)
    seed: int = -1

    # KV cache optimization
    cache_type_k: str = "f16"
    cache_type_v: Optional[str] = None

    # Architecture-specific
    rope_freq_base: Optional[float] = None
    rope_freq_scale: Optional[float] = None
    rope_scaling: str = ""
    yarn_ext_factor: float = 1.0
    yarn_attn_factor: float = 1.0

    # MoE configuration
    moe_offload_pattern: str = "none"
    moe_offload_custom: str = ""

    # Special flags
    embedding: bool = False
    jinja: bool = False

    # Server parameters
    host: str = "0.0.0.0"
    port: int = 0
    timeout: int = 300

    # Additional fields for backward compatibility
    yaml: str = ""
    customArgs: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for llama-swap integration."""
        result = {}
        for key, value in self.__dict__.items():
            # Skip None values to keep config clean
            if value is not None and value != []:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Create GenerationConfig from dictionary."""
        # Filter out only fields that exist in the dataclass
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in field_names}
        return cls(**filtered_data)

    def update(self, updates: Dict[str, Any]) -> "GenerationConfig":
        """Create a new config with updates applied."""
        new_dict = self.to_dict()
        new_dict.update(updates)
        # Filter out None values and empty lists
        new_dict = {k: v for k, v in new_dict.items() if v is not None and v != []}
        return self.from_dict(new_dict)
