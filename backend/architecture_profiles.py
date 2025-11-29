"""
Architecture-aware profiles for interpreting GGUF metadata.

Each profile is responsible for turning raw GGUF metadata into:
- block_count: architectural depth (number of transformer blocks)
- effective_layer_count: layers llama.cpp can offload (including output layer)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class LayerConfig:
    """Standardized output for layer calculations."""
    block_count: int
    effective_layer_count: int


# --- Helper Utilities ---

def _get_first_valid_int(
    metadata: Dict[str, Any], keys: List[str], default: Optional[int] = None
) -> Optional[int]:
    """
    Scans metadata for the first key that contains a valid >0 number.
    """
    for key in keys:
        val = metadata.get(key)
        # GGUF metadata values can be various numeric types
        if isinstance(val, (int, float)) and val > 0:
            return int(val)
    return default


# --- Registry System ---

_PROFILE_REGISTRY: List["ArchitectureProfile"] = []


def register_profile(cls: Type["ArchitectureProfile"]) -> Type["ArchitectureProfile"]:
    """
    Registers a profile class.
    Profiles are stored and later sorted by specificity (longest name match first).
    """
    _PROFILE_REGISTRY.append(cls())
    return cls


# --- Base Class ---

class ArchitectureProfile(ABC):
    """Base class for architecture-specific GGUF metadata interpretation."""

    def __init__(self, names: Tuple[str, ...]):
        self.names = names

    def matches(self, architecture: str) -> bool:
        """
        Checks if the architecture string matches this profile.
        """
        arch = architecture.lower()
        # "llama" should match "llama", "llama-2", etc.
        return any(arch == n or arch.startswith(n) for n in self.names)

    def compute(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> LayerConfig:
        """
        Public interface that wraps the calculation with standardized logging.
        """
        result = self._calculate_layers(metadata, base_block_count)
        
        logger.debug(
            "%s: matched. block_count=%s, effective_layer_count=%s (base=%s)",
            self.__class__.__name__,
            result.block_count,
            result.effective_layer_count,
            base_block_count,
        )
        return result

    @abstractmethod
    def _calculate_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> LayerConfig:
        """Implementation specific logic."""
        raise NotImplementedError


# --- Standard Profile (Handles 95% of cases) ---

class StandardDecoderProfile(ArchitectureProfile):
    """
    Generic profile for standard decoder-only models (Llama, Qwen, DeepSeek, etc.).
    
    Logic: 
    1. Look for specific keys (names.block_count, names.n_layer).
    2. Fallback to base_block_count.
    3. Effective layers = block_count + 1 (for the output head).
    
    Note on MoEs: Even for MoE models (Qwen2-MoE, DeepSeek-V2), llama.cpp 
    counts the 'offloadable layers' as the number of transformer blocks. 
    Expert offloading is managed internally within those blocks.
    """
    
    def _calculate_layers(
        self, metadata: Dict[str, Any], base_block_count: int
    ) -> LayerConfig:
        # Generate candidate keys based on architecture names
        # e.g., ["llama.block_count", "llama.n_layer", ...]
        candidate_keys = []
        for name in self.names:
            candidate_keys.extend([
                f"{name}.block_count", 
                f"{name}.n_layer",
                f"{name}.n_layers" # Some older models use plural
            ])
            
        block_count = _get_first_valid_int(
            metadata, candidate_keys, default=base_block_count
        ) or 0
        
        # Standard decoder: blocks + output head
        effective = (block_count + 1) if block_count > 0 else 0
        
        return LayerConfig(block_count=block_count, effective_layer_count=effective)


# --- Concrete Profiles ---

@register_profile
class GlmProfile(StandardDecoderProfile):
    """
    Profile for GLM family (GLM-4, GLM-4-MoE, etc.).
    """
    def __init__(self) -> None:
        super().__init__(names=("glm", "glm4", "glm4moe"))

    def _calculate_layers(
        self, metadata: Dict[str, Any], base_block_count: int
    ) -> LayerConfig:
        # GLM GGUFs often use 'glm4' or 'glm4moe' prefixes
        # Note: 'nextn_predict_layers' exists in metadata but typically resolves to 1
        # and is covered by the standard +1 output layer logic.
        return super()._calculate_layers(metadata, base_block_count)


@register_profile
class DeepseekProfile(StandardDecoderProfile):
    """
    DeepSeek decoder LMs and MoE variants.
    Crucial: Must check 'deepseek2' for V2/V3 models.
    """
    def __init__(self) -> None:
        super().__init__(names=("deepseek", "deepseek2"))


@register_profile
class QwenFamilyProfile(StandardDecoderProfile):
    """Qwen / Qwen2 / Qwen2.5 / Qwen2-MoE."""
    def __init__(self) -> None:
        super().__init__(names=("qwen", "qwen2", "qwen3", "qwen2moe", "qwen3moe"))


@register_profile
class SeedProfile(StandardDecoderProfile):
    """
    Profile for Seed OSS models (Seed-OSS-36B, etc.).
    Seed models are Llama-based but may use different metadata keys.
    Seed-OSS-36B has 64 transformer layers (not 32).
    """
    def __init__(self) -> None:
        super().__init__(names=("seed", "seed-oss", "seedoss"))

    def _calculate_layers(
        self, metadata: Dict[str, Any], base_block_count: int
    ) -> LayerConfig:
        # Seed models might use 'llama' prefix or 'seed' prefix
        # Check multiple candidate keys, prioritizing seed-specific keys
        candidate_keys = [
            "seed.block_count",
            "seed.n_layer",
            "seed.n_layers",
            "llama.block_count",
            "llama.n_layer",
            "llama.n_layers",
            "general.block_count",
            "general.n_layer",
        ]
        
        metadata_block_count = _get_first_valid_int(
            metadata, candidate_keys, default=None
        )
        
        # Use the maximum of metadata value or tensor-based count
        # This is important because Seed models might have incorrect metadata
        # but correct tensor counts
        if metadata_block_count is not None:
            block_count = max(metadata_block_count, base_block_count) if base_block_count > 0 else metadata_block_count
            if metadata_block_count < base_block_count:
                logger.info(
                    f"Seed profile: metadata shows {metadata_block_count} layers but tensor count suggests {base_block_count}, "
                    f"using tensor count ({base_block_count})"
                )
        else:
            block_count = base_block_count if base_block_count > 0 else 0
        
        # Special handling: If we detect around 32 layers but the model size suggests 64,
        # it might be a Seed model with incorrect metadata
        if block_count > 0 and block_count < 40:
            # Check if tensor count suggests more (Seed-OSS-36B should have ~64)
            if base_block_count > block_count * 1.5:
                logger.warning(
                    f"Seed profile: Detected potential mismatch - metadata={block_count}, "
                    f"tensor_count={base_block_count}. Using tensor_count."
                )
                block_count = base_block_count
        
        # Standard decoder: blocks + output head
        effective = (block_count + 1) if block_count > 0 else 0
        
        return LayerConfig(block_count=block_count, effective_layer_count=effective)


@register_profile
class LlamaLikeProfile(StandardDecoderProfile):
    """LLaMA, Mistral, Mixtral, Gemma, Phi, etc."""
    def __init__(self) -> None:
        # "phi" added as it follows the same decoder structure in GGUF
        super().__init__(names=("llama", "mistral", "mixtral", "gemma", "phi"))


# --- Main Accessor ---

def get_sorted_profiles() -> List[ArchitectureProfile]:
    """
    Returns profiles sorted by specificity (longest name match first).
    Example: 'glm4moe' (len 7) is checked before 'glm' (len 3).
    """
    return sorted(
        _PROFILE_REGISTRY,
        key=lambda p: max(len(n) for n in p.names),
        reverse=True
    )


def compute_layers_for_architecture(
    architecture: str,
    metadata: Dict[str, Any],
    base_block_count: int,
) -> Dict[str, int]:
    """
    Compute block_count and effective_layer_count.
    """
    arch = architecture.lower()
    
    # Iterate through automatically sorted profiles
    for profile in get_sorted_profiles():
        if profile.matches(arch):
            result = profile.compute(metadata, base_block_count)
            return {
                "block_count": result.block_count,
                "effective_layer_count": result.effective_layer_count
            }

    # --- Generic Fallback ---
    # Even if unknown architecture, if we have a base_block_count, 
    # it's safe to assume it's a decoder stack + 1 output head.
    block_count = base_block_count or 0

    if block_count > 0:
        effective_layer_count = block_count + 1
        logger.info(
            "Generic profile: architecture=%s, block_count=%s, "
            "effective_layer_count=%s",
            arch, block_count, effective_layer_count,
        )
        return {
            "block_count": block_count,
            "effective_layer_count": effective_layer_count,
        }

    # Complete fallback
    logger.warning(
        "Could not determine block_count for architecture=%s; "
        "using default effective_layer_count=32",
        arch,
    )
    return {"block_count": 0, "effective_layer_count": 32}