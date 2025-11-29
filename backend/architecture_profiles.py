"""
Architecture-aware profiles for interpreting GGUF metadata.

Each profile is responsible for turning raw GGUF metadata into:
- block_count: architectural depth (number of transformer blocks)
- effective_layer_count: layers llama.cpp can offload (including output layer)
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ArchitectureProfile:
    """
    Base class for architecture-specific GGUF metadata interpretation.
    """

    names: Tuple[str, ...]

    def matches(self, architecture: str) -> bool:
        arch = architecture.lower()
        return any(arch == n or arch.startswith(n) for n in self.names)

    def compute_block_and_effective_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> Dict[str, int]:
        """
        Compute block_count and effective_layer_count for this architecture.

        base_block_count is a pre-computed best-effort value from generic
        logic (e.g. _extract_layer_count) and can be used as a fallback.
        """
        raise NotImplementedError


class Glm4MoeProfile(ArchitectureProfile):
    def __init__(self) -> None:
        super().__init__(names=("glm4moe",))

    def compute_block_and_effective_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> Dict[str, int]:
        glm_block_count = metadata.get("glm4moe.block_count")
        nextn_layers = metadata.get("glm4moe.nextn_predict_layers")

        block_count = base_block_count
        if isinstance(glm_block_count, (int, float)) and glm_block_count > 0:
            block_count = int(glm_block_count)

        effective_layer_count = block_count or 0
        if isinstance(nextn_layers, (int, float)) and nextn_layers > 0:
            effective_layer_count = int(effective_layer_count) + int(nextn_layers)

        logger.info(
            "glm4moe profile: block_count=%s, effective_layer_count=%s "
            "(base_block_count=%s, glm_block_count=%s, nextn_predict_layers=%s)",
            block_count,
            effective_layer_count,
            base_block_count,
            glm_block_count,
            nextn_layers,
        )

        return {
            "block_count": int(block_count) if block_count else 0,
            "effective_layer_count": int(effective_layer_count)
            if effective_layer_count
            else 0,
        }


class GlmProfile(ArchitectureProfile):
    """Profile for GLM models (non-MoE variants like GLM-Z1, GLM-4)."""
    def __init__(self) -> None:
        super().__init__(names=("glm", "glm4"))

    def compute_block_and_effective_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> Dict[str, int]:
        # Try GLM-specific keys first (check both glm.* and glm4.*)
        glm_block_count = (
            metadata.get("glm4.block_count")
            or metadata.get("glm.block_count")
            or metadata.get("glm4.layer_count")
            or metadata.get("glm.layer_count")
            or metadata.get("glm4.n_layer")
            or metadata.get("glm.n_layer")
        )
        
        # Also check for nextn_predict_layers (used in some GLM variants)
        nextn_layers = (
            metadata.get("glm4.nextn_predict_layers")
            or metadata.get("glm.nextn_predict_layers")
        )

        block_count = base_block_count
        if isinstance(glm_block_count, (int, float)) and glm_block_count > 0:
            block_count = int(glm_block_count)

        # For GLM models, effective_layer_count = block_count (no +1 like Llama)
        # Some GLM variants have nextn_predict_layers that should be added
        effective_layer_count = block_count or 0
        if isinstance(nextn_layers, (int, float)) and nextn_layers > 0:
            effective_layer_count = int(effective_layer_count) + int(nextn_layers)

        logger.info(
            "glm profile: block_count=%s, effective_layer_count=%s "
            "(base_block_count=%s, glm_block_count=%s, nextn_predict_layers=%s)",
            block_count,
            effective_layer_count,
            base_block_count,
            glm_block_count,
            nextn_layers,
        )

        return {
            "block_count": int(block_count) if block_count else 0,
            "effective_layer_count": int(effective_layer_count)
            if effective_layer_count
            else 0,
        }


class LlamaLikeProfile(ArchitectureProfile):
    """
    LLaMA-style decoder-only LMs (llama, mistral, mixtral, gemma, etc.).

    These typically have:
    - N transformer blocks (llama.block_count)
    - separate output head that llama.cpp counts as an additional layer.
    """

    def __init__(self) -> None:
        super().__init__(names=("llama", "mistral", "mixtral", "gemma"))

    def compute_block_and_effective_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> Dict[str, int]:
        block_count = (
            metadata.get("llama.block_count")
            or metadata.get("llama.n_layer")
            or base_block_count
            or 0
        )
        if isinstance(block_count, (int, float)):
            block_count = int(block_count)
        else:
            block_count = 0

        # Offloadable layers = transformer stack + output head
        effective_layer_count = block_count + 1 if block_count > 0 else 0

        logger.info(
            "LLaMA-like profile: block_count=%s, effective_layer_count=%s "
            "(base_block_count=%s)",
            block_count,
            effective_layer_count,
            base_block_count,
        )

        return {
            "block_count": block_count,
            "effective_layer_count": effective_layer_count,
        }


class QwenFamilyProfile(ArchitectureProfile):
    """
    Qwen / Qwen2 / Qwen3 / Qwen3-MoE decoder LMs.
    """

    def __init__(self) -> None:
        super().__init__(names=("qwen", "qwen2", "qwen3", "qwen3moe"))

    def compute_block_and_effective_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> Dict[str, int]:
        block_count = (
            metadata.get("qwen.block_count")
            or metadata.get("qwen2.block_count")
            or metadata.get("qwen3.block_count")
            or metadata.get("qwen3moe.block_count")
            or metadata.get("qwen.n_layer")
            or base_block_count
            or 0
        )
        if isinstance(block_count, (int, float)):
            block_count = int(block_count)
        else:
            block_count = 0

        effective_layer_count = block_count + 1 if block_count > 0 else 0

        logger.info(
            "Qwen-family profile: block_count=%s, effective_layer_count=%s "
            "(base_block_count=%s)",
            block_count,
            effective_layer_count,
            base_block_count,
        )

        return {
            "block_count": block_count,
            "effective_layer_count": effective_layer_count,
        }


class DeepseekProfile(ArchitectureProfile):
    """
    DeepSeek decoder LMs and MoE variants.
    """

    def __init__(self) -> None:
        super().__init__(names=("deepseek",))

    def compute_block_and_effective_layers(
        self,
        metadata: Dict[str, Any],
        base_block_count: int,
    ) -> Dict[str, int]:
        block_count = (
            metadata.get("deepseek.block_count")
            or metadata.get("deepseek.n_layer")
            or base_block_count
            or 0
        )
        if isinstance(block_count, (int, float)):
            block_count = int(block_count)
        else:
            block_count = 0

        effective_layer_count = block_count + 1 if block_count > 0 else 0

        logger.info(
            "DeepSeek profile: block_count=%s, effective_layer_count=%s "
            "(base_block_count=%s)",
            block_count,
            effective_layer_count,
            base_block_count,
        )

        return {
            "block_count": block_count,
            "effective_layer_count": effective_layer_count,
        }


PROFILES: List[ArchitectureProfile] = [
    Glm4MoeProfile(),
    GlmProfile(),  # Must come after Glm4MoeProfile so glm4moe matches first
    LlamaLikeProfile(),
    QwenFamilyProfile(),
    DeepseekProfile(),
]


def find_profile(architecture: str) -> Optional[ArchitectureProfile]:
    """
    Find a profile matching the given architecture string.
    """
    for profile in PROFILES:
        if profile.matches(architecture):
            return profile
    return None


def compute_layers_for_architecture(
    architecture: str,
    metadata: Dict[str, Any],
    base_block_count: int,
) -> Dict[str, int]:
    """
    Helper used by gguf_reader to compute block_count and effective_layer_count.

    If a concrete profile is found, uses it. Otherwise applies a generic
    decoder heuristic: effective_layer_count = base_block_count + 1 when
    a non-zero block count is available, or falls back to 32.
    """
    arch = architecture.lower()
    profile = find_profile(arch)

    if profile:
        return profile.compute_block_and_effective_layers(metadata, base_block_count)

    block_count = base_block_count or 0

    if block_count > 0:
        effective_layer_count = block_count + 1
        logger.info(
            "Generic profile: architecture=%s, block_count=%s, "
            "effective_layer_count=%s",
            arch,
            block_count,
            effective_layer_count,
        )
        return {
            "block_count": int(block_count),
            "effective_layer_count": int(effective_layer_count),
        }

    # Complete fallback: nothing useful in metadata. Keep existing behaviour
    # of assuming a 32-layer model, but log it clearly.
    logger.warning(
        "Could not determine block_count for architecture=%s; "
        "using default effective_layer_count=32",
        arch,
    )
    return {"block_count": 0, "effective_layer_count": 32}


