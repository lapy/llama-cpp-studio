"""Compatibility accessor for the definition-owned upstream vLLM engine.

vLLM has no Python installer implementation: the definition owns identity,
recipes, lifecycle (including the declared status projection) and launch.
``get_vllm_manager`` returns the shared definition installer so older call
sites and tests keep a stable import.
"""

from __future__ import annotations

from backend.engines.python.generic import GenericVenvInstaller, get_definition_installer


def get_vllm_manager() -> GenericVenvInstaller:
    return get_definition_installer("vllm")


__all__ = ["get_vllm_manager"]
