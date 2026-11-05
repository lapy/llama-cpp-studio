"""Test helpers for the generic release host."""

from __future__ import annotations

from typing import Optional

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.release_host import GenericReleaseInstaller, get_release_installer

ENGINE_ID = "unsloth_llama"


class UnslothLlamaInstaller(GenericReleaseInstaller):
    def __init__(self, *, base_dir: Optional[str] = None) -> None:
        definition = load_default_engine_definitions().definitions[ENGINE_ID]
        super().__init__(definition, base_dir=base_dir)


def get_unsloth_llama_manager() -> GenericReleaseInstaller:
    return get_release_installer(ENGINE_ID)
