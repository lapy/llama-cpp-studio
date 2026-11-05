"""Compatibility accessors for the definition-owned 1Cat-vLLM engine.

1Cat-vLLM has no installer subclass: its recipes run on the shared
``GenericVenvInstaller``. The GitHub-wheel selection, Rust toolchain
provisioning, requirement fallbacks and wheel building are reusable primitives
(``prereqs.resolve_release_assets``, ``toolchain.ensure``,
``pip.install_requirements``, ``artifact.build_wheels``) driven by the
definition. The names below remain only for older call sites and tests.
"""

from __future__ import annotations

from typing import Optional

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller, get_definition_installer

ENGINE_ID = "1cat_vllm"
GITHUB_REPO = "1CatAI/1Cat-vLLM"
DEFAULT_SOURCE_REPO = "https://github.com/1CatAI/1Cat-vLLM.git"


def get_onecat_vllm_manager() -> GenericVenvInstaller:
    return get_definition_installer(ENGINE_ID)


class OneCatVllmInstaller(GenericVenvInstaller):
    """Compatibility constructor; binds the definition to the shared host."""

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        definition = load_default_engine_definitions().definitions.get(ENGINE_ID)
        if definition is None:
            raise RuntimeError(f"{ENGINE_ID} engine definition is unavailable")
        super().__init__(
            definition,
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
        )
        self.MANAGER_NAME = "onecat_vllm"


OneCatInstaller = OneCatVllmInstaller  # historical name
OneCatVllmManager = OneCatVllmInstaller
