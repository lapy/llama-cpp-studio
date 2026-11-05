"""Test-only constructors for venv engines.

Production selects these engines exclusively through their YAML lifecycle;
these names preserve the historical test setup without keeping per-engine
compatibility modules in ``backend.engines``.
"""

from __future__ import annotations

from typing import Optional

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller, get_definition_installer

ENGINE_ID = "1cat_vllm"
GITHUB_REPO = "1CatAI/1Cat-vLLM"


def _installer(engine_id: str, **kwargs) -> GenericVenvInstaller:
    definition = load_default_engine_definitions().definitions.get(engine_id)
    if definition is None:
        raise RuntimeError(f"{engine_id} engine definition is unavailable")
    return GenericVenvInstaller(definition, **kwargs)


class LMDeployInstaller(GenericVenvInstaller):
    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            load_default_engine_definitions().definitions["lmdeploy"],
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
        )


LMDeployManager = LMDeployInstaller


class OneCatVllmInstaller(GenericVenvInstaller):
    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        super().__init__(
            load_default_engine_definitions().definitions["1cat_vllm"],
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
        )
        self.MANAGER_NAME = "onecat_vllm"


OneCatVllmManager = OneCatVllmInstaller


class SglangManager:
    def __new__(
        cls,
        engine_id: str = "sglang",
        *,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> GenericVenvInstaller:
        if engine_id not in {"sglang", "sglang_v100"}:
            raise ValueError(f"Unknown SGLang engine: {engine_id}")
        return _installer(engine_id, log_path=log_path, base_dir=base_dir)


SglangInstaller = SglangManager


def get_lmdeploy_manager() -> GenericVenvInstaller:
    return get_definition_installer("lmdeploy")


def get_onecat_vllm_manager() -> GenericVenvInstaller:
    return get_definition_installer("1cat_vllm")


def get_sglang_manager(engine_id: str = "sglang") -> GenericVenvInstaller:
    if engine_id not in {"sglang", "sglang_v100"}:
        raise ValueError(f"Unknown SGLang engine: {engine_id}")
    return get_definition_installer(engine_id)


def get_sglang_upstream_manager() -> GenericVenvInstaller:
    return get_sglang_manager("sglang")


def get_sglang_v100_manager() -> GenericVenvInstaller:
    return get_sglang_manager("sglang_v100")


def get_vllm_manager() -> GenericVenvInstaller:
    return get_definition_installer("vllm")
