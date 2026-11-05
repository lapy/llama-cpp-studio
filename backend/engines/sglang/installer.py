"""SGLang family accessors.

Both SGLang variants are definition-owned and run on the shared
``GenericVenvInstaller``. The fork's upstream rewrites are versioned transform
assets applied by the generic ``vcs.apply_transform`` primitive and its build
environment is declared through ``build.prepare_fork``; no engine-specific
installer subclass or mixin remains.
"""

from __future__ import annotations

from typing import Optional

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller, get_definition_installer

SGLANG_ENGINE_IDS = frozenset({"sglang", "sglang_v100"})


def _safe_variant(engine_id: str) -> str:
    value = str(engine_id or "").strip()
    if value not in SGLANG_ENGINE_IDS:
        raise ValueError(f"Unknown SGLang engine: {engine_id}")
    return value


def get_sglang_manager(engine_id: str = "sglang") -> GenericVenvInstaller:
    return get_definition_installer(_safe_variant(engine_id))


def get_sglang_upstream_manager() -> GenericVenvInstaller:
    return get_definition_installer("sglang")


def get_sglang_v100_manager() -> GenericVenvInstaller:
    return get_definition_installer("sglang_v100")


class SglangManager:
    """Compatibility facade: binds the variant's definition to the shared host."""

    def __new__(
        cls,
        engine_id: str = "sglang",
        *,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> GenericVenvInstaller:
        variant = _safe_variant(engine_id)
        definition = load_default_engine_definitions().definitions.get(variant)
        if definition is None:
            raise RuntimeError(f"{variant} engine definition is unavailable")
        return GenericVenvInstaller(definition, log_path=log_path, base_dir=base_dir)


SglangInstaller = SglangManager
