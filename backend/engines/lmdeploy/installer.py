"""LMDeploy compatibility installer — definition-bound, no engine logic.

The former class owned LMDeploy's binary-resolution hook (``LMDEPLOY_BIN``)
plus a venv/pip shell. Both are gone: the override is the generic
``lifecycle.binary_override_env`` mechanism in
:class:`backend.engines.python.generic.GenericVenvInstaller`, and every flow is
the definition's recipes. What remains here is a constructor shim for call
sites/tests that still instantiate ``LMDeployManager(...)`` without supplying
the definition. It adds no decision of its own.
"""

from __future__ import annotations

from typing import Optional

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller, get_definition_installer

_MANAGER_INSTANCE: Optional["LMDeployInstaller"] = None


def get_lmdeploy_manager() -> "LMDeployInstaller":
    """Singleton accessor kept for older call sites."""
    global _MANAGER_INSTANCE
    if _MANAGER_INSTANCE is None:
        _MANAGER_INSTANCE = LMDeployInstaller()
    return _MANAGER_INSTANCE


class LMDeployInstaller(GenericVenvInstaller):
    """Compatibility constructor; behaves exactly like the shared installer."""

    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        state_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        definition = load_default_engine_definitions().definitions.get("lmdeploy")
        if definition is None:
            raise RuntimeError("lmdeploy engine definition is unavailable")
        super().__init__(
            definition,
            log_path=log_path,
            base_dir=base_dir,
            state_path=state_path,
        )


# Backwards-compatible alias (tests and older call sites import this name).
LMDeployManager = LMDeployInstaller


__all__ = [
    "LMDeployInstaller",
    "LMDeployManager",
    "GenericVenvInstaller",
    "get_definition_installer",
    "get_lmdeploy_manager",
]
