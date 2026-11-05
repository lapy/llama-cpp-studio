"""Compatibility surface for the definition-owned LMDeploy engine.

LMDeploy has no Python engine implementation any more: its identity, settings,
recipes, lifecycle, discovery and launch profile live in
``backend/engines/definitions/lmdeploy.yaml``. The names below remain solely
for older call sites and tests; deletion is blocked only by those consumers
(production routing and adapters resolve the shared definition installer).
"""

from backend.engines.lmdeploy.installer import (
    LMDeployInstaller,
    LMDeployManager,
    get_lmdeploy_manager,
)

__all__ = ["LMDeployInstaller", "LMDeployManager", "get_lmdeploy_manager"]
