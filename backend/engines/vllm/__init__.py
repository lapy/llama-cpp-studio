"""vLLM family: upstream vLLM and the 1Cat-vLLM fork."""

from backend.engines.vllm.installer import VllmInstaller, VllmManager, get_vllm_manager
from backend.engines.vllm.onecat import (
    ENGINE_ID as ONECAT_ENGINE_ID,
    GITHUB_REPO,
    OneCatVllmInstaller,
    OneCatVllmManager,
    get_onecat_vllm_manager,
)

__all__ = [
    "GITHUB_REPO",
    "ONECAT_ENGINE_ID",
    "OneCatVllmInstaller",
    "OneCatVllmManager",
    "VllmInstaller",
    "VllmManager",
    "get_onecat_vllm_manager",
    "get_vllm_manager",
]
