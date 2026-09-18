"""Vanilla vLLM manager using Studio's versioned Python-engine lifecycle."""

from typing import Optional

from backend.sglang_manager import SglangManager


_manager_instance: Optional["VllmManager"] = None


class VllmManager(SglangManager):
    def __init__(
        self,
        *,
        log_path: Optional[str] = None,
        base_dir: Optional[str] = None,
    ) -> None:
        super().__init__("vllm", log_path=log_path, base_dir=base_dir)


def get_vllm_manager() -> VllmManager:
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = VllmManager()
    return _manager_instance


__all__ = ["VllmManager", "get_vllm_manager"]
