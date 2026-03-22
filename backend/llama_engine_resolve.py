"""Resolve active llama.cpp / ik_llama binary paths from the engines store (lightweight; no hf/swap imports)."""

from __future__ import annotations

import os
from typing import Any, Optional

from backend.logging_config import get_logger

logger = get_logger(__name__)


def abs_llama_binary_path(p: Optional[str]) -> str:
    if not p:
        return ""
    if os.path.isabs(p):
        return p
    return os.path.join("/app", p.lstrip("/"))


def get_active_llama_swap_binary_path(store: Any) -> Optional[str]:
    """
    Same resolution as llama-swap: first existing ``binary_path`` on active ``llama_cpp``,
    else on active ``ik_llama``.
    """
    try:
        for engine in ("llama_cpp", "ik_llama"):
            active_version = store.get_active_engine_version(engine)
            if not active_version or not active_version.get("binary_path"):
                continue
            binary_path = active_version["binary_path"]
            if not os.path.isabs(binary_path):
                binary_path = os.path.join("/app", binary_path)
            if os.path.exists(binary_path):
                return binary_path
            abs_path = os.path.abspath(binary_path)
            if os.path.exists(abs_path):
                return abs_path
        logger.warning("No active llama-cpp version found in data store")
        return None
    except Exception as e:
        logger.error("Error getting active llama swap binary path: %s", e)
        return None


def infer_llama_engine_for_binary(store: Any, binary_path: str) -> str:
    """Return ``llama_cpp`` or ``ik_llama`` depending on which active row references this path."""
    try:
        norm = os.path.abspath(abs_llama_binary_path(binary_path))
        for eng in ("ik_llama", "llama_cpp"):
            av = store.get_active_engine_version(eng)
            if av and av.get("binary_path"):
                if os.path.abspath(abs_llama_binary_path(av["binary_path"])) == norm:
                    return eng
    except Exception as e:
        logger.debug("infer_llama_engine_for_binary: %s", e)
    return "llama_cpp"
