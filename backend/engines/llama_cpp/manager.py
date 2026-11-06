"""Compatibility re-exports for the generic CMake build host."""

from backend.engines.python.cmake_build import (
    BuildConfig,
    CmakeBuildHost,
    LlamaCppManager,
)

LlamaManager = CmakeBuildHost

__all__ = ["BuildConfig", "CmakeBuildHost", "LlamaCppManager", "LlamaManager"]
