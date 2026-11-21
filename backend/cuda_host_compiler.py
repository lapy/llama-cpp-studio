"""Pick a CUDA host compiler the installed toolkit actually supports."""

from __future__ import annotations

import os
import re
import shutil
from typing import Optional, Tuple

# nvcc rejects host compilers newer than the toolkit's supported ceiling. The
# image can carry several GCC versions (26.04 ships GCC 12/15 and adds 14 for
# CUDA), so Studio must choose one per toolkit instead of using unversioned
# ``g++`` (which is the distro default and may be too new).
_VERSION_RE = re.compile(r"cuda-(\d+)\.(\d+)(?:\.(\d+))?")


def cuda_toolkit_version(cuda_root: str) -> Optional[Tuple[int, int]]:
    """Return ``(major, minor)`` from a managed ``.../cuda-12.8.2`` path."""
    match = _VERSION_RE.search(os.path.basename(str(cuda_root or "")))
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def max_supported_gcc(cuda_version: Optional[Tuple[int, int]]) -> int:
    """Newest GCC major nvcc accepts for this CUDA version.

    Mirrors NVIDIA's host-compiler tables: CUDA 12.0-12.3 supports GCC 12,
    12.4-12.6 supports GCC 13, 12.8-12.9 supports GCC 14, and 13.x supports
    GCC 15. Unknown versions stay conservative (GCC 12).
    """
    if cuda_version is None:
        return 12
    if cuda_version >= (13, 0):
        return 15
    if cuda_version >= (12, 8):
        return 14
    if cuda_version >= (12, 4):
        return 13
    return 12


def select_cuda_host_compiler(cuda_root: str) -> Optional[str]:
    """Absolute path of a supported ``g++`` for this toolkit, or ``None``.

    Versioned compilers are preferred so a distro's unversioned ``g++`` cannot
    silently be newer than nvcc accepts. Only CUDA 13 (max GCC 15) may fall
    back to the unversioned binary.
    """
    ceiling = max_supported_gcc(cuda_toolkit_version(cuda_root))
    for major in range(ceiling, 11, -1):
        path = shutil.which(f"g++-{major}")
        if path:
            return path
    if ceiling >= 15:
        return shutil.which("g++")
    return None


__all__ = [
    "cuda_toolkit_version",
    "max_supported_gcc",
    "select_cuda_host_compiler",
]
