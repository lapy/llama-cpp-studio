"""Host-pass compatibility shim for glibc >= 2.41 with older CUDA headers."""

from __future__ import annotations

import os
import platform
import sys
from typing import Optional, Tuple

from backend.cuda_host_compiler import cuda_toolkit_version

SHIM_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "engines",
    "cuda",
    "glibc_c23_math_compat.h",
)

# CUDA 13.2 is the first toolkit documented to agree with glibc's C23 math
# declarations. CUDA <= 12.9 and 13.0/13.1 need the host-pass shim.
_FIXED_CUDA = (13, 2)


def host_glibc_version() -> Optional[Tuple[int, int]]:
    """Return the host glibc ``(major, minor)`` or ``None`` when not glibc."""
    name, version = platform.libc_ver()
    if name != "glibc" or not version:
        return None
    parts = version.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return None


def needs_cuda_c23_shim(
    glibc_version: Optional[Tuple[int, int]] = None,
    cuda_version: Optional[Tuple[int, int]] = None,
) -> bool:
    """True when CUDA's math headers clash with this glibc's C23 additions."""
    if not sys.platform.startswith("linux"):
        return False
    glibc = host_glibc_version() if glibc_version is None else glibc_version
    cuda = cuda_version
    if glibc is None:
        # Unknown libc on Linux: the header is a no-op where __GLIBC__ is not
        # defined, so including it is still safe.
        return True
    if glibc < (2, 41):
        return False
    return cuda is None or cuda < _FIXED_CUDA


def cuda_c23_shim_flag(cuda_root: str = "") -> str:
    """CMake CUDA flag force-including the shim into nvcc's host pass."""
    del cuda_root  # kept for symmetry with the host-compiler selector
    return f"-Xcompiler=-include,{SHIM_PATH}"


__all__ = [
    "SHIM_PATH",
    "cuda_c23_shim_flag",
    "host_glibc_version",
    "needs_cuda_c23_shim",
]
