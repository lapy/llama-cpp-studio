"""Managed JIT/kernel caches kept inside an engine install directory.

Engine builds and the servers they launch compile kernels at first use: nvcc
JIT (``CUDA_CACHE_PATH``), Triton (``TRITON_CACHE_DIR``), TileLang
(``TILELANG_CACHE_DIR``), TVM-FFI (``TVM_FFI_CACHE_DIR``), FlashInfer
(``SGLANG_CACHE_DIR``), TorchInductor (``TORCHINDUCTOR_CACHE_DIR``) and torch
extension builds (``TORCH_EXTENSIONS_DIR``). Left to their defaults they land
under ``$HOME``: shared between engines and versions, invisible to Studio, and
never invalidated when a build changes - so a stale entry can be reused by a
different toolchain, and a fresh install still pays the full JIT cost.

A definition therefore DECLARES the cache variables it cares about
(``runtime_profiles.<name>.env.caches`` for the server,
``build.prepare_fork.caches`` for the installer). Studio resolves each one to
``<install_dir>/caches/<var-slug>`` and creates it, so:

* the installer's smoke/precompile step fills the same cache the server reads,
  which is what makes the first real start fast;
* a new install gets an empty cache by construction, and
  :func:`flush_caches` empties it for in-place rebuilds and source syncs;
* two engine versions never share compiled artifacts.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Iterable, Mapping

CACHES_DIRNAME = "caches"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def cache_slug(name: str) -> str:
    """Directory name for a cache variable (``TRITON_CACHE_DIR`` -> ``triton-cache-dir``)."""
    slug = _SLUG_RE.sub("-", str(name).strip().lower()).strip("-")
    return slug or "cache"


def caches_root(install_dir: str) -> str:
    return os.path.join(str(install_dir), CACHES_DIRNAME)


def cache_path(install_dir: str, name: str) -> str:
    return os.path.join(caches_root(install_dir), cache_slug(name))


def cache_environment(install_dir: str, names: Iterable[str]) -> dict[str, str]:
    """Environment entries pointing each declared cache at the install dir."""
    root = str(install_dir or "")
    if not root:
        return {}
    return {
        str(name): cache_path(root, str(name))
        for name in names or ()
        if str(name).strip()
    }


def ensure_cache_dirs(install_dir: str, names: Iterable[str]) -> None:
    """Create the declared cache directories (best effort; launch may be read-only)."""
    for path in cache_environment(install_dir, names).values():
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            continue


def flush_caches(install_dir: str) -> None:
    """Drop every managed cache for an install (in-place rebuild / source sync)."""
    root = caches_root(install_dir) if str(install_dir or "") else ""
    if not root:
        return
    shutil.rmtree(root, ignore_errors=True)
