"""Managed JIT/kernel caches kept inside an engine install directory.

Engine builds and the servers they launch compile kernels at first use: nvcc
JIT (``CUDA_CACHE_PATH``), Triton (``TRITON_CACHE_DIR``), TileLang
(``TILELANG_CACHE_DIR``), TVM-FFI (``TVM_FFI_CACHE_DIR``), FlashInfer
(``FLASHINFER_WORKSPACE_BASE``), TorchInductor (``TORCHINDUCTOR_CACHE_DIR``),
torch extension builds (``TORCH_EXTENSIONS_DIR``) and CuTe DSL
(``CUTE_DSL_CACHE_DIR``, which otherwise defaults to ``/tmp``). Left to their
defaults they land under ``$HOME`` or a temporary directory: shared between
engines and versions, invisible to Studio, and never invalidated - so a stale
entry can be reused by a different toolchain, and a cold container still pays
the full JIT cost on every start.

A definition therefore DECLARES the cache variables it cares about
(``runtime_profiles.<name>.env.caches`` for the server,
``build.prepare_fork.caches`` for the installer). Studio resolves each one to
``<install_dir>/caches/<var-slug>`` and creates it, so:

* the installer's build and smoke/precompile steps fill the same cache the
  server reads, which is what makes the first real start fast;
* a new install gets an empty cache by construction, and
  :func:`flush_caches` empties it for in-place rebuilds and source syncs;
* two engine versions never share compiled artifacts.

``PYTHON_JIT_CACHES`` / ``SGLANG_JIT_CACHES`` / ``VLLM_JIT_CACHES`` are the
Studio-owned baselines every shipped definition in that family must declare;
the engine definition files remain the editable data, and a test in
``backend/tests/test_engine_caches.py`` pins the coverage. Model-WEIGHT
caches (``HF_HOME``, ``HF_HUB_CACHE``, ``TRANSFORMERS_CACHE``,
``MODELSCOPE_CACHE``, ``TORCH_HOME``) are deliberately NOT moved: they hold
multi-GB downloads, not compiled artifacts, and belong to Studio's model
storage rather than to one engine version.
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Iterable, Mapping

CACHES_DIRNAME = "caches"

# Every path-valued compile/JIT cache a Python inference stack can read. The
# list stays a superset on purpose: an unused variable is inert, while a
# missing one silently falls back to $HOME (or /tmp for CuTe DSL) and is paid
# for again on every cold start.
PYTHON_JIT_CACHES: tuple[str, ...] = (
    "XDG_CACHE_HOME",
    "TRITON_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "TORCH_EXTENSIONS_DIR",
    "TVM_FFI_CACHE_DIR",
    "TILELANG_CACHE_DIR",
    "CUTE_DSL_CACHE_DIR",
    "FLASHINFER_WORKSPACE_BASE",
    "OUTLINES_CACHE_DIR",
    "DG_JIT_CACHE_DIR",
    "TIKTOKEN_CACHE_DIR",
    "DATA_GYM_CACHE_DIR",
    "KERNELS_CACHE",
    "CUDA_CACHE_PATH",
)

# SGLang's own caches. Several have lazy defaults under SGLANG_CACHE_DIR, but
# SGLANG_JIT_CACHE_DIR is hard-coded to ~/.cache/sglang/jit, the CuTe AOT /
# DeepGEMM roots are resolved at import, and the V100/SM70 forks build some
# torch extensions into /tmp paths of their own, so each is declared
# explicitly. Variables a fork does not read are inert.
SGLANG_JIT_CACHES: tuple[str, ...] = PYTHON_JIT_CACHES + (
    "SGLANG_CACHE_DIR",
    "SGLANG_DIFFUSION_CACHE_ROOT",
    "SGLANG_DIFFUSION_HOST_SPILL_DIR",
    "SGLANG_CUTE_AOT_CACHE_DIR",
    "SGLANG_DG_CACHE_DIR",
    "SGLANG_JIT_CACHE_DIR",
    "SGLANG_QWEN4_PLE_FILE_DIR",
    "SGLANG_SM70_GLM_NVFP4_BUILD_DIR",
    "SGLANG_SM70_QPN8_BUILD_DIR",
    "SGLANG_V100_DECODE_CUDA_BUILD_DIR",
    "SGLANG_V100_NVFP4_MOE_BUILD_DIR",
    "HUMMING_CACHE_DIR",
    "HUMMING_TMP_DIR",
)

# vLLM's compile root and the caches it derives from that root (XLA, FlashInfer
# autotune, DeepGEMM) plus its asset cache. VLLM_MEDIA_CACHE and
# VLLM_LORA_RESOLVER_CACHE_DIR are intentionally absent: both are disabled
# unless a user opts in, so pointing them at an engine path would turn caches
# ON rather than move an existing one.
VLLM_JIT_CACHES: tuple[str, ...] = PYTHON_JIT_CACHES + (
    "VLLM_CACHE_ROOT",
    "VLLM_ASSETS_CACHE",
    "VLLM_XLA_CACHE_PATH",
    "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR",
    # vLLM creates this itself for the AMD FlyDSL runtime; declaring it keeps
    # the ROCm JIT cache beside the install too.
    "FLYDSL_RUNTIME_CACHE_DIR",
)

# Native CUDA servers keep compiled kernels in the driver's compute cache;
# XDG_CACHE_HOME also covers driver/OpenGL/Vulkan shader caches and (for
# audio.cpp) its eSpeak extraction cache. Both are process-scoped, so nothing
# outside the launched server moves.
NATIVE_JIT_CACHES: tuple[str, ...] = ("XDG_CACHE_HOME", "CUDA_CACHE_PATH")

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
    return {str(name): cache_path(root, str(name)) for name in names or () if str(name).strip()}


def ensure_cache_dirs(install_dir: str, names: Iterable[str]) -> None:
    """Create the declared cache directories (best effort; launch may be read-only)."""
    for path in cache_environment(install_dir, names).values():
        try:
            os.makedirs(path, exist_ok=True)
        except OSError:
            continue


def declared_caches(env_spec: Mapping | None) -> tuple[str, ...]:
    """Cache variable names a runtime profile's ``env`` block declares."""
    env = env_spec if isinstance(env_spec, Mapping) else {}
    return tuple(str(name) for name in (env.get("caches") or ()) if str(name).strip())


def install_dir_for(active: Mapping | None) -> str:
    """Directory a managed cache tree belongs to for the ACTIVE install row.

    Installed rows carry ``install_dir`` for every install kind. The legacy
    venv fallback keeps rows written before that field existed working: the
    install directory is the venv's parent.
    """
    row = active if isinstance(active, Mapping) else {}
    install_dir = str(row.get("install_dir") or "")
    if not install_dir:
        venv_path = str(row.get("venv_path") or "")
        if venv_path:
            install_dir = os.path.dirname(venv_path)
    return install_dir


def resolve_profile_caches(env_spec: Mapping | None, active: Mapping | None) -> dict[str, str]:
    """Managed cache env for a launch profile and active install row.

    Creates the directories as a side effect so a first start writes into a
    real directory. Returns ``{}`` when the profile declares nothing or the row
    has no install directory (nothing safe to anchor the tree to).
    """
    names = declared_caches(env_spec)
    if not names:
        return {}
    install_dir = install_dir_for(active)
    if not install_dir:
        return {}
    ensure_cache_dirs(install_dir, names)
    return cache_environment(install_dir, names)


def flush_caches(install_dir: str) -> None:
    """Drop every managed cache for an install (in-place rebuild / source sync)."""
    root = caches_root(install_dir) if str(install_dir or "") else ""
    if not root:
        return
    shutil.rmtree(root, ignore_errors=True)
