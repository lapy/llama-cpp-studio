"""Which CUDA toolkit a native build used, recorded once when the build starts.

A native source build inherits a CUDA toolkit from one of two places: the
Studio-managed installation under ``<data>/cuda``, which the CMake build
handlers bind to whenever one exists (``llama.cpp`` passes the validated root
explicitly, ``audio.cpp`` overrides ``CUDA_HOME``/``CUDA_PATH`` from
``CUDAInstaller.get_cuda_env()``), or the ``CUDA_HOME``/``CUDA_PATH`` the
process inherited when Studio has no managed toolkit. This module resolves
that environment ONCE, when the llama-family source build starts, and stores
it in the build task's metadata so the task detail reports the toolkit instead
of it being guessed from whatever happens to be installed later.

The managed-first priority mirrors the build environment: the same toolkit
wins for the spawned build commands. Recording is deliberately total — a build
must never fail or slow down because provenance could not be recorded, and no
subprocess is spawned: the version is parsed from the toolkit path (Studio
installs are ``cuda-<x.y[.z]>`` directories).
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Mapping, Optional

from backend.paths import studio_data_dir

__all__ = ["record_cuda_provenance", "resolve_cuda_provenance"]

STUDIO_MANAGED = "studio-managed"
SYSTEM = "system"
UNKNOWN = "unknown"

# Studio installs live in ``<data>/cuda/cuda-<version>``; system toolkits commonly
# live in ``/usr/local/cuda-<version>``. Both name the version in the directory.
_VERSION_DIR = re.compile(r"^cuda[-_]?(\d+(?:\.\d+){1,2})$")
_NVCC_NAMES = ("nvcc", "nvcc.exe")


def _has_nvcc(home: str) -> bool:
    return any(os.path.isfile(os.path.join(home, "bin", name)) for name in _NVCC_NAMES)


def _version_key(entry: str) -> tuple:
    match = _VERSION_DIR.match(entry)
    if not match:
        return ()
    return tuple(int(part) for part in match.group(1).split("."))


def _managed_cuda_home() -> Optional[str]:
    """The Studio-managed toolkit the CMake build handlers bind to.

    Mirrors ``CUDAInstaller._get_cuda_path`` discovery without its state file:
    the ``current`` pointer activation maintains, then the newest complete
    ``cuda-*`` install. The native build environment binds these explicitly, so
    this is checked before the inherited environment.
    """
    root = os.path.join(studio_data_dir(), "cuda")
    current = os.path.join(root, "current")
    try:
        pointed = os.path.realpath(current)
    except (OSError, ValueError):
        pointed = ""
    if pointed and os.path.isdir(pointed) and _has_nvcc(pointed):
        return pointed
    try:
        entries = [
            entry for entry in os.listdir(root) if entry != "current" and _VERSION_DIR.match(entry)
        ]
    except OSError:
        return None
    for entry in sorted(entries, key=_version_key, reverse=True):
        candidate = os.path.join(root, entry)
        if os.path.isdir(candidate) and _has_nvcc(candidate):
            return os.path.realpath(candidate)
    return None


def _absolute(path: str) -> str:
    expanded = os.path.expanduser(str(path).strip())
    resolved = os.path.realpath(expanded)
    return resolved if os.path.exists(resolved) else os.path.abspath(expanded)


def _is_studio_managed(home: str) -> bool:
    root = os.path.realpath(os.path.join(studio_data_dir(), "cuda"))
    try:
        return os.path.commonpath([root, os.path.realpath(home)]) == root
    except ValueError:  # different drives (Windows): not comparable
        return False


def _cuda_version(home: str) -> Optional[str]:
    for part in reversed(re.split(r"[/\\]+", home.rstrip("/\\"))):
        match = _VERSION_DIR.match(part)
        if match:
            return match.group(1)
    return None


def resolve_cuda_provenance(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """Resolve the toolkit a native build's spawned commands inherit.

    ``env`` defaults to this process's environment; callers may pass the build
    environment they are about to spawn. Returns ``cuda_home`` (absolute, or
    ``None``), ``cuda_version`` (parsed from the path, or ``None``) and
    ``source`` (``studio-managed``, ``system`` or ``unknown``). Never raises.
    """
    try:
        environ = dict(os.environ if env is None else env)
        home = _managed_cuda_home()
        if home is None:
            inherited = str(environ.get("CUDA_HOME") or environ.get("CUDA_PATH") or "").strip()
            if inherited:
                home = _absolute(inherited)
        if not home:
            return {"cuda_home": None, "cuda_version": None, "source": UNKNOWN}
        return {
            "cuda_home": home,
            "cuda_version": _cuda_version(home),
            "source": STUDIO_MANAGED if _is_studio_managed(home) else SYSTEM,
        }
    except Exception:  # noqa: BLE001 - provenance must never fail a build
        return {"cuda_home": None, "cuda_version": None, "source": UNKNOWN}


def record_cuda_provenance(
    progress_manager: Any,
    task_id: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    """Record the resolved toolkit in the build task's metadata.

    Returns the recorded ``{"cuda_home", "cuda_version", "source"}`` mapping, or
    ``None`` when there is no retained task to attach it to. The task detail
    (SSE ``task_updated``/``task_snapshot`` payloads and the retained-task
    route) serializes ``metadata`` as-is, so this is the only write.
    """
    if not progress_manager or not task_id:
        return None
    provenance = resolve_cuda_provenance(env)
    progress_manager.update_task(task_id, metadata_update={"cuda": provenance})
    return provenance
