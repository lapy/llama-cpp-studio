"""Shared runtime environment helpers for engine subprocesses."""

from __future__ import annotations

import os
from typing import Dict, List, Optional


def cuda_lib64_path() -> Optional[str]:
    try:
        from backend.cuda_installer import get_cuda_installer

        cuda_path = get_cuda_installer()._get_cuda_path()
        if not cuda_path:
            return None
        cuda_lib = os.path.join(cuda_path, "lib64")
        return cuda_lib if os.path.isdir(cuda_lib) else None
    except Exception:
        return None


def cuda_runtime_env() -> Dict[str, str]:
    try:
        from backend.cuda_installer import get_cuda_installer

        return get_cuda_installer().get_cuda_env() or {}
    except Exception:
        return {}


def merge_ld_library_path(*segments: Optional[str]) -> str:
    paths: List[str] = []
    seen: set[str] = set()
    for segment in segments:
        if not segment:
            continue
        for part in str(segment).split(os.pathsep):
            part = part.strip()
            if not part:
                continue
            abs_part = os.path.abspath(part)
            if abs_part in seen:
                continue
            seen.add(abs_part)
            paths.append(abs_part)
    return os.pathsep.join(paths)


def audio_cpp_library_dirs(
    server_binary_path: str,
    source_path: Optional[str] = None,
) -> List[str]:
    binary_dir = os.path.dirname(os.path.abspath(server_binary_path))
    root = os.path.abspath(source_path or binary_dir)
    dirs: List[str] = []
    for path in (
        binary_dir,
        os.path.join(root, "build", "bin"),
        os.path.join(root, "build", "lib"),
    ):
        if os.path.isdir(path):
            dirs.append(path)
    return dirs


def build_swap_process_env(
    user_env: Dict[str, str],
    *,
    library_dirs: Optional[List[str]] = None,
    include_cuda: bool = True,
) -> Dict[str, str]:
    """Merge per-model swap env with CUDA toolkit paths and local engine libs."""
    env = dict(user_env)
    cuda_env = cuda_runtime_env() if include_cuda else {}

    for key in ("CUDA_HOME", "CUDA_PATH"):
        if key in cuda_env and key not in env:
            env[key] = cuda_env[key]

    if cuda_env.get("PATH"):
        env["PATH"] = (
            cuda_env["PATH"]
            if "PATH" not in env
            else f"{cuda_env['PATH']}:{env['PATH']}"
        )

    user_ld = str(env.pop("LD_LIBRARY_PATH", "") or "").strip()
    ld_segments: List[Optional[str]] = []
    if include_cuda:
        cuda_lib = cuda_lib64_path()
        if cuda_lib:
            ld_segments.append(cuda_lib)
        elif cuda_env.get("LD_LIBRARY_PATH"):
            ld_segments.append(cuda_env["LD_LIBRARY_PATH"])
    if library_dirs:
        ld_segments.append(os.pathsep.join(library_dirs))
    if user_ld:
        ld_segments.append(user_ld)
    proc_ld = os.environ.get("LD_LIBRARY_PATH", "").strip()
    if proc_ld:
        ld_segments.append(proc_ld)

    merged_ld = merge_ld_library_path(*ld_segments)
    if merged_ld:
        env["LD_LIBRARY_PATH"] = merged_ld
    return env
