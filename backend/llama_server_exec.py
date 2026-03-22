"""Resolve llama-server executable + cwd to match llama-swap ``cd …/build/bin && ./llama-server`` layout."""

from __future__ import annotations

import os
from typing import Optional, Tuple


def sibling_build_bin_from_install_bin(install_bin_dir: str) -> Optional[str]:
    """
    If ``install_bin_dir`` is ``PREFIX/bin`` (not already ``…/build/bin``), return ``PREFIX/build/bin``.

    Handles paths that end with ``/bin`` but do not contain the substring ``/bin/`` (e.g. ``…/install/bin``).
    """
    wd = os.path.abspath(install_bin_dir)
    norm = wd.rstrip(os.sep)
    if os.path.basename(norm) != "bin":
        return None
    parent = os.path.dirname(norm)
    if os.path.basename(parent) == "build":
        return None
    return os.path.join(parent, "build", "bin")


def resolve_llama_server_invocation_paths(abs_binary_path: str) -> Tuple[str, str]:
    """
    Match ``generate_llama_swap_config`` launcher: ``cd working_dir && ./<binary>`` where
    ``working_dir`` is ``…/build/bin`` when that directory exists and the stored path was under ``…/bin``.

    Args:
        abs_binary_path: Absolute path from engines.yaml (after ``/app`` join if needed).

    Returns:
        ``(executable_path, cwd)`` for subprocess (argv0 and working directory).
    """
    p = os.path.abspath(abs_binary_path)
    working_dir = os.path.dirname(p)
    binary_name = os.path.basename(p)
    alt_dir = sibling_build_bin_from_install_bin(working_dir)
    if alt_dir:
        alt_exec = os.path.join(alt_dir, binary_name)
        if os.path.isfile(alt_exec):
            return os.path.abspath(alt_exec), os.path.abspath(alt_dir)
    # Legacy layout: path segment contains ``/bin/`` (e.g. ``…/something/bin/extra``)
    if "/bin/" in working_dir and "/build/bin/" not in working_dir:
        legacy_alt = working_dir.replace("/bin/", "/build/bin/")
        alt_exec = os.path.join(legacy_alt, binary_name)
        if os.path.isfile(alt_exec):
            return os.path.abspath(alt_exec), os.path.abspath(legacy_alt)
    return p, os.path.abspath(working_dir)


def llama_help_ld_library_path(binary_dir: str) -> str:
    """Dirs to search for ggml/CUDA .so when running ``llama-server --help`` (scan / flag probes)."""
    candidates: list[str] = []

    def consider(path: str) -> None:
        if not path:
            return
        ap = os.path.abspath(path)
        if os.path.isdir(ap) and ap not in candidates:
            candidates.append(ap)

    consider(binary_dir)
    sbb = sibling_build_bin_from_install_bin(binary_dir)
    if sbb:
        consider(sbb)
    if "/bin/" in binary_dir and "/build/bin/" not in binary_dir:
        consider(binary_dir.replace("/bin/", "/build/bin/"))
    consider(os.path.join(binary_dir, "build", "bin"))
    consider(os.path.join(binary_dir, "build"))
    consider(os.path.join(binary_dir, "..", "build", "bin"))

    seen = set(candidates)
    tail = os.environ.get("LD_LIBRARY_PATH", "").strip()
    if tail:
        for part in tail.split(os.pathsep):
            p = part.strip()
            if p and p not in seen:
                ap = os.path.abspath(p)
                seen.add(ap)
                candidates.append(ap)
    return os.pathsep.join(candidates)
