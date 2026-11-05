"""Native (cmake-built) engine helpers shared by audio.cpp and the native
llama engines.

Binaries live under ``build/bin`` (and nested multi-config dirs); locating
them is identical across these engines, so the logic has one home here.
"""

from __future__ import annotations

import os
from typing import List


def native_binary_candidates(build_dir: str, name: str) -> List[str]:
    executable = f"{name}.exe" if os.name == "nt" else name
    return [
        os.path.join(build_dir, "bin", executable),
        os.path.join(build_dir, executable),
        os.path.join(build_dir, "Release", executable),
        os.path.join(build_dir, "bin", "Release", executable),
    ]


def native_find_binary(build_dir: str, name: str) -> str:
    for candidate in native_binary_candidates(build_dir, name):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise RuntimeError(f"Built {name} binary was not found under {build_dir}")
