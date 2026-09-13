"""Discover engine-owned utility identities through the audio.cpp inspect API."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable

from backend.audio_cpp_artifact import BUILTIN_AUDIO_FAMILY
from backend.audio_cpp_inspect import (
    audio_cpp_inspect_env,
    build_audio_cpp_inspect_argv,
    select_inspect_payload,
)
from backend.audio_cpp_model_contracts import load_family_contract


def discover_builtin_audio_packages(active: dict, families: Iterable[str]) -> list[dict]:
    """Expose only ids accepted by the active engine, with its own task metadata.

    Upstream has no public utility-id enumeration yet. Packaged asset directory
    names and weight stems are candidates, never proof of a supported utility.
    No package copies, C++ parsing, or frozen utility registry are needed.
    """
    if BUILTIN_AUDIO_FAMILY not in set(families):
        return []
    source = str(active.get("source_path") or "")
    cli = str(active.get("cli_binary_path") or "")
    contract = load_family_contract(source, BUILTIN_AUDIO_FAMILY)
    root = Path(source) / "assets" / "framework" / "audio_utilities"
    if not contract or contract.get("packages") or not source or not root.is_dir() or not os.path.isfile(cli):
        return []
    candidates = set()
    try:
        for directory in root.iterdir():
            if directory.is_dir():
                candidates.add(directory.name)
                candidates.update(path.stem for path in directory.glob("*.safetensors"))
    except OSError:
        return []
    env = {**os.environ, **audio_cpp_inspect_env(cli, source_path=source)}
    packages = []
    for model_id in sorted(candidates):
        argv = build_audio_cpp_inspect_argv(
            cli, model_id, family=BUILTIN_AUDIO_FAMILY,
            model_spec_override=str(Path(source) / "model_specs"),
        )
        try:
            process = subprocess.run(
                [*argv, "--inspect", "--json"], cwd=source, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=15,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if process.returncode:
            continue
        _, inspection = select_inspect_payload(process.stdout)
        if inspection.get("family") != BUILTIN_AUDIO_FAMILY or not inspection.get("task_names"):
            continue
        modes = list(dict.fromkeys(
            mode for row in inspection.get("tasks") or []
            for mode in row.get("modes") or []
        ))
        packages.append({
            "id": f"builtin-{model_id}",
            "display_name": f"{model_id} (built-in)",
            "description": contract.get("description") or "Built into audio.cpp",
            "family": BUILTIN_AUDIO_FAMILY,
            "tasks": inspection["task_names"],
            "modes": modes,
            "standalone": True,
            "installable": True,
            "install_kind": "builtin",
            "format": "builtin",
            "source": {"kind": "builtin", "model_id": model_id},
            "required_files": [],
            "size_bytes": 0,
        })
    return packages
