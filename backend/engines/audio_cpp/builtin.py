"""Discover engine-owned utility identities through the audio.cpp inspect API."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Tuple

from backend.engines.audio_cpp.artifact import BUILTIN_AUDIO_FAMILY
from backend.engines.audio_cpp.inspect import (
    audio_cpp_inspect_env,
    build_audio_cpp_inspect_argv,
    select_inspect_payload,
)
from backend.engines.audio_cpp.contracts import load_family_contract
from backend.logging_config import get_logger
from backend.paths import studio_data_dir

logger = get_logger(__name__)

_PROBE_CONCURRENCY = 4
_PROBE_TIMEOUT = 15


def _cache_path() -> str:
    return os.path.join(studio_data_dir(), "config", "audio-cpp", "builtin-utilities.json")


def _cache_key(active: dict, candidates: List[str]) -> str:
    listing = "\n".join(sorted(candidates))
    commit = str(active.get("source_commit") or active.get("source_ref") or "")
    source = str(active.get("source_path") or "")
    root = Path(source) / "assets" / "framework" / "audio_utilities"
    try:
        mtime_ns = root.stat().st_mtime_ns
    except OSError:
        mtime_ns = 0
    return hashlib.sha256(
        f"{source}\n{commit}\n{mtime_ns}\n{listing}".encode("utf-8")
    ).hexdigest()


def _read_cache(key: str) -> List[dict] | None:
    try:
        payload = json.loads(Path(_cache_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if isinstance(payload, dict) and payload.get("key") == key:
        packages = payload.get("packages")
        if isinstance(packages, list):
            return [row for row in packages if isinstance(row, dict)]
    return None


def _write_cache(key: str, packages: List[dict]) -> None:
    path = _cache_path()
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump({"key": key, "packages": packages}, handle)
        os.replace(tmp, path)
    except OSError as exc:
        logger.debug("audio.cpp builtin utility cache write failed: %s", exc)


def _candidate_utility_ids(root: Path) -> List[str]:
    candidates = set()
    try:
        for directory in root.iterdir():
            if directory.is_dir():
                candidates.add(directory.name)
                candidates.update(path.stem for path in directory.glob("*.safetensors"))
    except OSError:
        return []
    return sorted(candidates)


def discover_builtin_audio_packages(active: dict, families: Iterable[str]) -> list[dict]:
    """Expose only ids accepted by the active engine, with its own task metadata.

    Candidates come from packaged asset directory names and weight stems; the
    engine's inspect API alone decides acceptance. Probes run with bounded
    concurrency and the verified result set is cached on
    (engine commit, utility directory listing) so catalog refreshes do not
    re-probe an unchanged engine.
    """
    if BUILTIN_AUDIO_FAMILY not in set(families):
        return []
    source = str(active.get("source_path") or "")
    cli = str(active.get("cli_binary_path") or "")
    contract = load_family_contract(source, BUILTIN_AUDIO_FAMILY)
    root = Path(source) / "assets" / "framework" / "audio_utilities"
    if not contract or contract.get("packages") or not source or not root.is_dir() or not os.path.isfile(cli):
        return []
    candidates = _candidate_utility_ids(root)
    if not candidates:
        return []

    cache_key = _cache_key(active, candidates)
    cached = _read_cache(cache_key)
    if cached is not None:
        return cached

    env = {**os.environ, **audio_cpp_inspect_env(cli, source_path=source)}

    def _probe(model_id: str) -> dict | None:
        argv = build_audio_cpp_inspect_argv(
            cli, model_id, family=BUILTIN_AUDIO_FAMILY,
            model_spec_override=str(Path(source) / "model_specs"),
        )
        try:
            process = subprocess.run(
                [*argv, "--inspect", "--json"], cwd=source, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, timeout=_PROBE_TIMEOUT,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        if process.returncode:
            return None
        _, inspection = select_inspect_payload(process.stdout)
        if inspection.get("family") != BUILTIN_AUDIO_FAMILY or not inspection.get("task_names"):
            return None
        modes = list(dict.fromkeys(
            mode for row in inspection.get("tasks") or []
            for mode in row.get("modes") or []
        ))
        return {
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
        }

    packages: List[dict] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_PROBE_CONCURRENCY, max(1, len(candidates)))
    ) as pool:
        for row in pool.map(_probe, candidates):
            if row:
                packages.append(row)
    packages.sort(key=lambda row: str(row.get("id") or ""))
    _write_cache(cache_key, packages)
    return packages
