"""Canonical audio.cpp model path / artifact contract.

Install, validate, inspect, and runtime must agree on one resolvable path shape:
directory packages (preferred) or a single ``.gguf`` file when the package root
cannot host ``model.gguf``.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, Optional


BUILTIN_AUDIO_FAMILY = "builtin_audio_utils"


def audio_builtin_model_id(model: dict) -> str:
    """Return an explicit engine-owned utility id, never a filesystem path.

    audio.cpp currently grants this path exception only to builtin_audio_utils.
    Actual id/asset availability is validated by the engine's inspect API.
    """
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    if artifact.get("package_kind") != "builtin" or model.get("family") != BUILTIN_AUDIO_FAMILY:
        return ""
    model_id = artifact.get("model_id")
    if not isinstance(model_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", model_id):
        return ""
    return model_id


def audio_model_ready(model: dict) -> bool:
    """Validate artifact identity; builtin assets are checked by engine inspect."""
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    if artifact.get("package_kind") == "builtin":
        return bool(audio_builtin_model_id(model))
    return audio_model_path_ready(resolve_audio_model_path(model))


def build_builtin_artifact_descriptor(model_id: str) -> Dict[str, Any]:
    descriptor = {"format": "builtin", "package_kind": "builtin", "model_id": model_id,
                  "layout": "builtin", "size": 0}
    if not audio_builtin_model_id({"family": BUILTIN_AUDIO_FAMILY, "artifact": descriptor}):
        raise ValueError("Invalid audio.cpp builtin utility id")
    return descriptor


def studio_data_root() -> str:
    if os.path.isdir("/app/data"):
        return "/app/data"
    return os.path.abspath("data")


def audio_sidecar_root() -> str:
    return os.path.join(studio_data_root(), "config", "audio-cpp", "servers")


def audio_model_path_ready(path: str) -> bool:
    """True when ``path`` is usable as audio.cpp ``--model``."""
    if not path:
        return False
    if os.path.isdir(path):
        return True
    return os.path.isfile(path) and str(path).lower().endswith(".gguf")


def _abspath(path: str) -> str:
    return os.path.abspath(path) if path else ""


def prefer_directory_model_path(path: str, *, bundle_path: str = "") -> str:
    """If ``path`` is a GGUF and a package root hosts ``model.gguf``, use that dir."""
    abs_path = _abspath(path)
    if not abs_path or not os.path.isfile(abs_path):
        return abs_path
    if not abs_path.lower().endswith(".gguf"):
        return abs_path

    bundle = _abspath(bundle_path) if bundle_path else ""
    if bundle and os.path.isdir(bundle):
        root_gguf = os.path.join(bundle, "model.gguf")
        if os.path.isfile(root_gguf):
            return bundle

    parent = os.path.dirname(abs_path)
    parent_gguf = os.path.join(parent, "model.gguf")
    if os.path.isfile(parent_gguf):
        return parent
    return abs_path


def resolve_audio_model_path(model: dict) -> str:
    """Resolve the canonical ``--model`` path for a stored audio model record."""
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    if artifact.get("package_kind") == "builtin":
        return audio_builtin_model_id(model)
    bundle = str(
        artifact.get("bundle_path") or model.get("bundle_path") or ""
    ).strip()
    candidates = [
        artifact.get("runtime_path"),
        artifact.get("path"),
        model.get("local_path"),
        model.get("model_path"),
        artifact.get("bundle_path"),
        model.get("bundle_path"),
    ]
    for raw in candidates:
        path = str(raw or "").strip()
        if not path:
            continue
        abs_path = prefer_directory_model_path(path, bundle_path=bundle)
        if audio_model_path_ready(abs_path):
            return abs_path
    return ""


def resolve_audio_bundle_root(model: dict) -> str:
    """Package root directory (manifest / references), even when runtime is a file."""
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    if artifact.get("package_kind") == "builtin":
        return ""
    for raw in (
        artifact.get("bundle_path"),
        model.get("bundle_path"),
        artifact.get("path"),
        model.get("local_path"),
    ):
        path = str(raw or "").strip()
        if not path:
            continue
        abs_path = _abspath(path)
        if os.path.isdir(abs_path):
            return abs_path
        if os.path.isfile(abs_path):
            return os.path.dirname(abs_path)
    runtime = resolve_audio_model_path(model)
    if runtime and os.path.isdir(runtime):
        return runtime
    if runtime and os.path.isfile(runtime):
        return os.path.dirname(runtime)
    return ""


def build_artifact_descriptor(
    *,
    bundle_path: str,
    runtime_path: str,
    size: Optional[int] = None,
    package_kind: str = "prepared_bundle",
) -> Dict[str, Any]:
    """Persist install-time layout so later stages do not re-guess."""
    bundle = _abspath(bundle_path)
    runtime = prefer_directory_model_path(
        runtime_path, bundle_path=bundle
    ) or _abspath(runtime_path)
    layout = "directory" if os.path.isdir(runtime) else "gguf_file"
    gguf_path = None
    if layout == "gguf_file" and runtime.lower().endswith(".gguf"):
        gguf_path = runtime
    else:
        candidate = os.path.join(runtime if layout == "directory" else bundle, "model.gguf")
        if os.path.isfile(candidate):
            gguf_path = candidate
    return {
        "format": "mixed",
        "package_kind": package_kind,
        "path": runtime,
        "runtime_path": runtime,
        "bundle_path": bundle,
        "layout": layout,
        "gguf_path": gguf_path,
        "has_root_model_gguf": bool(
            gguf_path
            and os.path.isfile(gguf_path)
            and os.path.basename(gguf_path) == "model.gguf"
            and os.path.realpath(os.path.dirname(gguf_path)) == os.path.realpath(bundle)
        ),
        "size": size,
    }
