"""Persisted install/build defaults for python_venv engines (LMDeploy, 1Cat-vLLM)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.engine_registry import get_engine_spec

LMDEPLOY_DEFAULTS: Dict[str, Any] = {
    "source_repo": "https://github.com/InternLM/lmdeploy.git",
    "source_branch": "main",
    "pip_version": "",
}

ONECAT_VLLM_DEFAULTS: Dict[str, Any] = {
    "source_repo": "https://github.com/1CatAI/1Cat-vLLM.git",
    "source_branch": "main",
    "release_version": "",
}

_ENGINE_DEFAULTS = {
    "lmdeploy": LMDEPLOY_DEFAULTS,
    "1cat_vllm": ONECAT_VLLM_DEFAULTS,
}


def default_install_settings(engine_id: str) -> Dict[str, Any]:
    defaults = _ENGINE_DEFAULTS.get(str(engine_id or ""))
    if not defaults:
        raise ValueError(f"No install settings defaults for engine: {engine_id}")
    return dict(defaults)


def coerce_install_settings(engine_id: str, settings: Optional[dict]) -> Dict[str, Any]:
    base = default_install_settings(engine_id)
    raw = settings if isinstance(settings, dict) else {}
    out = dict(base)
    for key in base:
        if key not in raw:
            continue
        value = raw.get(key)
        if value is None:
            out[key] = ""
        else:
            out[key] = str(value).strip()
    # Keep empty optional version fields as blank (= latest).
    if not out.get("source_repo"):
        out["source_repo"] = base["source_repo"]
    if not out.get("source_branch"):
        out["source_branch"] = base["source_branch"]
    return out


def assert_venv_engine(engine_id: str) -> None:
    spec = get_engine_spec(engine_id)
    if not spec or spec.install_kind != "python_venv":
        raise ValueError(f"Engine is not a python_venv installer: {engine_id}")
    if engine_id not in _ENGINE_DEFAULTS:
        raise ValueError(f"No install settings defaults for engine: {engine_id}")
