"""Persisted install/build defaults for Python virtual-environment engines.

The frozen per-engine defaults (source repo/branch/version) are engine DATA: they
live on the engine spec (`EngineSpec.settings_defaults`), are exported into the
generated definitions, and are read from the spec here. Authoritative (YAML-only)
engines register theirs at startup through ``register_engine_install_settings``
from their definition's ``settings_defaults`` block.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Authoritative (YAML-only) engine registrations. Generated engines never land
# here: their defaults come from the spec. Kept module-level so startup wiring
# and tests can isolate registrations by patching this dict.
_ENGINE_DEFAULTS: Dict[str, Dict[str, Any]] = {}


def _declared_defaults(engine_id: str) -> Dict[str, Any]:
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(str(engine_id or ""))
    if spec is None:
        return {}
    return dict(spec.settings_defaults)


def default_install_settings(engine_id: str) -> Dict[str, Any]:
    registered = _ENGINE_DEFAULTS.get(str(engine_id or ""))
    if registered:
        return dict(registered)
    declared = _declared_defaults(engine_id)
    if not declared:
        raise ValueError(f"No install settings defaults for engine: {engine_id}")
    return declared


def register_engine_install_settings(engine_id: str, defaults: Dict[str, Any]) -> None:
    """YAML campaign S3: authoritative (YAML-only) engines declare their
    install-settings defaults through the definition lifecycle section;
    ``register_definition_lifecycles`` funnels them in at startup so the
    shared HTTP surface's ``_saved()``/settings verbs work with zero code.
    Generated engines declare theirs on the spec instead."""
    _ENGINE_DEFAULTS[str(engine_id)] = dict(defaults)


def coerce_install_settings(engine_id: str, settings: Optional[dict]) -> Dict[str, Any]:
    base = default_install_settings(engine_id)
    raw = settings if isinstance(settings, dict) else {}
    out = dict(base)
    for key in base:
        if key not in raw:
            continue
        value = raw.get(key)
        if isinstance(base[key], bool):
            if isinstance(value, str):
                out[key] = value.strip().lower() in {"1", "true", "yes", "on"}
            else:
                out[key] = bool(value)
            continue
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
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(engine_id)
    if not spec or spec.install_kind != "python_venv":
        raise ValueError(f"Engine is not a python_venv installer: {engine_id}")
    if not _declared_defaults(engine_id) and engine_id not in _ENGINE_DEFAULTS:
        raise ValueError(f"No install settings defaults for engine: {engine_id}")
