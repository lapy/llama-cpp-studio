"""Per-engine model configuration: normalize stored YAML, effective flat view, merge on PUT."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from backend.engine_param_catalog import (
    embedding_mode_config_key_from_entry,
    get_version_entry,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_ENGINE = "llama_cpp"
VALID_ENGINE_IDS = frozenset({"llama_cpp", "ik_llama", "lmdeploy"})
EMBEDDINGS_ENGINE_IDS = frozenset({"llama_cpp", "ik_llama"})


def _coerce_raw(config_value: Optional[Any]) -> Dict[str, Any]:
    if not config_value:
        return {}
    if isinstance(config_value, dict):
        return dict(config_value)
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            logger.warning("Failed to parse model config JSON")
            return {}
    return {}


def normalize_model_config(raw: Optional[Any]) -> Dict[str, Any]:
    """
    Return canonical stored shape: {"engine": str, "engines": {engine_id: {...}}}.
    Migrates flat dicts (no `engines` key) into engines[engine].
    """
    c = _coerce_raw(raw)
    if not c:
        return {"engine": DEFAULT_ENGINE, "engines": {}}

    if isinstance(c.get("engines"), dict):
        engine = c.get("engine") or DEFAULT_ENGINE
        if engine not in VALID_ENGINE_IDS:
            engine = DEFAULT_ENGINE
        engines: Dict[str, Dict[str, Any]] = {}
        for k, v in c["engines"].items():
            if k not in VALID_ENGINE_IDS:
                continue
            if isinstance(v, dict):
                engines[k] = dict(v)
            else:
                engines[k] = {}
        return {"engine": engine, "engines": engines}

    engine = c.get("engine") or DEFAULT_ENGINE
    if engine not in VALID_ENGINE_IDS:
        engine = DEFAULT_ENGINE
    reserved = frozenset({"engine", "engines"})
    payload = {k: v for k, v in c.items() if k not in reserved}
    return {"engine": engine, "engines": {engine: payload}}


def effective_model_config(normalized: Dict[str, Any]) -> Dict[str, Any]:
    """Flat dict: engine + params for the active engine (runtime, swap, proxy alias)."""
    eng = normalized.get("engine") or DEFAULT_ENGINE
    if eng not in VALID_ENGINE_IDS:
        eng = DEFAULT_ENGINE
    section = dict((normalized.get("engines") or {}).get(eng) or {})
    return {"engine": eng, **section}


def effective_model_config_from_raw(raw: Optional[Any]) -> Dict[str, Any]:
    return effective_model_config(normalize_model_config(raw))


def config_api_response(normalized: Dict[str, Any]) -> Dict[str, Any]:
    """GET/PUT payload: flattened effective params plus `engines` map for the UI."""
    eff = effective_model_config(normalized)
    engines = {k: dict(v) for k, v in (normalized.get("engines") or {}).items()}
    return {**eff, "engines": engines}


def _strip_empty_values(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in d.items():
        if value is None:
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        out[key] = value
    return out


def merge_model_config_put(existing_raw: Optional[Any], body: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merge client PUT into stored config. If body contains `engines`, merge each
    provided engine section; omitted engine keys are left unchanged. Always sets `engine`
    from body when provided.
    """
    existing = normalize_model_config(existing_raw)
    body = body or {}

    if isinstance(body.get("engines"), dict):
        eng = body.get("engine") or existing["engine"]
        if eng not in VALID_ENGINE_IDS:
            eng = existing["engine"]
        merged_engines = {k: dict(v) for k, v in existing["engines"].items()}
        for k, v in body["engines"].items():
            if k not in VALID_ENGINE_IDS or not isinstance(v, dict):
                continue
            merged_engines[k] = _strip_empty_values(dict(v))
        return {"engine": eng, "engines": merged_engines}

    eng = body.get("engine") or existing["engine"]
    if eng not in VALID_ENGINE_IDS:
        eng = existing["engine"]
    reserved = frozenset({"engine", "engines"})
    incoming = _strip_empty_values(
        {k: v for k, v in body.items() if k not in reserved}
    )
    section = dict((existing["engines"] or {}).get(eng) or {})
    section.update(incoming)
    merged_engines = {k: dict(v) for k, v in existing["engines"].items()}
    merged_engines[eng] = section
    return {"engine": eng, "engines": merged_engines}


def default_engine_for_format(model_format: Optional[str]) -> str:
    if (model_format or "").lower() == "safetensors":
        return "lmdeploy"
    return "llama_cpp"


def set_embedding_flag(raw: Optional[Any], *, model_format: Optional[str]) -> Dict[str, Any]:
    """Return normalized config with embeddings=True on the appropriate engine section."""
    n = normalize_model_config(raw)
    c = _coerce_raw(raw)
    has_explicit_engine = isinstance(c, dict) and c.get("engine") in EMBEDDINGS_ENGINE_IDS
    if has_explicit_engine:
        eng = c["engine"]
    else:
        eng = default_engine_for_format(model_format)
    n["engine"] = eng
    n.setdefault("engines", {})
    n["engines"].setdefault(eng, {})

    # Local import avoids circular import (data_store imports model_config).
    from backend.data_store import get_store

    store = get_store()
    active = store.get_active_engine_version(eng)
    if not active or not active.get("version"):
        return n
    entry = get_version_entry(store, eng, active["version"])
    if not entry or entry.get("scan_error"):
        return n

    embeddings_key = embedding_mode_config_key_from_entry(entry)
    if embeddings_key:
        n["engines"][eng][embeddings_key] = True
        # Routes/UI use ``embedding`` in effective config; keep in sync if catalog uses another key.
        if embeddings_key != "embedding":
            n["engines"][eng]["embedding"] = True
    else:
        # No embedding param in this engine's catalog (e.g. scan pending); keep previous behavior.
        n["engines"][eng]["embedding"] = True
    return n
