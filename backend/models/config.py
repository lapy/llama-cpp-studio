"""Per-engine model configuration: normalize stored YAML, effective flat view, merge on PUT."""

from __future__ import annotations

from typing import Any, Dict, Optional

from backend.engines.registry import EMBEDDINGS_ENGINE_IDS, VALID_ENGINE_IDS
from backend.engines.params import (
    embedding_mode_config_key_from_entry,
    get_version_entry,
)
from backend.utils.coercion import coerce_json_dict

DEFAULT_ENGINE = "llama_cpp"
DOWNLOAD_CONFIG_REVIEW_SOURCE = "download"
USER_CONFIG_REVIEW_SOURCE = "user"
LEGACY_CONFIG_REVIEW_SOURCE = "legacy"
_UNREVIEWED_CONFIG_SOURCES = {DOWNLOAD_CONFIG_REVIEW_SOURCE, "installer"}


def note_legacy_config_review(model: Dict[str, Any]) -> None:
    """Mark a pre-stamp engine map so a later download is not treated the same way.

    Records that already name their provenance are left alone. A saved stamp
    becomes ``user``. An engine map with no stamp becomes ``legacy``.
    """
    if not isinstance(model, dict) or model.get("config_review_source"):
        return
    if model.get("config_reviewed_at"):
        model["config_review_source"] = USER_CONFIG_REVIEW_SOURCE
        return
    raw = model.get("config")
    if isinstance(raw, dict) and isinstance(raw.get("engines"), dict):
        model["config_review_source"] = LEGACY_CONFIG_REVIEW_SOURCE


def config_was_reviewed(model: Any) -> bool:
    """True only for a user save or a migrated historical engine map.

    A download can store engine defaults. That is not a review.
    """
    if not isinstance(model, dict):
        return False
    if model.get("config_reviewed_at"):
        return True
    source = model.get("config_review_source")
    if source == LEGACY_CONFIG_REVIEW_SOURCE:
        return True
    if source in _UNREVIEWED_CONFIG_SOURCES:
        return False
    raw = model.get("config")
    return isinstance(raw, dict) and isinstance(raw.get("engines"), dict)


def unreviewed_download_update(model: Any) -> Dict[str, Any]:
    """Provenance to store when a download writes config and the user has not saved."""
    if not isinstance(model, dict):
        return {"config_review_source": DOWNLOAD_CONFIG_REVIEW_SOURCE}
    if model.get("config_reviewed_at"):
        return {}
    source = model.get("config_review_source")
    if source in {USER_CONFIG_REVIEW_SOURCE, LEGACY_CONFIG_REVIEW_SOURCE}:
        return {}
    if source == DOWNLOAD_CONFIG_REVIEW_SOURCE:
        return {}
    return {"config_review_source": DOWNLOAD_CONFIG_REVIEW_SOURCE}


def _coerce_raw(config_value: Optional[Any]) -> Dict[str, Any]:
    return coerce_json_dict(config_value, copy=True)


def normalize_model_config(raw: Optional[Any]) -> Dict[str, Any]:
    """Return canonical stored shape: {"engine": str, "engines": {engine_id: {...}}}."""
    c = _coerce_raw(raw)
    if not c or not isinstance(c.get("engines"), dict):
        return {"engine": DEFAULT_ENGINE, "engines": {}}

    engine = c.get("engine") or DEFAULT_ENGINE
    if engine not in VALID_ENGINE_IDS:
        engine = DEFAULT_ENGINE
    engines: Dict[str, Dict[str, Any]] = {}
    for k, v in c["engines"].items():
        if k not in VALID_ENGINE_IDS:
            continue
        engines[k] = dict(v) if isinstance(v, dict) else {}
    return {"engine": engine, "engines": engines}


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
        if key == "swap_env" and isinstance(value, dict):
            # An explicit empty string is a value. An empty map replaces the collection.
            cleaned: Dict[str, str] = {}
            for env_key, env_value in value.items():
                name = str(env_key)
                if env_value is None:
                    continue
                cleaned[name] = env_value if isinstance(env_value, str) else str(env_value)
            out[key] = cleaned
            continue
        if key == "swap_env_unset" and isinstance(value, list):
            out[key] = [str(item) for item in value if str(item).strip()]
            continue
        if key == "gpu_devices" and isinstance(value, list):
            out[key] = [str(item) for item in value if str(item).strip()]
            continue
        if value is None:
            out[key] = None
            continue
        if isinstance(value, str) and value == "":
            continue
        if isinstance(value, float) and value != value:  # NaN
            continue
        if isinstance(value, dict):
            nested = _strip_empty_values(value)
            if nested:
                out[key] = nested
            continue
        if isinstance(value, list) and not value:
            continue
        out[key] = value
    return out


def merge_model_config_put(
    existing_raw: Optional[Any], body: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
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
    incoming = _strip_empty_values({k: v for k, v in body.items() if k not in reserved})
    section = dict((existing["engines"] or {}).get(eng) or {})
    section.update(incoming)
    merged_engines = {k: dict(v) for k, v in existing["engines"].items()}
    merged_engines[eng] = section
    return {"engine": eng, "engines": merged_engines}


def _section_has_settings(section: Any) -> bool:
    """True when a stored engine section has a value a user could rely on later."""
    if not isinstance(section, dict):
        return False
    for value in section.values():
        if value is None or value == "":
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        return True
    return False


def _model_reference_label(model: Dict[str, Any]) -> str:
    return str(
        model.get("display_name")
        or model.get("name")
        or model.get("id")
        or "model"
    )


def collect_engine_model_references(models: Any) -> Dict[str, Any]:
    """Group models by the engine they run and by unused saved engine sections.

    A model with no stored config uses the default engine. A section counts as
    unused only when it has settings and is not the model's selected engine.
    """
    selected = {engine: [] for engine in VALID_ENGINE_IDS}
    dormant = {engine: [] for engine in VALID_ENGINE_IDS}
    for model in models or []:
        if not isinstance(model, dict):
            continue
        normalized = normalize_model_config(model.get("config"))
        label = _model_reference_label(model)
        chosen = str(normalized.get("engine") or DEFAULT_ENGINE)
        if chosen in selected:
            selected[chosen].append(label)
        for engine, section in (normalized.get("engines") or {}).items():
            if engine == chosen or engine not in dormant:
                continue
            if _section_has_settings(section):
                dormant[engine].append({"name": label, "engine": chosen})
    return {"selected": selected, "dormant": dormant}


def default_engine_for_format(model_format: Optional[str]) -> str:
    if (model_format or "").lower() == "safetensors":
        return "lmdeploy"
    return "llama_cpp"


def set_embedding_flag(
    raw: Optional[Any],
    *,
    model_format: Optional[str],
    store: Any,
) -> Dict[str, Any]:
    """Return normalized config with embeddings=True on the appropriate engine section.

    Callers must pass the data store so this module stays free of store singleton access.
    """
    n = normalize_model_config(raw)
    c = _coerce_raw(raw)
    has_explicit_engine = (
        isinstance(c, dict) and c.get("engine") in EMBEDDINGS_ENGINE_IDS
    )
    if has_explicit_engine:
        eng = c["engine"]
    else:
        eng = default_engine_for_format(model_format)
    n["engine"] = eng
    n.setdefault("engines", {})
    n["engines"].setdefault(eng, {})

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
