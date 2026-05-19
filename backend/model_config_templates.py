"""Model configuration templates: snapshot, store, and apply to other models."""

from __future__ import annotations

import copy
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from backend.model_config import DEFAULT_ENGINE, VALID_ENGINE_IDS, normalize_model_config

# Per-model routing; omitted from templates unless explicitly included.
ROUTING_ONLY_KEYS = frozenset({"model_alias", "swap_aliases"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _copy_engine_section(section: Dict[str, Any], *, include_routing: bool) -> Dict[str, Any]:
    sec = copy.deepcopy(section)
    if not include_routing:
        for key in ROUTING_ONLY_KEYS:
            sec.pop(key, None)
    return sec


def extract_template_config(
    normalized: Dict[str, Any],
    *,
    include_routing: bool = False,
    engines_scope: str = "all",
) -> Dict[str, Any]:
    """
    Build storable template payload from a normalized model config.

    ``engines_scope``: ``all`` copies every engine section; ``active`` copies only
    the active ``engine`` section.
    """
    norm = normalize_model_config(normalized)
    engine = norm.get("engine") or DEFAULT_ENGINE
    if engine not in VALID_ENGINE_IDS:
        engine = DEFAULT_ENGINE
    engines_out: Dict[str, Dict[str, Any]] = {}
    source_engines = norm.get("engines") or {}
    if engines_scope == "active":
        section = source_engines.get(engine)
        if isinstance(section, dict):
            engines_out[engine] = _copy_engine_section(section, include_routing=include_routing)
    else:
        for eng, section in source_engines.items():
            if eng not in VALID_ENGINE_IDS or not isinstance(section, dict):
                continue
            engines_out[eng] = _copy_engine_section(section, include_routing=include_routing)
    return {"engine": engine, "engines": engines_out}


def apply_template_to_config(
    existing_raw: Optional[Any],
    template_config: Dict[str, Any],
    *,
    include_routing: bool = False,
    apply_engines: str = "active",
) -> Dict[str, Any]:
    """
    Merge a template into an existing normalized config.

    ``apply_engines``:
    - ``active``: merge template's section for ``template.engine`` into the target's
      current active engine (does not change target ``engine``).
    - ``all``: merge every engine section present in the template.
    - ``set_engine``: merge active section and set target ``engine`` to template's.
    """
    existing = normalize_model_config(existing_raw)
    template = normalize_model_config(template_config)
    merged_engines = {k: dict(v) for k, v in (existing.get("engines") or {}).items()}

    target_engine = existing.get("engine") or DEFAULT_ENGINE
    if target_engine not in VALID_ENGINE_IDS:
        target_engine = DEFAULT_ENGINE

    template_engine = template.get("engine") or DEFAULT_ENGINE
    if template_engine not in VALID_ENGINE_IDS:
        template_engine = DEFAULT_ENGINE

    template_engines = template.get("engines") or {}

    def _merge_section(target_eng: str, source_section: Dict[str, Any]) -> None:
        base = dict(merged_engines.get(target_eng) or {})
        incoming = _copy_engine_section(source_section, include_routing=include_routing)
        if not include_routing:
            for key in ROUTING_ONLY_KEYS:
                incoming.pop(key, None)
        base.update(incoming)
        merged_engines[target_eng] = base

    if apply_engines == "all":
        for eng, section in template_engines.items():
            if eng in VALID_ENGINE_IDS and isinstance(section, dict):
                _merge_section(eng, section)
        result_engine = target_engine
    elif apply_engines == "set_engine":
        section = template_engines.get(template_engine)
        if isinstance(section, dict):
            _merge_section(template_engine, section)
        result_engine = template_engine
    else:
        section = template_engines.get(template_engine)
        if isinstance(section, dict):
            _merge_section(target_engine, section)
        result_engine = target_engine

    return {"engine": result_engine, "engines": merged_engines}


def new_template_record(
    *,
    name: str,
    config: Dict[str, Any],
    description: str = "",
    source_model_id: Optional[str] = None,
    include_routing: bool = False,
    engines_scope: str = "all",
) -> Dict[str, Any]:
    """Create a template dict ready for persistence."""
    trimmed_name = (name or "").strip()
    if not trimmed_name:
        raise ValueError("Template name is required")
    payload = extract_template_config(
        config,
        include_routing=include_routing,
        engines_scope=engines_scope,
    )
    if not payload.get("engines"):
        raise ValueError("Template has no engine settings to save")
    now = _utc_now_iso()
    return {
        "id": str(uuid.uuid4()),
        "name": trimmed_name,
        "description": (description or "").strip(),
        "created_at": now,
        "updated_at": now,
        "source_model_id": source_model_id,
        "include_routing": bool(include_routing),
        "engines_scope": engines_scope,
        "config": payload,
    }
