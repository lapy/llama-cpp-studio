"""Studio-managed llama-swap profiles and selectors (v241+).

Stored in ``data/config/llama_swap_routing.yaml`` and emitted into the generated
``llama-swap-config.yaml`` by :func:`backend.llama_swap_config.generate_llama_swap_config`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from backend.data_store import (
    collect_claimed_swap_names,
    get_store,
    normalize_proxy_alias,
    resolve_llama_swap_id,
)
from backend.logging_config import get_logger
from backend.model_config import effective_model_config, normalize_model_config
from backend.utils.coercion import coerce_json_dict

logger = get_logger(__name__)

ROUTING_FILENAME = "llama_swap_routing.yaml"
SELECTOR_STRATEGIES = frozenset({"warm", "pin", "spillover"})


def empty_routing_document() -> Dict[str, Any]:
    return {"profiles": {}, "selectors": {}}


def _normalize_id(raw: Any) -> str:
    if raw is None:
        return ""
    return normalize_proxy_alias(str(raw))


def _normalize_target_name(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    # Allow peer FQNs (peer/model) without forcing alias normalization that
    # would rewrite the slash.
    if "/" in text:
        return text
    return normalize_proxy_alias(text) or text


def _normalize_pins(raw: Any) -> Dict[str, str]:
    if not isinstance(raw, dict):
        return {}
    pins: Dict[str, str] = {}
    for key, value in raw.items():
        pin_id = _normalize_id(key)
        if not pin_id:
            continue
        if value is None:
            pins[pin_id] = ""
            continue
        text = str(value).strip()
        if not text or text.lower() in {"~", "null", "none"}:
            pins[pin_id] = ""
            continue
        pins[pin_id] = _normalize_target_name(text)
    return pins


def _normalize_targets(raw: Any) -> List[str]:
    if not isinstance(raw, list):
        return []
    out: List[str] = []
    seen: set[str] = set()
    for item in raw:
        target = _normalize_target_name(item)
        if not target or target in seen:
            continue
        seen.add(target)
        out.append(target)
    return out


def _normalize_metadata(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        name = str(key).strip()
        if not name:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[name] = value
        else:
            out[name] = str(value)
    return out


def normalize_profile(raw: Any) -> Optional[Dict[str, Any]]:
    data = coerce_json_dict(raw, copy=False)
    if not data and not isinstance(raw, dict):
        return None
    description = str(data.get("description") or "").strip()
    pins = _normalize_pins(data.get("pins"))
    return {"description": description, "pins": pins}


def normalize_selector(raw: Any) -> Optional[Dict[str, Any]]:
    data = coerce_json_dict(raw, copy=False)
    if not data and not isinstance(raw, dict):
        return None
    strategy = str(data.get("strategy") or "").strip().lower()
    targets = _normalize_targets(data.get("targets"))
    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    unlisted = bool(data.get("unlisted", False))
    metadata = _normalize_metadata(data.get("metadata"))

    settings_raw = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    spillover = settings_raw.get("spillover", data.get("spillover", 1))
    try:
        spillover_n = int(spillover)
    except (TypeError, ValueError):
        spillover_n = 1
    if spillover_n < 1:
        spillover_n = 1

    block: Dict[str, Any] = {
        "strategy": strategy,
        "targets": targets,
    }
    if name:
        block["name"] = name
    if description:
        block["description"] = description
    if unlisted:
        block["unlisted"] = True
    if metadata:
        block["metadata"] = metadata
    if strategy == "spillover":
        block["settings"] = {"spillover": spillover_n}
    return block


def _normalized_id_collisions(raw_map: Any, section: str) -> List[str]:
    """Detect distinct keys that collapse to the same normalized id."""
    if not isinstance(raw_map, dict):
        return []
    seen: Dict[str, str] = {}
    errors: List[str] = []
    for key in raw_map.keys():
        original = str(key)
        nid = _normalize_id(original)
        if not nid:
            errors.append(f"{section}: empty or invalid id {original!r}")
            continue
        prior = seen.get(nid)
        if prior is not None and prior != original:
            errors.append(
                f"{section}: {prior!r} and {original!r} both normalize to {nid!r}"
            )
        else:
            seen[nid] = original
    return errors


def normalize_routing_document(raw: Any) -> Dict[str, Any]:
    data = coerce_json_dict(raw, copy=False)
    profiles_in = data.get("profiles") if isinstance(data.get("profiles"), dict) else {}
    selectors_in = (
        data.get("selectors") if isinstance(data.get("selectors"), dict) else {}
    )

    profiles: Dict[str, Any] = {}
    for key, value in profiles_in.items():
        profile_id = _normalize_id(key)
        if not profile_id:
            continue
        normalized = normalize_profile(value)
        if normalized is None:
            continue
        profiles[profile_id] = normalized

    selectors: Dict[str, Any] = {}
    for key, value in selectors_in.items():
        selector_id = _normalize_id(key)
        if not selector_id:
            continue
        normalized = normalize_selector(value)
        if normalized is None:
            continue
        selectors[selector_id] = normalized

    return {"profiles": profiles, "selectors": selectors}


def catalog_swap_names(store=None) -> Tuple[set[str], set[str]]:
    """Return (model YAML keys, all claimed names including aliases/sub-ids)."""
    store = store or get_store()
    model_ids: set[str] = set()
    claimed: set[str] = set()
    for model in store.list_models():
        stable = resolve_llama_swap_id(model)
        if not stable:
            continue
        model_ids.add(stable)
        config = effective_model_config(normalize_model_config(model.get("config")))
        claimed.update(collect_claimed_swap_names(model, config))
    return model_ids, claimed


def validate_routing_document(
    routing: Dict[str, Any],
    *,
    store=None,
    raw: Any = None,
) -> List[str]:
    """Return hard validation errors (empty list means saveable)."""
    source = raw if raw is not None else routing
    source_dict = coerce_json_dict(source, copy=False)
    errors: List[str] = []
    errors.extend(_normalized_id_collisions(source_dict.get("profiles"), "profiles"))
    errors.extend(_normalized_id_collisions(source_dict.get("selectors"), "selectors"))

    doc = normalize_routing_document(routing if raw is None else source)
    model_ids, claimed = catalog_swap_names(store)
    selector_ids = set(doc["selectors"].keys())
    profile_ids = set(doc["profiles"].keys())

    for selector_id, selector in doc["selectors"].items():
        strategy = selector.get("strategy") or ""
        if strategy not in SELECTOR_STRATEGIES:
            errors.append(
                f"selectors.{selector_id}.strategy must be one of: "
                + ", ".join(sorted(SELECTOR_STRATEGIES))
            )
        targets = selector.get("targets") or []
        if not targets:
            errors.append(
                f"selectors.{selector_id}.targets must contain at least one entry"
            )
        if selector_id in model_ids or selector_id in claimed:
            errors.append(
                f"selectors.{selector_id}: id conflicts with an existing model or alias"
            )
        for idx, target in enumerate(targets):
            if target in selector_ids:
                errors.append(
                    f"selectors.{selector_id}.targets[{idx}] references selector "
                    f"{target!r}; selector chaining is not supported"
                )
            if strategy == "warm" and "/" in target:
                errors.append(
                    f"selectors.{selector_id}.targets[{idx}] must be a local model "
                    f"for strategy 'warm' (got peer id {target!r})"
                )
        if strategy == "spillover":
            settings = selector.get("settings") or {}
            spillover = settings.get("spillover", 1)
            try:
                if int(spillover) < 1:
                    errors.append(
                        f"selectors.{selector_id}.settings.spillover must be >= 1"
                    )
            except (TypeError, ValueError):
                errors.append(
                    f"selectors.{selector_id}.settings.spillover must be an integer >= 1"
                )

    for profile_id, profile in doc["profiles"].items():
        pins = profile.get("pins") or {}
        if not pins:
            errors.append(f"profiles.{profile_id}.pins must contain at least one pin")

    for shared_id in sorted(selector_ids & profile_ids):
        errors.append(f"id {shared_id!r} is used by both a profile and a selector")

    # Preserve stable order while deduping.
    return list(dict.fromkeys(errors))


def routing_warnings(routing: Dict[str, Any], *, store=None) -> List[str]:
    """Non-fatal warnings (unknown targets, etc.)."""
    doc = normalize_routing_document(routing)
    warnings: List[str] = []
    model_ids, claimed = catalog_swap_names(store)
    selector_ids = set(doc["selectors"].keys())
    known = claimed | model_ids | selector_ids

    for selector_id, selector in doc["selectors"].items():
        for idx, target in enumerate(selector.get("targets") or []):
            if target not in known and "/" not in target:
                warnings.append(
                    f"selectors.{selector_id}.targets[{idx}] references unknown "
                    f"model/alias {target!r}"
                )

    for profile_id, profile in doc["profiles"].items():
        for pin_id, target in (profile.get("pins") or {}).items():
            if not target:
                continue
            if target not in known and "/" not in target:
                warnings.append(
                    f"profiles.{profile_id}.pins.{pin_id} references unknown "
                    f"model/alias/selector {target!r}"
                )
    return list(dict.fromkeys(warnings))


def get_routing_document(store=None) -> Dict[str, Any]:
    store = store or get_store()
    return normalize_routing_document(store.get_llama_swap_routing())


def save_routing_document(raw: Any, *, store=None) -> Dict[str, Any]:
    """Normalize, validate, persist. Raises ValueError on hard errors."""
    store = store or get_store()
    errors = validate_routing_document(raw, store=store, raw=raw)
    if errors:
        raise ValueError("; ".join(errors))
    doc = normalize_routing_document(raw)
    store.set_llama_swap_routing(doc)
    return doc


def routing_for_yaml(store=None) -> Dict[str, Any]:
    """
    Slice of routing suitable for embedding in generated llama-swap YAML.

    Omits empty top-level maps so the generated file stays minimal.
    """
    doc = get_routing_document(store)
    out: Dict[str, Any] = {}
    if doc.get("profiles"):
        out["profiles"] = doc["profiles"]
    if doc.get("selectors"):
        out["selectors"] = doc["selectors"]
    return out
