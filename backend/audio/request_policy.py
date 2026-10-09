"""Schema-driven request policy for audio.cpp models (endpoint + instructions)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from backend.engines.audio_cpp.discovery import (
    infer_instructions_policy,
    resolve_api_endpoint,
    resolve_defaults_key_for_endpoint,
)


def _inspection_tasks(inspection: Optional[dict]) -> List[str]:
    if not isinstance(inspection, dict):
        return []
    tasks: List[str] = []
    raw_tasks = inspection.get("tasks")
    if isinstance(raw_tasks, list):
        for item in raw_tasks:
            if isinstance(item, dict):
                name = item.get("task") or item.get("name") or item.get("id")
                if name:
                    tasks.append(str(name))
            elif item:
                tasks.append(str(item))
    for key in ("supported_tasks", "task_names"):
        value = inspection.get(key)
        if isinstance(value, list):
            tasks.extend(str(v) for v in value if v)
    single = inspection.get("task")
    if single:
        tasks.append(str(single))
    return list(dict.fromkeys(t.strip().lower() for t in tasks if str(t).strip()))


def _help_option_keys(profile: Optional[dict]) -> List[str]:
    keys: List[str] = []
    if not isinstance(profile, dict):
        return keys
    for section in profile.get("sections") or []:
        if not isinstance(section, dict):
            continue
        for param in section.get("params") or []:
            if not isinstance(param, dict) or param.get("scope") not in {None, "request_option"}:
                continue
            name = param.get("name") or param.get("key") or param.get("cli")
            if name:
                keys.append(str(name).lstrip("-"))
            for alias in param.get("aliases") or []:
                keys.append(str(alias).lstrip("-"))
    # Flat param lists used by some scans
    for param in profile.get("params") or []:
        if isinstance(param, dict) and param.get("scope") in {None, "request_option"}:
            name = param.get("name") or param.get("key")
            if name:
                keys.append(str(name).lstrip("-"))
    return list(dict.fromkeys(k for k in keys if k))


def build_request_policy(
    *,
    task: Optional[str] = None,
    family: Optional[str] = None,
    inspection: Optional[dict] = None,
    model_profile: Optional[dict] = None,
    source_path: Optional[str] = None,
    help_option_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build endpoint/defaults/instructions policy from inspect + help signals."""
    from backend.audio.task_profiles import (
        _family_key,
        _synthetic_inspection_tasks,
    )

    # Prefer inspect/loaders-advertised tasks. Synthetic multi-route expansion
    # is only a fallback when the engine did not advertise tasks.
    inspected = _inspection_tasks(inspection)
    inspected_family = ""
    if isinstance(inspection, dict):
        inspected_family = str(inspection.get("family") or "").strip().lower()
    family_key = _family_key(family)
    # Draft family overrides must not inherit another family's inspect task list.
    if family_key and inspected_family and family_key != inspected_family:
        inspected = []
        inspection = None
        model_profile = None
    synthetic = _synthetic_inspection_tasks(task, family)
    if inspected:
        tasks = list(dict.fromkeys(inspected))
        task_key = str(task or "").strip().lower()
        if task_key and task_key not in tasks:
            tasks.append(task_key)
    else:
        tasks = list(dict.fromkeys(synthetic))
    keys = list(help_option_keys or _help_option_keys(model_profile))
    from backend.engines.audio_cpp.contracts import load_family_contract
    from backend.engines.audio_cpp.spec_fields import iter_spec_options

    contract = load_family_contract(source_path, family_key) if source_path and family_key else None
    keys = list(dict.fromkeys(keys + [
        key for scope, key, _ in iter_spec_options(contract) if scope == "request"
    ]))
    supports_style = None
    if isinstance(inspection, dict):
        caps = inspection.get("capabilities")
        if isinstance(caps, dict) and "supports_style_condition" in caps:
            supports_style = bool(caps.get("supports_style_condition"))
        elif "supports_style_condition" in inspection:
            raw = inspection.get("supports_style_condition")
            if isinstance(raw, str):
                supports_style = raw.strip().lower() in {"1", "true", "yes"}
            else:
                supports_style = bool(raw)

    route_task = task

    preferred_endpoint = None
    upstream_policy = None
    upstream_vocab: Any = None
    if isinstance(inspection, dict):
        preferred_endpoint = (
            inspection.get("preferred_api_endpoint")
            or inspection.get("request_surface")
        )
        upstream_policy = inspection.get("instructions_policy")
        upstream_vocab = (
            inspection.get("instructions_vocabulary")
            or inspection.get("instructions_vocab")
        )

    if isinstance(inspection, dict):
        for row in inspection.get("tasks") or []:
            if isinstance(row, dict) and (row.get("task") or row.get("name") or row.get("id")) == task:
                preferred_endpoint = row.get("preferred_api_endpoint") or row.get("request_surface") or preferred_endpoint
                upstream_policy = row.get("instructions_policy") or upstream_policy
                upstream_vocab = row.get("instructions_vocabulary") or upstream_vocab
                break
    preferred_endpoint = preferred_endpoint or (contract or {}).get("preferred_api_endpoint")
    upstream_policy = upstream_policy or (contract or {}).get("instructions_policy")

    endpoint = resolve_api_endpoint(
        task=route_task,
        inspection_tasks=tasks,
        help_option_keys=keys,
        preferred_api_endpoint=(
            str(preferred_endpoint) if preferred_endpoint is not None else None
        ),
    )
    defaults_key = resolve_defaults_key_for_endpoint(endpoint)
    instructions_policy = infer_instructions_policy(
        help_option_keys=keys,
        supports_style_condition=supports_style,
        family=family,
        inspection_policy=(
            str(upstream_policy) if upstream_policy is not None else None
        ),
    )
    vocabulary: Optional[List[str]] = None
    if isinstance(upstream_vocab, list):
        vocabulary = [str(item).strip() for item in upstream_vocab if str(item).strip()]
    elif isinstance(upstream_vocab, dict):
        # Allow { "tags": [...] } or category -> list maps
        collected: List[str] = []
        for value in upstream_vocab.values():
            if isinstance(value, list):
                collected.extend(str(item).strip() for item in value if str(item).strip())
            elif value:
                collected.append(str(value).strip())
        vocabulary = list(dict.fromkeys(collected)) or None

    return {
        "api_endpoint": endpoint,
        "request_defaults_key": defaults_key,
        "instructions_policy": instructions_policy,
        "instructions_policy_source": (
            "engine" if upstream_policy else "engine_options" if instructions_policy != "none" else "unreported"
        ),
        "instructions_vocabulary": vocabulary,
        "inspection_tasks": tasks,
        "help_option_keys": keys,
    }


def _normalize_vocab_token(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ").replace("-", " ")


def validate_instructions_against_policy(
    instructions: Optional[str],
    *,
    policy: str,
    family: Optional[str] = None,
    vocabulary: Optional[Sequence[str]] = None,
) -> List[str]:
    text = str(instructions or "").strip()
    if not text:
        return []
    policy_key = str(policy or "none").strip().lower()
    vocab = [
        str(item).strip()
        for item in (vocabulary or [])
        if str(item).strip()
    ]

    if policy_key == "caption_option":
        return [
            "This model uses options.caption for voice design, not instructions. "
            "Set caption under the request defaults options object."
        ]
    if policy_key == "text_prefix":
        return [
            "This model does not use instructions. Put style prefixes at the start of "
            "input text instead, e.g. '(calm) Hello world'."
        ]
    if policy_key == "soft_tags":
        if vocab:
            allowed = {_normalize_vocab_token(item) for item in vocab}
            parts = [p.strip() for p in text.split(",") if p.strip()]
            if not parts:
                return [
                    "Instructions should be a comma-separated list of voice attributes."
                ]
            unknown = [
                part
                for part in parts
                if _normalize_vocab_token(part) not in allowed
            ]
            if unknown:
                examples = ", ".join(vocab[:8])
                return [
                    "Unsupported instruction attribute(s): "
                    + ", ".join(unknown)
                    + (f". Engine vocabulary examples: {examples}." if examples else ".")
                ]
            return []
        parts = [p.strip() for p in text.split(",") if p.strip()]
        if len(parts) < 1:
            return ["Instructions should be a comma-separated list of voice attributes."]
        return []
    if policy_key in {"none", ""}:
        return [
            "This model does not accept an instructions field in request defaults."
        ]
    return []


__all__ = [
    "build_request_policy",
    "validate_instructions_against_policy",
]
