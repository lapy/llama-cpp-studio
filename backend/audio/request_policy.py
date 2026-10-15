"""Engine-observed request policy for audio.cpp models.

Endpoint and defaults-key resolution is a contract lookup
(:mod:`backend.engines.audio_cpp.integration`): inspect
``preferred_api_endpoint`` first, then the family's advertised
``api_endpoints`` from ``--list-loaders --json``, then the model spec, then the
registry's deterministic task table. There is no task-set arbitration tree and
no option-name heuristics — an unreported field stays ``none`` / ``unreported``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from backend.engines.audio_cpp.integration import compose_model_contract


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


def _family_capability_fields(version_row: Optional[dict], family_key: str) -> tuple:
    """Family endpoints/policy observed by the version scan (loaders JSON)."""
    caps = (version_row or {}).get("capabilities") if isinstance(version_row, dict) else None
    if not isinstance(caps, dict) or not family_key:
        return [], ""
    endpoints_map = caps.get("family_endpoints")
    policies_map = caps.get("family_policies")
    endpoints = (
        [str(e).strip() for e in endpoints_map.get(family_key) or [] if str(e).strip()]
        if isinstance(endpoints_map, dict)
        else []
    )
    policy = ""
    if isinstance(policies_map, dict):
        policy = str(policies_map.get(family_key) or "")
    return endpoints, policy


def build_request_policy(
    *,
    task: Optional[str] = None,
    family: Optional[str] = None,
    inspection: Optional[dict] = None,
    model_profile: Optional[dict] = None,
    source_path: Optional[str] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    active_version: Optional[dict] = None,
) -> Dict[str, Any]:
    """Build endpoint/defaults/instructions policy from engine-observed data."""
    family_key = str(family or "").strip().lower()
    inspected_family = ""
    if isinstance(inspection, dict):
        inspected_family = str(inspection.get("family") or "").strip().lower()
    # Draft family overrides must not inherit another family's inspect payload.
    if family_key and inspected_family and family_key != inspected_family:
        inspection = None

    inspected = _inspection_tasks(inspection)
    task_key = str(task or "").strip().lower()
    tasks = list(dict.fromkeys([*inspected, *([task_key] if task_key else [])]))

    keys = list(help_option_keys or _help_option_keys(model_profile))
    from backend.engines.audio_cpp.contracts import load_family_contract
    from backend.engines.audio_cpp.spec_fields import iter_spec_options

    contract = load_family_contract(source_path, family_key) if source_path and family_key else None
    keys = list(dict.fromkeys(keys + [
        key for scope, key, _ in iter_spec_options(contract) if scope == "request"
    ]))

    family_endpoints, family_policy = _family_capability_fields(active_version, family_key)
    version_descriptor = (
        (active_version or {}).get("contract")
        if isinstance(active_version, dict)
        else None
    )
    composed = compose_model_contract(
        version_descriptor if isinstance(version_descriptor, dict) else None,
        family=family_key or None,
        task=task_key or None,
        inspection=inspection if isinstance(inspection, dict) else None,
        family_endpoints=family_endpoints,
        family_policy=family_policy,
        spec=contract,
        declared_option_keys=keys,
    )

    return {
        "api_endpoint": composed["api_endpoint"],
        "studio_path": composed["studio_path"],
        "operation_id": composed["operation_id"],
        "upstream_only": composed["upstream_only"],
        "encodings": composed["encodings"],
        "responses": composed["responses"],
        "errors": composed["errors"],
        "endpoint_provenance": composed["endpoint_provenance"],
        "request_defaults_key": composed["request_defaults_key"],
        "instructions_policy": composed["instructions_policy"],
        "instructions_policy_source": composed["instructions_policy_source"],
        "instructions_vocabulary": composed["instructions_vocabulary"],
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
