"""Compose the effective audio.cpp integration contract per engine version.

The composed descriptor is the single source of truth every Studio consumer
(routes, proxy, param registry, frontend) reads. It is derived, cached, and
reproducible: registry entries (provenance ``studio``) overlaid by engine
observations from ``--list-loaders --json`` (provenance ``loaders-json``),
``--inspect`` (``inspect``), and ``model_specs`` (``spec``). An upstream
contract document (audio.cpp#852), once available, overlays everything with
provenance ``upstream``.

Conflict policy: observed beats studio. When an observed endpoint cannot be
matched to any registry operation, that is drift and becomes a warning on the
descriptor, surfaced through the same activation warnings channel as
capability deltas.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.engines.audio_cpp.integration import upstream_adapter
from backend.engines.audio_cpp.integration.contract_registry import (
    CONTRACT_SCHEMA,
    DEFAULTS_KEYS,
    TASKS_RUN_ALIAS,
    defaults_key_for,
    operation_for_native_path,
    registry_descriptor,
    resolve_operation,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)

_INSTRUCTIONS_POLICIES = frozenset(
    {
        "caption_option",
        "text_prefix",
        "openai_instruct",
        "soft_tags",
        "none",
    }
)


def normalize_instructions_policy(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return text if text in _INSTRUCTIONS_POLICIES else ""


# Deterministic alias table for instructions policy inferred from *declared*
# engine options (model spec request options / model-aware help). No fuzzy
# matching: the option key must equal one of these or start with ``caption_``.
_INSTRUCT_OPTION_KEYS = frozenset({"instruction", "instruct", "instructions"})


def policy_from_declared_options(keys: Optional[Sequence[str]]) -> str:
    """Instructions policy from option names the engine itself declares."""
    names = {
        str(key or "").strip().lower().replace("-", "_")
        for key in (keys or [])
        if str(key or "").strip()
    }
    if not names:
        return ""
    if "caption" in names or any(name.startswith("caption_") for name in names):
        return "caption_option"
    if names & _INSTRUCT_OPTION_KEYS:
        return "openai_instruct"
    return ""


def compose_version_contract(
    capabilities: Optional[dict],
    *,
    source_path: Optional[str] = None,
    upstream: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compose the version-level descriptor from a version scan's capabilities.

    ``capabilities`` is the dict persisted by ``scan_audio_cpp_version``
    (``families``, ``family_tasks``, ``family_modes``, ``family_policies``,
    ``family_endpoints``, ``family_contract_sources`` ...). Missing pieces
    degrade to the registry; they are never invented.
    """
    caps = capabilities if isinstance(capabilities, dict) else {}
    family_endpoints = _string_map_list(caps.get("family_endpoints"))
    family_policies = {
        str(family).strip().lower(): normalize_instructions_policy(policy)
        for family, policy in (caps.get("family_policies") or {}).items()
        if normalize_instructions_policy(policy)
    }
    family_modes = _string_map_list(caps.get("family_modes"))
    family_tasks = _string_map_list(caps.get("family_tasks"))
    families = [
        str(f).strip().lower()
        for f in (caps.get("families") or [])
        if str(f).strip()
    ]

    operations = registry_descriptor()
    provenance_by_id = {
        str(op["id"]): str(op.get("provenance") or "studio") for op in operations
    }
    drift_warnings: List[str] = []

    # Upstream document (audio.cpp#852) overlays the Studio table wholesale.
    upstream_ops = upstream_adapter.parse_upstream_operations(upstream or {})
    if upstream_ops:
        by_id = {str(op["id"]): op for op in operations}
        for op in upstream_ops:
            existing = by_id.get(op["id"])
            if existing is not None and existing["native_path"] != op["native_path"]:
                drift_warnings.append(
                    f"Upstream contract moved operation {op['id']} from "
                    f"{existing['native_path']} to {op['native_path']}"
                )
            by_id[op["id"]] = op
            provenance_by_id[op["id"]] = "upstream"
        operations = list(by_id.values())

    # Family-observed endpoints validate/annotate the operation table; a
    # family advertising an unknown route is drift, not a silent remap.
    observed_endpoints = sorted(
        {endpoint for values in family_endpoints.values() for endpoint in values}
    )
    for endpoint in observed_endpoints:
        if endpoint in (TASKS_RUN_ALIAS, "/v1/tasks/run"):
            continue
        if operation_for_native_path(endpoint) is None:
            drift_warnings.append(
                "Engine advertises an endpoint Studio has no operation for: "
                f"{endpoint}; requests fall through to /v1/tasks/run handling"
            )

    family_routing: Dict[str, Dict[str, Any]] = {}
    for family in families:
        endpoints = family_endpoints.get(family) or []
        op = resolve_operation(family_endpoints=endpoints) if endpoints else None
        entry: Dict[str, Any] = {
            "tasks": family_tasks.get(family) or [],
            "modes": family_modes.get(family) or [],
            "endpoints": list(endpoints),
            "api_endpoint": endpoints[0] if endpoints else None,
            "instructions_policy": family_policies.get(family) or None,
        }
        if op is not None:
            entry["operation_id"] = op.id
            entry["studio_path"] = op.studio_path
            entry["request_defaults_key"] = defaults_key_for(op)
        family_routing[family] = entry

    task_routing: Dict[str, str] = {}
    for op in operations:
        for task in op.get("serves_tasks") or []:
            task_routing.setdefault(str(task), str(op["id"]))

    descriptor: Dict[str, Any] = {
        "schema": CONTRACT_SCHEMA,
        "operations": operations,
        "provenance": provenance_by_id,
        "defaults_keys": dict(DEFAULTS_KEYS),
        "task_routing": task_routing,
        "generic_studio_path": TASKS_RUN_ALIAS,
        "families": families,
        "family_routing": family_routing,
        "drift_warnings": drift_warnings,
        "upstream_contract": bool(upstream_ops),
    }
    if source_path:
        descriptor["spec_source"] = str(source_path)
    descriptor["descriptor_fingerprint"] = hashlib.sha256(
        json.dumps(
            _fingerprint_input(descriptor), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return descriptor


def _fingerprint_input(descriptor: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: descriptor.get(key)
        for key in (
            "schema",
            "operations",
            "defaults_keys",
            "task_routing",
            "generic_studio_path",
            "family_routing",
        )
    }


def compose_model_contract(
    version_descriptor: Optional[dict],
    *,
    family: Optional[str] = None,
    task: Optional[str] = None,
    inspection: Optional[dict] = None,
    family_endpoints: Optional[List[str]] = None,
    family_policy: Optional[str] = None,
    spec: Optional[dict] = None,
    declared_option_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Compose the per-model request policy from engine-observed fields only.

    Resolution order (first hit wins, no heuristics): inspect
    ``preferred_api_endpoint`` (or per-task row), family ``api_endpoints`` from
    loaders JSON (via the version descriptor's family routing), model spec
    ``preferred_api_endpoint``, deterministic task table. The operation's
    encodings/responses ride along so callers never re-derive them.
    """
    descriptor = version_descriptor if isinstance(version_descriptor, dict) else {}
    family_key = str(family or "").strip().lower()
    routing = (descriptor.get("family_routing") or {}).get(family_key) or {}
    preferred = ""
    policy_raw = ""
    vocabulary: Any = None
    if isinstance(inspection, dict):
        preferred = str(
            inspection.get("preferred_api_endpoint")
            or inspection.get("request_surface")
            or ""
        ).strip()
        policy_raw = str(inspection.get("instructions_policy") or "")
        vocabulary = (
            inspection.get("instructions_vocabulary")
            or inspection.get("instructions_vocab")
        )
        task_key = str(task or "").strip().lower()
        for row in inspection.get("tasks") or []:
            if (
                isinstance(row, dict)
                and task_key
                and str(row.get("task") or row.get("name") or row.get("id") or "").lower()
                == task_key
            ):
                preferred = str(
                    row.get("preferred_api_endpoint")
                    or row.get("request_surface")
                    or preferred
                ).strip()
                policy_raw = str(row.get("instructions_policy") or policy_raw)
                vocabulary = row.get("instructions_vocabulary") or vocabulary
                break

    endpoints = [str(e).strip() for e in (family_endpoints or []) if str(e).strip()]
    if not endpoints:
        endpoints = [str(e).strip() for e in (routing.get("endpoints") or []) if str(e).strip()]
    spec_endpoint = str((spec or {}).get("preferred_api_endpoint") or "").strip()
    op = resolve_operation(
        task=task or "",
        preferred_api_endpoint=preferred or None,
        family_endpoints=endpoints or ([spec_endpoint] if spec_endpoint else None),
    )

    declared_policy = policy_from_declared_options(declared_option_keys)
    policy = (
        normalize_instructions_policy(policy_raw)
        or normalize_instructions_policy(family_policy)
        or normalize_instructions_policy((spec or {}).get("instructions_policy"))
        or declared_policy
    )
    provenance = "engine"
    if not policy_raw and not family_policy and not (spec or {}).get("instructions_policy"):
        provenance = "engine_options" if declared_policy else "unreported"

    vocab_list: Optional[List[str]] = None
    if isinstance(vocabulary, list):
        vocab_list = [str(item).strip() for item in vocabulary if str(item).strip()]
    elif isinstance(vocabulary, dict):
        collected: List[str] = []
        for value in vocabulary.values():
            if isinstance(value, list):
                collected.extend(str(item).strip() for item in value if str(item).strip())
            elif value:
                collected.append(str(value).strip())
        vocab_list = list(dict.fromkeys(collected)) or None

    observed_source = (
        "inspect" if preferred else "loaders-json" if endpoints else "spec" if spec_endpoint else "registry"
    )
    return {
        "operation_id": op.id,
        # ``api_endpoint`` is the URL Studio and the UI must call: the public
        # llama-swap path for the generic route, the native path elsewhere.
        # ``native_path`` stays engine-side for proxy /upstream rewriting.
        "api_endpoint": op.studio_path,
        "native_path": op.native_path,
        "studio_path": op.studio_path,
        "upstream_only": op.upstream_only,
        "request_defaults_key": defaults_key_for(op),
        "body_shape": op.as_dict()["body_shape"],
        "encodings": list(op.encodings),
        "responses": list(op.responses),
        "errors": op.errors,
        "endpoint_provenance": observed_source if observed_source != "registry" else "studio",
        "instructions_policy": policy or "none",
        "instructions_policy_source": provenance,
        "instructions_vocabulary": vocab_list,
    }


def _string_map_list(value: Any) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    if not isinstance(value, dict):
        return out
    for key, raw in value.items():
        family = str(key).strip().lower()
        if not family:
            continue
        if isinstance(raw, str):
            out[family] = [raw.strip()] if raw.strip() else []
        elif isinstance(raw, list):
            out[family] = [str(item).strip() for item in raw if str(item).strip()]
    return out


__all__ = [
    "compose_model_contract",
    "compose_version_contract",
    "normalize_instructions_policy",
    "policy_from_declared_options",
]
