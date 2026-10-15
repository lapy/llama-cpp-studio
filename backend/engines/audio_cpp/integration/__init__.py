"""Single-source audio.cpp integration contract (shim for audio.cpp#852).

- :mod:`contract_registry` — the Studio-owned operation table.
- :mod:`composer` — overlays live engine observations on the registry.
- :mod:`upstream_adapter` — parses the upstream contract once #852 ships.
"""

from __future__ import annotations

from backend.engines.audio_cpp.integration.composer import (
    compose_model_contract,
    compose_version_contract,
    normalize_instructions_policy,
    policy_from_declared_options,
)
from backend.engines.audio_cpp.integration.contract_registry import (
    CONTRACT_SCHEMA,
    DEFAULTS_KEYS,
    NATIVE_AUDIO_BASE,
    STUDIO_OWNED_FLAGS,
    TASKS_RUN_ALIAS,
    defaults_key_for,
    is_generic_route,
    is_native_api_path,
    operation_by_id,
    operation_for_native_path,
    operation_for_studio_path,
    operation_for_task,
    operations,
    registry_descriptor,
    resolve_operation,
)
from backend.engines.audio_cpp.integration.upstream_adapter import (
    fetch_upstream_contract,
)

__all__ = [
    "CONTRACT_SCHEMA",
    "DEFAULTS_KEYS",
    "NATIVE_AUDIO_BASE",
    "STUDIO_OWNED_FLAGS",
    "TASKS_RUN_ALIAS",
    "compose_model_contract",
    "compose_version_contract",
    "defaults_key_for",
    "fetch_upstream_contract",
    "is_generic_route",
    "is_native_api_path",
    "normalize_instructions_policy",
    "operation_by_id",
    "operation_for_native_path",
    "operation_for_studio_path",
    "operation_for_task",
    "operations",
    "policy_from_declared_options",
    "registry_descriptor",
    "resolve_operation",
]
