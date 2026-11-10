"""Declarative inference-engine capabilities shared by backend and frontend APIs.

The registry intentionally contains data only.  Engine-specific installers, scanners,
and runtime adapters remain in their own modules and are selected by the string keys
stored here.  This keeps model/config validation independent from heavyweight manager
imports and avoids circular dependencies with the YAML data store.

One deliberate mutation exists: definition-owned engines are validated and
loaded from ``backend/engines/definitions/*.yaml`` at import (see the bootstrap
at the end of this module), so the registry is a DERIVED VIEW of the validated
YAML definitions.  The derived id-sets are SHARED
mutable objects precisely so that startup registrations (fixture personas) and
the import-time bootstrap are observed by ``from ... import VALID_ENGINE_IDS``
holders.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from backend.engines.spec import EngineSpec


ENGINE_REGISTRY: Dict[str, EngineSpec] = {}

# SHARED mutable id-sets (see module docstring): startup registrations
# update these objects IN PLACE — never rebind them.
VALID_ENGINE_IDS: Set[str] = set(ENGINE_REGISTRY)
EMBEDDINGS_ENGINE_IDS: Set[str] = {
    key for key, spec in ENGINE_REGISTRY.items() if spec.supports_embeddings
}
NATIVE_ENGINE_IDS: Set[str] = {
    key for key, spec in ENGINE_REGISTRY.items() if spec.install_kind == "cmake"
}
GGUF_ENGINE_IDS: Set[str] = {
    key for key, spec in ENGINE_REGISTRY.items() if "gguf" in spec.artifact_formats
}


def inferred_engines_for_artifact_format(artifact_format: Optional[str]) -> List[str]:
    """Return engines whose compatibility can be inferred from format alone.

    Engines that also accept curated or mixed packages (currently audio.cpp) are
    excluded so a ``.safetensors`` file never implies verified audio support.
    """
    fmt = str(artifact_format or "").strip().lower()
    if not fmt:
        return []
    return [
        spec.id for spec in ENGINE_REGISTRY.values() if spec.artifact_formats == frozenset({fmt})
    ]


HF_SNAPSHOT_ENGINE_IDS: Set[str] = set(inferred_engines_for_artifact_format("safetensors"))


def _refresh_derived_engine_id_sets() -> None:
    """Recompute the SHARED id-sets IN PLACE after a registration."""
    VALID_ENGINE_IDS.update(ENGINE_REGISTRY)
    EMBEDDINGS_ENGINE_IDS.update(
        key for key, spec in ENGINE_REGISTRY.items() if spec.supports_embeddings
    )
    NATIVE_ENGINE_IDS.update(
        key for key, spec in ENGINE_REGISTRY.items() if spec.install_kind == "cmake"
    )
    GGUF_ENGINE_IDS.update(
        key for key, spec in ENGINE_REGISTRY.items() if "gguf" in spec.artifact_formats
    )
    HF_SNAPSHOT_ENGINE_IDS.update(inferred_engines_for_artifact_format("safetensors"))


def register_engine_spec(spec: EngineSpec) -> bool:
    """Register an authoritative engine spec at startup (YAML campaign S2).

    Idempotent when an existing entry agrees exactly (same ``to_api_dict``);
    any disagreement REFUSES rather than overwriting — generated engines are
    never passed here, their equality is enforced by startup drift checks.
    Returns True when the registry grew.
    """
    existing = ENGINE_REGISTRY.get(spec.id)
    if existing is not None:
        if existing.to_api_dict() == spec.to_api_dict():
            return False
        raise RuntimeError(
            f"Engine spec conflict for {spec.id!r}: authoritative definition "
            "disagrees with the existing registry entry"
        )
    ENGINE_REGISTRY[spec.id] = spec
    _refresh_derived_engine_id_sets()
    return True


def get_engine_spec(engine_id: Optional[str]) -> Optional[EngineSpec]:
    return ENGINE_REGISTRY.get(str(engine_id or ""))


def engine_specs() -> Iterable[EngineSpec]:
    return ENGINE_REGISTRY.values()


def engine_ids_with_install_kind(kind: str) -> Tuple[str, ...]:
    """Engine ids whose declared ``install_kind`` matches (registry-derived).

    Replaces hardcoded per-family id tuples (venv retry/orphan discovery, cmake
    build-editability): the capability follows the declaration, so a YAML engine
    that declares the same install kind gets the same behavior with no code edit.
    """
    return tuple(
        sorted(
            engine_id
            for engine_id, spec in ENGINE_REGISTRY.items()
            if spec.install_kind == str(kind)
        )
    )


def native_source_engine_ids() -> Tuple[str, ...]:
    """Native source-build engines: ``install_kind=cmake`` + ``runtime_kind=
    llama_server`` (R105).

    Replaces the ``("llama_cpp", "ik_llama")`` tuples in the llama-versions
    routes: audio.cpp is cmake but runtime ``audio_cpp``; unsloth is prebuilt.
    """
    return tuple(
        sorted(
            engine_id
            for engine_id, spec in ENGINE_REGISTRY.items()
            if spec.install_kind == "cmake" and spec.runtime_kind == "llama_server"
        )
    )


def llama_server_engine_ids() -> Tuple[str, ...]:
    """Every engine whose runtime is ``llama_server`` (R106).

    This is the llama-family scan/help family: llama.cpp, ik_llama and the
    prebuilt unsloth build. audio.cpp has its own runtime and is excluded.
    """
    return tuple(
        sorted(
            engine_id
            for engine_id, spec in ENGINE_REGISTRY.items()
            if spec.runtime_kind == "llama_server"
        )
    )


def engine_registry_payload() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "engines": [spec.to_api_dict() for spec in ENGINE_REGISTRY.values()],
    }


def required_active_path_fields(engine_id: str, row: Optional[dict]) -> Tuple[str, ...]:
    spec = get_engine_spec(engine_id)
    if spec is None:
        return ()
    return spec.active_path_fields


def active_engine_row_is_runnable(engine_id: str, row: Optional[dict]) -> bool:
    """Return whether a stored active-version row has the paths its engine needs."""
    spec = get_engine_spec(engine_id)
    if not spec or not isinstance(row, dict):
        return False
    status = str(row.get("build_status") or "").strip().lower()
    if status and status != "ready":
        return False
    return all(bool(row.get(field)) for field in required_active_path_fields(engine_id, row))


# ---------------------------------------------------------------------------
# Definition-owned engines: YAML is the source of truth
# ---------------------------------------------------------------------------
# An engine whose definition is CONVERTED (hand-authored, not exported from the
# literal map above) is validated and loaded here, at import, into the SAME
# typed registry. That keeps every import-time consumer (valid-id sets, model
# compatibility, launch/discovery dispatch, API descriptors) identical for
# converted and generated engines, so a YAML edit changes production behavior
# with no Python edit and no regeneration step.
#
# Definition-owned engines are validated and loaded here, at import, into the
# SAME typed registry, so the registry stays a derived view of the YAML.
DEFINITION_ENGINE_IDS: Set[str] = set()


def _bootstrap_definition_engines() -> None:
    from backend.engines.definitions.loader import load_default_engine_definitions

    try:
        registry = load_default_engine_definitions()
    except Exception as exc:  # DefinitionError and friends
        raise RuntimeError(
            f"Engine definitions invalid; refusing to build the engine registry: {exc}"
        ) from exc
    for engine_id, definition in sorted(registry.definitions.items()):
        if engine_id in ENGINE_REGISTRY:
            continue
        ENGINE_REGISTRY[engine_id] = definition.to_spec()
        DEFINITION_ENGINE_IDS.add(engine_id)
    if DEFINITION_ENGINE_IDS:
        _refresh_derived_engine_id_sets()


_bootstrap_definition_engines()

# Every production engine the application ships (generated + definition-owned),
# captured after the bootstrap. Import-time consumers that mean "the shipped
# engines" (rather than "the exporter-generated tier") should use this.
SHIPPED_ENGINE_IDS: FrozenSet[str] = frozenset(ENGINE_REGISTRY)
