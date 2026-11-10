"""YAML campaign S2 — startup acceptance inversion (registry as derived view).

The deliberate behavior move: hand-authored AUTHORITATIVE definitions are
now LOADED INTO ``ENGINE_REGISTRY`` at startup (decided Q2: registry
becomes a derived view; one identity system). Everything generated-tier
keeps its prior contract byte-for-byte: the nine exported definitions
must still equal their Python specs, invalid definitions still refuse
startup, and ``missing=`` semantics are unchanged.

RED-first record (pre-inversion legacy bytes): with an authoritative
definition present, ``verify_engine_definitions()`` raised
``RuntimeError("Engine definitions diverged from the Python registry;
refusing startup (drifted=['synthcraft'], …)")`` — the assertions below
expect the NEW acceptance behavior and failed against the old code with
exactly that refusal.
"""

from __future__ import annotations

import pytest

from backend.engines import registry as registry_module
from backend.engines.definitions import startup_check
from backend.engines.definitions.loader import (
    EngineDefinitionRegistry,
    load_default_engine_definitions,
)
from backend.engines.definitions.schema import validate_definition


def _craft_raw(**overrides):
    # SYNTHETIC engine id: never a shipped engine, so these pins cannot
    # collide with tree contents.
    doc = {
        "schema_version": 2,
        "id": "synthcraft",
        "label": "Synthy",
        "repository_source": "synthcraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary"],
        "artifact_formats": ["gguf"],
        "package_kinds": ["binary"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "recipes": {"build": [{"id": "compose", "uses": "env.compose", "produces": ["env"]}]},
    }
    doc.update(overrides)
    return doc


def _expected_count() -> int:
    return len(load_default_engine_definitions().definitions) + 1


@pytest.fixture
def registry_sandbox():
    """Restore every registry surface (dict + derived id sets) after use —
    registration mutates SHARED objects, so tests must hand them back."""
    dict_copy = dict(registry_module.ENGINE_REGISTRY)
    sets = {
        name: set(getattr(registry_module, name))
        for name in (
            "VALID_ENGINE_IDS",
            "EMBEDDINGS_ENGINE_IDS",
            "NATIVE_ENGINE_IDS",
            "GGUF_ENGINE_IDS",
            "HF_SNAPSHOT_ENGINE_IDS",
        )
    }
    yield
    ENGINE_REGISTRY = registry_module.ENGINE_REGISTRY
    ENGINE_REGISTRY.clear()
    ENGINE_REGISTRY.update(dict_copy)
    for name, snapshot in sets.items():
        target = getattr(registry_module, name)
        target.clear()
        target.update(snapshot)


def _verify_with_extra(definition) -> int:
    combined = EngineDefinitionRegistry(
        definitions={**load_default_engine_definitions().definitions, definition.id: definition}
    )
    original = startup_check.load_default_engine_definitions
    startup_check.load_default_engine_definitions = lambda: combined
    try:
        return startup_check.verify_engine_definitions()
    finally:
        startup_check.load_default_engine_definitions = original


def test_authoritative_definition_is_accepted_and_registered(registry_sandbox):
    craft = validate_definition(_craft_raw(), source="synthcraft.yaml")
    assert startup_check.definition_tier("synthcraft") == "authoritative"

    count = _verify_with_extra(craft)

    assert count == _expected_count()  # shipped definitions + the synthetic
    spec = registry_module.ENGINE_REGISTRY.get("synthcraft")
    assert spec is not None
    assert spec.to_api_dict() == craft.to_api_dict()  # Q2: identical shape
    # SHARED derived sets see the registration (from-import holders included)
    assert "synthcraft" in registry_module.VALID_ENGINE_IDS
    assert "synthcraft" in registry_module.NATIVE_ENGINE_IDS  # install_kind cmake
    assert "synthcraft" in registry_module.GGUF_ENGINE_IDS  # gguf artifact format
    assert "synthcraft" not in registry_module.EMBEDDINGS_ENGINE_IDS


def test_reverification_is_idempotent_and_conflicts_still_refuse(registry_sandbox):
    craft = validate_definition(_craft_raw(), source="synthcraft.yaml")
    assert _verify_with_extra(craft) == _expected_count()
    # second pass over an already-registered identical spec: no error, no dup
    assert _verify_with_extra(craft) == _expected_count()

    conflicting = validate_definition(_craft_raw(label="Different"), source="synthcraft.yaml")
    with pytest.raises(RuntimeError, match="refusing startup"):
        _verify_with_extra(conflicting)


def test_registry_drift_refusal_message_is_stable(registry_sandbox):
    """A definition mutated after import must refuse startup, naming the ids."""
    from dataclasses import replace

    base = load_default_engine_definitions().definitions["lmdeploy"]
    drifted = replace(base, label="Drifted LMDeploy")
    mutated = EngineDefinitionRegistry(
        definitions={
            **{
                engine_id: definition
                for engine_id, definition in load_default_engine_definitions().definitions.items()
                if engine_id != "lmdeploy"
            },
            "lmdeploy": drifted,
        }
    )
    original = startup_check.load_default_engine_definitions
    startup_check.load_default_engine_definitions = lambda: mutated
    try:
        with pytest.raises(RuntimeError) as excinfo:
            startup_check.verify_engine_definitions()
    finally:
        startup_check.load_default_engine_definitions = original
    message = str(excinfo.value)
    assert "Engine definitions diverged from the derived engine registry" in message
    assert "drifted=['lmdeploy']" in message
