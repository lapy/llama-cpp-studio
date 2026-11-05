"""R92 pins: per-engine build-settings defaults are engine data.

`_apply_engine_specific_build_defaults` branched on ``engine == "ik_llama"`` to
force ``build_examples``. The requirement is now declared on the spec
(``EngineSpec.build_defaults`` + schema-v2 ``build_defaults``) and exported into
the generated definition; the route layers the declaration over the catalog
defaults.
"""

from __future__ import annotations

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import ENGINE_REGISTRY, EngineSpec, get_engine_spec
from backend.routes.llama_versions import _apply_engine_specific_build_defaults


def test_ik_declares_the_examples_requirement():
    assert dict(get_engine_spec("ik_llama").build_defaults) == {"build_examples": True}
    assert dict(get_engine_spec("llama_cpp").build_defaults) == {}
    definition = load_default_engine_definitions().definitions["ik_llama"]
    assert dict(definition.build_defaults) == {"build_examples": True}
    assert "build_defaults" not in get_engine_spec("ik_llama").to_api_dict()


def test_route_layers_the_declared_defaults():
    applied = _apply_engine_specific_build_defaults(
        "ik_llama", {"build_examples": False, "build_type": "Debug"}
    )
    assert applied["build_examples"] is True
    assert applied["build_type"] == "Debug"
    # llama.cpp declares nothing: settings pass through untouched.
    assert _apply_engine_specific_build_defaults("llama_cpp", {"build_examples": False}) == {
        "build_examples": False
    }


def test_a_declared_engine_can_pin_build_defaults(monkeypatch):
    fake = EngineSpec(
        id="fixture_build",
        label="fixture_build",
        repository_source="fixture_build",
        install_kind="cmake",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"gguf"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        build_defaults={"native": True},
    )
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_build", fake)
    applied = _apply_engine_specific_build_defaults("fixture_build", {"native": False})
    assert applied["native"] is True


def _raw(**extra):
    raw = {
        "schema_version": 2,
        "id": "builddefaultscraft",
        "label": "Builddefaultscraft",
        "repository_source": "Example Org/Builddefaultscraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary_path"],
        "artifact_formats": ["builtin"],
        "package_kinds": ["builtin"],
        "tasks": ["tts"],
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
    }
    raw.update(extra)
    return raw


def test_definition_round_trips_build_defaults():
    definition = validate_definition(
        _raw(build_defaults={"build_examples": True, "native": True}), source="b.yaml"
    )
    assert dict(definition.build_defaults) == {"build_examples": True, "native": True}
    assert dict(definition.to_spec().build_defaults) == dict(definition.build_defaults)


def test_malformed_build_defaults_are_rejected():
    with pytest.raises(DefinitionError, match="must be a mapping"):
        validate_definition(_raw(build_defaults=["x"]), source="b.yaml")
    with pytest.raises(DefinitionError, match="schema_version 2"):
        validate_definition(_raw(schema_version=1, build_defaults={"a": 1}), source="b.yaml")
