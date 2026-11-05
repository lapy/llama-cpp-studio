"""R76 pins: model-compatibility rules are engine data, not id branches.

`models/config.default_engine_for_format` hardcoded ``safetensors -> lmdeploy``
and `models/schema._default_package_kind` branched on ``engine == "audio_cpp"``.
Both are now declared on `EngineSpec.model_compat` (schema-v2 `model_compat`)
and exported into the definitions.
"""

from __future__ import annotations

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import ENGINE_REGISTRY, EngineSpec, get_engine_spec
from backend.models.config import DEFAULT_ENGINE, default_engine_for_format
from backend.models.schema import _default_package_kind


def test_declared_model_compat_matches_the_historical_rules():
    assert dict(get_engine_spec("lmdeploy").model_compat) == {
        "default_for_formats": ["safetensors"]
    }
    assert dict(get_engine_spec("audio_cpp").model_compat) == {
        "prepared_bundle_formats": ["mixed", "original"]
    }
    for engine_id, spec in ENGINE_REGISTRY.items():
        assert "model_compat" not in spec.to_api_dict(), engine_id


def test_model_compat_is_exported_with_the_definition():
    definitions = load_default_engine_definitions().definitions
    assert dict(definitions["lmdeploy"].model_compat) == {"default_for_formats": ["safetensors"]}
    assert dict(definitions["audio_cpp"].model_compat) == {
        "prepared_bundle_formats": ["mixed", "original"]
    }


def test_default_engine_for_format_reads_the_declaration():
    assert default_engine_for_format("safetensors") == "lmdeploy"
    assert default_engine_for_format("SAFETENSORS") == "lmdeploy"
    assert default_engine_for_format("gguf") == DEFAULT_ENGINE
    assert default_engine_for_format(None) == DEFAULT_ENGINE
    assert default_engine_for_format("") == DEFAULT_ENGINE


def test_a_declared_engine_can_claim_a_default_format(monkeypatch):
    fake = EngineSpec(
        id="fixture_default",
        label="fixture_default",
        repository_source="fixture_default",
        install_kind="prebuilt",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"customfmt"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        model_compat={"default_for_formats": ["customfmt"]},
    )
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_default", fake)
    assert default_engine_for_format("customfmt") == "fixture_default"


def test_default_package_kind_reads_the_declaration():
    # audio's declared bundle formats.
    assert _default_package_kind({"engine": "audio_cpp"}, "mixed") == "prepared_bundle"
    assert _default_package_kind({"engine": "audio_cpp"}, "original") == "prepared_bundle"
    # audio with a non-bundle format keeps the generic rules.
    assert _default_package_kind({"engine": "audio_cpp"}, "gguf") == "single_file"
    # A record with a family is treated as a package even without an engine.
    assert _default_package_kind({"family": "speech"}, "mixed") == "prepared_bundle"
    # Engines without the declaration are unaffected.
    assert _default_package_kind({"engine": "llama_cpp"}, "mixed") == "unknown"


def test_a_declared_engine_can_declare_bundle_formats(monkeypatch):
    fake = EngineSpec(
        id="fixture_bundle",
        label="fixture_bundle",
        repository_source="fixture_bundle",
        install_kind="prebuilt",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"mixed"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        model_compat={"prepared_bundle_formats": ["mixed"]},
    )
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_bundle", fake)
    assert _default_package_kind({"engine": "fixture_bundle"}, "mixed") == "prepared_bundle"


def _raw(**extra):
    raw = {
        "schema_version": 2,
        "id": "compatcraft",
        "label": "Compatcraft",
        "repository_source": "Example Org/Compatcraft",
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


def test_definition_round_trips_model_compat():
    block = {
        "default_for_formats": ["safetensors"],
        "prepared_bundle_formats": ["mixed"],
    }
    definition = validate_definition(_raw(model_compat=block), source="c.yaml")
    assert dict(definition.model_compat) == block
    assert dict(definition.to_spec().model_compat) == block


def test_malformed_model_compat_is_rejected():
    with pytest.raises(DefinitionError, match="unknown keys"):
        validate_definition(_raw(model_compat={"nope": []}), source="c.yaml")
    with pytest.raises(DefinitionError, match="non-empty list"):
        validate_definition(_raw(model_compat={"default_for_formats": []}), source="c.yaml")
    with pytest.raises(DefinitionError, match="duplicates"):
        validate_definition(_raw(model_compat={"default_for_formats": ["a", "a"]}), source="c.yaml")
    with pytest.raises(DefinitionError, match="schema_version 2"):
        validate_definition(
            _raw(schema_version=1, model_compat={"default_for_formats": ["a"]}),
            source="c.yaml",
        )
