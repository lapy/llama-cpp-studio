"""Definition descriptors and kind vocabularies stay consistent."""

from __future__ import annotations

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import (
    INSTALL_KINDS,
    RUNTIME_KINDS,
    SCANNER_KINDS,
)
from backend.engines.registry import (
    ENGINE_REGISTRY,
    engine_registry_payload,
)


@pytest.mark.parametrize("engine_id", sorted(ENGINE_REGISTRY))
def test_definition_descriptor_equals_registry_spec(engine_id):
    definition = load_default_engine_definitions().get(engine_id)
    assert definition is not None, engine_id
    assert definition.to_api_dict() == ENGINE_REGISTRY[engine_id].to_api_dict()


@pytest.mark.parametrize("engine_id", sorted(ENGINE_REGISTRY))
def test_definition_reconstructs_engine_spec_fields(engine_id):
    definition = load_default_engine_definitions().get(engine_id)
    spec = definition.to_spec()
    assert spec == ENGINE_REGISTRY[engine_id]


def test_default_registry_is_cached_singleton():
    assert load_default_engine_definitions() is load_default_engine_definitions()


def test_schema_vocabularies_cover_registry_kinds():
    assert {spec.install_kind for spec in ENGINE_REGISTRY.values()} <= INSTALL_KINDS
    assert {spec.runtime_kind for spec in ENGINE_REGISTRY.values()} <= RUNTIME_KINDS
    assert {spec.scanner_kind for spec in ENGINE_REGISTRY.values()} <= SCANNER_KINDS


def test_scanner_vocabulary_matches_scanner_dispatch():
    # The loader vocabulary, the legacy adapters and the definition scanners
    # must never diverge. Legacy adapters are optional now that shipped engines
    # can declare discovery; every kind WITHOUT a legacy adapter must have a
    # parse-only counterpart the definition scanner can run.
    from backend.engines.scan.scanner import (
        DEFINITION_HELP_PARSERS,
        SCANNER_KIND_DISPATCH,
    )

    assert set(SCANNER_KIND_DISPATCH) <= set(SCANNER_KINDS)
    assert set(SCANNER_KINDS) == set(SCANNER_KIND_DISPATCH) | set(DEFINITION_HELP_PARSERS)


def test_python_installer_adapters_are_within_install_vocabulary():
    from backend.engines.adapters import _INSTALLERS

    # All adapter-backed engines are install_kind python_venv; cmake/prebuilt
    # engines are driven by their native routes, never this map.
    for engine_id in _INSTALLERS:
        assert ENGINE_REGISTRY[engine_id].install_kind == "python_venv"


def test_registry_api_payload_shape_unchanged():
    payload = engine_registry_payload()
    assert payload["schema_version"] == 1
    assert {engine["id"] for engine in payload["engines"]} == set(ENGINE_REGISTRY)


def test_startup_verification_passes_and_detects_drift(monkeypatch):
    from dataclasses import replace

    from backend.engines.definitions import startup_check

    assert startup_check.verify_engine_definitions() == len(ENGINE_REGISTRY)

    registry = load_default_engine_definitions()

    # Drift in any engine descriptor must refuse startup.
    drifted = dict(registry.definitions)
    drifted["lmdeploy"] = replace(drifted["lmdeploy"], label="Drifted")
    monkeypatch.setattr(
        startup_check,
        "load_default_engine_definitions",
        lambda: type(registry)(definitions=drifted),
    )
    with pytest.raises(RuntimeError, match="refusing startup"):
        startup_check.verify_engine_definitions()

    # A definition file missing an engine must also refuse startup.
    missing = {k: v for k, v in registry.definitions.items() if k != "audio_cpp"}
    monkeypatch.setattr(
        startup_check,
        "load_default_engine_definitions",
        lambda: type(registry)(definitions=missing),
    )
    with pytest.raises(RuntimeError, match="missing=\\['audio_cpp'\\]"):
        startup_check.verify_engine_definitions()
