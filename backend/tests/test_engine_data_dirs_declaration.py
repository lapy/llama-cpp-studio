"""R75 pins: startup data directories are engine data, not a hardcoded list.

`backend/main.py::ensure_data_directories` used to hardcode every engine's data
folder (`llama-cpp`, `audio-cpp`, `models/audio-cpp`, `1cat-vllm`, ...). The
folders now come from `EngineSpec.data_dirs` (schema-v2 `data_dirs`), exported
into the definitions, so a YAML-onboarded engine gets its folders too.
"""

from __future__ import annotations

import os

import pytest

from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import ENGINE_REGISTRY, EngineSpec, get_engine_spec

EXPECTED_DIRS = {
    "llama_cpp": ("llama-cpp",),
    "ik_llama": ("llama-cpp",),
    "unsloth_llama": ("unsloth-llama",),
    "audio_cpp": ("audio-cpp", "models/audio-cpp"),
    "lmdeploy": ("lmdeploy",),
    "1cat_vllm": ("1cat-vllm",),
    "sglang": ("sglang",),
    "sglang_v100": ("sglang-v100",),
    "vllm": ("vllm",),
}
BASE_DIRS = {"config", "logs", "temp"}


def test_generated_engines_declare_their_historical_directories():
    for engine_id, dirs in EXPECTED_DIRS.items():
        spec = get_engine_spec(engine_id)
        assert spec is not None, engine_id
        assert tuple(spec.data_dirs) == dirs, engine_id
        assert "data_dirs" not in spec.to_api_dict(), engine_id


def test_directories_are_exported_with_the_definition():
    definitions = load_default_engine_definitions()
    for engine_id, dirs in EXPECTED_DIRS.items():
        definition = definitions.get(engine_id)
        assert definition is not None, engine_id
        assert tuple(definition.data_dirs) == dirs, engine_id


def test_startup_creates_declared_directories(tmp_path, monkeypatch):
    from backend import main as main_module

    monkeypatch.setattr("backend.data_store.studio_data_dir", lambda: str(tmp_path))
    main_module.ensure_data_directories()

    for subdir in BASE_DIRS | {d for dirs in EXPECTED_DIRS.values() for d in dirs}:
        assert (tmp_path / subdir).is_dir(), subdir
    assert (tmp_path / "models" / "audio-cpp").is_dir()


def test_a_declared_engine_gets_its_directories(tmp_path, monkeypatch):
    """Proves the list is spec-driven: a new declared dir shows up next run."""
    from backend import main as main_module

    fake = EngineSpec(
        id="fixture_dirs",
        label="fixture_dirs",
        repository_source="fixture_dirs",
        install_kind="cmake",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"gguf"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        data_dirs=("fixture-dirs", "models/fixture"),
    )
    monkeypatch.setitem(ENGINE_REGISTRY, "fixture_dirs", fake)
    monkeypatch.setattr("backend.data_store.studio_data_dir", lambda: str(tmp_path))
    main_module.ensure_data_directories()
    assert (tmp_path / "fixture-dirs").is_dir()
    assert (tmp_path / "models" / "fixture").is_dir()


def _raw(**extra):
    raw = {
        "schema_version": 2,
        "id": "dircraft",
        "label": "Dircraft",
        "repository_source": "Example Org/Dircraft",
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


def test_definition_round_trips_data_dirs():
    definition = validate_definition(
        _raw(data_dirs=["dir-craft", "models/dir-craft"]), source="dircraft.yaml"
    )
    assert tuple(definition.data_dirs) == ("dir-craft", "models/dir-craft")
    assert tuple(definition.to_spec().data_dirs) == ("dir-craft", "models/dir-craft")


def test_malformed_data_dirs_are_rejected():
    with pytest.raises(DefinitionError, match="relative path"):
        validate_definition(_raw(data_dirs=["/abs"]), source="d.yaml")
    with pytest.raises(DefinitionError, match="relative path"):
        validate_definition(_raw(data_dirs=["../escape"]), source="d.yaml")
    with pytest.raises(DefinitionError, match="duplicate"):
        validate_definition(_raw(data_dirs=["a", "a"]), source="d.yaml")
    with pytest.raises(DefinitionError, match="schema_version 2"):
        validate_definition(_raw(schema_version=1, data_dirs=["a"]), source="d.yaml")
