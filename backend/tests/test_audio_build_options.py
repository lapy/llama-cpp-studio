"""Tests for the audio.cpp CMake build-option catalog."""

import pytest

from backend.engines.audio_cpp import build_options as bo
from backend.engines.audio_cpp.build_options import (
    catalog_for_ui,
    coerce_build_settings,
    default_build_settings,
)
from backend.engines.audio_cpp.manager import AudioCppBuildConfig, AudioCppManager
from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import get_engine_spec


def test_catalog_exposes_backends_and_iqk_style_sections():
    cat = catalog_for_ui()
    ids = {c["id"] for c in cat["categories"]}
    assert "backends" in ids
    assert "models" in ids
    keys = {o["key"] for c in cat["categories"] for o in c["options"]}
    for expected in (
        "cuda",
        "llamafile",
        "deployment_build",
        "native_model_manager",
        "use_system_openssl",
        "build_server_frontends",
        "build_model_tests",
        "model_set",
        "static_espeak",
        "build_c_api",
        "build_extended_tests",
    ):
        assert expected in keys
    for dropped in ("hip", "vulkan", "metal", "hip_strix_halo"):
        assert dropped not in keys
    assert cat["defaults"]["native_model_manager"] is False


def test_backend_is_derived_from_cuda_toggle():
    settings = coerce_build_settings({"cuda": True, "native_cpu": False})
    assert settings["cuda"] is True
    assert settings["backend"] == "cuda"
    assert settings["native_cpu"] is False

    cpu = coerce_build_settings({"backend": "metal"})
    assert cpu["cuda"] is False
    assert cpu["backend"] == "cpu"


def test_defaults_cover_catalog():
    defaults = default_build_settings()
    assert defaults["build_type"] == "RelWithDebInfo"
    assert "llamafile" in defaults


def test_build_options_api(client):
    r = client.get("/api/audio-cpp/build-options")
    # Feature flag may 404 in some envs; accept catalog or gated 404
    if r.status_code == 404:
        return
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "audio_cpp"
    assert any(c["id"] == "backends" for c in body["categories"])


def test_cuda_cmake_forces_unsupported_backends_off(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.build_progress.shutil.which",
        lambda name: "/usr/bin/ninja" if name == "ninja" else None,
    )
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    config = AudioCppBuildConfig(cuda=True).normalized()
    args = manager._cmake_args("/s", "/b", config)
    assert "-DENGINE_ENABLE_CUDA=ON" in args
    assert "-DENGINE_ENABLE_HIP=OFF" in args
    assert "-DENGINE_ENABLE_VULKAN=OFF" in args
    assert "-DENGINE_ENABLE_METAL=OFF" in args
    assert "-DENGINE_HIP_STRIX_HALO_OPTIMIZATIONS=OFF" in args
    assert "-DAUDIOCPP_STATIC_ESPEAK=OFF" in args
    assert "-DAUDIOCPP_BUILD_C_API=OFF" in args
    assert "-DENGINE_BUILD_EXTENDED_TESTS=OFF" in args
    assert "-DENGINE_BUILD_MODEL_TESTS=OFF" in args
    assert "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=OFF" in args
    assert config.backend == "cuda"


# --- R68: the catalog is engine data (spec + generated definition) ---------------


def test_catalog_is_exported_with_the_definition():
    spec = get_engine_spec("audio_cpp")
    definition = load_default_engine_definitions().get("audio_cpp")
    assert spec is not None and definition is not None

    spec_catalog = dict(spec.build_options)
    assert spec_catalog["build_type_default"] == "RelWithDebInfo"
    assert len(spec_catalog["options"]) == 23
    assert [c["id"] for c in spec_catalog["categories"]] == [
        "backends",
        "cuda",
        "artifacts",
        "cpu",
        "models",
        "advanced",
    ]
    # The generated definition carries the same catalog, byte-for-byte.
    assert dict(definition.build_options) == spec_catalog
    # And it never reaches the descriptor payload.
    assert "build_options" not in spec.to_api_dict()


def test_catalog_module_derives_from_the_spec(monkeypatch):
    """The module must not keep its own table: render from whatever the spec says."""
    fake = {
        "build_types": ["Fast", "Slow"],
        "build_type_default": "Slow",
        "primary_keys": ["turbo"],
        "categories": [
            {"id": "backends", "label": "Backends", "collapsed": False},
            {"id": "extra", "label": "Extra", "collapsed": True, "requires": "turbo"},
        ],
        "options": [
            {
                "key": "turbo",
                "field": "turbo",
                "kind": "bool",
                "default": False,
                "label": "Turbo",
                "desc": "ENGINE_TURBO",
                "category": "backends",
                "cmake": "ENGINE_TURBO",
                "special": "backend",
            },
            {
                "key": "mode",
                "field": "mode",
                "kind": "enum",
                "default": "a",
                "label": "Mode",
                "desc": "ENGINE_MODE",
                "category": "extra",
                "enum_values": ["a", "b"],
                "requires": "turbo",
            },
        ],
    }

    class FakeSpec:
        build_options = fake

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())

    defaults = bo.default_build_settings()
    assert defaults["build_type"] == "Slow"
    assert defaults["turbo"] is False
    assert defaults["mode"] == "a"
    settings = bo.coerce_build_settings({"turbo": True, "mode": "b"})
    assert settings["backend"] == "cuda"
    assert settings["mode"] == "b"
    assert bo.catalog_for_ui()["build_types"] == ["Fast", "Slow"]


def test_definition_validates_a_catalog_round_trip():
    raw = {
        "schema_version": 2,
        "id": "buildcraft",
        "label": "Buildcraft",
        "repository_source": "Example Org/Buildcraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary_path"],
        "artifact_formats": ["builtin"],
        "package_kinds": ["builtin"],
        "tasks": ["tts"],
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "build_options": {
            "build_types": ["Release"],
            "build_type_default": "Release",
            "primary_keys": [],
            "categories": [{"id": "main", "label": "Main", "collapsed": False}],
            "options": [
                {
                    "key": "opt",
                    "field": "opt",
                    "kind": "int",
                    "default": 2,
                    "label": "Option",
                    "desc": "ENGINE_OPT",
                    "category": "main",
                }
            ],
        },
    }
    definition = validate_definition(raw, source="buildcraft.yaml")
    assert dict(definition.build_options) == raw["build_options"]
    assert dict(definition.to_spec().build_options) == raw["build_options"]


def test_definition_rejects_malformed_catalogs():
    base = {
        "schema_version": 2,
        "id": "buildcraft",
        "label": "Buildcraft",
        "repository_source": "Example Org/Buildcraft",
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
    good = {
        "build_types": ["Release"],
        "build_type_default": "Release",
        "primary_keys": [],
        "categories": [{"id": "main", "label": "Main"}],
        "options": [
            {
                "key": "opt",
                "field": "opt",
                "kind": "bool",
                "default": False,
                "label": "Option",
                "desc": "ENGINE_OPT",
                "category": "main",
            }
        ],
    }

    def _bad(mutate):
        block = {**good}
        mutate(block)
        with pytest.raises(DefinitionError):
            validate_definition({**base, "build_options": block}, source="buildcraft.yaml")

    _bad(lambda b: b.update({"nope": 1}))
    _bad(lambda b: b.update({"build_type_default": "Debug"}))
    _bad(
        lambda b: b["options"][0].update({"category": "missing"}),
    )
    _bad(lambda b: b["options"][0].update({"default": "no"}))
    _bad(lambda b: b["options"][0].update({"requires": "unknown_parent"}))
    _bad(lambda b: b["options"][0].update({"kind": "enum"}))
    _bad(
        lambda b: b["options"].append(dict(b["options"][0])),
    )
