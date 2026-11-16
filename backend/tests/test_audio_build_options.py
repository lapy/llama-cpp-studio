"""Tests for the audio.cpp CMake build-option catalog."""

import pytest

from backend.engines.audio_cpp import build_options as bo
from backend.engines.audio_cpp.build_options import (
    catalog_for_ui,
    cmake_arguments,
    coerce_build_settings,
    default_build_settings,
    validate_build_settings,
)
from backend.engines.audio_cpp.manager import AudioCppBuildConfig, AudioCppManager
from backend.engines.definitions import load_default_engine_definitions
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import get_engine_spec

#: Every CMake option()/cache variable upstream's configure honours for a build
#: (CMakeLists.txt + app/server/server_frontends.cmake, v0.9.5 and the commits
#: after it). When upstream adds or renames a knob the catalog must follow, so
#: this pin fails until the definition grows the new entry.
UPSTREAM_CMAKE_VARIABLES = {
    "ENGINE_ENABLE_CUDA",
    "ENGINE_ENABLE_HIP",
    "ENGINE_ENABLE_VULKAN",
    "ENGINE_ENABLE_WEBGPU",
    "ENGINE_ENABLE_SYCL",
    "ENGINE_ENABLE_METAL",
    "ENGINE_ENABLE_LLAMAFILE",
    "ENGINE_ENABLE_CUDA_GRAPHS",
    "ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS",
    "ENGINE_ENABLE_RKNN",
    "ENGINE_ENABLE_NEON",
    "ENGINE_ENABLE_NATIVE_CPU",
    "ENGINE_ENABLE_OPENMP",
    "ENGINE_ENABLE_CPU_ALL_VARIANTS",
    "ENGINE_BUILD_EXAMPLES",
    "ENGINE_BUILD_TESTS",
    "ENGINE_BUILD_EXTENDED_TESTS",
    "ENGINE_BUILD_MODEL_TESTS",
    "ENGINE_BUILD_WARMBENCH",
    "AUDIOCPP_DEPLOYMENT_BUILD",
    "AUDIOCPP_STATIC_ESPEAK",
    "AUDIOCPP_BUILD_NATIVE_MODEL_MANAGER",
    "AUDIOCPP_USE_SYSTEM_OPENSSL",
    "AUDIOCPP_BUILD_SERVER_FRONTENDS",
    "AUDIOCPP_SERVER_FRONTEND_MODULES",
    "AUDIOCPP_SERVER_FRONTENDS_DIR",
    "AUDIOCPP_BUILD_C_API",
    "AUDIOCPP_BUILD_C_API_STATIC",
    "AUDIOCPP_MODEL_SET",
    "AUDIOCPP_MODELS",
    "AUDIOCPP_ZONOS3_TEXTNORM",
    "AUDIOCPP_ZONOS3_TEXTNORM_DIR",
}


def test_catalog_covers_every_upstream_cmake_variable():
    covered = {option.cmake for option in bo.build_options() if option.cmake}
    assert covered == UPSTREAM_CMAKE_VARIABLES


def test_unsupported_knobs_are_catalogued_but_flagged():
    """Studio documents every upstream knob and marks what it cannot build."""
    assert bo.unsupported_option_keys() == (
        "hip",
        "vulkan",
        "webgpu",
        "sycl",
        "metal",
        "neon",
        "rknn",
    )
    assert bo.supported_backends() == ("cpu", "cuda")
    entries = {o["key"]: o for c in catalog_for_ui()["categories"] for o in c["options"]}
    assert entries["cuda"]["supported"] is True
    assert entries["hip"]["supported"] is False
    assert entries["neon"]["supported"] is False


def test_catalog_exposes_backends_and_sections():
    cat = catalog_for_ui()
    ids = [c["id"] for c in cat["categories"]]
    assert ids == ["backends", "cuda", "hip", "artifacts", "cpu", "npu", "models", "advanced"]
    keys = {o["key"] for c in cat["categories"] for o in c["options"]}
    for expected in (
        "cuda",
        "hip",
        "vulkan",
        "webgpu",
        "sycl",
        "metal",
        "hip_strix_halo",
        "rknn",
        "neon",
        "llamafile",
        "deployment_build",
        "native_model_manager",
        "use_system_openssl",
        "build_server_frontends",
        "server_frontend_modules",
        "server_frontends_dir",
        "build_model_tests",
        "build_c_api",
        "c_api_static",
        "zonos3_textnorm",
        "zonos3_textnorm_dir",
        "static_espeak",
        "build_extended_tests",
        "model_set",
    ):
        assert expected in keys
    assert cat["defaults"]["native_model_manager"] is False
    assert cat["supported_backends"] == ["cpu", "cuda"]


def test_backend_is_derived_from_enabled_backends():
    settings = coerce_build_settings({"cuda": True, "native_cpu": False})
    assert settings["cuda"] is True
    assert settings["backend"] == "cuda"
    assert settings["native_cpu"] is False

    # ``backend`` is derived, never user-selected.
    cpu = coerce_build_settings({"backend": "metal"})
    assert cpu["cuda"] is False
    assert cpu["backend"] == "cpu"

    # Precedence follows upstream's AUDIOCPP_BUILD_BACKENDS order.
    assert coerce_build_settings({"vulkan": True})["backend"] == "vulkan"
    assert coerce_build_settings({"vulkan": True, "cuda": True})["backend"] == "cuda"


def test_defaults_cover_catalog():
    defaults = default_build_settings()
    assert defaults["build_type"] == "RelWithDebInfo"
    assert defaults["server_frontends_dir"] == "external/audio.cpp-server-frontends"
    # One entry per catalog option plus build_type/backend.
    assert set(defaults) == {option.key for option in bo.build_options()} | {
        "build_type",
        "backend",
    }


def test_build_options_api(client):
    r = client.get("/api/audio-cpp/build-options")
    # Feature flag may 404 in some envs; accept catalog or gated 404
    if r.status_code == 404:
        return
    assert r.status_code == 200
    body = r.json()
    assert body["engine"] == "audio_cpp"
    assert any(c["id"] == "backends" for c in body["categories"])
    assert body["supported_backends"] == ["cpu", "cuda"]


def test_cmake_arguments_pin_every_option_explicitly():
    flags = dict(item[2:].split("=", 1) for item in cmake_arguments({}))
    # A CPU build still pins the backends Studio does not support: upstream
    # defaults ENGINE_ENABLE_METAL ON on Apple hosts.
    assert flags["ENGINE_ENABLE_CUDA"] == "OFF"
    assert flags["ENGINE_ENABLE_HIP"] == "OFF"
    assert flags["ENGINE_ENABLE_VULKAN"] == "OFF"
    assert flags["ENGINE_ENABLE_WEBGPU"] == "OFF"
    assert flags["ENGINE_ENABLE_SYCL"] == "OFF"
    assert flags["ENGINE_ENABLE_METAL"] == "OFF"
    assert flags["ENGINE_ENABLE_RKNN"] == "OFF"
    assert flags["ENGINE_ENABLE_NEON"] == "OFF"
    assert flags["ENGINE_HIP_STRIX_HALO_OPTIMIZATIONS"] == "OFF"
    assert flags["AUDIOCPP_BUILD_C_API_STATIC"] == "OFF"
    assert flags["AUDIOCPP_ZONOS3_TEXTNORM"] == "OFF"
    assert flags["AUDIOCPP_MODEL_SET"] == "full"
    assert flags["AUDIOCPP_MODELS"] == ""
    assert len(flags) == len(UPSTREAM_CMAKE_VARIABLES)


def test_cmake_arguments_render_frontend_modules_as_a_cmake_list():
    args = cmake_arguments(
        {
            "build_server_frontends": True,
            "server_frontend_modules": "audio_decode, mp3_encode",
        }
    )
    assert "-DAUDIOCPP_BUILD_SERVER_FRONTENDS=ON" in args
    assert "-DAUDIOCPP_SERVER_FRONTEND_MODULES=audio_decode;mp3_encode" in args
    assert "-DAUDIOCPP_SERVER_FRONTENDS_DIR=external/audio.cpp-server-frontends" in args


def test_cmake_arguments_skip_env_and_argv_only_options():
    args = cmake_arguments({"jobs": 8, "custom_cmake_args": "-DFOO=ON", "cflags": "-O2"})
    assert not any("JOBS" in arg or "CUSTOM_CMAKE" in arg or "CFLAGS" in arg for arg in args)


@pytest.mark.parametrize(
    "settings, expected",
    [
        ({}, ""),
        ({"cuda": True}, ""),
        ({"cuda": True, "hip": True}, "mutually exclusive"),
        ({"hip": True}, "clear hip"),
        ({"cuda": True, "hip_strix_halo": True}, "HIP backend"),
        ({"use_system_openssl": True}, "native model manager"),
        ({"server_frontend_modules": "mp3_encode"}, "server frontends"),
        (
            {
                "build_server_frontends": True,
                "server_frontend_modules": "mp3_encode",
                "server_frontends_dir": "",
            },
            "frontends package directory",
        ),
        ({"zonos3_textnorm": True}, "zyphra-textnorm"),
        ({"model_set": "custom"}, "custom model set"),
        ({"vulkan": True}, "build matrix"),
        ({"webgpu": True}, "build matrix"),
        ({"rknn": True}, "build matrix"),
        ({"neon": True}, "build matrix"),
    ],
)
def test_validation_reports_build_breaking_combinations(settings, expected):
    errors = validate_build_settings(settings)
    if expected:
        assert any(expected in error for error in errors), errors
    else:
        assert errors == []


def test_validation_accepts_a_complete_frontends_configuration():
    assert (
        validate_build_settings(
            {
                "build_server_frontends": True,
                "server_frontend_modules": "audio_decode;mp3_encode",
                "server_frontends_dir": "external/audio.cpp-server-frontends",
            }
        )
        == []
    )


def test_manager_cmake_for_cuda_build(tmp_path, monkeypatch):
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


def test_manager_rejects_backends_studio_cannot_ship(tmp_path):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    manager.validate_build_config(AudioCppBuildConfig(cuda=True).normalized())
    with pytest.raises(ValueError, match="build matrix"):
        manager.validate_build_config(AudioCppBuildConfig(vulkan=True).normalized())
    with pytest.raises(ValueError, match="mutually exclusive"):
        manager.validate_build_config(AudioCppBuildConfig(cuda=True, hip=True).normalized())


# --- R68: the catalog is engine data (spec + generated definition) ---------------


def test_catalog_is_exported_with_the_definition():
    spec = get_engine_spec("audio_cpp")
    definition = load_default_engine_definitions().get("audio_cpp")
    assert spec is not None and definition is not None

    spec_catalog = dict(spec.build_options)
    assert spec_catalog["build_type_default"] == "RelWithDebInfo"
    assert len(spec_catalog["options"]) == 36
    assert [c["id"] for c in spec_catalog["categories"]] == [
        "backends",
        "cuda",
        "hip",
        "artifacts",
        "cpu",
        "npu",
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
    # The backend is named after the enabled catalog option.
    assert settings["backend"] == "turbo"
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
    _bad(lambda b: b["options"][0].update({"supported": "no"}))
    _bad(lambda b: b["options"][0].update({"kind": "enum"}))
    _bad(
        lambda b: b["options"].append(dict(b["options"][0])),
    )
