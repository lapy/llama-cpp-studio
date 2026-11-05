"""R69 pins: audio's protocol contract and extension surfaces are engine data.

The inspect-probe flags and the artifact path shape used to be constants in
`inspect.py`/`artifact.py`; the typed extension endpoints used to be implicit in
the router. Both are now declared on the engine spec, exported into the
definition, and read from there — so the surface is explicit and gate-checkable.
"""

from __future__ import annotations

import pytest

from backend.engines.audio_cpp import artifact as artifact_module
from backend.engines.audio_cpp import inspect as inspect_module
from backend.engines.definitions.schema import DefinitionError, validate_definition
from backend.engines.registry import get_engine_spec

EXPECTED_EXTENSIONS = [
    {"id": "contract", "path": "/contract", "methods": ["GET"]},
    {"id": "status", "path": "/status", "methods": ["GET"]},
    {"id": "build_options", "path": "/build-options", "methods": ["GET"]},
    {"id": "build_settings", "path": "/build-settings", "methods": ["GET", "PUT"]},
    {"id": "cancel", "path": "/cancel", "methods": ["POST"]},
    {"id": "community_voices", "path": "/community-voices", "methods": ["GET", "DELETE"]},
    {
        "id": "community_voices_install",
        "path": "/community-voices/install",
        "methods": ["POST"],
    },
]


def test_protocol_is_engine_data_and_exported_with_the_definition():
    from backend.engines.definitions import load_default_engine_definitions

    spec = get_engine_spec("audio_cpp")
    definition = load_default_engine_definitions().get("audio_cpp")
    assert spec is not None and definition is not None
    protocol = dict(spec.protocol)
    assert protocol["probe"]["model_flag"] == "--model"
    assert protocol["probe"]["inspect_flag"] == "--inspect"
    assert protocol["probe"]["json_flag"] == "--json"
    assert protocol["launch"] == {
        "config_flag": "--config",
        "config_macro": "studio_audio_config",
    }
    assert protocol["artifact"] == {
        "builtin_family": "builtin_audio_utils",
        "root_gguf_name": "model.gguf",
        "gguf_suffix": ".gguf",
        "builtin_id_pattern": "[a-zA-Z0-9][a-zA-Z0-9_-]*",
    }
    assert dict(definition.protocol) == protocol
    assert "protocol" not in spec.to_api_dict()


def test_extensions_are_declared_and_actually_served():
    from backend.routes import audio_cpp_versions as routes

    spec = get_engine_spec("audio_cpp")
    declared = [dict(item) for item in spec.extensions]
    assert declared == EXPECTED_EXTENSIONS
    assert "extensions" not in spec.to_api_dict()
    # The router enforces the same declaration at import time; the shared
    # helper is the single implementation (no test-local copy of the check).
    from backend.engines.definitions.extensions import (
        assert_extensions_served,
        served_surfaces,
    )

    assert_extensions_served("audio_cpp", routes.router.routes)
    served = served_surfaces(routes.router.routes)
    for extension in declared:
        assert set(extension["methods"]) <= served[extension["path"]], extension


def test_extension_enforcement_rejects_a_drifting_router():
    from backend.engines.definitions.extensions import assert_extensions_served

    class _Route:
        def __init__(self, path, methods):
            self.path = path
            self.methods = methods

    complete = [
        _Route(extension["path"], set(extension["methods"])) for extension in EXPECTED_EXTENSIONS
    ]
    assert_extensions_served("audio_cpp", complete)
    with pytest.raises(RuntimeError, match="not served"):
        assert_extensions_served("audio_cpp", complete[:-1])
    wrong_method = complete[:-1] + [_Route("/community-voices/install", {"GET"})]
    with pytest.raises(RuntimeError, match="missing"):
        assert_extensions_served("audio_cpp", wrong_method)


def test_inspect_argv_reads_the_declared_flags(monkeypatch):
    class FakeSpec:
        protocol = {
            "probe": {
                "model_flag": "--m",
                "family_flag": "--f",
                "spec_override_flag": "--o",
                "load_option_flag": "--l",
                "inspect_flag": "--i",
                "json_flag": "--j",
            },
            "artifact": {},
        }

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())
    argv = inspect_module.build_audio_cpp_inspect_argv(
        "/cli",
        "/model",
        family="speech",
        model_spec_override="family.json",
        load_options={"device": "cpu", "empty": ""},
    )
    assert argv == [
        "/cli",
        "--m",
        "/model",
        "--f",
        "speech",
        "--o",
        "family.json",
        "--l",
        "device=cpu",
    ]
    json_argv, text_argv = inspect_module.inspect_command_variants(["/cli", "/model"])
    assert json_argv[-2:] == ["--i", "--j"]
    assert text_argv[-1] == "--i"


def test_artifact_contract_reads_the_declared_shape(monkeypatch):
    class FakeSpec:
        protocol = {
            "probe": {},
            "artifact": {
                "builtin_family": "other_family",
                "root_gguf_name": "weights.bin",
                "gguf_suffix": ".bin",
                "builtin_id_pattern": "[a-z]+",
            },
        }

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())
    assert artifact_module.artifact_contract()["root_gguf_name"] == "weights.bin"

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "weights.bin").write_text("x")
        (root / "model.gguf").write_text("x")
        single = root / "single.bin"
        single.write_text("x")

        assert artifact_module.audio_model_path_ready(str(root / "weights.bin")) is True
        assert artifact_module.audio_model_path_ready(str(root / "model.gguf")) is False
        # A package root hosting the declared root file wins over the single file.
        assert artifact_module.prefer_directory_model_path(str(single)) == str(root)
        descriptor = artifact_module.build_artifact_descriptor(
            bundle_path=str(root), runtime_path=str(single)
        )
        assert descriptor["gguf_path"] == str(root / "weights.bin")
        assert descriptor["layout"] == "directory"


def _raw(**extra):
    raw = {
        "schema_version": 2,
        "id": "protocolcraft",
        "label": "Protocolcraft",
        "repository_source": "Example Org/Protocolcraft",
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


def test_definition_round_trips_protocol_and_extensions():
    protocol = {
        "probe": {"model_flag": "--model", "inspect_flag": "--inspect"},
        "artifact": {"root_gguf_name": "model.gguf", "gguf_suffix": ".gguf"},
    }
    extensions = [
        {"id": "contract", "path": "/contract", "methods": ["GET"]},
        {"id": "voices", "path": "/voices", "methods": ["GET", "POST"], "description": "x"},
    ]
    definition = validate_definition(
        _raw(protocol=protocol, extensions=extensions), source="protocolcraft.yaml"
    )
    assert dict(definition.protocol) == protocol
    assert [dict(item) for item in definition.extensions] == extensions
    spec = definition.to_spec()
    assert dict(spec.protocol) == protocol
    assert [dict(item) for item in spec.extensions] == extensions
    assert "protocol" not in definition.to_api_dict()
    assert "extensions" not in definition.to_api_dict()


def test_malformed_protocol_and_extensions_are_rejected():
    with pytest.raises(DefinitionError, match="unknown keys"):
        validate_definition(_raw(protocol={"probe": {}, "nope": 1}), source="p.yaml")
    with pytest.raises(DefinitionError, match="must be a flag"):
        validate_definition(_raw(protocol={"probe": {"model_flag": "model"}}), source="p.yaml")
    with pytest.raises(DefinitionError, match="gguf_suffix"):
        validate_definition(
            _raw(protocol={"probe": {}, "artifact": {"gguf_suffix": "gguf"}}),
            source="p.yaml",
        )
    with pytest.raises(DefinitionError, match="not a regex"):
        validate_definition(
            _raw(protocol={"probe": {}, "artifact": {"builtin_id_pattern": "["}}),
            source="p.yaml",
        )
    with pytest.raises(DefinitionError, match="must start with '/'"):
        validate_definition(_raw(extensions=[{"id": "x", "path": "x"}]), source="p.yaml")
    with pytest.raises(DefinitionError, match="upper-case"):
        validate_definition(
            _raw(extensions=[{"id": "x", "path": "/x", "methods": ["get"]}]),
            source="p.yaml",
        )
    with pytest.raises(DefinitionError, match="duplicate extension path"):
        validate_definition(
            _raw(
                extensions=[
                    {"id": "a", "path": "/x", "methods": ["GET"]},
                    {"id": "b", "path": "/x", "methods": ["GET"]},
                ]
            ),
            source="p.yaml",
        )


def test_schema_v1_rejects_protocol_and_extensions():
    for extra in (
        {"protocol": {"probe": {"model_flag": "--model"}}},
        {"extensions": [{"id": "x", "path": "/x"}]},
        {"request_defaults": {"task": {"int_keys": ["seed"]}}},
    ):
        with pytest.raises(DefinitionError, match="schema_version 2"):
            validate_definition(_raw(schema_version=1, **extra), source="p.yaml")


# --- request-default vocabularies (S4: the task/request surface) ----------------


def test_request_default_vocabularies_are_engine_data():
    from backend.engines.definitions import load_default_engine_definitions

    spec = get_engine_spec("audio_cpp")
    block = dict(spec.request_defaults)
    assert set(block) == {"task", "transcription", "speech"}
    assert block["task"]["int_keys"] == [
        "seed",
        "max_tokens",
        "top_k",
        "num_inference_steps",
        "text_chunk_size",
    ]
    assert block["transcription"]["float_keys"] == ["temperature", "top_p"]
    assert block["transcription"]["bool_keys"] == ["stream"]
    assert "return_text" in block["speech"]["bool_keys"]
    assert "return_text" in block["speech"]["keys"]
    definition = load_default_engine_definitions().get("audio_cpp")
    assert dict(definition.request_defaults) == block
    assert "request_defaults" not in spec.to_api_dict()


def test_request_default_normalizers_read_the_vocabulary(monkeypatch):
    from backend.audio import request_defaults as request_module
    from backend.audio import transcription_defaults, voice_presets

    class FakeSpec:
        request_defaults = {
            "task": {"bool_keys": ["flag"], "int_keys": ["count"], "float_keys": ["scale"]},
            "transcription": {
                "bool_keys": ["return_text"],
                "int_keys": ["count"],
                "float_keys": ["scale"],
            },
            "speech": {
                "keys": ["flag", "count", "scale", "text"],
                "bool_keys": ["flag"],
                "int_keys": ["count"],
                "float_keys": ["scale"],
            },
        }

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())

    task = request_module.normalize_task_defaults(
        {"flag": 1, "count": "3", "scale": "0.5", "other": "x", "options": {"a": 1}}
    )
    assert task == {"flag": True, "count": 3, "scale": 0.5, "other": "x", "options": {"a": 1}}
    transcription = transcription_defaults.normalize_transcription_defaults(
        {"language": "en", "return_text": 1, "stream": True, "count": "4", "scale": 0.25}
    )
    # Declared boolean accepted; undeclared boolean dropped.
    assert transcription == {"language": "en", "return_text": True, "count": 4, "scale": 0.25}
    speech = voice_presets.normalize_speech_defaults(
        {"flag": 1, "count": "2", "scale": "1.5", "text": " hi ", "ignored": "x"}
    )
    assert speech == {"flag": True, "count": 2, "scale": 1.5, "text": "hi"}


def test_request_default_vocabulary_validation():
    good = {
        "task": {
            "bool_keys": ["stream"],
            "int_keys": ["seed"],
            "float_keys": ["temperature"],
        },
        "speech": {"keys": ["voice", "seed"], "int_keys": ["seed"]},
    }
    definition = validate_definition(_raw(request_defaults=good), source="p.yaml")
    assert dict(definition.request_defaults) == good

    with pytest.raises(DefinitionError, match="unknown sections"):
        validate_definition(_raw(request_defaults={"video": {}}), source="p.yaml")
    with pytest.raises(DefinitionError, match="unknown keys"):
        validate_definition(_raw(request_defaults={"task": {"nope": []}}), source="p.yaml")
    with pytest.raises(DefinitionError, match="more than one type list"):
        validate_definition(
            _raw(request_defaults={"task": {"int_keys": ["seed"], "float_keys": ["seed"]}}),
            source="p.yaml",
        )
    with pytest.raises(DefinitionError, match="duplicates"):
        validate_definition(
            _raw(request_defaults={"task": {"int_keys": ["seed", "seed"]}}),
            source="p.yaml",
        )


# --- R71: launch macro + catalog surface ---------------------------------------


def test_launch_macro_reads_the_declared_protocol():
    from backend.engines.audio_cpp import runtime as audio_runtime

    assert audio_runtime.launch_macro() == {
        "config_flag": "--config",
        "config_macro": "studio_audio_config",
    }

    class FakeSpec:
        protocol = {
            "probe": {},
            "artifact": {},
            "launch": {"config_flag": "--cfg", "config_macro": "audio_cfg"},
        }

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(
            "backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec()
        )
        assert audio_runtime.launch_macro() == {
            "config_flag": "--cfg",
            "config_macro": "audio_cfg",
        }
    finally:
        monkeypatch.undo()


def test_catalog_providers_are_engine_data():
    expected = {
        "llama_cpp": ["huggingface"],
        "ik_llama": ["huggingface"],
        "unsloth_llama": ["huggingface"],
        "lmdeploy": ["huggingface"],
        "vllm": ["huggingface"],
        "1cat_vllm": ["huggingface"],
        "sglang": ["huggingface"],
        "sglang_v100": ["huggingface"],
        "audio_cpp": ["audio_cpp"],
    }
    for engine_id, providers in expected.items():
        spec = get_engine_spec(engine_id)
        assert list(dict(spec.catalog).get("providers") or ()) == providers, engine_id
        assert "catalog" not in spec.to_api_dict(), engine_id

    from backend.engines.definitions import load_default_engine_definitions

    definition = load_default_engine_definitions().get("audio_cpp")
    assert dict(definition.catalog) == dict(get_engine_spec("audio_cpp").catalog)


def test_catalog_service_reads_the_declared_providers(monkeypatch):
    from backend.model_catalog.service import ModelCatalogService

    class FakeSpec:
        catalog = {"providers": ["audio_cpp"]}

    monkeypatch.setattr("backend.engines.registry.get_engine_spec", lambda engine_id: FakeSpec())
    assert ModelCatalogService._provider_ids({"engine": "audio_cpp"}) == ["audio_cpp"]
    # An explicit provider filter still wins; unknown engines keep both catalogs.
    assert ModelCatalogService._provider_ids(
        {"engine": "audio_cpp", "provider": "huggingface"}
    ) == ["huggingface"]
    assert ModelCatalogService._provider_ids({}) == ["audio_cpp", "huggingface"]
