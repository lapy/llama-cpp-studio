"""Capability registry and model-schema contracts."""

from backend.engine_registry import (
    ENGINE_REGISTRY,
    active_engine_row_is_runnable,
    engine_registry_payload,
)
from backend.model_schema import normalize_model_record


def test_audio_engine_descriptor_is_capability_driven():
    spec = ENGINE_REGISTRY["audio_cpp"]
    assert spec.runtime_kind == "audio_cpp"
    assert spec.scanner_kind == "audio_cpp"
    assert spec.active_path_fields == ("server_binary_path", "cli_binary_path")
    assert "prepared_bundle" in spec.package_kinds
    assert {"tts", "asr", "vad"} <= spec.tasks
    descriptor = next(
        row
        for row in engine_registry_payload()["engines"]
        if row["id"] == "audio_cpp"
    )
    assert descriptor["experimental"] is False
    assert descriptor["maturity_surfaces"]["speech_asr"] == "stable"
    assert descriptor["maturity_surfaces"]["generic_tasks"] == "stable"
    assert descriptor["maturity_surfaces"]["heuristic_discovery"] == "experimental"
    assert descriptor["active_path_fields"] == [
        "server_binary_path",
        "cli_binary_path",
    ]


def test_audio_engine_runnable_requires_both_binaries():
    assert not active_engine_row_is_runnable(
        "audio_cpp", {"server_binary_path": "/server"}
    )
    assert active_engine_row_is_runnable(
        "audio_cpp",
        {
            "server_binary_path": "/server",
            "cli_binary_path": "/cli",
        },
    )


def test_audio_compatibility_is_never_inferred_from_safetensors_extension():
    generic = normalize_model_record(
        {"id": "generic", "format": "safetensors", "family": "audio-looking"}
    )
    verified = normalize_model_record(
        {
            "id": "verified",
            "format": "mixed",
            "family": "pocket_tts",
            "task": "tts",
            "local_path": "/models/pocket-tts",
            "compatible_engines": ["audio_cpp"],
        }
    )
    assert "audio_cpp" not in generic["compatible_engines"]
    assert verified["artifact"]["package_kind"] == "prepared_bundle"
    assert verified["compatible_engines"] == ["audio_cpp"]
    assert verified["input_modalities"] == ["text", "audio"]
    assert verified["output_modalities"] == ["audio"]

