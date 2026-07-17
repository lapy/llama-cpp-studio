"""Capability registry and lossless model-schema migration contracts."""

from backend.engine_registry import (
    ENGINE_REGISTRY,
    active_engine_row_is_runnable,
    engine_registry_payload,
)
from backend.model_schema import migrate_models_document, normalize_model_record


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
    assert descriptor["maturity_surfaces"]["generic_tasks"] == "limited"
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


def test_legacy_model_migration_is_idempotent_and_preserves_fields():
    legacy = {
        "models": [
            {
                "id": "org--demo--Q4_K_M",
                "huggingface_id": "org/demo",
                "format": "gguf",
                "quantization": "Q4_K_M",
                "custom_legacy_field": {"keep": True},
                "pipeline_tag": "text-generation",
            }
        ],
        "custom_root": "keep",
    }
    migrated, changed = migrate_models_document(legacy)
    again, changed_again = migrate_models_document(migrated)
    model = migrated["models"][0]

    assert changed is True
    assert changed_again is False
    assert again == migrated
    assert migrated["custom_root"] == "keep"
    assert model["custom_legacy_field"] == {"keep": True}
    assert model["source"] == {"provider": "huggingface", "id": "org/demo"}
    assert model["artifact"]["format"] == "gguf"
    assert model["compatible_engines"] == ["llama_cpp", "ik_llama"]


def test_audio_compatibility_is_never_inferred_from_safetensors_extension():
    generic = normalize_model_record(
        {"id": "generic", "format": "safetensors", "family": "audio-looking"}
    )
    verified = normalize_model_record(
        {
            "id": "verified",
            "format": "mixed",
            "family": "kokoro",
            "task": "tts",
            "local_path": "/models/kokoro",
            "compatible_engines": ["audio_cpp"],
        }
    )
    assert "audio_cpp" not in generic["compatible_engines"]
    assert verified["artifact"]["package_kind"] == "prepared_bundle"
    assert verified["compatible_engines"] == ["audio_cpp"]
    assert verified["input_modalities"] == ["text", "audio"]
    assert verified["output_modalities"] == ["audio"]

