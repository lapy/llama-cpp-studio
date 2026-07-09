"""Normalized model catalog providers and filters."""

from backend.model_catalog.audio_cpp_provider import (
    AudioCppCatalogProvider,
    parse_model_manager_catalog,
)
from backend.model_catalog.base import item_matches_filters


def test_parse_audio_model_manager_catalog_without_execution():
    source = """
from dataclasses import dataclass
CATALOG = (
    ModelPackage(
        id="demo_tts",
        display_name="Demo TTS",
        target_directory="demo",
        source=SnapshotSource(repo_id="org/demo", revision="abc"),
        required_files=("config.json", "model.safetensors"),
        description="Demo package",
    ),
    ModelPackage(
        id="converted",
        display_name="Converted",
        target_directory="converted",
        source=ConverterSource(kind="nemo", description="Convert NeMo"),
        required_files=(),
    ),
)
"""
    packages = parse_model_manager_catalog(source)
    assert [package["id"] for package in packages] == ["demo_tts", "converted"]
    assert packages[0]["source"]["kind"] == "huggingface_snapshot"
    assert packages[0]["source"]["repo_id"] == "org/demo"
    assert packages[1]["source"]["kind"] == "composite"


def test_audio_catalog_requires_loader_scan_for_verified_compatibility(monkeypatch):
    class Store:
        def get_active_engine_version(self, engine):
            assert engine == "audio_cpp"
            return {"version": "v1", "source_commit": "abc"}

    provider = AudioCppCatalogProvider(Store())
    monkeypatch.setattr(
        "backend.model_catalog.audio_cpp_provider.get_version_entry",
        lambda *args: {
            "capabilities": {"families": ["qwen3_tts"]},
        },
    )
    items = provider._normalize_packages(
        [
            {
                "id": "qwen3_tts_0_6b_base",
                "display_name": "Qwen TTS",
                "installable": True,
                "source": {"kind": "huggingface_snapshot", "repo_id": "Qwen/test"},
            },
            {
                "id": "unknown",
                "display_name": "Unknown",
                "installable": True,
                "source": {"kind": "huggingface_snapshot", "repo_id": "org/unknown"},
            },
        ],
        {"version": "v1", "source_commit": "abc"},
    )
    assert items[0]["compatible_engines"] == ["audio_cpp"]
    assert items[1]["compatible_engines"] == []
    assert items[1]["unavailable_reason"]
    assert item_matches_filters(items[0], {"engine": "audio_cpp", "task": "tts"})
    assert not item_matches_filters(items[1], {"engine": "audio_cpp"})

