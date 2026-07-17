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


def test_audio_catalog_marks_subcomponent_with_parent_hint(monkeypatch):
    class Store:
        def get_active_engine_version(self, engine):
            return {"version": "v1", "source_commit": "abc", "source_path": ""}

    provider = AudioCppCatalogProvider(Store())
    monkeypatch.setattr(
        "backend.model_catalog.audio_cpp_provider.get_version_entry",
        lambda *args: {
            "capabilities": {
                "families": ["moss_tts"],
                "family_tasks": {"moss_tts": ["tts"]},
            },
        },
    )
    items = provider._normalize_packages(
        [
            {
                "id": "moss_tts_nano_100m",
                "display_name": "MOSS Nano",
                "target_directory": "MOSS-TTS-Nano-100M",
                "description": "Framework-ready MOSS Nano",
                "installable": True,
                "source": {"kind": "composite_snapshot", "placements": []},
            },
            {
                "id": "moss_audio_tokenizer_nano",
                "display_name": "MOSS tokenizer",
                "target_directory": "MOSS-TTS-Nano-100M",
                "description": "Subcomponent only. Use moss_tts_nano_100m.",
                "installable": True,
                "source": {"kind": "huggingface_snapshot"},
            },
        ],
        {"version": "v1", "source_commit": "abc", "source_path": ""},
    )
    by_id = {item["provider_item_id"]: item for item in items}
    assert by_id["moss_tts_nano_100m"]["compatible_engines"] == ["audio_cpp"]
    dep = by_id["moss_audio_tokenizer_nano"]
    assert dep["compatible_engines"] == []
    assert "moss_tts_nano_100m" in (dep.get("unavailable_reason") or "")
    assert dep["metadata"]["discovery"]["standalone"] is False
    assert dep["metadata"]["discovery"]["parent_package_id"] == "moss_tts_nano_100m"


def test_audio_catalog_utility_packages_are_non_standalone(monkeypatch):
    class Store:
        def get_active_engine_version(self, engine):
            return {"version": "v1", "source_commit": "abc"}

    provider = AudioCppCatalogProvider(Store())
    monkeypatch.setattr(
        "backend.model_catalog.audio_cpp_provider.get_version_entry",
        lambda *args: {"capabilities": {"families": ["voxcpm2"]}},
    )
    items = provider._normalize_packages(
        [
            {
                "id": "voxcpm2",
                "display_name": "VoxCPM2",
                "target_directory": "VoxCPM2",
                "description": "Framework-ready",
                "installable": True,
                "source": {"kind": "composite_snapshot"},
            },
            {
                "id": "voxcpm2_audiovae",
                "display_name": "AudioVAE utility",
                "target_directory": "VoxCPM2",
                "description": "Utility only.",
                "installable": True,
                "source": {
                    "kind": "utility",
                    "operation_kind": "pytorch_to_safetensors",
                },
            },
        ],
        {"version": "v1", "source_commit": "abc"},
    )
    by_id = {item["provider_item_id"]: item for item in items}
    assert by_id["voxcpm2_audiovae"]["metadata"]["discovery"]["standalone"] is False
    assert by_id["voxcpm2_audiovae"]["install_variants"][0]["installable"] is False


def test_audio_catalog_matches_higgs_and_miocodec_without_overlay(monkeypatch):
    class Store:
        def get_active_engine_version(self, engine):
            return {"version": "v1", "source_commit": "abc"}

    provider = AudioCppCatalogProvider(Store())
    monkeypatch.setattr(
        "backend.model_catalog.audio_cpp_provider.get_version_entry",
        lambda *args: {
            "capabilities": {
                "families": ["higgs_tts", "miocodec"],
                "family_tasks": {
                    "higgs_tts": ["tts"],
                    "miocodec": ["codec"],
                },
            },
        },
    )
    items = provider._normalize_packages(
        [
            {
                "id": "higgs_audio_v3_tts_4b",
                "display_name": "Higgs TTS",
                "installable": True,
                "source": {"kind": "huggingface_snapshot", "repo_id": "org/higgs"},
            },
            {
                "id": "miocodec_25hz_44k_v2",
                "display_name": "MioCodec",
                "installable": True,
                "source": {"kind": "huggingface_snapshot", "repo_id": "org/mio"},
            },
        ],
        {"version": "v1", "source_commit": "abc"},
    )
    by_id = {item["provider_item_id"]: item for item in items}
    assert by_id["higgs_audio_v3_tts_4b"]["family"] == "higgs_tts"
    assert by_id["higgs_audio_v3_tts_4b"]["compatible_engines"] == ["audio_cpp"]
    assert by_id["miocodec_25hz_44k_v2"]["family"] == "miocodec"
    assert by_id["miocodec_25hz_44k_v2"]["compatible_engines"] == ["audio_cpp"]


def test_parse_composite_snapshot_preserves_placements():
    source = """
CATALOG = (
    ModelPackage(
        id="bundle",
        display_name="Bundle",
        target_directory="Bundle",
        source=CompositeSnapshotSource(
            placements=(
                Placement(
                    target_subdir="weights",
                    source=SnapshotSource(repo_id="org/weights", revision="main"),
                    required_files=("model.safetensors",),
                ),
            ),
        ),
        required_files=("config.json",),
        description="Composite bundle",
    ),
)
"""
    packages = parse_model_manager_catalog(source)
    assert len(packages) == 1
    placements = packages[0]["source"].get("placements") or []
    assert placements
    assert placements[0]["repo_id"] == "org/weights"
    assert placements[0]["target_subdir"] == "weights"
    assert "model.safetensors" in placements[0]["required_files"]


def test_coerce_positive_int_ignores_malformed_page_values():
    from backend.routes.model_catalog import _coerce_positive_int

    assert _coerce_positive_int({"value": "audio_cpp"}, 1) == 1
    assert _coerce_positive_int(None, 2) == 2
    assert _coerce_positive_int("5", 1) == 5
    assert _coerce_positive_int(3, 1, maximum=2) == 2


def test_search_catalog_post_tolerates_event_like_page(client, monkeypatch):
    captured = {}

    class FakeService:
        async def search(self, **kwargs):
            captured.update(kwargs)
            return {
                "items": [],
                "facets": {},
                "provider_status": {},
                "total": 0,
                "page": kwargs["page"],
                "has_more": False,
            }

    monkeypatch.setattr(
        "backend.routes.model_catalog.ModelCatalogService",
        FakeService,
    )
    r = client.post(
        "/api/model-catalog/search",
        json={"query": "tts", "page": {"value": "audio_cpp"}, "engine": "audio_cpp"},
    )
    assert r.status_code == 200
    assert captured["page"] == 1
    assert captured["filters"]["engine"] == "audio_cpp"


def test_huggingface_gguf_variants_use_files_field():
    from backend.model_catalog.huggingface_provider import HuggingFaceCatalogProvider

    raw = {
        "id": "org/vision-model",
        "name": "org/vision-model",
        "quantizations": {
            "Q4_K_M": {
                "quantization": "Q4_K_M",
                "files": [
                    {"filename": "model-Q4_K_M.gguf", "size": 1000},
                    {"filename": "model-Q4_K_M-00002-of-00002.gguf", "size": 500},
                ],
                "total_size": 1500,
            },
            "Q5_K_M": {
                "quantization": "Q5_K_M",
                "filenames": ["legacy-Q5_K_M.gguf"],
                "total_size": 2000,
            },
        },
        "mmproj_files": [{"filename": "mmproj-F16.gguf", "size": 50}],
    }

    variants = HuggingFaceCatalogProvider._install_variants(raw, "gguf")
    by_id = {variant["id"]: variant for variant in variants}

    assert by_id["Q4_K_M"]["installable"] is True
    assert by_id["Q4_K_M"]["files"] == [
        "model-Q4_K_M.gguf",
        "model-Q4_K_M-00002-of-00002.gguf",
    ]
    assert by_id["Q4_K_M"]["sharded"] is True
    assert by_id["Q4_K_M"]["size_bytes"] == 1500

    assert by_id["Q5_K_M"]["installable"] is True
    assert by_id["Q5_K_M"]["files"] == ["legacy-Q5_K_M.gguf"]

    empty = HuggingFaceCatalogProvider._install_variants(
        {"quantizations": {"Q8_0": {"quantization": "Q8_0", "files": []}}},
        "gguf",
    )
    assert empty[0]["installable"] is False

