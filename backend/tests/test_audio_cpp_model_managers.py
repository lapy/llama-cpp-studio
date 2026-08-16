"""Tests for audio.cpp model-manager path resolution and v2 catalog shaping."""

from __future__ import annotations

from backend.audio_cpp_model_managers import (
    catalog_json_has_identity,
    gguf_snapshot_sidecar_prefixes,
    manager_paths_for_source,
    merge_catalog_packages,
    normalize_v2_catalog_packages,
    resolve_model_manager_legacy_path,
    resolve_model_manager_path,
    resolve_model_manager_v2_path,
)
from backend.model_catalog.audio_cpp_provider import resolve_studio_install_method


def test_manager_paths_prefer_v2_over_deprecated(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "model_manager_v2.py").write_text("# v2\n", encoding="utf-8")
    (tools / "model_manager_deprecated.py").write_text("# legacy\n", encoding="utf-8")

    paths = manager_paths_for_source(str(tmp_path))
    assert paths["model_manager_path"].endswith("model_manager_v2.py")
    assert paths["model_manager_v2_path"].endswith("model_manager_v2.py")
    assert paths["model_manager_legacy_path"].endswith("model_manager_deprecated.py")
    assert resolve_model_manager_path(str(tmp_path)).endswith("model_manager_v2.py")
    assert resolve_model_manager_legacy_path(str(tmp_path)).endswith(
        "model_manager_deprecated.py"
    )


def test_manager_paths_fall_back_to_legacy_model_manager_py(tmp_path):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "model_manager.py").write_text("# legacy\n", encoding="utf-8")

    paths = manager_paths_for_source(str(tmp_path))
    assert paths["model_manager_path"].endswith("model_manager.py")
    assert paths["model_manager_v2_path"] == ""
    assert paths["model_manager_legacy_path"].endswith("model_manager.py")


def test_normalize_v2_catalog_packages_maps_to_direct_install():
    packages = normalize_v2_catalog_packages(
        [
            {
                "family": "qwen3_tts",
                "id": "qwen3_tts_q8",
                "display_name": "Qwen3 TTS Q8",
                "format": "gguf",
                "precision": "q8_0",
                "default": True,
                "target_directory": "qwen3_tts",
                "repo": "audio-cpp/audio.cpp-gguf",
            }
        ]
    )
    assert len(packages) == 1
    package = packages[0]
    assert package["manager_backend"] == "v2"
    assert package["source"]["kind"] == "huggingface_snapshot"
    assert package["source"]["repo_id"] == "audio-cpp/audio.cpp-gguf"
    assert package["family"] == "qwen3_tts"
    assert "qwen3_tts/embeddings/" in package["source"]["include_prefixes"]
    assert resolve_studio_install_method(package) == "direct"
    assert catalog_json_has_identity(
        {
            "family": "qwen3_tts",
            "id": "qwen3_tts_q8",
            "target_directory": "qwen3_tts",
            "repo": "audio-cpp/audio.cpp-gguf",
        }
    )


def test_merge_catalog_packages_keeps_v2_and_adds_legacy_leftovers():
    preferred = normalize_v2_catalog_packages(
        [
            {
                "family": "qwen3_tts",
                "id": "qwen3_tts",
                "display_name": "Qwen3 TTS",
                "format": "gguf",
                "precision": "q8_0",
                "default": True,
                "target_directory": "qwen3_tts",
                "repo": "audio-cpp/audio.cpp-gguf",
            }
        ]
    )
    extra = [
        {
            "id": "qwen3_tts",
            "display_name": "Legacy duplicate",
            "source": {"kind": "huggingface_snapshot", "repo_id": "old/repo"},
            "install_kind": "snapshot",
            "manager_backend": "legacy",
        },
        {
            "id": "vibevoice_asr",
            "display_name": "VibeVoice ASR",
            "source": {"kind": "composite_snapshot"},
            "install_kind": "composite",
            "manager_backend": "legacy",
        },
    ]
    merged = merge_catalog_packages(preferred, extra)
    assert [row["id"] for row in merged] == ["qwen3_tts", "vibevoice_asr"]
    assert merged[0]["manager_backend"] == "v2"
    assert merged[1]["manager_backend"] == "legacy"


def test_resolve_v2_from_version_row_without_source_path(tmp_path):
    v2 = tmp_path / "model_manager_v2.py"
    v2.write_text("# v2\n", encoding="utf-8")
    assert (
        resolve_model_manager_v2_path(version_row={"model_manager_v2_path": str(v2)})
        == str(v2)
    )


def test_gguf_snapshot_sidecar_prefixes_add_embeddings_next_to_weight():
    prefixes = gguf_snapshot_sidecar_prefixes(
        ["PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf"]
    )
    assert "PocketTTS-GGUF/english/embeddings/" in prefixes
    assert "PocketTTS-GGUF/english/tokenizer.model" in prefixes


def test_normalize_v2_pocket_tts_includes_embedding_prefix_without_files_list():
    packages = normalize_v2_catalog_packages(
        [
            {
                "family": "pocket_tts",
                "id": "pocket_tts_english_q8_0",
                "format": "gguf",
                "target_directory": "PocketTTS-GGUF/english",
                "repo": "audio-cpp/audio.cpp-gguf",
            }
        ]
    )
    prefixes = packages[0]["source"]["include_prefixes"]
    assert "PocketTTS-GGUF/english/embeddings/" in prefixes
