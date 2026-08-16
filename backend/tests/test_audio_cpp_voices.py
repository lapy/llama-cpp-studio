"""On-disk packaged TTS voice discovery matching audio.cpp layouts."""

import json

from backend.audio_cpp_voices import (
    apply_packaged_voice_field_options,
    attach_packaged_voices,
    discover_packaged_voices,
    merge_voice_ids,
)


def test_discover_embeddings_safetensors_next_to_model(tmp_path):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "alba.safetensors").write_bytes(b"x")
    (embeddings / "cosette.safetensors").write_bytes(b"x")
    (embeddings / "model.safetensors").write_bytes(b"x")
    (embeddings / "skip.bin").write_bytes(b"x")
    nested = tmp_path / "en" / "embeddings"
    nested.mkdir(parents=True)
    (nested / "clara.safetensors").write_bytes(b"x")
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "M1.wav").write_bytes(b"x")

    voices = discover_packaged_voices(str(tmp_path), family="pocket_tts")
    assert voices == ["alba", "cosette", "model"]


def test_discover_pocket_tts_spec_language_root(tmp_path):
    source = tmp_path / "src"
    (source / "model_specs").mkdir(parents=True)
    (source / "model_specs" / "pocket_tts.json").write_text(
        json.dumps({"sources": [{"roots": {"language": "languages/english"}}]}),
        encoding="utf-8",
    )
    pkg = tmp_path / "pkg"
    english = pkg / "languages" / "english" / "embeddings"
    english.mkdir(parents=True)
    (english / "alba.safetensors").write_bytes(b"x")
    other = pkg / "en" / "embeddings"
    other.mkdir(parents=True)
    (other / "clara.safetensors").write_bytes(b"x")

    assert discover_packaged_voices(str(pkg), family="pocket_tts") == []
    assert discover_packaged_voices(
        str(pkg), family="pocket_tts", source_path=str(source)
    ) == ["alba"]


def test_discover_gguf_file_uses_parent_package_root(tmp_path):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "alba.safetensors").write_bytes(b"x")
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUF")

    assert discover_packaged_voices(str(gguf), family="pocket_tts") == ["alba"]
    assert discover_packaged_voices(str(tmp_path), family="pocket_tts") == ["alba"]


def test_discover_supertonic_voice_styles(tmp_path):
    styles = tmp_path / "voice_styles"
    styles.mkdir()
    (styles / "M1.json").write_text("{}", encoding="utf-8")
    (styles / "F1.json").write_text("{}", encoding="utf-8")
    (styles / "notes.txt").write_text("x", encoding="utf-8")

    assert discover_packaged_voices(str(tmp_path), family="supertonic") == ["F1", "M1"]
    assert discover_packaged_voices(str(tmp_path), family="chatterbox") == []


def test_discover_supertonic_spec_ids_when_sidecars_missing(tmp_path):
    source = tmp_path / "src"
    (source / "model_specs").mkdir(parents=True)
    (source / "model_specs" / "supertonic.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "files": {
                            "voice_style_M1": "model:voice_styles/M1.json",
                            "voice_style_F1": "model:voice_styles/F1.json",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "model.gguf").write_bytes(b"GGUF")

    assert discover_packaged_voices(
        str(pkg), family="supertonic", source_path=str(source)
    ) == ["F1", "M1"]


def test_discover_supertonic_prefers_existing_style_files_over_full_spec(tmp_path):
    source = tmp_path / "src"
    (source / "model_specs").mkdir(parents=True)
    (source / "model_specs" / "supertonic.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "files": {
                            "voice_style_M1": "model:voice_styles/M1.json",
                            "voice_style_F1": "model:voice_styles/F1.json",
                        }
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    styles = tmp_path / "voice_styles"
    styles.mkdir()
    (styles / "M1.json").write_text("{}", encoding="utf-8")

    assert discover_packaged_voices(
        str(tmp_path), family="supertonic", source_path=str(source)
    ) == ["M1"]


def test_discover_neutts_samples_and_spec_enum(tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "emily.txt").write_text("hello", encoding="utf-8")
    (samples / "dave.txt").write_text("hello", encoding="utf-8")

    assert discover_packaged_voices(str(tmp_path), family="neutts") == ["dave", "emily"]

    source = tmp_path / "src"
    (source / "model_specs").mkdir(parents=True)
    (source / "model_specs" / "neutts.json").write_text(
        json.dumps(
            {
                "options": {
                    "request": [
                        {
                            "name": "voice_id",
                            "values": ["dave", "emily", "greta"],
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    empty = tmp_path / "gguf-pkg"
    empty.mkdir()
    assert discover_packaged_voices(
        str(tmp_path), family="neutts", source_path=str(source)
    ) == ["dave", "emily"]
    assert discover_packaged_voices(
        str(empty), family="neutts", source_path=str(source)
    ) == ["dave", "emily", "greta"]


def test_discover_qwen3_custom_speakers_from_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "tts_model_type": "custom_voice",
                "talker_config": {"spk_id": {"Vivian": 0, "Ryan": 1}},
            }
        ),
        encoding="utf-8",
    )
    assert discover_packaged_voices(str(tmp_path), family="qwen3_tts") == [
        "Ryan",
        "Vivian",
    ]
    assert discover_packaged_voices(str(tmp_path), family="pocket_tts") == []


def test_discover_qwen3_base_without_spk_id_is_empty(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"tts_model_type": "base", "talker_config": {"hidden_size": 1}}),
        encoding="utf-8",
    )
    assert discover_packaged_voices(str(tmp_path), family="qwen3_tts") == []


def test_discover_missing_path_is_empty():
    assert discover_packaged_voices("/no/such/audio-bundle") == []
    assert discover_packaged_voices("") == []


def test_attach_writes_disk_embeddings_only(tmp_path):
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "alba.safetensors").write_bytes(b"x")
    inspection = {"family": "pocket_tts", "voices": ["custom"]}
    voices = attach_packaged_voices(inspection, str(tmp_path), "pocket_tts")
    assert voices == ["alba"]
    assert inspection["packaged_voices"] == ["alba"]


def test_apply_packaged_voice_field_options_overlays_voice_id_and_speaker():
    groups = [
        {
            "id": "voice",
            "fields": [
                {"key": "voice_id", "label": "Built-in voice id", "type": "string"},
                {"key": "speaker", "label": "Built-in speaker", "type": "string"},
                {"key": "voice_ref", "label": "Reference", "type": "path"},
            ],
        }
    ]
    out = apply_packaged_voice_field_options(groups, ["cosette", "alba"])
    by_key = {field["key"]: field for field in out[0]["fields"]}
    options = [
        {"value": "alba", "label": "alba"},
        {"value": "cosette", "label": "cosette"},
    ]
    assert by_key["voice_id"]["options"] == options
    assert by_key["speaker"]["options"] == options
    assert "options" not in by_key["voice_ref"]


def test_merge_voice_ids_sorts_and_dedupes():
    assert merge_voice_ids(["cosette"], ["alba", "cosette"]) == ["alba", "cosette"]
