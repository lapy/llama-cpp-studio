"""Source-assisted audio.cpp option discovery."""

from __future__ import annotations

import os
from pathlib import Path

from backend.audio_cpp_option_discovery import (
    discover_family_options,
    merge_discovered_options_into_sections,
    normalize_audio_option_key,
    parse_loader_cli_options,
    parse_session_accepted_keys,
    scanned_request_field_groups,
)
from backend.cli_help_parsers import parse_audio_cpp_help_to_sections

_REPO = Path(__file__).resolve().parents[2]
_AUDIO_SRC = _REPO / "data" / "audio-cpp" / "src"


def test_normalize_strips_transport_prefix_and_bool_default():
    assert normalize_audio_option_key(
        "--session-option chatterbox.mem_saver=true"
    ) == ("chatterbox.mem_saver", "true")
    assert normalize_audio_option_key("nemotron_asr.weight_type") == (
        "nemotron_asr.weight_type",
        None,
    )


def test_parse_chatterbox_loader_session_options():
    text = """
        inspection.cli.session_options = {
            {
                "--session-option chatterbox.conditionals_cache_slots",
                "n",
                "Prepared voice-condition cache slots; default 1, set 0 to disable.",
            },
            {
                "--session-option chatterbox.mem_saver=true",
                "",
                "Free non-conditional runtime graphs after each request chunk; default false.",
            },
            {
                "--source-audio",
                "wav",
                "Source speech audio for Chatterbox voice conversion.",
            },
        };
"""
    params = parse_loader_cli_options(text)
    by_key = {p["key"]: p for p in params}
    assert "chatterbox.conditionals_cache_slots" in by_key
    assert by_key["chatterbox.mem_saver"]["type"] == "bool"
    assert by_key["source_audio"]["scope"] == "request_option"


def test_parse_qwen3_asr_session_accepted_keys_from_fixture_text():
    text = """
            key != "qwen3_asr.audio_encoder_graph_arena_mb" &&
            key != "qwen3_asr.thinker_prefill_graph_arena_mb" &&
            key != "qwen3_asr.weight_type" &&
            key != "qwen3_asr.forced_aligner_model_path" &&
            key != "qwen3_asr.vad_model_path") {
        auto path = runtime::find_option(options.options, {"qwen3_asr.aligner_model_path"});
"""
    keys = {p["key"] for p in parse_session_accepted_keys(text, "qwen3_asr")}
    assert "qwen3_asr.forced_aligner_model_path" in keys
    assert "qwen3_asr.vad_model_path" in keys
    assert "qwen3_asr.weight_type" in keys
    assert "qwen3_asr.aligner_model_path" in keys


def test_parse_request_options_from_session_cpp():
    text = """
    if (const auto value = runtime::parse_int_option(request.options, {"max_tokens"})) {
        out.max_tokens = *value;
    }
    if (const auto value = runtime::parse_float_option(request.options, {"temperature"})) {
        out.temperature = *value;
    }
    if (const auto value = runtime::find_option(request.options, {"do_sample"})) {
        out.do_sample = runtime::parse_bool_option(*value, "do_sample");
    }
    if (key != "miotts.weight_type") {
    }
"""
    params = parse_session_accepted_keys(text, "miotts")
    by_key = {p["key"]: p for p in params}
    assert by_key["max_tokens"]["scope"] == "request_option"
    assert by_key["temperature"]["scope"] == "request_option"
    assert by_key["do_sample"]["scope"] == "request_option"
    assert by_key["miotts.weight_type"]["scope"] == "session_option"
    # request.options keys must not leak into session scope
    assert by_key["max_tokens"]["scope"] != "session_option"


def test_merge_fills_qwen3_asr_help_gaps_from_discovery():
    here = os.path.dirname(__file__)
    with open(
        os.path.join(here, "fixtures", "audio_cpp_qwen3_asr_help_live.txt"),
        encoding="utf-8",
    ) as handle:
        sections = parse_audio_cpp_help_to_sections(handle.read(), source="cli")
    discovered = [
        {
            "key": "qwen3_asr.forced_aligner_model_path",
            "scope": "session_option",
            "type": "path",
            "section_id": "model_session_options",
            "section_label": "Model session options",
            "label": "Forced Aligner Model Path",
            "description": "aligner",
        },
        {
            "key": "qwen3_asr.weight_type",
            "scope": "session_option",
            "type": "select",
            "options": [{"value": "native", "label": "native"}],
            "section_id": "model_session_options",
            "section_label": "Model session options",
            "label": "Weight Type",
        },
    ]
    merged = merge_discovered_options_into_sections(sections, discovered)
    scoped = {
        (p["scope"], p["key"]): p
        for section in merged
        for p in section["params"]
    }
    assert ("session_option", "qwen3_asr.forced_aligner_model_path") in scoped
    assert ("session_option", "qwen3_asr.weight_type") in scoped
    # Existing help rows remain.
    assert ("request_option", "audio_chunk_mode") in scoped


def test_scanned_request_field_groups_nested():
    groups = scanned_request_field_groups(
        [
            {
                "id": "model_request_options",
                "params": [
                    {
                        "key": "language",
                        "scope": "request_option",
                        "type": "string",
                        "label": "Language",
                    }
                ],
            }
        ]
    )
    assert groups[0]["fields"][0]["nested"] is True
    assert groups[0]["fields"][0]["options_key"] == "language"


def test_scanned_request_field_groups_drops_cli_only_io_flags():
    groups = scanned_request_field_groups(
        [
            {
                "id": "model_request_options",
                "params": [
                    {"key": "language", "scope": "request_option", "type": "string"},
                    {"key": "out", "scope": "request_option", "type": "path"},
                    {"key": "out_dir", "scope": "request_option", "type": "path"},
                    {"key": "batch_audio_dir", "scope": "request_option", "type": "path"},
                    {"key": "text_out", "scope": "request_option", "type": "path"},
                    {"key": "return_timestamps", "scope": "request_option", "type": "bool"},
                    {"key": "audio", "scope": "request_option", "type": "path"},
                ],
            }
        ]
    )
    keys = {field["key"] for field in groups[0]["fields"]}
    assert keys == {"language", "return_timestamps"}


def test_live_source_discovers_qwen3_asr_unadvertised_options():
    if not (_AUDIO_SRC / "src" / "models" / "qwen3_asr" / "session.cpp").is_file():
        return
    params = discover_family_options(str(_AUDIO_SRC), "qwen3_asr")
    keys = {p["key"] for p in params}
    assert "qwen3_asr.forced_aligner_model_path" in keys
    assert "qwen3_asr.vad_model_path" in keys
    assert "qwen3_asr.weight_type" in keys
    weight = next(p for p in params if p["key"] == "qwen3_asr.weight_type")
    assert weight["type"] == "select"
    assert [o["value"] for o in weight["options"]][:3] == ["native", "f32", "f16"]


def test_live_source_discovers_nemotron_loader_session_options():
    if not (_AUDIO_SRC / "src" / "models" / "nemotron_asr" / "loader.cpp").is_file():
        return
    params = discover_family_options(str(_AUDIO_SRC), "nemotron_asr")
    keys = {p["key"] for p in params}
    assert "nemotron_asr.mem_saver" in keys
    assert "nemotron_asr.weight_type" in keys


def test_live_source_discovers_sortformer_and_silero_bare_keys():
    if not (_AUDIO_SRC / "src" / "models" / "sortformer_diar").is_dir():
        return
    sortformer = {p["key"] for p in discover_family_options(str(_AUDIO_SRC), "sortformer_diar")}
    assert "speaker_threshold" in sortformer
    assert "sortformer_diar.weight_type" in sortformer
    silero = {p["key"] for p in discover_family_options(str(_AUDIO_SRC), "silero_vad")}
    assert "threshold" in silero
    assert "silero_vad.weight_type" in silero


def test_live_source_discovers_mel_band_roformer_concat_key():
    if not (_AUDIO_SRC / "src" / "models" / "roformer" / "loader.cpp").is_file():
        return
    keys = {p["key"] for p in discover_family_options(str(_AUDIO_SRC), "mel_band_roformer")}
    assert "mel_band_roformer.weight_type" in keys


def test_live_source_qwen3_asr_does_not_steal_sibling_docs_keys():
    if not (_AUDIO_SRC / "docs" / "qwen3.md").is_file():
        return
    keys = {p["key"] for p in discover_family_options(str(_AUDIO_SRC), "qwen3_asr")}
    assert "qwen3_asr.forced_aligner_model_path" in keys
    assert "qwen3_tts.weight_type" not in keys
    assert "vibevoice_asr.vad_model_path" not in keys


def test_family_discovery_coverage_smoke():
    """Most loader families should yield at least one discovered option from source."""
    if not (_AUDIO_SRC / "src" / "models").is_dir():
        return
    from pathlib import Path

    models = _AUDIO_SRC / "src" / "models"
    families = sorted({p.parent.name for p in models.rglob("loader.cpp")})
    # Map directory names to runtime family ids where they differ.
    family_ids = {
        "demucs": "htdemucs",
        "roformer": "mel_band_roformer",
        "moss": None,  # nested; handled via child dirs
    }
    checked = 0
    empty = []
    for fam_dir in families:
        fam = family_ids.get(fam_dir, fam_dir)
        if not fam:
            continue
        opts = discover_family_options(str(_AUDIO_SRC), fam)
        checked += 1
        if not opts and fam not in {"miocodec", "miotts"}:
            empty.append(fam)
    assert checked >= 20
    # Allow a couple of tiny/utility families with no session surface.
    assert len(empty) <= 3, f"unexpected empty families: {empty}"


def test_parse_chatterbox_help_normalizes_mem_saver_key():
    text = """
family=chatterbox
  Model session options:
    --session-option chatterbox.conditionals_cache_slots <n>  Prepared voice-condition cache slots
    --session-option chatterbox.mem_saver=true  Free graphs after each request
    --source-audio <wav>  Source speech audio
"""
    scoped = {
        (p["scope"], p["key"]): p
        for section in parse_audio_cpp_help_to_sections(text, source="cli")
        for p in section["params"]
    }
    assert ("session_option", "chatterbox.mem_saver") in scoped
    assert scoped[("session_option", "chatterbox.mem_saver")]["type"] == "bool"
    assert ("request_option", "source_audio") in scoped
