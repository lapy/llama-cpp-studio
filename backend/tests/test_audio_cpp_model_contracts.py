"""Tests for typed audio.cpp model-spec contract loading."""

from __future__ import annotations

import json

from backend.audio_cpp_model_contracts import (
    dependency_sidecar_fields,
    load_family_contract,
    load_family_contracts,
    public_option_key,
)
from backend.audio_task_profiles import sidecar_session_fields_for


def test_public_option_key_prefers_runtime_model_path_alias():
    assert (
        public_option_key("qwen3_asr", "forced_aligner_path")
        == "qwen3_asr.forced_aligner_model_path"
    )
    assert public_option_key("miotts", "codec_path") == "miotts.codec_model_path"
    assert (
        public_option_key(
            "qwen3_asr",
            "forced_aligner_path",
            known_keys={"qwen3_asr.forced_aligner_path"},
        )
        == "qwen3_asr.forced_aligner_path"
    )


def test_load_family_contract_prefers_typed_model_specs(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "outetts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "outetts",
                "display_name": "OuteTTS",
                "category": "community",
                "status": "community",
                "tasks": ["tts", "clone"],
                "modes": ["offline"],
                "languages": ["en"],
                "capabilities": {"clone": ["speaker_reference"]},
                "options": {"request": [], "session": [], "load": []},
                "packages": [{"id": "outetts_q8", "display_name": "Q8"}],
                "dependencies": [],
                "ui": {
                    "recommended_package": "outetts_q8",
                    "tags": ["TTS"],
                    "docs": [],
                },
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "outetts")
    assert contract is not None
    assert contract["typed"] is True
    assert contract["source"] == "model_specs"
    assert contract["tasks"] == ["tts", "clone"]


def test_load_family_contract_overlays_model_specs_v1(tmp_path):
    specs = tmp_path / "model_specs"
    preview = tmp_path / "model_specs_v1"
    specs.mkdir()
    preview.mkdir()
    (specs / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "family": "qwen3_asr",
                "sources": [{"files": {"config": "model:config.json"}}],
                "packages": [{"id": "qwen3_asr_1_7b_q8_0"}],
            }
        ),
        encoding="utf-8",
    )
    (preview / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "family": "qwen3_asr",
                "display_name": "Qwen3-ASR",
                "category": "asr",
                "status": "supported",
                "tasks": ["asr"],
                "modes": ["offline"],
                "capabilities": {"asr": ["word_timestamps"]},
                "options": {
                    "session": [
                        {"name": "forced_aligner_path", "type": "path", "required": False}
                    ]
                },
                "dependencies": [
                    {
                        "kind": "model",
                        "family": "qwen3_forced_aligner",
                        "scope": "session",
                        "option": "forced_aligner_path",
                        "required": False,
                        "required_when": [
                            {
                                "scope": "request",
                                "option_key": "return_timestamps",
                                "equals": True,
                            }
                        ],
                    },
                    {
                        "kind": "bundled_model",
                        "family": "silero_vad",
                        "path": "assets/framework/models/silero_vad",
                        "scope": "session",
                        "option": "vad_path",
                        "required": False,
                        "required_when": [
                            {
                                "scope": "request",
                                "option_key": "audio_chunk_mode",
                                "equals": "vad",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "qwen3_asr")
    assert contract is not None
    assert contract["source"] == "model_specs_v1"
    assert contract["typed"] is False
    deps = {dep["option_key"]: dep for dep in contract["dependencies"]}
    assert "qwen3_asr.forced_aligner_model_path" in deps
    assert deps["qwen3_asr.forced_aligner_model_path"]["family"] == "qwen3_forced_aligner"
    assert deps["qwen3_asr.vad_model_path"]["kind"] == "bundled_model"
    assert deps["qwen3_asr.vad_model_path"]["path"].endswith("silero_vad")


def test_sidecar_session_fields_from_contract_dependencies(tmp_path):
    preview = tmp_path / "model_specs_v1"
    preview.mkdir()
    (preview / "miotts.json").write_text(
        json.dumps(
            {
                "family": "miotts",
                "category": "tts",
                "tasks": ["tts"],
                "modes": ["offline"],
                "dependencies": [
                    {
                        "kind": "model",
                        "family": "miocodec",
                        "scope": "session",
                        "option": "codec_path",
                        "required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fields = sidecar_session_fields_for(
        "tts", "miotts", source_path=str(tmp_path)
    )
    keys = {field["key"] for field in fields}
    assert "miotts.codec_model_path" in keys
    codec = next(field for field in fields if field["key"] == "miotts.codec_model_path")
    assert codec["dependency"]["family"] == "miocodec"
    assert codec["dependency"]["required"] is True


def test_dependency_sidecar_omits_scanned_keys(tmp_path):
    preview = tmp_path / "model_specs_v1"
    preview.mkdir()
    (preview / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "family": "qwen3_asr",
                "category": "asr",
                "tasks": ["asr"],
                "modes": ["offline"],
                "dependencies": [
                    {
                        "kind": "model",
                        "family": "qwen3_forced_aligner",
                        "scope": "session",
                        "option": "forced_aligner_path",
                        "required": False,
                        "required_when": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    fields = sidecar_session_fields_for(
        "asr",
        "qwen3_asr",
        source_path=str(tmp_path),
        profile_sections=[
            {
                "params": [
                    {
                        "key": "qwen3_asr.forced_aligner_model_path",
                        "scope": "session_option",
                    }
                ]
            }
        ],
    )
    assert fields == []


def test_load_family_contracts_collects_dependency_map(tmp_path):
    preview = tmp_path / "model_specs_v1"
    preview.mkdir()
    (preview / "vibevoice_asr.json").write_text(
        json.dumps(
            {
                "family": "vibevoice_asr",
                "category": "asr",
                "tasks": ["asr"],
                "modes": ["offline"],
                "dependencies": [
                    {
                        "kind": "bundled_model",
                        "family": "silero_vad",
                        "path": "assets/framework/models/silero_vad",
                        "scope": "session",
                        "option": "vad_path",
                        "required": False,
                        "required_when": [
                            {
                                "scope": "request",
                                "option_key": "audio_chunk_mode",
                                "equals": "vad",
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    contracts = load_family_contracts(str(tmp_path), families=["vibevoice_asr"])
    fields = dependency_sidecar_fields(
        "vibevoice_asr", contracts["vibevoice_asr"]["dependencies"]
    )
    assert fields[0]["key"] == "vibevoice_asr.vad_model_path"
    assert fields[0]["placeholder"].endswith("silero_vad")
