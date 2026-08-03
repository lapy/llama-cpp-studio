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
    assert contract["temporary"] is False
    assert contract["adapter"] is None
    assert contract["source"] == "model_specs"
    assert contract["tasks"] == ["tts", "clone"]
    assert contract["temporary_peer_seeds"] is True
    deps = {dep["option_key"]: dep for dep in contract["dependencies"]}
    assert "outetts.aligner_model_path" in deps
    assert deps["outetts.aligner_model_path"]["temporary_seed"] is True
    assert deps["outetts.aligner_model_path"]["family"] == "qwen3_forced_aligner"


def test_temporary_peer_seeds_vevo2_whisper(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "vevo2.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "vevo2",
                "display_name": "VeVo2",
                "category": "vc",
                "status": "supported",
                "tasks": ["tts", "vc", "s2s", "svc"],
                "modes": ["offline"],
                "languages": ["en"],
                "capabilities": {},
                "options": {"request": [], "session": [], "load": []},
                "packages": [],
                "dependencies": [],
                "ui": {},
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "vevo2")
    assert contract is not None
    assert contract["typed"] is True
    assert contract["temporary_peer_seeds"] is True
    deps = {dep["option_key"]: dep for dep in contract["dependencies"]}
    assert "vevo2.whisper_model_path" in deps
    assert deps["vevo2.whisper_model_path"]["kind"] == "external"
    assert deps["vevo2.whisper_model_path"]["temporary_seed"] is True


def test_temporary_peer_seeds_skip_when_upstream_declares(tmp_path):
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
                "tasks": ["tts"],
                "modes": ["offline"],
                "languages": ["en"],
                "capabilities": {},
                "options": {"request": [], "session": [], "load": []},
                "packages": [],
                "dependencies": [
                    {
                        "kind": "model",
                        "family": "qwen3_forced_aligner",
                        "scope": "session",
                        "option": "aligner_path",
                        "required": False,
                    }
                ],
                "ui": {},
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "outetts")
    assert contract is not None
    assert contract.get("temporary_peer_seeds") is not True
    deps = contract["dependencies"]
    assert len(deps) == 1
    assert deps[0]["option_key"] == "outetts.aligner_model_path"
    assert deps[0].get("temporary_seed") is not True


def test_temporary_peer_seed_stub_without_spec(tmp_path):
    contract = load_family_contract(str(tmp_path), "vevo2")
    assert contract is not None
    assert contract["source"] == "temporary_peer_seed"
    assert contract["temporary"] is True
    assert contract["temporary_peer_seeds"] is True
    assert any(
        dep["option_key"] == "vevo2.whisper_model_path" for dep in contract["dependencies"]
    )


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
    assert contract["temporary"] is True
    assert contract["adapter"] == "pre_v1_model_specs_v1"
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
    from backend.audio_cpp_model_contracts import temporary_adapter_families

    assert temporary_adapter_families(contracts) == ["vibevoice_asr"]

    seeded = load_family_contracts(
        str(tmp_path), families=["vibevoice_asr", "vevo2", "outetts"]
    )
    assert set(temporary_adapter_families(seeded)) == {
        "outetts",
        "vevo2",
        "vibevoice_asr",
    }


def test_dependency_sidecar_dedupes_path_aliases():
    fields = dependency_sidecar_fields(
        "qwen3_asr",
        [
            {
                "kind": "model",
                "family": "qwen3_forced_aligner",
                "scope": "session",
                "option": "forced_aligner_path",
                "option_key": "qwen3_asr.forced_aligner_model_path",
                "required": False,
                "required_when": [],
            },
            {
                "kind": "model",
                "family": "qwen3_forced_aligner",
                "scope": "session",
                "option": "forced_aligner_path",
                "option_key": "qwen3_asr.forced_aligner_path",
                "required": False,
                "required_when": [],
            },
        ],
    )
    assert [field["key"] for field in fields] == [
        "qwen3_asr.forced_aligner_model_path"
    ]
    assert fields[0]["dependency"]["family"] == "qwen3_forced_aligner"


def test_dependency_sidecar_marks_temporary_seed():
    fields = dependency_sidecar_fields(
        "vevo2",
        [
            {
                "kind": "external",
                "family": "whisper",
                "scope": "load",
                "option": "whisper_model_path",
                "option_key": "vevo2.whisper_model_path",
                "required": False,
                "required_when": [],
                "temporary_seed": True,
            }
        ],
    )
    assert fields[0]["scope"] == "load_option"
    assert fields[0]["dependency"]["temporary_seed"] is True
    assert "until upstream" in fields[0]["description"]


def test_apply_dependency_overlays_enriches_scanned_params_and_keeps_gaps():
    from backend.audio_task_profiles import apply_dependency_field_overlays

    sections = [
        {
            "id": "session",
            "params": [
                {
                    "key": "qwen3_asr.forced_aligner_model_path",
                    "scope": "session_option",
                    "label": "forced_aligner_model_path",
                    "description": "raw help text",
                }
            ],
        }
    ]
    dependency_fields = [
        {
            "key": "qwen3_asr.forced_aligner_model_path",
            "label": "Forced aligner model path",
            "description": "Path to an installed Qwen3 Forced Aligner bundle.",
            "placeholder": "/models/aligner",
            "scope": "session_option",
            "install_hint": "Install a `qwen3_forced_aligner` package from Models search.",
            "dependency": {
                "family": "qwen3_forced_aligner",
                "required": False,
                "temporary_seed": False,
            },
        },
        {
            "key": "qwen3_asr.vad_model_path",
            "label": "VAD model path",
            "scope": "session_option",
            "dependency": {"family": "silero_vad", "required": False},
            "install_hint": "Install a `silero_vad` package from Models search.",
        },
    ]
    overlaid, gaps = apply_dependency_field_overlays(sections, dependency_fields)
    param = overlaid[0]["params"][0]
    assert param["label"] == "Forced aligner model path"
    assert param["dependency"]["family"] == "qwen3_forced_aligner"
    assert "Install a `qwen3_forced_aligner`" in param["install_hint"]
    assert [field["key"] for field in gaps] == ["qwen3_asr.vad_model_path"]
