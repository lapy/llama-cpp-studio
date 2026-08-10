"""Tests for versioned audio.cpp model-spec contract loading."""

from __future__ import annotations

import json

from backend.audio_cpp_model_contracts import (
    dependency_sidecar_fields,
    load_family_contract,
    load_family_contracts,
    public_option_key,
)
from backend.audio_task_profiles import (
    apply_dependency_field_overlays,
    sidecar_session_fields_for,
)


def _write_spec(root, family, **overrides):
    specs = root / "model_specs"
    specs.mkdir(exist_ok=True)
    payload = {
        "schema_version": 1,
        "family": family,
        "display_name": family,
        "category": "audio",
        "status": "supported",
        "tasks": ["asr"],
        "modes": ["offline"],
        "languages": [],
        "capabilities": {},
        "options": {"request": [], "session": [], "load": []},
        "packages": [],
        "dependencies": [],
        "ui": {},
        "sources": [],
        **overrides,
    }
    (specs / f"{family}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_public_option_key_uses_declared_name():
    assert (
        public_option_key("qwen3_asr", "forced_aligner_path")
        == "qwen3_asr.forced_aligner_path"
    )
    assert public_option_key("miotts", "codec_path") == "miotts.codec_path"


def test_load_family_contract_requires_versioned_model_spec(tmp_path):
    _write_spec(
        tmp_path,
        "qwen3_asr",
        dependencies=[
            {
                "kind": "model",
                "family": "qwen3_forced_aligner",
                "scope": "session",
                "option": "forced_aligner_path",
                "required": False,
            }
        ],
    )
    contract = load_family_contract(str(tmp_path), "qwen3_asr")
    assert contract is not None
    assert contract["typed"] is True
    assert contract["source"] == "model_specs"
    assert contract["dependencies"][0]["option_key"] == (
        "qwen3_asr.forced_aligner_path"
    )


def test_load_family_contract_rejects_unversioned_spec(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "legacy.json").write_text(
        json.dumps({"family": "legacy", "tasks": ["asr"]}),
        encoding="utf-8",
    )
    assert load_family_contract(str(tmp_path), "legacy") is None


def test_sidecar_fields_come_from_versioned_dependencies(tmp_path):
    _write_spec(
        tmp_path,
        "miotts",
        tasks=["tts"],
        dependencies=[
            {
                "kind": "model",
                "family": "miocodec",
                "scope": "session",
                "option": "codec_path",
                "required": True,
            }
        ],
    )
    fields = sidecar_session_fields_for(
        "tts", "miotts", source_path=str(tmp_path)
    )
    assert [field["key"] for field in fields] == ["miotts.codec_path"]
    assert fields[0]["dependency"]["family"] == "miocodec"
    assert fields[0]["dependency"]["required"] is True


def test_load_family_contracts_collects_versioned_specs(tmp_path):
    _write_spec(tmp_path, "vibevoice_asr")
    contracts = load_family_contracts(str(tmp_path))
    assert list(contracts) == ["vibevoice_asr"]


def test_dependency_sidecar_dedupes_exact_keys():
    dependency = {
        "kind": "model",
        "family": "peer",
        "scope": "session",
        "option": "peer_path",
        "option_key": "demo.peer_path",
        "required": False,
        "required_when": [],
    }
    fields = dependency_sidecar_fields("demo", [dependency, dict(dependency)])
    assert [field["key"] for field in fields] == ["demo.peer_path"]


def test_apply_dependency_overlays_enriches_scanned_params_and_keeps_gaps():
    sections = [{
        "id": "session",
        "params": [{
            "key": "qwen3_asr.forced_aligner_path",
            "scope": "session_option",
            "label": "forced_aligner_path",
        }],
    }]
    dependency_fields = [
        {
            "key": "qwen3_asr.forced_aligner_path",
            "label": "Forced aligner model path",
            "scope": "session_option",
            "dependency": {"family": "qwen3_forced_aligner", "required": False},
        },
        {
            "key": "qwen3_asr.vad_path",
            "label": "VAD model path",
            "scope": "session_option",
            "dependency": {"family": "silero_vad", "required": False},
        },
    ]
    overlaid, gaps = apply_dependency_field_overlays(sections, dependency_fields)
    assert overlaid[0]["params"][0]["label"] == "Forced aligner model path"
    assert [field["key"] for field in gaps] == ["qwen3_asr.vad_path"]
