"""Tests for typed audio.cpp model-spec contract loading."""

from __future__ import annotations

import json
from pathlib import Path

from backend.audio_cpp_model_contracts import (
    contracts_fingerprint,
    dependency_sidecar_fields,
    family_dependencies_map,
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


def test_load_family_contract_accepts_unversioned_typed_shaped_spec(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "family": "qwen3_asr",
                "tasks": ["asr"],
                "capabilities": {"asr": ["word_timestamps"]},
                "options": {
                    "request": [
                        {
                            "name": "clamp_timestamps_to_audio",
                            "type": "bool",
                            "required": False,
                        }
                    ],
                    "session": [],
                    "load": [],
                },
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "qwen3_asr")
    assert contract is not None
    assert contract["typed"] is False
    assert contract["source"] == "model_specs"
    assert contract["capabilities"]["asr"] == ["word_timestamps"]


def test_load_family_contract_rejects_unversioned_spec(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "legacy.json").write_text(
        json.dumps({"family": "legacy", "tasks": ["asr"]}),
        encoding="utf-8",
    )
    assert load_family_contract(str(tmp_path), "legacy") is None


def test_load_family_contract_seeds_missing_peer_paths(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "vevo2.json").write_text(
        json.dumps(
            {
                "family": "vevo2",
                "tasks": ["vc"],
                "capabilities": {"vc": ["speaker_reference"]},
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "vevo2")
    assert contract is not None
    keys = [row["option_key"] for row in contract["dependencies"]]
    assert "vevo2.whisper_model_path" in keys


def test_load_family_contract_seeds_outetts_aligner_path(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "outetts.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "outetts",
                "tasks": ["tts", "clone"],
                "capabilities": {"clone": ["speaker_reference"]},
                "options": {"request": [], "session": [], "load": []},
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "outetts")
    assert contract is not None
    keys = [row["option_key"] for row in contract["dependencies"]]
    assert keys.count("outetts.aligner_path") == 1
    aligner = next(
        row for row in contract["dependencies"] if row["option_key"] == "outetts.aligner_path"
    )
    assert aligner["family"] == "qwen3_forced_aligner"
    assert aligner["option"] == "aligner_path"
    assert aligner["required"] is False


def test_peer_seeds_do_not_duplicate_declared_dependencies(tmp_path):
    _write_spec(
        tmp_path,
        "vevo2",
        tasks=["vc"],
        dependencies=[
            {
                "kind": "external",
                "family": "whisper",
                "scope": "load",
                "option": "whisper_model_path",
                "required": False,
            }
        ],
    )
    contract = load_family_contract(str(tmp_path), "vevo2")
    keys = [row["option_key"] for row in contract["dependencies"]]
    assert keys.count("vevo2.whisper_model_path") == 1


def test_load_family_contract_accepts_capabilities_only_unversioned_spec(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "pocket_tts.json").write_text(
        json.dumps(
            {
                "family": "pocket_tts",
                "tasks": ["tts", "clone"],
                "capabilities": {"clone": ["speaker_reference"]},
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "pocket_tts")
    assert contract is not None
    assert contract["typed"] is False
    assert contract["source"] == "model_specs"
    assert contract["capabilities"]["clone"] == ["speaker_reference"]


def test_load_family_contract_rejects_package_only_junk(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "junk.json").write_text(
        json.dumps(
            {
                "family": "junk",
                "packages": [{"id": "junk_q8", "files": ["junk.gguf"]}],
            }
        ),
        encoding="utf-8",
    )
    assert load_family_contract(str(tmp_path), "junk") is None


def test_load_family_contract_ignores_model_specs_v1_tree(tmp_path):
    live = tmp_path / "model_specs"
    preview = tmp_path / "model_specs_v1"
    live.mkdir()
    preview.mkdir()
    (live / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "qwen3_asr",
                "tasks": ["asr"],
                "display_name": "live",
                "options": {"request": [], "session": [], "load": []},
            }
        ),
        encoding="utf-8",
    )
    (preview / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "qwen3_asr",
                "tasks": ["asr"],
                "display_name": "preview-v1",
                "options": {"request": [], "session": [], "load": []},
            }
        ),
        encoding="utf-8",
    )
    contract = load_family_contract(str(tmp_path), "qwen3_asr")
    assert contract is not None
    assert contract["display_name"] == "live"
    assert contract["source"] == "model_specs"


def test_load_family_contract_does_not_fall_back_to_model_specs_v1(tmp_path):
    preview = tmp_path / "model_specs_v1"
    preview.mkdir()
    (preview / "qwen3_asr.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "family": "qwen3_asr",
                "tasks": ["asr"],
                "options": {"request": [], "session": [], "load": []},
            }
        ),
        encoding="utf-8",
    )
    assert load_family_contract(str(tmp_path), "qwen3_asr") is None
    assert load_family_contracts(str(tmp_path)) == {}


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


def test_family_dependencies_map_and_fingerprint_cover_seeded_contracts(tmp_path):
    _write_spec(tmp_path, "vevo2", tasks=["vc"], capabilities={"vc": ["speaker_reference"]})
    contracts = load_family_contracts(str(tmp_path))
    mapped = family_dependencies_map(contracts)
    assert "vevo2.whisper_model_path" in [
        row["option_key"] for row in mapped["vevo2"]
    ]
    digest = contracts_fingerprint(contracts)
    assert len(digest) == 64
    assert contracts_fingerprint(contracts) == digest


def test_pinned_checkout_loads_live_specs_without_model_specs_v1():
    source = (
        Path(__file__).resolve().parents[2] / "data" / "audio-cpp" / "src"
    )
    if not (source / "model_specs").is_dir():
        return
    assert not (source / "model_specs_v1").exists()
    assert (source / "tools" / "model_manager_v2.py").is_file()

    pocket = load_family_contract(str(source), "pocket_tts")
    assert pocket is not None
    assert pocket["typed"] is False
    assert pocket["source"] == "model_specs"
    assert pocket["capabilities"]["clone"] == ["speaker_reference"]

    vevo = load_family_contract(str(source), "vevo2")
    assert vevo is not None
    assert "vevo2.whisper_model_path" in [
        row["option_key"] for row in vevo["dependencies"]
    ]

    oute = load_family_contract(str(source), "outetts")
    assert oute is not None
    assert oute["typed"] is True
    assert "outetts.aligner_path" in [
        row["option_key"] for row in oute["dependencies"]
    ]
