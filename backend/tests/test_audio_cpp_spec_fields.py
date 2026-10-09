"""Engine metadata must support new families without Studio code changes."""

import copy
import json

import pytest

from backend.audio.request_defaults_validation import validate_saved_request_defaults
from backend.audio.task_profiles import request_field_groups_for
from backend.engines.audio_cpp.contracts import load_family_contract
from backend.engines.audio_cpp.spec_fields import merge_spec_contract_into_sections
from backend.engines.scan.scanner import audio_cpp_model_profile_fingerprint


@pytest.fixture
def spec_tree(tmp_path):
    directory = tmp_path / "model_specs"
    directory.mkdir()
    payload = {
        "family": "future_synth",
        "display_name": "Future Synth",
        "tasks": ["tts"],
        "modes": ["offline"],
        "options": {
            "request": [
                {"name": "variance", "type": "float", "min": 0, "max": 2, "default": 0},
                {"name": "strategy", "type": "enum", "values": ["fast", "precise"]},
                {"name": "sample", "type": "bool", "default": False},
            ],
            "session": [{"name": "budget", "type": "int", "min": 1}],
            "load": [{"name": "precision", "type": "enum", "values": ["f16", "f32"]}],
        },
    }
    path = directory / "future_synth.json"
    path.write_text(json.dumps(payload))
    return tmp_path, path, payload


def test_spec_overrides_help_types_constraints_without_mutating_input(spec_tree):
    root, _, _ = spec_tree
    sections = [
        {
            "id": "request",
            "params": [
                {
                    "key": "variance",
                    "scope": "request_option",
                    "type": "int",
                    "default": 7,
                    "minimum": 5,
                    "maximum": 9,
                    "options": [{"value": "stale"}],
                    "value_kind": "enum",
                }
            ],
        }
    ]
    original = copy.deepcopy(sections)
    result = merge_spec_contract_into_sections(
        sections, load_family_contract(str(root), "future_synth")
    )
    rows = {(p["scope"], p["key"]): p for s in result for p in s["params"]}
    variance = rows[("request_option", "variance")]
    assert variance["type"] == "float"
    assert variance["default"] == 0
    assert variance["minimum"] == 0
    assert variance["maximum"] == 2
    assert "options" not in variance
    assert variance["value_kind"] == "scalar"
    assert rows[("load_option", "future_synth.precision")]["type"] == "select"
    assert rows[("session_option", "future_synth.budget")]["emission"]["flag"] == "--session-option"
    assert sections == original


def test_request_fields_merge_help_gaps_and_prefer_spec(spec_tree):
    root, _, _ = spec_tree
    groups = request_field_groups_for(
        "tts",
        "future_synth",
        source_path=str(root),
        profile_sections=[
            {
                "params": [
                    {"key": "variance", "scope": "request_option", "type": "string"},
                    {"key": "new_switch", "scope": "request_option", "type": "bool"},
                ],
            }
        ],
    )
    fields = {f["key"]: f for g in groups for f in g["fields"]}
    assert fields["variance"]["type"] == "float"
    assert fields["sample"]["default"] is False
    assert fields["new_switch"]["type"] == "bool"
    assert [o["value"] for o in fields["strategy"]["options"]] == ["fast", "precise"]


@pytest.mark.parametrize(
    "options, message",
    [
        ({"variance": 3}, "at most 2"),
        ({"variance": float("nan")}, "finite"),
        ({"variance": True}, "must be float"),
        ({"strategy": "obsolete"}, "unsupported"),
        ({"sample": "yes"}, "must be bool"),
        ([], "must be an object"),
    ],
)
def test_request_defaults_validate_discovered_schema(spec_tree, options, message):
    root, _, _ = spec_tree
    errors = validate_saved_request_defaults(
        task="tts",
        family="future_synth",
        source_path=str(root),
        config={"speech_defaults": {"options": options}},
    )
    assert any(message in error for error in errors)


def test_request_defaults_remain_partial_and_preserve_unknown_options(spec_tree):
    root, _, _ = spec_tree
    assert (
        validate_saved_request_defaults(
            task="tts",
            family="future_synth",
            source_path=str(root),
            config={"speech_defaults": {"options": {"sample": False, "variance": 0, "future": 42}}},
        )
        == []
    )


def test_request_numeric_strings_follow_schema_not_option_name(spec_tree):
    root, _, _ = spec_tree
    config = {"speech_defaults": {"options": {"variance": "1.25"}}}
    assert (
        validate_saved_request_defaults(
            task="tts",
            family="future_synth",
            source_path=str(root),
            config=config,
        )
        == []
    )
    assert config["speech_defaults"]["options"]["variance"] == 1.25


def test_profile_cache_tracks_specs_ui_and_load_configuration(spec_tree):
    root, path, payload = spec_tree
    version = {"version": "v1", "source_path": str(root)}
    model = {
        "family": "future_synth",
        "config": {"engine": "audio_cpp", "engines": {"audio_cpp": {}}},
    }
    first = audio_cpp_model_profile_fingerprint(version, model)
    payload["ui"] = {"default_voice": "new_voice"}
    path.write_text(json.dumps(payload))
    second = audio_cpp_model_profile_fingerprint(version, model)
    assert first != second
    model["config"]["engines"]["audio_cpp"]["load_options"] = {"future_synth.precision": "f32"}
    third = audio_cpp_model_profile_fingerprint(version, model)
    assert third != second
    model["config"]["engines"]["audio_cpp"]["speech_defaults"] = {"text": "unrelated"}
    assert audio_cpp_model_profile_fingerprint(version, model) == third


def test_invalid_spec_is_ignored_and_family_cannot_escape_spec_directory(tmp_path):
    directory = tmp_path / "model_specs"
    directory.mkdir()
    (directory / "broken.json").write_bytes(b"\xff")
    assert load_family_contract(str(tmp_path), "broken") is None
    (tmp_path / "outside.json").write_text('{"schema_version":1,"family":"outside"}')
    assert load_family_contract(str(tmp_path), "../outside") is None


def test_preset_enum_keeps_choices_expanded_by_engine_help(spec_tree):
    root, path, payload = spec_tree
    payload["options"]["request"] = [{"name": "strategy", "type": "enum", "preset": "future_modes"}]
    path.write_text(json.dumps(payload))
    sections = [
        {
            "params": [
                {
                    "key": "strategy",
                    "scope": "request_option",
                    "type": "select",
                    "options": [{"value": "new", "label": "New"}],
                    "transport": "key_value_option",
                },
                {
                    "key": "batch_manifest_out",
                    "scope": "request_option",
                    "transport": "request_only",
                },
            ]
        }
    ]
    merged = merge_spec_contract_into_sections(
        sections, load_family_contract(str(root), "future_synth")
    )
    assert merged[0]["params"][0]["type"] == "select"
    groups = request_field_groups_for(
        "tts", "future_synth", source_path=str(root), profile_sections=merged
    )
    fields = {f["key"]: f for g in groups for f in g["fields"]}
    assert fields["strategy"]["options"] == [{"value": "new", "label": "New"}]
    assert "batch_manifest_out" not in fields


def test_custom_spec_override_is_used_and_invalidates_cache(spec_tree):
    root, _, payload = spec_tree
    override = root / "custom.json"
    payload["options"]["request"][0]["max"] = 10
    override.write_text(json.dumps(payload))
    version = {"version": "v1", "source_path": str(root)}
    model = {
        "family": "future_synth",
        "config": {
            "engines": {
                "audio_cpp": {
                    "model_spec_override": str(override),
                }
            }
        },
    }
    first = audio_cpp_model_profile_fingerprint(version, model)
    assert load_family_contract(str(override), "future_synth")["options"]["request"][0]["max"] == 10
    assert load_family_contract(str(root / "model_specs"), "future_synth") is not None
    assert load_family_contract(str(override), "other_family") is None
    payload["options"]["request"][0]["max"] = 20
    override.write_text(json.dumps(payload))
    assert first != audio_cpp_model_profile_fingerprint(version, model)


def test_source_fallback_preserves_same_key_in_different_scopes():
    from backend.engines.audio_cpp.option_discovery import merge_discovered_options_into_sections

    sections = [{"params": [{"key": "future.shared", "scope": "load_option"}]}]
    merged = merge_discovered_options_into_sections(
        sections,
        [
            {"key": "future.shared", "scope": "session_option"},
        ],
    )
    assert {(p["scope"], p["key"]) for s in merged for p in s["params"]} == {
        ("load_option", "future.shared"),
        ("session_option", "future.shared"),
    }


def test_scan_discovers_new_spec_options_after_engine_edit(spec_tree, monkeypatch):
    from backend.engines.scan import scanner

    root, path, payload = spec_tree
    cli = root / "audiocpp_cli"
    cli.touch()
    model_path = root / "model"
    model_path.mkdir()
    version = {"version": "v1", "source_path": str(root), "cli_binary_path": str(cli)}
    model = {"family": "future_synth", "artifact": {"path": str(model_path)}}
    cache = {}
    calls = []
    monkeypatch.setenv("AUDIO_CPP_SOURCE_OPTION_DISCOVERY", "0")
    monkeypatch.setattr(
        scanner, "get_model_profile_entry", lambda _, engine, version, fp: cache.get(fp)
    )
    monkeypatch.setattr(
        scanner,
        "upsert_model_profile_entry",
        lambda _, engine, version, fp, row: cache.update({fp: row}),
    )

    def inspect(*args, **kwargs):
        calls.append(args)
        return "family=future_synth\ntask=tts modes=offline\n", None

    monkeypatch.setattr(scanner, "_run_audio_cpp_inspect", inspect)
    monkeypatch.setattr(
        scanner,
        "_run_help_argv",
        lambda *a, **k: (
            "  Model request options:\n    variance <n>  Variance\n",
            None,
        ),
    )
    first = scanner.scan_audio_cpp_model_profile(None, version, model)
    assert not first.get("scan_error")
    assert scanner.scan_audio_cpp_model_profile(None, version, model) == first
    assert len(calls) == 1
    payload["options"]["session"].append({"name": "new_option", "type": "bool"})
    path.write_text(json.dumps(payload))
    second = scanner.scan_audio_cpp_model_profile(None, version, model)
    assert not second.get("scan_error")
    assert len(calls) == 2
    assert any(
        p["key"] == "future_synth.new_option" for s in second["sections"] for p in s["params"]
    )


def test_workspace_fields_preserve_transport_and_ignore_cli_utilities():
    from backend.engines.audio_cpp.spec_fields import workspace_request_fields

    contract = {"family": "future", "options": {"request": [
        {"name": "text", "type": "string", "required": True},
        {"name": "sample", "type": "bool", "default": False},
        {"name": "variance", "type": "float", "default": 0, "min": 0},
    ]}}
    sections = [
        {"id": "available_input_options", "params": [
            {"key": "source_audio", "scope": "request_option", "transport": "request_only"},
        ]},
        {"id": "utility", "params": [
            {"key": "help", "scope": "request_option", "transport": "request_only"},
        ]},
        {"params": [{"key": "threads", "scope": "model"}]},
    ]
    fields = {field["key"]: field for field in workspace_request_fields(contract, sections)}
    assert set(fields) == {"text", "source_audio", "sample", "variance"}
    assert fields["text"]["required"] is True
    assert fields["text"]["nested"] is False
    assert fields["source_audio"]["nested"] is False
    assert fields["sample"]["nested"] is True
    assert fields["sample"]["default"] is False
    assert fields["variance"]["default"] == 0
    assert workspace_request_fields(None) == []


def test_instruction_policy_requires_declared_option_not_family_or_docs(spec_tree):
    from backend.audio.request_policy import build_request_policy
    from backend.engines.audio_cpp.discovery import infer_instructions_policy

    root, path, payload = spec_tree
    assert infer_instructions_policy(family="future", docs_text="Speech docs") == "none"
    assert build_request_policy(task="tts", family="future_synth", source_path=str(root))["instructions_policy"] == "none"
    payload["options"]["request"].append({"name": "instruction", "type": "string"})
    path.write_text(json.dumps(payload))
    policy = build_request_policy(task="tts", family="future_synth", source_path=str(root))
    assert policy["instructions_policy"] == "openai_instruct"
    assert policy["instructions_policy_source"] == "engine_options"


def test_conversion_profile_never_requires_session_voice(spec_tree):
    from backend.audio.task_profiles import overlay_task_profile_from_spec

    root, path, payload = spec_tree
    payload["ui"] = {"default_voice": "demo"}
    payload["tasks"] = ["tts", "vc"]
    path.write_text(json.dumps(payload))
    assert overlay_task_profile_from_spec({}, str(root), "future_synth", task="tts")["requires_session_voice"] is True
    assert overlay_task_profile_from_spec({}, str(root), "future_synth", task="vc")["requires_session_voice"] is False
