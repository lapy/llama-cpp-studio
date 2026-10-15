"""Contract-derived audio.cpp package discovery (no heuristics)."""

import json

import pytest

from backend.engines.audio_cpp.discovery import build_discovery_index
from backend.engines.audio_cpp.integration import (
    TASKS_RUN_ALIAS,
    is_generic_route,
    operation_by_id,
    operation_for_native_path,
    operation_for_task,
    resolve_operation,
)
from backend.engines.audio_cpp.integration.composer import policy_from_declared_options
from backend.engines.audio_cpp.tracking import merge_settings, split_settings
from backend.audio.request_policy import (
    build_request_policy,
    validate_instructions_against_policy,
)


def test_declared_package_family_wins_without_any_scoring():
    index = build_discovery_index(
        packages=[
            {
                "id": "weird_id_not_matching",
                "family": "omnivoice",
                "standalone": True,
                "tasks": ["tts"],
                "modes": ["offline"],
                "description": "Authoritative package JSON",
                "source": {"kind": "huggingface_snapshot"},
                "installable": True,
            }
        ],
        families=["omnivoice"],
        family_tasks={"omnivoice": ["tts"]},
    )
    pkg = index.get("weird_id_not_matching")
    assert pkg is not None
    assert pkg.family == "omnivoice"
    assert pkg.discovery_source == "json"
    assert pkg.match_reason == "package_json"
    assert pkg.tasks == ["tts"]


def test_model_spec_package_list_declares_identity(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "future_asr.json").write_text(
        json.dumps(
            {
                "family": "future_asr",
                "tasks": ["gen"],
                "modes": ["streaming"],
                "packages": [{"id": "opaque-package-17"}],
            }
        )
    )
    index = build_discovery_index(
        packages=[{"id": "opaque-package-17"}, {"id": "future_asr_guessed"}],
        families=["future_asr"],
        source_path=str(tmp_path),
    )
    declared = index.get("opaque-package-17")
    assert declared.family == "future_asr"
    assert declared.tasks == ["gen"]
    assert declared.modes == ["streaming"]
    unknown = index.get("future_asr_guessed")
    assert unknown.family is None
    assert unknown.tasks == []


def test_undeclared_package_stays_unclassified():
    index = build_discovery_index(
        packages=[
            {
                "id": "qwen3_tts_0_6b_base",
                "required_files": ["config.json"],
                "source": {"kind": "huggingface_snapshot"},
                "installable": True,
            }
        ],
        families=["qwen3_tts"],
    )
    pkg = index.get("qwen3_tts_0_6b_base")
    assert pkg is not None
    assert pkg.family is None
    assert pkg.match_reason == "no_declared_family"


def test_declared_standalone_and_subcomponent_markers():
    index = build_discovery_index(
        packages=[
            {
                "id": "moss_tts_nano_100m",
                "family": "moss_tts",
                "description": "Framework-ready MOSS Nano",
                "source": {"kind": "composite_snapshot", "placements": []},
                "installable": True,
            },
            {
                "id": "moss_audio_tokenizer_nano",
                "family": "moss_tts",
                "description": "Subcomponent only. Use moss_tts_nano_100m.",
                "parent_package_id": "moss_tts_nano_100m",
                "source": {"kind": "huggingface_snapshot"},
                "installable": True,
            },
            {
                "id": "voxcpm2_audiovae",
                "family": "voxcpm2",
                "description": "Utility only.",
                "source": {"kind": "utility", "operation_kind": "pytorch_to_safetensors"},
                "installable": True,
            },
        ],
        families=["moss_tts", "voxcpm2"],
        family_tasks={"moss_tts": ["tts"]},
    )
    assert index.get("moss_tts_nano_100m").standalone is True
    assert index.get("moss_audio_tokenizer_nano").standalone is False
    assert index.get("moss_audio_tokenizer_nano").parent_package_id == "moss_tts_nano_100m"
    assert index.get("voxcpm2_audiovae").standalone is False


def test_task_resolution_is_a_deterministic_table():
    assert operation_for_task("tts").id == "speech"
    assert operation_for_task("asr").id == "transcriptions"
    assert operation_for_task("align").id == "alignments"
    assert operation_for_task("vc").id == "tasks_run"
    assert operation_for_task("brand-new-task").id == "tasks_run"
    assert operation_for_task("").id == "tasks_run"


def test_resolve_operation_prefers_engine_observed_endpoints():
    assert resolve_operation(
        task="vc", preferred_api_endpoint="/v1/audio/speech"
    ).id == "speech"
    assert resolve_operation(
        task="tts", family_endpoints=["/v1/audio/transcriptions"]
    ).id == "transcriptions"
    assert resolve_operation(preferred_api_endpoint="/v1/tasks/run").id == "tasks_run"
    # GET-only routes never win a request-resolution lookup.
    assert resolve_operation(task="tts", family_endpoints=["/v1/models"]).id == "speech"


def test_tasks_run_maps_to_llama_swap_audioapi_alias():
    tasks_run = operation_by_id("tasks_run")
    assert tasks_run.studio_path == TASKS_RUN_ALIAS == "/audioapi/v1/tasks/run"
    assert tasks_run.native_path == "/v1/tasks/run"
    # The upstream-only flag marks routes llama-swap cannot dispatch directly.
    assert is_generic_route("/v1/audio/alignments") is True
    assert is_generic_route("/v1/audio/speech") is False


def test_unknown_native_path_resolves_to_no_operation():
    assert operation_for_native_path("/v1/audio/brand-new") is None
    assert operation_for_native_path("/v1/audio/speech").id == "speech"


def test_instructions_policy_from_declared_options_only():
    assert policy_from_declared_options(None) == ""
    assert policy_from_declared_options(["weight_type", "language"]) == ""
    assert policy_from_declared_options(["caption"]) == "caption_option"
    assert policy_from_declared_options(["caption_temperature"]) == "caption_option"
    assert policy_from_declared_options(["instruction"]) == "openai_instruct"
    assert policy_from_declared_options(["instruct"]) == "openai_instruct"


def test_instruction_validation_policies():
    assert validate_instructions_against_policy("hello", policy="caption_option")
    assert validate_instructions_against_policy("hello", policy="text_prefix")
    assert validate_instructions_against_policy("hello", policy="none")


def test_upstream_vocabulary_validates_soft_tags():
    errors = validate_instructions_against_policy(
        "female, unknown_token",
        policy="soft_tags",
        vocabulary=["female", "british accent"],
    )
    assert errors
    assert "unknown_token" in errors[0]

    assert (
        validate_instructions_against_policy(
            "female, british accent",
            policy="soft_tags",
            vocabulary=["female", "british accent"],
        )
        == []
    )


def test_build_request_policy_prefers_inspect_endpoint_and_policy():
    policy = build_request_policy(
        task="tts",
        family="future_tts",
        inspection={
            "family": "future_tts",
            "tasks": [{"task": "tts"}],
            "preferred_api_endpoint": "/v1/audio/speech",
            "instructions_policy": "soft_tags",
            "instructions_vocabulary": ["calm", "bright"],
        },
    )
    assert policy["api_endpoint"] == "/v1/audio/speech"
    assert policy["operation_id"] == "speech"
    assert policy["instructions_policy"] == "soft_tags"
    assert policy["instructions_policy_source"] == "engine"
    assert policy["instructions_vocabulary"] == ["calm", "bright"]


def test_build_request_policy_vevo2_like():
    policy = build_request_policy(
        task="tts",
        family="vevo2",
        inspection={"tasks": [{"task": "tts"}, {"task": "vc"}]},
        help_option_keys=["task-route", "source-audio"],
    )
    assert policy["api_endpoint"] == "/v1/audio/speech"
    assert policy["request_defaults_key"] == "speech_defaults"


def test_build_request_policy_ignores_inspect_tasks_from_other_family():
    policy = build_request_policy(
        task="gen",
        family="ace_step",
        inspection={
            "family": "omnivoice",
            "tasks": [{"task": "tts"}],
        },
    )
    assert policy["api_endpoint"] == TASKS_RUN_ALIAS
    assert policy["request_defaults_key"] == "task_defaults"
    assert "tts" not in policy["inspection_tasks"]


def test_inspected_conversion_task_uses_tasks_run():
    policy = build_request_policy(
        task="vc",
        family="chatterbox",
        inspection={
            "family": "chatterbox",
            "tasks": [{"task": "tts"}, {"task": "vc"}],
        },
        model_profile={"sections": []},
    )
    assert policy["api_endpoint"] == TASKS_RUN_ALIAS
    assert policy["request_defaults_key"] == "task_defaults"
    assert "vc" in policy["inspection_tasks"]


def test_build_request_policy_uses_version_descriptor_family_endpoints():
    descriptor = {
        "schema": "audio-cpp.studio-contract/v0",
        "family_routing": {
            "future_tts": {"endpoints": ["/v1/audio/transcriptions"]},
        },
    }
    policy = build_request_policy(
        task="tts",
        family="future_tts",
        active_version={"contract": descriptor},
    )
    # Engine-observed family endpoints beat the deterministic task table.
    assert policy["api_endpoint"] == "/v1/audio/transcriptions"
    assert policy["endpoint_provenance"] == "loaders-json"


def test_tracking_settings_envelope_round_trip():
    merged = merge_settings(
        tracking_ref="release-0.3",
        repository_url="https://github.com/0xShug0/audio.cpp.git",
        build_config={"cuda": True, "jobs": 8},
        existing={},
    )
    tracking, cmake = split_settings(merged)
    assert tracking["tracking_ref"] == "release-0.3"
    assert tracking["repository_url"].endswith("audio.cpp.git")
    assert cmake["cuda"] is True
    assert cmake["backend"] == "cuda"
    assert cmake["jobs"] == 8
    assert "tracking_ref" not in cmake or True  # tracking keys also present in merged
    # cmake extract should not drop backend when tracking keys coexist
    assert set(cmake) >= {"backend", "build_type", "jobs"}


@pytest.mark.asyncio
async def test_ensure_tracking_settings_bootstraps_from_github(monkeypatch, tmp_path):
    from backend import data_store
    from backend.engines.audio_cpp.tracking import ensure_tracking_settings

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(
        "backend.engines.audio_cpp.tracking.resolve_bootstrap_tracking_ref",
        lambda: "release-from-github",
    )

    settings = await ensure_tracking_settings(store)
    assert settings["tracking_ref"] == "release-from-github"
    assert "audio.cpp" in settings["repository_url"]
    stored = store.get_engine_build_settings("audio_cpp")
    assert stored["tracking_ref"] == "release-from-github"
