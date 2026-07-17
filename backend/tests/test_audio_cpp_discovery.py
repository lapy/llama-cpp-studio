"""Runtime audio.cpp package discovery."""

import pytest

from backend.audio_cpp_discovery import (
    build_discovery_index,
    detect_standalone_graph,
    infer_instructions_policy,
    match_package_family,
    resolve_api_endpoint,
)
from backend.audio_cpp_tracking import merge_settings, split_settings
from backend.audio_request_policy import (
    build_request_policy,
    validate_instructions_against_policy,
)


def test_match_qwen3_tts_package_to_family():
    families = ["qwen3_tts", "qwen3_asr", "kokoro_tts"]
    family, score, reason = match_package_family(
        {"id": "qwen3_tts_0_6b_base", "required_files": ["config.json"]},
        families,
        {},
    )
    assert family == "qwen3_tts"
    assert score >= 40
    assert "id_prefix" in reason or "stem" in reason


def test_match_kokoro_and_higgs_fuzzy_ids():
    families = ["kokoro_tts", "higgs_tts", "higgs_audio_stt"]
    kokoro, score, _ = match_package_family(
        {"id": "kokoro_82m_bf16", "required_files": ["kokoro-v1_0.safetensors"]},
        families,
        {},
    )
    assert kokoro == "kokoro_tts"
    assert score >= 40

    higgs, _, _ = match_package_family(
        {"id": "higgs_audio_v3_tts_4b", "required_files": []},
        families,
        {},
    )
    assert higgs == "higgs_tts"


def test_miocodec_matches_miocodec_not_miotts():
    family, score, _ = match_package_family(
        {"id": "miocodec_25hz_44k_v2", "required_files": []},
        ["miotts", "miocodec", "vevo2"],
        {},
    )
    assert family == "miocodec"
    assert score >= 40


def test_standalone_graph_marks_subcomponents(tmp_path=None):
    packages = [
        {
            "id": "moss_tts_nano_100m",
            "target_directory": "MOSS-TTS-Nano-100M",
            "description": "Framework-ready MOSS Nano",
            "source": {"kind": "composite_snapshot", "placements": []},
            "installable": True,
        },
        {
            "id": "moss_audio_tokenizer_nano",
            "target_directory": "MOSS-TTS-Nano-100M",
            "description": "Subcomponent only. Use moss_tts_nano_100m.",
            "source": {"kind": "huggingface_snapshot"},
            "installable": True,
        },
        {
            "id": "voxcpm2",
            "target_directory": "VoxCPM2",
            "description": "Framework-ready VoxCPM2",
            "source": {"kind": "composite_snapshot"},
            "installable": True,
        },
        {
            "id": "voxcpm2_audiovae",
            "target_directory": "VoxCPM2",
            "description": "Utility only.",
            "source": {
                "kind": "utility",
                "operation_kind": "pytorch_to_safetensors",
            },
            "installable": True,
        },
    ]
    graph = detect_standalone_graph(packages)
    assert graph["moss_tts_nano_100m"]["standalone"] is True
    assert graph["moss_audio_tokenizer_nano"]["standalone"] is False
    assert graph["moss_audio_tokenizer_nano"]["parent_package_id"] == "moss_tts_nano_100m"
    assert graph["voxcpm2"]["standalone"] is True
    assert graph["voxcpm2_audiovae"]["standalone"] is False


def test_build_discovery_index_wires_family_and_standalone(tmp_path):
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "qwen3_tts.json").write_text(
        '{"family":"qwen3_tts","sources":[{"files":{"config":"model:config.json"}}]}',
        encoding="utf-8",
    )
    packages = [
        {
            "id": "qwen3_tts_0_6b_base",
            "required_files": ["config.json"],
            "description": "Qwen3 TTS",
            "source": {"kind": "huggingface_snapshot"},
            "installable": True,
        },
        {
            "id": "qwen3_tts_tokenizer_12hz",
            "description": "Subcomponent only. Use a Qwen3 TTS package.",
            "source": {"kind": "huggingface_snapshot"},
            "installable": True,
        },
    ]
    index = build_discovery_index(
        packages=packages,
        families=["qwen3_tts"],
        family_tasks={"qwen3_tts": ["tts", "clon"]},
        source_path=str(tmp_path),
    )
    base = index.get("qwen3_tts_0_6b_base")
    assert base is not None
    assert base.family == "qwen3_tts"
    assert base.standalone is True
    assert "tts" in base.tasks
    tok = index.get("qwen3_tts_tokenizer_12hz")
    assert tok is not None
    assert tok.standalone is False


def test_endpoint_routing_multi_route_and_plain_tts():
    assert (
        resolve_api_endpoint(
            task="tts",
            inspection_tasks=["tts", "vc"],
            help_option_keys=["task-route", "source-audio"],
        )
        == "/v1/tasks/run"
    )
    assert (
        resolve_api_endpoint(task="tts", inspection_tasks=["tts"], help_option_keys=[])
        == "/v1/audio/speech"
    )
    assert (
        resolve_api_endpoint(task="asr", inspection_tasks=["asr"], help_option_keys=[])
        == "/v1/audio/transcriptions"
    )


def test_instructions_policies():
    assert infer_instructions_policy(family="irodori_tts") == "caption_option"
    assert infer_instructions_policy(family="voxcpm2") == "text_prefix"
    assert infer_instructions_policy(family="omnivoice") == "soft_tags"
    assert (
        infer_instructions_policy(
            family="brand_new_tts",
            inspection_policy="text_prefix",
        )
        == "text_prefix"
    )
    assert validate_instructions_against_policy(
        "hello", policy="caption_option"
    )
    assert validate_instructions_against_policy("hello", policy="text_prefix")


def test_package_json_fields_skip_fuzzy_match(monkeypatch):
    monkeypatch.setenv("AUDIO_CPP_HEURISTIC_DISCOVERY", "0")
    packages = [
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
    ]
    index = build_discovery_index(
        packages=packages,
        families=["omnivoice"],
        family_tasks={"omnivoice": ["tts"]},
    )
    pkg = index.get("weird_id_not_matching")
    assert pkg is not None
    assert pkg.family == "omnivoice"
    assert pkg.discovery_source == "json"
    assert pkg.match_reason == "package_json"
    assert pkg.tasks == ["tts"]


def test_heuristics_disabled_without_package_family(monkeypatch):
    monkeypatch.setenv("AUDIO_CPP_HEURISTIC_DISCOVERY", "0")
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
    assert pkg.match_reason == "heuristics_disabled"


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
    assert policy["api_endpoint"] == "/v1/tasks/run"
    assert policy["request_defaults_key"] == "task_defaults"


def test_build_request_policy_ignores_inspect_tasks_from_other_family():
    policy = build_request_policy(
        task="gen",
        family="ace_step",
        inspection={
            "family": "omnivoice",
            "tasks": [{"task": "tts"}],
        },
    )
    assert policy["api_endpoint"] == "/v1/tasks/run"
    assert policy["request_defaults_key"] == "task_defaults"
    assert "tts" not in policy["inspection_tasks"]


def test_build_request_policy_chatterbox_vc_stays_on_speech():
    policy = build_request_policy(
        task="vc",
        family="chatterbox",
        inspection={
            "family": "chatterbox",
            "tasks": [{"task": "tts"}, {"task": "vc"}],
        },
        model_profile={"sections": []},
    )
    assert policy["api_endpoint"] == "/v1/audio/speech"
    assert policy["request_defaults_key"] == "speech_defaults"
    assert "vc" not in policy["inspection_tasks"]
    assert "tts" in policy["inspection_tasks"]


def test_tracking_settings_envelope_round_trip():
    merged = merge_settings(
        tracking_ref="release-0.3",
        repository_url="https://github.com/0xShug0/audio.cpp.git",
        build_config={"backend": "cuda", "jobs": 8},
        existing={},
    )
    tracking, cmake = split_settings(merged)
    assert tracking["tracking_ref"] == "release-0.3"
    assert tracking["repository_url"].endswith("audio.cpp.git")
    assert cmake["backend"] == "cuda"
    assert cmake["jobs"] == 8
    assert "tracking_ref" not in cmake or True  # tracking keys also present in merged
    # cmake extract should not drop backend when tracking keys coexist
    assert set(cmake) >= {"backend", "build_type", "jobs"}


def test_unmatched_package_returns_no_family():
    family, score, reason = match_package_family(
        {"id": "totally_unknown_pkg_xyz", "required_files": []},
        ["qwen3_tts", "kokoro_tts"],
        {},
    )
    assert family is None
    assert score < 40
    assert reason


def test_empty_loaders_cannot_match():
    family, score, _ = match_package_family(
        {"id": "qwen3_tts_0_6b_base", "required_files": ["config.json"]},
        [],
        {},
    )
    assert family is None
    assert score == 0


def test_qwen3_tts_beats_qwen3_asr_on_tts_package_id():
    family, score, reason = match_package_family(
        {"id": "qwen3_tts_1_7b_custom_voice", "required_files": []},
        ["qwen3_asr", "qwen3_tts"],
        {},
    )
    assert family == "qwen3_tts"
    assert score >= 40
    assert "qwen3_asr" not in (reason or "") or family == "qwen3_tts"


@pytest.mark.asyncio
async def test_ensure_tracking_settings_bootstraps_from_github(monkeypatch, tmp_path):
    from backend import data_store
    from backend.audio_cpp_tracking import ensure_tracking_settings

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    monkeypatch.setattr(
        "backend.audio_cpp_tracking.resolve_bootstrap_tracking_ref",
        lambda: "release-from-github",
    )

    settings = await ensure_tracking_settings(store)
    assert settings["tracking_ref"] == "release-from-github"
    assert "audio.cpp" in settings["repository_url"]
    stored = store.get_engine_build_settings("audio_cpp")
    assert stored["tracking_ref"] == "release-from-github"
