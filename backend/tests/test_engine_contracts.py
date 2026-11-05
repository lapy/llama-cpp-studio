"""Structural pins: the typed contract mirrors validate real payloads.

Each literal below is transcribed from the CURRENT producer (file:line noted)
so a future delegation tranche that swaps a route's response_model for these
contracts cannot silently change a field. The models use extra="allow", so
these pins guard the TYPED fields, not exhaustiveness — exhaustiveness is
docs/engine-surface-contract-matrix.md's job.
"""

from __future__ import annotations

import pytest

from backend.models.engine_contracts import (
    AudioCheckUpdates,
    AudioContractResponse,
    AudioEngineStatus,
    CancelAccepted,
    CommitCheckUpdates,
    CommunityVoicesStatus,
    InstallAccepted,
    PypiCheckUpdates,
    ReleasesCheckUpdates,
    UnslothCheckUpdates,
    UpdateCheckResponse,
    VenvEngineStatus,
)


def test_pypi_check_updates_matches_factory_branch():
    # routes/engine_versions.py:69-75
    payload = {
        "latest_version": "0.7.1",
        "current_version": "0.7.0",
        "update_available": True,
        "releases": ["0.6.0", "0.7.0", "0.7.1"],
        "url": "https://pypi.org/project/lmdeploy/",
    }
    parsed = PypiCheckUpdates.model_validate(payload)
    assert parsed.model_dump() == payload


def test_commit_check_updates_has_no_releases_field():
    # routes/engine_versions.py:88-94
    payload = {
        "latest_version": "a" * 12,
        "latest_commit": "a" * 40,
        "current_version": "b" * 12,
        "update_available": True,
        "url": "https://github.com/sgl-project/sglang/commit/" + "a" * 40,
    }
    parsed = CommitCheckUpdates.model_validate(payload)
    assert "releases" not in parsed.model_dump()


def test_releases_check_updates_lacks_current_and_available():
    # routes/engine_versions.py:115-118 — the factory's THIRD shape.
    payload = {"latest_version": "1.2.3", "releases": ["1.2.2", "1.2.3"]}
    parsed = ReleasesCheckUpdates.model_validate(payload)
    # This variant declares NEITHER update_available NOR current_version —
    # the absence is the finding, so it must stay an absence in the model.
    assert not hasattr(parsed, "update_available")
    assert "current_version" not in ReleasesCheckUpdates.model_fields


def test_unsloth_check_updates_url_equals_release_url():
    # routes/unsloth_llama_versions.py:32-38
    url = "https://github.com/unslothai/llama.cpp/releases/tag/b9999"
    parsed = UnslothCheckUpdates.model_validate(
        {
            "latest_version": "b9999",
            "current_version": "b9998",
            "update_available": True,
            "url": url,
            "release_url": url,
        }
    )
    assert parsed.url == parsed.release_url


def test_llama_update_check_response_is_reexported():
    parsed = UpdateCheckResponse.model_validate(
        {
            "latest_release": {"tag_name": "b1234", "published_at": None, "html_url": None},
            "latest_commit": {"sha": "c" * 40, "commit_date": None, "message": None},
        }
    )
    assert parsed.latest_release.tag_name == "b1234"


def test_venv_status_fallback_fields_all_optional():
    # routes/engine_versions.py:184-197 (the exact fallback dict)
    payload = {
        "engine": "vllm",
        "installed": False,
        "version": None,
        "binary_path": None,
        "venv_path": None,
        "operation": None,
        "operation_started_at": None,
        "progress_task_id": None,
        "last_error": "boom",
        "install_type": None,
        "source_repo": None,
        "source_branch": None,
    }
    assert VenvEngineStatus.model_validate(payload).last_error == "boom"


def test_audio_status_drift_fields_typed():
    # routes/audio_cpp_versions.py:964-1004 (shape; values elided)
    payload = {
        "installed": True,
        "active": {"version": "release-0.9.4"},
        "runnable": True,
        "models_root": "/data/models/audio-cpp",
        "model_manager_ready": True,
        "tracking_ref": "release-0.9.4",
        "repository_url": "https://github.com/0xShug0/audio.cpp",
        "contract_fingerprint": "f" * 64,
        "contract_changed": False,
        "previous_contract_fingerprint": None,
        "capability_delta": {"added_families": ["tts"], "warnings": []},
        "families": ["tts"],
        "tasks": ["speech"],
        "discovery_source": "json",
        "catalog_source": "json",
        "contract_schema": "audio-cpp.studio-contract/v0",
        "drift_warnings": [
            "Engine advertises an endpoint Studio has no operation for: /x; "
            "requests fall through to /v1/tasks/run handling"
        ],
        "affected_models": [],
        "supported_build_backends": ["cpu", "cuda"],
    }
    parsed = AudioEngineStatus.model_validate(payload)
    assert parsed.capability_delta.added_families == ["tts"]
    # The real delta producer carries extra keys beyond the route fallback.
    assert (
        AudioEngineStatus.model_validate(
            {
                **payload,
                "capability_delta": {**payload["capability_delta"], "contract_changed": True},
            }
        ).capability_delta.contract_changed
        is True
    )


def test_audio_contract_response_envelope():
    # routes/audio_cpp_versions.py:1030-1036 + composer descriptor keys
    parsed = AudioContractResponse.model_validate(
        {
            "schema": "audio-cpp.studio-contract/v0",
            "operations": [{"id": "speech", "studio_path": "/audioapi/v1/audio/speech"}],
            "task_routing": {"speech": "speech"},
            "generic_studio_path": "/audioapi/v1/tasks/run",
            "families": ["tts"],
            "family_routing": {"tts": {"tasks": ["speech"], "endpoints": ["/v1/audio/speech"]}},
            "drift_warnings": [],
            "upstream_contract": False,
            "descriptor_fingerprint": "a" * 64,
            "engine_version": "release-0.9.4",
            "contract_fingerprint": "f" * 64,
            "contract_changed": False,
            "scanned_at": "2026-01-01T00:00:00Z",
        }
    )
    assert parsed.generic_studio_path == "/audioapi/v1/tasks/run"


def test_community_voices_shape():
    # backend/audio/community_voices.py:220-250
    parsed = CommunityVoicesStatus.model_validate(
        {
            "source": {
                "repository": "https://github.com/0xShug0/audio.cpp",
                "path": "webui/native/demo_voices",
                "checkout": "",
            },
            "installed": False,
            "voice_dir": None,
            "items": [
                {
                    "id": "voice_a",
                    "filename": "voice_a.wav",
                    "label": "Voice A",
                    "reference_text": "hello",
                    "size_bytes": 123,
                    "installed": False,
                    "path": "",
                    "voice_id": "voice_a",
                    "storage": "community",
                    "modified_at": None,
                }
            ],
        }
    )
    assert parsed.items[0].storage == "community"


@pytest.mark.parametrize(
    "payload",
    [
        # audio _schedule_prebuilt (routes/audio_cpp_versions.py:939-949)
        {
            "message": "Installing audio.cpp release-0.9.4",
            "task_id": "install_audio_cpp_release-0.9.4_1730000000",
            "status": "started",
            "version_name": "release-0.9.4",
            "source_ref": "release-0.9.4",
            "source_ref_type": "release",
            "prebuilt": True,
            "asset_name": "audio.tar.gz",
            "prebuilt_reason": "cuda 12 matches",
        },
        # audio _schedule_build (routes/audio_cpp_versions.py:651-658)
        {
            "message": "Building audio.cpp master",
            "task_id": "build_audio_cpp_master_1730000000",
            "status": "started",
            "version_name": "source-master-1730000000",
            "source_ref": "master",
            "source_ref_type": "branch",
        },
        # llama-family build scheduling extras ride the same core
        {
            "message": "Building",
            "task_id": "build_task_1",
            "status": "started",
            "commit_sha": "abc1234",
        },
    ],
)
def test_install_accepted_core_validates(payload):
    parsed = InstallAccepted.model_validate(payload)
    assert parsed.status == "started" and parsed.task_id == payload["task_id"]


def test_cancel_accepted_covers_build_task_manager_envelope():
    # backend/build_task_manager.py: {ok, terminated, message, task_id}
    parsed = CancelAccepted.model_validate(
        {
            "ok": True,
            "terminated": False,
            "message": "Cancellation was requested. The build has not been verified as stopped.",
            "task_id": "build_task_1",
        }
    )
    assert parsed.ok is True
