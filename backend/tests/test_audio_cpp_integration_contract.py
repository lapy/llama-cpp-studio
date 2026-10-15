"""Contract registry / composer / activation-gate / endpoint behavior.

Uses the captured v0.9.1 ``--list-loaders --json`` payload from
``audio_profile_fixtures`` as the engine-observation source.
"""

import pytest

from backend.engines.audio_cpp.integration import (
    CONTRACT_SCHEMA,
    DEFAULTS_KEYS,
    NATIVE_AUDIO_BASE,
    STUDIO_OWNED_FLAGS,
    TASKS_RUN_ALIAS,
    compose_model_contract,
    compose_version_contract,
    defaults_key_for,
    is_generic_route,
    is_native_api_path,
    operation_by_id,
    operation_for_native_path,
    operation_for_task,
    registry_descriptor,
    resolve_operation,
)
from backend.tests.audio_profile_fixtures import V091_LIST_LOADERS_JSON


def _caps_from_loaders(payload: dict) -> dict:
    """Capabilities shape persisted by the version scan for the JSON payload."""
    from backend.engines.scan.help_parsers import parse_audio_cpp_loaders_json

    return parse_audio_cpp_loaders_json(payload)


class TestRegistry:
    def test_registry_schema_and_operation_ids(self):
        ops = {op["id"] for op in registry_descriptor()}
        assert {"speech", "transcriptions", "transcriptions_details", "alignments",
                "batch_transcriptions", "voices", "models", "health",
                "tasks_run"} <= ops

    def test_operation_serialization_carries_defaults_key_and_body_shape(self):
        speech = operation_by_id("speech").as_dict()
        assert speech["request_defaults_key"] == "speech_defaults"
        assert speech["body_shape"] == "openai-json"
        asr = operation_by_id("transcriptions").as_dict()
        assert asr["request_defaults_key"] == "transcription_defaults"
        assert asr["body_shape"] == "multipart-form"
        generic = operation_by_id("tasks_run").as_dict()
        assert generic["request_defaults_key"] == "task_defaults"
        assert generic["body_shape"] == "tasks-run"

    def test_resolve_operation_prefers_preferred_then_family_then_task(self):
        assert resolve_operation(
            task="tts",
            preferred_api_endpoint="/v1/audio/transcriptions",
            family_endpoints=["/v1/audio/speech"],
        ).id == "transcriptions"
        assert resolve_operation(
            task="asr",
            family_endpoints=["/v1/audio/speech"],
        ).id == "speech"
        assert resolve_operation(task="tts").id == "speech"
        assert resolve_operation(task="vc").id == "tasks_run"
        assert resolve_operation(task="some_future_task").id == "tasks_run"

    def test_resolve_operation_accepts_coarse_engine_tokens(self):
        # v0.9.1 inspect reports coarse surface labels, not paths.
        assert resolve_operation(preferred_api_endpoint="tasks").id == "tasks_run"
        assert resolve_operation(preferred_api_endpoint="speech").id == "speech"
        assert resolve_operation(family_endpoints=["transcriptions"]).id == "transcriptions"
        assert resolve_operation(preferred_api_endpoint=TASKS_RUN_ALIAS).id == "tasks_run"
        # GET-only routes can never serve a model.
        assert resolve_operation(preferred_api_endpoint="voices").id == "tasks_run"

    def test_gets_only_routes_never_win_family_resolution(self):
        assert resolve_operation(family_endpoints=["/v1/models"]).id == "tasks_run"

    def test_defaults_key_table(self):
        assert defaults_key_for(operation_by_id("alignments")) == "task_defaults"
        assert DEFAULTS_KEYS["speech"] == "speech_defaults"

    def test_generic_route_flags_model_pinned_operations(self):
        assert is_generic_route("/v1/audio/alignments") is True
        assert is_generic_route("/v1/audio/speech") is False

    def test_native_path_validation(self):
        assert NATIVE_AUDIO_BASE == "/v1/audio"
        assert is_native_api_path("/v1/audio/speech") is True
        assert is_native_api_path("/v1/audio/whatever-new") is True
        assert is_native_api_path("/audioapi/v1/tasks/run") is False
        assert is_native_api_path("/v1/../etc/passwd") is False
        assert is_native_api_path("/v1/audio/speech?x=1") is False

    def test_operation_for_task_fallthrough(self):
        assert operation_for_task("tts").id == "speech"
        assert operation_for_task("unknown").id == "tasks_run"

    def test_operation_for_native_path(self):
        assert operation_for_native_path(TASKS_RUN_ALIAS) is None
        assert operation_by_id("tasks_run").studio_path == TASKS_RUN_ALIAS

    def test_studio_owned_flags_cover_runtime_injection(self):
        for flag in ("--host", "--port", "--model", "--backend", "--config"):
            assert flag in STUDIO_OWNED_FLAGS


class TestComposer:
    def test_version_descriptor_from_v091_capture(self):
        caps = _caps_from_loaders(V091_LIST_LOADERS_JSON)
        descriptor = compose_version_contract(caps)
        assert descriptor["schema"] == CONTRACT_SCHEMA
        routing = descriptor["family_routing"]
        assert routing["omnivoice"]["api_endpoint"] == "/v1/audio/speech"
        assert routing["omnivoice"]["operation_id"] == "speech"
        assert routing["omnivoice"]["request_defaults_key"] == "speech_defaults"
        assert routing["omnivoice"]["instructions_policy"] == "soft_tags"
        assert routing["qwen3_asr"]["operation_id"] == "transcriptions"
        assert routing["qwen3_asr"]["request_defaults_key"] == "transcription_defaults"
        assert descriptor["drift_warnings"] == []
        assert descriptor["task_routing"]["tts"] == "speech"
        assert descriptor["generic_studio_path"] == TASKS_RUN_ALIAS
        assert len(descriptor["descriptor_fingerprint"]) == 64

    def test_unknown_family_endpoint_becomes_drift_warning(self):
        caps = _caps_from_loaders({
            "loaders": [{
                "family": "future_family",
                "tasks": [{"id": "tts"}],
                "api_endpoints": ["/v1/future/route"],
            }],
        })
        descriptor = compose_version_contract(caps)
        assert any("/v1/future/route" in warning for warning in descriptor["drift_warnings"])

    def test_model_contract_resolution_order(self):
        caps = _caps_from_loaders(V091_LIST_LOADERS_JSON)
        descriptor = compose_version_contract(caps)
        # Inspect beats loaders.
        composed = compose_model_contract(
            descriptor,
            family="omnivoice",
            task="tts",
            inspection={"family": "omnivoice", "preferred_api_endpoint": "tasks"},
        )
        assert composed["operation_id"] == "tasks_run"
        assert composed["api_endpoint"] == TASKS_RUN_ALIAS
        assert composed["request_defaults_key"] == "task_defaults"
        assert composed["body_shape"] == "tasks-run"
        # Loaders beat the task table.
        composed = compose_model_contract(
            descriptor, family="omnivoice", task="vc", inspection=None,
        )
        assert composed["operation_id"] == "speech"
        # No observations: deterministic task table.
        composed = compose_model_contract(None, task="asr")
        assert composed["operation_id"] == "transcriptions"
        assert composed["endpoint_provenance"] == "studio"
        composed = compose_model_contract(
            descriptor, family="omnivoice", task="tts",
        )
        assert composed["endpoint_provenance"] == "loaders-json"

    def test_model_contract_declared_option_policy(self):
        composed = compose_model_contract(
            None,
            task="tts",
            declared_option_keys=["instructions", "temperature"],
        )
        assert composed["instructions_policy"] == "openai_instruct"
        assert composed["instructions_policy_source"] == "engine_options"
        composed = compose_model_contract(None, task="tts", declared_option_keys=[])
        assert composed["instructions_policy"] == "none"
        assert composed["instructions_policy_source"] == "unreported"

    def test_upstream_document_flips_provenance(self):
        descriptor = compose_version_contract(
            {},
            upstream={
                "operations": [{
                    "id": "speech",
                    "native_path": "/v1/audio/speech",
                    "method": "POST",
                }],
            },
        )
        assert descriptor["provenance"]["speech"] == "upstream"
        assert descriptor["upstream_contract"] is True


class TestActivationGate:
    def test_old_build_is_rejected(self, tmp_path, monkeypatch):
        from backend import data_store
        from backend.engines.audio_cpp.activation import assert_activation_supported
        from fastapi import HTTPException

        store = data_store.DataStore(config_dir=str(tmp_path / "config"))
        old = tmp_path / "audiocpp_cli_old"
        old.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        old.chmod(0o755)
        row = {
            "version": "v-old",
            "cli_binary_path": str(old),
            "server_binary_path": str(old),
        }
        store.add_engine_version(
            "audio_cpp",
            {
                "version": "v-old",
                "cli_binary_path": str(old),
                "server_binary_path": str(old),
            },
        )
        with pytest.raises(HTTPException) as excinfo:
            assert_activation_supported(store, row)
        assert excinfo.value.status_code == 400
        assert "v0.9.1" in str(excinfo.value.detail)

    def test_json_capable_build_passes(self, tmp_path):
        from backend import data_store
        from backend.engines.audio_cpp.activation import assert_activation_supported
        from backend.tests.audio_profile_fixtures import write_fake_v091_cli

        store = data_store.DataStore(config_dir=str(tmp_path / "config"))
        cli = write_fake_v091_cli(tmp_path / "audiocpp_cli")
        row = {
            "version": "v-new",
            "cli_binary_path": str(cli),
            "server_binary_path": str(cli),
        }
        store.add_engine_version(
            "audio_cpp",
            {
                "version": "v-new",
                "cli_binary_path": str(cli),
                "server_binary_path": str(cli),
            },
        )
        assert_activation_supported(store, row)  # no exception


class TestContractEndpoint:
    def test_409_without_active_contract(self, client, monkeypatch, tmp_path):
        from backend import data_store

        store = data_store.DataStore(config_dir=str(tmp_path / "config"))
        monkeypatch.setattr(data_store, "_store", store)
        r = client.get("/api/audio-cpp/contract")
        assert r.status_code == 409
        assert "v0.9.1" in r.json()["detail"]

    def test_returns_composed_descriptor(self, client, monkeypatch, tmp_path):
        from backend import data_store
        from backend.engines.params import upsert_version_entry
        from backend.tests.audio_profile_fixtures import fake_audio_binaries

        store = data_store.DataStore(config_dir=str(tmp_path / "config"))
        monkeypatch.setattr(data_store, "_store", store)
        server, cli = fake_audio_binaries(tmp_path)
        store.add_engine_version(
            "audio_cpp",
            {
                "version": "v-live",
                "server_binary_path": server,
                "cli_binary_path": cli,
            },
        )
        store.set_active_engine_version("audio_cpp", "v-live")
        caps = _caps_from_loaders(V091_LIST_LOADERS_JSON)
        upsert_version_entry(
            store,
            "audio_cpp",
            "v-live",
            {
                "contract": compose_version_contract(caps),
                "contract_fingerprint": "c" * 64,
                "contract_changed": False,
                "capabilities": caps,
            },
        )
        r = client.get("/api/audio-cpp/contract")
        assert r.status_code == 200
        data = r.json()
        assert data["schema"] == CONTRACT_SCHEMA
        assert data["engine_version"] == "v-live"
        assert data["contract_fingerprint"] == "c" * 64
        assert data["family_routing"]["omnivoice"]["operation_id"] == "speech"
        assert data["generic_studio_path"] == TASKS_RUN_ALIAS
        ids = {op["id"] for op in data["operations"]}
        assert "tasks_run" in ids
        # Operation rows carry the derived client hints.
        speech = next(op for op in data["operations"] if op["id"] == "speech")
        assert speech["request_defaults_key"] == "speech_defaults"
        assert speech["body_shape"] == "openai-json"
