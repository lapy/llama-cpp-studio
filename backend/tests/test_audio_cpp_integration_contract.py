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
    operations,
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
        assert {
            "speech",
            "transcriptions",
            "transcriptions_details",
            "alignments",
            "batch_transcriptions",
            "voices",
            "models",
            "health",
            "tasks_run",
        } <= ops

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
        assert (
            resolve_operation(
                task="tts",
                preferred_api_endpoint="/v1/audio/transcriptions",
                family_endpoints=["/v1/audio/speech"],
            ).id
            == "transcriptions"
        )
        assert (
            resolve_operation(
                task="asr",
                family_endpoints=["/v1/audio/speech"],
            ).id
            == "speech"
        )
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
        caps = _caps_from_loaders(
            {
                "loaders": [
                    {
                        "family": "future_family",
                        "tasks": [{"id": "tts"}],
                        "api_endpoints": ["/v1/future/route"],
                    }
                ],
            }
        )
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
            descriptor,
            family="omnivoice",
            task="vc",
            inspection=None,
        )
        assert composed["operation_id"] == "speech"
        # No observations: deterministic task table.
        composed = compose_model_contract(None, task="asr")
        assert composed["operation_id"] == "transcriptions"
        assert composed["endpoint_provenance"] == "studio"
        composed = compose_model_contract(
            descriptor,
            family="omnivoice",
            task="tts",
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
                "operations": [
                    {
                        "id": "speech",
                        "native_path": "/v1/audio/speech",
                        "method": "POST",
                    }
                ],
            },
        )
        assert descriptor["provenance"]["speech"] == "upstream"
        assert descriptor["upstream_contract"] is True


class TestV095Surface:
    """The documented v0.9.5 server surface is represented exactly once.

    Citations are ``app/server/api.md`` at tag ``v0.9.5`` (plus the server
    README for ``POST /v1/models/load``'s body) and live next to each registry
    entry. These tests pin the table a reviewer checks against that document:
    path, method, encodings, responses, error shape, task claims, whether
    Studio serves the route itself or only through /upstream, and whether a
    working Studio route exists at all (``served``).
    """

    EXPECTED = {
        # id: (native_path, studio_path, method, encodings, responses,
        #      upstream_only, served)
        "speech_live": (
            "/v1/audio/speech/live",
            "/v1/audio/speech/live",
            "POST",
            ["raw-pcm"],
            ["sse-events"],
            False,
            False,
        ),
        "transcriptions_live": (
            "/v1/audio/transcriptions/live",
            "/v1/audio/transcriptions/live",
            "POST",
            ["raw-pcm"],
            ["sse-events"],
            False,
            False,
        ),
        "models_load": (
            "/v1/models/load",
            "/v1/models/load",
            "POST",
            ["json"],
            ["json"],
            True,
            False,
        ),
        "tasks_stream": (
            "/v1/tasks/stream",
            "/v1/tasks/stream",
            "POST",
            ["json"],
            ["json", "sse-events"],
            True,
            True,
        ),
        "tasks_batch": (
            "/v1/tasks/batch",
            "/v1/tasks/batch",
            "POST",
            ["json"],
            ["sse-events"],
            True,
            True,
        ),
        "tasks_unload_models": (
            "/v1/tasks/unload_models",
            "/v1/tasks/unload_models",
            "POST",
            ["json"],
            ["json"],
            True,
            True,
        ),
        "tasks_unload_all_models": (
            "/v1/tasks/unload_all_models",
            "/v1/tasks/unload_all_models",
            "POST",
            [],
            ["json"],
            True,
            True,
        ),
        "recipes": ("/v1/recipes", "/v1/recipes", "GET", [], ["json"], True, False),
        "recipes_run": (
            "/v1/recipes/run",
            "/v1/recipes/run",
            "POST",
            ["json"],
            ["json", "sse-events"],
            True,
            False,
        ),
    }

    def test_every_documented_route_is_registered_exactly_once(self):
        by_native = {op.native_path: op for op in operations()}
        by_studio = {op.studio_path: op for op in operations()}
        assert len(by_native) == len(operations())
        assert len(by_studio) == len(operations())
        for op_id, expected in self.EXPECTED.items():
            op = operation_by_id(op_id)
            assert op is not None, op_id
            native, studio, method, encodings, responses, upstream_only, served = expected
            assert op.native_path == native
            assert op.studio_path == studio
            assert op.method == method
            assert list(op.encodings) == encodings
            assert list(op.responses) == responses
            assert op.upstream_only is upstream_only
            assert op.served is served
            assert op.provenance == "studio"
            # A transport/management route must never claim a task.
            assert op.serves_tasks == ()

    def test_registry_still_serves_the_pre_v095_table(self):
        assert {op.id for op in operations()} >= {
            "speech",
            "transcriptions",
            "transcriptions_details",
            "alignments",
            "batch_transcriptions",
            "voices",
            "models",
            "health",
            "tasks_run",
        }
        for op_id in (
            "speech",
            "transcriptions",
            "transcriptions_details",
            "alignments",
            "batch_transcriptions",
            "voices",
            "models",
            "health",
            "tasks_run",
        ):
            assert operation_by_id(op_id).served is True, op_id

    def test_served_flags_pin_the_working_surface(self):
        """``served`` is independent of ``upstream_only``.

        - tasks/stream & tasks/batch are served by management shims that
          themselves forward through /upstream/{model}.
        - live audio is nominally routed through the /v1/audio catch-all but
          cannot work through the buffered proxy dataplane.
        - models/load and recipes have no Studio route at all.
        """
        unserved = {op.id for op in operations() if not op.served}
        assert unserved == {
            "speech_live",
            "transcriptions_live",
            "models_load",
            "recipes",
            "recipes_run",
        }
        # Served management shims still reach the engine through /upstream.
        for op_id in ("tasks_stream", "tasks_batch", "tasks_unload_models"):
            op = operation_by_id(op_id)
            assert op.served is True
            assert op.upstream_only is True
        # Nominally routed does not mean served.
        live = operation_by_id("speech_live")
        assert live.served is False
        assert live.upstream_only is False
        # Registry-only routes have no shim.
        for op_id in ("models_load", "recipes", "recipes_run"):
            op = operation_by_id(op_id)
            assert op.served is False
            assert op.upstream_only is True
        # The serialized descriptor carries the flag for every operation.
        for op in operations():
            assert op.as_dict()["served"] is op.served

    def test_body_shapes_for_new_envelopes(self):
        assert operation_by_id("tasks_stream").as_dict()["body_shape"] == "tasks-run"
        assert operation_by_id("tasks_batch").as_dict()["body_shape"] == "tasks-batch"
        assert operation_by_id("speech_live").as_dict()["body_shape"] == "raw-pcm"
        assert operation_by_id("transcriptions_live").as_dict()["body_shape"] == "raw-pcm"
        # Management/metadata routes fall back to the flat-JSON schema shape.
        assert operation_by_id("tasks_unload_models").as_dict()["body_shape"] == "openai-json"
        assert operation_by_id("recipes").as_dict()["body_shape"] == "openai-json"

    def test_defaults_keys_are_deterministic_for_new_operations(self):
        # Both generic transports take the tasks/run request vocabulary.
        assert defaults_key_for(operation_by_id("tasks_stream")) == "task_defaults"
        assert defaults_key_for(operation_by_id("tasks_batch")) == "task_defaults"
        # Everything else inherits the documented schema fallback, not a claim
        # that the route persists defaults.
        for op_id in (
            "speech_live",
            "transcriptions_live",
            "models_load",
            "tasks_unload_models",
            "tasks_unload_all_models",
            "recipes",
            "recipes_run",
        ):
            assert defaults_key_for(operation_by_id(op_id)) == "task_defaults"

    def test_task_routing_is_unchanged_and_new_routes_never_claim_tasks(self):
        descriptor = compose_version_contract(_caps_from_loaders(V091_LIST_LOADERS_JSON))
        assert descriptor["task_routing"] == {
            "tts": "speech",
            "vdes": "speech",
            "clon": "speech",
            "asr": "transcriptions",
            "stt": "transcriptions",
            "transcribe_details": "transcriptions_details",
            "align": "alignments",
            "alignment": "alignments",
            "batch_transcribe": "batch_transcriptions",
            "vc": "tasks_run",
            "svc": "tasks_run",
            "s2s": "tasks_run",
            "vad": "tasks_run",
            "diar": "tasks_run",
            "sep": "tasks_run",
            "gen": "tasks_run",
            "spk": "tasks_run",
            "midi": "tasks_run",
            "turn": "tasks_run",
            "wake": "tasks_run",
            "cls": "tasks_run",
        }
        v095_ids = set(self.EXPECTED)
        assert not v095_ids & set(descriptor["task_routing"].values())
        ids = {op["id"] for op in descriptor["operations"]}
        assert v095_ids <= ids
        assert all(descriptor["provenance"][op_id] == "studio" for op_id in v095_ids)

    def test_resolution_honors_generic_stream_but_never_live_or_management(self):
        # A family advertising the streaming generic route gets that operation;
        # it takes the same request envelope as tasks/run.
        assert resolve_operation(preferred_api_endpoint="/v1/tasks/stream").id == "tasks_stream"
        assert resolve_operation(family_endpoints=["tasks/stream"]).id == "tasks_stream"
        # Transport variants and management routes are not model surfaces.
        for endpoint in (
            "/v1/audio/speech/live",
            "/v1/audio/transcriptions/live",
            "/v1/tasks/batch",
            "/v1/tasks/unload_models",
            "/v1/tasks/unload_all_models",
            "/v1/recipes",
            "/v1/recipes/run",
            "/v1/models/load",
        ):
            assert resolve_operation(preferred_api_endpoint=endpoint).id == "tasks_run", endpoint
        assert operation_for_task("vc").id == "tasks_run"

    def test_native_path_lookup_and_validation_for_new_routes(self):
        for op_id, expected in self.EXPECTED.items():
            native = expected[0]
            assert operation_for_native_path(native).id == op_id
            assert is_native_api_path(native) is True
        assert is_native_api_path("/v1/tasks/../unload_models") is False
        assert is_native_api_path("/v1/recipes?x=1") is False
        assert is_generic_route("/v1/tasks/stream") is True
        assert is_generic_route("/v1/audio/speech/live") is False

    def test_advertised_new_route_is_not_drift(self):
        from backend.engines.scan.help_parsers import parse_audio_cpp_loaders_json

        caps = parse_audio_cpp_loaders_json(
            {
                "loaders": [
                    {
                        "family": "future_v095",
                        "tasks": [{"id": "vc"}],
                        "api_endpoints": ["/v1/tasks/stream"],
                    }
                ],
            }
        )
        descriptor = compose_version_contract(caps)
        assert descriptor["drift_warnings"] == []
        assert descriptor["family_routing"]["future_v095"]["operation_id"] == "tasks_stream"
        assert descriptor["family_routing"]["future_v095"]["studio_path"] == "/v1/tasks/stream"


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
