"""Diagnostics keep proxy health separate from runtime age and redact secrets."""

import asyncio

import pytest

from backend.tests.test_api_routes import _install_temp_store


def test_unreachable_runtime_keeps_the_health_timestamp(client):
    from backend.proxy.llama_swap import runtime_observation as observation

    observation.clear_runtime_observation()
    payload = client.get("/api/status").json()
    health = payload["proxy_status"]["health_observed_at"]
    assert health
    assert payload["proxy_status"]["observed_at"] == health
    runtime = payload["runtime_observation"]
    assert runtime["quality"] == "unreachable"
    assert runtime["observed_at"] is None
    assert runtime["detail"] == "No successful running-model observation yet."
    assert health != runtime["observed_at"]


def test_old_runtime_snapshot_can_accompany_a_fresh_health_check(client):
    from backend.proxy.llama_swap import runtime_observation as observation

    observation.remember_running_models({"running": [{"model": "demo", "state": "running"}]})
    with observation._lock:
        observation._snapshot["observed_at"] = "2020-01-01T00:00:00+00:00"
    payload = client.get("/api/status").json()
    assert payload["proxy_status"]["health_observed_at"].startswith("20")
    assert not payload["proxy_status"]["health_observed_at"].startswith("2020-")
    runtime = payload["runtime_observation"]
    assert runtime["quality"] == "verified"
    assert runtime["observed_at"] == "2020-01-01T00:00:00+00:00"
    assert runtime["age_seconds"] > 60
    assert runtime["detail"] is None
    observation.clear_runtime_observation()


def test_saturation_is_recorded_when_the_queue_rejects_a_write():
    from backend import store_io

    store_io.clear_persistence_events()
    acquired = 0

    async def raise_busy():
        nonlocal acquired
        while store_io._acquire_slot(blocking=False):
            acquired += 1
        with pytest.raises(store_io.StoreIoBusy):
            store_io.run_store(lambda: None)

    try:
        asyncio.run(raise_busy())
        status = store_io.persistence_status()
        assert status["pending_store_writes"] == store_io.MAX_PENDING_STORE_WRITES
        assert status["max_pending_store_writes"] == store_io.MAX_PENDING_STORE_WRITES
        assert status["saturated"] is True
        event = status["recent_saturation"][-1]
        assert event["code"] == "STORE_QUEUE_FULL"
        assert event["category"] == "saturation"
        assert "message" not in event
        assert "exception_type" not in event
    finally:
        for _ in range(acquired):
            store_io._release_slot()
        store_io.clear_persistence_events()


def test_failed_store_write_is_remembered_without_a_document():
    from backend import store_io

    store_io.clear_persistence_events()

    def boom():
        raise RuntimeError("disk full hf_MESSAGESECRET")

    with pytest.raises(RuntimeError):
        store_io.run_store(boom)
    failure = store_io.persistence_status()["latest_failure"]
    assert failure["code"] == "PERSISTENCE_FAILED"
    assert failure["category"] == "unknown"
    assert failure["description"] == "A persistence error occurred. Its details were not exported."
    assert "message" not in failure
    assert "exception_type" not in failure
    assert "hf_MESSAGESECRET" not in str(failure)
    assert "document" not in failure
    store_io.clear_persistence_events()


def test_durability_event_keeps_the_phase_and_drops_the_os_text():
    from backend import store_io
    from backend.store_io import StoreDurabilityError

    store_io.clear_persistence_events()
    store_io._record_persistence_event(
        "failure",
        StoreDurabilityError(
            "No space left hf_PHASESECRET cmake -DFOO=1",
            phase="fsync",
            committed=False,
        ),
    )
    failure = store_io.persistence_status()["latest_failure"]
    assert failure["code"] == "STORE_WRITE_FAILED"
    assert failure["category"] == "durability"
    assert failure["phase"] == "fsync"
    assert failure["committed"] is False
    assert "unchanged" in failure["description"]
    assert "hf_PHASESECRET" not in str(failure)
    assert "cmake" not in str(failure)
    store_io.clear_persistence_events()


def test_an_unknown_phase_is_not_exported():
    from backend import store_io
    from backend.store_io import StoreDurabilityError

    store_io.clear_persistence_events()
    store_io._record_persistence_event(
        "failure",
        StoreDurabilityError("leaked", phase="cmake -DSECRET=1", committed="unknown"),
    )
    failure = store_io.persistence_status()["latest_failure"]
    assert "phase" not in failure
    assert failure["committed"] == "unknown"
    assert "could not be established" in failure["description"]
    assert "leaked" not in str(failure)
    store_io.clear_persistence_events()


def test_persistence_event_ring_stays_bounded():
    from backend import store_io

    store_io.clear_persistence_events()
    try:
        for index in range(20):
            store_io._record_persistence_event("failure", RuntimeError(f"secret-{index}"))
        assert len(store_io._EVENTS) == 16
        status = store_io.persistence_status()
        assert len(status["recent_failures"]) == 8
        assert "secret-" not in str(status)
        assert status["latest_failure"]["code"] == "PERSISTENCE_FAILED"
    finally:
        store_io.clear_persistence_events()


def test_bundle_redacts_nested_settings_urls_and_failure_messages(client, monkeypatch, tmp_path):
    from backend import store_io
    from backend.proxy.llama_swap import runtime_observation as observation

    store = _install_temp_store(monkeypatch, tmp_path)
    store.update_settings(
        {
            "proxy_port": 2345,
            "huggingface_token": "hf_DROPPEDTOKEN",
            "public_inference_url": "https://alice:PLANTED_USERINFO@infer.example.test/v1?api_key=URLSECRETVALUE",
            "build_type": "Release",
            "studio_options": {
                "label": "studio",
                "note": "prefix hf_NESTEDSECRETVALUE",
                "headers": {"Authorization": "Bearer PLANTED_HEADER_SECRET"},
                "command": "cmake -DSECRET=PLANTED_COMMAND_SECRET",
            },
        }
    )
    store_io.clear_persistence_events()
    store_io._record_persistence_event(
        "failure",
        RuntimeError(
            "Authorization: Bearer PLANTED_BEARER_SECRET "
            "Authorization: Basic PLANTED_BASIC_SECRET "
            "token: Bearer PLANTED_TOKEN_SECRET "
            "connection failed password=PLANTED_PASSWORD "
            "https://alice:PLANTED_PASSWORD@example.test/ write failed hf_MESSAGESECRET"
        ),
    )
    observation.clear_runtime_observation()

    response = client.get("/api/diagnostics/bundle")
    assert response.status_code == 200
    assert "attachment" in response.headers["content-disposition"]
    assert 'filename="studio-diagnostics.json"' in response.headers["content-disposition"]
    text = response.text
    for secret in (
        "URLSECRETVALUE",
        "NESTEDSECRETVALUE",
        "MESSAGESECRET",
        "DROPPEDTOKEN",
        "PLANTED_PASSWORD",
        "PLANTED_USERINFO",
        "PLANTED_BEARER_SECRET",
        "PLANTED_BASIC_SECRET",
        "PLANTED_TOKEN_SECRET",
        "PLANTED_HEADER_SECRET",
        "PLANTED_COMMAND_SECRET",
    ):
        assert secret not in text
    body = response.json()
    assert body["schema_version"] == 1
    assert body["settings"]["proxy_port"] == 2345
    assert body["settings"]["build_type"] == "Release"
    assert "huggingface_token" not in body["settings"]
    assert "studio_options" not in body["settings"]
    assert body["settings"]["public_inference_url"] == (
        "https://infer.example.test/v1?api_key=[redacted]"
    )
    failure = body["persistence"]["latest_failure"]
    assert failure["code"] == "PERSISTENCE_FAILED"
    assert failure["description"] == "A persistence error occurred. Its details were not exported."
    assert "message" not in failure
    assert "exception_type" not in failure
    assert body["proxy_status"]["health_observed_at"]
    assert body["runtime_observation"]["quality"] == "unreachable"
    assert "headers" not in body["proxy_status"]
    store_io.clear_persistence_events()


def test_permitted_strings_redact_bearer_basic_userinfo_and_assignments():
    from backend.diagnostics import build_diagnostics_bundle, redact_text

    text = redact_text(
        "Authorization: Bearer PLANTED_BEARER_SECRET "
        "Authorization: Basic PLANTED_BASIC_SECRET "
        "password=PLANTED_PASSWORD "
        "https://alice:PLANTED_USERINFO@example.test/v1"
    )
    assert text.count("Authorization: [redacted]") == 2
    assert "password=[redacted]" in text
    assert "https://example.test/v1" in text
    assert "PLANTED_BEARER_SECRET" not in text
    assert "PLANTED_BASIC_SECRET" not in text
    assert "PLANTED_PASSWORD" not in text
    assert "PLANTED_USERINFO" not in text

    bundle = build_diagnostics_bundle(
        proxy_status={
            "healthy": False,
            "headers": {"Authorization": "Bearer PLANTED_HEADER_SECRET"},
            "command": "curl -H 'Authorization: Bearer PLANTED_HEADER_SECRET'",
        },
        runtime_observation={
            "quality": "stale",
            "observed_at": "2020-01-01T00:00:00+00:00",
            "age_seconds": 90,
            "detail": "Authorization: Bearer PLANTED_BEARER_SECRET password=PLANTED_PASSWORD",
            "model_count": 0,
            "command": "cmake -DSECRET=PLANTED_COMMAND_SECRET",
            "states": {"demo": "running token=PLANTED_TOKEN_SECRET"},
        },
        settings={
            "proxy_port": 1,
            "studio_options": {"note": "hf_NESTEDSECRETVALUE"},
        },
    )
    dumped = str(bundle)
    for secret in (
        "PLANTED_HEADER_SECRET",
        "PLANTED_BEARER_SECRET",
        "PLANTED_PASSWORD",
        "PLANTED_COMMAND_SECRET",
        "PLANTED_TOKEN_SECRET",
        "NESTEDSECRETVALUE",
    ):
        assert secret not in dumped
    assert bundle["runtime_observation"]["quality"] == "stale"
    assert bundle["runtime_observation"]["detail"] == "Authorization: [redacted] password=[redacted]"
    assert "command" not in bundle["runtime_observation"]
    assert "states" not in bundle["runtime_observation"]
    assert "headers" not in bundle["proxy_status"]
    assert "studio_options" not in bundle["settings"]


def test_bundle_follows_management_access(monkeypatch):
    from fastapi.testclient import TestClient

    from backend.main import app

    monkeypatch.setenv("STUDIO_ACCESS_MODE", "remote")
    monkeypatch.setenv("STUDIO_API_TOKEN", "studio-test-token")
    with TestClient(app) as remote:
        assert remote.get("/api/diagnostics/bundle").status_code == 401
        allowed = remote.get(
            "/api/diagnostics/bundle",
            headers={"Authorization": "Bearer studio-test-token"},
        )
        assert allowed.status_code == 200
        assert allowed.json()["schema_version"] == 1

    monkeypatch.setenv("STUDIO_ACCESS_MODE", "local")
    monkeypatch.setattr("backend.access_policy.in_container", lambda: False)
    import httpx

    transport = httpx.ASGITransport(app=app, client=("10.9.8.7", 4000))

    async def denied_remote_in_local_mode():
        async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
            return (await client.get("/api/diagnostics/bundle")).status_code

    assert asyncio.run(denied_remote_in_local_mode()) == 403
