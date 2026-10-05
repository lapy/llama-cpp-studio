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
        assert status["recent_saturation"][-1]["exception_type"] == "StoreIoBusy"
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
    assert failure["exception_type"] == "RuntimeError"
    assert "hf_MESSAGESECRET" in failure["message"]
    assert "document" not in failure
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
            "studio_options": {
                "label": "studio",
                "note": "prefix hf_NESTEDSECRETVALUE",
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
    ):
        assert secret not in text
    body = response.json()
    assert body["settings"]["proxy_port"] == 2345
    assert "huggingface_token" not in body["settings"]
    assert body["settings"]["studio_options"]["label"] == "studio"
    assert body["settings"]["studio_options"]["note"] == "prefix [redacted]"
    assert body["settings"]["public_inference_url"] == (
        "https://infer.example.test/v1?api_key=[redacted]"
    )
    assert body["persistence"]["latest_failure"]["exception_type"] == "RuntimeError"
    failure_message = body["persistence"]["latest_failure"]["message"]
    assert "MESSAGESECRET" not in failure_message
    assert failure_message.count("Authorization: [redacted]") == 2
    assert "password=[redacted]" in failure_message
    assert "https://example.test/" in failure_message
    assert body["proxy_status"]["health_observed_at"]
    store_io.clear_persistence_events()
