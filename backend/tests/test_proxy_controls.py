import asyncio
from types import SimpleNamespace

import pytest

from backend.inference_auth import (
    authorize_headers,
    authorize_token,
    create_key,
    delete_key,
    verify_token,
)
from backend.proxy.studio.registry import Generation, ProcessRegistry, RegistryError
from backend.tests.test_api_routes import _install_temp_store


def test_scoped_expiring_rate_limited_key(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    created = client.post(
        "/api/inference/keys",
        json={
            "name": "restricted",
            "expires_at": "2099-01-01T00:00:00Z",
            "models": ["model-a"],
            "scopes": ["inference"],
            "rate_limit_per_minute": 1,
        },
    )
    assert created.status_code == 201
    payload = created.json()
    token = payload["key"]
    try:
        assert verify_token(token)
        assert authorize_token(token, model="model-b") is False
        assert authorize_token(token, model="model-a") is True
        assert authorize_token(token, model="model-a") is False
        listed = client.get("/api/inference/keys").json()["keys"][0]
        assert listed["models"] == ["model-a"]
        assert listed["expires_at"].startswith("2099-01-01")
        assert listed["rate_limit_per_minute"] == 1
        assert listed["last_used_at"]
    finally:
        delete_key(payload["id"])


def test_key_rejects_expired_or_unknown_scope(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    expired = client.post(
        "/api/inference/keys",
        json={"name": "expired", "expires_at": "2000-01-01T00:00:00Z"},
    )
    unknown = client.post(
        "/api/inference/keys",
        json={"name": "admin", "scopes": ["admin"]},
    )
    assert expired.status_code == 400
    assert unknown.status_code == 400


def test_anonymous_access_is_an_explicit_policy(monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.update_settings({"proxy": {"backend": "studio", "allow_anonymous": False}})
    assert authorize_headers({}) is False
    store.update_settings({"proxy": {"backend": "studio", "allow_anonymous": True}})
    assert authorize_headers({}) is True


def _generation(model: str) -> Generation:
    process = SimpleNamespace(pid=123, poll=lambda: None)
    return Generation(model, "a" * 64, 3001, process, "fake")


def test_runtime_metrics_include_latency_errors_and_traffic():
    registry = ProcessRegistry(lambda _: None)  # type: ignore[arg-type]
    generation = _generation("demo")
    registry.record_request(
        generation, status=200, latency_ms=10, input_bytes=100, output_bytes=200
    )
    registry.record_request(
        generation, status=500, latency_ms=30, input_bytes=50, output_bytes=10
    )
    row = registry.describe(generation)
    assert row["request_count"] == 2
    assert row["error_count"] == 1
    assert row["input_bytes"] == 150
    assert row["output_bytes"] == 210
    assert row["average_latency_ms"] == 20
    assert row["p50_latency_ms"] == 10
    assert row["p95_latency_ms"] == 30


@pytest.mark.asyncio
async def test_per_model_queue_is_bounded():
    registry = ProcessRegistry(
        lambda _: None, per_model_concurrency=1, queue_timeout=0.01  # type: ignore[arg-type]
    )
    await registry.enter_request("demo")
    with pytest.raises(RegistryError, match="queue timed out"):
        await registry.enter_request("demo")
    registry.leave_request("demo")
    await registry.enter_request("demo")
    registry.leave_request("demo")


@pytest.mark.asyncio
async def test_maximum_loaded_models_rejects_new_model():
    async def launch(_model):
        return None

    registry = ProcessRegistry(launch, max_loaded_models=1)  # type: ignore[arg-type]
    registry._active["one"] = _generation("one")
    with pytest.raises(RegistryError, match="maximum loaded models"):
        await registry.acquire("two")


def test_log_filter_search_and_levels():
    from backend.routes.proxy_admin import _filter_log_lines

    lines = [
        "stdout: server listening on 8080",
        "stderr: ggml_mmap: error mapping layer 7",
        "stdout: WARNING slow load detected",
    ]
    assert _filter_log_lines(lines, search="layer", level="") == [lines[1]]
    assert _filter_log_lines(lines, search="8080", level="error") == []
    assert _filter_log_lines(lines, search="", level="error") == [lines[1]]
    assert _filter_log_lines(lines, search="", level="warn") == [lines[1], lines[2]]
    assert _filter_log_lines(lines, search="", level="info") == lines
    assert _filter_log_lines(lines, search="no-match", level="") == []


def test_request_history_is_bounded_and_stats_reset_preserves_engines(monkeypatch):
    registry = ProcessRegistry(lambda _: None)
    generation = _generation("demo")
    registry._generations[123] = generation
    for index in range(105):
        registry.record_request(generation, status=200, latency_ms=index,
                                input_bytes=10, output_bytes=20,
                                method="POST", path="/v1/chat/completions")
    assert len(registry.requests.values()) == 100
    assert registry.requests.values()[0]["latency_ms"] == 5
    assert registry.requests.values()[-1]["path"] == "/v1/chat/completions"
    assert len({row["id"] for row in registry.requests.values()}) == 100
    registry.reset_stats()
    row = registry.describe(generation)
    assert row["request_count"] == row["output_bytes"] == 0
    assert row["p50_latency_ms"] is None
    assert len(registry.requests.values()) == 100
    assert registry._generations[123] is generation
    monkeypatch.setattr("backend.proxy.studio.registry.time.time", lambda: 9000)
    registry._emit("ready", generation)
    assert registry.activities.values()[-1]["timestamp"] == 9000
    assert registry.activities.values()[-1]["created_at"] == generation.created_at
