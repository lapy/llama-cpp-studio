"""Memory-only request payload capture and replay."""

from types import SimpleNamespace

import pytest

from backend.proxy.studio.registry import (
    CAPTURE_MAX_BODY_BYTES,
    CAPTURE_MAX_REQUESTS,
    Generation,
    ProcessRegistry,
)
from backend.tests.test_api_routes import _install_temp_store


def _generation(model: str = "demo-model") -> Generation:
    process = SimpleNamespace(pid=123, poll=lambda: None)
    return Generation(model, "a" * 64, 3001, process, "fake")


def _registry() -> ProcessRegistry:
    async def _launch(_model_id):
        raise AssertionError("launch is not expected in capture tests")

    return ProcessRegistry(_launch)


def _record(registry: ProcessRegistry, request_id: str, **overrides) -> None:
    kwargs = {
        "status": 200,
        "latency_ms": 12.0,
        "input_bytes": 10,
        "output_bytes": 20,
        "method": "POST",
        "path": "/v1/chat/completions",
        "observation": {"id": request_id},
    }
    kwargs.update(overrides)
    registry.record_request(_generation(), **kwargs)


def test_capture_keeps_body_and_redacts_sensitive_headers():
    registry = _registry()
    _record(
        registry,
        "req-1",
        request_headers={
            "Authorization": "Bearer super-secret",
            "X-Api-Key": "key-secret",
            "X-Custom-API-Key": "vendor-secret",
            "X-Client-Trace": "trace-1",
            "Content-Type": "application/json",
            "Host": "studio.test",
            "User-Agent": "pytest",
        },
        request_body=b'{"model":"demo","messages":[]}',
    )

    payload = registry.replay_payload("req-1")
    assert payload is not None
    assert payload["model"] == "demo-model"
    assert payload["method"] == "POST"
    assert payload["path"] == "/v1/chat/completions"
    assert payload["body"] == '{"model":"demo","messages":[]}'
    assert payload["body_truncated"] is False
    assert payload["headers"] == {
        "content-type": "application/json",
        "user-agent": "pytest",
        "x-client-trace": "trace-1",
    }
    assert sorted(payload["redacted_headers"]) == [
        "authorization",
        "x-api-key",
        "x-custom-api-key",
    ]


def test_capture_is_bounded_and_truncates_large_bodies():
    registry = _registry()
    for index in range(CAPTURE_MAX_REQUESTS + 5):
        _record(registry, f"req-{index}", request_body=b"x")

    assert len(registry.replay_payloads) == CAPTURE_MAX_REQUESTS
    assert registry.replay_payload("req-0") is None
    assert registry.replay_payload(f"req-{CAPTURE_MAX_REQUESTS + 4}") is not None

    oversized = b"y" * (CAPTURE_MAX_BODY_BYTES + 10)
    _record(registry, "oversized", request_body=oversized)
    payload = registry.replay_payload("oversized")
    assert payload["body_truncated"] is True
    assert len(payload["body"].encode("utf-8")) == CAPTURE_MAX_BODY_BYTES


def test_capture_without_headers_or_body_is_not_stored():
    registry = _registry()
    _record(registry, "empty")
    assert registry.replay_payload("empty") is None


def test_payload_capture_setting_defaults_off_and_round_trips(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)

    initial = client.get("/api/proxy").json()
    assert initial["settings"]["capture_request_payloads"] is False

    saved = client.put(
        "/api/proxy/settings",
        json={"backend": "studio", "capture_request_payloads": True},
    )
    assert saved.status_code == 200
    assert saved.json()["settings"]["capture_request_payloads"] is True

    payload = client.get("/api/proxy").json()
    assert payload["settings"]["capture_request_payloads"] is True


class _FakeRegistry:
    def __init__(self, payloads=None, generation=None):
        self.replay_payloads = dict(payloads or {})
        self.activities = []
        self._generation = generation

    def replay_payload(self, request_id):
        return self.replay_payloads.get(request_id)

    def active(self, model_id):
        return self._generation if model_id else None


class _FakeManager:
    def __init__(self, registry):
        self.registry = registry


class _FakeResponse:
    status_code = 200
    content = b'{"ok": true}'
    headers = {"content-type": "application/json"}


class _FakeClient:
    calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False

    async def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return _FakeResponse()


PAYLOAD = {
    "id": "req-1",
    "model": "demo-model",
    "method": "POST",
    "path": "/v1/chat/completions",
    "headers": {"content-type": "application/json"},
    "body": '{"messages":[]}',
}


def test_overview_rows_receive_the_same_replay_fields(monkeypatch):
    from backend.routes.proxy_admin import _with_replay_payloads

    monkeypatch.setattr(
        "backend.routes.proxy_admin.get_studio_proxy_manager",
        lambda: _FakeManager(_FakeRegistry(payloads={"req-1": PAYLOAD})),
    )
    rows = _with_replay_payloads([{"id": "req-1"}, {"id": "req-2"}])
    assert rows[0]["replayable"] is True
    assert rows[0]["captured_body"] == '{"messages":[]}'
    assert rows[1] == {"id": "req-2"}


def test_request_list_marks_captured_payloads(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)

    class _Accounting:
        def flush(self, *_args):
            return None

        def query(self, **_kwargs):
            return {"total": 1, "requests": [{"id": "req-1"}, {"id": "req-2"}]}

    monkeypatch.setattr(
        "backend.routes.proxy_admin.get_studio_proxy_manager",
        lambda: _FakeManager(_FakeRegistry(payloads={"req-1": PAYLOAD})),
    )
    monkeypatch.setattr("backend.proxy.studio.accounting.get_accounting", lambda: _Accounting())

    response = client.get("/api/proxy/requests")
    assert response.status_code == 200
    rows = response.json()["requests"]
    assert rows[0]["replayable"] is True
    assert rows[0]["captured_body"] == '{"messages":[]}'
    assert rows[1].get("replayable") is None


def test_replay_posts_captured_payload_to_the_loaded_engine(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    generation = SimpleNamespace(upstream="http://127.0.0.1:3001")
    registry = _FakeRegistry(payloads={"req-1": PAYLOAD}, generation=generation)
    monkeypatch.setattr(
        "backend.routes.proxy_admin.get_studio_proxy_manager",
        lambda: _FakeManager(registry),
    )
    client_calls = []

    def _client_factory(**_kwargs):
        instance = _FakeClient()
        instance.calls = client_calls
        return instance

    monkeypatch.setattr("backend.routes.proxy_admin.httpx.AsyncClient", _client_factory)

    response = client.post("/api/proxy/requests/req-1/replay")
    assert response.status_code == 200
    payload = response.json()
    assert payload["replayed"] is True
    assert payload["status"] == 200
    assert payload["body"] == '{"ok": true}'
    assert registry.activities[-1]["type"] == "replay"
    assert registry.activities[-1]["status"] == 200
    assert registry.activities[-1]["request_id"] == "req-1"
    assert "body" not in registry.activities[-1]
    assert client_calls == [
        (
            "POST",
            "http://127.0.0.1:3001/v1/chat/completions",
            {"headers": {"content-type": "application/json"}, "content": b'{"messages":[]}'},
        )
    ]


def test_replay_rejects_missing_payload_or_unloaded_model(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    registry = _FakeRegistry(payloads={}, generation=None)
    monkeypatch.setattr(
        "backend.routes.proxy_admin.get_studio_proxy_manager",
        lambda: _FakeManager(registry),
    )

    missing = client.post("/api/proxy/requests/nope/replay")
    assert missing.status_code == 404
    assert missing.json()["detail"]["code"] == "PAYLOAD_NOT_CAPTURED"

    registry.replay_payloads["req-1"] = {**PAYLOAD, "model": "demo-model"}
    unloaded = client.post("/api/proxy/requests/req-1/replay")
    assert unloaded.status_code == 409
    assert unloaded.json()["detail"]["code"] == "MODEL_NOT_LOADED"
    assert registry.activities[-1]["type"] == "replay_failed"
    assert registry.activities[-1]["error"] == "model_not_loaded"


@pytest.mark.parametrize("method", ["GET", "POST"])
def test_capture_accepts_methods_without_bodies(method):
    registry = _registry()
    _record(registry, f"req-{method}", method=method, request_headers={"accept": "*/*"})
    payload = registry.replay_payload(f"req-{method}")
    assert payload["body"] is None
    assert payload["method"] == method
