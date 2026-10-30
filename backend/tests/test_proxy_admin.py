import httpx

from backend.proxy.studio.ring import EventRing
from backend.tests.test_api_routes import _install_temp_store


def test_proxy_settings_are_visible_and_persisted(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)

    initial = client.get("/api/proxy")
    assert initial.status_code == 200
    assert initial.json()["settings"]["backend"] == "llama_swap"

    saved = client.put(
        "/api/proxy/settings",
        json={
            "backend": "studio",
            "idle_ttl_seconds": 600,
            "max_body_bytes": 1048576,
        },
    )
    assert saved.status_code == 200
    assert saved.json()["restart_required"] is True
    proxy = store.get_settings()["proxy"]
    assert proxy["backend"] == "studio"
    assert proxy["idle_ttl_seconds"] == 600.0
    assert proxy["max_body_bytes"] == 1048576
    assert proxy["allow_anonymous"] is True


def test_proxy_settings_validate_backend_and_limits(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    assert client.put("/api/proxy/settings", json={"backend": "other"}).status_code == 400
    assert (
        client.put(
            "/api/proxy/settings",
            json={"backend": "studio", "max_body_bytes": 0},
        ).status_code
        == 422
    )


def test_admin_get_reports_configured_and_active_backends(client, monkeypatch, tmp_path):
    """Configured backend and actually-running backend stay distinct."""
    _install_temp_store(monkeypatch, tmp_path)
    client.put("/api/proxy/settings", json={"backend": "studio"})
    payload = client.get("/api/proxy").json()
    assert payload["configured_backend"] == "studio"
    # Nothing was started in this test process: the truth is "not running".
    assert payload["active_backend"] == ""
    assert payload["proxy_running"] is False
    assert payload["studio_running"] is False
    assert isinstance(payload["catalog"], list)
    assert payload["network"]["port"] == payload["settings"]["proxy_port"]
    assert payload["network"]["listen_host"] in {"127.0.0.1", "0.0.0.0"}


def test_proxy_switch_persists_backend_and_drives_lifecycle(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    calls: list[str] = []

    class FakeStudio:
        running = False

        async def stop_proxy(self):
            calls.append("studio.stop")

        async def start_proxy(self):
            calls.append("studio.start")
            self.running = True

    class FakeSwap:
        process = None

        async def stop_proxy(self):
            calls.append("swap.stop")

    monkeypatch.setattr(
        "backend.routes.proxy_admin.get_studio_proxy_manager", lambda: FakeStudio()
    )
    monkeypatch.setattr(
        "backend.proxy.llama_swap.manager.get_llama_swap_manager", lambda: FakeSwap()
    )
    switched = client.post("/api/proxy/switch", json={"backend": "studio"})
    assert switched.status_code == 200
    assert switched.json() == {
        "backend": "studio",
        "stopped": [],
        "running": True,
        "restart_required": False,
    }
    assert store.get_settings()["proxy"]["backend"] == "studio"
    assert calls == ["studio.start"]
    assert client.put("/api/proxy/settings", json={"backend": "studio"}).json()[
        "restart_required"
    ] is False


def test_proxy_switch_rejects_unknown_backend_and_start_failures(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)

    class Broken:
        running = False

        async def stop_proxy(self):
            return None

        async def start_proxy(self):
            raise RuntimeError("port in use")

    monkeypatch.setattr("backend.routes.proxy_admin.get_studio_proxy_manager", lambda: Broken())
    assert client.post("/api/proxy/switch", json={"backend": "nope"}).status_code == 400
    response = client.post("/api/proxy/switch", json={"backend": "studio"})
    assert response.status_code == 502
    assert "port in use" in response.json()["detail"]


def test_lifecycle_dispatches_to_the_configured_backend(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.update_settings({"proxy": {"backend": "llama_swap"}})
    calls: list[str] = []

    class FakeSwap:
        process = None

        async def start_proxy(self):
            calls.append("start")

        async def stop_proxy(self):
            calls.append("stop")

        async def restart_proxy(self):
            calls.append("restart")

    monkeypatch.setattr(
        "backend.proxy.llama_swap.manager.get_llama_swap_manager", lambda: FakeSwap()
    )
    assert client.post("/api/proxy/lifecycle/start").json() == {
        "action": "start",
        "backend": "llama_swap",
        "running": False,
    }
    assert client.post("/api/proxy/lifecycle/apply").json()["action"] == "apply"
    assert client.post("/api/proxy/lifecycle/bogus").status_code == 404
    assert calls == ["start", "restart"]


def test_model_logs_support_search_level_and_download(client, monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "backend.routes.proxy_admin.configured_proxy_backend", lambda: "studio"
    )
    manager = get_studio_proxy_manager()

    class StubbornProcess:
        pid = 999_998

        def poll(self):
            return None

    from backend.proxy.studio.registry import Generation

    generation = Generation("demo", "a" * 64, 3001, StubbornProcess(), "llama_cpp")
    generation.logs.append("stdout: server listening")
    generation.logs.append("stderr: ggml_mmap: error mapping layer 7")
    generation.logs.append("stdout: WARNING slow load")
    manager.registry._starting["demo"] = generation
    try:
        logs = client.get("/api/proxy/models/demo/logs", params={"search": "layer"})
        assert logs.status_code == 200
        assert logs.json()["lines"] == ["stderr: ggml_mmap: error mapping layer 7"]

        errors = client.get("/api/proxy/models/demo/logs", params={"level": "error"})
        assert errors.json()["lines"] == ["stderr: ggml_mmap: error mapping layer 7"]
        warnings = client.get("/api/proxy/models/demo/logs", params={"level": "warn"})
        assert len(warnings.json()["lines"]) == 2

        assert client.get(
            "/api/proxy/models/demo/logs", params={"level": "verbose"}
        ).status_code == 400

        download = client.get("/api/proxy/models/demo/logs", params={"download": True})
        assert download.status_code == 200
        assert "attachment; filename=" in download.headers["content-disposition"]
        assert download.text == "\n".join(
            [
                "stdout: server listening",
                "stderr: ggml_mmap: error mapping layer 7",
                "stdout: WARNING slow load",
            ]
        ) + "\n"

        assert client.get("/api/proxy/models/missing/logs").status_code == 404
    finally:
        manager.registry._starting.pop("demo", None)


def test_key_rotation_replaces_the_secret(client, monkeypatch, tmp_path):
    from backend.inference_auth import verify_token

    _install_temp_store(monkeypatch, tmp_path)
    created = client.post("/api/inference/keys", json={"name": "rotate me"}).json()
    old_token = created["key"]
    rotated = client.post(f"/api/inference/keys/{created['id']}/rotate")
    assert rotated.status_code == 200
    new_token = rotated.json()["key"]
    assert new_token != old_token
    assert rotated.json()["prefix"] == new_token[:20]
    assert verify_token(old_token) is False
    assert verify_token(new_token) is True
    # Policy survives rotation.
    named = client.post(
        "/api/inference/keys",
        json={"name": "scoped", "models": ["m1"], "rate_limit_per_minute": 5},
    ).json()
    rotated_scoped = client.post(f"/api/inference/keys/{named['id']}/rotate").json()
    assert rotated_scoped["models"] == ["m1"]
    assert rotated_scoped["rate_limit_per_minute"] == 5
    assert client.post("/api/inference/keys/missing/rotate").status_code == 404


def test_preload_reports_warmup_best_effort(client, monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager
    from backend.proxy.studio.registry import Generation

    _install_temp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "backend.routes.proxy_admin.configured_proxy_backend", lambda: "studio"
    )
    manager = get_studio_proxy_manager()

    class StubbornProcess:
        pid = 999_997

        def poll(self):
            return None

    generation = Generation("demo", "a" * 64, 3001, StubbornProcess(), "fake")
    generation.ready = True
    released: list[str] = []

    async def fake_acquire(model_id):
        return generation

    monkeypatch.setattr(manager.registry, "acquire", fake_acquire)
    monkeypatch.setattr(manager.registry, "release", lambda gen: released.append(gen.model_id))

    class WarmResponse:
        status_code = 200

    async def fake_post(self, url, **kwargs):
        return WarmResponse()

    async def fake_aclose(self):
        return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        post = fake_post

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    warmed = client.post("/api/proxy/models/demo/preload", params={"warm": True})
    assert warmed.status_code == 200
    assert warmed.json() == {"model": "demo", "ready": True, "warmed": True}
    assert released == ["demo"]

    async def failing_post(self, url, **kwargs):
        raise httpx.ConnectError("engine not answering")

    FakeClient.post = failing_post
    cooled = client.post("/api/proxy/models/demo/preload", params={"warm": True})
    assert cooled.status_code == 200
    assert cooled.json()["warmed"] is False


def test_native_only_model_endpoints_guard_legacy_backend(client, monkeypatch, tmp_path):
    """Reload and log-clear must 409 in llama_swap mode, never spawn engines."""
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    launched: list[str] = []
    manager = get_studio_proxy_manager()

    async def spy_acquire(model_id):
        launched.append(model_id)
        raise AssertionError("reload must not launch in legacy mode")

    monkeypatch.setattr(manager.registry, "acquire", spy_acquire)
    reload_response = client.post("/api/proxy/models/demo/reload")
    clear_response = client.delete("/api/proxy/models/demo/logs")
    assert reload_response.status_code == 409
    assert "Studio backend" in reload_response.json()["detail"]
    assert clear_response.status_code == 409
    assert launched == []
