import httpx

from backend.proxy.studio.ring import EventRing
from backend.tests.test_api_routes import _install_temp_store


def test_proxy_settings_are_visible_and_persisted(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)

    initial = client.get("/api/proxy")
    assert initial.status_code == 200
    assert initial.json()["settings"]["backend"] == "studio"

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
    assert client.put("/api/proxy/settings", json={"max_body_bytes": 0}).status_code == 422
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


def test_lifecycle_dispatches_only_to_the_native_proxy(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    calls = []

    class Manager:
        running = False

        async def start_proxy(self):
            calls.append("start")
            self.running = True

        async def stop_proxy(self):
            calls.append("stop")
            self.running = False

        async def restart_proxy(self):
            calls.append("restart")
            self.running = True

    manager = Manager()
    monkeypatch.setattr("backend.routes.proxy_admin.get_studio_proxy_manager", lambda: manager)
    assert client.post("/api/proxy/lifecycle/start").json()["running"] is True
    assert client.post("/api/proxy/lifecycle/stop").json()["running"] is False
    assert client.post("/api/proxy/lifecycle/restart").json()["running"] is True
    assert calls == ["start", "stop", "restart"]
    assert client.post("/api/proxy/switch", json={"backend": "other"}).status_code in {404, 405}


def test_model_logs_support_search_level_and_download(client, monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("backend.routes.proxy_admin.configured_proxy_backend", lambda: "studio")
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

        assert (
            client.get("/api/proxy/models/demo/logs", params={"level": "verbose"}).status_code
            == 400
        )

        download = client.get("/api/proxy/models/demo/logs", params={"download": True})
        assert download.status_code == 200
        assert "attachment; filename=" in download.headers["content-disposition"]
        assert (
            download.text
            == "\n".join(
                [
                    "stdout: server listening",
                    "stderr: ggml_mmap: error mapping layer 7",
                    "stdout: WARNING slow load",
                ]
            )
            + "\n"
        )

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
    monkeypatch.setattr("backend.routes.proxy_admin.configured_proxy_backend", lambda: "studio")
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


def test_settings_migrate_retired_backend_without_losing_native_preferences(monkeypatch, tmp_path):
    from backend.tests.test_api_routes import _install_temp_store

    store = _install_temp_store(monkeypatch, tmp_path)
    store.write_document(
        "settings.yaml",
        {
            "proxy_port": 2345,
            "proxy": {
                "backend": "retired",
                "routing_aliases": {"fast": ["demo"]},
                "default_model": "demo",
                "allow_anonymous": False,
            },
        },
    )
    settings = store.get_settings()
    assert settings["proxy_port"] == 2345
    assert settings["proxy"] == {
        "backend": "studio",
        "default_model": "demo",
        "routing_aliases": {"fast": ["demo"]},
        "allow_anonymous": False,
    }
    store.update_settings({"public_inference_url": "http://localhost:2345"})
    assert store._read_yaml("settings.yaml")["proxy"]["routing_aliases"] == {"fast": ["demo"]}
