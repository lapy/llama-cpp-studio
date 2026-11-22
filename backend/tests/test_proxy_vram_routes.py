"""GPU admission API surface: settings, snapshot, 409 refusal, catalog evidence."""

from types import SimpleNamespace

import httpx
import pytest

from backend.proxy.studio.dataplane import ResolvedModel, StudioDataPlane
from backend.proxy.studio.registry import RegistryError
from backend.proxy.studio.vram import VramAdmissionError
from backend.tests.test_api_routes import _install_temp_store

MIB = 1024 * 1024


def refusal(reason="Insufficient VRAM to start demo", **extra):
    decision = {
        "decision": "rejected",
        "model": "demo",
        "engine": "llama_cpp",
        "mode": "enforce",
        "estimate_mib": 16384,
        "estimate_source": "settings",
        "safety_margin_mib": 1024,
        "available_mib": 7168,
        "deficit_mib": 9216,
        "devices": [{"index": 0, "free_mib": 8192, "margin_mib": 1024, "available_mib": 7168}],
        "reason": reason,
        "message": reason,
    }
    decision.update(extra)
    return VramAdmissionError(decision)


def test_vram_endpoint_defaults_to_off_without_probing(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    payload = client.get("/api/proxy/vram").json()
    assert payload["mode"] == "off"
    assert payload["safety_margin_mib"] == 1024
    assert payload["estimates_configured"] == {}
    assert payload["probe"]["status"] == "never"
    assert payload["last_admission"] is None
    assert payload["reservations"] == []


def test_vram_settings_are_visible_and_apply_live(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    initial = client.get("/api/proxy").json()
    assert initial["settings"]["vram_admission"] == "off"
    assert initial["settings"]["vram_safety_margin_mib"] == 1024
    assert initial["settings"]["vram_estimates"] == {}
    assert initial["vram"]["mode"] == "off"

    saved = client.put(
        "/api/proxy/settings",
        json={
            "vram_admission": "warn",
            "vram_safety_margin_mib": 2048,
            "vram_estimates": {"demo": 8192},
        },
    )
    assert saved.status_code == 200
    body = saved.json()
    assert body["restart_required"] is False
    assert body["applied_live_fields"] == [
        "vram_admission",
        "vram_estimates",
        "vram_safety_margin_mib",
    ]
    assert store.get_settings()["proxy"]["vram_admission"] == "warn"

    after = client.get("/api/proxy").json()
    assert after["settings"]["vram_estimates"] == {"demo": 8192}
    assert after["vram"]["mode"] == "warn"
    assert after["vram"]["safety_margin_mib"] == 2048
    assert after["vram"]["estimates_configured"] == {"demo": 8192}


def test_vram_settings_validate_mode_margin_and_estimates(client, monkeypatch, tmp_path):
    _install_temp_store(monkeypatch, tmp_path)
    url = "/api/proxy/settings"
    assert client.put(url, json={"vram_admission": "loud"}).status_code == 422
    assert client.put(url, json={"vram_safety_margin_mib": -1}).status_code == 422
    assert client.put(url, json={"vram_safety_margin_mib": 1048577}).status_code == 422
    assert client.put(url, json={"vram_estimates": {"demo": 0}}).status_code == 422
    assert client.put(url, json={"vram_estimates": {" ": 4096}}).status_code == 422
    too_many = {f"model-{index}": 1 for index in range(501)}
    assert client.put(url, json={"vram_estimates": too_many}).status_code == 422


def test_preload_maps_an_admission_refusal_to_structured_409(client, monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    manager = get_studio_proxy_manager()

    async def deny(model_id):
        raise refusal()

    monkeypatch.setattr(manager.registry, "acquire", deny)
    response = client.post("/api/proxy/models/demo/preload")
    assert response.status_code == 409
    body = response.json()
    assert body["code"] == "INSUFFICIENT_VRAM"
    assert body["error"] == "Insufficient VRAM to start demo"
    detail = body["detail"]
    assert detail["decision"] == "rejected"
    assert detail["message"] == body["error"]
    assert detail["estimate_mib"] == 16384


def test_reload_maps_an_admission_refusal_to_structured_409(client, monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    manager = get_studio_proxy_manager()

    async def stop(model_id):
        return False

    async def deny(model_id):
        raise refusal()

    monkeypatch.setattr(manager.registry, "stop", stop)
    monkeypatch.setattr(manager.registry, "acquire", deny)
    response = client.post("/api/proxy/models/demo/reload")
    assert response.status_code == 409
    assert response.json()["code"] == "INSUFFICIENT_VRAM"


def test_ordinary_start_failures_keep_the_existing_502_body(client, monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    manager = get_studio_proxy_manager()

    async def fail(model_id):
        raise RegistryError("engine readiness timed out after 2700s")

    monkeypatch.setattr(manager.registry, "acquire", fail)
    response = client.post("/api/proxy/models/demo/preload")
    assert response.status_code == 502
    assert response.json() == {"detail": "engine readiness timed out after 2700s"}


@pytest.mark.asyncio
async def test_inference_cold_start_returns_structured_409_without_dispatching():
    decision_error = refusal()

    class DenyRegistry:
        async def acquire(self, model_id):
            raise decision_error

        def release(self, generation):
            raise AssertionError("a refused start must not release a generation")

        def snapshot(self):
            return []

    app = StudioDataPlane(DenyRegistry(), lambda requested: ResolvedModel(requested or "demo"))
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://proxy"
        ) as client:
            response = await client.post(
                "/v1/chat/completions",
                json={"model": "demo", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 409
        body = response.json()
        assert body["code"] == "INSUFFICIENT_VRAM"
        assert body["error"] == "Insufficient VRAM to start demo"
        assert body["detail"]["deficit_mib"] == 9216
        assert body["detail"]["devices"][0]["index"] == 0
    finally:
        await app.close()


@pytest.mark.asyncio
async def test_queue_and_engine_failures_keep_their_status_and_body():
    class FailingRegistry:
        def __init__(self, error):
            self.error = error

        async def acquire(self, model_id):
            raise self.error

        def release(self, generation):
            return None

        def snapshot(self):
            return []

    for error, status, expected in (
        (RegistryError("maximum loaded models reached (1)"), 429, "maximum loaded models"),
        (RegistryError("engine exited before readiness (code 1)"), 502, "engine exited"),
    ):
        app = StudioDataPlane(FailingRegistry(error), lambda requested: ResolvedModel("demo"))
        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://proxy"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions", json={"model": "demo", "messages": []}
                )
            assert response.status_code == status
            assert expected in response.json()["error"]
        finally:
            await app.close()


def test_catalog_rows_expose_estimate_evidence(monkeypatch, tmp_path):
    from backend.proxy.studio.manager import get_studio_proxy_manager

    _install_temp_store(monkeypatch, tmp_path)
    manager = get_studio_proxy_manager()
    manager.vram.apply_settings({"vram_admission": "warn", "vram_estimates": {"demo": 9000}})
    compiled = SimpleNamespace(
        proxy=SimpleNamespace(model_id="demo", engine_id="llama_cpp", aliases=[], capabilities=None)
    )
    model = {
        "id": "stored-demo",
        "display_name": "Demo",
        "huggingface_id": "org/demo",
        "file_size": 3 * 1024 * MIB,
    }
    monkeypatch.setattr(manager, "_compiled_models", lambda: [(model, compiled)])

    assert manager._catalog()[0]["vram_estimate"] == {"mib": 9000, "source": "settings"}

    manager.vram.apply_settings({"vram_estimates": {}})
    assert manager._catalog()[0]["vram_estimate"] == {"mib": 3072, "source": "file_size"}


@pytest.mark.asyncio
async def test_admission_activity_is_recorded_in_the_registry_ring(tmp_path):
    from backend.proxy.studio.vram import VramAdmission
    from backend.proxy.launch_spec import EnvSpec, LaunchSpec

    async def probe():
        return {
            "vendor": "nvidia",
            "gpus": [
                {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "Fake",
                    "total_mib": 24576,
                    "free_mib": 4096,
                    "used_mib": 20480,
                }
            ],
            "tooling": "nvml",
            "error": None,
        }

    async def model_info(model_id):
        return {"id": model_id, "file_size": 0}

    admission = VramAdmission(model_info=model_info, probe=probe)
    admission.apply_settings({"vram_admission": "warn", "vram_estimates": {"demo": 16000}})
    spec = LaunchSpec(
        model_id="demo",
        engine_id="llama_cpp",
        engine_install_id="fake",
        executable="/bin/true",
        argv=[],
        cwd=None,
        env=EnvSpec(set={"CUDA_VISIBLE_DEVICES": "0"}),
    )
    token = await admission.admit(spec)
    admission.release(token)
    activity = admission.snapshot()["last_admission"]
    assert activity["decision"] == "warned"
    # The registry wires this emitter to its own activity ring in production.
    activities = []
    admission.bind_activity(lambda kind, payload: activities.append({"type": kind, **payload}))
    await admission.admit(spec)
    assert [item["type"] for item in activities] == ["vram_warning"]
    assert activities[0]["decision"] == "warned"
    assert activities[0]["deficit_mib"] == 16000 - (4096 - 1024)
