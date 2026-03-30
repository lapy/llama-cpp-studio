"""Additional API route coverage (FastAPI TestClient)."""

from urllib.parse import quote

import httpx
import pytest


def _install_temp_store(monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    return store


def _seed_model(store, model_id="org/model", quantization="Q4_K_M"):
    store.add_model(
        {
            "id": model_id,
            "name": "Test Model",
            "display_name": "Test Model",
            "huggingface_id": "org/model",
            "quantization": quantization,
            "format": "gguf",
            "config": {},
        }
    )


def test_openapi_schema_available(client):
    r = client.get("/openapi.json")
    assert r.status_code == 200
    data = r.json()
    assert data["info"]["title"]
    assert "/api/status" in str(data.get("paths", {}))


def test_gpu_info_returns_cpu_threads(client):
    r = client.get("/api/gpu-info")
    assert r.status_code == 200
    data = r.json()
    assert "cpu_threads" in data
    assert isinstance(data["cpu_threads"], int)


def test_llama_versions_list(client):
    r = client.get("/api/llama-versions/")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_lmdeploy_status_route(client):
    r = client.get("/api/lmdeploy/status")
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_llama_swap_pending_route(client):
    r = client.get("/api/llama-swap/pending")
    assert r.status_code == 200


def test_llama_swap_stale_route(client):
    r = client.get("/api/llama-swap/stale")
    assert r.status_code == 200
    data = r.json()
    assert "applicable" in data
    assert "stale" in data


def test_llama_swap_apply_route_success(client, monkeypatch):
    from backend.routes import llama_swap as llama_swap_routes

    called = {}

    class FakeManager:
        async def user_apply_regenerate_config(self):
            called["applied"] = True

    monkeypatch.setattr(llama_swap_routes, "get_llama_swap_manager", lambda: FakeManager())

    r = client.post("/api/llama-swap/apply-config")
    assert r.status_code == 200
    assert r.json()["message"] == "llama-swap configuration applied"
    assert called["applied"] is True


def test_llama_swap_apply_route_value_error_maps_to_400(client, monkeypatch):
    from backend.routes import llama_swap as llama_swap_routes

    class FakeManager:
        async def user_apply_regenerate_config(self):
            raise ValueError("bad config")

    monkeypatch.setattr(llama_swap_routes, "get_llama_swap_manager", lambda: FakeManager())

    r = client.post("/api/llama-swap/apply-config")
    assert r.status_code == 400
    assert r.json()["detail"] == "bad config"


def test_preview_llama_swap_cmd_unknown_model(client):
    r = client.post(
        "/api/models/nonexistent-model-id-xyz/preview-llama-swap-cmd",
        json={"engine": "llama_cpp", "engines": {}},
    )
    assert r.status_code == 404


def test_saved_llama_swap_cmd_unknown_model(client):
    r = client.get("/api/models/nonexistent-model-id-xyz/saved-llama-swap-cmd")
    assert r.status_code == 404


def test_saved_llama_swap_cmd_route_uses_stored_config(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.update_model(
        "org/model",
        {
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": {"temperature": 0.7}},
            }
        },
    )

    from backend import llama_swap_config

    seen = {}

    def fake_preview(model):
        seen["config"] = model.get("config")
        return {"ok": True, "cmd": "saved-cmd", "proxy_name": "org-model.q4_k_m"}

    monkeypatch.setattr(llama_swap_config, "preview_llama_swap_command_for_model", fake_preview)

    r = client.get(f"/api/models/{quote('org/model', safe='')}/saved-llama-swap-cmd")
    assert r.status_code == 200
    assert r.json()["cmd"] == "saved-cmd"
    assert seen["config"]["engines"]["llama_cpp"]["temperature"] == 0.7


def test_preview_llama_swap_cmd_route_applies_engine_section_replacement(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.update_model(
        "org/model",
        {
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": {"temperature": 0.7, "threads": 4}},
            }
        },
    )

    from backend import llama_swap_config

    seen = {}

    def fake_preview(model):
        seen["config"] = model.get("config")
        return {"ok": True, "cmd": "preview-cmd", "proxy_name": "org-model.q4_k_m"}

    monkeypatch.setattr(llama_swap_config, "preview_llama_swap_command_for_model", fake_preview)

    r = client.post(
        f"/api/models/{quote('org/model', safe='')}/preview-llama-swap-cmd",
        json={
            "engine": "llama_cpp",
            "engines": {"llama_cpp": {"temperature": 0.9}},
        },
    )
    assert r.status_code == 200
    assert r.json()["cmd"] == "preview-cmd"
    assert seen["config"]["engines"]["llama_cpp"]["temperature"] == 0.9
    assert "threads" not in seen["config"]["engines"]["llama_cpp"]


def test_preview_llama_swap_cmd_route_merges_flat_payload_into_active_engine(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.update_model(
        "org/model",
        {
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": {"temperature": 0.7, "threads": 4}},
            }
        },
    )

    from backend import llama_swap_config

    seen = {}

    def fake_preview(model):
        seen["config"] = model.get("config")
        return {"ok": True, "cmd": "preview-flat-cmd", "proxy_name": "org-model.q4_k_m"}

    monkeypatch.setattr(llama_swap_config, "preview_llama_swap_command_for_model", fake_preview)

    r = client.post(
        f"/api/models/{quote('org/model', safe='')}/preview-llama-swap-cmd",
        json={"threads": 8},
    )
    assert r.status_code == 200
    assert r.json()["cmd"] == "preview-flat-cmd"
    assert seen["config"]["engines"]["llama_cpp"]["temperature"] == 0.7
    assert seen["config"]["engines"]["llama_cpp"]["threads"] == 8


def test_model_list_exposes_raw_llama_swap_status(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)

    from backend.llama_swap_client import LlamaSwapClient

    async def fake_running_models(self):
        return {
            "running": [
                {
                    "model": "org-model.q4_k_m",
                    "state": "ready",
                }
            ]
        }

    monkeypatch.setattr(
        LlamaSwapClient, "get_running_models", fake_running_models
    )

    r = client.get("/api/models")
    assert r.status_code == 200
    payload = r.json()
    assert len(payload) == 1
    quant = payload[0]["quantizations"][0]
    assert quant["status"] == "ready"
    assert quant["run_state"] == "running"


def test_search_route_preserves_validation_errors(client):
    missing_query = client.post("/api/models/search", json={})
    assert missing_query.status_code == 400
    assert missing_query.json()["detail"] == "query parameter is required"

    bad_format = client.post(
        "/api/models/search",
        json={"query": "qwen", "model_format": "tarball"},
    )
    assert bad_format.status_code == 400
    assert "model_format must be either" in bad_format.json()["detail"]


def test_download_route_preserves_validation_errors(client):
    missing = client.post("/api/models/download", json={"huggingface_id": "org/repo"})
    assert missing.status_code == 400
    assert "huggingface_id and filename are required" in missing.json()["detail"]

    wrong_ext = client.post(
        "/api/models/download",
        json={
            "huggingface_id": "org/repo",
            "filename": "model.bin",
            "model_format": "gguf",
        },
    )
    assert wrong_ext.status_code == 400
    assert "must end with .gguf" in wrong_ext.json()["detail"]


def test_set_huggingface_token_route_preserves_validation_errors(client, monkeypatch):
    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)

    r = client.post("/api/models/huggingface-token", json={"token": "short"})
    assert r.status_code == 400
    assert r.json()["detail"] == "Invalid token format"


def test_search_file_size_route_and_removed_search_helpers(client, monkeypatch):
    from backend.routes import models as models_routes

    monkeypatch.setattr(
        models_routes,
        "get_accurate_file_sizes",
        lambda model_id, files: {name: 123 for name in files},
    )

    sizes = client.get(
        "/api/models/search/org/repo/file-sizes",
        params={"filenames": "a.gguf, b.gguf"},
    )
    assert sizes.status_code == 200
    assert sizes.json()["sizes"] == {"a.gguf": 123, "b.gguf": 123}

    missing = client.get(
        "/api/models/search/org/repo/file-sizes",
        params={"filenames": " , "},
    )
    assert missing.status_code == 400
    assert client.post("/api/models/search/clear-cache").status_code in {404, 405}
    assert client.get("/api/models/search/org-repo/details").status_code in {404, 405}
    assert client.get("/api/models/safetensors/org/repo/metadata").status_code in {404, 405}
    assert client.post("/api/models/safetensors/org/repo/metadata/regenerate").status_code in {404, 405}
    assert client.post("/api/models/safetensors/reload-from-disk").status_code in {404, 405}
    assert client.get("/api/models/supported-flags").status_code in {404, 405}
    assert client.post("/api/models/org/repo/regenerate-info").status_code in {404, 405}


def test_safetensors_list_and_token_status_routes(client, monkeypatch):
    from backend.routes import models as models_routes

    monkeypatch.setattr(
        models_routes,
        "list_grouped_safetensors_downloads",
        lambda: [{"huggingface_id": "org/repo"}],
    )
    monkeypatch.setattr(models_routes, "get_huggingface_token", lambda: "hf_1234567890")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_env_token_123")

    safetensors = client.get("/api/models/safetensors")
    assert safetensors.status_code == 200
    assert safetensors.json() == [{"huggingface_id": "org/repo"}]

    token = client.get("/api/models/huggingface-token")
    assert token.status_code == 200
    assert token.json()["has_token"] is True
    assert token.json()["token_preview"] == "hf_12345..."
    assert token.json()["from_environment"] is True


def test_quantization_sizes_preserves_validation_errors_as_400(client):
    response = client.post("/api/models/quantization-sizes", json={"huggingface_id": "org/repo"})
    assert response.status_code == 400
    assert "quantizations are required" in response.json()["detail"]


def test_set_huggingface_token_route_handles_env_override_clear_and_success(client, monkeypatch):
    from backend.routes import models as models_routes

    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_env_token_123")
    locked = client.post("/api/models/huggingface-token", json={"token": "hf_override_123"})
    assert locked.status_code == 200
    assert locked.json()["from_environment"] is True

    monkeypatch.delenv("HUGGINGFACE_API_KEY", raising=False)
    captured = []
    monkeypatch.setattr(models_routes, "set_huggingface_token", lambda token: captured.append(token))

    cleared = client.post("/api/models/huggingface-token", json={"token": ""})
    assert cleared.status_code == 200
    assert cleared.json()["has_token"] is False

    saved = client.post("/api/models/huggingface-token", json={"token": "hf_valid_token_123"})
    assert saved.status_code == 200
    assert saved.json()["has_token"] is True
    assert captured == ["", "hf_valid_token_123"]


def test_model_config_routes_round_trip_and_mark_stale(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)
    store.update_model(
        "org/model",
        {
            "config": {
                "engine": "llama_cpp",
                "engines": {"llama_cpp": {"temperature": 0.7}},
            }
        },
    )
    marked = {}

    from backend.routes import models as models_routes

    monkeypatch.setattr(models_routes, "_mark_llama_swap_stale", lambda: marked.setdefault("stale", True))

    original = client.get(f"/api/models/{quote('org/model', safe='')}/config")
    assert original.status_code == 200
    assert original.json()["engine"] == "llama_cpp"

    updated = client.put(
        f"/api/models/{quote('org/model', safe='')}/config",
        json={"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 0.9}}},
    )
    assert updated.status_code == 200
    assert updated.json()["engines"]["llama_cpp"]["temperature"] == 0.9
    assert marked["stale"] is True


def test_model_start_route_passthroughs_llama_swap_response(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)

    from backend.llama_swap_client import LlamaSwapClient

    async def fake_start(self, model_name):
        assert model_name == "org-model.q4_k_m"
        return httpx.Response(
            202,
            json={"model": model_name, "state": "loading"},
        )

    monkeypatch.setattr(
        LlamaSwapClient, "start_model_passthrough", fake_start
    )

    r = client.post(f"/api/models/{quote('org/model', safe='')}/start")
    assert r.status_code == 202
    assert r.json() == {"model": "org-model.q4_k_m", "state": "loading"}


def test_model_stop_route_passthroughs_llama_swap_response(
    client, monkeypatch, tmp_path
):
    store = _install_temp_store(monkeypatch, tmp_path)
    _seed_model(store)

    from backend.llama_swap_client import LlamaSwapClient

    async def fake_stop(self, model_name):
        assert model_name == "org-model.q4_k_m"
        return httpx.Response(
            200,
            text="OK",
            headers={"content-type": "text/plain; charset=utf-8"},
        )

    monkeypatch.setattr(
        LlamaSwapClient, "stop_model_passthrough", fake_stop
    )

    r = client.post(f"/api/models/{quote('org/model', safe='')}/stop")
    assert r.status_code == 200
    assert r.text == "OK"
    assert r.headers["content-type"].startswith("text/plain")


def test_build_cancel_requires_task_id(client):
    r = client.post("/api/llama-versions/build-cancel", json={})
    assert r.status_code == 400


def test_build_cancel_unknown_task(client):
    r = client.post(
        "/api/llama-versions/build-cancel",
        json={"task_id": "nonexistent-task-id-xyz"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body.get("ok") is False


def test_task_status_placeholder(client):
    r = client.get("/api/llama-versions/task-status/abc123")
    assert r.status_code == 200
    data = r.json()
    assert data["task_id"] == "abc123"
    assert "status" in data


@pytest.mark.parametrize(
    "engine",
    ["llama_cpp", "ik_llama", "lmdeploy"],
)
def test_param_registry_by_engine(client, engine):
    r = client.get("/api/models/param-registry", params={"engine": engine})
    assert r.status_code == 200
    data = r.json()
    assert "sections" in data
    assert isinstance(data["sections"], list)
