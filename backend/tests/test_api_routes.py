"""Additional API route coverage (FastAPI TestClient)."""

import pytest


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


def test_preview_llama_swap_cmd_unknown_model(client):
    r = client.post(
        "/api/models/nonexistent-model-id-xyz/preview-llama-swap-cmd",
        json={"engine": "llama_cpp", "engines": {}},
    )
    assert r.status_code == 404


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
