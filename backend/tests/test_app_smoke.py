"""
Smoke tests to verify the app and key routes work after refactoring.
Run with: pytest backend/tests/test_app_smoke.py -v
(Requires: pip install -r requirements-dev.txt)
"""
import pytest


def test_app_starts(client):
    """App should start and respond."""
    # Root or health-style endpoint may redirect; use API
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert "system" in data


def test_param_registry_route(client):
    """Param registry should return catalog-backed sections."""
    response = client.get("/api/models/param-registry")
    assert response.status_code == 200
    data = response.json()
    assert "sections" in data
    assert isinstance(data["sections"], list)


def test_models_list_route(client):
    """Models list should return 200 and a list (possibly empty)."""
    response = client.get("/api/models/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


def test_models_list_route_no_trailing_slash(client):
    """GET /api/models (no trailing slash) should return model list, not param-registry."""
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)


@pytest.mark.skip(reason="SSE stream never ends; TestClient blocks on full response")
def test_sse_events_route(client):
    """SSE events endpoint returns 200 and event-stream content-type."""
    response = client.get("/api/events")
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")
