"""HTTP contracts for storage inventory and guarded cleanup."""

import asyncio
import threading
import time

from backend.routes import storage as storage_routes
from backend.services import storage_cleanup as cleanup
from backend.services import storage_inventory as inventory
from backend.tests.test_storage_inventory import (
    _add_models,
    _build_tree,
    _seed_storage,
)


def _category(payload, category_id):
    return next(item for item in payload["categories"] if item["id"] == category_id)


def test_inventory_route_reports_categories_and_uses_the_ttl_cache(client, monkeypatch, tmp_path):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)

    response = client.get("/api/storage/inventory")
    assert response.status_code == 200
    payload = response.json()
    assert [item["id"] for item in payload["categories"]] == list(inventory.CATEGORY_ORDER)
    assert payload["filesystem"]["data_root"]
    assert payload["scan"]["quality"] == "complete"
    assert all("path" not in unit for item in payload["categories"] for unit in item["units"])

    cached = client.get("/api/storage/inventory").json()
    assert cached["cached"] is True
    assert cached["generated_at"] == payload["generated_at"]

    fresh = client.get("/api/storage/inventory", params={"refresh": "true"}).json()
    assert fresh["cached"] is False


def test_preview_apply_contract_with_confirmation(client, monkeypatch, tmp_path):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    orphan = data / "models" / "audio-cpp" / "orphan-a"

    listing = client.get("/api/storage/inventory").json()
    cleanable = [unit for unit in _category(listing, "models")["units"] if unit["cleanable"]]
    assert len(cleanable) == 1

    preview = client.post(
        "/api/storage/cleanup/preview",
        json={"category": "models", "unit_ids": [cleanable[0]["id"]]},
    )
    assert preview.status_code == 200
    plan = preview.json()
    assert plan["units"][0]["id"] == cleanable[0]["id"]
    assert plan["confirmation_phrase"] == "DELETE 1 MODEL DIRECTORIES"
    assert plan["reclaimable_bytes"] == 50

    unconfirmed = client.post(
        "/api/storage/cleanup/apply",
        json={"plan_id": plan["plan_id"], "confirm": False},
    )
    assert unconfirmed.status_code == 400
    assert unconfirmed.json()["detail"]["code"] == "CONFIRMATION_REQUIRED"
    assert orphan.exists()

    applied = client.post(
        "/api/storage/cleanup/apply",
        json={"plan_id": plan["plan_id"], "confirm": True},
    )
    assert applied.status_code == 200
    result = applied.json()
    assert result["outcome"] == "completed"
    assert result["freed_bytes"] == 50
    assert not orphan.exists()

    stale = client.post(
        "/api/storage/cleanup/apply",
        json={"plan_id": plan["plan_id"], "confirm": True},
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["code"] == "PLAN_STALE"


def test_preview_refuses_report_only_categories(client, monkeypatch, tmp_path):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)

    response = client.post(
        "/api/storage/cleanup/preview",
        json={"category": "engine_installs"},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "CATEGORY_NOT_CLEANABLE"


def test_apply_unknown_plan_returns_409(client, monkeypatch, tmp_path):
    data, store = _seed_storage(tmp_path, monkeypatch)
    response = client.post(
        "/api/storage/cleanup/apply",
        json={"plan_id": "0" * 64, "confirm": True},
    )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "PLAN_STALE"


def test_apply_while_another_cleanup_runs_returns_409(client, monkeypatch, tmp_path):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)
    preview = client.post("/api/storage/cleanup/preview", json={"category": "models"}).json()
    assert cleanup._APPLY_LOCK.acquire(blocking=False)
    try:
        response = client.post(
            "/api/storage/cleanup/apply",
            json={"plan_id": preview["plan_id"], "confirm": True},
        )
    finally:
        cleanup._APPLY_LOCK.release()

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "CLEANUP_IN_FLIGHT"


def test_path_like_unit_ids_are_refused_not_interpreted(client, monkeypatch, tmp_path):
    data, store = _seed_storage(tmp_path, monkeypatch)
    _build_tree(tmp_path, data)
    _add_models(store, data)

    response = client.post(
        "/api/storage/cleanup/preview",
        json={"category": "models", "unit_ids": ["/etc/passwd"]},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["units"] == []
    assert payload["refused"][0]["code"] == "UNKNOWN_UNIT"


def test_preview_body_validation(client, monkeypatch, tmp_path):
    _seed_storage(tmp_path, monkeypatch)
    assert client.post("/api/storage/cleanup/preview", json={}).status_code == 422
    assert (
        client.post(
            "/api/storage/cleanup/preview",
            json={"category": "models", "unit_ids": ["x"] * 201},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/storage/cleanup/preview",
            json={"category": "models", "keep_history": 0},
        ).status_code
        == 422
    )


def test_inventory_scan_runs_on_the_blocking_pool(monkeypatch):
    observed = {}

    def slow_scan(*, refresh=False):
        observed["thread"] = threading.current_thread().name
        time.sleep(0.08)
        return {
            "schema_version": inventory.INVENTORY_SCHEMA_VERSION,
            "categories": [],
            "scan": {"quality": "complete"},
        }

    monkeypatch.setattr(storage_routes, "collect_storage_inventory", slow_scan)
    ticks = 0

    async def scenario():
        nonlocal ticks

        async def ticker():
            nonlocal ticks
            for _ in range(50):
                await asyncio.sleep(0.002)
                ticks += 1

        task = asyncio.create_task(ticker())
        try:
            payload = await storage_routes.storage_inventory(refresh=True)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        return payload

    payload = asyncio.run(scenario())

    assert payload["schema_version"] == inventory.INVENTORY_SCHEMA_VERSION
    assert observed["thread"].startswith("studio-blocking")
    assert ticks > 0
