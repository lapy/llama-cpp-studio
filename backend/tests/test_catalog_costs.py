"""Catalog size, operation retention, and off-loop store I/O."""

import asyncio
import gzip
import time
from urllib.parse import quote

import httpx
import pytest
from fastapi import FastAPI

from backend.data_store import DataStore
from backend.tests.test_api_routes import _install_temp_store


def _operation(operation_id, status, updated_at):
    return {
        "operation_id": operation_id,
        "kind": "build",
        "status": status,
        "resource_key": "",
        "resumable": False,
        "detail": {},
        "message": "",
        "updated_at": updated_at,
    }


def test_terminal_history_is_pruned_and_survives_restart(tmp_path):
    from backend.operations.retention import (
        MAX_RETAINED_TERMINAL_OPERATIONS,
        TERMINAL_OPERATION_MAX_AGE_SECONDS,
    )

    config_dir = tmp_path / "config"
    store = DataStore(config_dir=str(config_dir))
    now = time.time()
    store.upsert_operation(_operation("running-old", "running", now - 10 * TERMINAL_OPERATION_MAX_AGE_SECONDS))
    store.upsert_operation(_operation("failed-recent", "failed", now))
    store.upsert_operation(_operation("failed-stale", "failed", now - TERMINAL_OPERATION_MAX_AGE_SECONDS - 5))
    store.upsert_operation(_operation("done", "succeeded", now))

    kept = {row["operation_id"] for row in store.list_operations()}
    assert "running-old" in kept
    assert "failed-recent" in kept
    assert "failed-stale" not in kept
    assert "done" not in kept

    for index in range(MAX_RETAINED_TERMINAL_OPERATIONS + 25):
        store.upsert_operation(_operation(f"failed-{index}", "failed", now + index + 1))

    restarted = DataStore(config_dir=str(config_dir))
    rows = restarted.list_operations()
    ids = [row["operation_id"] for row in rows]
    assert "running-old" in ids
    assert "failed-recent" not in ids
    failed = [row_id for row_id in ids if row_id.startswith("failed-")]
    assert len(failed) == MAX_RETAINED_TERMINAL_OPERATIONS
    assert "failed-0" not in ids
    assert f"failed-{MAX_RETAINED_TERMINAL_OPERATIONS + 24}" in ids


def test_failed_write_does_not_replace_the_cached_document(tmp_path):
    store = DataStore(config_dir=str(tmp_path / "config"))
    store.update_settings({"proxy_port": 2001})

    def fail_write(path, data):
        raise OSError("disk full")

    store._write_yaml = fail_write
    with pytest.raises(OSError, match="disk full"):
        store.update_settings({"proxy_port": 2002})

    assert store.get_settings()["proxy_port"] == 2001
    other = DataStore(config_dir=str(tmp_path / "config"))
    assert other.get_settings()["proxy_port"] == 2001


def test_parsed_document_cache_reloads_after_another_process_write(tmp_path):
    config_dir = str(tmp_path / "config")
    first = DataStore(config_dir=config_dir)
    second = DataStore(config_dir=config_dir)
    assert first.get_settings()["proxy_port"] == 2000
    second.update_settings({"proxy_port": 2345})
    assert first.get_settings()["proxy_port"] == 2345


def test_read_cache_cannot_restore_bytes_from_before_a_concurrent_write(tmp_path):
    """A stamp taken after the read must not bless the bytes from before the write."""
    config_dir = str(tmp_path / "config")
    reader = DataStore(config_dir=config_dir)
    writer = DataStore(config_dir=config_dir)
    original = reader._load_document
    injected = {"done": False}

    def load_then_let_the_other_store_commit(path):
        status, data = original(path)
        if not injected["done"] and path.endswith("settings.yaml"):
            injected["done"] = True
            writer.update_settings({"proxy_port": 2345})
        return status, data

    reader._load_document = load_then_let_the_other_store_commit
    seen = reader.get_settings()
    assert injected["done"] is True
    assert seen["proxy_port"] == 2345
    assert writer.get_settings()["proxy_port"] == 2345

    reader.update_settings({"public_inference_url": "https://infer.example.test"})
    fresh = DataStore(config_dir=config_dir)
    settings = fresh.get_settings()
    assert settings["proxy_port"] == 2345
    assert settings["public_inference_url"] == "https://infer.example.test"


def test_catalog_summary_omits_engine_parameters(client, monkeypatch, tmp_path):
    store = _install_temp_store(monkeypatch, tmp_path)
    store.add_model(
        {
            "id": "org/model",
            "name": "Test Model",
            "display_name": "Test Model",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {
                "engine": "llama_cpp",
                "engines": {
                    "llama_cpp": {
                        "temperature": 0.2,
                        "ctx_size": 4096,
                        "model_alias": "friendly",
                        "set_params_by_id": [{"sub_id": "high", "temperature": 0.9}],
                    }
                },
            },
        }
    )

    listed = client.get("/api/models")
    assert listed.status_code == 200
    config = listed.json()[0]["quantizations"][0]["config"]
    assert config["engine"] == "llama_cpp"
    assert config["engines"]["llama_cpp"]["model_alias"] == "friendly"
    assert config["engines"]["llama_cpp"]["set_params_by_id"] == [{"sub_id": "high"}]
    assert "temperature" not in config["engines"]["llama_cpp"]
    assert "ctx_size" not in config["engines"]["llama_cpp"]

    saved = client.get(f"/api/models/{quote('org/model', safe='')}/config")
    assert saved.status_code == 200
    assert saved.json()["temperature"] == 0.2


@pytest.mark.asyncio
async def test_catalog_read_stays_off_the_event_loop(monkeypatch):
    from backend.main import app

    def slow_list(self):
        time.sleep(0.45)
        return []

    monkeypatch.setattr(DataStore, "list_models", slow_list)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        started = time.perf_counter()
        listing = asyncio.create_task(client.get("/api/models"))
        await asyncio.sleep(0.05)
        elapsed = time.perf_counter() - started
        response = await listing

    assert elapsed < 0.25
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_operation_write_is_durable_without_stalling_the_loop(tmp_path, monkeypatch):
    from backend import data_store
    from backend.operations.supervisor import get_supervisor
    from backend.store_io import StoreIoMiddleware

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    original = DataStore.upsert_operation

    def slow_upsert(self, operation):
        time.sleep(0.45)
        return original(self, operation)

    monkeypatch.setattr(DataStore, "upsert_operation", slow_upsert)

    mini = FastAPI()
    mini.add_middleware(StoreIoMiddleware)

    @mini.post("/save")
    async def save():
        get_supervisor().start_operation("op-durable", "build", "install-dir")
        return {"ok": True}

    transport = httpx.ASGITransport(app=mini)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        started = time.perf_counter()
        saving = asyncio.create_task(client.post("/save"))
        await asyncio.sleep(0.05)
        elapsed = time.perf_counter() - started
        response = await saving

    assert elapsed < 0.25
    assert response.status_code == 200
    rows = store.list_operations()
    assert [row["operation_id"] for row in rows] == ["op-durable"]
    assert rows[0]["status"] == "running"


def test_proxy_http_client_is_reused(monkeypatch):
    from backend.proxy.llama_swap.client import LlamaSwapClient, reset_proxy_clients

    created = {"count": 0}

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            created["count"] += 1

        async def get(self, url, timeout):
            request = httpx.Request("GET", url)
            return httpx.Response(
                200,
                request=request,
                json={"running": []},
            )

    reset_proxy_clients()
    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    asyncio.run(LlamaSwapClient().get_running_models())
    asyncio.run(LlamaSwapClient().get_running_models())
    assert created["count"] == 1


@pytest.mark.asyncio
async def test_hashed_assets_negotiate_gzip(tmp_path):
    from backend.static_assets import HashedAssetFiles, ensure_precompressed_assets

    assets = tmp_path / "assets"
    assets.mkdir()
    source = assets / "app.js"
    source.write_text("console.log('studio');\n" * 40, encoding="utf-8")
    assert ensure_precompressed_assets(str(assets)) == 1
    compressed = gzip.decompress((assets / "app.js.gz").read_bytes())
    assert compressed == source.read_bytes()

    app = FastAPI()
    app.mount("/assets", HashedAssetFiles(directory=str(assets)), name="assets")
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://studio") as client:
        plain = await client.get("/assets/app.js", headers={"Accept-Encoding": "identity"})
        encoded = await client.get("/assets/app.js", headers={"Accept-Encoding": "gzip"})
        rejected = await client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip;q=0, identity;q=1"},
        )
        identity_preferred = await client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip;q=0.1, identity;q=1"},
        )
        gzip_preferred = await client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip;q=0.8, identity;q=0.1"},
        )
        wildcard_gzip = await client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "*;q=0, gzip;q=1"},
        )
        refused = await client.get(
            "/assets/app.js",
            headers={"Accept-Encoding": "gzip;q=0, identity;q=0, *;q=0"},
        )

    assert plain.status_code == 200
    assert "content-encoding" not in {key.lower() for key in plain.headers}
    assert plain.headers.get("vary") == "Accept-Encoding"
    assert plain.content == source.read_bytes()
    assert encoded.status_code == 200
    assert encoded.headers.get("content-encoding") == "gzip"
    assert encoded.headers.get("vary") == "Accept-Encoding"
    # The client gunzips because of Content-Encoding. A raw JS body labeled
    # gzip would fail this read instead of matching the source file.
    assert encoded.content == source.read_bytes()
    assert "immutable" in encoded.headers.get("cache-control", "")

    assert rejected.status_code == 200
    assert "content-encoding" not in {key.lower() for key in rejected.headers}
    assert rejected.headers.get("vary") == "Accept-Encoding"
    assert rejected.content == source.read_bytes()

    assert identity_preferred.status_code == 200
    assert "content-encoding" not in {key.lower() for key in identity_preferred.headers}
    assert identity_preferred.content == source.read_bytes()

    assert gzip_preferred.status_code == 200
    assert gzip_preferred.headers.get("content-encoding") == "gzip"
    assert gzip_preferred.headers.get("vary") == "Accept-Encoding"

    assert wildcard_gzip.status_code == 200
    assert wildcard_gzip.headers.get("content-encoding") == "gzip"

    assert refused.status_code == 406
    assert refused.headers.get("vary") == "Accept-Encoding"
    assert "content-encoding" not in {key.lower() for key in refused.headers}


def test_negotiated_asset_path_ranks_quality_wildcard_and_brotli(tmp_path):
    from backend.static_assets import negotiated_asset_path

    source = tmp_path / "app.js"
    source.write_text("var studio = 1;\n", encoding="utf-8")
    (tmp_path / "app.js.gz").write_bytes(b"gzip-bytes")
    (tmp_path / "app.js.br").write_bytes(b"br-bytes")

    def scope(header):
        return {"type": "http", "headers": [(b"accept-encoding", header.encode("latin1"))]}

    gzip_path, gzip_coding = negotiated_asset_path(str(source), scope("br;q=0.2, gzip;q=0.9"))
    assert gzip_path.endswith(".gz")
    assert gzip_coding == "gzip"
    br_path, br_coding = negotiated_asset_path(str(source), scope("br;q=0.9, gzip;q=0.2"))
    assert br_path.endswith(".br")
    assert br_coding == "br"
    identity = negotiated_asset_path(str(source), scope("*;q=0, identity;q=1"))
    assert identity == (str(source), None)
    assert negotiated_asset_path(str(source), scope("*;q=0")) is None
    assert negotiated_asset_path(str(source), scope("gzip;q=0, identity;q=1")) == (
        str(source),
        None,
    )
