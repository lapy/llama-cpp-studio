"""Regression ceilings for catalog, history, and the compressed asset-size budget.

The asset-size check recompresses the production bundle. It is not a
measurement of network transfer.

The 2026-10-05 review measured warm catalog medians of 0.68 / 3.10 / 32.84 ms
for 10 / 100 / 1,000 models, against 1.06 / 7.08 / 162.96 ms before the catalog
cache. A repeated operation upsert after a large success history was 4.25 ms
against 1,349 ms. These ceilings sit above the new measurements. They are not
a claim that every old cost would miss them: the 1,000-model HTTP ceiling is
200 ms. That allows the TestClient boundary's small hosted-runner variance over
the review's 162.96 ms handler median while remaining far below the old path.
"""

import gzip
import json
import re
import statistics
import threading
import time
from pathlib import Path

import pytest
import yaml

from backend.data_store import DataStore

CATALOG_WARM_BUDGET_MS = {10: 25.0, 100: 50.0, 1000: 200.0}
# Catalog reads issued while one store write is still inside its lock.
# This is not a warm-cache ceiling.
CATALOG_WRITE_OVERLAP_BUDGET_MS = 2_000.0
CATALOG_JSON_BUDGET_BYTES = 1_600_000
HISTORY_CLEANUP_BUDGET_SECONDS = 3.0
HISTORY_UPSERT_BUDGET_MS = 80.0
COLD_MODELS_GZIP_BUDGET_BYTES = 250_000
_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def _model(index):
    return {
        "id": f"org/model-{index}",
        "huggingface_id": f"org/model-{index}",
        "display_name": f"Model {index}",
        "base_model_name": f"Model {index}",
        "format": "gguf",
        "quantization": "Q4_K_M",
        "config": {
            "engine": "llama_cpp",
            "engines": {"llama_cpp": {"ctx_size": 4096, "n_gpu_layers": 20}},
        },
    }


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


def _install_catalog(monkeypatch, tmp_path, count):
    from backend import data_store

    config = tmp_path / "config"
    config.mkdir()
    document = {"schema_version": 3, "models": [_model(index) for index in range(count)]}
    (config / "models.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    store = DataStore(config_dir=str(config))
    monkeypatch.setattr(data_store, "_store", store)
    return store


@pytest.mark.parametrize("count", [10, 100, 1000])
def test_catalog_warm_path_stays_within_budget(count, client, monkeypatch, tmp_path):
    _install_catalog(monkeypatch, tmp_path, count)

    class IdleProxy:
        async def get_running_models(self):
            return {"running": []}

    monkeypatch.setattr(
        "backend.proxy.llama_swap.client.get_llama_swap_client",
        lambda: IdleProxy(),
    )

    first = client.get("/api/models")
    assert first.status_code == 200
    payload = first.content
    if count == 1000:
        assert len(payload) <= CATALOG_JSON_BUDGET_BYTES

    samples = []
    for _ in range(7):
        started = time.perf_counter()
        response = client.get("/api/models")
        samples.append((time.perf_counter() - started) * 1000)
        assert response.status_code == 200
    median = statistics.median(samples)
    assert median <= CATALOG_WARM_BUDGET_MS[count]


def test_catalog_reads_during_a_store_write_stay_within_their_own_ceiling(
    client, monkeypatch, tmp_path
):
    """Concurrent catalog reads while one settings write holds the store lock."""
    count = 100
    store = _install_catalog(monkeypatch, tmp_path, count)

    class IdleProxy:
        async def get_running_models(self):
            return {"running": []}

    monkeypatch.setattr(
        "backend.proxy.llama_swap.client.get_llama_swap_client",
        lambda: IdleProxy(),
    )
    assert client.get("/api/models").status_code == 200

    started = threading.Event()
    release = threading.Event()
    original = store._mutate

    def holding_mutate(filename, mutator):
        def wrapped(data):
            if filename == "settings.yaml":
                started.set()
                assert release.wait(3)
            return mutator(data)

        return original(filename, wrapped)

    monkeypatch.setattr(store, "_mutate", holding_mutate)
    writer = threading.Thread(target=lambda: store.update_settings({"overlap": "held"}))
    writer.start()
    assert started.wait(3)

    samples = []
    errors = []

    def read_catalog():
        try:
            moment = time.perf_counter()
            response = client.get("/api/models")
            samples.append((time.perf_counter() - moment) * 1000)
            if response.status_code != 200:
                errors.append(response.status_code)
        except Exception as exc:  # reported below
            errors.append(exc)

    readers = [threading.Thread(target=read_catalog) for _ in range(4)]
    for reader in readers:
        reader.start()
    time.sleep(0.05)
    release.set()
    writer.join(timeout=5)
    for reader in readers:
        reader.join(timeout=5)

    assert not writer.is_alive()
    assert errors == []
    assert len(samples) == 4
    assert max(samples) <= CATALOG_WRITE_OVERLAP_BUDGET_MS
    assert CATALOG_WRITE_OVERLAP_BUDGET_MS > CATALOG_WARM_BUDGET_MS[count]


def test_large_success_history_cleans_up_and_later_upserts_stay_fast(tmp_path):
    from backend.operations.retention import MAX_RETAINED_TERMINAL_OPERATIONS

    config = tmp_path / "config"
    config.mkdir()
    now = time.time()
    seeded = [_operation(f"failed-{index}", "failed", now + index) for index in range(5000)]
    (config / "operations.yaml").write_text(
        yaml.safe_dump({"schema_version": 1, "operations": seeded}, sort_keys=False),
        encoding="utf-8",
    )
    store = DataStore(config_dir=str(config))
    started = time.perf_counter()
    store.upsert_operation(_operation("failed-new", "failed", now + 6000))
    cleanup = time.perf_counter() - started
    assert cleanup <= HISTORY_CLEANUP_BUDGET_SECONDS
    assert len(store.list_operations()) <= MAX_RETAINED_TERMINAL_OPERATIONS

    samples = []
    for index in range(20):
        started = time.perf_counter()
        store.upsert_operation(_operation(f"failed-later-{index}", "failed", now + 7000 + index))
        samples.append((time.perf_counter() - started) * 1000)
    assert statistics.median(samples) <= HISTORY_UPSERT_BUDGET_MS
    assert len(store.list_operations()) <= MAX_RETAINED_TERMINAL_OPERATIONS


def _gzip_size(path: Path) -> int:
    return len(gzip.compress(path.read_bytes(), compresslevel=6))


def _add_asset(chosen: list[Path], seen: set[Path], dist: Path, relative: str) -> Path | None:
    relative = relative.split("?", 1)[0].split("#", 1)[0].lstrip("./")
    if relative.startswith("/"):
        relative = relative[1:]
    path = dist / relative
    if path in seen or not path.is_file():
        return None
    seen.add(path)
    chosen.append(path)
    return path


def _cold_models_assets(dist: Path) -> list[Path]:
    """Files counted by the compressed asset-size budget for /models.

    The route's Vite preload list supplies the library script, its static
    imports, and its stylesheet. Stylesheets contribute the font file a
    browser actually selects (woff2), not the older fallbacks.
    """
    html_path = dist / "index.html"
    html = html_path.read_text(encoding="utf-8")
    chosen: list[Path] = [html_path]
    seen = {html_path}

    def add(relative: str) -> Path | None:
        return _add_asset(chosen, seen, dist, relative)

    for ref in re.findall(r'(?:src|href)="([^"]+)"', html):
        if ref.endswith((".js", ".css", ".svg")):
            add(ref)
    entry_name = next(
        (Path(ref).name for ref in re.findall(r'src="([^"]+\.js)"', html)),
        "",
    )
    entry = next((path for path in chosen if path.name == entry_name), None)
    if entry is None:
        raise AssertionError("production build has no entry script")
    entry_text = entry.read_text(encoding="utf-8", errors="ignore")
    mapped = re.search(
        r"__vite__mapDeps=\(i,m=__vite__mapDeps,d=\(m\.f\|\|\(m\.f=(\[[\s\S]*?\])\)\)\)",
        entry_text,
    )
    route = re.search(
        r"path:(?:`/models`|\"/models\"|'/models')[\s\S]{0,800}?__vite__mapDeps\(\[([0-9,\s]+)\]\)",
        entry_text,
    )
    if mapped is None or route is None:
        raise AssertionError("production build does not expose the /models preload list")
    dep_files = json.loads(mapped.group(1))
    for index in (int(part) for part in route.group(1).split(",") if part.strip()):
        add(dep_files[index])
    for imported in re.findall(r'from["\']\./([^"\']+\.js)["\']', entry_text):
        add(f"assets/{imported}")

    pending = [path for path in chosen if path.suffix == ".js" and path != entry]
    while pending:
        script = pending.pop()
        text = script.read_text(encoding="utf-8", errors="ignore")
        for imported in set(re.findall(r'["\']\./([^"\']+\.(?:js|css))["\']', text)):
            added = add(f"assets/{imported}")
            if added is not None and added.suffix == ".js":
                pending.append(added)
    for stylesheet in [path for path in list(chosen) if path.suffix == ".css"]:
        text = stylesheet.read_text(encoding="utf-8", errors="ignore")
        for url in re.findall(r"url\(\s*['\"]?([^)'\"]+)", text):
            if url.startswith("data:"):
                continue
            if url.split("?", 1)[0].split("#", 1)[0].endswith(".woff2"):
                add(url)
    return chosen


@pytest.mark.skipif(not (_DIST / "index.html").is_file(), reason="production build is not present")
def test_compressed_asset_size_stays_within_budget():
    assets = _cold_models_assets(_DIST)
    names = {path.name for path in assets}
    assert any(name.startswith("ModelLibrary-") and name.endswith(".js") for name in names)
    assert any(name.startswith("ModelLibrary-") and name.endswith(".css") for name in names)
    assert any(name.startswith("primeicons-") and name.endswith(".woff2") for name in names)
    assert sum(_gzip_size(path) for path in assets) <= COLD_MODELS_GZIP_BUDGET_BYTES
