"""Launch campaign R3+R4 — a YAML-only engine launches AND serves chat through
the Studio proxy (nothing stubbed that could fabricate a response shape).

The end-to-end journey for ``launchcraft`` (docs/yaml-engine-launch-campaign.md,
owner decisions Q1/Q2/Q4): REAL generic router → REAL recipes/executor →
REAL ``python -m venv`` → REAL ``pip install`` from a hand-built local wheel
(no network, no build backend) → activation row → the production
``compile_model_runtime`` path → REAL ``subprocess.Popen`` through
``ProcessRegistry`` (Studio's only server runtime) → REAL ``/health``
readiness → ``POST /api/models/{id}/start`` answered by the real route →
one streamed ``/v1/chat/completions`` through the studio dataplane into the
YAML-launched server's SSE stream → ``POST /api/models/{id}/stop`` drains and
reaps the process group with a journal receipt → the same definition-owned
installer's ``remove`` recipe deletes the version row and venv.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import threading
import zipfile
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.data_store import DataStore

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "launchcraft_server"
PACKAGE_DIR = FIXTURE_DIR / "launchcraft_fixture"
DIST = "launchcraft_fixture-0.1.0"
WHEEL_NAME = f"{DIST}-py3-none-any.whl"
ARGV_GOLDEN = ["--model", "fixture/model", "--port", {"runtime": "port"}]


def _take_lifecycle_snapshot(lifecycle) -> dict:
    """Snapshot the process-global LIFECYCLE registry and start from empty.

    Registration skips ids that already exist, so a leftover entry from
    another test would keep THAT test's (already restored) install-settings
    defaults and make this registration silently incomplete.
    """
    snapshot = dict(lifecycle)
    lifecycle.clear()
    return snapshot


def _restore_lifecycle_snapshot(lifecycle, snapshot: dict) -> None:
    lifecycle.clear()
    lifecycle.update(snapshot)


def _build_wheel(wheelhouse: Path) -> Path:
    """Hand-build a valid pure-python wheel — no build backend, no network.

    PEP 427 is just a zip + METADATA/WHEEL/entry_points/RECORD; pip
    installs it fine. The console script pip generates from
    entry_points.txt is what the YAML profile names.
    """
    files: dict[str, bytes] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        rel = f"launchcraft_fixture/{path.relative_to(PACKAGE_DIR).as_posix()}"
        files[rel] = path.read_bytes()
    files[f"{DIST}.dist-info/METADATA"] = (
        "Metadata-Version: 2.1\nName: launchcraft-fixture\nVersion: 0.1.0\n\n"
        "Hermetic launch fixture (not for publication).\n"
    ).encode("utf-8")
    files[f"{DIST}.dist-info/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: handrolled\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    ).encode("utf-8")
    files[f"{DIST}.dist-info/entry_points.txt"] = (
        "[console_scripts]\nlaunchcraft-server = launchcraft_fixture.server:main\n"
    ).encode("utf-8")
    record_lines = []
    for rel, data in files.items():
        digest = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        )
        record_lines.append(f"{rel},sha256={digest},{len(data)}")
    record_lines.append(f"{DIST}.dist-info/RECORD,,")
    files[f"{DIST}.dist-info/RECORD"] = ("\n".join(record_lines) + "\n").encode("utf-8")
    wheel = wheelhouse / WHEEL_NAME
    with zipfile.ZipFile(wheel, "w", zipfile.ZIP_DEFLATED) as archive:
        for rel, data in files.items():
            archive.writestr(rel, data)
    return wheel


@pytest.mark.asyncio
async def test_launchcraft_installs_launches_and_serves_chat_through_the_proxy(
    tmp_path, monkeypatch
):
    import backend.data_store as ds
    from backend.engines.python import generic as gen
    from backend.engines.python import installer as venv_installer
    from backend.proxy.launch_spec import compile_model_runtime
    from backend.proxy.studio.dataplane import StudioDataPlane
    from backend.proxy.studio.manager import StudioProxyManager
    from backend.proxy.studio.registry import ProcessRegistry
    from backend.routes import engine_versions as ev
    from backend.routes import models as models_routes
    from backend.routes.engine_lifecycle import router

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(venv_installer, "studio_data_dir", lambda: str(tmp_path / "data"))
    monkeypatch.setattr(venv_installer, "get_store", lambda: store)
    monkeypatch.setattr(gen, "get_store", lambda: store)
    monkeypatch.setattr(ev, "get_store", lambda: store)
    monkeypatch.setattr(ds, "get_store", lambda: store)
    monkeypatch.setattr(ds, "_store", store)
    monkeypatch.setattr("backend.proxy.studio.manager.get_store", lambda: store)
    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    import backend.venv_install_settings as vis

    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))

    # REAL pip, offline: every pip invocation gets a local wheelhouse.
    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    wheel = _build_wheel(wheelhouse)
    assert wheel.is_file()
    real_run_pip = gen.GenericVenvInstaller._run_pip

    async def pip_with_wheelhouse(
        self, args, operation, ensure_venv=True, *, cwd=None, env=None, append=True
    ):
        argv = list(args)
        if argv and argv[0] == "install":
            argv = [*argv, "--no-index", "--find-links", str(wheelhouse)]
        return await real_run_pip(
            self,
            argv,
            operation,
            ensure_venv,
            cwd=cwd,
            env=env,
            append=append,
        )

    monkeypatch.setattr(gen.GenericVenvInstaller, "_run_pip", pip_with_wheelhouse)

    # Activation's runtime side effects are out of scope here (they start the
    # shared proxy manager); the ACTIVE ROW is what the launch compiler consumes.
    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        return None

    monkeypatch.setattr(gen, "activate_engine_version", fake_activate)

    lifecycle_snapshot = _take_lifecycle_snapshot(ev.LIFECYCLE)
    ev.register_definition_lifecycles()
    assert "launchcraft" in ev.LIFECYCLE

    async def settle_journal() -> None:
        """Drain the store thread exactly like the app's request gate does.

        Successful operations are pruned in the same write by design
        (backend/operations/retention.py), so a finished start/stop leaves
        no durable row; the journal recorder below captures what was
        actually submitted to disk.
        """
        from backend.store_io import drain_store_io, persistence_status

        await drain_store_io()
        status = persistence_status()
        assert status["latest_failure"] is None, status

    # Record every operation row submitted to the journal (the supervisor
    # writes through store.upsert_operation) so success receipts stay visible
    # even though retention prunes finished successes from the document.
    journal: list[dict] = []
    real_upsert = store.upsert_operation

    def recording_upsert(operation: dict) -> dict:
        journal.append(dict(operation))
        return real_upsert(operation)

    monkeypatch.setattr(store, "upsert_operation", recording_upsert)

    # The app's startup gate is what publishes YAML-only engines into
    # VALID_ENGINE_IDS (importing the loader cannot); run boot's real gate and
    # restore the derived sets afterwards so other tests see pristine globals.
    from backend.engines import registry as registry_module
    from backend.engines.definitions.startup_check import verify_engine_definitions

    derived = {
        name: set(getattr(registry_module, name))
        for name in (
            "VALID_ENGINE_IDS",
            "EMBEDDINGS_ENGINE_IDS",
            "NATIVE_ENGINE_IDS",
            "GGUF_ENGINE_IDS",
            "HF_SNAPSHOT_ENGINE_IDS",
        )
    }
    engine_registry_snapshot = dict(registry_module.ENGINE_REGISTRY)
    verify_engine_definitions()
    assert "launchcraft" in registry_module.VALID_ENGINE_IDS

    registry = None
    dataplane = None
    proxy_asgi_client = None
    try:
        app = FastAPI()
        app.include_router(router, prefix="/api/engines")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            # 1) pin the version offline (the pypi-kind resolver uses the
            # build setting; no network is touched).
            response = await client.put(
                "/api/engines/launchcraft/build-settings", json={"pip_version": "0.1.0"}
            )
            assert response.status_code == 200

            # 2) real install through the generic verb. No version in the
            # request: it must resolve from the SAVED build settings, and the
            # snapshot ledger must record that resolution.
            response = await client.post("/api/engines/launchcraft/install", json={})
            assert response.status_code == 200
            assert response.json()["task_id"]

            status = None
            for _ in range(600):
                status = (await client.get("/api/engines/launchcraft/status")).json()
                if not status.get("operation"):
                    break
                await asyncio.sleep(0.1)
            assert status is not None
            assert not status.get("operation"), status
            assert status.get("installed") is True, status
            assert status.get("version") == "0.1.0", status

            # 3) the install is REAL: a venv python and the console script
            installer = ev.LIFECYCLE["launchcraft"].get_installer()
            venv_path = str(installer._venv_path)
            console_script = os.path.join(venv_path, "bin", "launchcraft-server")
            assert os.path.isfile(os.path.join(venv_path, "bin", "python"))
            assert os.path.isfile(console_script), os.listdir(os.path.join(venv_path, "bin"))

            # 3b) snapshot-backed install: the ledger records the RESOLVED
            # dispatch inputs (version from saved settings) and, once the
            # installation exists, a published row associated with THAT
            # version (so a retry selects by installation, not by "latest").
            from backend.engines.definitions.loader import load_default_engine_definitions

            snaps = store.list_install_snapshots("launchcraft")
            installed_definition = load_default_engine_definitions().definitions["launchcraft"]
            scheduled = [row for row in snaps if row.get("phase") == "scheduled"]
            published = [row for row in snaps if row.get("phase") == "published"]
            assert scheduled and scheduled[-1]["resolved_inputs"]["version"] == "0.1.0"
            assert scheduled[-1]["definition_digest"] == installed_definition.definition_digest
            assert published and published[-1]["version"] == "0.1.0"
            assert published[-1]["resolved_inputs"]["package_version"] == "0.1.0"
            assert published[-1]["definition_digest"] == installed_definition.definition_digest

            # 4) make the version the ACTIVE row (what activation would do)
            store.update_engine_version("launchcraft", "0.1.0", {"venv_path": venv_path})
            store.set_active_engine_version("launchcraft", "0.1.0")

        store.add_model(
            {
                "id": "lc-model",
                "name": "Launchcraft Model",
                "display_name": "Launchcraft Model",
                "huggingface_id": "fixture/model",
                "format": "safetensors",
                # canonical stored config shape (as the config API persists it)
                "config": {"engine": "launchcraft", "engines": {"launchcraft": {}}},
            }
        )
        model = store.get_model("lc-model")
        assert model is not None

        # 4b) DISCOVERY (gate G5): the real scan route runs the YAML-declared
        # probe (--scan-help — NOT --help, so success proves the declared
        # prefix ran), parses it with the existing llama_help parser, and
        # writes the catalog. The param registry then serves it to ModelConfig,
        # and runnability flips ONLY by the definition's active_path_fields.
        from backend.engines.registry import active_engine_row_is_runnable
        from backend.routes import llama_versions as llama_versions_routes

        api = FastAPI()
        api.include_router(models_routes.router, prefix="/api/models")
        api.include_router(llama_versions_routes.router, prefix="/api/llama-versions")
        async with AsyncClient(
            transport=ASGITransport(app=api), base_url="http://studio"
        ) as api_client:
            scanned = await api_client.post(
                "/api/llama-versions/scan-engine-params",
                json={"engine": "launchcraft"},
            )
            assert scanned.status_code == 200, scanned.text
            scan_body = scanned.json()
            assert scan_body["ok"] is True, scan_body
            assert scan_body["param_count"] >= 3, scan_body

            registry_payload = await api_client.get(
                "/api/models/param-registry", params={"engine": "launchcraft"}
            )
            assert registry_payload.status_code == 200, registry_payload.text
            registry_body = registry_payload.json()
            assert registry_body["scan_error"] is None, registry_body
            served = {
                param["key"] for section in registry_body["sections"] for param in section["params"]
            }
            assert "max_batch_size" in served, served

        assert active_engine_row_is_runnable("launchcraft", {"venv_path": venv_path}) is True
        assert active_engine_row_is_runnable("launchcraft", {}) is False
        assert active_engine_row_is_runnable("launchcraft", {"binary_path": "/nope"}) is False

        # 5) the PRODUCTION compile path (not the profile helper): engine
        # identity, proxy model id, and the YAML-declared argv.
        compiled = await asyncio.to_thread(compile_model_runtime, model)
        assert compiled.proxy.engine_id == "launchcraft"
        assert compiled.proxy.model_id == "fixture-model"
        assert compiled.proxy.catalog_id == "lc-model"
        spec = compiled.launch
        assert spec.executable == console_script
        assert spec.argv == ARGV_GOLDEN
        assert spec.env.set, "profile env defaults must reach the LaunchSpec"

        # 6) REAL ProcessRegistry + the real manager's resolver, behind the
        # real Studio dataplane ASGI app.
        manager = object.__new__(StudioProxyManager)
        manager._route_lock = threading.Lock()
        manager._route_cursor = {}
        manager._catalog_lock = threading.RLock()
        manager._compiled_cache = None

        async def factory(model_id):
            current = await asyncio.to_thread(store.get_model, "lc-model")
            runtime = await asyncio.to_thread(compile_model_runtime, current)
            return runtime.launch, runtime.revision, None

        manager.registry = registry = ProcessRegistry(factory, readiness_timeout=60.0)
        dataplane = StudioDataPlane(registry, manager._resolve)

        # The route's ProxyClient really speaks HTTP into the dataplane; only
        # the transport is in-process (no listening socket needed).
        proxy_asgi_client = httpx.AsyncClient(
            transport=ASGITransport(app=dataplane), base_url="http://proxy"
        )
        monkeypatch.setattr(
            "backend.proxy.client.shared_proxy_client", lambda base_url: proxy_asgi_client
        )
        async with AsyncClient(
            transport=ASGITransport(app=api), base_url="http://studio"
        ) as api_client:
            # 7) POST /api/models/{id}/start — real route, real journal, real
            # process spawn through the dataplane's lazy acquire.
            started = await api_client.post("/api/models/lc-model/start")
            assert started.status_code == 200, started.text
            payload = started.json()
            assert payload["object"] == "list"
            assert payload["data"][0]["id"] == "fixture/model"

            generation = registry.active("fixture-model")
            assert generation is not None and generation.ready is True
            port = generation.port
            pid = generation.process.pid
            assert generation.process.poll() is None

            # durable journal: the supervisor submitted a terminal success
            # for the start, and no admission fence survives it (a success is
            # pruned from operations.yaml by design, so a retry is admitted).
            await settle_journal()
            start_receipts = [row for row in journal if row.get("kind") == "model_start"]
            assert start_receipts, "start must write a journal receipt"
            assert start_receipts[-1]["status"] == "succeeded"
            assert start_receipts[-1]["detail"]["proxy_name"] == "fixture-model"
            from backend.operations.action_recovery import open_action

            assert open_action(store.list_operations(), "model_start:lc-model") is None

            # 8) argv proven from INSIDE the running process: prefix + model
            # binding + port flag with the ephemeral port the registry chose.
            async with httpx.AsyncClient(timeout=5.0) as probe:
                health = await probe.get(f"http://127.0.0.1:{port}/health")
                assert health.status_code == 200
                assert health.json() == {"status": "ok"}
                state = (await probe.get(f"http://127.0.0.1:{port}/__state")).json()
            assert state["pid"] == pid
            assert state["argv"][1:] == ["--model", "fixture/model", "--port", str(port)]
            assert state["model_ref"] == "fixture/model"

            # 9) ONE streamed chat completion through the Studio dataplane into
            # the YAML-launched server; bytes are the fixture's real SSE.
            async with AsyncClient(
                transport=ASGITransport(app=dataplane), base_url="http://proxy"
            ) as proxy_client:
                chat = await proxy_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fixture-model",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": True,
                    },
                )
            assert chat.status_code == 200, chat.text
            assert chat.headers["content-type"].startswith("text/event-stream")
            # The fixture streams two real content deltas, then a finish chunk.
            assert '"content":"launchcraft"' in chat.text
            assert '"content":"-ok"' in chat.text
            assert '"finish_reason":"stop"' in chat.text
            assert "[DONE]" in chat.text
            assert chat.headers.get("x-request-id")

            # The completion really reached the engine process (its own hit
            # counter), not a proxy cache.
            async with httpx.AsyncClient(timeout=5.0) as probe:
                hits = (await probe.get(f"http://127.0.0.1:{port}/__state")).json()["hits"]
            assert hits.get("/v1/models", 0) >= 1, hits  # the start passthrough
            assert hits.get("/v1/chat/completions", 0) >= 1, hits

            # 10) POST /api/models/{id}/stop — real route → dataplane unload →
            # drain; journal receipt; zero orphans.
            stopped = await api_client.post("/api/models/lc-model/stop")
            assert stopped.status_code == 200, stopped.text
            assert stopped.json() == {"model": "fixture-model", "stopped": True}

            for _ in range(200):
                if generation.process.poll() is not None:
                    break
                await asyncio.sleep(0.05)
            assert generation.process.poll() is not None, "process survived stop"
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert registry.active("fixture-model") is None

            await settle_journal()
            stop_receipts = [row for row in journal if row.get("kind") == "model_stop"]
            assert stop_receipts, "stop must write a journal receipt"
            assert stop_receipts[-1]["status"] == "succeeded"
            assert open_action(store.list_operations(), "model_stop:lc-model") is None

            # 11) lifecycle removal through the SAME definition-owned
            # installer: drop the model reference first (the real removal
            # route does the same through reference retirement), then the
            # remove recipe deletes the version row and venv.
            assert store.delete_model("lc-model") is True
            removed = await installer.remove()
            assert removed.get("message") or removed.get("task_id"), removed
            for _ in range(200):
                if not store.get_engine_versions("launchcraft"):
                    break
                await asyncio.sleep(0.05)
            assert store.get_engine_versions("launchcraft") == [], (
                installer._last_error,
                installer.status(),
            )
            assert store.get_active_engine_version("launchcraft") is None
            assert not os.path.exists(venv_path)
    finally:
        if proxy_asgi_client is not None:
            await proxy_asgi_client.aclose()
        if dataplane is not None:
            await dataplane.close()
        if registry is not None:
            await registry.close()
        _restore_lifecycle_snapshot(ev.LIFECYCLE, lifecycle_snapshot)
        registry_module.ENGINE_REGISTRY.clear()
        registry_module.ENGINE_REGISTRY.update(engine_registry_snapshot)
        for name, snapshot in derived.items():
            target = getattr(registry_module, name)
            target.clear()
            target.update(snapshot)


@pytest.mark.asyncio
async def test_launchforge_gguf_module_profile_launches_end_to_end(tmp_path, monkeypatch):
    """Second, materially different persona (review follow-up).

    Unlike the console-script fixture this one DECLARES gguf artifacts (so
    it joins GGUF_ENGINE_IDS and would previously die in the legacy GGUF
    compiler), launches a venv python with a module prefix, binds the model
    POSITIONALLY, runs with cwd=venv_root, and requires an env variable it
    satisfies through an engine default. Real install → compile → spawn →
    health → streamed chat through the proxy → drain.
    """
    import backend.data_store as ds
    from backend.engines import registry as registry_module
    from backend.engines.definitions.startup_check import verify_engine_definitions
    from backend.engines.python import generic as gen
    from backend.engines.python import installer as venv_installer
    from backend.proxy import profile_launch
    from backend.proxy.launch_spec import compile_model_runtime
    from backend.proxy.studio.dataplane import StudioDataPlane
    from backend.proxy.studio.manager import StudioProxyManager
    from backend.proxy.studio.registry import ProcessRegistry
    from backend.routes import engine_versions as ev
    from backend.routes import models as models_routes
    from backend.routes.engine_lifecycle import router

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(venv_installer, "studio_data_dir", lambda: str(tmp_path / "data"))
    monkeypatch.setattr(venv_installer, "get_store", lambda: store)
    monkeypatch.setattr(gen, "get_store", lambda: store)
    monkeypatch.setattr(ev, "get_store", lambda: store)
    monkeypatch.setattr(ds, "get_store", lambda: store)
    monkeypatch.setattr(ds, "_store", store)
    monkeypatch.setattr("backend.proxy.studio.manager.get_store", lambda: store)
    monkeypatch.setattr(models_routes, "get_store", lambda: store)
    import backend.venv_install_settings as vis

    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))

    wheelhouse = tmp_path / "wheelhouse"
    wheelhouse.mkdir()
    assert _build_wheel(wheelhouse).is_file()
    real_run_pip = gen.GenericVenvInstaller._run_pip

    async def pip_with_wheelhouse(
        self, args, operation, ensure_venv=True, *, cwd=None, env=None, append=True
    ):
        argv = list(args)
        if argv and argv[0] == "install":
            argv = [*argv, "--no-index", "--find-links", str(wheelhouse)]
        return await real_run_pip(
            self,
            argv,
            operation,
            ensure_venv,
            cwd=cwd,
            env=env,
            append=append,
        )

    monkeypatch.setattr(gen.GenericVenvInstaller, "_run_pip", pip_with_wheelhouse)

    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        return None

    monkeypatch.setattr(gen, "activate_engine_version", fake_activate)

    lifecycle_snapshot = _take_lifecycle_snapshot(ev.LIFECYCLE)
    ev.register_definition_lifecycles()
    derived = {
        name: set(getattr(registry_module, name))
        for name in (
            "VALID_ENGINE_IDS",
            "EMBEDDINGS_ENGINE_IDS",
            "NATIVE_ENGINE_IDS",
            "GGUF_ENGINE_IDS",
            "HF_SNAPSHOT_ENGINE_IDS",
        )
    }
    engine_registry_snapshot = dict(registry_module.ENGINE_REGISTRY)
    verify_engine_definitions()
    assert "launchforge" in registry_module.VALID_ENGINE_IDS
    # The trap is LIVE: declaring gguf artifacts joins the derived set, which
    # is exactly what used to route a YAML engine into the legacy compiler.
    assert "launchforge" in registry_module.GGUF_ENGINE_IDS

    registry = None
    dataplane = None
    try:
        app = FastAPI()
        app.include_router(router, prefix="/api/engines")
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://testserver"
        ) as client:
            response = await client.put(
                "/api/engines/launchforge/build-settings", json={"pip_version": "0.1.0"}
            )
            assert response.status_code == 200
            response = await client.post(
                "/api/engines/launchforge/install", json={"version": "0.1.0"}
            )
            assert response.status_code == 200
            status = None
            for _ in range(600):
                status = (await client.get("/api/engines/launchforge/status")).json()
                if not status.get("operation"):
                    break
                await asyncio.sleep(0.1)
            assert status is not None and not status.get("operation"), status
            assert status.get("installed") is True and status.get("version") == "0.1.0"

            venv_path = str(ev.LIFECYCLE["launchforge"].get_installer()._venv_path)
            assert os.path.isfile(os.path.join(venv_path, "bin", "python"))
            store.update_engine_version("launchforge", "0.1.0", {"venv_path": venv_path})
            store.set_active_engine_version("launchforge", "0.1.0")

        store.add_model(
            {
                "id": "lf-model",
                "name": "Launchforge Model",
                "display_name": "Launchforge Model",
                "huggingface_id": "fixture/model",
                "format": "gguf",
                "config": {"engine": "launchforge", "engines": {"launchforge": {}}},
            }
        )
        model = store.get_model("lf-model")
        assert model is not None

        # Fix 1 pin: the engine default satisfies the requirement, but an
        # explicit unset makes the FINAL launch env lack it — typed refusal
        # instead of a process that dies without the variable.
        with pytest.raises(Exception) as excinfo:
            profile_launch.compile_profile_launch(
                model,
                {"runtime_env_unset": ["LAUNCHFORGE_MODE"]},
                "fixture-model",
                "launchforge",
            )
        assert getattr(excinfo.value, "field", "") == "LAUNCHFORGE_MODE"

        # Fix 3 pin: gguf-declaring definition → profile compiler wins.
        compiled = await asyncio.to_thread(compile_model_runtime, model)
        assert compiled.launch.engine_id == "launchforge"
        assert compiled.launch.executable == os.path.join(venv_path, "bin", "python")
        assert compiled.launch.argv == [
            "-m",
            "launchcraft_fixture.server",
            "fixture/model",
            "--port",
            {"runtime": "port"},
        ]
        assert compiled.launch.cwd == venv_path
        assert compiled.launch.env.set.get("LAUNCHFORGE_MODE") == "fixture"
        assert compiled.proxy.use_model_name == "fixture/model"

        manager = object.__new__(StudioProxyManager)
        manager._route_lock = threading.Lock()
        manager._route_cursor = {}
        manager._catalog_lock = threading.RLock()
        manager._compiled_cache = None

        async def factory(model_id):
            current = await asyncio.to_thread(store.get_model, "lf-model")
            runtime = await asyncio.to_thread(compile_model_runtime, current)
            return runtime.launch, runtime.revision, None

        manager.registry = registry = ProcessRegistry(factory, readiness_timeout=60.0)
        dataplane = StudioDataPlane(registry, manager._resolve)
        # The start route's ProxyClient really speaks HTTP into the dataplane.
        proxy_asgi_client = httpx.AsyncClient(
            transport=ASGITransport(app=dataplane), base_url="http://proxy"
        )
        monkeypatch.setattr(
            "backend.proxy.client.shared_proxy_client", lambda base_url: proxy_asgi_client
        )
        api = FastAPI()
        api.include_router(models_routes.router, prefix="/api/models")
        async with AsyncClient(
            transport=ASGITransport(app=api), base_url="http://studio"
        ) as api_client:
            started = await api_client.post("/api/models/lf-model/start")
            assert started.status_code == 200, started.text
            assert started.json()["data"][0]["id"] == "fixture/model"

            generation = registry.active("fixture-model")
            assert generation is not None and generation.ready is True
            port = generation.port
            pid = generation.process.pid

            async with httpx.AsyncClient(timeout=5.0) as probe:
                state = (await probe.get(f"http://127.0.0.1:{port}/__state")).json()
            # The OS argv proves the module prefix ran; sys.argv (rewritten by
            # CPython for -m) proves the positional model + port substitution.
            assert state["os_argv"][1:] == [
                "-m",
                "launchcraft_fixture.server",
                "fixture/model",
                "--port",
                str(port),
            ]
            assert state["argv"][1:] == ["fixture/model", "--port", str(port)]
            assert os.path.realpath(state["cwd"]) == os.path.realpath(venv_path)
            assert state["launchforge_mode"] == "fixture"
            assert state["model_ref"] == "fixture/model"

            async with AsyncClient(
                transport=ASGITransport(app=dataplane), base_url="http://proxy"
            ) as proxy_client:
                chat = await proxy_client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "fixture-model",
                        "messages": [{"role": "user", "content": "ping"}],
                        "stream": True,
                    },
                )
            assert chat.status_code == 200, chat.text
            assert '"content":"-ok"' in chat.text
            assert "[DONE]" in chat.text

            stopped = await api_client.post("/api/models/lf-model/stop")
            assert stopped.status_code == 200, stopped.text
            for _ in range(200):
                if generation.process.poll() is not None:
                    break
                await asyncio.sleep(0.05)
            assert generation.process.poll() is not None, "process survived stop"
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
            assert registry.active("fixture-model") is None
    finally:
        if proxy_asgi_client is not None:
            await proxy_asgi_client.aclose()
        if dataplane is not None:
            await dataplane.close()
        if registry is not None:
            await registry.close()
        _restore_lifecycle_snapshot(ev.LIFECYCLE, lifecycle_snapshot)
        registry_module.ENGINE_REGISTRY.clear()
        registry_module.ENGINE_REGISTRY.update(engine_registry_snapshot)
        for name, snapshot in derived.items():
            target = getattr(registry_module, name)
            target.clear()
            target.update(snapshot)
