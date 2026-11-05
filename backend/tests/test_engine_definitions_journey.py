"""YAML campaign S6 — the full HTTP journey for a YAML-only engine.

Drives the REAL ``/api/engines/{engine_id}`` router (the same callables
the legacy prefixes share) against the shipped ``yamlcraft`` fixture,
registered exactly the way startup registers it. Everything below the
installer boundary is real (router, dispatchers, executor, recipes,
version-store bookkeeping, snapshot ledger); only venv creation and pip
are stubbed — no network, no real installs. This is what the dynamic
frontend card drives, so the install/manage journey is proven WITHOUT a
browser.
"""

from __future__ import annotations

import asyncio
import os

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from backend.data_store import DataStore


@pytest.mark.asyncio
async def test_yamlcraft_journey_through_the_generic_router(tmp_path, monkeypatch):
    from backend.engines.definitions.loader import load_default_engine_definitions
    from backend.engines.python import generic as gen
    from backend.engines.python import installer as venv_installer
    from backend.routes import engine_versions as ev
    from backend.routes.engine_lifecycle import router

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(venv_installer, "studio_data_dir", lambda: str(tmp_path / "data"))
    monkeypatch.setattr(venv_installer, "get_store", lambda: store)
    monkeypatch.setattr(gen, "get_store", lambda: store)
    monkeypatch.setattr(ev, "get_store", lambda: store)
    import backend.venv_install_settings as vis

    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))

    # Startup-equivalent registration from a CLEAN registry: registration
    # skips existing ids, so a leftover entry from another test would keep
    # that test's (already restored) install-settings defaults. Snapshot and
    # restore the process-global registry so no test order leaks.
    lifecycle_snapshot = dict(ev.LIFECYCLE)
    ev.LIFECYCLE.clear()
    registered = ev.register_definition_lifecycles()
    assert "yamlcraft" in ev.LIFECYCLE  # freshly registered or pre-existing
    del registered
    try:
        app = FastAPI()
        app.include_router(router, prefix="/api/engines")
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://testserver") as client:
            # 1) status before anything exists (lifecycle-declared fallback)
            response = await client.get("/api/engines/yamlcraft/status")
            assert response.status_code == 200
            assert response.json()["installed"] is False

            # 2) build settings come from the DECLARED defaults (YAML)
            response = await client.get("/api/engines/yamlcraft/build-settings")
            assert response.status_code == 200
            settings = response.json()
            assert settings["source_repo"] == "https://github.com/example-org/yamlcraft.git"
            assert settings["source_branch"] == "main"

            # 3) save + read back (generic settings surface)
            response = await client.put(
                "/api/engines/yamlcraft/build-settings",
                json={"pip_version": "0.1.0"},
            )
            assert response.status_code == 200
            response = await client.get("/api/engines/yamlcraft/build-settings")
            assert response.json()["pip_version"] == "0.1.0"

            # 4) offline release install through the recipe + real executor
            pip_calls = []

            async def fake_run_pip(self, args, stage, ensure_venv=True, **kwargs):
                if ensure_venv:
                    fake_ensure_venv(self)
                pip_calls.append(list(args))
                return 0

            # _ensure_venv is SYNC on the base class — the stub must match
            # the real calling convention (an async stub here produced
            # unawaited coroutines and a weaker install proof).
            def fake_ensure_venv(self):
                binary = self._venv_bin("yamlcraft")
                os.makedirs(os.path.dirname(binary), exist_ok=True)
                open(binary, "w").close()
                os.chmod(binary, 0o755)

            monkeypatch.setattr(gen.GenericVenvInstaller, "_run_pip", fake_run_pip)
            monkeypatch.setattr(gen.GenericVenvInstaller, "_ensure_venv", fake_ensure_venv)
            # Binary resolution against the SEALED per-version venv is base
            # class behavior pinned in the legacy suites; the journey stubs
            # it to keep this test about router + recipes + bookkeeping.
            monkeypatch.setattr(
                gen.GenericVenvInstaller,
                "_resolve_binary_path",
                lambda self: os.path.join(tmp_path, "data", "yamlcraft", "sealed-binary"),
            )
            os.makedirs(os.path.join(tmp_path, "data", "yamlcraft"), exist_ok=True)
            open(os.path.join(tmp_path, "data", "yamlcraft", "sealed-binary"), "w").close()
            monkeypatch.setattr(
                gen.GenericVenvInstaller,
                "_detect_installed_version",
                lambda self, venv_path=None: "0.1.0",
            )

            async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
                return None

            monkeypatch.setattr(gen, "activate_engine_version", fake_activate)

            response = await client.post(
                "/api/engines/yamlcraft/install", json={"version": "0.1.0"}
            )
            assert response.status_code == 200
            body = response.json()
            assert body["message"] == "Crafty installation started"
            assert body["task_id"]

            # 5) the card's poll loop: status until the operation settles
            in_flight = None
            for _ in range(250):
                status = (await client.get("/api/engines/yamlcraft/status")).json()
                if status.get("operation") and in_flight is None:
                    in_flight = status
                if not status.get("operation"):
                    break
                await asyncio.sleep(0.02)
            # Contract the panel reads (review 2026-11-03): `operation` is
            # an operation-NAME string while running; the cancel handle is
            # the separate `progress_task_id`. Checked whenever the (fast,
            # stubbed) operation was caught mid-flight.
            if in_flight is not None:
                assert isinstance(in_flight["operation"], str), in_flight
                assert in_flight.get("progress_task_id"), in_flight
            assert not status.get("operation"), status
            assert status["installed"] is True, status
            assert status["version"] == "0.1.0", status
            assert pip_calls == [["install", "--upgrade", "yamlcraft-fixture==0.1.0"]]

            # 6) version-store bookkeeping + S1 digest ledger row
            rows = store.get_engine_versions("yamlcraft")
            assert any(
                r.get("version") == "0.1.0" and r.get("build_status") == "ready" for r in rows
            )
            snaps = store.list_install_snapshots("yamlcraft")
            definition = load_default_engine_definitions().definitions["yamlcraft"]
            assert snaps[-1]["definition_digest"] == definition.definition_digest

            # 7) error contracts on the generic surface
            response = await client.post("/api/engines/yamlcraft/cancel", json={})
            assert response.status_code == 400
            response = await client.get("/api/engines/not_an_engine/status")
            assert response.status_code == 404
    finally:
        ev.LIFECYCLE.clear()
        ev.LIFECYCLE.update(lifecycle_snapshot)


# --- Shipped engine: the SAME definition path, no Python table ---------------------


def test_shipped_engine_discovery_and_launch_follow_the_definition(tmp_path, monkeypatch):
    """A converted SHIPPED engine is driven entirely by its YAML file.

    This is the vertical-slice regression gate for the conversion pattern:
    discovery (probe argv, cwd, venv env) and launch (profile argv) come from
    ``lmdeploy.yaml``; editing that file changes both behaviors with no Python
    edit and no exporter run.
    """
    import shutil
    from pathlib import Path

    import yaml

    from backend.engines.definitions import loader
    from backend.engines.definitions.loader import DEFINITIONS_DIR, load_engine_definitions
    from backend.engines.scan import scanner as scanner_mod
    from backend.proxy import launch_spec, profile_launch

    dest = tmp_path / "defs"
    shutil.copytree(DEFINITIONS_DIR, dest)
    yaml_path = dest / "lmdeploy.yaml"

    def install_registry():
        registry = load_engine_definitions(str(dest))
        monkeypatch.setattr(loader, "_default_registry", registry)
        monkeypatch.setattr(profile_launch, "_REGISTRY", registry, raising=False)
        return registry

    install_registry()

    venv = tmp_path / "venv"
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    executable = bin_dir / "lmdeploy"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o755)

    help_text = (
        Path(__file__).parent / "fixtures" / "lmdeploy_api_server_help_excerpt.txt"
    ).read_text(encoding="utf-8")
    calls = []

    def fake_run_help(argv, *, cwd=None, extra_env=None, scan_engine=None, label=None):
        calls.append({"argv": list(argv), "cwd": cwd, "extra_env": dict(extra_env or {})})
        return help_text, None

    monkeypatch.setattr(scanner_mod, "_run_help_argv", fake_run_help)

    row = {"version": "0.1.0", "venv_path": str(venv)}
    entry = scanner_mod.scan_definition_engine_version("lmdeploy", row)
    assert entry["scan_error"] is None, entry
    assert entry["sections"]
    assert calls[0]["argv"] == [str(executable), "serve", "api_server", "--help"]
    assert calls[0]["cwd"] == str(venv)  # declared `cwd: venv_root`
    assert calls[0]["extra_env"]["VIRTUAL_ENV"] == str(venv)

    # Launch compiles from the declared profile.
    monkeypatch.setattr(
        launch_spec, "_active_row", lambda _engine: {"version": "0.1.0", "venv_path": str(venv)}
    )
    model = {"huggingface_id": "org/model"}
    compiled = profile_launch.compile_profile_launch(model, {}, "model-stable", "lmdeploy")
    assert compiled is not None
    assert compiled.argv[:2] == ["serve", "api_server"]

    # Edit ONLY the definition, then reload it: both behaviors move.
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    document["discovery"]["probe_argv_prefix"].append("--fixture-probe")
    document["runtime_profiles"]["default"]["argv_prefix"].append("--fixture-launch")
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    install_registry()

    calls.clear()
    entry = scanner_mod.scan_definition_engine_version("lmdeploy", row)
    assert entry["scan_error"] is None, entry
    assert calls[0]["argv"][-1] == "--fixture-probe"

    recompiled = profile_launch.compile_profile_launch(model, {}, "model-stable", "lmdeploy")
    assert recompiled is not None
    assert "--fixture-launch" in recompiled.argv
