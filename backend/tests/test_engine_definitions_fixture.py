"""YAML campaign S4 — the `yamlcraft` fixture end to end (control plane).

The fixture ships in the definitions dir: startup tier-check registers it
into the derived registry, generic lifecycle verbs run it through
``GenericVenvInstaller``, and the S1 ledger snapshots the YAML digest.
Gates: G1 startup byte-identity of the GENERATED nine + fixture
registration; G2 generic verbs on a YAML-only engine; G3 install via the
shared primitives (offline — stubbed venv/pip, real executor); G4 install
ledger row carries the fixture's definition digest; G5 architecture guard:
the literal string "yamlcraft" exists in NO non-test, non-fixture,
non-doc source file.

Launch/discovery is deliberately OUT of scope (separate runtime campaign):
the fixture's runtime/scanner kinds are vocabulary placeholders.
"""

from __future__ import annotations

import os

import pytest

from backend.data_store import DataStore
from backend.engines import registry as registry_module
from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.definitions.startup_check import verify_engine_definitions
from backend.engines.registry import (
    SHIPPED_ENGINE_IDS,
    engine_registry_payload,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_YAML = os.path.join(REPO_ROOT, "backend", "engines", "definitions", "yamlcraft.yaml")


@pytest.fixture
def registry_sandbox():
    dict_copy = dict(registry_module.ENGINE_REGISTRY)
    sets = {
        name: set(getattr(registry_module, name))
        for name in (
            "VALID_ENGINE_IDS",
            "EMBEDDINGS_ENGINE_IDS",
            "NATIVE_ENGINE_IDS",
            "GGUF_ENGINE_IDS",
            "HF_SNAPSHOT_ENGINE_IDS",
        )
    }
    yield
    registry_module.ENGINE_REGISTRY.clear()
    registry_module.ENGINE_REGISTRY.update(dict_copy)
    for name, snapshot in sets.items():
        target = getattr(registry_module, name)
        target.clear()
        target.update(snapshot)


# --- G5 architecture guard -----------------------------------------------------------


def test_yamlcraft_literal_never_appears_in_engine_code():
    """Onboarding MUST NOT require code edits: if this fails, someone
    hardcoded the fixture engine instead of letting its YAML speak."""
    allowed = {
        FIXTURE_YAML,  # the definition IS the engine
    }
    offenders = []
    for root, dirs, files in os.walk(REPO_ROOT):
        dirs[:] = [
            d
            for d in dirs
            if d not in {"__pycache__", "node_modules", ".git", "dist", ".venv", "venv"}
        ]
        root_path = os.path.abspath(root)
        rel = os.path.relpath(root_path, REPO_ROOT).replace("\\", "/")
        if rel.startswith("docs") or rel.startswith("backend/tests"):
            continue
        if "tests" in rel.split("/") and rel.startswith("frontend"):
            continue
        for name in files:
            if not name.endswith((".py", ".js", ".ts", ".vue", ".json", ".yaml", ".yml")):
                continue
            # Frontend tests are CO-LOCATED (*.test.js under src/) — they
            # replay live descriptor payloads (data, not engine code).
            if ".test." in name:
                continue
            path = os.path.abspath(os.path.join(root, name))
            if path in allowed:
                continue
            try:
                with open(path, encoding="utf-8") as fh:
                    content = fh.read()
            except (OSError, UnicodeDecodeError):
                continue
            if "yamlcraft" in content:
                offenders.append(os.path.relpath(path, REPO_ROOT).replace("\\", "/"))
    assert not offenders, f"yamlcraft hardcoded outside YAML/tests/docs: {sorted(offenders)}"


# --- G1 startup + registry derivation -------------------------------------------------


def test_startup_registers_the_fixture_and_descriptor_flows(registry_sandbox):
    count = verify_engine_definitions()
    # Every shipped engine (exporter-generated + definition-owned) plus the
    # three YAML-only fixtures (control plane + two launch personas).
    assert count == len(SHIPPED_ENGINE_IDS) + 4
    assert "yamlcraft" in registry_module.ENGINE_REGISTRY
    assert "yamlcraft" in registry_module.VALID_ENGINE_IDS
    definition = load_default_engine_definitions().definitions["yamlcraft"]
    descriptor = engine_registry_payload()
    payload = {engine["id"]: engine for engine in descriptor["engines"]}["yamlcraft"]
    assert payload == definition.to_api_dict()  # one identity system (Q2)
    assert payload["label"] == "Crafty"


# --- G2 + G3 + G4: generic verbs, offline persona install, ledger -------------------


@pytest.mark.asyncio
async def test_generic_verbs_run_the_yaml_engine_offline(tmp_path, monkeypatch, registry_sandbox):
    from backend.engines.python import generic as gen
    from backend.engines.python import installer as venv_installer
    from backend.routes import engine_versions as ev
    import backend.venv_install_settings as vis

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(venv_installer, "studio_data_dir", lambda: str(tmp_path / "data"))
    monkeypatch.setattr(venv_installer, "get_store", lambda: store)
    monkeypatch.setattr(gen, "get_store", lambda: store)
    monkeypatch.setattr(ev, "get_store", lambda: store)
    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))
    monkeypatch.setattr(ev, "LIFECYCLE", {})

    activations = []

    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        activations.append((engine_id, version_name))

    monkeypatch.setattr(gen, "activate_engine_version", fake_activate)

    registered = ev.register_definition_lifecycles()
    # Generated engines skipped; every converted shipped engine that declares a
    # lifecycle and both YAML-only fixtures get their record from definitions.
    assert registered == [
        "1cat_vllm",
        "launchcraft",
        "launchforge",
        "lmdeploy",
        "sglang",
        "sglang_sxm2",
        "sglang_v100",
        "vllm",
        "yamlcraft",
    ]
    lc = ev.LIFECYCLE["yamlcraft"]

    # G2: status verb (generic machinery, definition-provided fallback)
    status = await ev.lifecycle_status(lc)
    assert status["installed"] is False

    # G3: install through the definition's release recipe (offline stubs)
    pip_calls = []

    async def fake_run_pip(self, args, stage, **kwargs):
        pip_calls.append(list(args))
        return 0

    monkeypatch.setattr(gen.GenericVenvInstaller, "_run_pip", fake_run_pip)
    monkeypatch.setattr(
        gen.GenericVenvInstaller,
        "_ensure_venv",
        lambda self: os.makedirs(self._venv_path, exist_ok=True),
    )
    monkeypatch.setattr(
        gen.GenericVenvInstaller,
        "_detect_installed_version",
        lambda self, venv_path=None: "9.9.0",
    )

    response = await ev.lifecycle_install(lc, {"version": "9.9.0"})
    assert response["message"] == "Crafty installation started"

    loop_busy = 0
    while lc.get_installer()._operation is not None and loop_busy < 250:
        import asyncio

        await asyncio.sleep(0.02)
        loop_busy += 1
    installer = lc.get_installer()
    assert installer._last_error is None, installer._last_error
    assert pip_calls == [["install", "--upgrade", "yamlcraft-fixture==9.9.0"]]
    assert activations == [("yamlcraft", "9.9.0")]
    rows = store.get_engine_versions("yamlcraft")
    assert any(r.get("version") == "9.9.0" and r.get("build_status") == "ready" for r in rows)

    # G4: the S1 ledger froze WHICH definition served this install. Three rows
    # now: the dispatch-time record, the installer's PENDING record associated
    # with the pre-publication row name (so a failed attempt is still guarded),
    # and the PUBLISHED record for the final version.
    snaps = store.list_install_snapshots("yamlcraft")
    definition = load_default_engine_definitions().definitions["yamlcraft"]
    scheduled = [row for row in snaps if row.get("phase") == "scheduled"]
    published = [row for row in snaps if row.get("phase") == "published"]
    assert len(scheduled) == 2 and len(published) == 1
    assert all(row["definition_digest"] == definition.definition_digest for row in snaps)
    assert any(row["resolved_inputs"].get("version") == "9.9.0" for row in scheduled)
    pending = next(row for row in scheduled if row.get("version"))
    assert pending["version"] != "9.9.0"  # the timestamp-pip pre-publication name
    assert scheduled[0]["recipe_version"] == "schema-v2"
    assert published[0]["version"] == "9.9.0"
    assert published[0]["resolved_inputs"]["package_version"] == "9.9.0"


# --- S5: descriptor payload carries the routing facts the frontend needs ----------


@pytest.mark.asyncio
async def test_descriptor_route_advertises_generic_routing_for_authoritative_only(
    tmp_path, monkeypatch, registry_sandbox
):
    from backend.data_store import DataStore
    from backend.routes import engine_versions as ev
    from backend.routes import engines as engines_route

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(engines_route, "get_store", lambda: store)
    # Deterministic registry state (this suite may run before any lifespan).
    verify_engine_definitions()
    # Same call the app startup makes; idempotent if a prior test's lifespan
    # already registered the fixture (this mirrors production state).
    ev.register_definition_lifecycles()
    assert "yamlcraft" in ev.LIFECYCLE

    payload = await engines_route.list_engine_descriptors()
    by_id = {descriptor["id"]: descriptor for descriptor in payload["engines"]}

    craft = by_id["yamlcraft"]
    assert craft["generic_lifecycle"] is True
    assert set(craft["generic_verbs"]) == {
        "check-updates",
        "status",
        "build-settings",
        "install",
        "install-source",
        "remove",
        "cancel",
    }
    # Every shipped descriptor is definition-owned and advertises its routing
    # facts (additive keys only), so the UI can drive it without a local
    # descriptor.
    definition_owned = SHIPPED_ENGINE_IDS
    assert definition_owned
    for engine_id in definition_owned:
        if engine_id not in ev.LIFECYCLE:
            continue
        assert by_id[engine_id]["generic_lifecycle"] is True
        assert "install" in by_id[engine_id]["generic_verbs"]
