"""Generic native engine lifecycle (single-operator hardening WS3).

A YAML-only native engine — a definition with ``native_host`` plus a
``lifecycle`` block — must get the generic ``/api/engines/{id}`` verbs with no
legacy prefix and no engine-specific Python. This module pins:

* ``build_definition_lifecycle`` binds the installer from the declared native
  host (``get_native_installer``) instead of forcing the python-venv host;
* the generic verb dispatch (install/status/cancel/build-settings/remove)
  reaches that host, and an unresolvable host is loud instead of silently
  falling back to a venv installer;
* ``install-source`` is served only when the definition declares a ``source``
  recipe (405 otherwise) — and works when it does;
* ``register_definition_lifecycles`` registers the native record once, after
  the same startup verification every shipped definition passes.

The definition is a temp YAML file loaded through the production loader and the
native host is a stub: no data directory, network, subprocess or real engine is
touched.
"""

from __future__ import annotations

import pytest
import yaml
from fastapi import HTTPException

from backend.data_store import DataStore
from backend.engines.definitions import load_engine_definitions
from backend.engines.definitions.startup_check import (
    definition_tier,
    verify_engine_definitions,
)
from backend.routes import engine_versions as ev

ENGINE_ID = "nativecraft"
SOURCE_REPO = "https://github.com/example/nativecraft.git"
DECLARED_SETTINGS = {
    "source_repo": SOURCE_REPO,
    "source_branch": "main",
    "release_version": "",
}


class StubNativeInstaller:
    """Minimal native host: records generic-verb calls, touches nothing."""

    def __init__(self, engine_id: str = ENGINE_ID):
        self.engine_id = engine_id
        self.calls: list[tuple[str, dict]] = []

    @property
    def install_root(self) -> str:
        return "/nonexistent/nativecraft"

    @property
    def log_path(self) -> str:
        return "/nonexistent/nativecraft.log"

    def status(self) -> dict:
        self.calls.append(("status", {}))
        return {
            "engine": self.engine_id,
            "installed": False,
            "version": None,
            "install_dir": None,
            "log_path": self.log_path,
            "operation": None,
            "last_error": None,
        }

    async def install_release(
        self, version=None, force_reinstall=False, *, reuse_dir=None, existing_version=None
    ) -> dict:
        self.calls.append(("install", {"version": version, "force_reinstall": force_reinstall}))
        return {"message": f"{self.engine_id} installation started", "task_id": "t-1"}

    async def install_from_source(
        self, repo_url=None, branch="main", *, reuse_dir=None, existing_version=None
    ) -> dict:
        self.calls.append(("install_source", {"repo_url": repo_url, "branch": branch}))
        return {"message": f"{self.engine_id} source installation started", "task_id": "t-2"}

    async def remove(self, retire_references=False) -> dict:
        self.calls.append(("remove", {"retire_references": retire_references}))
        return {"message": f"{self.engine_id} removed", "task_id": "t-3"}

    def cancel_task(self, task_id: str) -> dict:
        self.calls.append(("cancel", {"task_id": task_id}))
        return {"ok": True, "task_id": task_id}


# ---------------------------------------------------------------------------
# Temp fixture: a native definition loaded through the production loader
# ---------------------------------------------------------------------------


def _document(**overrides) -> dict:
    document = {
        "schema_version": 2,
        "id": ENGINE_ID,
        "label": "Native Craft",
        "repository_source": "Native Craft",
        # The host is DECLARED; the installer is resolved from it, never from
        # the engine id.
        "native_host": "release_assets",
        "install_kind": "prebuilt",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary_path"],
        "artifact_formats": ["gguf"],
        "package_kinds": ["binary"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "lifecycle": {
            "binary_name": "nativecraft",
            "default_repo_url": SOURCE_REPO,
            "default_branch": "main",
            "version_setting_key": "release_version",
            "update_source": {"kind": "github_releases", "repo": "example/nativecraft"},
        },
        "settings_defaults": dict(DECLARED_SETTINGS),
        "recipes": {
            "release": [
                {
                    "id": "prepare",
                    "uses": "fs.ensure_dir",
                    "with": {"path": "install"},
                }
            ]
        },
    }
    document.update(overrides)
    return document


def _load_definition(tmp_path, **overrides):
    directory = tmp_path / "definitions"
    directory.mkdir()
    (directory / f"{ENGINE_ID}.yaml").write_text(
        yaml.safe_dump(_document(**overrides), sort_keys=False), encoding="utf-8"
    )
    registry = load_engine_definitions(str(directory))
    return registry.definitions[ENGINE_ID]


@pytest.fixture
def stub_native(monkeypatch):
    """Patch the ONE host seam; record which engine ids asked for a host."""
    stub = StubNativeInstaller()
    resolved: list[str] = []

    def fake_get_native_installer(engine_id: str):
        resolved.append(engine_id)
        return stub

    monkeypatch.setattr(ev, "get_native_installer", fake_get_native_installer)
    return stub, resolved


@pytest.fixture
def install_defaults(monkeypatch):
    """Register per-test install-settings defaults without touching real state."""
    from backend import venv_install_settings as vis

    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))

    def register(definition) -> dict:
        defaults = ev.definition_install_defaults(definition)
        vis.register_engine_install_settings(definition.id, defaults)
        return defaults

    return register


# ---------------------------------------------------------------------------
# Builder: installer follows native_host; 405 gate follows the source recipe
# ---------------------------------------------------------------------------


def test_builder_binds_the_declared_native_host_and_caches_it(tmp_path, stub_native):
    stub, resolved = stub_native
    definition = _load_definition(tmp_path)

    lifecycle = ev.build_definition_lifecycle(definition)

    assert lifecycle.engine_id == ENGINE_ID
    assert lifecycle.url_prefix == f"/{ENGINE_ID}"  # generic route, no legacy prefix
    # Lazy: building the record does not instantiate a host.
    assert resolved == []
    assert lifecycle.get_installer() is stub
    # The record owns ONE host instance, so in-memory operation state (and
    # cancellation) is the same across requests.
    assert lifecycle.get_installer() is stub
    assert resolved == [ENGINE_ID]

    # No source recipe declared -> no install-source plan exists.
    assert lifecycle.supports_source_install is False
    verbs = ev.generic_verb_names(lifecycle)
    assert set(verbs) == {
        "check-updates",
        "status",
        "build-settings",
        "install",
        "remove",
        "cancel",
    }


def test_builder_is_loud_when_the_declared_host_is_unresolvable(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "get_native_installer", lambda engine_id: None)
    definition = _load_definition(tmp_path)

    lifecycle = ev.build_definition_lifecycle(definition)

    with pytest.raises(RuntimeError, match="native_host"):
        lifecycle.get_installer()


# ---------------------------------------------------------------------------
# Generic verbs reach the native host
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generic_verbs_dispatch_to_the_native_installer(
    tmp_path, monkeypatch, stub_native, install_defaults
):
    stub, _resolved = stub_native
    definition = _load_definition(tmp_path)
    install_defaults(definition)
    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(ev, "get_store", lambda: store)
    lifecycle = ev.build_definition_lifecycle(definition)

    # install (release): the declared version travels to the host.
    assert await ev.lifecycle_install(lifecycle, {"version": "1.2.3"}) == {
        "message": f"{ENGINE_ID} installation started",
        "task_id": "t-1",
    }
    assert stub.calls[-1] == (
        "install",
        {"version": "1.2.3", "force_reinstall": False},
    )
    # No version -> falls back to the declared install-settings version key.
    await ev.lifecycle_install(lifecycle, {})
    assert stub.calls[-1][1] == {"version": None, "force_reinstall": False}

    status = await ev.lifecycle_status(lifecycle)
    assert status["engine"] == ENGINE_ID
    assert status["installed"] is False
    assert stub.calls[-1] == ("status", {})

    assert await ev.lifecycle_cancel(lifecycle, {"task_id": "t-7"}) == {
        "ok": True,
        "task_id": "t-7",
    }
    assert stub.calls[-1] == ("cancel", {"task_id": "t-7"})
    with pytest.raises(HTTPException) as missing_task:
        await ev.lifecycle_cancel(lifecycle, {})
    assert missing_task.value.status_code == 400
    assert missing_task.value.detail == "task_id is required"

    assert await ev.lifecycle_remove(lifecycle, {"retire_launch_references": True}) == {
        "message": f"{ENGINE_ID} removed",
        "task_id": "t-3",
    }
    assert stub.calls[-1] == ("remove", {"retire_references": True})

    # build-settings verbs use the declared settings surface (persisted
    # through the data store, like every other generic engine).
    assert await ev.lifecycle_get_build_settings(lifecycle) == DECLARED_SETTINGS
    saved = await ev.lifecycle_save_build_settings(lifecycle, {"release_version": "2.0"})
    assert saved == {**DECLARED_SETTINGS, "release_version": "2.0"}
    assert store.get_engine_build_settings(ENGINE_ID)["release_version"] == "2.0"


@pytest.mark.asyncio
async def test_install_source_is_405_without_a_source_recipe(
    tmp_path, stub_native, install_defaults
):
    stub, _resolved = stub_native
    definition = _load_definition(tmp_path)
    install_defaults(definition)
    lifecycle = ev.build_definition_lifecycle(definition)

    with pytest.raises(HTTPException) as unavailable:
        await ev.lifecycle_install_source(lifecycle, {})
    assert unavailable.value.status_code == 405
    assert unavailable.value.detail == (f"install-source is not available for engine '{ENGINE_ID}'")
    assert stub.calls == []  # never accepted, never partially executed


@pytest.mark.asyncio
async def test_install_source_is_served_when_the_source_recipe_is_declared(
    tmp_path, stub_native, install_defaults
):
    stub, _resolved = stub_native
    definition = _load_definition(
        tmp_path,
        recipes={
            "release": [{"id": "prepare", "uses": "fs.ensure_dir", "with": {"path": "x"}}],
            "source": [{"id": "checkout", "uses": "fs.ensure_dir", "with": {"path": "y"}}],
        },
    )
    install_defaults(definition)
    lifecycle = ev.build_definition_lifecycle(definition)

    assert lifecycle.supports_source_install is True
    assert "install-source" in ev.generic_verb_names(lifecycle)

    result = await ev.lifecycle_install_source(lifecycle, {})
    assert result == {
        "message": f"{ENGINE_ID} source installation started",
        "task_id": "t-2",
    }
    # The declared source repo/branch defaults are the resolved dispatch inputs.
    assert stub.calls[-1] == (
        "install_source",
        {"repo_url": SOURCE_REPO, "branch": "main"},
    )


# ---------------------------------------------------------------------------
# Startup registration: same gate and same one-record rule as every engine
# ---------------------------------------------------------------------------


@pytest.fixture
def startup_state_guard():
    """Restore the process-global registry state a startup gate mutates."""
    import backend.main  # noqa: F401 — production import-time registrations

    from backend.engines import registry as registry_module

    registry_snapshot = dict(registry_module.ENGINE_REGISTRY)
    id_set_snapshots = {
        name: set(getattr(registry_module, name))
        for name in (
            "VALID_ENGINE_IDS",
            "EMBEDDINGS_ENGINE_IDS",
            "NATIVE_ENGINE_IDS",
            "GGUF_ENGINE_IDS",
            "HF_SNAPSHOT_ENGINE_IDS",
        )
    }
    yield registry_module
    for name, snapshot in id_set_snapshots.items():
        target = getattr(registry_module, name)
        target.clear()
        target.update(snapshot)
    registry_module.ENGINE_REGISTRY.clear()
    registry_module.ENGINE_REGISTRY.update(registry_snapshot)


def test_startup_registers_the_native_record_once(
    tmp_path, monkeypatch, startup_state_guard, stub_native, install_defaults
):
    from backend import venv_install_settings as vis
    from backend.engines.definitions import loader as loader_module
    from backend.engines.definitions.loader import (
        EngineDefinitionRegistry,
        load_default_engine_definitions,
    )

    _stub, resolved = stub_native
    definition = _load_definition(tmp_path)
    shipped = load_default_engine_definitions()
    combined = EngineDefinitionRegistry(definitions={**shipped.definitions, ENGINE_ID: definition})
    monkeypatch.setattr(loader_module, "_default_registry", combined)
    monkeypatch.setattr(ev, "LIFECYCLE", dict(ev.LIFECYCLE))

    # Gate 1: the same startup verification every shipped definition passes.
    assert verify_engine_definitions() == len(shipped.ids()) + 1
    assert definition_tier(ENGINE_ID) == "authoritative"
    assert startup_state_guard.ENGINE_REGISTRY[ENGINE_ID].id == ENGINE_ID

    # Gate 2: the generic lifecycle record, registered from the YAML.
    registered = ev.register_definition_lifecycles()
    assert registered.count(ENGINE_ID) == 1
    record = ev.LIFECYCLE[ENGINE_ID]
    assert record.engine_id == ENGINE_ID
    assert record.get_installer().engine_id == ENGINE_ID
    assert resolved == [ENGINE_ID]
    assert vis.default_install_settings(ENGINE_ID) == DECLARED_SETTINGS

    # One-record rule: an existing registration is never replaced.
    sentinel = object()
    ev.LIFECYCLE[ENGINE_ID] = sentinel
    assert ev.register_definition_lifecycles() == []
    assert ev.LIFECYCLE[ENGINE_ID] is sentinel


def test_startup_registers_the_native_install_defaults(
    tmp_path, monkeypatch, startup_state_guard, stub_native, install_defaults
):
    from backend import venv_install_settings as vis
    from backend.engines.definitions import loader as loader_module
    from backend.engines.definitions.loader import (
        EngineDefinitionRegistry,
        load_default_engine_definitions,
    )

    definition = _load_definition(tmp_path)
    shipped = load_default_engine_definitions()
    combined = EngineDefinitionRegistry(definitions={**shipped.definitions, ENGINE_ID: definition})
    monkeypatch.setattr(loader_module, "_default_registry", combined)
    monkeypatch.setattr(ev, "LIFECYCLE", dict(ev.LIFECYCLE))

    ev.register_definition_lifecycles()

    assert vis.default_install_settings(ENGINE_ID) == DECLARED_SETTINGS


# ---------------------------------------------------------------------------
# Generic HTTP surface: one route space, no legacy prefix
# ---------------------------------------------------------------------------


def test_generic_http_surface_serves_the_native_engine(
    client, tmp_path, monkeypatch, stub_native, install_defaults
):
    stub, _resolved = stub_native
    definition = _load_definition(tmp_path)
    monkeypatch.setattr(ev, "get_store", lambda: DataStore(config_dir=str(tmp_path / "c")))
    install_defaults(definition)
    # Mutate the SHARED record map (the generic router resolves it per request),
    # and let monkeypatch restore it afterwards.
    monkeypatch.setitem(ev.LIFECYCLE, ENGINE_ID, ev.build_definition_lifecycle(definition))

    status = client.get(f"/api/engines/{ENGINE_ID}/status")
    assert status.status_code == 200, status.text
    assert status.json()["engine"] == ENGINE_ID
    assert status.json()["installed"] is False

    install = client.post(f"/api/engines/{ENGINE_ID}/install", json={"version": "4.2"})
    assert install.status_code == 200, install.text
    assert install.json() == {
        "message": f"{ENGINE_ID} installation started",
        "task_id": "t-1",
    }
    assert stub.calls[-1] == ("install", {"version": "4.2", "force_reinstall": False})

    cancel = client.post(f"/api/engines/{ENGINE_ID}/cancel", json={"task_id": "h-1"})
    assert cancel.status_code == 200, cancel.text
    assert cancel.json() == {"ok": True, "task_id": "h-1"}

    settings = client.get(f"/api/engines/{ENGINE_ID}/build-settings")
    assert settings.status_code == 200, settings.text
    assert settings.json() == DECLARED_SETTINGS

    # No source recipe -> the verb is refused as unavailable, not failed later.
    source = client.post(f"/api/engines/{ENGINE_ID}/install-source", json={})
    assert source.status_code == 405
    assert source.json() == {"detail": f"install-source is not available for engine '{ENGINE_ID}'"}

    # No legacy prefix was created for this engine.
    paths = client.get("/openapi.json").json()["paths"]
    assert not [path for path in paths if ENGINE_ID in path]
    assert f"/api/engines/{{engine_id}}/install" in paths
