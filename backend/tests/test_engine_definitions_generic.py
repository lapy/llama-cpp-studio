"""YAML campaign S3 — data-driven lifecycle + generic venv installer.

S3 adds code paths that ONLY YAML-onboarded engines traverse; the nine
generated engines are untouched (their existing suites passing unchanged is
the preservation evidence). Pins here cover:

* the optional ``lifecycle`` definition section (schema v2, strict keys);
* ``build_definition_lifecycle`` / ``register_definition_lifecycles``
  (generated engines skipped, Python registration wins collisions);
* ``GenericVenvInstaller`` end-to-end OFFLINE — venv/pip/metadata stubbed,
  the real executor + shared primitives + generic publish handlers run
  against a definition-driven ``release`` recipe;
* the persona boundary: a recipe reaching outside the venv+pip surface
  fails loudly with ``unimplemented primitive``.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.data_store import DataStore
from backend.engines.definitions.schema import (
    DefinitionError,
    validate_definition,
)


def _lifecycle(**overrides):
    doc = {
        "update_source": {"kind": "pypi", "package": "yamlcraft"},
        "binary_name": "yamlcraft",
        "distribution_names": ["yamlcraft"],
        "default_repo_url": "https://github.com/example/yamlcraft.git",
        "status_fallback": {"installed_at": None, "removed_at": None, "log_path": None},
    }
    doc.update(overrides)
    return doc


def _release_recipe():
    return {
        "release": [
            {"id": "resolve", "uses": "prereqs.resolve_version", "produces": ["version"]},
            {"id": "venv", "uses": "venv.create", "produces": ["venv"]},
            {
                "id": "pip",
                "uses": "pip.install_package",
                "requires": ["version"],
                "with": {"package": "yamlcraft"},
            },
            {"id": "meta", "uses": "pip.capture_metadata", "produces": ["metadata"]},
            {"id": "publish", "uses": "publish.install", "requires": ["metadata"]},
        ]
    }


def _craft(**overrides):
    doc = {
        "schema_version": 2,
        "id": "yamlcraft",
        "label": "Crafty",
        "repository_source": "yamlcraft",
        "install_kind": "python_venv",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["binary"],
        "artifact_formats": ["gguf"],
        "package_kinds": ["binary"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "recipes": _release_recipe(),
        "lifecycle": _lifecycle(),
    }
    doc.update(overrides)
    if doc.get("lifecycle") is None:
        doc.pop("lifecycle", None)
    return doc


# --- schema -----------------------------------------------------------------------


def test_lifecycle_section_validates_and_freezes_order():
    definition = validate_definition(_craft(), source="yamlcraft.yaml")
    assert dict(definition.lifecycle)["binary_name"] == "yamlcraft"
    assert tuple(k for k, _ in definition.lifecycle) == tuple(sorted(dict(_lifecycle())))


def test_lifecycle_rejects_unknown_keys_and_bad_sources():
    with pytest.raises(DefinitionError, match="lifecycle has unknown keys"):
        validate_definition(_craft(lifecycle=_lifecycle(magic=True)), source="y.yaml")
    with pytest.raises(DefinitionError, match="kind pypi requires package"):
        validate_definition(
            _craft(lifecycle=_lifecycle(update_source={"kind": "pypi"})), source="y.yaml"
        )
    with pytest.raises(DefinitionError, match="unsupported kind|update_source.kind"):
        validate_definition(
            _craft(lifecycle=_lifecycle(update_source={"kind": "carrier_pigeon"})),
            source="y.yaml",
        )


def test_lifecycle_is_v2_only_and_digest_sensitive():
    with pytest.raises(DefinitionError, match="require schema_version 2"):
        doc = _craft()
        doc["schema_version"] = 1
        doc.pop("recipes", None)
        validate_definition(doc, source="y.yaml")
    base = validate_definition(_craft(), source="y.yaml")
    shifted = validate_definition(
        _craft(lifecycle=_lifecycle(binary_name="crafty2")), source="y.yaml"
    )
    assert base.definition_digest != shifted.definition_digest
    assert base.to_api_dict() == shifted.to_api_dict()  # identity untouched


# --- builder / registration ---------------------------------------------------------


def test_builder_maps_lifecycle_onto_engine_lifecycle():
    from backend.routes.engine_versions import build_definition_lifecycle

    definition = validate_definition(_craft(), source="y.yaml")
    lc = build_definition_lifecycle(definition)
    assert lc.engine_id == "yamlcraft"
    assert lc.update_source.kind == "pypi"
    assert lc.update_source.package == "yamlcraft"
    assert lc.update_source.url == "https://pypi.org/project/yamlcraft/"
    assert lc.version_setting_key == "pip_version"
    assert lc.status_fallback == {"installed_at": None, "removed_at": None, "log_path": None}
    assert lc.prefer_source_install is False

    commit_doc = _craft(
        lifecycle=_lifecycle(
            update_source={"kind": "github_commit", "repo": "example/yamlcraft", "branch": "dev"}
        )
    )
    lc2 = build_definition_lifecycle(validate_definition(commit_doc, source="y.yaml"))
    assert lc2.update_source.kind == "github_commit"
    assert lc2.update_source.branch == "dev"


def _fake_loader(monkeypatch, combined):
    import backend.engines.definitions.loader as loader_mod

    monkeypatch.setattr(loader_mod, "load_default_engine_definitions", lambda: combined)


def _combined_registry(**extra):
    from backend.engines.definitions.loader import EngineDefinitionRegistry

    # Only the explicit authoritative fixtures: the exporter tier is empty.
    return EngineDefinitionRegistry(definitions=dict(extra))


def test_register_skips_generated_and_python_owned_but_adds_authoritative(monkeypatch, tmp_path):
    from backend.routes import engine_versions as ev
    import backend.venv_install_settings as vis

    craft = validate_definition(_craft(), source="y.yaml")
    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))
    _fake_loader(monkeypatch, _combined_registry(yamlcraft=craft))
    monkeypatch.setattr(ev, "LIFECYCLE", {})

    registered = ev.register_definition_lifecycles()
    assert registered == ["yamlcraft"]  # the nine generated ones were skipped
    lc = ev.LIFECYCLE["yamlcraft"]
    assert lc.engine_id == "yamlcraft"
    # install-settings defaults arrived from the lifecycle section, not code
    assert vis.default_install_settings("yamlcraft") == {
        "source_repo": "https://github.com/example/yamlcraft.git",
        "source_branch": "main",
        "pip_version": "",
    }
    assert vis.coerce_install_settings("yamlcraft", {})["source_repo"].endswith(".git")
    # generated engine defaults untouched
    assert vis.default_install_settings("lmdeploy")["source_repo"] == (
        "https://github.com/InternLM/lmdeploy.git"
    )

    # A Python registration wins (id already in LIFECYCLE): re-registering
    # keeps the existing record and adds nothing.
    sentinel = object()
    ev.LIFECYCLE["yamlcraft"] = sentinel
    assert ev.register_definition_lifecycles() == []
    assert ev.LIFECYCLE["yamlcraft"] is sentinel


def test_definition_without_lifecycle_is_not_registered(monkeypatch):
    from backend.routes import engine_versions as ev
    import backend.venv_install_settings as vis

    craft = validate_definition(_craft(lifecycle=None), source="y.yaml")
    monkeypatch.setattr(vis, "_ENGINE_DEFAULTS", dict(vis._ENGINE_DEFAULTS))
    vis._ENGINE_DEFAULTS.pop("yamlcraft", None)  # the shipped fixture may have
    # already registered defaults earlier in the process (lifespan ordering)
    _fake_loader(monkeypatch, _combined_registry(yamlcraft=craft))
    monkeypatch.setattr(ev, "LIFECYCLE", {})
    assert ev.register_definition_lifecycles() == []
    with pytest.raises(ValueError, match="No install settings defaults"):
        vis.default_install_settings("yamlcraft")


# --- GenericVenvInstaller (offline persona run) -------------------------------------


def _installer_env(tmp_path, monkeypatch, definition):
    from backend.engines.python import generic as gen
    from backend.engines.python import installer as venv_installer

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(venv_installer, "studio_data_dir", lambda: str(tmp_path / "data"))
    monkeypatch.setattr(venv_installer, "get_store", lambda: store)
    monkeypatch.setattr(gen, "get_store", lambda: store)
    activations = []

    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        activations.append((engine_id, version_name, row))

    monkeypatch.setattr(gen, "activate_engine_version", fake_activate)
    installer = gen.GenericVenvInstaller(definition, base_dir=str(tmp_path / "root"))
    return store, installer, activations


async def _settle(installer, timeout: float = 5.0):
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while installer._operation is not None:
        if loop.time() > deadline:
            raise AssertionError("operation did not settle")
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.02)  # let task post-processing land


@pytest.mark.asyncio
async def test_generic_release_install_runs_definition_recipe_offline(tmp_path, monkeypatch):
    definition = validate_definition(_craft(), source="y.yaml")
    store, installer, activations = _installer_env(tmp_path, monkeypatch, definition)

    pip_calls = []

    def fake_ensure_venv():
        import os

        os.makedirs(installer._venv_path, exist_ok=True)

    async def fake_run_pip(args, stage, **kwargs):
        pip_calls.append((args, stage))
        return 0

    installer._ensure_venv = fake_ensure_venv  # type: ignore[method-assign]
    installer._run_pip = fake_run_pip  # type: ignore[method-assign]
    installer._detect_installed_version = lambda venv_path=None: "1.2.3"  # type: ignore[assignment]

    response = await installer.install_release("1.2.3")
    assert response["message"] == "Crafty installation started"
    assert "task_id" in response
    await _settle(installer)

    assert installer._last_error is None, installer._last_error
    assert pip_calls == [(["install", "--upgrade", "yamlcraft==1.2.3"], "install")]
    assert activations and activations[0][0] == "yamlcraft"
    assert activations[0][1] == "1.2.3"
    rows = store.get_engine_versions("yamlcraft")
    assert any(row.get("version") == "1.2.3" and row.get("build_status") == "ready" for row in rows)
    # The installer records the AUTHORITATIVE, version-associated snapshot at
    # publish time: the resolved package version is frozen, not the request.
    snaps = store.list_install_snapshots("yamlcraft")
    published = [
        row for row in snaps if row.get("phase") == "published" and row.get("version") == "1.2.3"
    ]
    assert published, snaps
    assert published[-1]["definition_digest"] == definition.definition_digest
    assert published[-1]["resolved_inputs"]["package_version"] == "1.2.3"
    assert published[-1]["resolved_inputs"]["install_type"] == "pip"
    assert "task_id" in response


@pytest.mark.asyncio
async def test_failed_release_records_a_pending_snapshot_for_retry(tmp_path, monkeypatch):
    """A failed install must still be associated with the attempted definition.

    The pending row is named before execution (the timestamp-pip convention);
    the snapshot has to use that name, otherwise retry cannot find it and a
    changed definition is silently accepted.
    """
    from backend.engines.install_snapshots import retry_resolved_inputs

    definition = validate_definition(_craft(), source="y.yaml")
    store, installer, _activations = _installer_env(tmp_path, monkeypatch, definition)

    def fake_ensure_venv():
        import os

        os.makedirs(installer._venv_path, exist_ok=True)

    async def failing_pip(args, stage, **kwargs):
        return 1

    installer._ensure_venv = fake_ensure_venv  # type: ignore[method-assign]
    installer._run_pip = failing_pip  # type: ignore[method-assign]
    installer._detect_installed_version = lambda venv_path=None: None  # type: ignore[assignment]

    await installer.install_release("1.2.3")
    await _settle(installer)

    rows = store.get_engine_versions("yamlcraft")
    assert rows, "the failed install must leave its pending row"
    failed_version = rows[0]["version"]
    assert rows[0].get("build_status") in {"failed", "cancelled"}

    pending = [
        row
        for row in store.list_install_snapshots("yamlcraft")
        if row.get("phase") == "scheduled" and row.get("version") == failed_version
    ]
    assert pending, store.list_install_snapshots("yamlcraft")
    assert pending[-1]["definition_digest"] == definition.definition_digest
    assert pending[-1]["resolved_inputs"]["requested_version"] == "1.2.3"

    # And the retry guard now sees it: a changed definition is refused.
    with pytest.raises(ValueError, match="definition changed"):
        retry_resolved_inputs(
            store,
            "yamlcraft",
            failed_version,
            current_digest="sha256:" + "0" * 64,
            label="Crafty",
        )


@pytest.mark.asyncio
async def test_retry_of_pending_snapshot_does_not_use_the_row_name_as_a_version(
    tmp_path, monkeypatch
):
    """Installation identity is not a package version.

    Pending snapshots associate by the pre-publication row name; that name must
    never be passed to pip as the requested package version. A latest install
    with no resolved version retries as latest instead of failing on a
    timestamp-pip string.
    """
    definition = validate_definition(_craft(), source="y.yaml")
    store, installer, _activations = _installer_env(tmp_path, monkeypatch, definition)
    row_name = "20260101-000000-pip"
    store.record_install_snapshot(
        "yamlcraft",
        definition_digest=definition.definition_digest,
        resolved_inputs={
            "install_type": "pip",
            "install_dir": str(tmp_path / "root" / row_name),
            "venv_path": str(tmp_path / "root" / row_name / "venv"),
        },
        version=row_name,
        phase="scheduled",
    )
    captured = {}

    async def fake_install_release(
        version=None, force_reinstall=False, *, reuse_dir=None, existing_version=None
    ):
        captured.update({"version": version, "reuse_dir": reuse_dir})
        return {"message": "retry started"}

    installer.install_release = fake_install_release  # type: ignore[method-assign]
    await installer.retry_existing_install(
        {
            "version": row_name,
            "install_type": "pip",
            "install_dir": str(tmp_path / "root" / row_name),
            "venv_path": str(tmp_path / "root" / row_name / "venv"),
        }
    )
    assert captured["version"] is None
    assert captured["reuse_dir"] == str(tmp_path / "root" / row_name)


@pytest.mark.asyncio
async def test_retry_selects_the_snapshot_for_its_own_version(tmp_path, monkeypatch):
    definition = validate_definition(_craft(), source="y.yaml")
    store, installer, _activations = _installer_env(tmp_path, monkeypatch, definition)
    install_dir = str(tmp_path / "root" / "pip")

    def published(version: str, package_version: str, digest: str = ""):
        store.record_install_snapshot(
            "yamlcraft",
            definition_digest=digest or definition.definition_digest,
            resolved_inputs={
                "install_type": "pip",
                "version": version,
                "package_version": package_version,
                "install_dir": install_dir,
            },
            version=version,
            phase="published",
        )

    def row_for(version: str, package_version: str) -> dict:
        return {
            "version": version,
            "install_type": "pip",
            "install_dir": install_dir,
            "venv_path": str(tmp_path / "root" / "pip" / "venv"),
            "package_version": package_version,
        }

    published("1.2.3", "1.2.3")
    published("9.9.9", "9.9.9")  # installed AFTER 1.2.3

    captured = {}

    async def fake_install_release(
        version=None, force_reinstall=False, *, reuse_dir=None, existing_version=None
    ):
        captured.update(
            {
                "version": version,
                "force_reinstall": force_reinstall,
                "existing_version": existing_version,
            }
        )
        return {}

    installer.install_release = fake_install_release  # type: ignore[method-assign]

    # Retrying the OLDER installation must use ITS snapshot, not the latest
    # installation's resolution (the cross-installation bug).
    await installer.retry_existing_install(row_for("1.2.3", "1.2.3"))
    assert captured["version"] == "1.2.3"
    await installer.retry_existing_install(row_for("9.9.9", "9.9.9"))
    assert captured["version"] == "9.9.9"

    # A changed definition ON THE TARGET VERSION refuses; the other version's
    # digest is irrelevant.
    published("1.2.3", "1.2.3", digest="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="definition changed"):
        await installer.retry_existing_install(row_for("1.2.3", "1.2.3"))
    await installer.retry_existing_install(row_for("9.9.9", "9.9.9"))
    assert captured["version"] == "9.9.9"


@pytest.mark.asyncio
async def test_retry_matches_a_scheduled_snapshot_by_requested_version(tmp_path, monkeypatch):
    """Pre-publication rows (recorded at dispatch) still guard a retry when
    their recorded requested version matches the installation."""
    definition = validate_definition(_craft(), source="y.yaml")
    store, installer, _activations = _installer_env(tmp_path, monkeypatch, definition)
    store.record_install_snapshot(
        "yamlcraft",
        definition_digest=definition.definition_digest,
        resolved_inputs={"install_type": "pip", "version": "1.2.3"},
    )
    row = {
        "version": "1.2.3",
        "install_type": "pip",
        "install_dir": str(tmp_path / "root" / "pip"),
        "package_version": "1.2.3",
    }
    with pytest.raises(ValueError, match="definition changed"):
        # A different current definition makes the recorded digest stale.
        await _installer_env(
            tmp_path,
            monkeypatch,
            validate_definition({**_craft(), "label": "Crafty Relabeled"}, source="y.yaml"),
        )[1].retry_existing_install(dict(row))


@pytest.mark.asyncio
async def test_recipe_outside_persona_fails_loudly(tmp_path, monkeypatch):
    doc = _craft()
    doc["recipes"]["release"][1] = {
        "id": "cmake",
        "uses": "cmake.configure",
        "with": {"source_dir": "x"},
    }
    definition = validate_definition(doc, source="y.yaml")
    _, installer, activations = _installer_env(tmp_path, monkeypatch, definition)
    installer._ensure_venv = lambda: None  # type: ignore[method-assign]

    await installer.install_release("0.1")
    await _settle(installer)

    assert "unimplemented primitive" in (installer._last_error or "")
    assert activations == []


@pytest.mark.asyncio
async def test_compat_install_uses_the_declared_torch_profile():
    from backend.engines.python.generic import GenericVenvInstaller

    installer = object.__new__(GenericVenvInstaller)
    installer.label = "LMDeploy"
    calls = []

    async def fake_run_pip(args, operation, ensure_venv=True, **kwargs):
        calls.append((list(args), operation))
        return 0

    installer._run_pip = fake_run_pip  # type: ignore[method-assign]
    installer._install_settings = lambda: {  # type: ignore[method-assign]
        "compat_sm70": True,
        "compat_torch_spec": "torch==2.8.0+cu126 torchvision==0.23.0+cu126",
        "compat_torch_index_url": "https://download.pytorch.org/whl/cu126",
    }
    await installer._apply_compat_install()
    assert calls == [
        (
            [
                "install",
                "--upgrade",
                "torch==2.8.0+cu126",
                "torchvision==0.23.0+cu126",
                "--index-url",
                "https://download.pytorch.org/whl/cu126",
            ],
            "compatibility",
        )
    ]

    calls.clear()
    installer._install_settings = lambda: {"compat_sm70": False}  # type: ignore[method-assign]
    await installer._apply_compat_install()
    assert calls == []
