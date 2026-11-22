"""Acceptance: onboarding a new engine is a YAML definition, not new dispatch.

The project promise (docs/config-driven-engine-refactoring-plan.md, Objective):
*"Add an engine by adding one validated YAML definition, without adding an
installer class, runtime compiler branch, API router, frontend component, or
engine-ID allowlist."* This module is the acceptance test for that promise,
written so it cannot cheat: it starts from a REAL shipped definition, renames
it to a brand-new engine id (``fixture_echo``), writes it to a TEMP definitions
directory, and then only exercises production entry points.

What the fixture proves, in order:

1. ``load_engine_definitions(directory)`` — the loader's supported directory
   argument, used by existing schema tests — parses, template-expands and
   validates the new file (filename stem == id, strict schema);
2. the startup gates the application runs in ``backend/main.py``'s lifespan
   accept the new id: ``verify_engine_definitions()`` registers it as an
   AUTHORITATIVE definition (``register_engine_spec``) and
   ``register_definition_lifecycles()`` materializes its generic lifecycle
   record and install-settings defaults from the YAML;
3. the existing generic surfaces serve it through a TestClient with no new
   route: ``GET /api/engines`` (descriptor + ``generic_lifecycle`` /
   ``generic_verbs``), ``GET /api/engines/{id}/status`` and
   ``GET /api/engines/{id}/build-settings``. The status call only inspects the
   absence of an installed venv — it does not spawn a process;
4. the declared ``runtime_profiles.default`` compiles to argv/env via
   ``compile_profile_argv``/``compile_profile_env``, the declared ``release``
   recipe materializes through ``materialize_plan``, and the declared
   ``settings_defaults`` drive the generic settings verbs;
5. an architecture guard scans backend and frontend source for the fixture id:
   onboarding it required zero engine-specific Python or frontend file.

The one hook that is NOT a public directory override, and why
--------------------------------------------------------------
The loader accepts a directory (``load_engine_definitions``), and that is the
hook used for the real load. The application, however, resolves definitions
through the process-wide singleton ``load_default_engine_definitions()`` —
``verify_engine_definitions()`` and ``register_definition_lifecycles()`` both
call it, and there is no environment variable or setting that points the
process at another definitions directory (the shipped directory is fixed:
``backend/engines/definitions/``). To run the REAL startup gates against the
temp fixture without copying a fixture engine into the shipped tree, this test
patches exactly that one singleton (``loader._default_registry``) to the union
of the shipped registry and the temp-loaded one. It does not patch individual
module constants, route modules, or lifecycle registration; one seam, restored
by pytest, and the rest is production code.

Recommendation (non-blocking): if running the app against an alternate
definitions directory ever becomes a supported deployment mode (plugins,
per-user engines), expose it as an explicit setting and have
``load_default_engine_definitions()`` read it; then this monkeypatch disappears.
It is not required to onboard an engine into the shipped directory today — that
path is already data-only.

Honest static boundaries this test PINS instead of hiding
---------------------------------------------------------
* ``install_kind`` / ``runtime_kind`` / ``scanner_kind`` are closed vocabularies
  in ``backend/engines/definitions/schema.py``. A new engine that joins an
  existing mechanism family (here: ``python_venv`` install, the LMDeploy-style
  server runtime, the ``lmdeploy_help`` scanner) is YAML-only. A genuinely new
  MECHANISM (new inference protocol, native host, help parser) still needs one
  shared, kind-keyed implementation — see docs/engine-host-and-presentation-
  boundaries.md. The test asserts the fixture's kinds are existing members
  rather than pretending engine ids and mechanism kinds are the same thing.
* The frontend ``ENGINE_DESCRIPTORS`` table in ``frontend/src/engines/
  registry.js`` remains static for the shipped engines (icons, copy, specialty
  panels). It is NOT required for a new engine: the server descriptor feeds the
  server-backed overlay (``generic_lifecycle`` + ``generic_verbs`` + declared
  ``ui_kind``), which is covered by the frontend vitest suites
  (``presentationOverlay.test.js``, ``engines.server.test.js``). This Python
  test proves the payload inputs those helpers consume; it does not re-run JS.
* Generated assets only change when a definition lands in the SHIPPED directory:
  ``docs/engine-parity-inventory.{json,md}`` captures the registry, and
  ``frontend/src/api/openapi.json`` captures routes (unchanged here — the
  fixture adds no route). Regeneration commands are documented in
  docs/engine-onboarding.md. Because this fixture lives in ``tmp_path``, no
  generated asset or shipped inventory changes.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import backend.venv_install_settings as install_settings
from backend.data_store import DataStore
from backend.engines import registry as registry_module
from backend.engines.definitions import DEFINITIONS_DIR, load_engine_definitions
from backend.engines.definitions import loader as loader_module
from backend.engines.definitions.loader import (
    EngineDefinitionRegistry,
    load_default_engine_definitions,
)
from backend.engines.definitions.plan import materialize_plan
from backend.engines.definitions.profiles import (
    compile_profile_argv,
    compile_profile_env,
)
from backend.engines.definitions.schema import (
    INSTALL_KINDS,
    RUNTIME_KINDS,
    SCANNER_KINDS,
    DefinitionError,
    EngineDefinition,
    load_yaml_file_strict,
)
from backend.engines.definitions.startup_check import (
    definition_tier,
    verify_engine_definitions,
)
from backend.engines.python import generic as generic_module
from backend.engines.python import installer as venv_installer_module
from backend.engines.registry import (
    ENGINE_REGISTRY,
    engine_registry_payload,
    get_engine_spec,
)
from backend.routes import engine_versions as engine_versions_route
from backend.routes import engines as engines_route

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ID = "fixture_echo"
# A real shipped definition supplies the STRUCTURE (template inheritance,
# lifecycle, discovery, runtime profile, recipes, presentation kind); only the
# identity and the engine-specific strings below are replaced.
SOURCE_DEFINITION = os.path.join(DEFINITIONS_DIR, "lmdeploy.yaml")
TEMPLATES_FILENAME = "_templates.yaml"
FIXTURE_REPO_URL = "https://github.com/example/fixture-echo.git"
EXPECTED_SETTINGS = {
    "pip_version": "",
    "source_branch": "main",
    "source_repo": FIXTURE_REPO_URL,
}
GENERIC_VERBS = {
    "build-settings",
    "cancel",
    "check-updates",
    "install",
    "install-source",
    "remove",
    "status",
}


# ---------------------------------------------------------------------------
# The fixture engine: a real definition's structure under a new id
# ---------------------------------------------------------------------------


def _fixture_document() -> dict:
    """Re-identify the real LMDeploy definition as the new fixture engine."""
    document = load_yaml_file_strict(SOURCE_DEFINITION)
    assert isinstance(document, dict)
    document["id"] = FIXTURE_ID
    document["label"] = "Fixture Echo"
    document["repository_source"] = "Fixture Echo"
    document["lifecycle"].update(
        {
            "binary_name": "fixture-echo",
            "distribution_names": ["fixture-echo"],
            "root_name": FIXTURE_ID,
            "log_name": "fixture_echo_install.log",
            "state_name": "fixture_echo_manager.json",
            "default_repo_url": FIXTURE_REPO_URL,
            "binary_override_env": "FIXTURE_ECHO_BIN",
            "update_source": {
                "kind": "pypi",
                "package": "fixture-echo",
                "timeout": 10.0,
                "error_status": 500,
            },
        }
    )
    document["runtime_profiles"]["default"]["executable"]["name"] = "fixture-echo"
    document["settings_defaults"] = dict(EXPECTED_SETTINGS)
    document["upstream"] = {"repository_url": FIXTURE_REPO_URL}
    document["data_dirs"] = [FIXTURE_ID]
    document["launch_policy"] = {
        "cpu_supported": True,
        "gpu_archs": ["sm_90"],
        "runtime_context_key": "max_model_len",
    }
    # Never claim a shipped artifact format: a test engine must not leak into
    # model-compatibility inference for real models (a lesson recorded when the
    # previous shipped persona was retired).
    document.pop("model_compat", None)
    for step in document["recipes"]["release"]:
        if step.get("uses") == "pip.install_package":
            step["with"]["package"] = "fixture-echo"
    for step in document["recipes"]["remove"]:
        if step.get("uses") == "pip.uninstall":
            step["with"]["package"] = "fixture-echo"
    return document


def _write_fixture_definition(directory: Path, *, document: dict | None = None) -> Path:
    """Write templates + the new engine YAML into a temp definitions directory."""
    directory.mkdir(parents=True, exist_ok=True)
    templates_path = os.path.join(DEFINITIONS_DIR, TEMPLATES_FILENAME)
    (directory / TEMPLATES_FILENAME).write_text(
        Path(templates_path).read_text(encoding="utf-8"), encoding="utf-8"
    )
    payload = document if document is not None else _fixture_document()
    (directory / f"{FIXTURE_ID}.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False), encoding="utf-8"
    )
    return directory


# ---------------------------------------------------------------------------
# Process-global isolation (registry/lifecycle/defaults are SHARED objects)
# ---------------------------------------------------------------------------


@pytest.fixture
def production_state_guard():
    """Snapshot and restore every process-global onboarding mutates.

    ``ENGINE_REGISTRY`` and its derived id-sets are shared mutable objects by
    design (see backend/engines/registry.py), so a registration cannot be
    undone with monkeypatch alone. Importing ``backend.main`` first captures the
    same startup-time state production has.
    """
    import backend.main  # noqa: F401 — production import-time registrations

    registry_snapshot = dict(ENGINE_REGISTRY)
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
    lifecycle_snapshot = dict(engine_versions_route.LIFECYCLE)
    defaults_snapshot = dict(install_settings._ENGINE_DEFAULTS)
    yield
    for name, snapshot in id_set_snapshots.items():
        target = getattr(registry_module, name)
        target.clear()
        target.update(snapshot)
    ENGINE_REGISTRY.clear()
    ENGINE_REGISTRY.update(registry_snapshot)
    engine_versions_route.LIFECYCLE.clear()
    engine_versions_route.LIFECYCLE.update(lifecycle_snapshot)
    install_settings._ENGINE_DEFAULTS.clear()
    install_settings._ENGINE_DEFAULTS.update(defaults_snapshot)


@pytest.fixture
def onboarded(tmp_path, monkeypatch, production_state_guard):
    """Load the temp definition through the loader, then run the real startup gates."""
    shipped = load_default_engine_definitions()
    definitions_dir = _write_fixture_definition(tmp_path / "definitions")
    loaded = load_engine_definitions(str(definitions_dir))
    definition = loaded.get(FIXTURE_ID)
    assert isinstance(definition, EngineDefinition)

    combined = EngineDefinitionRegistry(definitions={**shipped.definitions, **loaded.definitions})
    # The single documented seam in this module's docstring: the startup gates
    # resolve definitions through the process-wide singleton, which has no
    # directory override. Patch that one object, not module constants.
    monkeypatch.setattr(loader_module, "_default_registry", combined)

    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(engine_versions_route, "get_store", lambda: store)
    monkeypatch.setattr(generic_module, "get_store", lambda: store)
    monkeypatch.setattr(engines_route, "get_store", lambda: store)
    data_root = tmp_path / "data"
    monkeypatch.setattr(venv_installer_module, "studio_data_dir", lambda: str(data_root))

    verified_count = verify_engine_definitions()
    registered = engine_versions_route.register_definition_lifecycles()
    return SimpleNamespace(
        definition=definition,
        shipped_ids=shipped.ids(),
        verified_count=verified_count,
        registered=registered,
        lifecycle=engine_versions_route.LIFECYCLE[FIXTURE_ID],
    )


# ---------------------------------------------------------------------------
# 1. Loader + schema: a new file is a new engine
# ---------------------------------------------------------------------------


def test_loader_accepts_a_new_definition_and_reports_declared_metadata(tmp_path):
    definitions_dir = _write_fixture_definition(tmp_path / "definitions")

    registry = load_engine_definitions(str(definitions_dir))

    assert registry.ids() == (FIXTURE_ID,)
    definition = registry.get(FIXTURE_ID)
    assert isinstance(definition, EngineDefinition)
    assert definition.source_path == str(definitions_dir / f"{FIXTURE_ID}.yaml")
    assert definition.schema_version == 2
    # Identity/capabilities are exactly the declared values.
    assert definition.label == "Fixture Echo"
    assert definition.repository_source == "Fixture Echo"
    assert definition.install_kind == "python_venv"
    assert definition.runtime_kind == "lmdeploy"
    assert definition.scanner_kind == "lmdeploy_help"
    assert definition.active_path_fields == ("venv_path",)
    # These values come from the real templates the source definition extends,
    # so template expansion is exercised from data as well.
    assert definition.artifact_formats == ("safetensors",)
    assert definition.package_kinds == ("hf_snapshot",)
    assert definition.tasks == (
        "embeddings",
        "reranking",
        "text-generation",
        "text2text-generation",
    )
    assert dict(definition.discovery)["parser"] == "lmdeploy_help"
    assert dict(definition.discovery)["probe_argv_prefix"] == ["serve", "api_server", "--help"]
    assert definition.ui_kind == "venv"
    # Nested sections survive validation as declared data.
    assert dict(definition.lifecycle)["update_source"]["package"] == "fixture-echo"
    assert dict(definition.lifecycle)["status_fallback"] == {
        "installed_at": None,
        "removed_at": None,
        "log_path": None,
    }
    assert dict(definition.settings_defaults) == EXPECTED_SETTINGS
    assert dict(definition.launch_policy)["gpu_archs"] == ("sm_90",)
    assert set(dict(definition.runtime_profiles)) == {"default"}
    assert dict(definition.runtime_profiles)["default"]["executable"] == {
        "kind": "venv_console_script",
        "name": "fixture-echo",
    }
    assert set(dict(definition.recipes)) == {
        "release",
        "remove",
        "remove_version",
        "source",
        "sync",
    }
    # The definition is fully validated and content-addressed (install identity).
    assert definition.definition_digest.startswith("sha256:")


def test_loader_still_rejects_an_invalid_new_definition(tmp_path):
    """The new file is VALIDATED, not merely accepted: schema gates stay live."""
    document = _fixture_document()
    document["invented_section"] = {"anything": True}
    bad_dir = _write_fixture_definition(tmp_path / "bad", document=document)

    with pytest.raises(DefinitionError, match="unknown definition fields"):
        load_engine_definitions(str(bad_dir))


# ---------------------------------------------------------------------------
# 2. Startup: the real gates register the new id as authoritative
# ---------------------------------------------------------------------------


def test_startup_registers_the_new_definition_without_touching_shipped_engines(onboarded):
    # Every shipped engine was re-verified (no drift/missing) and the fixture was
    # accepted — the count is exactly shipped + 1.
    assert onboarded.verified_count == len(onboarded.shipped_ids) + 1
    assert definition_tier(FIXTURE_ID) == "authoritative"

    spec = get_engine_spec(FIXTURE_ID)
    assert spec is not None
    # One identity system: the registry entry IS the validated definition.
    assert spec.to_api_dict() == onboarded.definition.to_api_dict()
    assert FIXTURE_ID in registry_module.VALID_ENGINE_IDS
    assert FIXTURE_ID not in registry_module.NATIVE_ENGINE_IDS  # python_venv
    assert FIXTURE_ID not in registry_module.GGUF_ENGINE_IDS  # safetensors

    payload = engine_registry_payload()
    entry = next(engine for engine in payload["engines"] if engine["id"] == FIXTURE_ID)
    assert entry == onboarded.definition.to_api_dict()
    assert entry["label"] == "Fixture Echo"
    assert entry["ui_kind"] == "venv"
    # Launch-policy facts are exposed generically for frontend filters.
    assert entry["cpu_supported"] is True
    assert entry["gpu_archs"] == ["sm_90"]

    # Startup added the new id exactly once. A YAML-only shipped engine with no
    # compatibility router (e.g. ReVolta) may also be registered when nothing
    # registered it earlier in this process, so assert membership/uniqueness
    # rather than an order-dependent exact list: no shipped engine needed code
    # work for the new definition either way.
    assert onboarded.registered.count(FIXTURE_ID) == 1
    assert set(onboarded.registered) <= set(onboarded.shipped_ids) | {FIXTURE_ID}
    assert onboarded.lifecycle.engine_id == FIXTURE_ID
    assert onboarded.lifecycle.update_source.package == "fixture-echo"
    # Install-settings defaults came from the declared settings_defaults block.
    assert install_settings.default_install_settings(FIXTURE_ID) == EXPECTED_SETTINGS


# ---------------------------------------------------------------------------
# 3. Generic HTTP surfaces serve it (TestClient, no process spawn)
# ---------------------------------------------------------------------------


def test_generic_http_surfaces_serve_the_new_engine(client, onboarded):
    listing = client.get("/api/engines")
    assert listing.status_code == 200
    engines = listing.json()["engines"]
    assert len(engines) == len(onboarded.shipped_ids) + 1
    entry = next(engine for engine in engines if engine["id"] == FIXTURE_ID)
    assert entry["label"] == "Fixture Echo"
    assert entry["generic_lifecycle"] is True
    assert set(entry["generic_verbs"]) == GENERIC_VERBS
    assert entry["active_version"] is None
    assert entry["installed_versions"] == 0
    assert entry["runnable"] is False

    # Status is the generic lifecycle machinery (GenericVenvInstaller), resolved
    # from the definition: nothing installed, no venv, no process spawned.
    status = client.get(f"/api/engines/{FIXTURE_ID}/status")
    assert status.status_code == 200
    body = status.json()
    installer = onboarded.lifecycle.get_installer()
    assert body["installed"] is False
    assert body["version"] is None
    assert body["venv_path"] == os.path.join(installer.install_root, "venv")
    assert body["log_path"] == installer.log_path
    assert body["binary_path"] is None

    settings = client.get(f"/api/engines/{FIXTURE_ID}/build-settings")
    assert settings.status_code == 200
    assert settings.json() == EXPECTED_SETTINGS

    # Onboarding added NO route: the fixture appears in no path, and the generic
    # routes it uses already exist in the checked-in OpenAPI document.
    paths = client.get("/openapi.json").json()["paths"]
    assert f"/api/engines/{{engine_id}}/status" in paths
    assert f"/api/engines/{{engine_id}}/build-settings" in paths
    assert not [path for path in paths if FIXTURE_ID in path]


# ---------------------------------------------------------------------------
# 4. Launch/build behaviour is resolved from declared data
# ---------------------------------------------------------------------------


def test_declared_profile_recipes_and_settings_resolve_from_data(onboarded):
    definition = onboarded.definition

    argv = compile_profile_argv(definition, model_ref="example/echo-model", port="8123")
    assert argv == ["serve", "api_server", "example/echo-model", "--server-port", "8123"]
    env = compile_profile_env(definition, {"PATH": "/usr/bin"})
    assert env == {
        "env": {"PATH": "/usr/bin"},
        "missing_required": [],
        "prepend_bin_to_path": True,
        "cwd": "none",
    }

    # The registry carries the same declared profile (no per-engine compiler).
    spec = get_engine_spec(FIXTURE_ID)
    assert spec.runtime_profile == dict(definition.profile())

    # The release install path is a declared plan over the shared primitive
    # vocabulary; it materializes without executing anything.
    plan = materialize_plan(definition, "release")
    assert plan.engine_id == FIXTURE_ID
    assert [step.primitive for step in plan.steps] == [
        "prereqs.resolve_version",
        "venv.create",
        "pip.install_package",
        "pip.capture_metadata",
        "publish.install",
    ]
    assert plan.required_inputs == ("input.engine_root", "input.force_reinstall")
    with pytest.raises(DefinitionError, match="no recipe"):
        materialize_plan(definition, "shell")

    # Settings verbs coerce through the registered declared defaults.
    coerced = install_settings.coerce_install_settings(
        FIXTURE_ID, {"pip_version": "1.2.3", "source_branch": "dev"}
    )
    assert coerced == {
        "pip_version": "1.2.3",
        "source_branch": "dev",
        "source_repo": FIXTURE_REPO_URL,
    }
    # The engine reuses declared MECHANISM kinds rather than declaring a new
    # engine-id branch; these vocabularies are the boundary where new
    # mechanisms (not new engines) still need code.
    assert definition.install_kind in INSTALL_KINDS
    assert definition.runtime_kind in RUNTIME_KINDS
    assert definition.scanner_kind in SCANNER_KINDS


# ---------------------------------------------------------------------------
# 5. Architecture guard: no engine-specific source file was needed
# ---------------------------------------------------------------------------


def test_onboarding_needed_no_engine_specific_python_or_frontend_file():
    """If this fails, someone hardcoded the fixture instead of letting YAML speak."""
    scan_roots = (
        (REPO_ROOT / "backend", {".py"}),
        (REPO_ROOT / "frontend" / "src", {".js", ".ts", ".vue", ".json"}),
    )
    skipped_parts = {"tests", "node_modules", "dist", "__pycache__"}
    offenders = []
    for base, suffixes in scan_roots:
        for path in base.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            if skipped_parts & set(path.parts):
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            if FIXTURE_ID in path.name.lower():
                offenders.append(relative)
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            if FIXTURE_ID in content:
                offenders.append(relative)
    assert not offenders, f"{FIXTURE_ID} hardcoded outside tests/YAML/docs: {offenders}"
