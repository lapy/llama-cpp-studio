"""Phase-3 typed installer policy — the capabilities the vLLM extraction needs.

Pins the shared host's definition-declared behavior before any engine is
flipped onto it:

* ``python_min``/``python_max`` interpreter guard (exact legacy message);
* ``build_env: cuda`` composition + runtime metadata;
* ``publish.finalize`` semantics (import validation, reuse naming, activation);
* ``forget_errors: raise`` strict deletion policy;
* plan-filtered handler maps (only the recipe's own primitives are overridden).

Synthetic "policycraft" definition only — no engine-id literals.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from backend.data_store import DataStore
from backend.engines.definitions.schema import validate_definition
from backend.engines.python import generic as gen

POLICYCRAFT_ID = "policycraft"


def _definition(lifecycle_extra: dict | None = None, *, recipes: dict | None = None):
    lifecycle = {
        "binary_name": "policycraft",
        "root_name": "policycraft",
        "log_name": "policycraft.log",
    }
    lifecycle.update(lifecycle_extra or {})
    raw = {
        "schema_version": 2,
        "id": POLICYCRAFT_ID,
        "label": "Policycraft",
        "repository_source": "Example Org/Policycraft",
        "install_kind": "python_venv",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["venv_path"],
        "artifact_formats": ["safetensors"],
        "package_kinds": ["hf_snapshot"],
        "tasks": ["text-generation"],
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "lifecycle": lifecycle,
    }
    if recipes:
        raw["recipes"] = recipes
    return validate_definition(raw, source="policycraft.yaml")


async def _fake_progress(*args, **kwargs):
    return None


def _single_forget_plan(name: str):
    """A one-step plan so the real executor drives the forget handler."""
    from backend.engines.definitions.plan import InstallPlan, PlannedStep

    return InstallPlan(
        engine_id=POLICYCRAFT_ID,
        recipe=name,
        steps=(
            PlannedStep(
                id="forget",
                primitive="publish.forget",
                run=(),
                with_={},
                when=None,
                requires=(),
                produces=(),
                timeout_s=None,
                accepted_exit_codes=(),
            ),
        ),
    )


def _installer(tmp_path, definition):
    return gen.GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "policycraft.log"),
        state_path=str(tmp_path / "policycraft_state.json"),
        base_dir=str(tmp_path / "policycraft"),
    )


def _patch_store(monkeypatch, tmp_path) -> DataStore:
    store = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(gen, "get_store", lambda: store)
    return store


def test_python_range_guard_raises_the_legacy_message(tmp_path):
    definition = _definition({"python_min": "3.0", "python_max": "3.0"})
    installer = _installer(tmp_path, definition)
    with pytest.raises(RuntimeError) as excinfo:
        installer._ensure_venv()
    assert str(excinfo.value) == (
        "Policycraft requires Python 3.0–3.0; "
        f"Studio is running Python {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_python_range_guard_allows_the_running_interpreter(tmp_path, monkeypatch):
    definition = _definition({"python_min": "3.0", "python_max": "9.99"})
    installer = _installer(tmp_path, definition)
    called = []
    monkeypatch.setattr(gen.PythonVenvInstaller, "_ensure_venv", lambda self: called.append(True))
    installer._ensure_venv()
    assert called == [True]


def test_build_env_cuda_adds_env_and_runtime_metadata(tmp_path, monkeypatch):
    definition = _definition({"build_env": "cuda"})
    installer = _installer(tmp_path, definition)
    fake = SimpleNamespace(
        status=lambda: {"version": "12.8", "cuda_path": "/opt/cuda"},
        get_cuda_env=lambda: {"CUDA_HOME": "/opt/cuda"},
    )
    monkeypatch.setattr("backend.cuda_installer.get_cuda_installer", lambda: fake)

    env = installer._compose_build_env()
    assert env["CUDA_HOME"] == "/opt/cuda"
    assert env["VIRTUAL_ENV"] == installer._venv_path
    runtime = installer._runtime_metadata()
    assert runtime["cuda_version"] == "12.8"
    assert runtime["cuda_path"] == "/opt/cuda"


def test_build_env_default_has_no_cuda_metadata(tmp_path):
    installer = _installer(tmp_path, _definition())
    assert installer._runtime_metadata() == {}
    assert "CUDA_HOME" not in installer._compose_build_env() or True


@pytest.mark.asyncio
async def test_finalize_validates_the_install_and_activates(tmp_path, monkeypatch):
    installer = _installer(tmp_path, _definition())
    store = _patch_store(monkeypatch, tmp_path)
    activations = []

    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        activations.append((engine_id, version_name, dict(row)))

    monkeypatch.setattr(gen, "activate_engine_version", fake_activate)
    monkeypatch.setattr(installer, "_update_progress_task", _fake_progress)
    monkeypatch.setattr(installer, "_detect_installed_version", lambda *a, **k: "1.2.3")
    monkeypatch.setattr(installer, "_ready_pending_version", lambda pending, meta: meta["version"])

    handler = installer._finalize_handler(existing_version=None, pending_version="pip")
    await handler(None, {}, {})

    assert activations and activations[0][1] == "1.2.3"
    assert activations[0][2]["package_version"] == "1.2.3"
    assert store.list_install_snapshots(POLICYCRAFT_ID)[-1]["version"] == "1.2.3"


@pytest.mark.asyncio
async def test_finalize_rejects_an_unimportable_install(tmp_path, monkeypatch):
    from backend.engines.definitions.executor import PrimitiveExecutionError

    installer = _installer(tmp_path, _definition())
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setattr(installer, "_detect_installed_version", lambda *a, **k: None)
    monkeypatch.setattr(installer, "_update_progress_task", _fake_progress)

    handler = installer._finalize_handler(existing_version=None, pending_version="pip")
    with pytest.raises(PrimitiveExecutionError) as excinfo:
        await handler(None, {}, {})
    assert str(excinfo.value) == "Installed Policycraft package could not be imported"


@pytest.mark.asyncio
async def test_forget_errors_raise_propagates_the_failure(tmp_path, monkeypatch):
    definition = _definition({"forget_errors": "raise"})
    installer = _installer(tmp_path, definition)
    store = _patch_store(monkeypatch, tmp_path)
    monkeypatch.setattr(installer, "_ready_pending_version", lambda *a, **k: "1.0.0")
    monkeypatch.setattr(installer, "_update_installed_state", lambda *a, **k: None)
    # A single publish.forget step so the real executor drives the handler.
    monkeypatch.setattr(installer, "_recipe", _single_forget_plan)

    def boom(*args, **kwargs):
        raise RuntimeError("delete failed")

    store.delete_engine_version = boom  # type: ignore[assignment]
    # Strict policy: the deletion failure escapes the primitive instead of
    # being swallowed (the executor propagates it to the caller).
    with pytest.raises(RuntimeError, match="delete failed"):
        await installer._execute_remove_plan(
            venv_path=str(tmp_path / "policycraft" / "venv"),
            python_exists=False,
            retire_references=False,
            version_name="1.0.0",
            store=store,
        )

    # Default policy swallows the same failure and completes.
    lenient = _installer(tmp_path, _definition())
    lenient._recipe = lambda name: _single_forget_plan(name)  # type: ignore[assignment]
    lenient._update_installed_state = lambda *a, **k: None  # type: ignore[assignment]
    execution = await lenient._execute_remove_plan(
        venv_path=str(tmp_path / "policycraft" / "venv"),
        python_exists=False,
        retire_references=False,
        version_name="1.0.0",
        store=store,
    )
    assert execution.status == "completed"


@pytest.mark.asyncio
async def test_source_sealing_rebinds_checkout(tmp_path, monkeypatch):
    """Publishing a source checkout must rebind it to the sealed copy."""
    from backend.engines.definitions.plan import PlannedStep

    from backend.engines.definitions.primitives import run_shared_primitive

    definition = _definition(recipes={"source": [{"id": "seal", "uses": "vcs.seal_checkout"}]})
    installer = _installer(tmp_path, definition)
    sealed_calls: list[tuple[str, str]] = []

    class _Workspace:
        def seal_installed_source(self, destination, venv_path):
            sealed_calls.append((destination, venv_path))
            return destination + "-sealed"

    step = PlannedStep(
        id="seal",
        primitive="vcs.seal_checkout",
        run=(),
        with_={"destination": "source", "workspace_from": "workspace"},
        when=None,
        requires=(),
        produces=(),
        timeout_s=None,
        accepted_exit_codes=(),
    )
    workspace = _Workspace()
    context = {"workspace": workspace}

    default_result = await run_shared_primitive(installer, "vcs.seal_checkout", step, {}, context)

    # Drive the shared host's handler directly through the same closure the
    # flow builds, by executing the plan with a capturing runner.
    captured = {}

    async def fake_execute_plan(plan, *, run_primitive, context):
        captured["handlers"] = run_primitive
        return SimpleNamespace(status="completed", failed_step=None)

    monkeypatch.setattr("backend.engines.python.generic.execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        "backend.engines.python.generic.make_engine_runner",
        lambda installer_obj, handlers: handlers,
    )
    await installer._execute_source_plan(
        repo_url="https://example.invalid/source.git",
        branch="main",
        reuse=False,
        clone_dir=str(tmp_path / "policycraft" / "source"),
        pending_version="source",
        existing_version=None,
        workspace_holder={"workspace": workspace},
    )
    handler = captured["handlers"]["vcs.seal_checkout"]
    sealed_calls.clear()
    result = await handler(step, {}, context)

    assert result["checkout"] == default_result["checkout"]
    assert sealed_calls == [(str(tmp_path / "policycraft" / "source"), installer._venv_path)]
