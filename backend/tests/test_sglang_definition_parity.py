"""SGLang parity pins — both variants are fully definition-owned.

Neither SGLang variant has an installer class: flows run on the shared
``GenericVenvInstaller``. The fork's upstream rewrites are versioned transform
assets applied by ``vcs.apply_transform`` and its builder environment is
declared through ``build.prepare_fork``; the byte-contract deltas (status
projection, start wording, log milestones, interpreter requirement, release
delegation) are declared data.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.engines.definitions import load_default_engine_definitions, materialize_plan
from backend.engines.python.generic import GenericVenvInstaller

FLOWS = ("release", "source", "sync", "remove")

EXPECTED_HANDLER_KEYS = {
    "release": ["publish.finalize"],
    "source": ["env.compose", "publish.finalize", "vcs.seal_checkout", "vcs.workspace_checkout"],
    "sync": ["env.compose", "publish.finalize", "vcs.sync"],
    "remove": ["publish.forget"],
}

EXPECTED_CONTEXT_KEYS = {
    "release": {"input.engine_root", "input.force_reinstall", "input.requested_version"},
    "source": {
        "input.branch",
        "input.dist_dir",
        "input.engine_root",
        "input.repo_url",
        "input.reuse_mode",
    },
    "sync": {"input.branch", "input.engine_root", "input.source_repo", "input.version_name"},
    "remove": {
        "input.engine_root",
        "input.install_dir",
        "input.installs_root",
        "input.retire_references",
        "input.venv_path",
        "input.venv_python_exists",
        "input.version_name",
    },
}

SPECIALIZED_PRIMITIVES = ("vcs.apply_transform", "build.prepare_fork")


def _installer(tmp_path) -> GenericVenvInstaller:
    definition = load_default_engine_definitions().definitions["sglang"]
    return GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "sglang.log"),
        base_dir=str(tmp_path / "sglang"),
    )


class _CompletedExecution:
    status = "completed"
    failed_step = None


def _capture(monkeypatch):
    calls: list[dict] = []

    async def fake_execute_plan(plan, *, run_primitive, context):
        calls[-1]["plan"] = _step_signature(plan)
        calls[-1]["context"] = sorted(context)
        return _CompletedExecution()

    def fake_make_engine_runner(installer, handlers):
        calls.append({"handlers": sorted(handlers)})
        return lambda *args, **kwargs: None

    monkeypatch.setattr("backend.engines.python.generic.execute_plan", fake_execute_plan)
    monkeypatch.setattr(
        "backend.engines.python.generic.make_engine_runner", fake_make_engine_runner
    )
    return calls


def _step_signature(plan) -> list[tuple[str, str]]:
    steps = getattr(plan, "steps", plan)
    return [(str(getattr(step, "id", "")), str(getattr(step, "primitive", ""))) for step in steps]


def _flow_kwargs(tmp_path, flow: str) -> dict:
    root = tmp_path / "sglang"
    return {
        "release": {
            "requested_version": "0.9.0",
            "pending_version": "pip",
            "existing_version": None,
            "force_reinstall": False,
        },
        "source": {
            "repo_url": "https://example.invalid/sglang.git",
            "branch": "main",
            "reuse": False,
            "clone_dir": str(root / "source"),
            "pending_version": "source",
            "existing_version": None,
            "workspace_holder": {},
        },
        "sync": {
            "branch": "main",
            "clone_dir": str(root / "source"),
            "version_name": "0.9.0",
            "source_repo": "https://example.invalid/sglang.git",
        },
        "remove": {
            "venv_path": str(root / "venv"),
            "python_exists": False,
            "retire_references": False,
            "version_name": "0.9.0",
            "store": SimpleNamespace(),
        },
    }[flow]


def _flow_method(installer, flow: str):
    return {
        "release": installer._execute_release_plan,
        "source": installer._execute_source_plan,
        "sync": installer._execute_sync_plan,
        "remove": installer._execute_remove_plan,
    }[flow]


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", FLOWS)
async def test_upstream_flow_executes_the_definition_plan(tmp_path, monkeypatch, flow):
    installer = _installer(tmp_path)
    calls = _capture(monkeypatch)

    await _flow_method(installer, flow)(**_flow_kwargs(tmp_path, flow))

    assert len(calls) == 1, calls
    definition = load_default_engine_definitions().get(installer.engine_id)
    assert definition is not None
    expected = materialize_plan(definition, flow)
    assert expected is not None, flow
    assert calls[0]["plan"] == _step_signature(expected)
    assert calls[0]["handlers"] == EXPECTED_HANDLER_KEYS[flow]
    assert calls[0]["context"] == sorted(EXPECTED_CONTEXT_KEYS[flow])


def test_upstream_sglang_has_one_manager_everywhere():
    """P1: routes, adapters and retry/sync must share the same singleton.

    The versions router used to select the LEGACY installer while the adapter
    registry selected the upstream one, so a retried operation could be
    invisible to the status/cancel path.
    """
    from backend.engines.adapters import get_engine_installer
    from backend.engines.sglang import get_sglang_manager, get_sglang_upstream_manager

    upstream = get_sglang_upstream_manager()
    assert get_sglang_manager("sglang") is upstream
    assert get_engine_installer("sglang") is upstream


@pytest.mark.asyncio
@pytest.mark.parametrize("flow", ("source", "sync"))
async def test_v100_flow_executes_its_recipe_with_shared_primitives(tmp_path, monkeypatch, flow):
    from backend.engines.sglang import get_sglang_v100_manager

    V100_HANDLERS = {
        "source": [
            "build.prepare_fork",
            "env.compose",
            "publish.finalize",
            "toolchain.ensure",
            "vcs.apply_transform",
            "vcs.seal_checkout",
            "vcs.workspace_checkout",
        ],
        "sync": [
            "build.prepare_fork",
            "env.compose",
            "publish.finalize",
            "toolchain.ensure",
            "vcs.apply_transform",
            "vcs.sync",
        ],
    }

    manager = get_sglang_v100_manager()
    calls = _capture(monkeypatch)
    root = tmp_path / "v100"
    if flow == "source":
        await manager._execute_source_plan(
            repo_url="https://example.invalid/sglang-v100.git",
            branch="main",
            reuse=False,
            clone_dir=str(root / "source"),
            pending_version="source",
            existing_version=None,
            workspace_holder={},
        )
    else:
        await manager._execute_sync_plan(
            branch="main",
            clone_dir=str(root / "source"),
            version_name="0.4.0",
            source_repo="https://example.invalid/sglang-v100.git",
        )
    assert len(calls) == 1, calls
    definition = load_default_engine_definitions().get(manager.engine_id)
    expected = materialize_plan(definition, flow)
    assert calls[0]["plan"] == _step_signature(expected)
    assert calls[0]["handlers"] == V100_HANDLERS[flow]
