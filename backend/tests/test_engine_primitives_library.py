"""Unit tests for the shared primitive library (Phase 3 extraction).

The engine gate suites (test_lmdeploy_legacy_plan_shadow.py,
test_vllm_release_plan_shadow.py) prove end-to-end byte preservation; these
tests pin the library-level contracts the engines rely on: tri-state append
(ABSENT with_ key must not override _run_pip's append=True default), error
templates resolved from the recipe, and make_engine_runner dispatch rules.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from backend.engines.definitions import PrimitiveExecutionError, make_engine_runner
from backend.engines.definitions.primitives import SHARED_PRIMITIVES, run_shared_primitive


class _FakeInstaller:
    engine_id = "fake"

    def __init__(self) -> None:
        self.calls: List[Any] = []
        self._venv_path = "/venv"

    async def _run_pip(self, args, stage, **kwargs):
        self.calls.append(("pip", list(args), stage, kwargs))
        return 0

    def _ensure_venv(self):
        self.calls.append(("ensure_venv",))

    def _detect_installed_version(self):
        return "1.2.3"

    def _update_installed_state(self, *a, **k):
        self.calls.append(("state", a))


def _step(**with_) -> SimpleNamespace:
    return SimpleNamespace(id="s", primitive="x", with_=with_)


def _run(installer, primitive, step, requirements=None, context=None):
    return run_shared_primitive(installer, primitive, step, requirements or {}, context or {})


@pytest.mark.asyncio
async def test_append_is_tri_state_absent_key_keeps_pip_default():
    installer = _FakeInstaller()
    # Legacy LMDeploy release omitted append -> _run_pip default True applies.
    await _run(
        installer,
        "pip.install_package",
        _step(package="x", version_from="version", stage="install"),
        {"version": "1.0"},
    )
    _, args, _, kwargs = installer.calls[0]
    assert "append" not in kwargs  # must NOT be forced to False
    assert args == ["install", "--upgrade", "x==1.0"]

    # vLLM pins append=False in the recipe -> passed through.
    installer.calls.clear()
    await _run(
        installer,
        "pip.install_package",
        _step(package="x", version_from="version", stage="install", append=False),
        {"version": ""},
    )
    _, args, _, kwargs = installer.calls[0]
    assert kwargs["append"] is False
    assert args == ["install", "--upgrade", "x"]  # empty pin floats, as legacy


@pytest.mark.asyncio
async def test_error_template_comes_from_the_recipe():
    installer = _FakeInstaller()

    async def failing_pip(args, stage, **kwargs):
        return 7

    installer._run_pip = failing_pip
    with pytest.raises(PrimitiveExecutionError) as excinfo:
        await _run(
            installer,
            "pip.install_package",
            _step(
                package="x", version_from="version", error="pip install failed with status {code}"
            ),
            {"version": ""},
        )
    assert str(excinfo.value) == "pip install failed with status 7"
    assert excinfo.value.exit_code == 7


@pytest.mark.asyncio
async def test_force_reinstall_flag_inserts_flag_before_package():
    installer = _FakeInstaller()
    await _run(
        installer,
        "pip.install_package",
        _step(package="x", version_from="version"),
        {"version": "2.0", "input.force_reinstall": True},
    )
    assert installer.calls[0][1] == ["install", "--upgrade", "--force-reinstall", "x==2.0"]


@pytest.mark.asyncio
async def test_ccache_env_is_composed_around_cwd(monkeypatch):
    import shutil

    import backend.engines.build_workspace as workspace

    if shutil.which("ccache") is None:
        monkeypatch.setattr(
            workspace.shutil,
            "which",
            lambda name: "/usr/bin/ccache" if name == "ccache" else None,
        )
    installer = _FakeInstaller()
    await _run(
        installer,
        "pip.install_package",
        _step(
            package=".", editable=True, stage="install_source", cwd_from="checkout", env="ccache"
        ),
        {"checkout": "/tmp"},
    )
    _, args, stage, kwargs = installer.calls[0]
    assert args == ["install", "-v", "-e", "."]
    assert stage == "install_source"
    assert kwargs["cwd"] == "/tmp"
    assert any(key.startswith("CCACHE") for key in kwargs["env"])


@pytest.mark.asyncio
async def test_uninstall_skips_venv_ensure():
    installer = _FakeInstaller()
    await _run(installer, "pip.uninstall", _step(package="x", stage="remove"))
    assert installer.calls[0][1] == ["uninstall", "-y", "x"]
    assert installer.calls[0][3]["ensure_venv"] is False


@pytest.mark.asyncio
async def test_capture_metadata_updates_state_and_wraps_version():
    installer = _FakeInstaller()
    outputs = await _run(installer, "pip.capture_metadata", _step())
    assert outputs == {"metadata": {"version": "1.2.3"}}
    assert ("state", (True, "1.2.3")) in installer.calls


@pytest.mark.asyncio
async def test_resolve_version_echoes_requested_input():
    installer = _FakeInstaller()
    outputs = await _run(
        installer, "prereqs.resolve_version", _step(), {}, {"input.requested_version": "9.9"}
    )
    assert outputs == {"version": "9.9"}


@pytest.mark.asyncio
async def test_make_engine_runner_routes_engine_then_shared_then_raises():
    installer = _FakeInstaller()
    seen: Dict[str, Any] = {}

    async def engine_handler(step, requirements, context):
        seen["called"] = True
        return {"done": True}

    runner = make_engine_runner(installer, {"publish.install": engine_handler})
    out = await runner("publish.install", _step(), {}, {})
    assert out == {"done": True} and seen["called"]

    out = await runner("venv.create", _step(), {}, {})
    assert out == {"venv": "/venv"}

    with pytest.raises(PrimitiveExecutionError):
        await runner("vcs.teleport", _step(), {}, {})


def test_shared_names_are_declared_primitives():
    from backend.engines.definitions import INSTALL_PRIMITIVES

    assert SHARED_PRIMITIVES <= INSTALL_PRIMITIVES
