"""Phase-3 engine #2 gate: upstream vLLM release installs via the plan.

The vLLM release cutover must reproduce legacy behavior byte-for-byte —
including the places vLLM deliberately DIFFERS from LMDeploy: error strings
("pip install failed with status N", "Installed vLLM package could not be
imported"), the publish.finalize semantics (import must detect; reuse keeps
the pending name; the row is keyed by _ready_pending_version's RETURN), and
the absence of the state-refresh step on failure. These tests pin all of it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller


def _make_installer(tmp_path: Path, calls: list, *, detected="0.6.3"):
    definition = load_default_engine_definitions().definitions["vllm"]
    installer = GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "vllm.log"),
        base_dir=str(tmp_path / "vllm"),
    )

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, kwargs.get("append")))
        return 0

    async def fake_progress(progress, message, **kwargs):
        calls.append(("progress", progress, message))

    installer._create_task = None  # replaced per test after task capture
    task_holder: dict = {}

    def capture(coro):
        task_holder["coro"] = coro

    installer._create_task = capture
    installer._run_pip = fake_run_pip
    installer._update_progress_task = fake_progress
    installer._detect_installed_version = lambda *a, **k: detected
    installer._ready_pending_version = lambda pending, meta: (
        calls.append(("ready", pending, dict(meta))),
        "published-name",
    )[1]
    installer._fail_pending_version = lambda pending, message, extra, **kw: calls.append(
        ("fail", message, bool(kw.get("cancelled")))
    )
    return installer, task_holder


async def _drive(installer, task_holder, calls, **kwargs):
    import backend.engines.python.generic as generic_module

    async def fake_activate(engine, version_name, **kw):
        calls.append(("activate", engine, version_name))

    original = generic_module.activate_engine_version
    generic_module.activate_engine_version = fake_activate
    try:
        response = await installer.install_release(**kwargs)
        await task_holder["coro"]
    finally:
        generic_module.activate_engine_version = original
        from backend.operations.supervisor import get_supervisor

        get_supervisor().finish_operation(response["task_id"], "succeeded", "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "expected_package"),
    [("0.6.3", "vllm==0.6.3"), (None, "vllm")],
)
async def test_release_argv_and_publish_match_legacy(
    tmp_path: Path, monkeypatch, requested, expected_package
):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(
        generic_module,
        "unique_version_name",
        lambda store, engine, base: calls.append(("unique", engine, bool(base))) or "unique-name",
    )

    await _drive(installer, task_holder, calls, version=requested)

    pip = next(call for call in calls if call[0] == "pip")
    assert pip[1] == ["install", "--upgrade", expected_package]
    assert pip[2] == "install"
    assert pip[3] is False  # legacy passed append=False explicitly
    assert ("progress", 98, "Validating vLLM installation") in calls
    assert ("unique", "vllm", True) in calls  # fresh install takes the unique path
    ready = next(call for call in calls if call[0] == "ready")
    row = ready[2]
    assert row["version"] == "unique-name"
    assert row["package_version"] == "0.6.3"
    assert row["install_type"] == "pip"
    assert "reuse_existing" not in row
    assert row["venv_path"] == installer._venv_path
    assert ("activate", "vllm", "published-name") in calls  # ready()'s RETURN wins
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_release_force_reinstall_argv(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "unique_version_name", lambda *a: "u")
    await _drive(installer, task_holder, calls, version="0.6.3", force_reinstall=True)
    pip = next(call for call in calls if call[0] == "pip")
    assert pip[1] == ["install", "--upgrade", "--force-reinstall", "vllm==0.6.3"]


@pytest.mark.asyncio
async def test_release_reuse_keeps_pending_name_without_uniquifying(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    def no_unique(*a):
        calls.append(("unique", "FORBIDDEN"))
        return "should-not-be-used"

    monkeypatch.setattr(generic_module, "unique_version_name", no_unique)
    reuse_dir = tmp_path / "vllm"
    reuse_dir.mkdir(parents=True, exist_ok=True)

    await _drive(
        installer,
        task_holder,
        calls,
        version="0.6.3",
        reuse_dir=str(reuse_dir),
        existing_version="0.6.3-old",
    )

    assert not [call for call in calls if call[0] == "unique"]
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[1] == "0.6.3-old"  # pending = existing_version, as legacy
    assert ready[2]["version"] == "0.6.3-old"
    assert "reuse_existing" not in ready[2]  # popped before publishing


@pytest.mark.asyncio
async def test_release_pip_failure_uses_legacy_error_string(tmp_path: Path):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)

    async def failing_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage))
        return 7

    installer._run_pip = failing_run_pip
    await _drive(installer, task_holder, calls, version="0.6.3")

    assert installer._last_error == "pip install failed with status 7"
    assert ("fail", "pip install failed with status 7", False) in calls
    assert not [call for call in calls if call[0] == "ready"]


@pytest.mark.asyncio
async def test_release_import_detection_failure_matches_legacy(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls, detected=None)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "unique_version_name", lambda *a: "u")

    await _drive(installer, task_holder, calls, version="0.6.3")

    # vLLM (unlike LMDeploy) hard-fails when the installed package cannot be
    # imported — pinned string and pending-version failure.
    assert installer._last_error == "Installed vLLM package could not be imported"
    assert ("fail", "Installed vLLM package could not be imported", False) in calls
    assert not [call for call in calls if call[0] == "ready"]
