"""Phase-3 engine #3 gate: upstream SGLang release installs via the plan.

SGLang shares the PythonVenvInstaller surface but keeps its own release
quirks, all pinned here: the ``--pre`` flag between --upgrade and
--force-reinstall, the runner-side detection feeding the row's base version
before finalize's OWN detection (two detects, no installed-state update),
the "pip install failed with status N" / "Installed SGLang package could not
be imported" strings, and the v100 twin's release-to-source delegation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller


def _make_installer(tmp_path: Path, calls: list, *, detected="0.4.0"):
    definition = load_default_engine_definitions().definitions["sglang"]
    installer = GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "sglang.log"),
        base_dir=str(tmp_path / "sglang"),
    )

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, kwargs.get("append")))
        return 0

    async def fake_progress(progress, message, **kwargs):
        calls.append(("progress", progress, message))

    def fake_detect():
        calls.append(("detect",))
        return detected

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._update_progress_task = fake_progress
    installer._ensure_venv = lambda: calls.append(("ensure_venv",))
    installer._detect_installed_version = fake_detect
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
    [("0.4.0", "sglang==0.4.0"), (None, "sglang")],
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
    assert pip[1] == ["install", "--upgrade", "--pre", expected_package]
    assert pip[2] == "install"
    assert pip[3] is False  # legacy passed append=False explicitly
    # Two detections in legacy order: runner's (feeds the row base), then
    # finalize's own — progress(98) sits between them.
    order = [call[0] for call in calls]
    assert order[:6] == ["ensure_venv", "pip", "detect", "progress", "detect", "unique"]
    assert ("progress", 98, "Validating SGLang installation") in calls
    assert ("unique", "sglang", True) in calls
    ready = next(call for call in calls if call[0] == "ready")
    row = ready[2]
    assert row["version"] == "unique-name"
    assert row["package_version"] == "0.4.0"
    assert row["install_type"] == "pip"
    assert "reuse_existing" not in row
    assert row["venv_path"] == installer._venv_path
    assert ("activate", "sglang", "published-name") in calls  # ready()'s RETURN wins
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_release_force_reinstall_argv_keeps_pre_flag_order(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "unique_version_name", lambda *a: "u")
    await _drive(installer, task_holder, calls, version="0.4.0", force_reinstall=True)
    pip = next(call for call in calls if call[0] == "pip")
    assert pip[1] == ["install", "--upgrade", "--pre", "--force-reinstall", "sglang==0.4.0"]


@pytest.mark.asyncio
async def test_release_reuse_keeps_pending_name_without_uniquifying(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    def no_unique(*a):
        calls.append(("unique", "FORBIDDEN"))
        return "should-not-be-used"

    monkeypatch.setattr(generic_module, "unique_version_name", no_unique)
    reuse_dir = tmp_path / "sglang"
    reuse_dir.mkdir(parents=True, exist_ok=True)

    await _drive(
        installer,
        task_holder,
        calls,
        version="0.4.0",
        reuse_dir=str(reuse_dir),
        existing_version="0.4.0-old",
    )

    assert not [call for call in calls if call[0] == "unique"]
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[1] == "0.4.0-old"  # pending = existing_version, as legacy
    assert ready[2]["version"] == "0.4.0-old"
    assert "reuse_existing" not in ready[2]  # popped before publishing


@pytest.mark.asyncio
async def test_release_pip_failure_uses_legacy_error_string(tmp_path: Path):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)

    async def failing_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage))
        return 7

    installer._run_pip = failing_run_pip
    await _drive(installer, task_holder, calls, version="0.4.0")

    assert installer._last_error == "pip install failed with status 7"
    assert ("fail", "pip install failed with status 7", False) in calls
    # The shared host refreshes install state after a failure (documented
    # normalization, R15); what must NOT happen is finalize publishing.
    assert not [call for call in calls if call[0] == "ready"]
    assert not [call for call in calls if call[0] == "ready"]


@pytest.mark.asyncio
async def test_release_import_detection_failure_matches_legacy(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls, detected=None)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "unique_version_name", lambda *a: "u")

    await _drive(installer, task_holder, calls, version="0.4.0")

    assert installer._last_error == "Installed SGLang package could not be imported"
    assert ("fail", "Installed SGLang package could not be imported", False) in calls
    assert not [call for call in calls if call[0] == "ready"]


@pytest.mark.asyncio
async def test_v100_release_delegates_to_source_build(tmp_path: Path):
    # The v100 twin never takes the pip release path; that routing lives
    # OUTSIDE any plan and must survive the cutover untouched.
    calls: list = []
    installer = GenericVenvInstaller(
        load_default_engine_definitions().definitions["sglang_v100"],
        log_path=str(tmp_path / "sg.log"),
        base_dir=str(tmp_path / "sg-v100"),
    )

    async def fake_source(repo_url, branch, *, reuse_dir=None, existing_version=None):
        calls.append(("source", repo_url, branch, reuse_dir, existing_version))
        return {"started": True}

    installer.install_from_source = fake_source
    result = await installer.install_release(
        "0.4.0", reuse_dir="/tmp/x", existing_version="v100-old"
    )

    assert result == {"started": True}
    assert calls == [
        (
            "source",
            dict(load_default_engine_definitions().definitions["sglang_v100"].lifecycle)[
                "default_repo_url"
            ],
            "main",
            "/tmp/x",
            "v100-old",
        )
    ]
