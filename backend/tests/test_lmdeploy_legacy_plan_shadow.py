"""Phase-3 gates: LMDeploy installs, syncs, and removes via definition plans.

Gates 1-4 (release, source install, source sync, removal) are cut over:
LMDeployInstaller executes the committed recipes with primitive handlers that
must reproduce legacy behavior byte-for-byte (git/pip argv, error strings,
workspace lifecycle, swallow-on-metadata-failure, activation payloads).
These tests are the regression gates: they capture argv and call sequences at
the process/store boundary and pin them against the legacy behavior recorded
before each cutover. status() stays a legacy read path by design.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.engines.definitions import (
    REFERENCE_DEFINITIONS_DIR,
    execute_plan,
    load_engine_definitions,
    materialize_plan,
)
from backend.tests.venv_fixtures import LMDeployManager


@pytest.fixture(autouse=True)
def _ccache_available(monkeypatch):
    """CI runners may not ship ccache; the plan still declares its env layer."""
    import shutil

    import backend.engines.build_workspace as workspace

    if shutil.which("ccache") is None:
        monkeypatch.setattr(
            workspace.shutil,
            "which",
            lambda name: "/usr/bin/ccache" if name == "ccache" else None,
        )


async def _run_release(tmp_path: Path, **kwargs) -> tuple[list, LMDeployManager]:
    """Drive install_release with the pip subprocess faked; return ops + manager."""
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )
    ops: list = []

    async def fake_run_pip(args, stage, **kwargs_):
        ops.append(("venv.ensure", "implicit"))
        ops.append(("pip.install", list(args)))
        return 0

    async def fake_activate(engine, version_name, **kwargs_):
        ops.append(("publish.activate", engine, version_name))

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._detect_installed_version = lambda: "0.10.1"
    installer._update_installed_state = lambda *a, **k: None
    installer._ready_pending_version = lambda *a, **k: None

    import backend.engines.python.generic as activation

    original = activation.activate_engine_version
    activation.activate_engine_version = fake_activate
    try:
        response = await installer.install_release(**kwargs)
        await task_holder["coro"]
    finally:
        activation.activate_engine_version = original
        from backend.operations.supervisor import get_supervisor

        get_supervisor().finish_operation(response["task_id"], "succeeded", "")
    return ops, installer


@pytest.mark.asyncio
@pytest.mark.parametrize("requested_version", ["0.10.1", None])
async def test_legacy_release_flow_maps_onto_reference_plan(tmp_path: Path, requested_version):
    ops, installer = await _run_release(tmp_path, version=requested_version)

    assert [op[0] for op in ops] == ["venv.ensure", "pip.install", "publish.activate"]
    legacy_pip = next(op for op in ops if op[0] == "pip.install")
    legacy_publish = next(op for op in ops if op[0] == "publish.activate")

    # ---- the plan side, over the committed reference definition ----
    registry = load_engine_definitions(REFERENCE_DEFINITIONS_DIR)
    plan = materialize_plan(registry.get("lmdeploy"), "release")
    plan_ops: list = []

    def runner(primitive, step, requirements, context):
        plan_ops.append((primitive, dict(step.with_)))
        return {name: ("0.10.1" if name == "version" else f"{name}-x") for name in step.produces}

    execution = await execute_plan(
        plan, run_primitive=runner, context={"input.engine_root": installer._base_dir}
    )
    assert execution.status == "completed"
    assert [prim for prim, _ in plan_ops] == [
        "prereqs.resolve_version",
        "venv.create",
        "pip.install_package",
        "pip.capture_metadata",
        "publish.install",
    ]

    # ---- step-for-step mapping (input to the Phase 3 migration table) ----
    #   legacy venv.ensure (implicit inside _run_pip)  <->  venv.create
    #   legacy pip.install                             <->  pip.install_package
    #   legacy _detect_installed_version               <->  pip.capture_metadata
    #   legacy activate_engine_version                 <->  publish.install
    install_with = next(with_ for prim, with_ in plan_ops if prim == "pip.install_package")
    assert install_with["package"] == "lmdeploy"
    assert install_with["version_from"] == "version"
    assert legacy_publish[1] == "lmdeploy"

    # ---- byte-for-byte legacy argv (the cutover's behavior-preservation bar)
    if requested_version:
        assert legacy_pip[1] == ["install", "--upgrade", f"lmdeploy=={requested_version}"]
    else:
        # Latest: legacy floated ("lmdeploy"); the cutover preserves that
        # byte-for-byte. Pinning the resolved version would be a future,
        # explicit behavior change — not smuggled into a migration.
        assert legacy_pip[1] == ["install", "--upgrade", "lmdeploy"]


@pytest.mark.asyncio
async def test_force_reinstall_argv_matches_legacy_exactly(tmp_path: Path):
    ops, _ = await _run_release(tmp_path, version="1.2.3", force_reinstall=True)
    pip_install = next(op for op in ops if op[0] == "pip.install")
    assert pip_install[1] == ["install", "--upgrade", "--force-reinstall", "lmdeploy==1.2.3"]


@pytest.mark.asyncio
async def test_pip_failure_preserves_the_legacy_error_string(tmp_path: Path):
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    async def failing_run_pip(args, stage, **kwargs):
        return 7

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = failing_run_pip
    installer._fail_pending_version = lambda *a, **k: None
    installer._refresh_state_from_environment = lambda: None

    response = await installer.install_release(version="0.10.1")
    await task_holder["coro"]
    assert installer._last_error == "pip exited with status 7"


# ---------------------------------------------------------------------------
# Gate 2: source install (reuse-mode shallow clone / shared-workspace modes)
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, returncode: int) -> None:
        self.returncode = returncode

    async def wait(self) -> int:
        return self.returncode


def _make_source_installer(tmp_path: Path, monkeypatch, calls: list, clone_calls: list):
    """Manager with pip/clone/activation faked; returns (installer, task_holder,
    activation module, original activate)."""
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, kwargs.get("cwd"), kwargs.get("env") or {}))
        return 0

    async def fake_exec(*argv, **kwargs):
        clone_calls.append([str(a) for a in argv])
        return _FakeProc(0)

    async def fake_activate(engine, version_name, **kwargs):
        calls.append(("activate", engine, version_name))

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._detect_installed_version = lambda: "0.7.0.dev0"
    installer._update_installed_state = lambda *a, **k: None
    installer._ready_pending_version = lambda *a, **k: None

    import asyncio

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)

    import backend.engines.python.generic as activation

    original = activation.activate_engine_version
    activation.activate_engine_version = fake_activate
    return installer, task_holder, activation, original


@pytest.mark.asyncio
async def test_source_reuse_mode_preserves_clone_and_pip_argv(tmp_path: Path, monkeypatch):
    calls: list = []
    clone_calls: list = []
    installer, task_holder, activation, original = _make_source_installer(
        tmp_path, monkeypatch, calls, clone_calls
    )
    try:
        await installer.install_from_source(
            "https://example.invalid/lmdeploy.git",
            "trunk",
            reuse_dir=str(tmp_path / "lmdeploy"),
            existing_version="source-existing",
        )
        await task_holder["coro"]
    finally:
        activation.activate_engine_version = original

    clone_dir = str(Path(tmp_path / "lmdeploy" / "source"))
    assert clone_calls == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "trunk",
            "https://example.invalid/lmdeploy.git",
            clone_dir,
        ]
    ]
    install = next(call for call in calls if call[0] == "pip")
    assert install[1] == ["install", "-v", "-e", "."]
    assert install[2] == "install_source"
    assert install[3] == clone_dir
    assert any(key.startswith("CCACHE") for key in install[4])
    activate = next(call for call in calls if call[0] == "activate")
    assert activate[1] == "lmdeploy"


@pytest.mark.asyncio
async def test_source_workspace_mode_lifecycle_matches_legacy(tmp_path: Path, monkeypatch):
    calls: list = []
    clone_calls: list = []
    installer, task_holder, activation, original = _make_source_installer(
        tmp_path, monkeypatch, calls, clone_calls
    )

    class _FakeWorkspace:
        def __init__(self) -> None:
            self.checkout_dir = str(tmp_path / "shared-checkout")

        def acquire(self):
            calls.append("acquire")

        def sync_git(self, repo, branch):
            calls.append(("sync_git", repo, branch))

        def seal_installed_source(self, destination, venv_path):
            calls.append(("seal", destination, venv_path))
            return destination

    import backend.engines.build_workspace as bw

    def fake_open(engine, repo, meta):
        calls.append(("open", engine, repo, meta))
        return _FakeWorkspace()

    released: list = []
    monkeypatch.setattr(bw.BuildWorkspace, "open", staticmethod(fake_open))
    monkeypatch.setattr(bw, "release_held", lambda workspace: released.append(workspace))

    try:
        await installer.install_from_source("https://example.invalid/lmdeploy.git", "main")
        await task_holder["coro"]
    finally:
        activation.activate_engine_version = original

    assert clone_calls == []  # workspace mode never shallow-clones
    assert calls[0] == (
        "open",
        "lmdeploy",
        "https://example.invalid/lmdeploy.git",
        {"kind": "source"},
    )
    assert calls[1] == "acquire"
    assert calls[2] == ("sync_git", "https://example.invalid/lmdeploy.git", "main")
    install = next(call for call in calls if call[0] == "pip")
    assert install[3] == str(tmp_path / "shared-checkout")
    seal = next(call for call in calls if call[0] == "seal")
    # seal destination is the operation-bound install dir (timestamped on a
    # fresh install), exactly as legacy computed it.
    assert seal[1] == str(Path(installer._base_dir) / "source")
    assert len(released) == 1 and isinstance(released[0], _FakeWorkspace)


@pytest.mark.asyncio
async def test_source_clone_failure_preserves_legacy_error_string(tmp_path: Path, monkeypatch):
    calls: list = []
    clone_calls: list = []
    installer, task_holder, activation, original = _make_source_installer(
        tmp_path, monkeypatch, calls, clone_calls
    )

    async def failing_exec(*argv, **kwargs):
        clone_calls.append([str(a) for a in argv])
        return _FakeProc(128)

    import asyncio

    monkeypatch.setattr(asyncio, "create_subprocess_exec", failing_exec)
    installer._fail_pending_version = lambda *a, **k: None
    installer._refresh_state_from_environment = lambda: None

    try:
        await installer.install_from_source(
            "https://example.invalid/lmdeploy.git",
            "main",
            reuse_dir=str(tmp_path / "lmdeploy"),
            existing_version="source-existing",
        )
        await task_holder["coro"]
    finally:
        activation.activate_engine_version = original
    assert installer._last_error == "git clone failed with code 128"
    assert not [call for call in calls if call[0] == "activate"]


# ---------------------------------------------------------------------------
# Gate 3: source sync (vcs.sync -> editable reinstall -> metadata update)
# ---------------------------------------------------------------------------


def _make_sync_installer(tmp_path: Path, monkeypatch, calls: list):
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(
            (
                "pip",
                list(args),
                stage,
                kwargs.get("cwd"),
                kwargs.get("env") or {},
                kwargs.get("append"),
            )
        )
        return 0

    async def fake_sync_git(checkout_dir, branch):
        calls.append(("sync_git", checkout_dir, branch))

    async def fake_head(checkout_dir):
        return "deadbeef"

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._ensure_venv = lambda: None
    installer._sync_git_checkout = fake_sync_git
    installer._detect_installed_version = lambda: "0.7.0.dev0"
    installer._update_installed_state = lambda *a, **k: None
    installer._git_head = fake_head

    import backend.engines.python.generic as installer_module

    monkeypatch.setattr(installer_module, "mark_runtime_changed", lambda: calls.append("mark"))
    return installer, task_holder


_SYNC_ENTRY = {
    "install_type": "source",
    "source_branch": "main",
    "source_repo": "https://example.invalid/lmdeploy.git",
    "version": "0.7.0-dev1",
    "venv_path": "VENVPATH",
}


def _sync_entry(tmp_path: Path) -> dict:
    entry = dict(_SYNC_ENTRY)
    entry["venv_path"] = str(tmp_path / "lmdeploy" / "venv")
    return entry


@pytest.mark.asyncio
async def test_sync_plan_preserves_legacy_calls_and_payload(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_sync_installer(tmp_path, monkeypatch, calls)

    class _FakeStore:
        def update_engine_version(self, engine, version, patch):
            calls.append(("update", engine, version, patch))
            return {"version": version}

    import backend.data_store  # noqa: F401  (import target for realism)
    import backend.engines.python.generic as installer_module
    import backend.engines.scan.scanner as scanner_module

    monkeypatch.setattr(installer_module, "get_store", lambda: _FakeStore())
    monkeypatch.setattr(
        scanner_module,
        "scan_engine_version",
        lambda store, engine, row: calls.append(("scan", engine, row)),
    )

    await installer.sync_source_version(_sync_entry(tmp_path))
    await task_holder["coro"]

    sync_git = calls[0]
    assert sync_git == ("sync_git", str(tmp_path / "lmdeploy" / "source"), "main")
    pip = next(call for call in calls if call[0] == "pip")
    assert pip[1] == ["install", "-v", "-e", "."]
    assert pip[2] == "sync_source"
    assert pip[3] == str(tmp_path / "lmdeploy" / "source")
    assert pip[5] is True  # legacy appended to the install log
    assert any(key.startswith("CCACHE") for key in pip[4])
    update = next(call for call in calls if call[0] == "update")
    assert update[1] == "lmdeploy"
    assert update[2] == "0.7.0-dev1"
    assert update[3]["source_commit"] == "deadbeef"
    assert update[3]["source_branch"] == "main"
    assert update[3]["source_repo"] == "https://example.invalid/lmdeploy.git"
    assert update[3]["venv_path"] == str(tmp_path / "lmdeploy" / "venv")
    assert ("scan", "lmdeploy", {"version": "0.7.0-dev1"}) in calls
    assert "mark" in calls
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_sync_pip_failure_preserves_legacy_error_string(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_sync_installer(tmp_path, monkeypatch, calls)

    async def failing_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage))
        return 5

    installer._run_pip = failing_run_pip
    installer._refresh_state_from_environment = lambda: None

    await installer.sync_source_version(_sync_entry(tmp_path))
    await task_holder["coro"]
    assert installer._last_error == "pip install -e -v . failed with code 5"
    assert not [call for call in calls if call[0] == "update"]


@pytest.mark.asyncio
async def test_sync_succeeds_when_metadata_update_fails(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_sync_installer(tmp_path, monkeypatch, calls)

    class _BrokenStore:
        def update_engine_version(self, *args):
            calls.append(("update", "attempted"))
            raise RuntimeError("store is down")

    import backend.engines.python.generic as installer_module

    monkeypatch.setattr(installer_module, "get_store", lambda: _BrokenStore())

    await installer.sync_source_version(_sync_entry(tmp_path))
    await task_holder["coro"]
    # Legacy swallow-everything block: sync SUCCEEDS, mark_runtime_changed
    # is skipped, no error is recorded.
    assert ("update", "attempted") in calls
    assert installer._last_error is None
    assert "mark" not in calls


# ---------------------------------------------------------------------------
# Gate 4: removal (uninstall -> rmtree -> forget), including partial installs
# ---------------------------------------------------------------------------


def _make_remove_installer(tmp_path: Path, monkeypatch, calls: list, *, active):
    import backend.data_store
    import backend.engines.python.generic as installer_module
    import backend.utils.fs_ops as fs_ops

    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )
    venv_dir = tmp_path / "lmdeploy" / "venv"
    fake_python = venv_dir / "bin" / "python"
    fake_python.parent.mkdir(parents=True, exist_ok=True)
    fake_python.write_text("")  # presence = "venv python exists"

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, kwargs.get("ensure_venv")))
        return 0

    # remove() fetches the active row with a method-local get_store import,
    # so the fake lives on the real store class's two methods — the
    # supervisor keeps the untouched real store for everything else.
    store_cls = type(backend.data_store.get_store())
    monkeypatch.setattr(store_cls, "get_active_engine_version", lambda self, engine: active)
    # The post-remove stale-version sweep is engine-wide; this harness pins the
    # legacy version-scoped recipe, so keep the global store out of it.
    monkeypatch.setattr(store_cls, "get_engine_versions", lambda self, engine: [])
    monkeypatch.setattr(
        store_cls,
        "delete_engine_version",
        lambda self, engine, version: calls.append(("delete", engine, version)),
    )

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._update_installed_state = lambda *a, **k: calls.append(("state", a))

    monkeypatch.setattr(installer_module, "mark_runtime_changed", lambda: calls.append(("mark",)))
    monkeypatch.setattr(
        fs_ops,
        "release_launch_hold",
        lambda path, retire_references=False: calls.append(("hold", path, retire_references)),
    )
    monkeypatch.setattr(
        fs_ops,
        "robust_rmtree",
        lambda path, retire_references=False: calls.append(("rmtree", path, retire_references)),
    )
    return installer, task_holder, venv_dir


_ACTIVE_ROW = {"version": "0.7.0-dev1"}


@pytest.mark.asyncio
async def test_remove_plan_preserves_legacy_step_sequence(tmp_path: Path, monkeypatch):
    calls: list = []
    active = dict(_ACTIVE_ROW, venv_path=str(tmp_path / "lmdeploy" / "venv"))
    installer, task_holder, venv_dir = _make_remove_installer(
        tmp_path, monkeypatch, calls, active=active
    )
    await installer.remove(retire_references=True)
    await task_holder["coro"]

    kinds = [call[0] for call in calls]
    assert kinds == ["hold", "pip", "rmtree", "delete", "state", "mark"]
    hold = calls[0]
    assert hold[1] == str(venv_dir) and hold[2] is True  # pre-operation, as legacy
    pip = calls[1]
    assert pip[1] == ["uninstall", "-y", "lmdeploy"]
    assert pip[2] == "remove"
    assert pip[3] is False  # legacy passed ensure_venv=False
    rmtree = calls[2]
    assert rmtree[1] == str(venv_dir) and rmtree[2] is True
    assert calls[3] == ("delete", "lmdeploy", "0.7.0-dev1")
    assert calls[4] == ("state", (False, None))
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_remove_half_deleted_install_skips_uninstall(tmp_path: Path, monkeypatch):
    calls: list = []
    active = dict(_ACTIVE_ROW, venv_path=str(tmp_path / "lmdeploy" / "venv"))
    installer, task_holder, venv_dir = _make_remove_installer(
        tmp_path, monkeypatch, calls, active=active
    )
    (venv_dir / "bin" / "python").unlink()  # broken venv: no interpreter

    await installer.remove()
    await task_holder["coro"]

    kinds = [call[0] for call in calls]
    assert "pip" not in kinds  # the step condition skipped the uninstall
    assert kinds == ["hold", "rmtree", "delete", "state", "mark"]
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_remove_pip_failure_stops_before_teardown(tmp_path: Path, monkeypatch):
    calls: list = []
    active = dict(_ACTIVE_ROW, venv_path=str(tmp_path / "lmdeploy" / "venv"))
    installer, task_holder, venv_dir = _make_remove_installer(
        tmp_path, monkeypatch, calls, active=active
    )

    async def failing_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage))
        return 9

    installer._run_pip = failing_run_pip
    installer._refresh_state_from_environment = lambda: None

    await installer.remove()
    await task_holder["coro"]
    assert installer._last_error == "pip exited with status 9"
    kinds = [call[0] for call in calls]
    assert "rmtree" not in kinds and "delete" not in kinds  # legacy aborted here too


@pytest.mark.asyncio
async def test_remove_swallows_metadata_delete_failure(tmp_path: Path, monkeypatch):
    calls: list = []
    active = dict(_ACTIVE_ROW, venv_path=str(tmp_path / "lmdeploy" / "venv"))
    installer, task_holder, venv_dir = _make_remove_installer(
        tmp_path, monkeypatch, calls, active=active
    )

    def broken_delete(self, engine, version):
        calls.append(("delete", "attempted"))
        raise RuntimeError("store is read-only")

    import backend.data_store

    monkeypatch.setattr(
        type(backend.data_store.get_store()), "delete_engine_version", broken_delete
    )

    await installer.remove()
    await task_holder["coro"]
    # Legacy swallowed the delete failure and still completed removal.
    assert ("delete", "attempted") in calls
    assert installer._last_error is None
    assert calls[-1] == ("mark",)
