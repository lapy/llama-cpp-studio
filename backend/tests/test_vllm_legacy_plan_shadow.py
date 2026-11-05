"""Phase-3 vLLM completion gates: source/sync/remove install paths via plans.

These paths deliberately differ from BOTH the vLLM release flow and
LMDeploy's equivalents, and the tests pin exactly those deltas: the logged
progress-clone with "git clone failed with status N" (vs LMDeploy's "code"),
the in-step pip bootstrap ("pip upgrade failed with status N") that creates
the venv via _run_pip's ensure_venv default, the absolute-checkout editable
install into a CUDA+ccache env, the seal-rebind feeding git-head, the
runtime_meta merged into SUCCESS and FAILURE bookkeeping alike, sync's
no-uniquify re-publish, and remove's parent-dir guard plus
non-swallowed delete failures.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.engines.definitions.loader import load_default_engine_definitions
from backend.engines.python.generic import GenericVenvInstaller

FAKE_RUNTIME = {"cuda_version": "12.8", "cuda_path": "/fake-cuda", "python_version": "3.12"}


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


def _make_installer(tmp_path: Path, calls: list, *, detected="0.6.3"):
    definition = load_default_engine_definitions().definitions["vllm"]
    installer = GenericVenvInstaller(
        definition,
        log_path=str(tmp_path / "vllm.log"),
        base_dir=str(tmp_path / "vllm"),
    )

    async def fake_run_pip(args, stage, **kwargs):
        calls.append(("pip", list(args), stage, dict(kwargs)))
        return 0

    async def fake_progress(progress, message, **kwargs):
        calls.append(("progress", progress, message))

    def fake_compose_build_env():
        calls.append(("cuda_env",))
        return {"CUDA_HOME": "/fake-cuda", "VIRTUAL_ENV": installer._venv_path}

    task_holder: dict = {}
    installer._create_task = lambda coro: task_holder.__setitem__("coro", coro)
    installer._run_pip = fake_run_pip
    installer._update_progress_task = fake_progress
    installer._compose_build_env = fake_compose_build_env
    installer._runtime_metadata = lambda: dict(FAKE_RUNTIME)
    installer._ensure_venv = lambda: calls.append(("ensure_venv",))
    installer._detect_installed_version = lambda *a, **k: detected
    installer._ready_pending_version = lambda pending, meta: (
        calls.append(("ready", pending, dict(meta))),
        "published-name",
    )[1]
    installer._fail_pending_version = lambda pending, message, extra, **kw: calls.append(
        ("fail", message, dict(extra or {}), bool(kw.get("cancelled")))
    )
    return installer, task_holder


async def _drive(installer, task_holder, calls, invoke):
    import backend.engines.python.generic as generic_module

    async def fake_activate(engine, version_name, **kw):
        calls.append(("activate", engine, version_name))

    original = generic_module.activate_engine_version
    generic_module.activate_engine_version = fake_activate
    try:
        response = await invoke()
        await task_holder["coro"]
    finally:
        generic_module.activate_engine_version = original
        from backend.operations.supervisor import get_supervisor

        get_supervisor().finish_operation(response["task_id"], "succeeded", "")


def _patch_git_clone(monkeypatch, calls, *, keep_dir: bool = True):
    """Reuse-mode checkout is an asyncio `git clone` subprocess now."""
    import asyncio

    class _Proc:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_exec(*argv, **kwargs):
        # ("git", "clone", "--depth", "1", "--branch", branch, repo, target)
        target = argv[-1]
        calls.append(("clone", argv[6], argv[5], target))
        os.makedirs(target, exist_ok=True)
        if not keep_dir:
            os.rmdir(target)
        return _Proc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)


class _FakeWorkspace:
    def __init__(self, calls, checkout_dir="/ws-checkout"):
        self._calls = calls
        self.checkout_dir = checkout_dir

    def acquire(self):
        self._calls.append(("acquire",))

    def sync_git(self, repo_url, branch):
        self._calls.append(("sync_git", repo_url, branch))

    def seal_installed_source(self, destination, venv_path):
        self._calls.append(("seal", destination, venv_path))
        return "/sealed-checkout"


# --------------------------------------------------------------------------
# source
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_source_reuse_matches_legacy_argv_order_and_rows(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(
        generic_module, "unique_version_name", lambda *a: calls.append(("unique",)) or "u"
    )

    clone_dir = os.path.join(installer._base_dir, "source")

    _patch_git_clone(monkeypatch, calls)

    async def fake_git_head(path):
        calls.append(("git_head", path))
        return "abc123456789def"

    installer._git_head = fake_git_head

    repo = "https://example.invalid/vllm.git"
    await _drive(
        installer,
        task_holder,
        calls,
        lambda: installer.install_from_source(
            repo, "trunk", reuse_dir=str(tmp_path / "vllm"), existing_version="v1"
        ),
    )

    kinds = [call[0] for call in calls]
    # Legacy sequence: cuda env -> logged clone (progress 3/8) -> pip bootstrap
    # -> editable install -> git head -> detect -> finalize (98, detect) -> row.
    assert kinds[:7] == [
        "cuda_env",
        "progress",
        "clone",
        "progress",
        "pip",
        "pip",
        "git_head",
    ]
    assert ("progress", 3, "Cloning source") in calls
    assert ("progress", 8, "Source checkout ready") in calls
    assert calls[kinds.index("clone")][1:] == (repo, "trunk", clone_dir)
    assert calls[kinds.index("git_head")][1] == clone_dir  # reuse mode: unsealed

    pips = [call for call in calls if call[0] == "pip"]
    bootstrap, editable = pips
    assert bootstrap[1] == ["install", "--upgrade", "pip"]
    assert bootstrap[2] == "install_source"
    assert bootstrap[3].get("append") is True
    assert "cwd" not in bootstrap[3] and "env" not in bootstrap[3]
    assert editable[1] == ["install", "-v", "-e", clone_dir]  # absolute path, not "."
    assert editable[2] == "install_source"
    assert editable[3]["cwd"] == clone_dir
    assert "append" not in editable[3]  # tri-state: legacy relied on the default
    assert editable[3]["env"]["CUDA_HOME"] == "/fake-cuda"
    assert any(key.startswith("CCACHE") for key in editable[3]["env"])

    assert not [call for call in calls if call[0] == "unique"]  # reuse keeps the name
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[1] == "v1"
    row = ready[2]
    assert row["version"] == "v1"
    assert row["source_commit"] == "abc123456789def"
    assert row["source_branch"] == "trunk"
    assert row["package_version"] == "0.6.3"
    # runtime_meta merged into the success row:
    assert row["cuda_version"] == "12.8"
    assert row["python_version"] == "3.12"
    assert "reuse_existing" not in row
    assert ("activate", "vllm", "published-name") in calls
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_source_workspace_mode_seals_and_rebinds_commit_capture(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.build_workspace as bw
    import backend.engines.python.generic as generic_module

    unique_calls: list = []

    def fake_unique(store, engine, base):
        unique_calls.append((engine, base))
        return "unique-name"

    monkeypatch.setattr(generic_module, "unique_version_name", fake_unique)

    checkout = tmp_path / "ws-checkout"
    checkout.mkdir()
    workspace = _FakeWorkspace(calls, str(checkout))

    class _Open:
        @staticmethod
        def open(engine_id, repo_url, meta):
            calls.append(("open", engine_id, repo_url, meta))
            return workspace

    monkeypatch.setattr(bw, "BuildWorkspace", _Open)
    monkeypatch.setattr(bw, "release_held", lambda ws: calls.append(("held", ws)))

    async def fake_git_head(path):
        calls.append(("git_head", path))
        return "abc123456789def"

    installer._git_head = fake_git_head

    repo = "https://example.invalid/vllm.git"
    await _drive(
        installer,
        task_holder,
        calls,
        lambda: installer.install_from_source(repo, "main"),
    )

    assert ("open", "vllm", repo, {"kind": "source"}) in calls
    assert ("seal", os.path.join(installer._base_dir, "source"), installer._venv_path) in calls
    assert ("held", workspace) in calls  # teardown stays manager-owned
    git_head = next(call for call in calls if call[0] == "git_head")
    assert git_head[1] == "/sealed-checkout"  # commit read from the SEALED copy
    pips = [call for call in calls if call[0] == "pip"]
    assert pips[1][1] == ["install", "-v", "-e", str(checkout)]
    assert pips[1][3]["cwd"] == str(checkout)
    # Fresh install: unique name from "{detected}-{commit8}".
    assert unique_calls == [("vllm", "0.6.3-abc12345")]
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[2]["version"] == "unique-name"


@pytest.mark.asyncio
async def test_source_pip_bootstrap_failure_keeps_runtime_meta_in_failure(
    tmp_path: Path, monkeypatch
):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    clone_dir = os.path.join(installer._base_dir, "source")

    _patch_git_clone(monkeypatch, calls)

    async def failing_bootstrap(args, stage, **kwargs):
        return 7

    installer._run_pip = failing_bootstrap

    await _drive(
        installer,
        task_holder,
        calls,
        lambda: installer.install_from_source(
            "https://example.invalid/vllm.git", "main", reuse_dir=str(tmp_path / "vllm")
        ),
    )

    assert installer._last_error == "pip upgrade failed with status 7"
    fail = next(call for call in calls if call[0] == "fail")
    assert fail[1] == "pip upgrade failed with status 7"
    # runtime_meta was captured BEFORE the failure and merges in, as legacy.
    assert fail[2]["cuda_version"] == "12.8"
    assert not [call for call in calls if call[0] == "ready"]


@pytest.mark.asyncio
async def test_source_editable_failure_and_missing_checkout_guard(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)

    _patch_git_clone(monkeypatch, calls, keep_dir=False)
    original_run_pip = installer._run_pip

    async def pip_then_fail(args, stage, **kwargs):
        calls.append(("pip", list(args), stage))
        if "-e" in args:
            return 9
        return await original_run_pip(args, stage, **kwargs)

    installer._run_pip = pip_then_fail

    await _drive(
        installer,
        task_holder,
        calls,
        lambda: installer.install_from_source(
            "https://example.invalid/vllm.git", "main", reuse_dir=str(tmp_path / "vllm")
        ),
    )

    # The assert_dir step sits between bootstrap and editable install.
    assert installer._last_error == "vLLM source package directory is unavailable"
    editable = [call for call in calls if call[0] == "pip" and "-e" in call[1]]
    assert editable == []  # guard stopped the plan before the editable install


@pytest.mark.asyncio
async def test_source_editable_failure_uses_legacy_string(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    clone_dir = os.path.join(installer._base_dir, "source")

    _patch_git_clone(monkeypatch, calls)
    original_run_pip = installer._run_pip

    async def pip_then_fail(args, stage, **kwargs):
        if "-e" in args:
            return 9
        return await original_run_pip(args, stage, **kwargs)

    installer._run_pip = pip_then_fail

    await _drive(
        installer,
        task_holder,
        calls,
        lambda: installer.install_from_source(
            "https://example.invalid/vllm.git", "main", reuse_dir=str(tmp_path / "vllm")
        ),
    )

    assert installer._last_error == "vLLM source install failed with status 9"


# --------------------------------------------------------------------------
# sync
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sync_republishes_same_version_without_uniquifying(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(
        generic_module, "unique_version_name", lambda *a: calls.append(("unique",)) or "u"
    )

    install_dir = tmp_path / "installs" / "v1"
    install_dir.mkdir(parents=True)
    row = {
        "version": "v1",
        "install_dir": str(install_dir),
        "source_branch": "main",
        "install_type": "source",
        "source_repo": "https://example.invalid/vllm.git",
    }

    async def fake_sync(target, branch):
        calls.append(("sync_git", target, branch))
        os.makedirs(target, exist_ok=True)  # a real sync leaves the checkout present

    installer._sync_git_checkout = fake_sync

    async def fake_git_head(path):
        return "deadbeefcafe"

    installer._git_head = fake_git_head

    await _drive(installer, task_holder, calls, lambda: installer.sync_source_version(row))

    clone_dir = os.path.join(str(install_dir), "source")  # base rebound like legacy
    assert ("sync_git", clone_dir, "main") in calls
    assert not [call for call in calls if call[0] == "unique"]
    ready = next(call for call in calls if call[0] == "ready")
    assert ready[1] == "v1"
    row_out = ready[2]
    assert row_out["version"] == "v1"
    assert row_out["source_commit"] == "deadbeefcafe"
    assert row_out["source_repo"] == "https://example.invalid/vllm.git"  # row passthrough
    assert row_out["cuda_version"] == "12.8"  # runtime_meta merged
    assert "reuse_existing" not in row_out
    assert ("activate", "vllm", "published-name") in calls
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_sync_failure_merges_row_and_runtime_meta(tmp_path: Path):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    install_dir = tmp_path / "installs" / "v1"
    install_dir.mkdir(parents=True)
    row = {
        "version": "v1",
        "install_dir": str(install_dir),
        "source_branch": "main",
        "install_type": "source",
    }

    async def failing_sync(target, branch):
        raise RuntimeError("git sync exploded")

    installer._sync_git_checkout = failing_sync

    await _drive(installer, task_holder, calls, lambda: installer.sync_source_version(row))

    assert installer._last_error == "git sync exploded"
    fail = next(call for call in calls if call[0] == "fail")
    assert fail[1] == "git sync exploded"
    assert fail[2]["install_type"] == "source"  # row merged
    assert fail[2]["cuda_version"] == "12.8"  # runtime_meta merged (captured pre-failure)


# --------------------------------------------------------------------------
# remove
# --------------------------------------------------------------------------


def _patch_store(monkeypatch, active, delete_calls, delete_error=None):
    import backend.data_store

    store_cls = type(backend.data_store.get_store())

    def fake_active(self, engine_id):
        return active

    def fake_delete(self, engine_id, version):
        delete_calls.append((engine_id, version))
        if delete_error is not None:
            raise RuntimeError(delete_error)

    monkeypatch.setattr(store_cls, "get_active_engine_version", fake_active)
    # The post-remove stale-version sweep is engine-wide; this harness pins the
    # legacy version-scoped recipe, so keep the global store out of it.
    monkeypatch.setattr(store_cls, "get_engine_versions", lambda self, engine: [])
    monkeypatch.setattr(store_cls, "delete_engine_version", fake_delete)


@pytest.mark.asyncio
async def test_remove_deletes_guarded_tree_and_forgets_row(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.utils.fs_ops as fs_ops

    install_dir = os.path.join(os.path.realpath(installer._root_dir), "v1")
    os.makedirs(install_dir)
    deletes: list = []
    _patch_store(
        monkeypatch,
        {"version": "v1", "install_dir": os.path.realpath(install_dir)},
        deletes,
    )
    monkeypatch.setattr(
        fs_ops,
        "robust_rmtree",
        lambda path, retire_references=False: calls.append(("rmtree", path, retire_references)),
    )
    monkeypatch.setattr(
        fs_ops,
        "release_launch_hold",
        lambda path, retire_references=False: calls.append(("hold", path, retire_references)),
    )
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "mark_runtime_changed", lambda: calls.append(("mark",)))

    await _drive(installer, task_holder, calls, lambda: installer.remove(retire_references=True))

    assert ("hold", os.path.realpath(install_dir), True) in calls  # pre-op, manager-owned
    assert ("rmtree", os.path.realpath(install_dir), True) in calls
    assert deletes == [("vllm", "v1")]
    assert ("mark",) in calls
    assert installer._last_error is None


@pytest.mark.asyncio
async def test_remove_skips_trees_outside_engine_root_but_still_forgets(
    tmp_path: Path, monkeypatch
):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    import backend.utils.fs_ops as fs_ops

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    deletes: list = []
    _patch_store(
        monkeypatch,
        {"version": "v1", "install_dir": os.path.realpath(str(outside))},
        deletes,
    )
    monkeypatch.setattr(
        fs_ops,
        "robust_rmtree",
        lambda path, retire_references=False: calls.append(("rmtree", path)),
    )
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "mark_runtime_changed", lambda: calls.append(("mark",)))

    await _drive(installer, task_holder, calls, lambda: installer.remove())

    assert not [call for call in calls if call[0] == "rmtree"]  # parent guard
    assert deletes == [("vllm", "v1")]
    assert ("mark",) in calls


@pytest.mark.asyncio
async def test_remove_delete_failure_is_not_swallowed(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    deletes: list = []
    _patch_store(
        monkeypatch,
        {"version": "v1", "install_dir": ""},
        deletes,
        delete_error="delete boom",
    )
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "mark_runtime_changed", lambda: calls.append(("mark",)))

    await _drive(installer, task_holder, calls, lambda: installer.remove())

    # LMDeploy swallows delete failures; vLLM does NOT — the operation fails.
    assert installer._last_error == "delete boom"
    assert not [call for call in calls if call[0] == "mark"]


@pytest.mark.asyncio
async def test_remove_without_active_row_still_marks_runtime(tmp_path: Path, monkeypatch):
    calls: list = []
    installer, task_holder = _make_installer(tmp_path, calls)
    deletes: list = []
    _patch_store(monkeypatch, None, deletes)
    import backend.engines.python.generic as generic_module

    monkeypatch.setattr(generic_module, "mark_runtime_changed", lambda: calls.append(("mark",)))

    await _drive(installer, task_holder, calls, lambda: installer.remove())

    assert deletes == []
    assert ("mark",) in calls
    assert installer._last_error is None
