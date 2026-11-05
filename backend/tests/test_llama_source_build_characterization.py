"""Characterization pins for the llama.cpp ``source`` recipe on the native host.

These pins freeze the source build's OBSERVABLE contract â€” command argv
sequences, stage/progress ordering, error strings (including the wrapper),
cancellation identity, cleanup points, and the finally bookkeeping â€” against a
fake manager, so the Phase-7 cutover onto ``GenericNativeInstaller`` must be
observably identical to the legacy ``source_build`` module (now deleted).

Key pinned asymmetries (deliberate, not bugs to fix during conversion):
- cancellation raises the ORIGINAL TaskCancelledError unwrapped; every other
  failure is re-raised as ``Failed to build from source {commit}: {e}``.
- partial checkouts are removed only at specific checkpoints (pre-command
  cancel, clone failure, configure/build failure) â€” the post-build binary
  hunt, the stage-3 cancel check, and validate failures leave it on disk.
- the cancel registry is always unregistered in ``finally``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import backend.task_cancel_registry as cancel_registry
from backend.engines.llama_cpp.installer import LlamaNativeInstaller
from backend.engines.llama_cpp.manager import BuildConfig
from backend.engines.python.native import run_checked_command
from backend.task_cancel_registry import TaskCancelledError

TASK = "char-build-task"
COMMIT = "abc1234def5678"


class FakePm:
    def __init__(self):
        self.events: list[dict] = []

    async def send_build_progress(self, **kwargs):
        self.events.append(kwargs)

    def get_task(self, task_id):
        return {"metadata": {"log_lines": []}}


class FakeHost:
    LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"
    REPOSITORY_SOURCES = {"llama.cpp": LLAMA_CPP_REPO}

    def __init__(self, llama_dir):
        self.llama_dir = str(llama_dir)
        self.commands: list[tuple[str, list[str], str]] = []
        self.patches: list[tuple[str, str]] = []
        self.syncs: list[tuple] = []
        self.pm_flushes: list[bool] = []
        self.valid = True
        self.exit_plan: dict = {}  # predicate -> returncode
        self.lines_plan: dict = {}  # predicate -> output lines
        self.after: dict = {}  # predicate -> callable()
        self.cancel_streams = True

    # -- manager seams used by the source recipe --------------------------
    def _create_build_log_batcher(self, progress_manager, task_id):
        ctx = {
            "stage": "init",
            "progress": 0,
            "message": "",
            "base_message": "",
            "progress_floor": 0,
            "progress_ceil": 2,
        }

        async def emit_line(line):
            return None

        async def flush(*, complete_stage: bool = False):
            self.pm_flushes.append(complete_stage)

        return ctx, emit_line, flush

    def _match(self, table, argv):
        for predicate in table:
            if predicate(argv):
                return table[predicate]
        return None

    def _rc(self, argv):
        override = self._match(self.exit_plan, argv)
        return override if override is not None else 0

    def _run_hooks(self, argv):
        hook = self._match(self.after, argv)
        if hook is not None:
            hook()

    async def _run_command_streaming(
        self,
        args,
        *,
        cwd=None,
        env=None,
        cancel_event=None,
        on_line=None,
        merge_stderr=True,
        timeout=None,
    ):
        argv = [str(a) for a in args]
        self.commands.append(("stream", argv, cwd))
        self._run_hooks(argv)
        if cancel_event is not None and self.cancel_streams and cancel_event.is_set():
            raise TaskCancelledError("Build cancelled by user")
        return SimpleNamespace(returncode=self._rc(argv), lines=self._lines(argv))

    async def _run_command(self, *args, cwd=None, env=None, timeout=None, **kwargs):
        argv = [str(a) for a in args]
        self.commands.append(("plain", argv, cwd))
        self._run_hooks(argv)
        stdout = b"llama-server ... (target list)\n"
        return SimpleNamespace(returncode=self._rc(argv), stdout=stdout, stderr=b"")

    def _lines(self, argv):
        override = self._match(self.lines_plan, argv)
        if override is not None:
            return list(override)
        if self._rc(argv):
            return [f"{argv[1]} failed"]
        return [f"{argv[1]} ok", "llama-server built target"]

    async def _apply_patch(self, repo_dir, patch_url):
        self.patches.append((repo_dir, patch_url))

    async def _sync_existing_checkout(self, clone_dir, ref, **kwargs):
        self.syncs.append((clone_dir, ref))

    def _check_cuda_toolkit_available(self, *args, **kwargs):
        return True, "/usr/local/cuda", ""

    def _verify_cuda_toolkit_complete(self, cuda_root):
        return True, []

    def _find_cmake_executable(self):
        return "cmake"

    def _get_cmake_version(self):
        return (3, 28, 0)

    def _get_cuda_version(self, nvcc_path):
        return (12, 4)

    async def _detect_cuda_architectures(self):
        return None

    def get_optimal_build_threads(self):
        return 4

    async def validate_build(self, binary_path, progress_manager=None, task_id=None):
        return self.valid


def _cpu_config() -> BuildConfig:
    cfg = BuildConfig()
    cfg.enable_cuda = False
    cfg.enable_ccache = False
    return cfg


@pytest.fixture
def host(tmp_path, monkeypatch):
    fake = FakeHost(tmp_path / "llama-dir")
    # pkg-config libcurl probe: report libcurl present so LLAMA_CURL is untouched.
    monkeypatch.setattr(
        "backend.engines.python.cmake_source.subprocess.run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    yield fake
    assert TASK not in cancel_registry._events, "finally must unregister the task"


def _make_build_side_effect(host):
    """Create <clone>/build/bin/llama-server when llama-server is compiled."""

    def effect():
        build_dir = host.version_build_dir
        bin_dir = build_dir / "bin"
        bin_dir.mkdir(parents=True, exist_ok=True)
        (bin_dir / "llama-server").write_bytes(b"#!/bin/sh\n")
        (bin_dir / "llama-server").chmod(0o644)

    def predicate(argv):
        return "--target" in argv and "llama-server" in argv

    host.after[predicate] = effect
    return predicate


async def _run(host, pm, commit_sha=COMMIT, **overrides):
    kwargs = dict(
        build_config=_cpu_config(),
        progress_manager=pm,
        task_id=TASK,
        version_name="v-test",
    )
    kwargs.update(overrides)
    installer = LlamaNativeInstaller("llama_cpp", manager=host)
    return await installer.run_source(commit_sha, **kwargs)


@pytest.mark.asyncio
async def test_happy_cpu_path_stage_sequence_commands_and_publish(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    pm = FakePm()

    result = await _run(host, pm)

    kinds = [(kind, argv[:2]) for kind, argv, _cwd in host.commands]
    assert kinds == [
        ("stream", ["git", "clone"]),
        ("plain", ["git", "checkout"]),
        ("stream", ["cmake", ".."]),
        ("plain", ["cmake", "--build"]),  # target 'help' probe (non-critical)
        ("stream", ["cmake", "--build"]),
    ]
    configure = host.commands[2][1]
    assert configure[0] == "cmake"
    assert f"-DCMAKE_BUILD_TYPE={_cpu_config().build_type}" in configure
    assert "-DGGML_CUDA=OFF" in configure
    assert "-DCMAKE_CUDA_COMPILER=" in configure  # explicit CUDA-off guard
    build = host.commands[4][1]
    assert build[build.index("--target") :][:4] == ["--target", "llama-server", "--parallel", "4"]
    assert host.pm_flushes[0] is True  # clone stage flushes as complete
    assert [e["stage"] for e in pm.events] == [
        "init",
        "clone",
        "checkout",
        "configure",
        "build",
        "verify",
        "validate",
        "complete",
    ]
    assert pm.events[0]["progress"] == 0 and pm.events[-1]["progress"] == 100
    assert pm.events[-1]["log_lines"] == [
        f"llama-server ready at: {result}",
        "Validation: Passed",
    ]
    assert result == str(tmp_path / "llama-dir" / "v-test" / "llama-server")
    assert (tmp_path / "llama-dir" / "v-test" / "llama-server").read_bytes() == b"#!/bin/sh\n"


@pytest.mark.asyncio
async def test_cancellation_mid_checkout_propagates_unwrapped(host):
    pm = FakePm()

    def cancel_during_checkout(argv):
        return argv[:2] == ["git", "checkout"]

    host.after[cancel_during_checkout] = lambda: cancel_registry.request_task_cancel(TASK)

    with pytest.raises(TaskCancelledError) as exc:
        await _run(host, pm)
    assert str(exc.value) == "Build cancelled by user"
    # The stage-3 checkpoint raises WITHOUT partial-checkout cleanup (pinned).
    assert [e["stage"] for e in pm.events][-1] == "checkout"
    assert TASK not in cancel_registry._events


@pytest.mark.asyncio
async def test_clone_failure_cleans_checkout_and_wraps_error(host, tmp_path):
    pm = FakePm()
    clone_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp"

    def clone_hook():
        (clone_dir / ".git").mkdir(parents=True)

    def is_clone(argv):
        return argv[:2] == ["git", "clone"]

    host.exit_plan[is_clone] = 128
    host.after[is_clone] = clone_hook

    with pytest.raises(Exception) as exc:
        await _run(host, pm)
    # lines[-40:] tail -> "clone failed" from the fake streaming runner.
    assert str(exc.value) == (
        f"Failed to build from source {COMMIT}: Git clone failed: clone failed"
    )
    assert not clone_dir.exists()
    assert [e["stage"] for e in pm.events][-1] == "error"
    assert pm.events[-1]["message"] == "Build failed: Git clone failed: clone failed"


@pytest.mark.asyncio
async def test_configure_failure_cleans_checkout_and_wraps_error(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    pm = FakePm()

    def is_configure(argv):
        return argv[:2] == ["cmake", ".."]

    host.exit_plan[is_configure] = 1

    with pytest.raises(Exception) as exc:
        await _run(host, pm)
    assert str(exc.value) == (
        f"Failed to build from source {COMMIT}: CMake configuration failed: .. failed"
    )
    assert not (tmp_path / "llama-dir" / "v-test" / "llama.cpp").exists()


@pytest.mark.asyncio
async def test_build_failure_reports_build_stage_then_wraps(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    pm = FakePm()

    def predicate(argv):
        return "--target" in argv and "llama-server" in argv

    host.exit_plan[predicate] = 2

    with pytest.raises(Exception) as exc:
        await _run(host, pm)
    assert str(exc.value) == (
        f"Failed to build from source {COMMIT}: Build failed (exit code 2). Check logs for details."
    )
    # NB: cmake_stage_start("build") is ALSO 28, so the stage-start event and
    # the error event share a number (pinned); select by message.
    build_errors = [e for e in pm.events if e["message"] == "Build failed (exit code 2)"]
    assert len(build_errors) == 1 and build_errors[0]["stage"] == "build"
    assert build_errors[0]["progress"] == 28
    assert not (tmp_path / "llama-dir" / "v-test" / "llama.cpp").exists()


@pytest.mark.asyncio
async def test_zero_returncode_with_error_lines_fails_the_build(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    pm = FakePm()

    def is_server_build(argv):
        return "--target" in argv and "llama-server" in argv

    host.lines_plan[is_server_build] = ["error: undefined reference to cudaLaunch"]

    with pytest.raises(Exception) as exc:
        await _run(host, pm)
    assert "Build completed but contains errors" in str(exc.value)
    assert str(exc.value).startswith(f"Failed to build from source {COMMIT}: ")


@pytest.mark.asyncio
async def test_missing_binary_keeps_checkout_and_raises_detailed_error(host, tmp_path):
    build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    (build_dir / "sub").mkdir(parents=True)
    (build_dir / "sub" / "README").write_text("not executable", encoding="utf-8")
    pm = FakePm()

    with pytest.raises(Exception) as exc:
        await _run(host, pm)
    message = str(exc.value)
    assert message.startswith(f"Failed to build from source {COMMIT}: ")
    assert "llama-server executable not found after build in" in message
    assert "No executables found" in message
    # Pinned: this failure path does NOT clean the checkout.
    assert build_dir.exists()
    assert [e["stage"] for e in pm.events][-1] == "error"


@pytest.mark.asyncio
async def test_reuse_existing_checkout_syncs_and_skips_clone(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-sync" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    pm = FakePm()

    result = await _run(
        host,
        pm,
        reuse_existing_checkout=True,
        source_branch="master",
        version_name="v-sync",
    )

    clone_dir = str(tmp_path / "llama-dir" / "v-sync" / "llama.cpp")
    assert host.syncs == [(clone_dir, "master")]
    assert not [argv for _kind, argv, _cwd in host.commands if argv[:2] == ["git", "clone"]]
    assert not [argv for _kind, argv, _cwd in host.commands if argv[:2] == ["git", "checkout"]]
    assert [e["stage"] for e in pm.events][:2] == ["init", "configure"]
    assert result.endswith("llama-server")


@pytest.mark.asyncio
async def test_master_checkout_falls_back_to_main(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    pm = FakePm()

    def is_master(argv):
        return argv[:3] == ["git", "checkout", "master"]

    host.exit_plan[is_master] = 1

    await _run(host, pm, commit_sha="master")

    checkouts = [argv for _kind, argv, _cwd in host.commands if argv[:2] == ["git", "checkout"]]
    assert [c[2] for c in checkouts] == ["master", "main"]


@pytest.mark.asyncio
async def test_patches_apply_in_order_with_progress(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    pm = FakePm()

    await _run(host, pm, patches=["https://p/one.patch", "https://p/two.patch"])

    clone_dir = str(tmp_path / "llama-dir" / "v-test" / "llama.cpp")
    assert host.patches == [(clone_dir, "https://p/one.patch"), (clone_dir, "https://p/two.patch")]
    assert [e["stage"] for e in pm.events][:5] == [
        "init",
        "clone",
        "checkout",
        "patch",
        "configure",
    ]


@pytest.mark.asyncio
async def test_failed_validation_still_completes_the_build(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    host.valid = False
    pm = FakePm()

    result = await _run(host, pm)

    assert pm.events[-1]["stage"] == "complete"
    assert pm.events[-1]["log_lines"][1] == "Validation: Failed"
    assert result.endswith("llama-server")


@pytest.mark.asyncio
async def test_run_checked_command_pre_cancel_removes_checkout(tmp_path):
    checkout = tmp_path / "partial"
    checkout.mkdir()

    async def never_run(*args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("command must not execute when cancelled")

    with pytest.raises(TaskCancelledError):
        await run_checked_command(
            never_run,
            cancelled=lambda: True,
            checkout_path=checkout,
        )
    assert not checkout.exists()


@pytest.mark.asyncio
async def test_run_checked_command_failure_cleans_and_raises_stderr(tmp_path):
    checkout = tmp_path / "partial"
    (checkout / ".git").mkdir(parents=True)

    async def failing(*args, **kwargs):
        return SimpleNamespace(returncode=3, stderr=b"boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_checked_command(
            failing,
            cancelled=lambda: False,
            checkout_path=checkout,
        )
    assert not checkout.exists()


class RecordingStore:
    """Minimal engine-version store: the publish step's upsert only."""

    def __init__(self) -> None:
        self.rows: dict = {}

    def update_engine_version(self, engine, version, updates):
        row = self.rows.get((engine, version))
        if row is None:
            return None
        row.update(updates)
        return row

    def add_engine_version(self, engine, row):
        self.rows[(engine, row["version"])] = dict(row)


@pytest.mark.asyncio
async def test_publish_finalize_writes_the_install_row(host, tmp_path):
    """The recipe's publish step owns the row the route used to write."""
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    store = RecordingStore()
    pm = FakePm()

    await _run(
        host,
        pm,
        store=store,
        publish_row=True,
        repository_url="https://github.com/ggerganov/llama.cpp.git",
        repository_source="llama.cpp",
        source_ref_type="branch",
    )

    row = store.rows[("llama_cpp", "v-test")]
    assert row["type"] == "source"
    assert row["install_type"] == "source"
    assert row["is_fork"] is False
    assert row["source_ref"] == COMMIT
    assert row["source_ref_type"] == "branch"
    assert row["source_branch"] == COMMIT
    assert row["source_commit"] == COMMIT
    assert row["repository_source"] == "llama.cpp"
    assert row["install_dir"] == str(tmp_path / "llama-dir" / "v-test")
    assert row["binary_path"].endswith("llama-server")
    assert row["build_config"]["repository_source"] == "llama.cpp"
    assert row["installed_at"]


@pytest.mark.asyncio
async def test_publish_finalize_labels_a_fork_repository(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    store = RecordingStore()

    await _run(
        host,
        FakePm(),
        store=store,
        publish_row=True,
        repository_url="https://github.com/alice/llama.cpp.git",
        repository_source="llama.cpp",
        source_ref_type="ref",
    )

    row = store.rows[("llama_cpp", "v-test")]
    assert row["type"] == "fork"
    assert row["is_fork"] is True
    assert row["source_branch"] is None
    assert row["source_ref_type"] == "ref"


@pytest.mark.asyncio
async def test_publish_finalize_upstream_stays_source(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    store = RecordingStore()

    await _run(
        host,
        FakePm(),
        store=store,
        publish_row=True,
        repository_url="https://github.com/ggerganov/llama.cpp.git",
        repository_source="llama.cpp",
        source_ref_type="branch",
    )

    row = store.rows[("llama_cpp", "v-test")]
    assert row["type"] == "source"
    assert row["is_fork"] is False


@pytest.mark.asyncio
async def test_publish_row_is_off_unless_the_caller_asks(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    store = RecordingStore()

    await _run(host, FakePm(), store=store)

    assert store.rows == {}


@pytest.mark.asyncio
async def test_publish_finalize_updates_an_existing_row_for_sync(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)
    store = RecordingStore()
    store.rows[("llama_cpp", "v-test")] = {
        "version": "v-test",
        "install_dir": "/old/install",
        "installed_at": "2026-01-01T00:00:00Z",
    }

    await _run(
        host,
        FakePm(),
        store=store,
        publish_row="update",
        repository_url="https://github.com/ggerganov/llama.cpp.git",
        repository_source="llama.cpp",
        source_branch="main",
        source_ref_type="branch",
    )

    row = store.rows[("llama_cpp", "v-test")]
    assert row["type"] == "source"
    assert row["source_ref"] == "main"
    assert row["source_branch"] == "main"
    assert row["source_ref_type"] == "branch"
    assert row["source_commit"] == COMMIT
    assert row["binary_path"].endswith("llama-server")
    assert row["updated_at"]
    # An update merges in place: the install provenance is untouched.
    assert row["install_dir"] == "/old/install"
    assert row["installed_at"] == "2026-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_publish_finalize_update_of_a_missing_row_raises(host, tmp_path):
    host.version_build_dir = tmp_path / "llama-dir" / "v-test" / "llama.cpp" / "build"
    _make_build_side_effect(host)

    with pytest.raises(Exception, match="disappeared during sync"):
        await _run(
            host,
            FakePm(),
            store=RecordingStore(),
            publish_row="update",
            source_branch="main",
        )


def test_build_target_resolution_reads_the_declared_upstream():
    """R93: repository identity comes from the spec, not an id map."""
    from fastapi import HTTPException

    from backend.engines.registry import ENGINE_REGISTRY, EngineSpec
    from backend.routes.llama_versions import _resolve_engine_build_target

    assert _resolve_engine_build_target("llama_cpp") == (
        "llama.cpp",
        "https://github.com/ggerganov/llama.cpp.git",
    )
    assert _resolve_engine_build_target("ik_llama") == (
        "ik_llama.cpp",
        "https://github.com/ikawrakow/ik_llama.cpp.git",
    )

    for rejected in ("audio_cpp", "unsloth_llama", "not-an-engine"):
        with pytest.raises(HTTPException) as caught:
            _resolve_engine_build_target(rejected)
        assert caught.value.status_code == 400

    fake = EngineSpec(
        id="fixture_native",
        label="fixture_native",
        repository_source="Fixture Native",
        install_kind="cmake",
        runtime_kind="llama_server",
        scanner_kind="llama_help",
        active_path_fields=("binary_path",),
        artifact_formats=frozenset({"gguf"}),
        package_kinds=frozenset({"builtin"}),
        tasks=frozenset({"text-generation"}),
        input_modalities=frozenset({"text"}),
        output_modalities=frozenset({"text"}),
        upstream={"repository_url": "https://github.com/acme/fixture.git"},
    )
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setitem(ENGINE_REGISTRY, "fixture_native", fake)
        assert _resolve_engine_build_target("fixture_native") == (
            "Fixture Native",
            "https://github.com/acme/fixture.git",
        )
    finally:
        monkeypatch.undo()


def test_source_repo_fallbacks_read_the_declared_upstream():
    """R96: the retry/sync fallbacks use the declared upstream, not a name map."""
    from backend.routes.llama_versions import _declared_source_repo, _spec_repository_source

    assert _spec_repository_source("llama_cpp") == "llama.cpp"
    assert _spec_repository_source("ik_llama") == "ik_llama.cpp"
    assert _spec_repository_source("not-an-engine") == ""

    assert _declared_source_repo("llama_cpp") == "https://github.com/ggerganov/llama.cpp.git"
    assert (
        _declared_source_repo("ik_llama", "ik_llama.cpp")
        == "https://github.com/ikawrakow/ik_llama.cpp.git"
    )
    # A recorded name that does not belong to the engine resolves to nothing.
    assert _declared_source_repo("llama_cpp", "ik_llama.cpp") == ""
    assert _declared_source_repo("not-an-engine") == ""
