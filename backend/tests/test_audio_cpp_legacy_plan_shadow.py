"""Behavior pins for the audio.cpp PREBUILT install after the Phase-7 cutover.

The prebuilt path now runs through the definition-driven
``GenericNativeInstaller`` (``AudioCppInstaller``); the route keeps the task,
admission, progress and terminal bookkeeping around it. These tests ARE the
contract: exact progress sequence, the SHA256SUMS-inside-archive verification,
the dual-binary locate with mode |= 0o111, the legacy archive-deletion point
(after the binaries are located, before the CLI companion fetch and the row
publish), the CLI companion download, the published snapshot, and every
error/cancel string.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import os
import tarfile

import pytest

from backend.engines.audio_cpp.prebuilt import PrebuiltPlan

ASSET = "audio-v0.9.1-bin-ubuntu-x64-cuda12.8.tar.gz"
URL = f"https://downloads.test/releases/{ASSET}"
CLI_ASSET = "audio-v0.9.1-bin-portable-x64.tar.gz"
CLI_URL = f"https://downloads.test/releases/{CLI_ASSET}"
TASK_ID = "install_audio_cpp_v0.9.1_1"


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tar_bytes(entries: dict[str, bytes], *, with_sums: bool = True) -> bytes:
    """Archive laid out like the upstream release: pack/<binaries> + SHA256SUMS."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        for name, data in entries.items():
            info = tarfile.TarInfo(f"pack/{name}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if with_sums:
            lines = [f"{_digest(data)}  {name}" for name, data in entries.items()]
            sums = ("\n".join(lines) + "\n").encode("utf-8")
            info = tarfile.TarInfo("pack/SHA256SUMS")
            info.size = len(sums)
            archive.addfile(info, io.BytesIO(sums))
    return buf.getvalue()


class FakePm:
    def __init__(self):
        self.calls: list[tuple] = []

    def update_task(self, task_id, **kwargs):
        self.calls.append(("progress", kwargs.get("progress"), kwargs.get("message")))

    def complete_task(self, task_id, message):
        self.calls.append(("complete", message))

    def fail_task(self, task_id, message):
        self.calls.append(("fail", message))

    async def send_notification(self, **kwargs):
        self.calls.append(("notify", kwargs["title"], kwargs["message"], kwargs["type"]))


class FakeManager:
    def __init__(self, builds_dir: str):
        self.builds_dir = builds_dir


def _plan(**overrides) -> PrebuiltPlan:
    base = dict(
        asset_name=ASSET,
        url=URL,
        backend="cuda",
        package_cuda="12.8",
        host_cuda="13.0",
        architectures=(75, 80, 86, 89, 90),
        portable=False,
        reason="host CUDA 13.0 is newer than package CUDA 12.8",
        cli_asset_name=CLI_ASSET,
        cli_url=CLI_URL,
    )
    base.update(overrides)
    return PrebuiltPlan(**base)


class FakeStore:
    """The host's store seam: only the published-snapshot write is exercised."""

    def __init__(self, calls: list) -> None:
        self.calls = calls

    def record_install_snapshot(self, engine, **kwargs):
        self.calls.append(("snapshot", engine, kwargs))
        return {"engine": engine, **kwargs}


def _patch_environment(
    monkeypatch, tmp_path, pm, *, archives: dict[str, bytes], patch_download=True
):
    """Wire routes AND the native host to fakes.

    The prebuilt plan now executes inside ``backend.engines.python.native``, so
    the progress manager, store, readiness hook and activation hook are patched
    at THAT module (the route keeps its own progress/complete calls). Downloads
    write prebuilt tar bytes, so digest and content always agree.
    """
    from backend.engines.python import native
    from backend.operations import cancellable
    from backend.routes import audio_cpp_versions as routes

    calls: list[tuple] = []
    install_root = str(tmp_path / "builds")
    monkeypatch.setattr(routes, "get_progress_manager", lambda: pm)
    # Shared primitives (archive/locate) report progress through the
    # cancellable base class, whose progress manager is the same singleton in
    # production and this fake in tests.
    monkeypatch.setattr(cancellable, "get_progress_manager", lambda: pm)
    monkeypatch.setattr(routes, "get_store", lambda: object())
    monkeypatch.setattr(routes, "get_audio_cpp_manager", lambda: FakeManager(install_root))
    monkeypatch.setattr(native, "get_store", lambda: FakeStore(calls))

    if patch_download:

        async def fake_stream(url, dest):
            calls.append(("download", url, dest))
            data = archives[url]
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "wb") as handle:
                handle.write(data)

        monkeypatch.setattr(
            "backend.engines.audio_cpp.prebuilt.stream_release_download", fake_stream
        )

    def fake_ready(store, engine, row):
        calls.append(("ready", engine, row))

    def fake_failed(store, engine, version, **kwargs):
        calls.append(("failed", engine, version, kwargs))

    monkeypatch.setattr(native, "mark_engine_version_ready", fake_ready)
    monkeypatch.setattr("backend.engines.lifecycle.mark_engine_version_failed", fake_failed)

    async def fake_activate(engine, version, *, covered_by=None, row=None):
        calls.append(("activate", version, covered_by))

    monkeypatch.setattr(native, "activate_engine_version", fake_activate)
    return calls, install_root


async def _run(routes_module, **overrides):
    kwargs = dict(
        task_id=TASK_ID,
        version_name="v0.9.1",
        source_ref="v0.9.1",
        repository_url="https://github.com/0xShug0/audio.cpp.git",
        plan=_plan(),
        auto_activate=True,
    )
    kwargs.update(overrides)
    await routes_module._prebuilt_task(**kwargs)


SERVER = b"server-elf\n"
CLI = b"cli-elf\n"


@pytest.mark.asyncio
async def test_prebuilt_installs_both_binaries_from_one_archive(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    main_tar = _tar_bytes({"audiocpp_server": SERVER, "audiocpp_cli": CLI})
    calls, install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={URL: main_tar})

    await _run(routes)

    install_dir = os.path.join(install_root, "v0.9.1")
    server = os.path.join(install_dir, "pack", "audiocpp_server")
    cli = os.path.join(install_dir, "pack", "audiocpp_cli")
    kinds = [call[0] for call in calls]
    assert kinds == ["download", "ready", "snapshot", "activate"]
    assert pm.calls == [
        ("progress", 15, f"Downloading {ASSET}"),
        ("complete", "Installed audio.cpp v0.9.1"),
        (
            "notify",
            "audio.cpp installed",
            "Installed audio.cpp v0.9.1 (host CUDA 13.0 is newer than package CUDA 12.8)",
            "success",
        ),
    ]
    ready = next(call for call in calls if call[0] == "ready")
    row = ready[2]
    assert row["version"] == "v0.9.1"
    assert row["source_ref"] == "v0.9.1"
    assert row["source_ref_type"] == "release"
    assert row["source_branch"] == "v0.9.1"
    assert row["source_repo"] == "https://github.com/0xShug0/audio.cpp.git"
    assert row["install_dir"] == install_dir
    assert row["server_binary_path"] == server
    assert row["cli_binary_path"] == cli
    assert row["cuda_version"] == "13.0"
    assert row["build_config"] == {
        "prebuilt": True,
        "release_tag": "v0.9.1",
        "backend": "cuda",
        "asset_name": ASSET,
        "cli_asset_name": CLI_ASSET,
        "package_cuda": "12.8",
        "host_cuda": "13.0",
        "selection_reason": "host CUDA 13.0 is newer than package CUDA 12.8",
        "architectures": [75, 80, 86, 89, 90],
    }
    assert row["server_binary_path"] and os.access(server, os.X_OK)
    assert os.access(cli, os.X_OK)
    snapshot = next(call for call in calls if call[0] == "snapshot")
    assert snapshot[1] == "audio_cpp"
    assert snapshot[2]["version"] == "v0.9.1"
    assert snapshot[2]["phase"] == "published"
    assert snapshot[2]["resolved_inputs"]["install_type"] == "release"
    assert snapshot[2]["resolved_inputs"]["install_dir"] == install_dir
    assert next(call for call in calls if call[0] == "activate")[2] == TASK_ID
    assert ("complete", "Installed audio.cpp v0.9.1") in pm.calls
    assert not os.path.exists(os.path.join(install_dir, ASSET))  # legacy archive cleanup


@pytest.mark.asyncio
async def test_cuda_archive_without_cli_pulls_the_cpu_companion(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    main_tar = _tar_bytes({"audiocpp_server": SERVER})
    # Legacy extract_archive also demands audiocpp_server from ANY archive it
    # unpacks, so the CPU companion ships both binaries (like upstream).
    companion_tar = _tar_bytes({"audiocpp_server": SERVER, "audiocpp_cli": CLI})
    calls, install_root = _patch_environment(
        monkeypatch, tmp_path, pm, archives={URL: main_tar, CLI_URL: companion_tar}
    )

    await _run(routes)

    install_dir = os.path.join(install_root, "v0.9.1")
    assert [call[0] for call in calls] == [
        "download",
        "download",
        "ready",
        "snapshot",
        "activate",
    ]
    assert [c for c in pm.calls if c[0] == "progress"] == [
        ("progress", 15, f"Downloading {ASSET}"),
        ("progress", 55, f"Downloading audiocpp_cli from {CLI_ASSET}"),
    ]
    assert ("complete", "Installed audio.cpp v0.9.1") in pm.calls
    row = next(call for call in calls if call[0] == "ready")[2]
    copied = os.path.join(install_dir, "audiocpp_cli")
    assert row["cli_binary_path"] == copied
    assert os.path.isfile(copied) and os.access(copied, os.X_OK)
    assert not os.path.isdir(os.path.join(install_dir, ".cli-archive"))  # companion dir removed


@pytest.mark.asyncio
async def test_missing_cli_and_cli_url_fails_with_legacy_error(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    main_tar = _tar_bytes({"audiocpp_server": SERVER})
    calls, install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={URL: main_tar})

    await _run(routes, plan=_plan(cli_url="", cli_asset_name=""))

    failed = next(call for call in calls if call[0] == "failed")
    assert failed[3]["error"] == "audiocpp_cli was not in the audio.cpp archive"
    assert ("fail", "audiocpp_cli was not in the audio.cpp archive") in pm.calls
    assert pm.calls[-1][0] == "notify"  # legacy order: fail_task, then notification
    assert not [call for call in calls if call[0] == "ready"]
    # The archive was already deleted at the legacy point (before the CLI fetch)
    assert not os.path.exists(os.path.join(install_root, "v0.9.1", ASSET))


@pytest.mark.asyncio
async def test_missing_server_fails_after_extraction_and_cleans_archive(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    broken = _tar_bytes({"README": b"hi"})
    calls, install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={URL: broken})

    await _run(routes)

    failed = next(call for call in calls if call[0] == "failed")
    assert failed[3]["error"] == "audiocpp_server was not found in the audio.cpp archive"
    assert not [call for call in calls if call[0] == "ready"]
    install_dir = os.path.join(install_root, "v0.9.1")
    assert os.path.isfile(os.path.join(install_dir, "pack", "README"))  # extracted first
    assert not os.path.exists(os.path.join(install_dir, ASSET))  # caller finally mirrors legacy


@pytest.mark.asyncio
async def test_bad_checksum_inside_archive_fails_install(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as archive:
        data = SERVER
        info = tarfile.TarInfo("pack/audiocpp_server")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
        sums = (f"{_digest(b'wrong')}  audiocpp_server\n").encode("utf-8")
        info = tarfile.TarInfo("pack/SHA256SUMS")
        info.size = len(sums)
        archive.addfile(info, io.BytesIO(sums))
    calls, install_root = _patch_environment(
        monkeypatch, tmp_path, pm, archives={URL: buf.getvalue()}
    )

    await _run(routes)

    failed = next(call for call in calls if call[0] == "failed")
    assert failed[3]["error"] == "Checksum mismatch for audiocpp_server"
    assert not [call for call in calls if call[0] == "ready"]


@pytest.mark.asyncio
async def test_non_http_url_uses_the_legacy_download_guard(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    calls, _install_root = _patch_environment(
        monkeypatch, tmp_path, pm, archives={}, patch_download=False
    )

    await _run(routes, plan=_plan(url="ftp://downloads.test/releases/audio.tar.gz"))

    failed = next(call for call in calls if call[0] == "failed")
    assert failed[3]["error"] == "Refusing download from a non-HTTP URL"


@pytest.mark.asyncio
async def test_cancellation_marks_cancelled_row_and_reraises(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    calls, _install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={})

    async def cancel_midway(url, dest):
        raise asyncio.CancelledError

    monkeypatch.setattr("backend.engines.audio_cpp.prebuilt.stream_release_download", cancel_midway)

    with pytest.raises(asyncio.CancelledError):
        await _run(routes)

    failed = next(call for call in calls if call[0] == "failed")
    assert failed[3]["error"] == "audio.cpp install cancelled"
    assert failed[3]["cancelled"] is True
    assert pm.calls[-1] == ("fail", "audio.cpp install cancelled")


@pytest.mark.asyncio
async def test_auto_activate_false_publishes_without_activating(monkeypatch, tmp_path):
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    main_tar = _tar_bytes({"audiocpp_server": SERVER, "audiocpp_cli": CLI})
    calls, _install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={URL: main_tar})

    await _run(routes, auto_activate=False)

    assert [call for call in calls if call[0] in ("ready", "activate")] == [
        next(call for call in calls if call[0] == "ready")
    ]


def test_prebuilt_recipe_shape_is_the_pinned_sequence():
    from backend.engines.definitions import (
        load_default_engine_definitions,
        materialize_plan,
    )

    definition = load_default_engine_definitions().get("audio_cpp")
    assert definition is not None
    steps = materialize_plan(definition, "prebuilt").steps
    assert [(step.id, step.primitive) for step in steps] == [
        ("download_archive", "archive.download"),
        ("ensure_dir", "fs.ensure_dir"),
        ("extract_archive", "archive.extract_tar"),
        ("verify_checksums", "archive.verify_checksums_file"),
        ("locate_server", "artifact.locate"),
        ("locate_cli", "artifact.locate"),
        ("remove_archive", "fs.remove_file"),
        ("materialize_cli", "artifact.materialize_cli"),
        ("publish_row", "publish.install"),
    ]


class _CompletedExecution:
    status = "completed"
    failed_step = None


@pytest.mark.asyncio
async def test_prebuilt_runs_the_definition_plan_through_the_native_host(monkeypatch, tmp_path):
    """The route executes audio's OWN recipe on ``AudioCppInstaller``.

    Pins the Phase-7 cutover structurally: definition plan step-for-step, the
    plan-filtered handler surface (only the two the recipe declares), the host
    instance being the audio subclass, the route's task id bound for progress,
    and the exact context vocabulary the definition consumes.
    """
    from backend.engines.audio_cpp.installer import AudioCppInstaller
    from backend.engines.definitions import (
        load_default_engine_definitions,
        materialize_plan,
    )
    from backend.engines.python import native
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    calls, install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={})
    captured: dict = {}

    def fake_make_engine_runner(installer, handlers):
        captured["installer"] = installer
        captured["handlers"] = sorted(handlers)
        captured["task_id"] = installer.progress_task_id
        return lambda *args, **kwargs: None

    async def fake_execute_plan(plan, *, run_primitive, context):
        captured["plan"] = [(step.id, step.primitive) for step in plan.steps]
        captured["context"] = sorted(context)
        captured["install_dir"] = captured["installer"].install_dir
        return _CompletedExecution()

    monkeypatch.setattr(native, "make_engine_runner", fake_make_engine_runner)
    monkeypatch.setattr(native, "execute_plan", fake_execute_plan)

    await _run(routes)

    definition = load_default_engine_definitions().get("audio_cpp")
    expected = materialize_plan(definition, "prebuilt")
    assert captured["plan"] == [(step.id, step.primitive) for step in expected.steps]
    assert isinstance(captured["installer"], AudioCppInstaller)
    assert captured["installer"].definition.id == "audio_cpp"
    assert captured["handlers"] == ["artifact.materialize_cli", "publish.install"]
    assert captured["task_id"] == TASK_ID
    assert captured["install_dir"] == os.path.join(install_root, "v0.9.1")
    assert captured["context"] == sorted(
        {
            "input.plan",
            "input.plan_url",
            "input.download_label",
            "input.archive_path",
            "input.install_dir",
            "input.install_type",
            "input.version_name",
            "input.source_ref",
            "input.source_ref_type",
            "input.source_repo",
            "input.repository_url",
            "input.auto_activate",
        }
    )
    assert ("complete", "Installed audio.cpp v0.9.1") in pm.calls

    # The row fields the definition declares travel through publish.install's
    # `<field>_from` convention (asserted end-to-end by the tests above).
    publish_step = expected.by_id("publish_row")
    assert publish_step.with_ == {
        "server_binary_path_from": "server",
        "cli_binary_path_from": "cli_final",
    }


@pytest.mark.asyncio
async def test_source_build_runs_the_definition_plan_through_the_native_host(monkeypatch, tmp_path):
    """Phase-7 source cutover: the route executes audio's OWN `source` recipe.

    Pins the definition plan step-for-step, the host's plan-filtered handler
    surface (workspace checkout, configure/build, workspace publication,
    binary --help gate, commit capture, publication) and the context vocabulary
    the recipe consumes — including the row labels the route owns.
    """
    from backend.engines.audio_cpp.installer import AudioCppInstaller
    from backend.engines.definitions import (
        load_default_engine_definitions,
        materialize_plan,
    )
    from backend.engines.python import native
    from backend.routes import audio_cpp_versions as routes

    pm = FakePm()
    _calls, install_root = _patch_environment(monkeypatch, tmp_path, pm, archives={})
    captured: dict = {}

    def fake_make_engine_runner(installer, handlers):
        captured["installer"] = installer
        captured["handlers"] = sorted(handlers)
        return lambda *args, **kwargs: None

    async def fake_execute_plan(plan, *, run_primitive, context):
        captured["plan"] = [(step.id, step.primitive) for step in plan.steps]
        captured["context"] = sorted(context)
        return _CompletedExecution()

    monkeypatch.setattr(native, "make_engine_runner", fake_make_engine_runner)
    monkeypatch.setattr(native, "execute_plan", fake_execute_plan)

    await routes._build_task(
        task_id=TASK_ID,
        version_name="source-v0.9.1",
        source_ref="v0.9.1",
        source_ref_type="release",
        repository_url="https://github.com/0xShug0/audio.cpp.git",
        build_config=routes.AudioCppBuildConfig(),
        auto_activate=False,
    )

    definition = load_default_engine_definitions().get("audio_cpp")
    expected = materialize_plan(definition, "source")
    assert captured["plan"] == [(step.id, step.primitive) for step in expected.steps]
    assert isinstance(captured["installer"], AudioCppInstaller)
    assert captured["installer"].definition.id == "audio_cpp"
    assert captured["handlers"] == [
        "cmake.build",
        "cmake.configure",
        "publish.install",
        "smoke.test",
        "vcs.seal_checkout",
        "vcs.workspace_checkout",
    ]
    assert captured["context"] == sorted(
        {
            "input.install_dir",
            "input.version_name",
            "input.install_type",
            "input.type_labels",
            "input.source_ref",
            "input.source_ref_type",
            "input.repository_url",
            "input.source_repo",
            "input.build_config",
            "input.replace_existing",
            "input.use_workspace",
            "input.auto_activate",
        }
    )
    assert ("complete", "Installed audio.cpp source-v0.9.1") in pm.calls
    # The row's installed-path fields come from the recipe, not the route.
    assert expected.by_id("publish_row").with_ == {
        "server_binary_path_from": "server",
        "cli_binary_path_from": "cli",
    }
