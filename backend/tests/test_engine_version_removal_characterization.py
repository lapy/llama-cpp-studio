"""Characterization pins for the cross-family version removal flow.

The DELETE surface at /api/llama-versions/{version_id} is ONE route running
ONE flow: the definition's pinned "remove_version" recipe against the
route's handler pack. The per-family behavior — the python engines' guarded
tree + row forget, audio.cpp's manager-tracked bundle, and the native
family's llama-root safety walk with its lone-binary fallback — now lives
in the recipes and the route's target resolution. Every observable below
matches the pre-cutover monolith route.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from backend.routes import llama_versions as lv
from backend.utils import fs_ops
from backend.utils.fs_ops import FilesystemRefusal

PYTHON_ENGINES = ["lmdeploy", "1cat_vllm", "sglang", "sglang_v100", "vllm"]


class FakeStore:
    def __init__(self):
        self.deleted: list[tuple[str, str]] = []

    def delete_engine_version(self, engine, version):
        self.deleted.append((engine, version))


class Recorder:
    def __init__(self, raises: list | None = None):
        self.calls: list[tuple[str, bool]] = []
        self._raises = list(raises or [])

    def __call__(self, path, *, retire_references=False):
        self.calls.append((str(path), bool(retire_references)))
        if self._raises:
            raise self._raises.pop(0)


def _patch(monkeypatch, *, install_dir="", raises=None):
    """Swap the fs/store seams. Every tree removal now runs through the
    shared primitives, which import robust_rmtree from fs_ops at call time,
    so fs_ops is the seam under test."""
    from backend.engines.python import native

    store = FakeStore()
    rmtree = Recorder(raises)
    marked: list[int] = []
    monkeypatch.setattr(lv, "resolve_install_dir", lambda engine, entry: install_dir)
    monkeypatch.setattr(fs_ops, "robust_rmtree", rmtree)
    monkeypatch.setattr(lv, "mark_runtime_changed", lambda: marked.append(1))
    # The host path (audio.cpp) marks through its own module binding.
    monkeypatch.setattr(native, "mark_runtime_changed", lambda: marked.append(1))
    return store, rmtree, marked


def _dir(tmp_path, *parts):
    path = tmp_path.joinpath(*parts)
    path.mkdir(parents=True)
    return str(path)


@pytest.mark.asyncio
async def test_python_engine_registered_deletes_install_dir_and_row(monkeypatch, tmp_path):
    install = _dir(tmp_path, "vllm", "v1")
    store, rmtree, marked = _patch(monkeypatch, install_dir=install)
    entry = {"version": "1.2.3", "binary_path": None, "venv_path": str(install) + "/venv"}

    result = await lv._remove_engine_version(store, entry, "vllm", "1.2.3", True)

    assert rmtree.calls == [(install, False)]
    assert store.deleted == [("vllm", "1.2.3")]
    assert marked == [1]
    assert result == {"message": "Deleted version 1.2.3"}


@pytest.mark.asyncio
async def test_python_engine_disk_only_skips_store_row_delete(monkeypatch, tmp_path):
    install = _dir(tmp_path, "vllm", "v9")
    store, rmtree, marked = _patch(monkeypatch, install_dir=install)

    result = await lv._remove_engine_version(store, {"version": "v9"}, "vllm", "v9", False)

    assert rmtree.calls == [(install, False)]
    assert store.deleted == []
    assert marked == [1]  # the log line + runtime mark ran regardless
    assert result == {"message": "Deleted version v9"}


@pytest.mark.asyncio
async def test_python_engine_all_five_share_the_branch(monkeypatch, tmp_path):
    for engine in PYTHON_ENGINES:
        install = _dir(tmp_path, engine, "v")
        store, rmtree, _marked = _patch(monkeypatch, install_dir=install)
        result = await lv._remove_engine_version(store, {"version": "v"}, engine, "v", True)
        assert rmtree.calls == [(install, False)], engine
        assert store.deleted == [(engine, "v")], engine
        assert result == {"message": "Deleted version v"}, engine


@pytest.mark.asyncio
async def test_python_engine_venv_fallback_removes_version_root(monkeypatch, tmp_path):
    # No install_dir: legacy derived the version root from venv_path's parent.
    version_root = tmp_path / "sglang" / "v1"
    (version_root / "venv").mkdir(parents=True)
    store, rmtree, _marked = _patch(monkeypatch, install_dir="")
    entry = {"version": "1", "venv_path": str(version_root / "venv")}

    await lv._remove_engine_version(store, entry, "sglang", "1", True)

    assert rmtree.calls == [(str(version_root), False)]


@pytest.mark.asyncio
async def test_python_engine_nothing_on_disk_still_forgets_row(monkeypatch):
    store, rmtree, marked = _patch(monkeypatch, install_dir="")

    result = await lv._remove_engine_version(store, {"version": "z"}, "lmdeploy", "z", True)

    assert rmtree.calls == []
    assert store.deleted == [("lmdeploy", "z")]
    assert marked == [1]
    assert result == {"message": "Deleted version z"}


@pytest.mark.asyncio
async def test_python_engine_missing_dir_skips_removal_but_forgets_row(monkeypatch, tmp_path):
    # Legacy parity: delete_tree -> robust_rmtree returns early for missing
    # paths; the row is still forgotten and the runtime marked.
    missing = str(tmp_path / "vllm" / "g")
    store, rmtree, marked = _patch(monkeypatch, install_dir=missing)

    result = await lv._remove_engine_version(store, {"version": "g"}, "vllm", "g", True)

    assert rmtree.calls == []
    assert store.deleted == [("vllm", "g")]
    assert marked == [1]
    assert result == {"message": "Deleted version g"}


@pytest.mark.asyncio
async def test_python_engine_refusal_wraps_as_version_deletion_rejected(monkeypatch, tmp_path):
    install = _dir(tmp_path, "vllm", "v1")
    refusal = FilesystemRefusal("held by a launch", code="LAUNCH_REFERENCE_RUNNING")
    store, _rmtree, _marked = _patch(monkeypatch, install_dir=install, raises=[refusal])

    with pytest.raises(lv.VersionDeletionRejected) as exc:
        await lv._remove_engine_version(store, {"version": "1"}, "vllm", "1", True)
    assert exc.value.detail == {
        "code": "LAUNCH_REFERENCE_RUNNING",
        "message": "held by a launch",
    }
    assert store.deleted == []  # the row survives a refused tree removal


@pytest.mark.asyncio
async def test_python_engine_unexpected_error_maps_to_500(monkeypatch, tmp_path):
    install = _dir(tmp_path, "vllm", "v1")
    store, _rmtree, _marked = _patch(
        monkeypatch, install_dir=install, raises=[OSError("disk on fire")]
    )

    with pytest.raises(lv.HTTPException) as exc:
        await lv._remove_engine_version(store, {"version": "1"}, "vllm", "1", True)
    assert exc.value.status_code == 500
    assert exc.value.detail == "Failed to delete version: disk on fire"


@pytest.mark.asyncio
async def test_retire_references_forwards_to_every_removal(monkeypatch, tmp_path):
    install = _dir(tmp_path, "vllm", "v1")
    store, rmtree, _marked = _patch(monkeypatch, install_dir=install)

    await lv._remove_engine_version(
        store, {"version": "1"}, "vllm", "1", True, retire_references=True
    )

    assert rmtree.calls == [(install, True)]


@pytest.mark.asyncio
async def test_audio_cpp_removes_dir_then_row(monkeypatch, tmp_path):
    install = _dir(tmp_path, "audio", "v1")
    store, rmtree, marked = _patch(monkeypatch, install_dir=install)
    import backend.engines.audio_cpp.manager as audio_manager_module

    class FakeAudioManager:
        builds_dir = str(tmp_path / "audio")

    monkeypatch.setattr(
        audio_manager_module,
        "get_audio_cpp_manager",
        lambda: FakeAudioManager(),
    )
    entry = {"version": "a1", "install_dir": install}

    result = await lv._remove_engine_version(store, entry, "audio_cpp", "a1", True)

    assert rmtree.calls == [(install, False)]
    assert store.deleted == [("audio_cpp", "a1")]
    assert marked == [1]
    assert result == {"message": "Deleted version a1"}


@pytest.mark.asyncio
async def test_audio_cpp_outside_builds_root_maps_to_400(monkeypatch, tmp_path):
    builds = tmp_path / "audio"
    install = _dir(tmp_path, "audio", "v1")  # the resolved delete target (inside)
    outside = _dir(tmp_path, "elsewhere", "v9")  # the row's recorded dir (outside)
    store, rmtree, _marked = _patch(monkeypatch, install_dir=install)
    import backend.engines.audio_cpp.manager as audio_manager_module

    class FakeAudioManager:
        builds_dir = str(builds)

    monkeypatch.setattr(audio_manager_module, "get_audio_cpp_manager", lambda: FakeAudioManager())

    with pytest.raises(lv.HTTPException) as exc:
        await lv._remove_engine_version(
            store, {"version": "a1", "install_dir": str(outside)}, "audio_cpp", "a1", True
        )
    assert exc.value.status_code == 400
    assert exc.value.detail == "Refusing to delete audio.cpp files outside the builds root"
    # The tree removal ran BEFORE the bundle refused (pinned ordering).
    assert rmtree.calls == [(install, False)]
    assert store.deleted == []


@pytest.mark.asyncio
async def test_audio_cpp_removal_runs_the_definition_plan_on_the_native_host(monkeypatch, tmp_path):
    """Phase 7: audio's removal executes its OWN recipe on the native host.

    Pins the cutover structurally — definition plan step-for-step, the
    plan-filtered handler surface (the host's bundle + forget steps), the
    `AudioCppInstaller` instance, and the context vocabulary (which now carries
    the version row so the bundle step resolves the recorded layout).
    """
    from backend.engines.audio_cpp.installer import AudioCppInstaller
    from backend.engines.definitions import (
        load_default_engine_definitions,
        materialize_plan,
    )
    from backend.engines.python import native

    install = _dir(tmp_path, "audio", "v1")
    store, _rmtree, _marked = _patch(monkeypatch, install_dir=install)
    captured: dict = {}

    def fake_make_engine_runner(installer, handlers):
        captured["installer"] = installer
        captured["handlers"] = sorted(handlers)
        return lambda *args, **kwargs: None

    async def fake_execute_plan(plan, *, run_primitive, context):
        captured["plan"] = [(step.id, step.primitive) for step in plan.steps]
        captured["context"] = sorted(context)
        return SimpleNamespace(status=lv.EXECUTION_COMPLETED, failed_step=None)

    monkeypatch.setattr(native, "make_engine_runner", fake_make_engine_runner)
    monkeypatch.setattr(native, "execute_plan", fake_execute_plan)

    result = await lv._remove_engine_version(
        store, {"version": "a1", "install_dir": install}, "audio_cpp", "a1", True
    )

    definition = load_default_engine_definitions().get("audio_cpp")
    expected = materialize_plan(definition, "remove_version")
    assert captured["plan"] == [(step.id, step.primitive) for step in expected.steps]
    assert isinstance(captured["installer"], AudioCppInstaller)
    assert captured["installer"].definition.id == "audio_cpp"
    assert captured["handlers"] == ["artifact.remove_bundle", "publish.forget"]
    assert captured["context"] == sorted(
        {
            "input.delete_target",
            "input.file_target",
            "input.retire_references",
            "input.registered",
            "input.version_name",
            "input.version_entry",
        }
    )
    assert result == {"message": "Deleted version a1"}


@pytest.mark.asyncio
async def test_native_engine_with_install_dir_deletes_it(monkeypatch, tmp_path):
    install = _dir(tmp_path, "llama", "v1")
    store, rmtree, marked = _patch(monkeypatch, install_dir=install)

    result = await lv._remove_engine_version(
        store,
        {"version": "b1", "binary_path": str(tmp_path / "llama/v1/bin/llama-server")},
        "llama_cpp",
        "b1",
        True,
    )

    assert rmtree.calls == [(install, False)]
    assert store.deleted == [("llama_cpp", "b1")]
    assert marked == [1]
    assert result == {"message": "Deleted version b1"}


@pytest.mark.asyncio
async def test_unsloth_uses_the_native_recipe_shape(monkeypatch, tmp_path):
    # Unsloth rows lived in the native branch; the consolidated flow must
    # keep deleting their install dirs and forgetting rows identically.
    install = _dir(tmp_path, "unsloth", "v1")
    store, rmtree, marked = _patch(monkeypatch, install_dir=install)

    result = await lv._remove_engine_version(store, {"version": "u1"}, "unsloth_llama", "u1", True)

    assert rmtree.calls == [(install, False)]
    assert store.deleted == [("unsloth_llama", "u1")]
    assert marked == [1]
    assert result == {"message": "Deleted version u1"}


@pytest.mark.asyncio
async def test_native_binary_walks_up_to_version_dir_never_the_root(monkeypatch, tmp_path):
    root = tmp_path / "llama-dir"
    binary = root / "v1" / "bin" / "llama-server"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"bin")
    store, rmtree, _marked = _patch(monkeypatch, install_dir="")
    monkeypatch.setattr(lv, "llama_manager", SimpleNamespace(llama_dir=str(root)))
    monkeypatch.setattr(lv, "_resolve_binary_path", lambda path: path)
    holds: list[str] = []
    monkeypatch.setattr(fs_ops, "release_launch_hold", lambda path, **kw: holds.append(str(path)))

    await lv._remove_engine_version(
        store, {"version": "b1", "binary_path": str(binary)}, "llama_cpp", "b1", True
    )

    assert rmtree.calls == [(os.path.realpath(str(root / "v1")), False)]
    assert holds == []


@pytest.mark.asyncio
async def test_native_binary_at_root_falls_back_to_single_file_removal(monkeypatch, tmp_path):
    root = tmp_path / "llama-dir"
    root.mkdir()
    binary = root / "llama-server"
    binary.write_bytes(b"bin")
    store, rmtree, _marked = _patch(monkeypatch, install_dir="")
    monkeypatch.setattr(lv, "llama_manager", SimpleNamespace(llama_dir=str(root)))
    monkeypatch.setattr(lv, "_resolve_binary_path", lambda path: path)
    holds: list[str] = []
    monkeypatch.setattr(fs_ops, "release_launch_hold", lambda path, **kw: holds.append(str(path)))

    await lv._remove_engine_version(
        store, {"version": "b0", "binary_path": str(binary)}, "llama_cpp", "b0", True
    )

    # The version-dir candidate would equal the root -> refused; only the
    # binary file itself is released+removed. The root survives.
    assert rmtree.calls == []
    assert holds == [os.path.realpath(str(binary))]
    assert not binary.exists()
    assert store.deleted == [("llama_cpp", "b0")]
