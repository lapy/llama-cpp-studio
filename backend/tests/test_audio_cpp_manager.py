"""Native audio.cpp build planning, validation, and cancellation."""

import sys

import pytest

from backend.audio_cpp_manager import (
    AUDIO_CPP_DEFAULT_REF,
    AudioCppBuildConfig,
    AudioCppManager,
)
from backend.git_https import git_argv, is_network_git_command
from backend.task_cancel_registry import (
    TaskCancelledError,
    register_task_cancel,
    request_task_cancel,
    unregister_task_cancel,
)


def test_cmake_plan_selects_one_backend_and_both_runtime_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "backend.build_progress.shutil.which",
        lambda name: "/usr/bin/ninja" if name == "ninja" else None,
    )
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    config = AudioCppBuildConfig(
        backend="cuda",
        build_type="Release",
        native_cpu=False,
        openmp=True,
        cuda_graphs=True,
        jobs=4,
    )
    args = manager._cmake_args("/source", "/build", config)

    assert "-DENGINE_ENABLE_CUDA=ON" in args
    assert "-DENGINE_ENABLE_HIP=OFF" in args
    assert "-DENGINE_ENABLE_VULKAN=OFF" in args
    assert "-DENGINE_ENABLE_METAL=OFF" in args
    assert "-DENGINE_BUILD_TESTS=OFF" in args
    assert "-DENGINE_ENABLE_LLAMAFILE=ON" in args
    assert "-DAUDIOCPP_MODEL_SET=full" in args
    assert "-DAUDIOCPP_BUILD_NATIVE_MODEL_MANAGER=ON" in args
    assert args[-2:] == ["-G", "Ninja"]
    assert config.cuda is True
    assert config.backend == "cuda"


def test_build_config_normalizes_invalid_values(tmp_path):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    config = manager.build_config_from_dict(
        {"backend": "unknown", "build_type": "Fast", "jobs": -4, "extra": True}
    )
    assert config.backend == "cpu"
    assert config.build_type == "RelWithDebInfo"
    assert config.jobs == 0


def test_metal_is_rejected_on_unsupported_host(tmp_path, monkeypatch):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(ValueError, match="not supported on linux"):
        manager.validate_build_config(AudioCppBuildConfig(backend="metal"))


def test_unsupported_secondary_backend_is_also_rejected(tmp_path, monkeypatch):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    monkeypatch.setattr(sys, "platform", "linux")

    with pytest.raises(ValueError, match="backend 'metal'.*not supported on linux"):
        manager.validate_build_config(AudioCppBuildConfig(cuda=True, metal=True))


def test_cancelled_build_fails_before_spawning_process(tmp_path):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    task_id = "cancel-build"
    register_task_cancel(task_id)
    request_task_cancel(task_id)
    try:
        with pytest.raises(TaskCancelledError, match="cancelled"):
            manager._raise_if_cancelled(task_id)
    finally:
        unregister_task_cancel(task_id)


def test_default_tracking_ref_is_main():
    assert AUDIO_CPP_DEFAULT_REF == "main"


@pytest.mark.asyncio
async def test_sync_source_requires_existing_checkout(tmp_path):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))

    with pytest.raises(ValueError, match="metadata is incomplete"):
        await manager.sync_source(version_entry={}, branch="release-0.2")

    source_dir = tmp_path / "audio-cpp" / "builds" / "source-test" / "source"
    source_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="checkout not found"):
        await manager.sync_source(
            version_entry={"version": "source-test", "source_path": str(source_dir)},
            branch="release-0.2",
        )


@pytest.mark.asyncio
async def test_sync_and_clone_use_https_instead_of_ssh(tmp_path, monkeypatch):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    source_dir = tmp_path / "audio-cpp" / "builds" / "source-test" / "source"
    source_dir.mkdir(parents=True)
    (source_dir / ".git").mkdir()
    seen = []

    async def fake_run(argv, **kwargs):
        seen.append(list(argv))
        return []

    async def fake_emit(*_args, **_kwargs):
        return None

    monkeypatch.setattr(manager, "_run_streaming", fake_run)
    monkeypatch.setattr(manager, "_emit", fake_emit)

    await manager._sync_git_checkout(str(source_dir), "main", task_id=None, progress_manager=None)
    network = [argv for argv in seen if is_network_git_command(argv)]
    assert network
    assert all(argv[:5] == git_argv("unused")[:5] for argv in network)
    assert any(argv[-4:] == ["submodule", "update", "--init", "--recursive"] for argv in seen)


def test_local_checkout_is_allowed_for_sync(tmp_path):
    manager = AudioCppManager(str(tmp_path / "audio-cpp"))
    local_src = tmp_path / "audio-cpp" / "src"
    local_src.mkdir(parents=True)
    build_dir = local_src / "build" / "linux-cpu-release"
    build_dir.mkdir(parents=True)
    (build_dir / "CMakeCache.txt").write_text("cmake", encoding="utf-8")
    server = build_dir / "bin" / "audiocpp_server"
    server.parent.mkdir(parents=True)
    server.write_text("x", encoding="utf-8")

    version = {
        "version": "linux-cpu-local",
        "type": "local",
        "install_type": "local",
        "source_path": str(local_src),
        "server_binary_path": str(server),
    }
    assert manager._allowed_sync_source_dir(str(local_src), version) is True
    assert manager._resolve_sync_build_dir(version, str(local_src)) == str(build_dir)

    outsider = tmp_path / "other" / "src"
    outsider.mkdir(parents=True)
    assert manager._allowed_sync_source_dir(
        str(outsider),
        {**version, "source_path": str(outsider)},
    ) is False
