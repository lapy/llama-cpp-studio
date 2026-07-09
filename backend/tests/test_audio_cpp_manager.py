"""Native audio.cpp build planning, validation, and cancellation."""

import sys

import pytest

from backend.audio_cpp_manager import AudioCppBuildConfig, AudioCppManager
from backend.task_cancel_registry import (
    TaskCancelledError,
    register_task_cancel,
    request_task_cancel,
    unregister_task_cancel,
)


def test_cmake_plan_selects_one_backend_and_both_runtime_targets(tmp_path):
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
    assert "-DENGINE_ENABLE_VULKAN=OFF" in args
    assert "-DENGINE_ENABLE_METAL=OFF" in args
    assert "-DENGINE_BUILD_TESTS=OFF" in args
    assert manager._binary_candidates("/build", "audiocpp_server")[0].endswith(
        "/build/bin/audiocpp_server"
    )


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

