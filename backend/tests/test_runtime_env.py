"""Runtime environment helpers for engine subprocesses."""

from pathlib import Path

import backend.runtime_env as runtime_env


def test_build_swap_process_env_includes_cuda_lib64(monkeypatch, tmp_path):
    cuda_lib = tmp_path / "cuda" / "lib64"
    cuda_lib.mkdir(parents=True)
    (cuda_lib / "libcublas.so.12").write_text("stub", encoding="utf-8")

    class FakeInstaller:
        def _get_cuda_path(self):
            return str(tmp_path / "cuda")

        def get_cuda_env(self):
            return {
                "CUDA_HOME": str(tmp_path / "cuda"),
                "CUDA_PATH": str(tmp_path / "cuda"),
                "LD_LIBRARY_PATH": f"{cuda_lib}:",
            }

    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer",
        lambda: FakeInstaller(),
    )

    env = runtime_env.build_swap_process_env(
        {"CUDA_VISIBLE_DEVICES": "0"},
        library_dirs=[str(tmp_path / "build" / "bin")],
        include_cuda=True,
    )

    assert env["CUDA_HOME"] == str(tmp_path / "cuda")
    assert str(cuda_lib) in env["LD_LIBRARY_PATH"]
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_audio_runtime_env_includes_cuda_for_cuda_build(tmp_path, monkeypatch):
    import backend.audio_cpp_runtime as audio_runtime

    cuda_lib = tmp_path / "cuda" / "lib64"
    cuda_lib.mkdir(parents=True)
    binary = tmp_path / "build" / "bin" / "audiocpp_server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")

    class FakeInstaller:
        def _get_cuda_path(self):
            return str(tmp_path / "cuda")

        def get_cuda_env(self):
            return {
                "CUDA_HOME": str(tmp_path / "cuda"),
                "CUDA_PATH": str(tmp_path / "cuda"),
                "LD_LIBRARY_PATH": f"{cuda_lib}:",
            }

    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer",
        lambda: FakeInstaller(),
    )
    monkeypatch.setattr(
        audio_runtime,
        "validate_audio_model_config",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    store = type(
        "Store",
        (),
        {
            "get_active_engine_version": staticmethod(
                lambda engine: {
                    "server_binary_path": str(binary),
                    "source_path": str(tmp_path),
                    "build_config": {"backend": "cuda"},
                }
            )
        },
    )()
    model = {
        "artifact": {"path": str(tmp_path / "model")},
        "config": {"engine": "audio_cpp"},
    }
    (tmp_path / "model").mkdir()
    config = {
        "family": "demo",
        "task": "tts",
        "mode": "offline",
        "backend": "cuda",
        "device": 0,
        "threads": 4,
    }

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )
    env_text = "\n".join(runtime["env"])
    assert "CUDA_HOME=" in env_text
    assert str(cuda_lib) in env_text
