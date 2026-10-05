"""Source-build contract: configuration is delegated, and failed work is removed."""

import asyncio
from types import SimpleNamespace

import pytest

from backend.engines.llama_cpp.manager import BuildConfig, LlamaManager
from backend.engines.llama_cpp.source_build import (
    cleanup_partial_build,
    run_checked_command,
    run_source_build,
)
from backend.task_cancel_registry import TaskCancelledError


def test_manager_delegates_the_build_configuration(monkeypatch):
    seen = {}

    async def fake_run(host, **kwargs):
        seen["host"] = host
        seen["config"] = kwargs["build_config"]
        return "/tmp/llama-server"

    monkeypatch.setattr(
        "backend.engines.llama_cpp.source_build.run_source_build",
        fake_run,
    )
    config = BuildConfig()
    manager = LlamaManager()
    result = asyncio.run(
        manager.build_source("abc123", build_config=config, version_name="source-test")
    )
    assert result == "/tmp/llama-server"
    assert seen["host"] is manager
    assert seen["config"] is config


def test_cancelled_command_removes_a_partial_checkout(tmp_path):
    partial = tmp_path / "llama.cpp"
    partial.mkdir()
    (partial / "partial.o").write_text("object", encoding="utf-8")

    async def run_command(*_args, **_kwargs):
        raise AssertionError("cancelled work must not run the command")

    with pytest.raises(TaskCancelledError):
        asyncio.run(
            run_checked_command(
                run_command,
                "cmake",
                "--build",
                cancelled=lambda: True,
                checkout_path=partial,
            )
        )
    assert not partial.exists()


def test_failed_command_removes_a_partial_checkout(tmp_path):
    partial = tmp_path / "llama.cpp"
    partial.mkdir()
    (partial / "partial.o").write_text("object", encoding="utf-8")

    async def run_command(*_args, **_kwargs):
        return SimpleNamespace(returncode=1, stderr=b"cmake failed")

    with pytest.raises(RuntimeError, match="cmake failed"):
        asyncio.run(
            run_checked_command(
                run_command,
                "cmake",
                "--build",
                cancelled=lambda: False,
                checkout_path=partial,
            )
        )
    assert not partial.exists()


def test_cleanup_partial_build_leaves_a_missing_path_alone(tmp_path):
    missing = tmp_path / "absent"
    cleanup_partial_build(missing)
    assert not missing.exists()


def test_source_build_cleans_a_failed_clone_and_a_cancellation(tmp_path, monkeypatch):
    class Host:
        LLAMA_CPP_REPO = "https://example.test/llama.cpp.git"
        REPOSITORY_SOURCES = {}
        llama_dir = str(tmp_path / "llama-cpp")

        def __init__(self):
            self.calls = []

        async def _run_command_streaming(self, args, **kwargs):
            self.calls.append(args)
            checkout = args[-1]
            marker = __import__("pathlib").Path(checkout)
            marker.mkdir(parents=True, exist_ok=True)
            (marker / "partial.o").write_text("object", encoding="utf-8")
            if kwargs.get("cancel_event") is not None and self.fail == "cancel":
                kwargs["cancel_event"].set()
                return SimpleNamespace(returncode=0, lines=[])
            return SimpleNamespace(returncode=1, lines=["clone failed"])

    host = Host()
    host.fail = "clone"
    with pytest.raises(Exception, match="Git clone failed"):
        asyncio.run(
            run_source_build(
                host,
                "abc12345",
                version_name="partial-clone",
                repository_url=host.LLAMA_CPP_REPO,
            )
        )
    assert not (tmp_path / "llama-cpp" / "partial-clone" / "llama.cpp").exists()

    host.fail = "cancel"
    with pytest.raises(TaskCancelledError):
        asyncio.run(
            run_source_build(
                host,
                "abc12345",
                version_name="partial-cancel",
                repository_url=host.LLAMA_CPP_REPO,
                task_id="source-build-test",
            )
        )
    assert not (tmp_path / "llama-cpp" / "partial-cancel" / "llama.cpp").exists()


def _cpu_host(manager, calls):
    async def streaming(args, **kwargs):
        argv = list(args)
        calls.append(argv)
        if argv[:2] == ["git", "clone"]:
            os_path = __import__("pathlib").Path(argv[-1])
            os_path.mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, lines=["cloned"])
        if "--build" in argv and "llama-server" in argv:
            binary = __import__("pathlib").Path(kwargs["cwd"]) / "bin" / "llama-server"
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_text("#!/bin/sh\n", encoding="utf-8")
            binary.chmod(0o755)
            return SimpleNamespace(returncode=0, lines=["built target llama-server"])
        if argv and argv[0] == "cmake" and "--build" not in argv:
            return SimpleNamespace(returncode=0, lines=["Configuring done"])
        return SimpleNamespace(returncode=1, lines=["cmake failed"])

    async def run_command(*args, **_kwargs):
        calls.append(list(args))
        return SimpleNamespace(returncode=0, stdout=b"llama-server\n", stderr=b"")

    async def validate(*_args, **_kwargs):
        return True

    manager._run_command_streaming = streaming
    manager._run_command = run_command
    manager._find_cmake_executable = lambda: "cmake"
    manager.get_optimal_build_threads = lambda: 2
    manager.validate_build = validate


def test_default_config_cpu_build_uses_the_checked_command_runner(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    from backend.engines.build_workspace import BuildWorkspace

    def sync_git(self, _repo_url, ref):
        __import__("os").makedirs(self.checkout_dir, exist_ok=True)
        return ref

    monkeypatch.setattr(BuildWorkspace, "sync_git", sync_git)
    manager = LlamaManager()
    calls = []
    _cpu_host(manager, calls)
    binary = asyncio.run(
        manager.build_source("abc123", use_workspace=True, version_name="cpu-default")
    )
    assert "BuildConfig" not in str(binary)
    assert binary.endswith("llama-server")
    configure = next(argv for argv in calls if "-DCMAKE_BUILD_TYPE=" in " ".join(argv))
    rendered = " ".join(configure)
    assert "-DGGML_CUDA=OFF" in rendered
    assert "-DCMAKE_CUDA_COMPILER=" in rendered
    assert "-DGGML_CUDA=ON" not in rendered


def test_failed_cpu_configure_removes_the_partial_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    manager = LlamaManager()
    calls = []

    async def streaming(args, **_kwargs):
        argv = list(args)
        calls.append(argv)
        if argv[:2] == ["git", "clone"]:
            __import__("pathlib").Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            return SimpleNamespace(returncode=0, lines=["cloned"])
        return SimpleNamespace(returncode=1, lines=["cmake failed"])

    async def run_command(*args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    manager._run_command_streaming = streaming
    manager._run_command = run_command
    manager._find_cmake_executable = lambda: "cmake"
    config = BuildConfig(enable_cuda=False, custom_cmake_args="-DCPU_PROBE=1")
    with pytest.raises(Exception, match="CMake configuration failed") as caught:
        asyncio.run(
            manager.build_source(
                "abc123",
                build_config=config,
                version_name="cpu-failed",
            )
        )
    assert "name 'shlex' is not defined" not in str(caught.value)
    assert "name 'BuildConfig' is not defined" not in str(caught.value)
    configure = next(argv for argv in calls if "-DCPU_PROBE=1" in argv)
    assert "-DGGML_CUDA=OFF" in configure
    checkout = tmp_path / "data" / "llama-cpp" / "cpu-failed" / "llama.cpp"
    assert not checkout.exists()
