"""Phase-7 native build primitives: the shared runner + cmake/smoke steps.

The plan's "one process runner for logs, cancellation, process-group
termination and progress events" now exists (``backend/engines/native_build.py``)
and the native host exposes the build vocabulary on top of it. Pinned here with
a synthetic ``nativecraft`` definition and REAL short-lived subprocesses (the
interpreter itself as the fake toolchain) — no engine-id literals.

Audio.cpp's own ``_run_streaming`` delegates to the same runner, so these pins
also protect the production source-build path; the audio manager's adapter is
pinned separately in ``test_audio_cpp_manager.py``.
"""

from __future__ import annotations

import os
import sys

import pytest

from backend.data_store import DataStore
from backend.engines.definitions.schema import validate_definition
from backend.engines.native_build import run_streaming_command
from backend.engines.python import native as native_mod
from backend.engines.python.native import GenericNativeInstaller

NATIVECRAFT_ID = "nativecraft"


def _script(body: str) -> list:
    return [sys.executable, "-c", body]


_RECIPES = {
    "source": [
        {
            "id": "configure",
            "uses": "cmake.configure",
            "requires": ["input.install_dir", "input.cmake_args", "input.build_env"],
            "with": {
                "argv_from": "input.cmake_args",
                "cwd_from": "input.install_dir",
                "env_from": "input.build_env",
                "progress": 30,
                "progress_message": "Configuring nativecraft",
            },
        },
        {
            "id": "build",
            "uses": "cmake.build",
            "requires": ["input.install_dir", "input.build_args"],
            "with": {
                "argv_from": "input.build_args",
                "cwd_from": "input.install_dir",
                "progress": 60,
                "progress_message": "Building nativecraft",
            },
        },
        {
            "id": "smoke",
            "uses": "smoke.test",
            "requires": ["input.smoke_argv"],
            "with": {
                "argv_from": "input.smoke_argv",
                "require_output": True,
                "error": "nativecraft_server returned no help",
            },
        },
        {
            "id": "publish_row",
            "uses": "publish.install",
            "requires": ["input.install_dir", "input.version_name"],
        },
    ]
}


def _definition(*, recipes: dict | None = None):
    raw = {
        "schema_version": 2,
        "id": NATIVECRAFT_ID,
        "label": "Nativecraft",
        "repository_source": "Example Org/Nativecraft",
        "install_kind": "cmake",
        "runtime_kind": "llama_server",
        "scanner_kind": "llama_help",
        "active_path_fields": ["server_binary_path"],
        "artifact_formats": ["builtin"],
        "package_kinds": ["builtin"],
        "tasks": ["tts"],
        "input_modalities": ["text"],
        "output_modalities": ["audio"],
        "lifecycle": {
            "binary_name": "nativecraft_server",
            "root_name": "nativecraft",
            "log_name": "nativecraft.log",
        },
        "recipes": recipes if recipes is not None else _RECIPES,
    }
    return validate_definition(raw, source="nativecraft.yaml")


@pytest.fixture()
def store(tmp_path, monkeypatch) -> DataStore:
    instance = DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(native_mod, "get_store", lambda: instance)
    return instance


@pytest.fixture()
def installer(tmp_path, store) -> GenericNativeInstaller:
    return GenericNativeInstaller(
        _definition(),
        install_root=str(tmp_path / "nativecraft"),
        log_path=str(tmp_path / "nativecraft.log"),
    )


@pytest.fixture()
def activations(monkeypatch) -> list:
    calls: list = []

    async def fake_activate(engine_id, version_name, *, covered_by=None, row=None):
        calls.append((engine_id, version_name, dict(row or {})))

    monkeypatch.setattr(native_mod, "activate_engine_version", fake_activate)
    return calls


def _context(install_dir, tmp_path, *, build_exit: int = 0, smoke_output: str = "usage: ok"):
    marker = tmp_path / "marker.txt"
    cmake_script = (
        "import os, sys\n"
        f"open({str(marker)!r}, 'w').write(os.getcwd() + '|' + os.environ.get('NATIVECRAFT_MARK', ''))\n"
        "print('configure line')\n"
    )
    build_script = (
        f"import sys\nprint('build line one')\nprint('build line two')\nsys.exit({build_exit})\n"
    )
    smoke_script = f"print({smoke_output!r})\n"
    return {
        "input.install_dir": str(install_dir),
        "input.version_name": "1.0.0",
        "input.cmake_args": _script(cmake_script),
        "input.build_args": _script(build_script),
        "input.smoke_argv": _script(smoke_script),
        "input.build_env": {**os.environ, "NATIVECRAFT_MARK": "1"},
    }, marker


@pytest.mark.asyncio
async def test_cmake_steps_run_with_the_declared_argv_cwd_env(
    tmp_path, installer, store, activations
):
    install_dir = tmp_path / "nativecraft" / "1.0.0"
    context, marker = _context(install_dir, tmp_path)

    execution = await installer.execute_operation(
        "source",
        task_id="",
        install_dir=str(install_dir),
        context=context,
    )

    assert execution.status == "completed", execution.failed_step
    # The configure step ran in the install dir with the declared env.
    assert marker.read_text() == f"{install_dir}|1"
    row = store.get_engine_versions(NATIVECRAFT_ID)[0]
    assert row["version"] == "1.0.0"
    assert row["install_dir"] == str(install_dir)
    assert activations and activations[0][1] == "1.0.0"


@pytest.mark.asyncio
async def test_build_failure_reports_the_runner_error(tmp_path, store, activations):
    installer = GenericNativeInstaller(
        _definition(),
        install_root=str(tmp_path / "nativecraft"),
        log_path=str(tmp_path / "nativecraft.log"),
    )
    install_dir = tmp_path / "nativecraft" / "1.0.0"
    context, _marker = _context(install_dir, tmp_path, build_exit=3)

    execution = await installer.execute_operation(
        "source", task_id="", install_dir=str(install_dir), context=context
    )

    assert execution.status == "failed"
    assert execution.failed_step is not None
    assert execution.failed_step.error == (
        "build failed with exit code 3:\nbuild line one\nbuild line two"
    )
    assert store.get_engine_versions(NATIVECRAFT_ID) == []


@pytest.mark.asyncio
async def test_smoke_test_requires_output(tmp_path, store, activations):
    installer = GenericNativeInstaller(
        _definition(),
        install_root=str(tmp_path / "nativecraft"),
        log_path=str(tmp_path / "nativecraft.log"),
    )
    install_dir = tmp_path / "nativecraft" / "1.0.0"
    context, _marker = _context(install_dir, tmp_path, smoke_output="")

    execution = await installer.execute_operation(
        "source", task_id="", install_dir=str(install_dir), context=context
    )

    assert execution.status == "failed"
    assert execution.failed_step.error == "nativecraft_server returned no help"


@pytest.mark.asyncio
async def test_cmake_step_requires_an_argv(tmp_path, store):
    installer = GenericNativeInstaller(
        _definition(
            recipes={
                "source": [
                    {
                        "id": "configure",
                        "uses": "cmake.configure",
                        "requires": ["input.install_dir"],
                        "with": {"cwd_from": "input.install_dir"},
                    }
                ]
            }
        ),
        install_root=str(tmp_path / "nativecraft"),
        log_path=str(tmp_path / "nativecraft.log"),
    )
    install_dir = tmp_path / "nativecraft" / "1.0.0"

    execution = await installer.execute_operation(
        "source",
        task_id="",
        install_dir=str(install_dir),
        context={"input.install_dir": str(install_dir)},
    )

    assert execution.status == "failed"
    assert execution.failed_step.error == "configure requires an argv (argv or argv_from)"


# --- the shared runner itself ---------------------------------------------------


def _noop_emit(calls: list):
    async def emit(stage, progress, message, new_lines, all_lines):
        calls.append((stage, progress, message, list(new_lines)))

    return emit


@pytest.mark.asyncio
async def test_runner_streams_lines_and_returns_them():
    async def noop() -> None:
        return None

    calls: list = []
    lines = await run_streaming_command(
        _script("for i in range(3): print('line %d' % i)"),
        stage="build",
        progress=10,
        base_message="Building",
        emit=_noop_emit(calls),
        is_cancelled=lambda: False,
        terminate=noop,
        command_label="test",
    )

    assert lines == ["line 0", "line 1", "line 2"]
    assert calls and calls[0][0] == "build"
    assert calls[-1][2] == "Building"  # final completion emit
    assert calls[-1][3] == []


@pytest.mark.asyncio
async def test_runner_reports_nonzero_exit_with_the_tail():
    async def noop() -> None:
        return None

    with pytest.raises(RuntimeError) as excinfo:
        await run_streaming_command(
            _script("print('boom'); raise SystemExit(7)"),
            stage="configure",
            progress=0,
            base_message="Configuring",
            emit=_noop_emit([]),
            is_cancelled=lambda: False,
            terminate=noop,
            command_label="test",
        )
    assert str(excinfo.value) == "configure failed with exit code 7:\nboom"


@pytest.mark.asyncio
async def test_runner_refuses_when_cancellation_is_requested():
    from backend.task_cancel_registry import TaskCancelledError

    terminated: list = []

    async def terminate() -> None:
        terminated.append(True)

    with pytest.raises(TaskCancelledError) as excinfo:
        await run_streaming_command(
            _script("print('never')"),
            stage="build",
            progress=0,
            base_message="Building",
            emit=_noop_emit([]),
            is_cancelled=lambda: True,
            terminate=terminate,
            cancel_message="nativecraft build cancelled",
            command_label="test",
        )
    assert str(excinfo.value) == "nativecraft build cancelled"
    assert terminated == []
