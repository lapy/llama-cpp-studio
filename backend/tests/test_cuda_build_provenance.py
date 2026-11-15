"""CUDA provenance recorded on native build tasks (and returned by the detail route).

A native source build must say which CUDA toolkit it used. The record is taken
once when the build starts, from the environment the build inherits plus the
Studio-managed install the CMake handlers bind to, and it is exposed through the
retained-task route the task detail reads.
"""

from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

import pytest

from backend.engines.cuda_provenance import record_cuda_provenance, resolve_cuda_provenance
from backend.engines.definitions.executor import EXECUTION_COMPLETED
from backend.engines.python.cmake_build import BuildConfig
from backend.engines.python.cmake_source import CmakeSourceInstaller
from backend.operations.progress import ProgressManager

TASK_ID = "build_provenance_test"


def _managed_cuda(data_root, version: str = "12.8.2"):
    home = data_root / "cuda" / f"cuda-{version}"
    (home / "bin").mkdir(parents=True)
    (home / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")).write_text("", encoding="utf-8")
    return home


def _clear_inherited_cuda(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.delenv("CUDA_PATH", raising=False)


def _start_native_build(monkeypatch, tmp_path, build_config: BuildConfig) -> dict:
    """Run ``CmakeSourceInstaller.run_source`` with the launch itself stubbed."""
    launched = []

    async def fake_execute(recipe, **_kwargs):
        launched.append(recipe)
        return SimpleNamespace(
            status=EXECUTION_COMPLETED,
            failed_step=None,
            outputs={"binary": str(tmp_path / "llama-server")},
        )

    installer = CmakeSourceInstaller("llama_cpp", install_root=str(tmp_path / "llama-cpp"))
    monkeypatch.setattr(installer, "execute_operation", fake_execute)
    progress = ProgressManager()
    progress.create_task("build", "Build llama.cpp main", {"engine": "llama_cpp"}, task_id=TASK_ID)

    asyncio.run(
        installer.run_source(
            "main",
            build_config=build_config,
            progress_manager=progress,
            task_id=TASK_ID,
            version_name="source-main-test",
        )
    )

    assert launched == ["source"]
    task = progress.get_task(TASK_ID)
    assert task is not None
    return task


def test_native_build_records_the_studio_managed_toolkit(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    home = _managed_cuda(data_root)
    monkeypatch.setenv("STUDIO_DATA_DIR", str(data_root))
    _clear_inherited_cuda(monkeypatch)

    task = _start_native_build(monkeypatch, tmp_path, BuildConfig(enable_cuda=True))

    assert task["metadata"]["cuda"] == {
        "cuda_home": os.path.realpath(home),
        "cuda_version": "12.8.2",
        "source": "studio-managed",
    }


def test_native_build_records_an_inherited_system_toolkit(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    system_home = tmp_path / "system" / "cuda-12.4"
    (system_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(system_home))
    monkeypatch.delenv("CUDA_PATH", raising=False)

    task = _start_native_build(monkeypatch, tmp_path, BuildConfig(enable_cuda=True))

    assert task["metadata"]["cuda"] == {
        "cuda_home": os.path.realpath(system_home),
        "cuda_version": "12.4",
        "source": "system",
    }


def test_native_build_records_no_toolkit_as_unknown(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    _clear_inherited_cuda(monkeypatch)

    task = _start_native_build(monkeypatch, tmp_path, BuildConfig(enable_cuda=True))

    assert task["metadata"]["cuda"] == {
        "cuda_home": None,
        "cuda_version": None,
        "source": "unknown",
    }


def test_cpu_only_native_build_records_no_cuda_provenance(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    _managed_cuda(data_root)
    monkeypatch.setenv("STUDIO_DATA_DIR", str(data_root))
    _clear_inherited_cuda(monkeypatch)

    task = _start_native_build(monkeypatch, tmp_path, BuildConfig(enable_cuda=False))

    # A CPU build did not use a toolkit; claiming one would be worse than the
    # panel's "CUDA: not recorded" fallback.
    assert "cuda" not in task["metadata"]


def test_task_detail_route_returns_the_build_cuda_provenance(client, monkeypatch, tmp_path):
    from backend.operations.progress import get_progress_manager

    data_root = tmp_path / "data"
    home = _managed_cuda(data_root, "12.9.0")
    monkeypatch.setenv("STUDIO_DATA_DIR", str(data_root))
    _clear_inherited_cuda(monkeypatch)

    progress = get_progress_manager()
    task_id = "build_provenance_detail_test"
    progress.create_task("build", "Build llama.cpp main", {"engine": "llama_cpp"}, task_id=task_id)
    record_cuda_provenance(progress, task_id)

    response = client.get(f"/api/operations/tasks/{task_id}/logs")

    assert response.status_code == 200
    body = response.json()
    assert body["task_id"] == task_id
    assert body["metadata"]["cuda"] == {
        "cuda_home": os.path.realpath(home),
        "cuda_version": "12.9.0",
        "source": "studio-managed",
    }


def test_recording_without_a_retained_task_is_a_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    _clear_inherited_cuda(monkeypatch)

    assert record_cuda_provenance(ProgressManager(), None) is None
    assert record_cuda_provenance(None, TASK_ID) is None


def test_system_toolkit_without_a_versioned_path_records_no_version(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDIO_DATA_DIR", str(tmp_path / "data"))
    plain_home = tmp_path / "opt" / "cuda"
    (plain_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("CUDA_PATH", str(plain_home))
    monkeypatch.delenv("CUDA_HOME", raising=False)

    assert resolve_cuda_provenance() == {
        "cuda_home": os.path.realpath(plain_home),
        "cuda_version": None,
        "source": "system",
    }


@pytest.mark.skipif(os.name == "nt", reason="creating symlinks needs privileges on Windows")
def test_current_pointer_wins_over_the_inherited_environment(monkeypatch, tmp_path):
    data_root = tmp_path / "data"
    home = _managed_cuda(data_root, "12.7.0")
    (home.parent / "current").symlink_to(home, target_is_directory=True)
    monkeypatch.setenv("STUDIO_DATA_DIR", str(data_root))
    system_home = tmp_path / "system" / "cuda-12.4"
    (system_home / "bin").mkdir(parents=True)
    monkeypatch.setenv("CUDA_HOME", str(system_home))

    assert resolve_cuda_provenance() == {
        "cuda_home": os.path.realpath(home),
        "cuda_version": "12.7.0",
        "source": "studio-managed",
    }
