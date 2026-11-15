"""CUDA toolkit disk-usage reporting and installer-cache cleanup."""

import shutil
from pathlib import Path

import pytest

from backend.cuda_installer import CUDAInstaller


def _installer(tmp_path: Path) -> CUDAInstaller:
    return CUDAInstaller(
        log_path=str(tmp_path / "logs" / "cuda.log"),
        state_path=str(tmp_path / "config" / "cuda.json"),
        download_dir=str(tmp_path / "downloads"),
    )


def _write_installation(installer: CUDAInstaller, tmp_path: Path, version: str) -> Path:
    install_path = tmp_path / f"cuda-{version}"
    (install_path / "bin").mkdir(parents=True)
    (install_path / "bin" / "nvcc").write_bytes(b"#!fake nvcc")
    (install_path / "lib64").mkdir()
    (install_path / "lib64" / "libcudart.so").write_bytes(b"r" * 1024)
    state = installer._load_state()
    state.setdefault("installations", {})[version] = {
        "path": str(install_path),
        "installed_at": "2026-01-01T00:00:00Z",
    }
    installer._save_state(state)
    return install_path


def _write_cache_file(installer: CUDAInstaller, relative_path: str, size: int) -> Path:
    file_path = Path(installer._download_dir) / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_bytes(b"c" * size)
    return file_path


def test_status_reports_install_size_and_installer_cache(tmp_path, monkeypatch):
    installer = _installer(tmp_path)
    install_path = _write_installation(installer, tmp_path, "12.8")
    _write_cache_file(installer, "cuda_12.8.2_linux.run", 4096)
    _write_cache_file(installer, "archive/cuda_12.7.1_linux.run", 1024)
    monkeypatch.setattr(installer, "_detect_cudnn_version", lambda path: None)
    monkeypatch.setattr(installer, "_detect_tensorrt_version", lambda path: None)

    status = installer.status()

    row = next(row for row in status["installed_versions"] if row["version"] == "12.8")
    assert row["path"] == str(install_path)
    expected_size = (install_path / "bin" / "nvcc").stat().st_size + (
        install_path / "lib64" / "libcudart.so"
    ).stat().st_size
    assert row["size_bytes"] == expected_size

    cache = status["installer_cache"]
    assert cache["bytes"] == 5120
    assert cache["files"] == 2
    assert cache["directory"] == installer._download_dir


def test_status_tolerates_missing_install_and_cache_directories(tmp_path, monkeypatch):
    installer = _installer(tmp_path)
    install_path = _write_installation(installer, tmp_path, "12.8")
    shutil.rmtree(install_path)
    shutil.rmtree(installer._download_dir)

    status = installer.status()

    assert status["installed_versions"] == []
    assert status["installer_cache"] == {
        "bytes": 0,
        "files": 0,
        "directory": installer._download_dir,
    }


@pytest.mark.asyncio
async def test_cleanup_installers_removes_cache_files_and_reports_freed_bytes(tmp_path):
    installer = _installer(tmp_path)
    install_path = _write_installation(installer, tmp_path, "12.8")
    _write_cache_file(installer, "cuda_12.8.2_linux.run", 4096)
    _write_cache_file(installer, "archive/cuda_12.7.1_linux.run", 1024)
    (Path(installer._download_dir) / "empty-subdir").mkdir()

    result = await installer.cleanup_installers()

    assert result == {"freed_bytes": 5120, "removed": 2}
    assert list(Path(installer._download_dir).rglob("*")) == []
    assert Path(installer._download_dir).is_dir()
    # Installed toolkits are untouched by cache cleanup.
    assert (install_path / "bin" / "nvcc").is_file()


@pytest.mark.asyncio
async def test_cleanup_installers_tolerates_missing_download_directory(tmp_path):
    installer = _installer(tmp_path)
    shutil.rmtree(installer._download_dir)

    assert await installer.cleanup_installers() == {"freed_bytes": 0, "removed": 0}


@pytest.mark.asyncio
async def test_cleanup_installers_is_refused_while_an_operation_is_running(tmp_path):
    installer = _installer(tmp_path)
    cache_file = _write_cache_file(installer, "cuda_12.8.2_linux.run", 4096)
    installer._operation = "install"

    with pytest.raises(RuntimeError, match="already running"):
        await installer.cleanup_installers()

    assert cache_file.is_file()


def test_cuda_cleanup_api_reports_freed_bytes(client, monkeypatch):
    class Installer:
        async def cleanup_installers(self):
            return {"freed_bytes": 5120, "removed": 2}

    monkeypatch.setattr("backend.routes.llama_versions.get_cuda_installer", lambda: Installer())

    response = client.post("/api/llama-versions/cuda-cleanup")

    assert response.status_code == 200
    assert response.json() == {"freed_bytes": 5120, "removed": 2}


def test_cuda_cleanup_api_refuses_during_running_operation(client, monkeypatch):
    class Installer:
        async def cleanup_installers(self):
            raise RuntimeError("Another CUDA installer operation is already running")

    monkeypatch.setattr("backend.routes.llama_versions.get_cuda_installer", lambda: Installer())

    response = client.post("/api/llama-versions/cuda-cleanup")

    assert response.status_code == 400
    assert "already running" in response.json()["detail"]
