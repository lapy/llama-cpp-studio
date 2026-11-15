"""Contract tests for the typed CUDA installer errors.

The Hardware UI switches on ``detail.code``, so these tests pin the HTTP status
and stable code for the public failures without starting real downloads.
"""

import pytest

from backend.cuda_errors import (
    CudaInstallerError,
    CudaInstallFailed,
    CudaInstallIncomplete,
    CudaOperationBusy,
    CudaUnsupportedPlatform,
    CudaVersionNotInstalled,
    CudaVersionUnsupported,
)
from backend.cuda_installer import CUDAInstaller


def _installer(tmp_path):
    return CUDAInstaller(
        log_path=str(tmp_path / "cuda.log"),
        state_path=str(tmp_path / "cuda-state.json"),
        download_dir=str(tmp_path / "downloads"),
    )


def _use_installer(monkeypatch, installer):
    monkeypatch.setattr("backend.routes.llama_versions.get_cuda_installer", lambda: installer)


def _write_installation(installer, version, install_path):
    state = installer._load_state()
    state.setdefault("installations", {})[version] = {
        "path": str(install_path),
        "installed_at": "2026-01-01T00:00:00Z",
    }
    installer._save_state(state)


def _detail(response):
    return response.json()["detail"]


@pytest.mark.parametrize(
    "error_cls,code,status",
    [
        (CudaOperationBusy, "cuda_operation_busy", 409),
        (CudaVersionUnsupported, "cuda_version_unsupported", 400),
        (CudaVersionNotInstalled, "cuda_version_not_installed", 404),
        (CudaInstallIncomplete, "cuda_install_incomplete", 409),
        (CudaInstallFailed, "cuda_install_failed", 500),
        (CudaUnsupportedPlatform, "cuda_platform_unsupported", 400),
    ],
)
def test_typed_error_defaults_are_stable(error_cls, code, status):
    err = error_cls()

    assert isinstance(err, CudaInstallerError)
    assert err.code == code
    assert err.http_status == status
    assert str(err) == err.message
    assert err.to_detail() == {"code": code, "message": err.message}


def test_error_keeps_the_explicit_message():
    message = "CUDA 12.8 installation is missing or incomplete"

    err = CudaInstallIncomplete(message)

    assert str(err) == message
    assert err.to_detail() == {"code": "cuda_install_incomplete", "message": message}


def test_activate_unknown_version_is_404_not_installed(client, monkeypatch, tmp_path):
    _use_installer(monkeypatch, _installer(tmp_path))

    response = client.post("/api/llama-versions/cuda-activate", json={"version": "12.8.2"})

    assert response.status_code == 404
    detail = _detail(response)
    assert detail["code"] == "cuda_version_not_installed"
    assert detail["message"] == "CUDA 12.8.2 installation not found in state"


def test_activate_empty_version_is_400_unsupported(client, monkeypatch, tmp_path):
    _use_installer(monkeypatch, _installer(tmp_path))

    response = client.post("/api/llama-versions/cuda-activate", json={"version": ""})

    assert response.status_code == 400
    detail = _detail(response)
    assert detail["code"] == "cuda_version_unsupported"
    assert detail["message"] == "CUDA version is required"


def test_activate_incomplete_installation_is_409(client, monkeypatch, tmp_path):
    installer = _installer(tmp_path)
    install_path = tmp_path / "cuda-12.8.2"
    install_path.mkdir()  # no bin/nvcc
    _write_installation(installer, "12.8.2", install_path)
    _use_installer(monkeypatch, installer)

    response = client.post("/api/llama-versions/cuda-activate", json={"version": "12.8.2"})

    assert response.status_code == 409
    detail = _detail(response)
    assert detail["code"] == "cuda_install_incomplete"
    assert detail["message"] == "CUDA 12.8.2 installation is missing or incomplete"


def test_install_unsupported_version_is_400(client, monkeypatch, tmp_path):
    installer = _installer(tmp_path)
    monkeypatch.setattr(installer, "_get_platform", lambda: ("linux", "x86_64"))

    async def discover_nothing():
        return []

    monkeypatch.setattr(installer, "get_available_versions", discover_nothing)
    _use_installer(monkeypatch, installer)

    response = client.post("/api/llama-versions/cuda-install", json={"version": "99.9"})

    assert response.status_code == 400
    detail = _detail(response)
    assert detail["code"] == "cuda_version_unsupported"
    assert detail["message"].startswith("Unsupported or unavailable CUDA version: 99.9.")


def test_install_invalid_version_is_400(client, monkeypatch, tmp_path):
    installer = _installer(tmp_path)
    monkeypatch.setattr(installer, "_get_platform", lambda: ("linux", "x86_64"))
    _use_installer(monkeypatch, installer)

    response = client.post("/api/llama-versions/cuda-install", json={"version": "v12.8"})

    assert response.status_code == 400
    detail = _detail(response)
    assert detail["code"] == "cuda_version_unsupported"
    assert detail["message"].startswith("Invalid CUDA version: v12.8")


def test_install_non_linux_is_400_platform_unsupported(client, monkeypatch, tmp_path):
    installer = _installer(tmp_path)
    monkeypatch.setattr(installer, "_get_platform", lambda: ("windows", "amd64"))
    _use_installer(monkeypatch, installer)

    response = client.post("/api/llama-versions/cuda-install", json={"version": "12.8"})

    assert response.status_code == 400
    detail = _detail(response)
    assert detail["code"] == "cuda_platform_unsupported"
    assert detail["message"] == "CUDA installation is only supported on Linux, not windows"


@pytest.mark.parametrize(
    "path,payload",
    [
        ("/api/llama-versions/cuda-install", {"version": "12.6"}),
        ("/api/llama-versions/cuda-uninstall", {"version": "12.6"}),
    ],
)
def test_concurrent_operation_is_409_busy(client, monkeypatch, tmp_path, path, payload):
    installer = _installer(tmp_path)
    installer._operation = "install"
    _use_installer(monkeypatch, installer)

    response = client.post(path, json=payload)

    assert response.status_code == 409
    detail = _detail(response)
    assert detail["code"] == "cuda_operation_busy"
    assert detail["message"] == "Another CUDA installer operation is already running"


def test_uninstall_not_installed_is_404(client, monkeypatch, tmp_path):
    _use_installer(monkeypatch, _installer(tmp_path))

    response = client.post("/api/llama-versions/cuda-uninstall", json={"version": "12.6"})

    assert response.status_code == 404
    detail = _detail(response)
    assert detail["code"] == "cuda_version_not_installed"
    assert detail["message"] == "CUDA 12.6 installation not found in state"


@pytest.mark.asyncio
async def test_install_linux_missing_installer_is_cuda_install_failed(tmp_path):
    installer = _installer(tmp_path)

    with pytest.raises(CudaInstallFailed, match="Installer file not found"):
        await installer._install_linux(str(tmp_path / "missing.run"), "12.8")
