import json
from pathlib import Path

import pytest

from backend.cuda_errors import CudaVersionNotInstalled
from backend.cuda_installer import CUDAInstaller


def _installer(tmp_path: Path) -> CUDAInstaller:
    return CUDAInstaller(
        log_path=str(tmp_path / "cuda.log"),
        state_path=str(tmp_path / "cuda-state.json"),
        download_dir=str(tmp_path / "downloads"),
    )


def _write_installation(installer: CUDAInstaller, tmp_path: Path, version: str) -> Path:
    install_path = tmp_path / f"cuda-{version}"
    (install_path / "bin").mkdir(parents=True)
    (install_path / "bin" / "nvcc").write_text("fixture", encoding="utf-8")
    state = installer._load_state()
    state.setdefault("installations", {})[version] = {
        "path": str(install_path),
        "installed_at": f"2026-01-{version[-1]}T00:00:00Z",
    }
    installer._save_state(state)
    return install_path


@pytest.mark.asyncio
async def test_activate_selects_an_existing_installation(tmp_path, monkeypatch):
    installer = _installer(tmp_path)
    old_path = _write_installation(installer, tmp_path, "12.8")
    selected_path = _write_installation(installer, tmp_path, "12.8.2")
    selected = {"path": str(old_path)}

    monkeypatch.setattr(
        installer, "_update_current_symlink", lambda path: selected.update(path=path)
    )
    monkeypatch.setattr(installer, "_get_cuda_path", lambda version=None: selected["path"])
    monkeypatch.setattr(installer, "get_cuda_env", lambda version=None: {})

    class Proxy:
        restarted = False

        async def restart_proxy(self):
            self.restarted = True

    proxy = Proxy()
    monkeypatch.setattr("backend.proxy.control.get_proxy_controller", lambda: proxy)

    result = await installer.activate("12.8.2")

    assert result["version"] == "12.8.2"
    assert selected["path"] == str(selected_path)
    state = json.loads(Path(installer._state_path).read_text(encoding="utf-8"))
    assert state["installed_version"] == "12.8.2"
    assert state["cuda_path"] == str(selected_path)
    assert proxy.restarted is True


@pytest.mark.asyncio
async def test_activate_rejects_missing_installation(tmp_path):
    installer = _installer(tmp_path)

    with pytest.raises(CudaVersionNotInstalled, match="installation not found"):
        await installer.activate("12.8.2")


def test_status_prefers_selected_version_over_latest_install(tmp_path, monkeypatch):
    installer = _installer(tmp_path)
    selected_path = _write_installation(installer, tmp_path, "12.8")
    _write_installation(installer, tmp_path, "12.8.2")
    state = installer._load_state()
    state["installed_version"] = "12.8"
    state["cuda_path"] = str(selected_path)
    installer._save_state(state)
    monkeypatch.setattr(installer, "_detect_cudnn_version", lambda path: None)
    monkeypatch.setattr(installer, "_detect_tensorrt_version", lambda path: None)

    status = installer.status()

    assert status["version"] == "12.8"
    assert (
        next(row for row in status["installed_versions"] if row["version"] == "12.8")["is_current"]
        is True
    )


def test_cuda_activate_api_selects_existing_version(client, monkeypatch):
    class Installer:
        async def activate(self, version):
            return {"message": f"CUDA {version} activated", "version": version}

    monkeypatch.setattr("backend.routes.llama_versions.get_cuda_installer", lambda: Installer())

    response = client.post("/api/llama-versions/cuda-activate", json={"version": "12.8.2"})

    assert response.status_code == 200
    assert response.json()["version"] == "12.8.2"
