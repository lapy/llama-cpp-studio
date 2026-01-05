from pathlib import Path

import pytest

from backend.lmdeploy_installer import LMDeployInstaller


@pytest.mark.asyncio
async def test_install_prevents_parallel_operations(tmp_path: Path, monkeypatch):
    installer = LMDeployInstaller(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    # Prevent the background task from executing pip in tests
    def prevent_task(coro):
        coro.close()

    monkeypatch.setattr(installer, "_create_task", prevent_task)

    result = await installer.install()
    assert result["message"].startswith("LMDeploy installation started")

    with pytest.raises(RuntimeError):
        await installer.install()


def test_status_reflects_detection(tmp_path: Path, monkeypatch):
    installer = LMDeployInstaller(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    monkeypatch.setattr(installer, "_detect_installed_version", lambda: "0.10.0")
    monkeypatch.setattr(installer, "_resolve_binary_path", lambda: "/opt/lmdeploy/bin/lmdeploy")

    status = installer.status()
    assert status["installed"] is True
    assert status["version"] == "0.10.0"
    assert status["binary_path"].endswith("lmdeploy")
    assert status["venv_path"] == installer._venv_path

