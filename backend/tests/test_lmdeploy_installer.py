from pathlib import Path
import sys

import pytest

from backend.tests.venv_fixtures import LMDeployManager


@pytest.mark.asyncio
async def test_run_logged_preserves_indentation_and_blank_lines(tmp_path: Path, monkeypatch):
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )
    broadcasted = []

    async def fake_broadcast(line: str, *, record: bool = True) -> None:
        if record:
            broadcasted.append(line)

    monkeypatch.setattr(installer, "_broadcast_log_line", fake_broadcast)
    script = "print('  /\\\\_/\\\\  ')\nprint()\nprint(' ( o.o ) ')\n"

    returncode = await installer._run_logged([sys.executable, "-c", script], "install")

    assert returncode == 0
    assert any(line.startswith("  /\\_/\\") for line in broadcasted)
    assert "" in broadcasted
    assert " ( o.o )" in broadcasted


@pytest.mark.asyncio
async def test_install_prevents_parallel_operations(tmp_path: Path, monkeypatch):
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    # Prevent the background task from executing pip in tests
    def prevent_task(coro):
        coro.close()

    monkeypatch.setattr(installer, "_create_task", prevent_task)

    result = await installer.install_release()
    assert result["message"].startswith("LMDeploy installation started")

    with pytest.raises(RuntimeError):
        await installer.install_release()
    from backend.operations.supervisor import get_supervisor

    get_supervisor().finish_operation(result["task_id"], "succeeded", "")


def test_status_reflects_detection(tmp_path: Path, monkeypatch):
    installer = LMDeployManager(
        log_path=str(tmp_path / "lmdeploy.log"),
        state_path=str(tmp_path / "lmdeploy_state.json"),
        base_dir=str(tmp_path / "lmdeploy"),
    )

    monkeypatch.setattr(installer, "_detect_installed_version", lambda: "0.10.0")
    monkeypatch.setattr(
        installer, "_resolve_binary_path", lambda: "/opt/lmdeploy/bin/lmdeploy"
    )

    status = installer.status()
    assert status["installed"] is True
    assert status["version"] == "0.10.0"
    assert status["binary_path"].endswith("lmdeploy")
    assert status["venv_path"] == installer._venv_path
