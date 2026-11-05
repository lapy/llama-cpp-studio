from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from devcontrol.config import Config
from devcontrol.local_runtime import LocalRuntime
from devcontrol.runtime import CommandResult
from devcontrol.service import DevControl


def make_config(tmp_path: Path, repo: Path) -> Config:
    config = Config(
        state_home=tmp_path / "state",
        repo=repo,
        data=tmp_path / "data",
        worktrees=tmp_path / "worktrees",
        shared_models=tmp_path / "shared",
        runtime="local",
    )
    config.validate()
    return config


def test_local_runtime_parses_supervisor_status(
    tmp_path: Path, git_repo: Path, monkeypatch
) -> None:
    config = make_config(tmp_path, git_repo)
    runtime = LocalRuntime(config)
    runtime.active_link = tmp_path / "active"

    def fake_ctl(*args: str, timeout: float = 30.0) -> CommandResult:
        assert args == ("status",)
        return CommandResult(
            argv=["supervisorctl"],
            returncode=0,
            stdout="backend RUNNING pid 1\nfrontend STOPPED\n",
            stderr="",
            duration_seconds=0.0,
        )

    monkeypatch.setattr(runtime, "_ctl", fake_ctl)
    assert runtime.running("backend") is True
    assert runtime.running("frontend") is False
    assert runtime.any_running() is True
    assert runtime.summary()["runtime"] == "local"


def test_local_runtime_exec_uses_active_checkout(tmp_path: Path, git_repo: Path) -> None:
    config = make_config(tmp_path, git_repo)
    runtime = LocalRuntime(config)
    result = runtime.exec("backend", [sys.executable, "-c", "import os; print(os.getcwd())"])
    assert result.ok is True
    assert str(git_repo) in result.stdout


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs Windows developer mode")
def test_local_runtime_switches_active_symlink(tmp_path: Path, git_repo: Path) -> None:
    config = make_config(tmp_path, git_repo)
    runtime = LocalRuntime(config)
    runtime.active_link = tmp_path / "active"
    runtime.set_active("main")
    assert runtime.active_link.is_symlink()
    assert runtime.active_link.resolve() == git_repo.resolve()


def test_devcontrol_selects_local_runtime(tmp_path: Path, git_repo: Path) -> None:
    config = make_config(tmp_path, git_repo)
    control = DevControl(config)
    assert isinstance(control.compose, LocalRuntime)
