from __future__ import annotations

from pathlib import Path

from devcontrol.compose import Compose
from devcontrol.config import Config
from devcontrol.runtime import CommandResult


def make_config(tmp_path: Path) -> Config:
    config = Config(
        state_home=tmp_path / "state",
        repo=tmp_path / "repo",
        data=tmp_path / "data",
        worktrees=tmp_path / "worktrees",
        shared_models=tmp_path / "shared",
    )
    config.validate()
    config.save()
    return config


def test_compose_base_uses_isolated_project_and_env_file(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    command = Compose(config)._base()
    assert command[:2] == ["docker", "compose"]
    assert "--project-name" in command
    assert command[command.index("--project-name") + 1] == "llama-cpp-studio-dev"
    assert "--env-file" in command
    assert str(config.env_path) in command


def test_compose_run_uses_state_directory_not_checkout(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    captured = {}

    def fake_run(argv, *, cwd=None, env=None, timeout=60.0):
        captured["argv"] = argv
        captured["cwd"] = cwd
        captured["env"] = env
        return CommandResult(
            argv=list(argv), returncode=0, stdout="", stderr="", duration_seconds=0.0
        )

    monkeypatch.setattr("devcontrol.compose.which", lambda name: "/usr/bin/docker")
    monkeypatch.setattr("devcontrol.compose.run_capture", fake_run)
    Compose(config)._run(["ps"], check=False)
    assert Path(captured["cwd"]) == config.state_home
    assert captured["env"]["DEV_DATA"] == str(config.data)
