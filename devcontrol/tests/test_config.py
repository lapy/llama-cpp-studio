from __future__ import annotations

from pathlib import Path

import pytest

from devcontrol.config import Config
from devcontrol.errors import ConfigError


def base_config(tmp_path: Path, repo: Path) -> Config:
    return Config(
        state_home=tmp_path / "state",
        repo=repo,
        data=tmp_path / "data",
        worktrees=tmp_path / "worktrees",
        shared_models=tmp_path / "shared",
    )


def test_valid_config_has_separate_defaults(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.validate()
    assert config.compose_project == "llama-cpp-studio-dev"
    assert config.backend_port != config.frontend_port
    assert config.cuda_visible_devices == ""


def test_rejects_live_ports(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.backend_port = 8080
    with pytest.raises(ConfigError):
        config.validate()


def test_local_runtime_may_use_internal_ports(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.runtime = "local"
    config.backend_port = 8081
    config.frontend_port = 5173
    config.proxy_port = 2000
    config.validate()
    assert config.runtime == "local"


def test_rejects_duplicate_ports(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.proxy_port = config.frontend_port
    with pytest.raises(ConfigError):
        config.validate()


def test_rejects_live_project_and_image(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.compose_project = "llama-cpp-studio"
    with pytest.raises(ConfigError):
        config.validate()

    config = base_config(tmp_path, git_repo)
    config.image = "llama-cpp-studio:latest"
    with pytest.raises(ConfigError):
        config.validate()


def test_rejects_data_inside_checkout(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.data = git_repo / "data"
    with pytest.raises(ConfigError):
        config.validate()


def test_gpu_selection_maps_visible_devices(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.gpu = "1,3"
    config.validate()
    assert config.cuda_visible_devices == "0,1"
    env = config.compose_env()
    assert env["DEV_GPU"] == "1,3"
    assert env["DEV_CUDA_VISIBLE_DEVICES"] == "0,1"

    config.gpu = "none"
    config.validate()
    assert config.gpu == ""


def test_rejects_bad_gpu_selection(tmp_path: Path, git_repo: Path) -> None:
    config = base_config(tmp_path, git_repo)
    config.gpu = "0,0"
    with pytest.raises(ConfigError):
        config.validate()
    config.gpu = "-1"
    with pytest.raises(ConfigError):
        config.validate()


def test_env_file_roundtrip_does_not_put_token_in_config_json(
    tmp_path: Path, git_repo: Path
) -> None:
    config = base_config(tmp_path, git_repo)
    config.api_token = "token-value"
    config.save()
    config_json = (tmp_path / "state" / "config.json").read_text(encoding="utf-8")
    env_text = (tmp_path / "state" / "dev.env").read_text(encoding="utf-8")
    assert "token-value" not in config_json
    assert "DEV_API_TOKEN=token-value" in env_text
