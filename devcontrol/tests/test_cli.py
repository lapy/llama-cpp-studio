from __future__ import annotations

import json
from pathlib import Path

from devcontrol.bootstrap import _write_wrapper
from devcontrol.cli import _normalise_argv, main


def test_bootstrap_and_journal_emit_json(tmp_path: Path, git_repo: Path, capsys) -> None:
    state = tmp_path / "state"
    argv = [
        "--state-home",
        str(state),
        "--repo",
        str(git_repo),
        "--data",
        str(tmp_path / "data"),
        "--worktrees",
        str(tmp_path / "worktrees"),
        "--shared-models",
        str(tmp_path / "shared"),
        "bootstrap",
        "--no-wrappers",
    ]
    assert main(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["revision"]
    assert payload["token_created"] is True
    assert "DEV_API_TOKEN" not in json.dumps(payload)

    assert main(["--state-home", str(state), "journal", "--limit", "10"]) == 0
    journal_payload = json.loads(capsys.readouterr().out)
    assert journal_payload["operations"][0]["action"] == "bootstrap"


def test_wrapper_carries_the_custom_state_home(tmp_path: Path) -> None:
    wrapper = tmp_path / "agent-dev"
    state = tmp_path / "state with space"
    _write_wrapper(wrapper, tmp_path / "repo", state, "mcp")
    body = wrapper.read_text(encoding="utf-8")
    assert "DEV_CONTROL_HOME=" in body
    assert "state with space" in body
    assert "exec python3 -m devcontrol mcp" in body


def test_json_flags_can_follow_subcommand() -> None:
    assert _normalise_argv(["inspect-instance", "--json", "--pretty"]) == [
        "--pretty",
        "--json",
        "inspect-instance",
    ]


def test_uninitialized_state_returns_error_json(tmp_path: Path, capsys) -> None:
    assert main(["--state-home", str(tmp_path / "missing"), "inspect-instance"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "CONFIG_INVALID"
