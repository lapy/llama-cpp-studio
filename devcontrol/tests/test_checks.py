from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from devcontrol.checks import expand_checks, run_checks, validate_test_targets
from devcontrol.errors import GuardError
from devcontrol.runtime import CommandResult


def test_expand_checks_aliases_and_all() -> None:
    names = [check.name for check in expand_checks(["pytest", "eslint", "all"])]
    assert "backend-tests" in names
    assert "frontend-lint" in names
    assert len(names) == len(set(names))


def test_unknown_check_is_rejected() -> None:
    with pytest.raises(GuardError):
        expand_checks(["format-production-database"])


def test_test_targets_must_stay_under_backend_tests(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    tests = root / "backend" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_demo.py").write_text("", encoding="utf-8")
    (root / "outside.py").write_text("", encoding="utf-8")

    assert validate_test_targets(root, ["backend/tests/test_demo.py"]) == [
        "backend/tests/test_demo.py"
    ]
    with pytest.raises(GuardError):
        validate_test_targets(root, ["backend/outside.py"])
    with pytest.raises(GuardError):
        validate_test_targets(root, ["backend/tests/../../outside.py"])


def test_run_checks_uses_selected_targets_and_user(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    tests = root / "backend" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_demo.py").write_text("", encoding="utf-8")
    calls = []

    class FakeCompose:
        config = SimpleNamespace(uid="1000", gid="1000")

        def exec(self, service, argv, *, timeout, user):
            calls.append((service, argv, timeout, user))
            return CommandResult(
                argv=list(argv), returncode=0, stdout="ok", stderr="", duration_seconds=0.1
            )

    result = run_checks(
        FakeCompose(),
        names=["backend-tests"],
        repo=root,
        tests=["backend/tests/test_demo.py"],
        timeout_seconds=30,
    )
    assert result["ok"] is True
    assert calls[0][0] == "backend"
    assert calls[0][1][:3] == ["python", "-m", "pytest"]
    assert "backend/tests/test_demo.py" in calls[0][1]
    assert calls[0][3] == "1000:1000"
