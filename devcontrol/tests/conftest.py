"""Test fixtures for the development control plane.

The suite deliberately keeps its temporary directories inside the workspace.
This makes the tests runnable in restricted Windows sandboxes where Python's
mode-0700 ``tmp_path`` directory cannot be reopened. The directory is ignored by
git and removed after the session.
"""

from __future__ import annotations

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
TEMP_ROOT = ROOT / ".pytest-devcontrol-tmp"


class _Factory:
    def __init__(self) -> None:
        self.base = TEMP_ROOT / uuid.uuid4().hex
        self.base.mkdir(parents=True)

    def mktemp(self, name: str = "test") -> Path:
        path = self.base / f"{name}-{uuid.uuid4().hex[:8]}"
        path.mkdir()
        return path

    def cleanup(self) -> None:
        shutil.rmtree(self.base, ignore_errors=True)


@pytest.fixture(scope="session")
def tmp_path_factory() -> _Factory:
    factory = _Factory()
    yield factory
    factory.cleanup()


@pytest.fixture
def tmp_path(tmp_path_factory: _Factory) -> Path:
    return tmp_path_factory.mktemp("case")


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        timeout=60,
        check=check,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "a.txt").write_text("one\n", encoding="utf-8")
    _git(repo, "add", "a.txt")
    _git(
        repo,
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "initial",
    )
    return repo


@pytest.fixture
def patch_text() -> str:
    return "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-one\n+two\n"
