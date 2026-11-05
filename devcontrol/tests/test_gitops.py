from __future__ import annotations

from pathlib import Path

import pytest

from devcontrol.errors import GuardError
from devcontrol.gitops import GitRepo, parse_patch_paths


def test_revision_identifies_dirty_tree(git_repo: Path) -> None:
    repo = GitRepo(git_repo)
    clean = repo.revision()
    assert clean.dirty is False
    assert clean.label == clean.commit

    (git_repo / "a.txt").write_text("two\n", encoding="utf-8")
    dirty = repo.revision()
    assert dirty.dirty is True
    assert dirty.label.startswith(f"{dirty.commit}+dirty.")
    assert dirty.changed_files == 1


def test_checkpoint_patch_and_rollback(git_repo: Path, patch_text: str) -> None:
    repo = GitRepo(git_repo)
    checkpoint = repo.checkpoint(worktree="main", label="before")
    assert repo.checkpoint_commit(checkpoint["id"]) == checkpoint["commit"]

    files = repo.apply_unified_patch(patch_text)
    assert files == ["a.txt"]
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "two\n"

    repo.restore_to_commit(checkpoint["commit"])
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "one\n"
    assert repo.revision().dirty is False


def test_rollback_restores_untracked_and_removes_new_files(git_repo: Path) -> None:
    repo = GitRepo(git_repo)
    (git_repo / "untracked.txt").write_text("keep me\n", encoding="utf-8")
    checkpoint = repo.checkpoint(worktree="main", label="with untracked")

    (git_repo / "untracked.txt").write_text("changed\n", encoding="utf-8")
    (git_repo / "later.txt").write_text("new\n", encoding="utf-8")
    repo.restore_to_commit(checkpoint["commit"])

    assert (git_repo / "untracked.txt").read_text(encoding="utf-8") == "keep me\n"
    assert not (git_repo / "later.txt").exists()


def test_rejects_parent_traversal_patch(git_repo: Path) -> None:
    repo = GitRepo(git_repo)
    patch = (
        "diff --git a/a.txt b/../outside.txt\n"
        "--- a/a.txt\n"
        "+++ b/../outside.txt\n"
        "@@ -1 +1 @@\n"
        "-one\n"
        "+two\n"
    )
    with pytest.raises(GuardError):
        repo.check_unified_patch(patch)


def test_parse_patch_paths_handles_quoted_paths() -> None:
    paths = parse_patch_paths(
        'diff --git "a/space name.txt" "b/space name.txt"\n'
        '--- "a/space name.txt"\n'
        '+++ "b/space name.txt"\n'
    )
    assert paths == ["space name.txt"]


def test_worktree_add_list_remove(git_repo: Path, tmp_path: Path) -> None:
    repo = GitRepo(git_repo)
    path = tmp_path / "worktrees" / "alice"
    created = repo.add_worktree(path, name="alice", base="HEAD")
    assert created["branch"] == "agent/alice"
    assert path.is_dir()
    paths = {item["path"] for item in repo.worktrees()}
    assert str(path) in paths or str(path).replace("\\", "/") in paths
    repo.remove_worktree(path, force=True)
    assert not path.exists()
