"""Git worktree, revision identity, checkpoint, patch, and rollback primitives."""

from __future__ import annotations

import os
import re
import shlex
import tempfile
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .errors import CommandError, ConfigError, GuardError, NotFoundError
from .locks import atomic_write_text
from .runtime import CommandResult, run_capture

CHECKPOINT_REF_PREFIX = "refs/agent-dev/checkpoints/"
MAX_PATCH_BYTES = 2 * 1024 * 1024
_PATH_RE = re.compile(r"^(?:[ab]/(.+)|(.+))$")
_BINARY_MARKER = re.compile(r"^(?:GIT binary patch|Binary files .* differ)$", re.MULTILINE)
_SPECIAL_MODE = re.compile(
    r"^(?:new file mode|old mode|index) .*\b(?:120000|160000)\b", re.MULTILINE
)
_DEV_IDENTITY = {
    "GIT_AUTHOR_NAME": "devcontrol",
    "GIT_AUTHOR_EMAIL": "devcontrol@localhost",
    "GIT_COMMITTER_NAME": "devcontrol",
    "GIT_COMMITTER_EMAIL": "devcontrol@localhost",
}


@dataclass
class Revision:
    commit: str
    tree: str
    head_tree: str
    branch: str
    dirty: bool
    changed_files: int
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "commit": self.commit,
            "tree": self.tree,
            "branch": self.branch,
            "dirty": self.dirty,
            "changed_files": self.changed_files,
        }


def _now_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


class GitRepo:
    def __init__(self, path: Path):
        self.path = Path(path)

    def _run(
        self,
        args: list[str],
        *,
        timeout: float = 30.0,
        env: Optional[dict[str, str]] = None,
        check: bool = True,
    ) -> CommandResult:
        result = run_capture(["git", "-C", str(self.path), *args], timeout=timeout, env=env)
        if check and not result.ok:
            raise CommandError(
                f"git {' '.join(args)} failed",
                argv=result.argv,
                returncode=result.returncode,
                stdout_tail=result.stdout[-4000:],
                stderr_tail=result.stderr[-4000:],
                timed_out=result.timed_out,
            )
        return result

    def ensure(self) -> None:
        if not self.path.is_dir():
            raise ConfigError(f"development checkout does not exist: {self.path}")
        result = self._run(["rev-parse", "--is-inside-work-tree"], check=False)
        if not result.ok or result.stdout.strip() != "true":
            raise ConfigError(f"development checkout is not a git worktree: {self.path}")

    def commit(self) -> str:
        return self._run(["rev-parse", "HEAD"]).stdout.strip()

    def tree(self, ref: str = "HEAD") -> str:
        return self._run(["rev-parse", f"{ref}^{{tree}}"]).stdout.strip()

    def branch(self) -> str:
        result = self._run(["symbolic-ref", "--quiet", "--short", "HEAD"], check=False)
        return result.stdout.strip() if result.ok else "HEAD"

    def resolve_ref(self, ref: str) -> str:
        if not ref or ref.startswith("-"):
            raise GuardError(f"invalid base ref: {ref!r}")
        result = self._run(["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
        if not result.ok:
            raise NotFoundError(f"git ref not found: {ref}")
        return result.stdout.strip()

    def status_counts(self) -> int:
        result = self._run(["status", "--porcelain=v1", "-z"], check=False)
        if not result.ok:
            return 0
        return len([item for item in result.stdout.split("\0") if item.strip()])

    def revision(self) -> Revision:
        commit = self.commit()
        head_tree = self.tree("HEAD")
        tree = self.working_tree()
        branch = self.branch()
        dirty = tree != head_tree
        label = commit if not dirty else f"{commit}+dirty.{tree[:12]}"
        return Revision(
            commit=commit,
            tree=tree,
            head_tree=head_tree,
            branch=branch,
            dirty=dirty,
            changed_files=self.status_counts(),
            label=label,
        )

    def working_tree(self) -> str:
        """Return a tree object for the working tree without touching the index.

        This is what makes a dirty revision identifiable: the returned label
        includes this tree id, so an agent can prove which uncommitted content
        the process actually loaded.
        """

        fd, temp_name = tempfile.mkstemp(prefix="devcontrol-index-")
        os.close(fd)
        try:
            os.unlink(temp_name)
            env = {"GIT_INDEX_FILE": temp_name}
            self._run(["read-tree", "HEAD"], env=env, timeout=60)
            self._run(["add", "-A", "--", "."], env=env, timeout=120)
            result = self._run(["write-tree"], env=env, timeout=60)
            return result.stdout.strip()
        finally:
            for candidate in (temp_name, f"{temp_name}.lock"):
                try:
                    os.unlink(candidate)
                except OSError:
                    pass

    def changed_files(self) -> list[str]:
        result = self._run(["diff", "--name-only", "HEAD"], check=False)
        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        untracked = self._run(["ls-files", "--others", "--exclude-standard"], check=False)
        files.extend(line.strip() for line in untracked.stdout.splitlines() if line.strip())
        return sorted(dict.fromkeys(files))

    def diff(self, *, max_bytes: int = 1_000_000) -> str:
        result = self._run(["diff", "--binary", "HEAD"], timeout=60, check=False)
        text = result.stdout
        if len(text) > max_bytes:
            text = text[:max_bytes] + "\n[truncated by devcontrol]\n"
        return text

    def untracked(self) -> list[str]:
        result = self._run(["ls-files", "--others", "--exclude-standard"], check=False)
        return [line.strip() for line in result.stdout.splitlines() if line.strip()]

    def worktrees(self) -> list[dict[str, Any]]:
        result = self._run(["worktree", "list", "--porcelain"], check=False)
        if not result.ok:
            return []
        records: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for line in result.stdout.splitlines() + [""]:
            if not line.strip():
                if current:
                    records.append(current)
                    current = {}
                continue
            key, _, value = line.partition(" ")
            if key == "worktree":
                current["path"] = value
            elif key == "HEAD":
                current["commit"] = value
            elif key == "branch":
                current["branch"] = value.removeprefix("refs/heads/")
            elif key in {"detached", "bare", "prunable", "locked"}:
                current[key] = True
            elif key == "locked" and value:
                current["locked"] = value
        return records

    def add_worktree(self, path: Path, *, name: str, base: str = "HEAD") -> dict[str, Any]:
        if not _valid_worktree_name(name):
            raise GuardError(f"invalid worktree name: {name!r}")
        if path.exists():
            raise GuardError(f"worktree path already exists: {path}")
        resolved = self.resolve_ref(base)
        branch = f"agent/{name}"
        existing = self._run(
            ["show-ref", "--verify", "--quiet", f"refs/heads/{branch}"], check=False
        )
        if existing.ok:
            raise GuardError(f"branch already exists: {branch}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._run(["worktree", "add", "-b", branch, str(path), resolved], timeout=120)
        except Exception:
            # A failed add can leave the new branch behind; remove it so a
            # corrected retry is not blocked by a half-created worktree.
            self._run(["branch", "-D", branch], check=False)
            self._run(["worktree", "prune"], check=False)
            raise
        return {"name": name, "path": str(path), "branch": branch, "base_commit": resolved}

    def remove_worktree(self, path: Path, *, force: bool = False) -> None:
        args = ["worktree", "remove"]
        if force:
            args.append("--force")
        args.append(str(path))
        self._run(args, timeout=120)

    def prune_worktrees(self) -> None:
        self._run(["worktree", "prune"], check=False)

    def checkpoint(
        self, *, worktree: str, label: str = "", operation_id: str = ""
    ) -> dict[str, Any]:
        revision = self.revision()
        checkpoint_id = f"cp-{_now_id()}"
        message = f"devcontrol checkpoint {checkpoint_id}"
        if label:
            message += f": {label.strip()[:120]}"
        env = dict(_DEV_IDENTITY)
        commit = self._run(
            ["commit-tree", revision.tree, "-p", revision.commit, "-m", message],
            env=env,
            timeout=60,
        ).stdout.strip()
        if not commit:
            raise CommandError("git commit-tree returned no commit")
        ref = CHECKPOINT_REF_PREFIX + checkpoint_id
        self._run(["update-ref", ref, commit], timeout=30)
        return {
            "id": checkpoint_id,
            "ref": ref,
            "commit": commit,
            "tree": revision.tree,
            "head_commit": revision.commit,
            "revision": revision.label,
            "worktree": worktree,
            "label": label,
            "operation_id": operation_id,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        }

    def checkpoint_commit(self, checkpoint_id: str) -> str:
        ref = CHECKPOINT_REF_PREFIX + checkpoint_id
        result = self._run(["rev-parse", "--verify", f"{ref}^{{commit}}"], check=False)
        if not result.ok:
            raise NotFoundError(f"checkpoint not found: {checkpoint_id}")
        return result.stdout.strip()

    def restore_to_commit(self, commit: str) -> None:
        """Restore index and working tree to a commit without moving HEAD.

        This keeps the branch pointer intact. The subsequent clean removes
        files created after the checkpoint; ignored paths (data, node_modules)
        are deliberately preserved. Every non-ignored file that existed at
        checkpoint time is represented in the checkpoint tree, so it comes
        back via ``read-tree -u`` and is not removed by ``clean``.
        """

        self.resolve_ref(commit)
        self._run(["read-tree", "--reset", "-u", commit], timeout=120)
        self._run(["clean", "-fd"], timeout=120)

    def _patch_bytes(self, patch_text: str) -> bytes:
        data = patch_text.encode("utf-8")
        if not data.strip():
            raise GuardError("patch is empty")
        if len(data) > MAX_PATCH_BYTES:
            raise GuardError(f"patch exceeds {MAX_PATCH_BYTES} bytes")
        if _BINARY_MARKER.search(patch_text):
            raise GuardError("binary patches are not accepted")
        if _SPECIAL_MODE.search(patch_text):
            raise GuardError("symlink and submodule patches are not accepted")
        paths = parse_patch_paths(patch_text)
        if not paths:
            raise GuardError("patch does not declare any file paths")
        for path in paths:
            _validate_patch_path(path)
        return data

    def check_unified_patch(self, patch_text: str) -> list[str]:
        data = self._patch_bytes(patch_text)
        paths = parse_patch_paths(patch_text)
        fd, temp_name = tempfile.mkstemp(prefix="devcontrol-patch-", suffix=".diff")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            check = self._run(["apply", "--check", "--whitespace=nowarn", temp_name], check=False)
            if not check.ok:
                raise GuardError(
                    "patch does not apply cleanly",
                    detail={"stderr": check.stderr[-4000:], "stdout": check.stdout[-4000:]},
                )
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return paths

    def apply_unified_patch(self, patch_text: str) -> list[str]:
        self.check_unified_patch(patch_text)
        data = self._patch_bytes(patch_text)
        fd, temp_name = tempfile.mkstemp(prefix="devcontrol-patch-", suffix=".diff")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            applied = self._run(["apply", "--whitespace=nowarn", temp_name], timeout=60)
            if not applied.ok:
                raise CommandError(
                    "git apply failed after a successful check",
                    argv=applied.argv,
                    returncode=applied.returncode,
                    stderr_tail=applied.stderr[-4000:],
                )
        finally:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
        return self.changed_files()


def _valid_worktree_name(name: str) -> bool:
    from .config import worktree_name_is_valid

    return worktree_name_is_valid(name)


def _clean_patch_path(candidate: str) -> str:
    value = candidate.strip()
    if not value or value == "/dev/null":
        return ""
    if value.startswith('"'):
        try:
            value = shlex.split(value)[0]
        except (ValueError, IndexError):
            value = value.strip('"')
    if value.startswith("a/") or value.startswith("b/"):
        value = value[2:]
    return value.split("\t", 1)[0].strip()


def parse_patch_paths(patch_text: str) -> list[str]:
    paths: list[str] = []
    for line in patch_text.splitlines():
        candidate = ""
        if line.startswith("diff --git "):
            try:
                parts = shlex.split(line)
            except ValueError:
                parts = line.split()
            if len(parts) >= 4:
                paths.extend(
                    filter(None, (_clean_patch_path(parts[2]), _clean_patch_path(parts[3])))
                )
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            candidate = line[4:].strip()
        elif line.startswith("rename from ") or line.startswith("rename to "):
            candidate = line.split(" ", 2)[2].strip()
        elif line.startswith("copy from ") or line.startswith("copy to "):
            candidate = line.split(" ", 2)[2].strip()
        cleaned = _clean_patch_path(candidate)
        if cleaned:
            paths.append(cleaned)
    return sorted(dict.fromkeys(paths))


def _validate_patch_path(path: str) -> None:
    value = path.strip()
    if not value or "\x00" in value:
        raise GuardError("patch contains an empty or NUL path")
    if value.startswith("/") or value.startswith("\\"):
        raise GuardError(f"absolute patch path is not allowed: {value}")
    if re.match(r"^[A-Za-z]:", value):
        raise GuardError(f"drive-qualified patch path is not allowed: {value}")
    parts = [part for part in re.split(r"[\\/]+", value) if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise GuardError(f"parent-directory patch path is not allowed: {value}")
    if any(part == ".git" for part in parts):
        raise GuardError(f"patch path targets .git: {value}")


def checkpoint_age(checkpoint: dict[str, Any]) -> float:
    try:
        created = datetime.fromisoformat(str(checkpoint.get("created_at"))).timestamp()
        return max(0.0, time.time() - created)
    except Exception:
        return 0.0


def write_patch_file(path: Path, text: str) -> None:
    atomic_write_text(path, text, mode=0o600)
