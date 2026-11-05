"""Cross-platform advisory locks for the instance guard and state writes."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from .errors import BusyError

if os.name == "nt":  # pragma: no cover - exercised on Windows hosts only
    import msvcrt
else:
    import fcntl


def atomic_write_text(path: Path, text: str, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            os.chmod(path, mode)
        except OSError:
            pass
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def atomic_write_json(path: Path, value: Any, *, mode: int = 0o600) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n", mode=mode)


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


class FileLock:
    """A small advisory file lock.

    ``blocking=False`` is used for the instance guard so an agent gets an
    immediate typed busy response rather than queuing behind another agent.
    Journal/state writes use the blocking form.
    """

    def __init__(self, path: Path, *, blocking: bool = True):
        self.path = Path(path)
        self.blocking = blocking
        self._fd: Optional[int] = None

    def _try_acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.name == "nt":  # pragma: no cover - Windows host
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                mode = msvcrt.LK_LOCK if self.blocking else msvcrt.LK_NBLCK
                msvcrt.locking(fd, mode, 1)
            else:
                flags = fcntl.LOCK_EX if self.blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
                fcntl.flock(fd, flags)
        except OSError:
            os.close(fd)
            return False
        self._fd = fd
        return True

    def acquire(self, *, wait_seconds: float = 0.0) -> None:
        deadline = time.monotonic() + max(0.0, wait_seconds)
        while True:
            if self._try_acquire():
                return
            if time.monotonic() >= deadline:
                raise BusyError(f"lock is held: {self.path}")
            time.sleep(0.1)

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            if os.name == "nt":  # pragma: no cover - Windows host
                os.lseek(self._fd, 0, os.SEEK_SET)
                msvcrt.locking(self._fd, msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()


class InstanceGuard:
    """Exclusive ownership of operations that can affect the running stack."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)
        self.lock_path = self.state_dir / "locks" / "instance.lock"
        self.owner_path = self.state_dir / "locks" / "instance.owner.json"
        self._lock: Optional[FileLock] = None

    def owner(self) -> dict[str, Any]:
        value = read_json(self.owner_path, {})
        return value if isinstance(value, dict) else {}

    def acquire(
        self,
        *,
        operation_id: str,
        action: str,
        worktree: str,
        actor: str,
    ) -> None:
        if self._lock is not None:
            raise BusyError(
                "another operation is active on the shared development instance",
                holder=self.owner(),
            )
        lock = FileLock(self.lock_path, blocking=False)
        try:
            lock.acquire()
        except BusyError:
            raise BusyError(
                "another operation is active on the shared development instance",
                holder=self.owner(),
            ) from None
        try:
            atomic_write_json(
                self.owner_path,
                {
                    "operation_id": operation_id,
                    "action": action,
                    "worktree": worktree,
                    "actor": actor,
                    "pid": os.getpid(),
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                },
            )
        except OSError:
            lock.release()
            raise
        self._lock = lock

    def release(self) -> None:
        if self._lock is None:
            return
        try:
            try:
                self.owner_path.unlink()
            except OSError:
                pass
        finally:
            self._lock.release()
            self._lock = None
