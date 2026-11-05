"""Small subprocess helper with bounded output and process-tree termination."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    env: dict[str, str] = field(default_factory=dict, repr=False)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out

    def tail(self, limit: int = 12_000) -> str:
        combined = "\n".join(part for part in (self.stdout.rstrip(), self.stderr.rstrip()) if part)
        return combined[-limit:] if limit > 0 else combined


def _read_tail(path: Path, limit: int) -> tuple[str, bool]:
    try:
        with open(path, "rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            truncated = size > limit
            handle.seek(max(0, size - limit))
            data = handle.read()
    except OSError:
        return "", False
    return data.decode("utf-8", errors="replace"), truncated


def _kill_tree(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
    except (OSError, ProcessLookupError, subprocess.TimeoutExpired):
        try:
            process.kill()
        except OSError:
            pass


def run_capture(
    argv: Sequence[str],
    *,
    cwd: Optional[Path | str] = None,
    env: Optional[dict[str, str]] = None,
    timeout: float = 60.0,
    max_output_bytes: int = 240_000,
    check: bool = False,
) -> CommandResult:
    """Run a command without a shell and retain only bounded tail output.

    Output goes to temporary files rather than pipes so a chatty build or test
    run cannot deadlock the parent on a full pipe buffer.
    """

    args = [str(item) for item in argv]
    merged_env = dict(os.environ)
    if env:
        merged_env.update({str(key): str(value) for key, value in env.items()})
    started = time.monotonic()
    tmp = Path(tempfile.gettempdir()) / f"devcontrol-{uuid.uuid4().hex[:12]}"
    # ``tempfile.mkdtemp`` creates mode 0700 on Windows, which some restricted
    # execution environments (including the development harness sandbox)
    # cannot reopen. A plain directory plus files created with os.open is
    # portable and still private on POSIX through the umask.
    tmp.mkdir()
    if os.name != "nt":
        os.chmod(tmp, 0o700)
    out_path = tmp / "stdout"
    err_path = tmp / "stderr"
    process: Optional[subprocess.Popen] = None
    timed_out = False
    try:
        with open(out_path, "wb") as out, open(err_path, "wb") as err:
            popen_kwargs: dict = {
                "cwd": str(cwd) if cwd else None,
                "env": merged_env,
                "stdin": subprocess.DEVNULL,
                "stdout": out,
                "stderr": err,
            }
            if os.name == "nt":
                popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                popen_kwargs["start_new_session"] = True
            process = subprocess.Popen(args, **popen_kwargs)
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _kill_tree(process)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
        stdout, stdout_truncated = _read_tail(out_path, max_output_bytes)
        stderr, stderr_truncated = _read_tail(err_path, max_output_bytes)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    duration = time.monotonic() - started
    result = CommandResult(
        argv=args,
        returncode=int(process.returncode if process.returncode is not None else -1),
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        timed_out=timed_out,
        stdout_truncated=stdout_truncated,
        stderr_truncated=stderr_truncated,
    )
    if check and not result.ok:
        from .errors import CommandError

        raise CommandError(
            f"command failed with exit code {result.returncode}: {args[0]}",
            argv=args,
            returncode=result.returncode,
            stdout_tail=stdout[-8000:],
            stderr_tail=stderr[-8000:],
            timed_out=timed_out,
        )
    return result


def which(name: str) -> Optional[str]:
    return shutil.which(name)
