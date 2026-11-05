"""In-container process runtime for the self-contained development image.

This is the Docker-socket-free counterpart to :class:`devcontrol.compose.Compose`.
It talks to supervisord over a Unix socket, runs checks directly in the active
checkout, and reads the service log files. It never invokes Docker and therefore
cannot affect anything outside the container.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import Config
from .errors import CommandError, GuardError, NotFoundError
from .locks import atomic_write_text
from .runtime import CommandResult, run_capture

SERVICES = ("backend", "frontend")


class LocalRuntime:
    def __init__(self, config: Config):
        self.config = config
        self.active_link = Path(os.getenv("DEV_ACTIVE_LINK", "/workspace/active"))
        self.log_dir = Path(os.getenv("DEV_LOG_DIR", "/state/logs"))
        self.supervisor_config = Path(
            os.getenv("DEV_SUPERVISOR_CONFIG", "/etc/supervisor/studio.conf")
        )
        self.supervisorctl = (
            os.getenv("DEV_SUPERVISORCTL")
            or shutil.which("supervisorctl")
            or "/opt/venv/bin/supervisorctl"
        )

    # ------------------------------------------------------------- status
    def _ctl(self, *args: str, timeout: float = 30.0) -> CommandResult:
        return run_capture(
            [self.supervisorctl, "-c", str(self.supervisor_config), *args],
            timeout=timeout,
        )

    def _status(self) -> dict[str, str]:
        result = self._ctl("status", timeout=15)
        states: dict[str, str] = {}
        if not result.ok:
            return states
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                states[parts[0]] = parts[1].lower()
        return states

    def service_rows(self) -> dict[str, dict[str, Any]]:
        states = self._status()
        return {
            name: {
                "Service": name,
                "State": "running" if state == "running" else "stopped",
                "Health": "healthy" if state == "running" else "unhealthy",
                "Image": "studio-standalone",
            }
            for name, state in states.items()
        }

    def ps(self) -> list[dict[str, Any]]:
        return [
            {
                "Service": row["Service"],
                "State": row["State"],
                "Health": row["Health"],
                "Image": row["Image"],
            }
            for row in self.service_rows().values()
        ]

    def summary(self) -> dict[str, Any]:
        rows = self.service_rows()
        return {
            "project": "studio-standalone",
            "runtime": "local",
            "services": {
                name: {
                    "state": row["State"],
                    "health": row["Health"],
                    "image": row["Image"],
                    "ports": [],
                }
                for name, row in sorted(rows.items())
            },
            "running": any(row["State"] == "running" for row in rows.values()),
        }

    def any_running(self) -> bool:
        return any(self.running(name) for name in SERVICES)

    def running(self, service: str) -> bool:
        row = self.service_rows().get(service)
        return bool(row and row["State"] == "running")

    def health(self, service: str) -> str:
        row = self.service_rows().get(service) or {}
        return str(row.get("Health") or "")

    # -------------------------------------------------------------- control
    def _sync_active(self) -> None:
        target = self.config.active_path
        if not target.is_dir():
            raise NotFoundError(f"active checkout does not exist: {target}")
        link = self.active_link
        link.parent.mkdir(parents=True, exist_ok=True)
        temporary = link.with_name(f".{link.name}.{os.getpid()}")
        try:
            temporary.unlink()
        except OSError:
            pass
        os.symlink(target.resolve(), temporary)
        os.replace(temporary, link)

    def set_active(self, name: str) -> None:
        old = self.config.active
        self.config.active = name
        try:
            self._sync_active()
        except Exception:
            self.config.active = old
            raise

    def restart(self, service: str) -> CommandResult:
        if service not in SERVICES:
            raise GuardError(f"unknown local service: {service}")
        result = self._ctl("restart", service, timeout=120)
        if not result.ok:
            # A stopped program cannot be restarted; start it instead.
            result = self._ctl("start", service, timeout=120)
        if not result.ok:
            raise CommandError(
                f"could not restart {service}",
                argv=result.argv,
                returncode=result.returncode,
                stdout_tail=result.stdout[-4000:],
                stderr_tail=result.stderr[-4000:],
            )
        return result

    def up(
        self,
        *,
        services: Optional[Sequence[str]] = None,
        build: bool = False,
        force_recreate: bool = False,
        no_deps: bool = False,
    ) -> CommandResult:
        del build, no_deps
        self._sync_active()
        selected = list(services or SERVICES)
        running = self._status()
        if any(running.get(name) == "running" for name in selected):
            result = self._ctl("restart", *selected, timeout=180)
        else:
            result = self._ctl("start", *selected, timeout=180)
        if not result.ok:
            result = self._ctl("start", *selected, timeout=180)
        if not result.ok:
            raise CommandError(
                "could not start local services",
                argv=result.argv,
                returncode=result.returncode,
                stdout_tail=result.stdout[-4000:],
                stderr_tail=result.stderr[-4000:],
            )
        return result

    def down(self, *, volumes: bool = False) -> CommandResult:
        del volumes
        result = self._ctl("stop", *SERVICES, timeout=60)
        if not result.ok:
            # Stopping already-stopped programs is not a useful failure.
            return result
        return result

    def run_init(self) -> CommandResult:
        self._sync_active()
        self._link_node_modules()
        return CommandResult(
            argv=[], returncode=0, stdout="local workspace ready\n", stderr="", duration_seconds=0.0
        )

    def build(self, *, no_cache: bool = False) -> dict[str, Any]:
        del no_cache
        self.run_init()
        return {
            "image": "studio-standalone",
            "runtime": "local",
            "ok": True,
            "detail": "dependencies are built into the standalone image",
        }

    def ensure_ready(self) -> None:
        if not self.config.active_path.is_dir():
            raise GuardError(f"active checkout does not exist: {self.config.active_path}")
        if not (self.config.active_path / ".git").exists():
            raise GuardError(f"active checkout is not a git repository: {self.config.active_path}")

    def _link_node_modules(self) -> None:
        shared = Path("/workspace/node_modules")
        if not shared.is_dir():
            return
        for root in [self.config.active_path, *self._worktree_paths()]:
            if not root.is_dir() or (root / "node_modules").exists():
                continue
            try:
                os.symlink(shared, root / "node_modules")
            except OSError:
                pass

    def _worktree_paths(self) -> list[Path]:
        if not self.config.worktrees.is_dir():
            return []
        return [path for path in self.config.worktrees.iterdir() if path.is_dir()]

    # ---------------------------------------------------------------- exec
    def exec(
        self,
        service: str,
        argv: Sequence[str],
        *,
        timeout: float = 1800.0,
        user: Optional[str] = None,
    ) -> CommandResult:
        del user
        if service not in SERVICES:
            raise GuardError(f"unknown local service: {service}")
        cwd = self.config.active_path
        if not cwd.is_dir():
            raise NotFoundError(f"active checkout does not exist: {cwd}")
        env = {
            "PYTHONPATH": str(cwd),
            "STUDIO_DATA_DIR": str(self.config.data),
            "HOME": "/workspace",
            "PATH": "/opt/venv/bin:/usr/local/bin:/usr/bin:/bin",
        }
        return run_capture([str(item) for item in argv], cwd=cwd, env=env, timeout=timeout)

    # ---------------------------------------------------------------- logs
    def logs(self, service: str, *, tail: int = 200, since: str = "") -> CommandResult:
        del since
        if service not in SERVICES:
            raise GuardError(f"unknown local service: {service}")
        path = self.log_dir / f"{service}.log"
        if not path.is_file():
            return CommandResult(argv=[], returncode=0, stdout="", stderr="", duration_seconds=0.0)
        try:
            size = path.stat().st_size
            with open(path, "rb") as handle:
                handle.seek(max(0, size - 1_000_000))
                text = handle.read().decode("utf-8", errors="replace")
        except OSError as exc:
            return CommandResult(
                argv=[], returncode=1, stdout="", stderr=str(exc), duration_seconds=0.0
            )
        lines = text.splitlines()[-max(1, min(tail, 5000)) :]
        return CommandResult(
            argv=[], returncode=0, stdout="\n".join(lines), stderr="", duration_seconds=0.0
        )


def write_local_runtime_files(config: Config) -> None:
    """Write a small state marker useful to diagnostics (not used by control)."""

    atomic_write_text(
        config.state_home / "runtime",
        "local\n",
        mode=0o600,
    )
