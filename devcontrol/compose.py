"""Docker Compose operations for the isolated development stack."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

from .config import Config
from .errors import CommandError, GuardError
from .http_client import request_json
from .runtime import CommandResult, run_capture, which

DEV_LABEL = "com.studio.dev"
DEV_PROJECT_LABEL = "com.studio.dev.project"
BASE_LICENSE_PLACEHOLDER = "dev-placeholder-not-for-distribution"


class Compose:
    def __init__(self, config: Config):
        self.config = config
        self.dir = config.stack_dir

    def _base(self) -> list[str]:
        command = ["docker", "compose"]
        if self.config.env_path.is_file():
            command.extend(["--env-file", str(self.config.env_path)])
        command.extend(
            [
                "--project-name",
                self.config.compose_project,
                "-f",
                str(self.config.compose_file),
            ]
        )
        if self.config.gpu:
            command.extend(["-f", str(self.config.gpu_compose_file)])
        return command

    def _run(
        self,
        args: Sequence[str],
        *,
        timeout: float = 120.0,
        check: bool = True,
    ) -> CommandResult:
        if not which("docker"):
            raise GuardError("docker is not installed on this host")
        # Run from the control state directory, not the checkout: Compose also
        # loads a .env from its working directory, and the live instance's .env
        # must never participate in a development command.
        self.config.state_home.mkdir(parents=True, exist_ok=True)
        result = run_capture(
            [*self._base(), *args],
            cwd=self.config.state_home,
            env=self.config.compose_env(),
            timeout=timeout,
        )
        if check and not result.ok:
            raise CommandError(
                f"docker compose {' '.join(str(item) for item in args)} failed",
                argv=result.argv,
                returncode=result.returncode,
                stdout_tail=result.stdout[-6000:],
                stderr_tail=result.stderr[-6000:],
                timed_out=result.timed_out,
            )
        return result

    def ps(self) -> list[dict[str, Any]]:
        result = self._run(["ps", "--all", "--format", "json"], timeout=30, check=False)
        if not result.ok or not result.stdout.strip():
            return []
        text = result.stdout.strip()
        rows: Any
        try:
            rows = json.loads(text)
        except ValueError:
            rows = []
            for line in text.splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
        if isinstance(rows, dict):
            rows = [rows]
        return [row for row in rows if isinstance(row, dict)]

    def service_rows(self) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for row in self.ps():
            service = str(row.get("Service") or row.get("service") or "")
            if service:
                rows[service] = row
        return rows

    def running(self, service: str) -> bool:
        row = self.service_rows().get(service)
        if not row:
            return False
        state = str(row.get("State") or row.get("state") or "").lower()
        return state == "running"

    def any_running(self) -> bool:
        rows = self.service_rows()
        return any(
            str(row.get("State") or row.get("state") or "").lower() == "running"
            for row in rows.values()
        )

    def health(self, service: str) -> str:
        row = self.service_rows().get(service) or {}
        return str(row.get("Health") or row.get("health") or "")

    def up(
        self,
        *,
        services: Optional[Sequence[str]] = None,
        build: bool = False,
        force_recreate: bool = False,
        no_deps: bool = False,
    ) -> CommandResult:
        args = ["up", "-d", "--remove-orphans"]
        if build:
            args.append("--build")
        if force_recreate:
            args.append("--force-recreate")
        if no_deps:
            args.append("--no-deps")
        args.extend(services or [])
        return self._run(args, timeout=1800)

    def down(self, *, volumes: bool = False) -> CommandResult:
        args = ["down", "--remove-orphans"]
        if volumes:
            args.append("--volumes")
        return self._run(args, timeout=300)

    def restart(self, service: str) -> CommandResult:
        return self._run(["restart", service], timeout=180)

    def run_init(self) -> CommandResult:
        return self._run(["run", "--rm", "--no-deps", "init"], timeout=1800)

    def exec(
        self,
        service: str,
        argv: Sequence[str],
        *,
        timeout: float = 1800.0,
        user: Optional[str] = None,
    ) -> CommandResult:
        args = ["exec", "-T"]
        if user:
            args.extend(["--user", user])
        args.append(service)
        args.extend(str(item) for item in argv)
        return self._run(args, timeout=timeout, check=False)

    def logs(self, service: str, *, tail: int = 200, since: str = "") -> CommandResult:
        args = ["logs", "--no-color", "--tail", str(max(1, min(tail, 5000)))]
        if since:
            args.extend(["--since", since])
        args.append(service)
        return self._run(args, timeout=60, check=False)

    def image_exists(self, image: str) -> bool:
        if not which("docker"):
            return False
        result = run_capture(["docker", "image", "inspect", image], timeout=30)
        return result.ok

    def build(self, *, no_cache: bool = False) -> dict[str, Any]:
        if not which("docker"):
            raise GuardError("docker is not installed on this host")
        license_value = self.config.primeui_license or BASE_LICENSE_PLACEHOLDER
        base_args = [
            "docker",
            "build",
            "-f",
            str(self.config.repo / "Dockerfile"),
            "--target",
            "runtime",
            "-t",
            self.config.base_image,
            "--build-arg",
            f"VITE_PRIMEUI_LICENSE={license_value}",
        ]
        if no_cache:
            base_args.append("--no-cache")
        base_args.append(str(self.config.repo))
        base = run_capture(base_args, cwd=self.config.repo, timeout=7200)
        if not base.ok:
            raise CommandError(
                "failed to build the base development image",
                argv=base.argv,
                returncode=base.returncode,
                stdout_tail=base.stdout[-6000:],
                stderr_tail=base.stderr[-6000:],
                timed_out=base.timed_out,
            )
        overlay_args = ["build"]
        if no_cache:
            overlay_args.append("--no-cache")
        overlay = self._run(overlay_args, timeout=3600)
        if not overlay.ok:
            raise CommandError(
                "failed to build the dev-tools image layer",
                argv=overlay.argv,
                returncode=overlay.returncode,
                stdout_tail=overlay.stdout[-6000:],
                stderr_tail=overlay.stderr[-6000:],
            )
        return {
            "base_image": self.config.base_image,
            "image": self.config.image,
            "base": tail_result(base),
            "overlay": tail_result(overlay),
        }

    def ensure_ready(self) -> None:
        if not self.config.compose_file.is_file():
            raise GuardError(f"compose file is missing: {self.config.compose_file}")
        if self.config.gpu and not self.config.gpu_compose_file.is_file():
            raise GuardError(f"GPU compose file is missing: {self.config.gpu_compose_file}")
        if not self.image_exists(self.config.image):
            raise GuardError(
                f"development image is missing: {self.config.image}; run 'devcontrol stack build'"
            )
        if self.config.gpu:
            gpu_check = run_capture(["nvidia-smi", "-L"], timeout=15)
            if not gpu_check.ok:
                raise GuardError(
                    "DEV_GPU is set but nvidia-smi is unavailable on this host; "
                    "install the NVIDIA container toolkit or clear DEV_GPU"
                )

    def wait_http(
        self, path: str, *, timeout: float = 60.0, require_auth: bool = False
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{self.config.backend_port}{path}"
        last = None
        while time.monotonic() < deadline:
            result = request_json(
                url,
                token=self.config.api_token if require_auth else "",
                timeout=3.0,
            )
            last = result
            if result.ok:
                return {"ok": True, "status": result.status, "url": url, "data": result.data}
            time.sleep(0.5)
        return {
            "ok": False,
            "url": url,
            "status": getattr(last, "status", 0),
            "error": getattr(last, "error", "timeout"),
        }

    def wait_ready(self, *, timeout: float = 90.0) -> dict[str, Any]:
        live = self.wait_http("/api/live", timeout=timeout)
        if not live.get("ok"):
            return {"ok": False, "live": live, "ready": None}
        ready = self.wait_http("/api/ready", timeout=min(timeout, 60.0))
        return {"ok": bool(ready.get("ok")), "live": live, "ready": ready}

    def summary(self) -> dict[str, Any]:
        rows = self.service_rows()
        return {
            "project": self.config.compose_project,
            "services": {
                name: {
                    "state": row.get("State") or row.get("state"),
                    "health": row.get("Health") or row.get("health"),
                    "image": row.get("Image") or row.get("image"),
                    "ports": row.get("Publishers") or row.get("Ports") or [],
                }
                for name, row in sorted(rows.items())
            },
            "running": self.any_running(),
        }


def tail_result(result: CommandResult, limit: int = 6000) -> dict[str, Any]:
    return {
        "ok": result.ok,
        "returncode": result.returncode,
        "duration_seconds": round(result.duration_seconds, 2),
        "output_tail": result.tail(limit),
        "timed_out": result.timed_out,
    }


def compose_binary_available() -> bool:
    return bool(which("docker"))
