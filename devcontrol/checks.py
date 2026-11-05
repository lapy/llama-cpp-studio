"""Allowlisted checks that run against the active development stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .compose import Compose
from .errors import GuardError
from .redact import redact_text
from .runtime import CommandResult


@dataclass(frozen=True)
class Check:
    name: str
    service: str
    argv: tuple[str, ...]
    timeout: float = 1800.0
    description: str = ""


CHECKS: dict[str, Check] = {
    "backend-tests": Check(
        "backend-tests",
        "backend",
        ("python", "-m", "pytest", "backend/tests", "-q", "--tb=short"),
        2400.0,
        "Full backend pytest suite",
    ),
    "backend-tests-fast": Check(
        "backend-tests-fast",
        "backend",
        ("python", "-m", "pytest", "backend/tests", "-q", "--tb=short", "-x"),
        1200.0,
        "Backend pytest suite, stop on first failure",
    ),
    "ruff": Check(
        "ruff",
        "backend",
        ("python", "-m", "ruff", "check", "backend", "scripts", "devcontrol"),
        300.0,
        "Ruff correctness checks",
    ),
    "frontend-tests": Check(
        "frontend-tests",
        "frontend",
        ("npm", "run", "test:frontend"),
        1200.0,
        "Vitest frontend unit suite",
    ),
    "frontend-lint": Check(
        "frontend-lint",
        "frontend",
        ("npm", "run", "lint:frontend"),
        600.0,
        "ESLint frontend sources",
    ),
    "typecheck": Check(
        "typecheck",
        "frontend",
        ("npm", "run", "typecheck:api"),
        600.0,
        "TypeScript API boundary check",
    ),
    "style": Check(
        "style",
        "frontend",
        ("npm", "run", "style:check"),
        600.0,
        "Stylelint",
    ),
    "format": Check(
        "format",
        "frontend",
        ("npm", "run", "format:check"),
        600.0,
        "Prettier check for the declared scope",
    ),
    "openapi": Check(
        "openapi",
        "backend",
        ("npm", "run", "openapi:check"),
        300.0,
        "Checked-in OpenAPI schema is current",
    ),
    "lint": Check(
        "lint",
        "backend",
        ("npm", "run", "lint"),
        1800.0,
        "The repository's aggregate static-check script",
    ),
}

ALIASES = {
    "backend": "backend-tests",
    "backend-test": "backend-tests",
    "pytest": "backend-tests",
    "frontend": "frontend-tests",
    "frontend-test": "frontend-tests",
    "vitest": "frontend-tests",
    "eslint": "frontend-lint",
    "tsc": "typecheck",
    "stylelint": "style",
    "prettier": "format",
    "backend-lint": "ruff",
}


def expand_checks(names: Iterable[str]) -> list[Check]:
    result: list[Check] = []
    seen = set()
    for raw in names:
        name = str(raw).strip()
        name = ALIASES.get(name, name)
        if name == "all":
            candidates = ["ruff", "backend-tests", "frontend-tests", "frontend-lint", "typecheck"]
        else:
            candidates = [name]
        for candidate in candidates:
            check = CHECKS.get(candidate)
            if check is None:
                raise GuardError(
                    f"unknown check: {raw!r}; allowed checks: {', '.join(sorted(CHECKS))}"
                )
            if check.name not in seen:
                result.append(check)
                seen.add(check.name)
    return result


def validate_test_targets(repo: Path, targets: Iterable[str]) -> list[str]:
    result = []
    root = (Path(repo) / "backend" / "tests").resolve()
    for raw in targets:
        relative = str(raw).replace("\\", "/").strip()
        if not relative:
            continue
        candidate = (Path(repo) / relative).resolve()
        if not candidate.is_relative_to(root):
            raise GuardError(f"test target must be under backend/tests: {raw}")
        if not candidate.exists():
            raise GuardError(f"test target does not exist in the active checkout: {raw}")
        result.append(relative)
    return result


def validate_frontend_targets(repo: Path, targets: Iterable[str]) -> list[str]:
    result = []
    roots = [
        (Path(repo) / "frontend" / "src").resolve(),
        (Path(repo) / "frontend" / "e2e").resolve(),
    ]
    for raw in targets:
        relative = str(raw).replace("\\", "/").strip()
        if not relative:
            continue
        candidate = (Path(repo) / relative).resolve()
        if not any(candidate.is_relative_to(root) for root in roots):
            raise GuardError(f"frontend target must be under frontend/src or frontend/e2e: {raw}")
        if not candidate.exists():
            raise GuardError(f"frontend target does not exist in the active checkout: {raw}")
        result.append(relative)
    return result


def _command_for(
    check: Check, *, backend_targets: list[str], frontend_targets: list[str]
) -> list[str]:
    argv = list(check.argv)
    if check.name in {"backend-tests", "backend-tests-fast"} and backend_targets:
        # Keep pytest flags, replace the default suite path with explicit targets.
        argv = ["python", "-m", "pytest", *backend_targets, "-q", "--tb=short"]
        if check.name == "backend-tests-fast":
            argv.append("-x")
    elif check.name == "frontend-tests" and frontend_targets:
        argv = ["npm", "run", "test:frontend", "--", *frontend_targets]
    return argv


def run_checks(
    compose: Compose,
    *,
    names: Iterable[str],
    repo: Path,
    tests: Optional[Iterable[str]] = None,
    frontend_targets: Optional[Iterable[str]] = None,
    timeout_seconds: Optional[float] = None,
    fail_fast: bool = True,
) -> dict[str, Any]:
    checks = expand_checks(names)
    backend_targets = validate_test_targets(repo, tests or [])
    frontend_paths = validate_frontend_targets(repo, frontend_targets or [])
    rows: list[dict[str, Any]] = []
    any_failed = False
    for check in checks:
        argv = _command_for(
            check,
            backend_targets=backend_targets,
            frontend_targets=frontend_paths,
        )
        timeout = float(timeout_seconds or check.timeout)
        result: CommandResult = compose.exec(
            check.service,
            argv,
            timeout=timeout,
            user=f"{compose.config.uid}:{compose.config.gid}",
        )
        tail = redact_text(result.tail(20_000))
        row = {
            "name": check.name,
            "service": check.service,
            "command": argv,
            "ok": result.ok,
            "returncode": result.returncode,
            "duration_seconds": round(result.duration_seconds, 2),
            "timed_out": result.timed_out,
            "output_tail": tail,
        }
        rows.append(row)
        if not result.ok:
            any_failed = True
            if fail_fast:
                break
    return {
        "ok": not any_failed,
        "checks": rows,
        "failed": [row["name"] for row in rows if not row["ok"]],
        "requested": [check.name for check in checks],
    }
