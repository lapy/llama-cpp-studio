"""Command-line surface for the development control service.

Every command prints one JSON object. Agents should pass ``--json`` for clarity
even though it is the default. Errors are JSON too and carry an ``operation_id``
when the failure happened after the control service was initialized.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

from . import __version__
from .bootstrap import bootstrap
from .config import Config
from .errors import DevControlError, GuardError
from .journal import new_operation_id
from .service import DevControl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="devcontrol",
        description="Control the isolated Studio development stack.",
    )
    parser.add_argument("--version", action="version", version=f"devcontrol {__version__}")
    parser.add_argument(
        "--state-home",
        help="control state directory (default: ~/.local/state/llama-cpp-studio-dev)",
    )
    parser.add_argument("--repo", help="development checkout")
    parser.add_argument("--data", help="development data directory")
    parser.add_argument("--worktrees", help="worktree root")
    parser.add_argument("--shared-models", help="read-only shared model root")
    parser.add_argument("--image", help="dev image tag")
    parser.add_argument("--backend-port", type=int)
    parser.add_argument("--frontend-port", type=int)
    parser.add_argument("--proxy-port", type=int)
    parser.add_argument("--gpu", help="comma-separated GPU ids; empty disables GPU")
    parser.add_argument("--actor", default="", help="actor recorded in the operation journal")
    parser.add_argument("--json", action="store_true", help="emit JSON (the default)")
    parser.add_argument("--pretty", action="store_true", help="indent JSON output")
    sub = parser.add_subparsers(dest="command", required=True)

    boot = sub.add_parser("bootstrap", help="initialize state and a development checkout")
    boot.add_argument("--repo-url", default="", help="clone URL when the checkout does not exist")
    boot.add_argument("--ref", default="", help="ref to check out")
    boot.add_argument("--reset-token", action="store_true", help="rotate the dev management token")
    boot.add_argument(
        "--no-wrappers", action="store_true", help="do not install ~/.local/bin helpers"
    )
    boot.add_argument("--no-checkpoint", action="store_true", help="skip the initial checkpoint")

    sub.add_parser("inspect-instance", help="revision, health, models, tasks, and GPU usage")

    logs = sub.add_parser("read-logs", help="read bounded application, build, or model logs")
    logs.add_argument("--source", choices=["app", "build", "model", "control"], default="app")
    logs.add_argument("--service", choices=["backend", "frontend", "all"], default="all")
    logs.add_argument("--model", default="")
    logs.add_argument("--tail", type=int, default=200)
    logs.add_argument("--since", default="")
    logs.add_argument("--search", default="")
    logs.add_argument("--level", choices=["error", "warn", "info", "debug"], default="")
    logs.add_argument("--max-bytes", type=int, default=256_000)

    patch = sub.add_parser("apply-patch", help="apply a unified diff to a worktree")
    patch.add_argument("--patch", dest="patch_file", help="path to a unified diff file")
    patch.add_argument("--stdin", action="store_true", help="read the diff from stdin")
    patch.add_argument("--worktree", default="")
    patch.add_argument("--no-checkpoint", action="store_true")
    patch.add_argument("--check-only", action="store_true")
    patch.add_argument("--label", default="")

    checks = sub.add_parser("run-checks", help="run allowlisted tests and lint")
    checks.add_argument("--check", action="append", default=[], help="check name; repeatable")
    checks.add_argument(
        "--test", action="append", default=[], help="backend test path under backend/tests"
    )
    checks.add_argument("--frontend-target", action="append", default=[])
    checks.add_argument("--timeout", type=float, default=None, help="per-check timeout seconds")
    checks.add_argument("--no-fail-fast", action="store_true")

    reload_parser = sub.add_parser("reload-service", help="restart a selected dev service")
    reload_parser.add_argument("service", choices=["backend", "frontend", "all"])
    reload_parser.add_argument("--wait-seconds", type=float, default=90.0)

    diagnostics = sub.add_parser("capture-diagnostics", help="collect a redacted evidence bundle")
    diagnostics.add_argument("--reason", default="")
    diagnostics.add_argument("--note", default="")

    rollback = sub.add_parser("rollback-changes", help="restore a worktree to a checkpoint")
    rollback.add_argument("--worktree", default="")
    rollback.add_argument("--checkpoint", default="")

    checkpoint = sub.add_parser("checkpoint-changes", help="checkpoint the current worktree")
    checkpoint.add_argument("--worktree", default="")
    checkpoint.add_argument("--label", default="")

    journal = sub.add_parser("journal", help="show recent control-plane operations")
    journal.add_argument("--limit", type=int, default=50)

    sub.add_parser("mcp", help="run the MCP stdio server")

    worktree = sub.add_parser("worktree", help="manage agent worktrees")
    worktree_sub = worktree.add_subparsers(dest="worktree_command", required=True)
    worktree_sub.add_parser("list")
    create = worktree_sub.add_parser("create")
    create.add_argument("name")
    create.add_argument("--base", default="HEAD")
    remove = worktree_sub.add_parser("remove")
    remove.add_argument("name")
    remove.add_argument("--force", action="store_true")
    activate = worktree_sub.add_parser("activate")
    activate.add_argument("name")
    activate.add_argument("--no-start", action="store_true")

    stack = sub.add_parser("stack", help="manage the isolated compose stack")
    stack_sub = stack.add_subparsers(dest="stack_command", required=True)
    build = stack_sub.add_parser("build")
    build.add_argument("--no-cache", action="store_true")
    up = stack_sub.add_parser("up")
    up.add_argument("--gpu", default=None, help="select GPU ids; use '' to disable")
    up.add_argument("--build", action="store_true")
    down = stack_sub.add_parser("down")
    down.add_argument("--volumes", action="store_true")
    down.add_argument("--yes", action="store_true")
    stack_sub.add_parser("status")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    values = {
        "repo": args.repo,
        "data": args.data,
        "worktrees": args.worktrees,
        "shared_models": args.shared_models,
        "image": args.image,
        "backend_port": args.backend_port,
        "frontend_port": args.frontend_port,
        "proxy_port": args.proxy_port,
        "gpu": args.gpu,
    }
    return {key: value for key, value in values.items() if value is not None}


def _load(args: argparse.Namespace) -> DevControl:
    config = Config.load(state_home=args.state_home, overrides=_overrides(args))
    config.assert_initialized()
    if args.actor:
        config_actor = args.actor
    else:
        config_actor = config.actor
    return DevControl(config, actor=config_actor)


def _error_payload(action: str, exc: BaseException) -> dict[str, Any]:
    if isinstance(exc, DevControlError):
        return {
            "ok": False,
            "operation_id": new_operation_id(),
            "action": action,
            "revision": None,
            "error": {
                "code": exc.code,
                "message": exc.message,
                "detail": exc.detail,
                "exit_code": exc.exit_code,
            },
        }
    return {
        "ok": False,
        "operation_id": new_operation_id(),
        "action": action,
        "revision": None,
        "error": {
            "code": "INTERNAL_ERROR",
            "message": f"{type(exc).__name__}: {exc}",
            "detail": {},
            "exit_code": 1,
        },
    }


def _read_patch(args: argparse.Namespace) -> str:
    if args.patch_file and args.stdin:
        raise GuardError("use either --patch or --stdin, not both")
    if args.patch_file:
        path = Path(args.patch_file).expanduser()
        if not path.is_file():
            raise GuardError(f"patch file does not exist: {path}")
        data = path.read_bytes()
    elif args.stdin:
        data = sys.stdin.buffer.read(2_000_001)
    else:
        raise GuardError("apply-patch needs --patch FILE or --stdin")
    if len(data) > 2_000_000:
        raise GuardError("patch exceeds 2,000,000 bytes")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GuardError("patch must be UTF-8 text") from exc


def dispatch(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "bootstrap":
        return bootstrap(
            state_home=args.state_home,
            repo=args.repo,
            repo_url=args.repo_url,
            ref=args.ref,
            data=args.data,
            worktrees=args.worktrees,
            shared_models=args.shared_models,
            image=args.image or "",
            backend_port=args.backend_port,
            frontend_port=args.frontend_port,
            proxy_port=args.proxy_port,
            gpu=args.gpu,
            reset_token=args.reset_token,
            wrappers=not args.no_wrappers,
            checkpoint=not args.no_checkpoint,
        )
    if args.command == "mcp":
        service = _load(args)
        from .mcp import serve

        serve(service)
        return {"ok": True, "action": "mcp", "operation_id": new_operation_id(), "revision": None}
    service = _load(args)
    if args.command == "inspect-instance":
        return service.inspect_instance()
    if args.command == "read-logs":
        if args.source == "control":
            return service.journal_tail(limit=args.tail)
        return service.read_logs(
            source=args.source,
            service=args.service,
            model=args.model,
            tail=args.tail,
            since=args.since,
            search=args.search,
            level=args.level,
            max_bytes=args.max_bytes,
        )
    if args.command == "apply-patch":
        return service.apply_patch(
            _read_patch(args),
            worktree=args.worktree,
            create_checkpoint=not args.no_checkpoint,
            check_only=args.check_only,
            label=args.label,
        )
    if args.command == "run-checks":
        return service.run_checks(
            checks=args.check or ["ruff"],
            worktree="",
            tests=args.test,
            frontend_targets=args.frontend_target,
            timeout_seconds=args.timeout,
            fail_fast=not args.no_fail_fast,
        )
    if args.command == "reload-service":
        return service.reload_service(service=args.service, wait_seconds=args.wait_seconds)
    if args.command == "capture-diagnostics":
        return service.capture_diagnostics(reason=args.reason, note=args.note)
    if args.command == "rollback-changes":
        return service.rollback_changes(worktree=args.worktree, checkpoint=args.checkpoint)
    if args.command == "checkpoint-changes":
        return service.checkpoint_changes(worktree=args.worktree, label=args.label)
    if args.command == "journal":
        return service.journal_tail(limit=args.limit)
    if args.command == "worktree":
        if args.worktree_command == "list":
            return service.list_worktrees()
        if args.worktree_command == "create":
            return service.create_worktree(name=args.name, base=args.base)
        if args.command == "worktree" and args.worktree_command == "remove":
            return service.remove_worktree(name=args.name, force=args.force)
        if args.worktree_command == "activate":
            return service.activate_worktree(name=args.name, start=not args.no_start)
    if args.command == "stack":
        if args.stack_command == "build":
            return service.stack_build(no_cache=args.no_cache)
        if args.stack_command == "up":
            return service.stack_up(gpu=args.gpu, build=args.build)
        if args.stack_command == "down":
            return service.stack_down(volumes=args.volumes, confirmed=args.yes)
        if args.stack_command == "status":
            return service.stack_status()
    raise GuardError(f"unsupported command: {args.command}")


def emit(payload: dict[str, Any], *, pretty: bool = False) -> None:
    if pretty:
        sys.stdout.write(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n")
    else:
        sys.stdout.write(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
        )
    sys.stdout.flush()


def _normalise_argv(argv: Sequence[str]) -> list[str]:
    """Allow the JSON flags after the subcommand as well as before it."""
    values = list(argv)
    for flag in ("--json", "--pretty"):
        if flag in values:
            values = [item for item in values if item != flag]
            values.insert(0, flag)
    return values


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(_normalise_argv(raw))
    if args.command == "mcp":
        try:
            dispatch(args)
        except Exception as exc:
            from .mcp import _error

            sys.stdout.write(json.dumps(_error(str(exc), code=-32603)) + "\n")
            sys.stdout.flush()
            return 1
        return 0
    try:
        payload = dispatch(args)
    except Exception as exc:  # keep JSON-only stdout even for a failed bootstrap
        payload = _error_payload(args.command, exc)
    emit(payload, pretty=bool(args.pretty))
    if payload.get("ok", True):
        return 0
    error = payload.get("error") or {}
    code = int(error.get("exit_code") or 1)
    return code if code in {1, 2, 3} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
