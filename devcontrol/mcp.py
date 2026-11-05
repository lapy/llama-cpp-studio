"""Minimal MCP stdio server (JSON-RPC 2.0, newline-delimited).

The protocol surface is intentionally small: initialize, tools/list, tools/call
and ping. It has no third-party dependency so it can run from the development
checkout on the remote host without installing the application virtualenv.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Optional, TextIO

from . import PROTOCOL_VERSION, __version__
from .errors import DevControlError
from .service import DevControl

SERVER_NAME = "llama-cpp-studio-devcontrol"
SUPPORTED_PROTOCOL_VERSIONS = {"2024-11-05", "2025-03-26", "2025-06-18"}

TOOLS: list[dict[str, Any]] = [
    {
        "name": "inspect_instance",
        "description": (
            "Return the active development revision, stack/health status, active models, "
            "tasks, and GPU usage."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "read_logs",
        "description": "Read a bounded tail of application, build, or model logs.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "enum": ["app", "build", "model"]},
                "service": {"type": "string", "enum": ["backend", "frontend", "all"]},
                "model": {"type": "string"},
                "tail": {"type": "integer", "minimum": 1, "maximum": 5000},
                "since": {"type": "string"},
                "search": {"type": "string"},
                "level": {"type": "string", "enum": ["error", "warn", "info", "debug"]},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "apply_patch",
        "description": (
            "Apply a unified diff to a worktree. A checkpoint is created first by default. "
            "The active worktree is watched by Uvicorn/Vite."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string"},
                "worktree": {"type": "string"},
                "checkpoint": {"type": "boolean"},
                "label": {"type": "string"},
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
    },
    {
        "name": "run_checks",
        "description": "Run allowlisted tests or lint against the active development stack.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "checks": {"type": "array", "items": {"type": "string"}},
                "tests": {"type": "array", "items": {"type": "string"}},
                "frontend_targets": {"type": "array", "items": {"type": "string"}},
                "worktree": {"type": "string"},
                "timeout_seconds": {"type": "number"},
                "fail_fast": {"type": "boolean"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "reload_service",
        "description": "Restart a selected development service when no conflicting operation is active.",
        "inputSchema": {
            "type": "object",
            "properties": {"service": {"type": "string", "enum": ["backend", "frontend", "all"]}},
            "additionalProperties": False,
        },
    },
    {
        "name": "capture_diagnostics",
        "description": "Collect a redacted evidence bundle around a failure.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "note": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "rollback_changes",
        "description": "Restore a worktree to its latest checkpoint or a named checkpoint.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "worktree": {"type": "string"},
                "checkpoint": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "checkpoint_changes",
        "description": "Create a named checkpoint of the current worktree content.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "worktree": {"type": "string"},
                "label": {"type": "string"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_worktrees",
        "description": "List development worktrees, their revisions, and the active checkout.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "create_worktree",
        "description": "Create a new git worktree for an agent.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "base": {"type": "string"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
    {
        "name": "activate_worktree",
        "description": "Point the running development stack at a worktree and restart its services.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "start": {"type": "boolean"},
            },
            "required": ["name"],
            "additionalProperties": False,
        },
    },
]

TOOL_NAMES = {tool["name"] for tool in TOOLS}


def _error(message: str, *, code: int = -32602, data: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        body["data"] = data
    return {"jsonrpc": "2.0", "id": None, "error": body}


def _jsonrpc_error(request_id: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        body["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": body}


def _require_object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def call_tool(service: DevControl, name: str, arguments: Any) -> dict[str, Any]:
    args = _require_object(arguments)
    if name == "inspect_instance":
        return service.inspect_instance()
    if name == "read_logs":
        source = str(args.get("source") or "app")
        if source == "control":
            return service.journal_tail(limit=int(args.get("tail") or 50))
        return service.read_logs(
            source=source,
            service=str(args.get("service") or "all"),
            model=str(args.get("model") or ""),
            tail=int(args.get("tail") or 200),
            since=str(args.get("since") or ""),
            search=str(args.get("search") or ""),
            level=str(args.get("level") or ""),
        )
    if name == "apply_patch":
        patch = args.get("patch")
        if not isinstance(patch, str):
            raise DevControlError("apply_patch requires a string 'patch' argument")
        return service.apply_patch(
            patch,
            worktree=str(args.get("worktree") or ""),
            create_checkpoint=bool(args.get("checkpoint", True)),
            label=str(args.get("label") or ""),
        )
    if name == "run_checks":
        checks = args.get("checks")
        if not isinstance(checks, list) or not checks:
            checks = ["ruff"]
        tests = args.get("tests") if isinstance(args.get("tests"), list) else []
        frontend_targets = (
            args.get("frontend_targets") if isinstance(args.get("frontend_targets"), list) else []
        )
        timeout = args.get("timeout_seconds")
        return service.run_checks(
            checks=[str(item) for item in checks],
            worktree=str(args.get("worktree") or ""),
            tests=[str(item) for item in tests],
            frontend_targets=[str(item) for item in frontend_targets],
            timeout_seconds=float(timeout) if timeout is not None else None,
            fail_fast=bool(args.get("fail_fast", True)),
        )
    if name == "reload_service":
        return service.reload_service(service=str(args.get("service") or "backend"))
    if name == "capture_diagnostics":
        return service.capture_diagnostics(
            reason=str(args.get("reason") or ""),
            note=str(args.get("note") or ""),
        )
    if name == "rollback_changes":
        return service.rollback_changes(
            worktree=str(args.get("worktree") or ""),
            checkpoint=str(args.get("checkpoint") or ""),
        )
    if name == "checkpoint_changes":
        return service.checkpoint_changes(
            worktree=str(args.get("worktree") or ""), label=str(args.get("label") or "")
        )
    if name == "list_worktrees":
        return service.list_worktrees()
    if name == "create_worktree":
        name_value = str(args.get("name") or "")
        if not name_value:
            raise DevControlError("create_worktree requires 'name'")
        return service.create_worktree(name=name_value, base=str(args.get("base") or "HEAD"))
    if name == "activate_worktree":
        name_value = str(args.get("name") or "")
        if not name_value:
            raise DevControlError("activate_worktree requires 'name'")
        return service.activate_worktree(name=name_value, start=bool(args.get("start", True)))
    raise DevControlError(f"unknown tool: {name}", detail={"code": "UNKNOWN_TOOL"})


def handle_message(service: DevControl, message: Any) -> Optional[dict[str, Any]]:
    if not isinstance(message, dict):
        return _jsonrpc_error(None, -32600, "invalid request")
    method = message.get("method")
    request_id = message.get("id")
    params = message.get("params")
    is_notification = "id" not in message
    if not isinstance(method, str):
        return None if is_notification else _jsonrpc_error(request_id, -32600, "missing method")
    if method == "notifications/initialized" or method == "notifications/cancelled":
        return None
    if method == "initialize":
        requested = _require_object(params).get("protocolVersion")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else "2025-06-18"
        result = {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            "instructions": (
                "Control plane for an isolated Studio development stack. "
                "Every call returns an operation_id and revision."
            ),
        }
        return {"jsonrpc": "2.0", "id": request_id, "result": result}
    if method == "ping":
        return {"jsonrpc": "2.0", "id": request_id, "result": {}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": request_id, "result": {"tools": TOOLS}}
    if method == "tools/call":
        params_obj = _require_object(params)
        name = str(params_obj.get("name") or "")
        if name not in TOOL_NAMES:
            return _jsonrpc_error(request_id, -32602, f"unknown tool: {name}")
        try:
            result = call_tool(service, name, params_obj.get("arguments") or {})
        except DevControlError as exc:
            result = {
                "ok": False,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "detail": exc.detail,
                },
                "operation_id": None,
                "revision": None,
            }
        except Exception as exc:  # keep the MCP stream alive on a tool bug
            result = {
                "ok": False,
                "error": {
                    "code": "INTERNAL_ERROR",
                    "message": f"{type(exc).__name__}: {exc}",
                    "detail": {},
                },
                "operation_id": None,
                "revision": None,
            }
        is_error = not bool(result.get("ok", False))
        text = json.dumps(result, ensure_ascii=False, default=str)
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{"type": "text", "text": text}],
                "structuredContent": result,
                "isError": is_error,
            },
        }
    return (
        None
        if is_notification
        else _jsonrpc_error(request_id, -32601, f"method not found: {method}")
    )


def serve(service: DevControl, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> int:
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError as exc:
            response = _error(f"parse error: {exc}", code=-32700)
            stdout.write(json.dumps(response, ensure_ascii=False) + "\n")
            stdout.flush()
            continue
        response = handle_message(service, message)
        if response is not None:
            stdout.write(json.dumps(response, ensure_ascii=False, default=str) + "\n")
            stdout.flush()
    return 0


def tool_names() -> list[str]:
    return sorted(TOOL_NAMES)


def protocol_version() -> str:
    return PROTOCOL_VERSION
