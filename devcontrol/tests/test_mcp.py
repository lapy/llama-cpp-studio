from __future__ import annotations

import io
import json
from typing import Any

from devcontrol.mcp import TOOL_NAMES, handle_message, serve, tool_names


class StubService:
    def inspect_instance(self) -> dict[str, Any]:
        return {
            "ok": True,
            "operation_id": "op-1",
            "revision": "abc",
            "health": {"overall": "healthy"},
        }

    def apply_patch(self, patch: str, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, "operation_id": "op-2", "revision": "abc", "bytes": len(patch)}

    def list_worktrees(self) -> dict[str, Any]:
        return {"ok": True, "operation_id": "op-3", "revision": "abc", "worktrees": []}


def test_initialize_and_tools_list() -> None:
    init = handle_message(
        StubService(),
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2025-06-18"},
        },
    )
    assert init is not None
    assert init["result"]["protocolVersion"] == "2025-06-18"
    assert "tools" in init["result"]["capabilities"]

    listed = handle_message(StubService(), {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    assert listed is not None
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert {"inspect_instance", "apply_patch", "reload_service"} <= names
    assert names == TOOL_NAMES


def test_tools_call_returns_structured_result() -> None:
    response = handle_message(
        StubService(),
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "apply_patch", "arguments": {"patch": "hello"}},
        },
    )
    assert response is not None
    result = response["result"]
    assert result["isError"] is False
    payload = json.loads(result["content"][0]["text"])
    assert payload["operation_id"] == "op-2"
    assert result["structuredContent"]["revision"] == "abc"


def test_unknown_tool_and_method_are_errors() -> None:
    unknown_tool = handle_message(
        StubService(),
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "nope"}},
    )
    assert unknown_tool is not None
    assert unknown_tool["error"]["code"] == -32602

    unknown_method = handle_message(
        StubService(), {"jsonrpc": "2.0", "id": 5, "method": "resources/list"}
    )
    assert unknown_method is not None
    assert unknown_method["error"]["code"] == -32601


def test_notifications_do_not_get_responses() -> None:
    assert (
        handle_message(StubService(), {"jsonrpc": "2.0", "method": "notifications/initialized"})
        is None
    )
    assert (
        handle_message(StubService(), {"jsonrpc": "2.0", "method": "notifications/cancelled"})
        is None
    )


def test_serve_handles_newline_delimited_stream() -> None:
    lines = [
        {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "list_worktrees"}},
    ]
    stdin = io.StringIO("\n".join(json.dumps(line) for line in lines) + "\n")
    stdout = io.StringIO()
    assert serve(StubService(), stdin, stdout) == 0
    responses = [json.loads(line) for line in stdout.getvalue().splitlines()]
    assert responses[0]["result"] == {}
    assert json.loads(responses[1]["result"]["content"][0]["text"])["operation_id"] == "op-3"


def test_tool_names_are_sorted() -> None:
    assert tool_names() == sorted(TOOL_NAMES)
