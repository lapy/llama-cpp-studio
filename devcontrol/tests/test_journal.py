from __future__ import annotations

from pathlib import Path

import pytest

from devcontrol.errors import BusyError
from devcontrol.journal import Journal, new_operation_id
from devcontrol.locks import InstanceGuard
from devcontrol.redact import redact_obj, redact_text


def test_operation_ids_are_unique() -> None:
    ids = {new_operation_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(item.startswith("op-") for item in ids)


def test_journal_append_and_read(tmp_path: Path) -> None:
    journal = Journal(tmp_path / "operations.jsonl")
    journal.append({"operation_id": "a", "action": "inspect_instance"})
    journal.append({"operation_id": "b", "action": "apply_patch"})
    records = journal.read(10)
    assert [record["operation_id"] for record in records] == ["a", "b"]
    assert journal.latest_for()["operation_id"] == "b"
    assert journal.latest_for("other") is None


def test_instance_guard_reports_owner(tmp_path: Path) -> None:
    guard = InstanceGuard(tmp_path / "state")
    guard.acquire(operation_id="op-1", action="reload_service", worktree="main", actor="alice")
    try:
        second = InstanceGuard(tmp_path / "state")
        with pytest.raises(BusyError) as exc:
            second.acquire(
                operation_id="op-2",
                action="run_checks",
                worktree="main",
                actor="bob",
            )
        assert exc.value.holder is not None
        assert exc.value.holder["operation_id"] == "op-1"
    finally:
        guard.release()


def test_redaction_covers_tokens_and_assignments() -> None:
    text = (
        "Authorization: Bearer abcdefghijklmnop\n"
        "DEV_API_TOKEN=supersecretvalue\n"
        "token=hf_abcdefghijklmnop\n"
    )
    redacted = redact_text(text)
    assert "abcdefghijklmnop" not in redacted
    assert "supersecretvalue" not in redacted
    assert "hf_abcdefghijklmnop" not in redacted
    payload = redact_obj({"api_token": "value", "nested": {"password": "value"}})
    assert payload["api_token"] == "[redacted]"
    assert payload["nested"]["password"] == "[redacted]"
