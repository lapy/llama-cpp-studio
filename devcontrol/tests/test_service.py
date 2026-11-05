from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from devcontrol.bootstrap import bootstrap
from devcontrol.config import Config
from devcontrol.http_client import HttpResult
from devcontrol.service import DevControl


@pytest.fixture
def service(tmp_path: Path, git_repo: Path) -> DevControl:
    bootstrap(
        state_home=tmp_path / "state",
        repo=git_repo,
        data=tmp_path / "data",
        worktrees=tmp_path / "worktrees",
        shared_models=tmp_path / "shared",
        wrappers=False,
    )
    config = Config.load(state_home=tmp_path / "state")
    return DevControl(config, actor="test-agent")


def fake_result(url: str, **kwargs: Any) -> HttpResult:
    if url.endswith("/api/live"):
        return HttpResult(status=200, data={"live": True})
    if url.endswith("/api/ready"):
        return HttpResult(status=200, data={"ready": True})
    if url.endswith("/api/status"):
        return HttpResult(
            status=200,
            data={
                "running_instances": [
                    {"model_id": "demo", "runtime_type": "llama_cpp", "port": 12000}
                ],
                "proxy_status": {"healthy": True},
            },
        )
    if url.endswith("/api/gpu-info"):
        return HttpResult(status=200, data={"gpus": [{"index": 0, "name": "test-gpu"}]})
    return HttpResult(status=200, data={})


def test_inspect_returns_revision_models_tasks_and_gpu(
    service: DevControl, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("devcontrol.service.request_json", fake_result)
    monkeypatch.setattr(
        "devcontrol.service.task_snapshot",
        lambda *args, **kwargs: [
            {"task_id": "t1", "type": "download", "status": "running", "description": "model"}
        ],
    )
    monkeypatch.setattr(
        "devcontrol.service.Compose.summary",
        lambda self: {"project": "llama-cpp-studio-dev", "running": True, "services": {}},
    )
    out = service.inspect_instance()
    assert out["ok"] is True
    assert out["operation_id"].startswith("op-")
    assert out["revision"] == out["revision_after"]
    assert out["health"]["overall"] == "healthy"
    assert out["models"][0]["model_id"] == "demo"
    assert out["tasks"]["active"][0]["task_id"] == "t1"
    assert out["gpu"]["devices"][0]["name"] == "test-gpu"


def test_apply_patch_is_checkpointed_and_rollback_restores(
    service: DevControl, patch_text: str, git_repo: Path
) -> None:
    applied = service.apply_patch(patch_text, label="agent change")
    assert applied["ok"] is True
    assert applied["checkpoint_id"]
    assert "+dirty." in applied["revision"]
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "two\n"

    rolled = service.rollback_changes(checkpoint=applied["checkpoint_id"])
    assert rolled["ok"] is True
    assert (git_repo / "a.txt").read_text(encoding="utf-8") == "one\n"


def test_rejected_patch_does_not_create_a_checkpoint(service: DevControl, git_repo: Path) -> None:
    before = service.checkpoints.list("main")
    out = service.apply_patch("not a patch at all\n")
    assert out["ok"] is False
    assert out["error"]["code"] == "GUARD_REFUSED"
    assert service.checkpoints.list("main") == before


def test_run_checks_refuses_inactive_worktree(service: DevControl) -> None:
    created = service.create_worktree(name="alice")
    assert created["ok"] is True
    out = service.run_checks(checks=["ruff"], worktree="alice")
    assert out["ok"] is False
    assert out["operation_id"].startswith("op-")
    assert out["error"]["code"] == "GUARD_REFUSED"
    assert "not active" in out["error"]["message"]


def test_reload_reports_busy_owner(service: DevControl) -> None:
    service.guard.acquire(
        operation_id="op-owner",
        action="run_checks",
        worktree="main",
        actor="other-agent",
    )
    try:
        out = service.reload_service(service="backend")
    finally:
        service.guard.release()
    assert out["ok"] is False
    assert out["error"]["code"] == "OPERATION_BUSY"
    assert out["error"]["detail"]["holder"]["operation_id"] == "op-owner"


def test_capture_diagnostics_writes_redacted_bundle(
    service: DevControl, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        service,
        "_inspect_payload",
        lambda: {"revision": "abc", "models": [], "health": {"overall": "down"}},
    )
    monkeypatch.setattr(
        "devcontrol.service.read_logs",
        lambda *args, **kwargs: {"lines": ["token=supersecretvalue"]},
    )
    monkeypatch.setattr("devcontrol.service.request_json", lambda *a, **k: HttpResult(status=0))
    monkeypatch.setattr("devcontrol.service.Compose.ps", lambda self: [])
    out = service.capture_diagnostics(reason="test failure")
    assert out["ok"] is True
    assert Path(out["path"]).is_dir()
    assert Path(out["archive"]).is_file()
    assert any(item["name"] == "manifest.json" for item in out["files"])


def test_activate_also_respects_instance_guard(service: DevControl) -> None:
    service.guard.acquire(
        operation_id="op-owner",
        action="run_checks",
        worktree="main",
        actor="other-agent",
    )
    try:
        out = service.activate_worktree(name="main", start=False)
    finally:
        service.guard.release()
    assert out["ok"] is False
    assert out["error"]["code"] == "OPERATION_BUSY"


def test_journal_records_each_action(
    service: DevControl, patch_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(service, "_inspect_payload", lambda: {"revision": "abc"})
    first = service.inspect_instance()
    service.apply_patch(patch_text)
    records = service.journal.read(10)
    assert any(record["operation_id"] == first["operation_id"] for record in records)
    assert any(record["action"] == "apply_patch" for record in records)
