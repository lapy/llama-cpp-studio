"""Management mutation audit: record shape, redaction, rotation, actor kinds."""

import json
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from backend import access_policy
from backend.observability import audit

RECORD_FIELDS = {
    "timestamp",
    "request_id",
    "method",
    "path",
    "status",
    "duration_ms",
    "actor",
}


def _read_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _audited_app(log: audit.AuditLog) -> FastAPI:
    app = FastAPI()

    @app.post("/api/widgets")
    async def create_widget():
        return {"ok": True}

    @app.get("/api/widgets")
    async def list_widgets():
        return []

    @app.post("/api/widgets/forbidden")
    async def forbidden():
        raise HTTPException(status_code=403, detail="not allowed")

    @app.post("/api/boom")
    async def boom():
        raise RuntimeError("boom")

    @app.post("/other")
    async def other():
        return {"ok": True}

    app.add_middleware(audit.AuditMiddleware, audit_log=log)
    return app


def _request(
    *,
    headers: dict[str, str] | None = None,
    client: tuple[str, int] = ("testclient", 50000),
) -> Request:
    raw_headers = [
        (name.lower().encode("latin-1"), value.encode("latin-1"))
        for name, value in (headers or {}).items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/widgets",
        "query_string": b"",
        "headers": raw_headers,
        "client": client,
    }
    return Request(scope)


def test_middleware_audits_mutation_without_bodies_headers_or_query(tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    log = audit.AuditLog(str(path))
    try:
        response = TestClient(_audited_app(log)).post(
            "/api/widgets?token=query-secret&model=org%2Fmodel",
            json={"secret": "body-secret"},
            headers={
                "Authorization": "Bearer header-secret",
                "X-Api-Key": "header-secret",
            },
        )

        assert response.status_code == 200
        log.flush()
        records = _read_records(path)
        assert len(records) == 1
        record = records[0]
        assert set(record) == RECORD_FIELDS
        assert record["method"] == "POST"
        assert record["path"] == "/api/widgets"
        assert record["status"] == 200
        assert record["actor"] == "local"
        assert isinstance(record["duration_ms"], float)
        assert record["duration_ms"] >= 0
        assert len(record["request_id"]) == 32
        assert record["timestamp"].endswith("Z")
        raw = path.read_text(encoding="utf-8")
        for secret in ("query-secret", "body-secret", "header-secret", "org%2Fmodel"):
            assert secret not in raw
    finally:
        log.close()


def test_middleware_records_rejected_and_failed_mutations(tmp_path):
    log = audit.AuditLog(str(tmp_path / "audit.jsonl"))
    try:
        client = TestClient(_audited_app(log), raise_server_exceptions=False)
        forbidden = client.post("/api/widgets/forbidden")
        failed = client.post("/api/boom")

        assert forbidden.status_code == 403
        assert failed.status_code == 500
        log.flush()
        records = _read_records(tmp_path / "audit.jsonl")
        assert [record["path"] for record in records] == [
            "/api/widgets/forbidden",
            "/api/boom",
        ]
        assert [record["status"] for record in records] == [403, 500]
    finally:
        log.close()


def test_middleware_ignores_reads_and_non_management_routes(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = audit.AuditLog(str(path))
    try:
        client = TestClient(_audited_app(log))
        assert client.get("/api/widgets").status_code == 200
        assert client.post("/other").status_code == 200
        log.flush()
        assert not path.exists() or path.read_text(encoding="utf-8") == ""
    finally:
        log.close()


def test_middleware_records_token_actor_without_the_token(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = audit.AuditLog(str(path))
    token = "token-value-that-must-not-be-logged"
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(access_policy, "management_token", lambda: token)
            TestClient(_audited_app(log)).post(
                "/api/widgets", headers={"Authorization": f"Bearer {token}"}
            )
        log.flush()
        record = _read_records(path)[-1]
        assert record["actor"] == "token"
        assert token not in path.read_text(encoding="utf-8")
    finally:
        log.close()


def test_request_actor_kind_distinguishes_session_local_and_anonymous(monkeypatch):
    monkeypatch.setattr(access_policy, "management_token", lambda: "expected-token")
    monkeypatch.setattr(access_policy, "_sessions", {"session-id": "csrf-value"})
    monkeypatch.setattr(access_policy, "in_container", lambda: False)

    assert access_policy.request_actor_kind(_request()) == "local"
    assert (
        access_policy.request_actor_kind(
            _request(headers={"Authorization": "Bearer expected-token"})
        )
        == "token"
    )
    assert (
        access_policy.request_actor_kind(_request(headers={"cookie": "studio_session=session-id"}))
        == "session"
    )
    assert access_policy.request_actor_kind(_request(client=("10.9.8.7", 1234))) == "anonymous"

    # Remote mode does not grant local-trust authorization to a loopback caller.
    monkeypatch.setenv("STUDIO_ACCESS_MODE", "remote")
    assert access_policy.request_actor_kind(_request()) == "anonymous"


def test_path_redaction_truncates_and_removes_secret_assignments():
    assert audit.redact_path("/api/hooks/token=abc123/run") == "/api/hooks/token=redacted/run"
    assert audit.redact_path("/api/x\ninjected") == "/api/xinjected"
    assert audit.redact_path("/api/key:secret-value/x") == "/api/key:redacted/x"
    assert audit.redact_path("/api/auth/bearer abc123/x") == "/api/auth/bearer redacted/x"

    truncated = audit.redact_path("/api/" + "a" * 2000)
    assert len(truncated) == audit.MAX_AUDIT_PATH_CHARS + 3
    assert truncated.endswith("...")


def test_full_queue_drops_records_without_blocking(tmp_path):
    log = audit.AuditLog(str(tmp_path / "audit.jsonl"), queue_size=1)
    # Keep the record in the bounded queue instead of racing the writer thread.
    log._ensure_thread = lambda: None

    for index in range(5):
        log.record(
            {
                "timestamp": audit.utc_timestamp(),
                "request_id": str(index),
                "method": "POST",
                "path": "/api/widgets",
                "status": 200,
                "duration_ms": 1.0,
                "actor": "local",
            }
        )

    assert log.queue_size == 1
    assert log.dropped == 4


def test_clean_record_keeps_only_the_audit_allow_list():
    record = audit.clean_record(
        {
            "method": "post",
            "path": "/api/widgets",
            "status": "201",
            "duration_ms": 1.23456,
            "actor": "token",
            "request_id": "r",
            "body": {"secret": "body-secret"},
            "headers": {"authorization": "header-secret"},
            "query": "token=query-secret",
        }
    )

    assert set(record) == RECORD_FIELDS
    assert record["method"] == "POST"
    assert record["status"] == 201
    assert record["duration_ms"] == 1.235
    text = json.dumps(record)
    for secret in ("body-secret", "header-secret", "query-secret"):
        assert secret not in text


def test_rotation_bounds_files_and_keeps_every_line_parseable(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = audit.AuditLog(str(path), max_bytes=320, backup_count=2)
    try:
        for index in range(15):
            log.record(
                {
                    "timestamp": audit.utc_timestamp(),
                    "request_id": f"{index:032d}",
                    "method": "POST",
                    "path": f"/api/widgets/{index}",
                    "status": 200,
                    "duration_ms": 1.0,
                    "actor": "local",
                }
            )
        log.flush()
    finally:
        log.close()

    files = sorted(tmp_path.glob("audit.jsonl*"))
    assert [file.name for file in files] == [
        "audit.jsonl",
        "audit.jsonl.1",
        "audit.jsonl.2",
    ]
    assert not (tmp_path / "audit.jsonl.3").exists()
    total_lines = 0
    for file in files:
        lines = file.read_text(encoding="utf-8").splitlines()
        assert lines
        total_lines += len(lines)
        for line in lines:
            parsed = json.loads(line)
            assert set(parsed) == RECORD_FIELDS
        # Rotation happens before a write, so only a single over-sized line can
        # push a file past the limit; no line is ever split.
        assert file.stat().st_size <= 320 + 512
    assert total_lines <= 15
    assert log.written == 15


def test_main_app_audits_a_management_mutation(client, monkeypatch, tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    log = audit.AuditLog(str(path))
    try:
        monkeypatch.setattr(audit, "_default_log", log)
        response = client.post("/api/tasks/dismiss", json={"task_id": "not-running"})
        assert response.status_code == 200

        log.flush()
        records = _read_records(path)
        assert records
        record = records[-1]
        assert record["method"] == "POST"
        assert record["path"] == "/api/tasks/dismiss"
        assert record["status"] == 200
        assert record["actor"] == "local"
    finally:
        log.close()
        monkeypatch.setattr(audit, "_default_log", None)


def test_main_app_audits_a_rejected_remote_mutation(client, monkeypatch, tmp_path):
    path = tmp_path / "logs" / "audit.jsonl"
    log = audit.AuditLog(str(path))
    try:
        monkeypatch.setattr(audit, "_default_log", log)
        monkeypatch.setenv("STUDIO_ACCESS_MODE", "remote")
        monkeypatch.setenv("STUDIO_API_TOKEN", "management-secret")

        response = client.post("/api/tasks/dismiss", json={"task_id": "not-running"})

        assert response.status_code == 401
        log.flush()
        record = _read_records(path)[-1]
        assert record["path"] == "/api/tasks/dismiss"
        assert record["status"] == 401
        assert record["actor"] == "anonymous"
    finally:
        log.close()
        monkeypatch.setattr(audit, "_default_log", None)
