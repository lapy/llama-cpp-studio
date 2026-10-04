"""Structured error text and API error logs."""

import logging

from fastapi import HTTPException

from backend.logging_config import describe_error
from backend.services.model_runtime_apply import PreflightError


def test_describe_error_includes_structured_failures_and_cause():
    try:
        raise PreflightError(["kept: llama-server binary not found"])
    except PreflightError as original:
        wrapped = HTTPException(status_code=original.status, detail=original.detail)
        wrapped.__cause__ = original

    text = describe_error(wrapped)
    assert "409" in text
    assert "llama-server binary not found" in text
    assert text.count("llama-server binary not found") == 1


def test_describe_error_formats_validation_locations():
    class Invalid(Exception):
        def errors(self):
            return [
                {"loc": ["body", "mode"], "msg": "field required"},
                {"loc": ["body", "models"], "msg": "value is not a valid list"},
            ]

    text = describe_error(Invalid())
    assert "body.mode: field required" in text
    assert "body.models: value is not a valid list" in text


def test_apply_rejection_is_logged_with_the_failure(client, caplog, monkeypatch):
    from backend.routes import llama_swap as llama_swap_routes

    class FakeManager:
        async def user_apply_regenerate_config(self):
            raise PreflightError(["kept: llama-server binary not found"])

    monkeypatch.setattr(
        llama_swap_routes, "get_llama_swap_manager", lambda: FakeManager()
    )
    with caplog.at_level(logging.WARNING, logger="backend.main"):
        response = client.post("/api/llama-swap/apply-config")

    assert response.status_code == 409
    assert any(
        "llama-server binary not found" in record.message
        and "POST /api/llama-swap/apply-config" in record.message
        for record in caplog.records
    )


def test_validation_errors_are_logged(client, caplog):
    with caplog.at_level(logging.WARNING, logger="backend.main"):
        response = client.post("/api/llama-swap/apply-launch", json={"mode": "nope"})

    assert response.status_code == 422
    assert any(
        "POST /api/llama-swap/apply-launch" in record.message and "422" in record.message
        for record in caplog.records
    )
