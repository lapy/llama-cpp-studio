"""Tests for build cancellation registry (in-flight source builds)."""

import asyncio

import pytest

import backend.build_cancel_registry as reg


@pytest.fixture(autouse=True)
def clear_registry():
    reg._events.clear()
    yield
    reg._events.clear()


def test_register_and_request_cancel():
    ev = reg.register_build_cancel("task-1")
    assert not ev.is_set()
    assert reg.request_build_cancel("task-1") is True
    assert ev.is_set()
    assert reg.is_build_cancel_requested("task-1") is True


def test_request_cancel_unknown_returns_false():
    assert reg.request_build_cancel("missing") is False


def test_request_cancel_twice_second_call_false():
    reg.register_build_cancel("t2")
    assert reg.request_build_cancel("t2") is True
    assert reg.request_build_cancel("t2") is False


def test_unregister_removes_cancel():
    reg.register_build_cancel("t3")
    reg.unregister_build_cancel("t3")
    assert reg.request_build_cancel("t3") is False
    assert reg.is_build_cancel_requested("t3") is False


def test_is_build_cancel_requested_none_and_empty():
    assert reg.is_build_cancel_requested(None) is False
    assert reg.is_build_cancel_requested("") is False


@pytest.mark.asyncio
async def test_register_returns_usable_event():
    ev = reg.register_build_cancel("async-task")
    await asyncio.sleep(0)
    reg.request_build_cancel("async-task")
    await ev.wait()
    assert ev.is_set()


def test_build_cancelled_error_is_exception():
    err = reg.BuildCancelledError("x")
    assert str(err) == "x"
