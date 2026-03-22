"""Progress manager task lifecycle and SSE subscribe (async)."""

import asyncio

import pytest

import backend.progress_manager as pm_mod


@pytest.fixture(autouse=True)
def isolated_progress_manager():
    pm_mod._progress_manager = pm_mod.ProgressManager()
    yield pm_mod.get_progress_manager()
    pm_mod._progress_manager = None


def test_create_and_complete_task():
    pm = pm_mod.get_progress_manager()
    tid = pm.create_task("download", "fetch model")
    assert len(tid) >= 4
    t = pm.get_task(tid)
    assert t["status"] == "running"
    pm.complete_task(tid, "ok")
    assert pm.get_task(tid)["status"] == "completed"
    assert pm.get_task(tid)["progress"] == 100.0


def test_update_task_clamps_progress():
    pm = pm_mod.get_progress_manager()
    tid = pm.create_task("x", "y")
    pm.update_task(tid, progress=150)
    assert pm.get_task(tid)["progress"] == 100.0
    pm.update_task(tid, progress=-10)
    assert pm.get_task(tid)["progress"] == 0.0


def test_fail_task():
    pm = pm_mod.get_progress_manager()
    tid = pm.create_task("x", "y")
    pm.fail_task(tid, "oops")
    assert pm.get_task(tid)["status"] == "failed"
    assert pm.get_task(tid)["message"] == "oops"


def test_get_active_tasks():
    pm = pm_mod.get_progress_manager()
    a = pm.create_task("a", "1")
    b = pm.create_task("b", "2")
    pm.complete_task(a)
    active = pm.get_active_tasks()
    assert len(active) == 1
    assert active[0]["task_id"] == b


@pytest.mark.asyncio
async def test_subscribe_yields_heartbeat_and_task_events():
    pm = pm_mod.get_progress_manager()
    pm.create_task("job", "running job")

    gen = pm.subscribe()
    first = await gen.__anext__()
    assert ": heartbeat" in first

    # Drain until we see a task-related event or timeout
    found = False
    for _ in range(20):
        chunk = await asyncio.wait_for(gen.__anext__(), timeout=2.0)
        if "task_created" in chunk or "task_updated" in chunk:
            found = True
            assert "data:" in chunk
            break
    assert found

    await gen.aclose()


@pytest.mark.asyncio
async def test_emit_notification():
    pm = pm_mod.get_progress_manager()
    q = asyncio.Queue()
    pm._subscribers.append(q)

    await pm.send_notification(title="T", message="M", type="info")

    event = await asyncio.wait_for(q.get(), timeout=2.0)
    assert event["event"] == "notification"
    assert event["data"]["title"] == "T"


@pytest.mark.asyncio
async def test_send_build_progress():
    pm = pm_mod.get_progress_manager()
    tid = pm.create_task("build", "compiling")
    await pm.send_build_progress(tid, "compile", 50, "cc main.c", log_lines=["line1"])
    t = pm.get_task(tid)
    assert t["metadata"]["stage"] == "compile"
    assert "line1" in t["metadata"]["log_lines"]
