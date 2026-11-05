"""Progress manager task lifecycle and SSE subscribe (async)."""

import asyncio

import pytest

import backend.operations.progress as pm_mod


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


def test_update_task_rounds_progress_for_api_payloads():
    pm = pm_mod.get_progress_manager()
    tid = pm.create_task("x", "y")
    pm.update_task(tid, progress=100 / 3)
    assert pm.get_task(tid)["progress"] == 33.3


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


def test_send_build_progress_now_is_sync():
    pm = pm_mod.get_progress_manager()
    tid = pm.create_task("param_scan", "Scan llama.cpp CLI parameters")
    pm.send_build_progress_now(tid, "parse", 55, "Extracting flags", log_lines=["EXTRACT accept"])
    t = pm.get_task(tid)
    assert t["metadata"]["stage"] == "parse"
    assert t["progress"] == 55.0
    assert "EXTRACT accept" in t["metadata"]["log_lines"]


@pytest.mark.asyncio
async def test_send_download_progress_updates_task_metadata():
    pm = pm_mod.get_progress_manager()
    queue = asyncio.Queue()
    pm._subscribers.append(queue)
    tid = pm.create_task("download", "fetch model")
    created = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert created["event"] == "task_created"

    await pm.send_download_progress(
        task_id=tid,
        progress=40,
        message="Downloading model.gguf",
        bytes_downloaded=400,
        total_bytes=1000,
        speed_mbps=12.5,
        eta_seconds=48,
        filename="model.gguf",
        model_format="gguf",
        huggingface_id="org/model",
    )
    task = pm.get_task(tid)
    assert task["progress"] == 40.0
    assert task["message"] == "Downloading model.gguf"
    assert task["metadata"]["bytes_downloaded"] == 400
    assert task["metadata"]["total_bytes"] == 1000
    assert task["metadata"]["speed_mbps"] == 12.5
    assert task["metadata"]["huggingface_id"] == "org/model"

    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["event"] == "download_progress"
    assert event["data"]["progress"] == 40
    # Hot-path ticks must not also spam task_updated (double UI writes).
    assert queue.empty()


@pytest.mark.asyncio
async def test_update_task_can_skip_broadcast():
    pm = pm_mod.get_progress_manager()
    queue = asyncio.Queue()
    pm._subscribers.append(queue)
    tid = pm.create_task("download", "fetch")
    await asyncio.wait_for(queue.get(), timeout=1.0)

    pm.update_task(tid, progress=12, message="silent", broadcast=False)
    assert pm.get_task(tid)["progress"] == 12.0
    assert queue.empty()

    pm.update_task(tid, progress=20, message="loud")
    event = await asyncio.wait_for(queue.get(), timeout=1.0)
    assert event["event"] == "task_updated"
    assert event["data"]["progress"] == 20.0


def test_broadcast_snapshots_do_not_follow_later_mutations():
    pm = pm_mod.get_progress_manager()
    queue = asyncio.Queue()
    pm._subscribers.append(queue)
    task_id = pm.create_task("job", "running")
    event = queue.get_nowait()
    assert event["data"]["status"] == "running"
    pm.complete_task(task_id)
    assert event["data"]["status"] == "running"
    assert pm.get_task(task_id)["status"] == "completed"


@pytest.mark.asyncio
async def test_subscribe_sends_terminal_tasks_in_the_snapshot():
    pm = pm_mod.get_progress_manager()
    task_id = pm.create_task("job", "done soon")
    pm.complete_task(task_id)
    gen = pm.subscribe()
    chunks = []
    for _ in range(3):
        chunks.append(await asyncio.wait_for(gen.__anext__(), timeout=1.0))
    body = "".join(chunks)
    assert "task_snapshot" in body
    assert "completed" in body
    await gen.aclose()


def test_restore_skips_finished_history_and_background_bookkeeping():
    pm = pm_mod.get_progress_manager()
    pm.restore_operations(
        [
            {
                "operation_id": "scan-1",
                "kind": "param_scan",
                "status": "failed",
                "message": "scan",
                "detail": {},
            },
            {
                "operation_id": "apply-1",
                "kind": "runtime_apply",
                "status": "failed",
                "message": "apply",
                "detail": {},
            },
            {
                "operation_id": "sync-1",
                "kind": "build",
                "status": "succeeded",
                "message": "Synced",
                "detail": {"description": "Sync llama.cpp"},
            },
            {
                "operation_id": "build-1",
                "kind": "build",
                "status": "failed",
                "message": "compiler error",
                "detail": {"description": "Build llama.cpp"},
            },
        ]
    )
    assert pm.get_task("scan-1") is None
    assert pm.get_task("apply-1") is None
    assert pm.get_task("sync-1") is None
    restored = pm.get_task("build-1")
    assert restored["status"] == "failed"
    assert restored["description"] == "Build llama.cpp"
    assert [task["task_id"] for task in pm.snapshot_tasks()] == ["build-1"]


def test_dismiss_finished_task_keeps_it_out_of_the_snapshot(monkeypatch):
    pm = pm_mod.get_progress_manager()
    monkeypatch.setattr(pm, "_remember_operation", lambda *args, **kwargs: None)
    monkeypatch.setattr(pm, "_finish_operation", lambda *args, **kwargs: None)
    forgotten = []
    monkeypatch.setattr(pm, "_forget_operation", lambda task_id: forgotten.append(task_id))
    task_id = pm.create_task("build", "Sync llama.cpp", task_id="build_sync_1")
    pm.complete_task(task_id)
    assert pm.dismiss_task(task_id) is True
    assert pm.get_task(task_id) is None
    assert forgotten == [task_id]
    assert pm.snapshot_tasks() == []

    running = pm.create_task("download", "Download model", task_id="download_1")
    assert pm.dismiss_task(running) is False
    assert pm.get_task(running)["status"] == "running"


def test_snapshot_omits_parameter_scans_and_runtime_applies(monkeypatch):
    pm = pm_mod.get_progress_manager()
    monkeypatch.setattr(pm, "_remember_operation", lambda *args, **kwargs: None)
    pm.create_task("param_scan", "Scan", task_id="scan_1")
    pm.create_task("runtime_apply", "runtime_apply", task_id="apply_1")
    pm.create_task("download", "Download", task_id="dl_1")
    assert {task["task_id"] for task in pm.snapshot_tasks()} == {"dl_1"}
