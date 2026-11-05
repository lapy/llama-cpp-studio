"""One streaming command runner for native builds (Phase 7).

The plan's "one process runner for logs, cancellation, process-group
termination and progress events" (config-driven refactoring plan, line 94).
Extracted VERBATIM from audio.cpp's historical ``AudioCppManager`` streaming
runner (since deleted) so the cmake build path keeps its exact
progress/failure behavior; the native host and the sync helper share this one
implementation. ``emit_build_progress`` is that runner's SSE payload, verbatim.

The runner is deliberately callback-shaped: the callers own their progress
transport (audio's SSE ``build_progress`` events, the native host's task
updates), their cancellation source (the task-cancel registry, the operation
supervisor) and their active-process bookkeeping. Nothing here knows an
engine, a stage vocabulary, or a progress manager.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import time
from typing import Awaitable, Callable, List, Mapping, Optional, Sequence

from backend.logging_config import get_logger


logger = get_logger(__name__)

EmitCallback = Callable[[str, int, str, List[str], List[str]], Awaitable[None]]
"""``emit(stage, progress, message, new_lines, all_lines)``."""


def emit_build_progress(
    progress_manager,
    task_id,
    stage: str,
    progress: int,
    message: str,
    *,
    new_lines: Optional[List[str]] = None,
    all_lines: Optional[List[str]] = None,
) -> None:
    """Report one build progress step (audio's historical payload, verbatim).

    Keeps a trailing window in the task metadata for reconnect and streams only
    newly produced lines on ``build_progress`` so the UI appends without
    duplicating the buffer.
    """
    from datetime import datetime

    if not progress_manager or not task_id:
        return
    history = list(all_lines or new_lines or [])[-100:]
    delta = list(new_lines or [])
    progress_manager.update_task(
        task_id,
        progress=float(progress),
        message=message,
        metadata_update={"stage": stage, "log_lines": history},
    )
    if delta:
        progress_manager.emit(
            "build_progress",
            {
                "task_id": task_id,
                "stage": stage,
                "progress": progress,
                "message": message,
                "log_lines": delta,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )


async def run_streaming_command(
    argv: Sequence[str],
    *,
    stage: str,
    progress: int,
    base_message: str,
    emit: EmitCallback,
    is_cancelled: Callable[[], bool],
    terminate: Callable[[], Awaitable[None]],
    on_spawn: Optional[Callable[[asyncio.subprocess.Process], None]] = None,
    on_finish: Optional[Callable[[], None]] = None,
    cancel_message: str = "build cancelled",
    cwd: Optional[str] = None,
    env: Optional[Mapping[str, str]] = None,
    start_new_session: Optional[bool] = None,
    command_label: str = "engine",
) -> List[str]:
    """Run ``argv`` to completion, streaming lines into progress callbacks.

    Raises ``RuntimeError`` on a non-zero exit (the message carries the last
    40 output lines), re-raises ``asyncio.CancelledError`` after terminating
    the process, and raises ``TaskCancelledError(cancel_message)`` when
    ``is_cancelled()`` reports a request (checked before spawn and per line).
    """
    from backend.build_progress import BuildProgressTracker, cmake_stage_window

    if is_cancelled():
        raise _cancelled(cancel_message)
    logger.info("%s command: %s", command_label, shlex.join([str(a) for a in argv]))
    process = await asyncio.create_subprocess_exec(
        *[str(a) for a in argv],
        cwd=cwd,
        env=dict(env) if env is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=(os.name != "nt") if start_new_session is None else start_new_session,
    )
    if on_spawn is not None:
        on_spawn(process)
    lines: List[str] = []
    pending: List[str] = []
    last_emit = time.monotonic()
    floor, ceil = cmake_stage_window(stage)
    # Prefer the shared cmake window; allow callers to start mid-stage.
    floor = min(floor, int(progress))
    tracker = BuildProgressTracker(floor=floor, ceil=ceil, progress=max(int(progress), floor))
    try:
        assert process.stdout is not None

        async def flush_pending(message: str = "") -> None:
            nonlocal pending, last_emit
            if not pending:
                return
            await emit(
                stage,
                tracker.progress,
                message or pending[-1],
                list(pending),
                lines,
            )
            pending = []
            last_emit = time.monotonic()

        while True:
            if is_cancelled():
                await terminate()
                raise _cancelled(cancel_message)
            try:
                chunk = await asyncio.wait_for(process.stdout.readline(), timeout=0.25)
            except asyncio.TimeoutError:
                if process.returncode is not None:
                    break
                if pending and (time.monotonic() - last_emit) >= 0.4:
                    await flush_pending()
                continue
            if not chunk:
                break
            line = chunk.decode("utf-8", errors="replace").rstrip()
            if line:
                lines.append(line)
                pending.append(line)
                step = tracker.apply_line(line)
                if step:
                    _progress, suffix = step
                    await flush_pending(f"{base_message} {suffix}")
                elif len(pending) >= 12 or (time.monotonic() - last_emit) >= 0.4:
                    await flush_pending(line)
        return_code = await process.wait()
        if return_code != 0:
            detail = "\n".join(lines[-40:]).strip()
            raise RuntimeError(
                f"{stage} failed with exit code {return_code}" + (f":\n{detail}" if detail else "")
            )
        tracker.complete()
        await emit(stage, tracker.progress, base_message, [], lines)
        return lines
    except asyncio.CancelledError:
        await terminate()
        raise
    finally:
        if on_finish is not None:
            on_finish()


def _cancelled(message: str) -> BaseException:
    from backend.task_cancel_registry import TaskCancelledError

    return TaskCancelledError(message)
