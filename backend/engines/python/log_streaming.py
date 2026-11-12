"""Streamed install-log runner selected by a definition's ``lifecycle.log``.

Extracted verbatim from the former SGLang log mixin so every engine that
declares log streaming gets ONE implementation: line-by-line log writes,
heartbeat notices while a compiler is silent, verbose-ptxas suppression,
failure-context replay, and monotonic progress milestones declared as data.

A definition declares::

    log:
      heartbeat_seconds: 45
      progress_milestones:
        - {contains: "collecting ", progress: 15, message: "Resolving {label} dependencies"}

Nothing engine-specific lives here; ``{label}`` is substituted from the host.
"""

from __future__ import annotations

import asyncio
import time
from asyncio.subprocess import PIPE, STDOUT
from collections import deque
from typing import Any, Dict, List, Mapping, Optional

from backend.engines.python.installer import utcnow


def _is_verbose_compiler_line(line: str) -> bool:
    text = str(line or "").lstrip().lower()
    if text.startswith("ptxas fatal"):
        return False
    return (
        text.startswith("ptxas info")
        or text.startswith("ptxas warning")
        or "bytes spill" in text
        or "bytes stack frame" in text
    )


def _is_failure_context_line(line: str) -> bool:
    text = str(line or "").lower()
    return any(
        marker in text
        for marker in (
            "failed:",
            "fatal error:",
            "ptxas fatal",
            " error:",
            "killed signal",
            "out of memory",
            "no space left on device",
            "subcommand failed",
        )
    )


def progress_milestone(
    milestones: List[Mapping[str, Any]], label: str, line: str, current: int
) -> tuple[int, str]:
    """Monotonic integer progress milestone for one installer output line."""
    text = str(line or "").lower()
    for milestone in reversed(milestones):
        if str(milestone.get("contains") or "").lower() in text:
            message = str(milestone.get("message") or "").replace("{label}", label)
            return max(current, int(milestone.get("progress") or 0)), message
    return current, "Working"


async def _update_milestone_progress(host, progress: float, message: str) -> None:
    """Publish rounded milestones without allowing regressions."""
    if host._progress_task_id:
        from backend.operations.progress import get_progress_manager

        task = get_progress_manager().get_task(host._progress_task_id) or {}
        progress = max(float(task.get("progress") or 0), float(progress))
    await host._update_progress_task(round(progress), message)


async def run_streamed_log(
    host,
    argv,
    operation: str,
    *,
    heartbeat_seconds: float,
    milestones: List[Mapping[str, Any]],
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    append: bool = True,
) -> int:
    argv_list = [str(item) for item in argv]
    mode = "a" if append else "w"
    with open(host._log_path, mode, encoding="utf-8") as log_file:
        log_file.write(f"[{utcnow()}] {host.label} {operation}: {' '.join(argv_list)}\n")
    await host._broadcast_log_line(f"$ {' '.join(argv_list)}")
    process = await asyncio.create_subprocess_exec(
        *argv_list,
        stdout=PIPE,
        stderr=STDOUT,
        cwd=cwd,
        env=env,
    )
    host._track_process(process)
    recent_lines: deque = deque(maxlen=12)
    failure_context: deque = deque(maxlen=120)

    async def stream() -> None:
        if process.stdout is None:
            return
        heartbeat = float(heartbeat_seconds or 0)
        last_output = time.monotonic()
        with open(host._log_path, "a", encoding="utf-8", buffering=1) as handle:
            while True:
                try:
                    if heartbeat > 0:
                        chunk = await asyncio.wait_for(process.stdout.readline(), timeout=heartbeat)
                    else:
                        chunk = await process.stdout.readline()
                except asyncio.TimeoutError:
                    if process.returncode is not None:
                        break
                    silent = int(time.monotonic() - last_output)
                    notice = (
                        f"[install] still compiling; no new output for {silent}s "
                        "(large CUDA files can take several minutes each)"
                    )
                    handle.write(notice + "\n")
                    await host._broadcast_log_line(notice)
                    continue
                if not chunk:
                    break
                last_output = time.monotonic()
                line = chunk.decode("utf-8", errors="replace")
                handle.write(line)
                clean_line = line.rstrip("\r\n")
                recent_lines.append(clean_line)
                if _is_failure_context_line(clean_line):
                    failure_context.extend(recent_lines)
                # Verbose ptxas output can be hundreds of lines per kernel.
                # Broadcasting each one stalls nvcc on a full stdout pipe.
                if not _is_verbose_compiler_line(clean_line):
                    await host._broadcast_log_line(clean_line)
                    current = host._stream_progress
                    value, message = progress_milestone(milestones, host.label, clean_line, current)
                    if value != current:
                        host._stream_progress = value
                        await _update_milestone_progress(host, value, message)

    await asyncio.gather(process.wait(), stream())
    host._clear_active_process()
    returncode = process.returncode or 0
    if returncode != 0 and failure_context:
        # Parallel CUDA jobs can continue printing thousands of ptxas lines
        # after an earlier command fails. Replay the useful diagnostic at the
        # end so it remains visible in both the UI and log-tail API.
        summary = ["--- captured failure context ---"]
        seen = set()
        for context_line in failure_context:
            if context_line and context_line not in seen:
                summary.append(context_line)
                seen.add(context_line)
        with open(host._log_path, "a", encoding="utf-8") as handle:
            handle.write("\n" + "\n".join(summary) + "\n")
        for context_line in summary:
            await host._broadcast_log_line(context_line)
    return returncode
