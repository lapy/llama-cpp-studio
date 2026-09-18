"""Live progress + debug log for engine ``--help`` parameter scans.

Scan progress reuses the same SSE ``build_progress`` channel as engine builds so
the existing notifications tray and collapsible log UI work without a new task
type on the frontend beyond ``param_scan``.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import queue
import shlex
import threading
import uuid
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence

from backend.engine_registry import get_engine_spec
from backend.logging_config import get_logger
from backend.progress_manager import get_progress_manager

logger = get_logger(__name__)

PARAM_SCAN_TASK_TYPE = "param_scan"
_FLUSH_BATCH = 80
_MAX_COVERAGE_FLAGS = 250

_current: contextvars.ContextVar[Optional["ParamScanSession"]] = contextvars.ContextVar(
    "param_scan_session", default=None
)


def get_param_scan_session() -> Optional["ParamScanSession"]:
    return _current.get()


def engine_scan_label(engine: str) -> str:
    spec = get_engine_spec(engine)
    return spec.label if spec else engine or "engine"


def _quote_argv(argv: Sequence[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def _field_text(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        return ",".join(_field_text(item) for item in value)
    text = str(value).replace("\n", " ").strip()
    if not text:
        return "-"
    if any(ch.isspace() for ch in text) or "=" in text:
        return json.dumps(text, ensure_ascii=False)
    return text


def format_extract(event: str, **fields: Any) -> str:
    parts = [f"EXTRACT {event}"]
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_field_text(value)}")
    return " ".join(parts)


def format_param_row(row: dict, *, section: Optional[str] = None) -> str:
    sid = section or row.get("section_id") or "options"
    flags = ",".join(str(flag) for flag in (row.get("flags") or []) if flag)
    options = row.get("options") or []
    option_values = [
        str(opt.get("value"))
        for opt in options
        if isinstance(opt, dict) and opt.get("value") is not None
    ]
    return format_extract(
        "accept",
        section=sid,
        flags=flags or row.get("primary_flag"),
        key=row.get("key"),
        kind=row.get("value_kind"),
        ui=row.get("type"),
        default=row.get("default"),
        reserved=bool(row.get("reserved")),
        options=option_values or None,
    )


class ParamScanSession:
    """Collects a scan log and streams it to the progress tray."""

    def __init__(
        self,
        *,
        engine: str,
        version: Optional[str] = None,
        model_id: Optional[str] = None,
        description: Optional[str] = None,
        task_id: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.engine = engine
        self.version = version
        self.model_id = model_id
        self.task_id = task_id or f"scan_{engine}_{uuid.uuid4().hex[:8]}"
        self.lines: List[str] = []
        self._pending: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._finished = False
        self._flush_scheduled = False
        self.progress = 1.0
        self.stage = "start"
        self.message = "Starting parameter scan"
        try:
            self._loop = loop or asyncio.get_running_loop()
        except RuntimeError:
            self._loop = loop
        spec_label = engine_scan_label(engine)
        version_bit = f" {version}" if version else ""
        self.description = description or f"Scan {spec_label} CLI parameters{version_bit}"
        self._token: Optional[contextvars.Token] = None

    def attach(self) -> "ParamScanSession":
        self._token = _current.set(self)
        pm = get_progress_manager()
        pm.create_task(
            PARAM_SCAN_TASK_TYPE,
            self.description,
            {
                "engine": self.engine,
                "version": self.version,
                "model_id": self.model_id,
                "stage": self.stage,
            },
            task_id=self.task_id,
        )
        self.log(f"=== Parameter scan: {self.engine} ===")
        if self.version:
            self.log(f"version: {self.version}")
        if self.model_id:
            self.log(f"model_id: {self.model_id}")
        self.log(f"started_at: {datetime.utcnow().isoformat()}Z")
        self.set_stage("start", 4, "Starting parameter scan")
        return self

    def detach(self) -> None:
        if self._token is not None:
            _current.reset(self._token)
            self._token = None

    def __enter__(self) -> "ParamScanSession":
        return self.attach()

    def __exit__(self, exc_type, exc, _tb) -> None:
        try:
            if not self._finished:
                if exc:
                    self.fail(str(exc))
                else:
                    self.complete("Parameter scan finished")
        finally:
            self.detach()
        return False

    def set_stage(self, stage: str, progress: float, message: str) -> None:
        self.stage = stage
        self.progress = max(self.progress, min(99.0, float(progress)))
        self.message = message
        self._emit(flush=True)

    def log(self, line: str) -> None:
        text = str(line).rstrip("\n")
        if not text:
            text = " "
        with self._lock:
            self.lines.append(text)
        self._pending.put(text)
        self._schedule_flush()

    def log_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.log(line)

    def extract(self, event: str, **fields: Any) -> None:
        self.log(format_extract(event, **fields))

    def log_command(
        self,
        argv: Sequence[Any],
        *,
        cwd: Optional[str] = None,
        extra_env: Optional[dict] = None,
        scan_engine: Optional[str] = None,
    ) -> None:
        self.log("----- command -----")
        self.log(f"argv: {_quote_argv(argv)}")
        if cwd:
            self.log(f"cwd: {cwd}")
        if scan_engine:
            self.log(f"scan_engine: {scan_engine}")
        if extra_env:
            keys = ", ".join(sorted(str(key) for key in extra_env))
            self.log(f"extra_env_keys: {keys}")

    def log_capture(
        self,
        argv: Sequence[Any],
        text: str,
        error: Optional[str],
        *,
        label: Optional[str] = None,
    ) -> None:
        body = text or ""
        lines = body.splitlines()
        title = label or _quote_argv(argv)
        self.log(
            f"===== HELP OUTPUT: {title} ({len(lines)} lines, {len(body)} chars) ====="
        )
        if error:
            self.log(f"command_warning: {error}")
        if not body.strip():
            self.log("(empty stdout/stderr)")
        else:
            width = max(4, len(str(len(lines))))
            for index, line in enumerate(lines, start=1):
                self.log(f"H{index:0{width}d}|{line}")
        self.log("===== END HELP OUTPUT =====")

    def log_catalog(self, entry: Optional[dict], *, title: str = "engine catalog") -> None:
        entry = entry or {}
        sections = entry.get("sections") or []
        params = [
            param
            for section in sections
            for param in (section.get("params") or [])
        ]
        error = entry.get("scan_error")
        self.log(
            f"===== CATALOG RESULT: {title} "
            f"({len(params)} params, {len(sections)} sections) ====="
        )
        if error:
            self.log(f"scan_error: {error}")
        if entry.get("binary_path"):
            self.log(f"binary_path: {entry.get('binary_path')}")
        if entry.get("scanned_at"):
            self.log(f"scanned_at: {entry.get('scanned_at')}")
        for section in sections:
            label = section.get("label") or section.get("id") or "options"
            sid = section.get("id") or "options"
            rows = section.get("params") or []
            self.log(f"--- section {sid} ({label}) · {len(rows)} params ---")
            for row in rows:
                self.log(format_param_row(row, section=sid))
        self.log("===== END CATALOG RESULT =====")

    def log_flag_coverage(self, help_text: str, sections: Sequence[dict]) -> None:
        from backend.cli_help_parsers import LONG_FLAG_RE, RESERVED_FLAGS

        help_flags = list(dict.fromkeys(LONG_FLAG_RE.findall(help_text or "")))
        parsed_flags: List[str] = []
        for section in sections or []:
            for row in section.get("params") or []:
                for flag in row.get("flags") or []:
                    if flag and flag not in parsed_flags:
                        parsed_flags.append(flag)
        help_set = set(help_flags)
        parsed_set = set(parsed_flags)
        missing = [flag for flag in help_flags if flag not in parsed_set]
        extra = [flag for flag in parsed_flags if flag not in help_set]
        reserved_missing = [flag for flag in missing if flag in RESERVED_FLAGS]
        interesting_missing = [flag for flag in missing if flag not in RESERVED_FLAGS]
        self.log(
            "===== FLAG COVERAGE "
            f"help={len(help_flags)} parsed={len(parsed_flags)} "
            f"missing={len(missing)} extra={len(extra)} ====="
        )
        if reserved_missing:
            self.log(
                "coverage_reserved_unparsed: "
                + ",".join(reserved_missing[:_MAX_COVERAGE_FLAGS])
            )
        if interesting_missing:
            self.log("coverage_missing_from_catalog (present in help, not parsed):")
            for flag in interesting_missing[:_MAX_COVERAGE_FLAGS]:
                self.log(f"MISSING {flag}")
            if len(interesting_missing) > _MAX_COVERAGE_FLAGS:
                self.log(
                    f"... {len(interesting_missing) - _MAX_COVERAGE_FLAGS} more missing flags"
                )
        else:
            self.log("coverage_missing_from_catalog: none")
        if extra:
            self.log("coverage_extra_parsed (not found as --flags in help text):")
            for flag in extra[:_MAX_COVERAGE_FLAGS]:
                self.log(f"EXTRA {flag}")
        else:
            self.log("coverage_extra_parsed: none")
        self.log("===== END FLAG COVERAGE =====")

    def complete(self, message: str = "Parameter scan complete") -> None:
        if self._finished:
            return
        self.log(f"status: completed ({message})")
        self._finished = True
        self.stage = "done"
        self.progress = 100.0
        self.message = message
        self._emit(flush=True, final_status="completed")

    def fail(self, error: str) -> None:
        if self._finished:
            return
        text = error or "Parameter scan failed"
        self.log(f"status: failed ({text})")
        self._finished = True
        self.stage = "failed"
        self.message = text
        self._emit(flush=True, final_status="failed")

    def finish_from_entry(
        self,
        entry: Optional[dict],
        *,
        profile: Optional[dict] = None,
    ) -> None:
        entry = entry or {}
        if profile is not None:
            self.log_catalog(profile, title="model profile")
        n_params = sum(len(section.get("params") or []) for section in entry.get("sections") or [])
        error = entry.get("scan_error")
        profile_error = (profile or {}).get("scan_error") if profile is not None else None
        if error:
            self.fail(str(error))
            return
        if profile_error:
            self.fail(f"Engine scan ok ({n_params} params); profile failed: {profile_error}")
            return
        self.complete(f"Indexed {n_params} CLI options")

    def _schedule_flush(self) -> None:
        with self._lock:
            if self._finished or self._flush_scheduled:
                return
            self._flush_scheduled = True
        loop = self._loop
        if loop is not None and loop.is_running() and not _on_loop_thread(loop):
            loop.call_soon_threadsafe(self._flush_on_loop)
        else:
            self._flush_on_loop()

    def _flush_on_loop(self) -> None:
        batch: List[str] = []
        while True:
            try:
                batch.append(self._pending.get_nowait())
            except queue.Empty:
                break
            if len(batch) >= _FLUSH_BATCH and not self._finished:
                break
        with self._lock:
            self._flush_scheduled = False
        if batch:
            self._emit(log_lines=batch)
        if not self._finished and not self._pending.empty():
            self._schedule_flush()

    def _emit(
        self,
        *,
        log_lines: Optional[List[str]] = None,
        flush: bool = False,
        final_status: Optional[str] = None,
    ) -> None:
        if flush:
            leftover: List[str] = []
            while True:
                try:
                    leftover.append(self._pending.get_nowait())
                except queue.Empty:
                    break
            if leftover:
                log_lines = list(log_lines or []) + leftover
        payload_lines = log_lines or []

        def _do() -> None:
            try:
                pm = get_progress_manager()
                metadata_update = {"stage": self.stage, "log_lines": payload_lines}
                if final_status == "completed":
                    pm.update_task(
                        self.task_id,
                        progress=100.0,
                        message=self.message,
                        status="completed",
                        metadata_update=metadata_update,
                    )
                elif final_status == "failed":
                    pm.update_task(
                        self.task_id,
                        message=self.message,
                        status="failed",
                        metadata_update=metadata_update,
                    )
                else:
                    pm.update_task(
                        self.task_id,
                        progress=self.progress,
                        message=self.message,
                        metadata_update=metadata_update,
                    )
                pm.emit(
                    "build_progress",
                    {
                        "task_id": self.task_id,
                        "stage": self.stage,
                        "progress": int(self.progress if final_status != "completed" else 100),
                        "message": self.message,
                        "log_lines": payload_lines,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                )
            except Exception as exc:
                logger.debug("param scan progress emit failed: %s", exc)

        loop = self._loop
        if loop is not None and loop.is_running() and not _on_loop_thread(loop):
            loop.call_soon_threadsafe(_do)
        else:
            _do()


def _on_loop_thread(loop: asyncio.AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() is loop
    except RuntimeError:
        return False


def start_param_scan(
    engine: str,
    *,
    version: Optional[str] = None,
    model_id: Optional[str] = None,
    description: Optional[str] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
) -> ParamScanSession:
    existing = get_param_scan_session()
    if existing and not existing._finished:
        return existing
    session = ParamScanSession(
        engine=engine,
        version=version,
        model_id=model_id,
        description=description,
        loop=loop,
    )
    session.attach()
    return session


def trace_extract(event: str, **fields: Any) -> None:
    session = get_param_scan_session()
    if session is None:
        return
    try:
        session.extract(event, **fields)
    except Exception:
        logger.debug("param scan extract log failed", exc_info=True)


def looks_like_cli_option_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if stripped.startswith("--"):
        return True
    return stripped.startswith("-") and "--" in stripped
