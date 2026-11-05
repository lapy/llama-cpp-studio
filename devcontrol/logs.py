"""Bounded log access for application, build, and model logs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, Optional

from .compose import Compose
from .errors import GuardError, NotFoundError
from .http_client import model_logs
from .redact import redact_text

MAX_LINES = 5000
MAX_BYTES = 2_000_000


def _tail_file(path: Path, *, max_bytes: int = 256_000) -> str:
    try:
        size = path.stat().st_size
        with open(path, "rb") as handle:
            handle.seek(max(0, size - max_bytes))
            data = handle.read()
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _filter(lines: Iterable[str], *, search: str = "", level: str = "") -> list[str]:
    values = list(lines)
    if search:
        needle = search.lower()
        values = [line for line in values if needle in line.lower()]
    if level == "error":
        values = [
            line
            for line in values
            if "error" in line.lower() or "traceback" in line.lower() or " stderr:" in line.lower()
        ]
    elif level == "warn":
        values = [
            line
            for line in values
            if "error" in line.lower() or "warn" in line.lower() or "traceback" in line.lower()
        ]
    elif level and level not in {"info", "debug"}:
        raise GuardError("level must be error, warn, info, or debug")
    return values


def _bounded(lines: list[str], *, tail: int, max_bytes: int) -> tuple[list[str], bool, int]:
    selected = lines[-tail:] if tail > 0 else lines
    encoded = "\n".join(selected).encode("utf-8", errors="replace")
    truncated = len(selected) < len(lines) or len(encoded) > max_bytes
    if len(encoded) > max_bytes:
        selected = encoded[-max_bytes:].decode("utf-8", errors="replace").splitlines()
    return selected, truncated, len(encoded)


def _build_log_files(data_dir: Path) -> list[Path]:
    roots = [
        data_dir / "logs",
        data_dir / "llama-cpp" / "builds",
        data_dir / "audio-cpp" / "builds",
        data_dir / "engines",
    ]
    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for current, _dirs, files in os.walk(root):
            depth = len(Path(current).relative_to(root).parts)
            if depth > 4:
                _dirs[:] = []
                continue
            for name in files:
                if name.endswith((".log", ".txt")) and (
                    "build" in name or "install" in name or name == "cuda_install.log"
                ):
                    found.append(Path(current) / name)

    def mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    return sorted(found, key=mtime, reverse=True)[:8]


def read_logs(
    compose: Compose,
    *,
    source: str = "app",
    service: str = "all",
    model: str = "",
    tail: int = 200,
    since: str = "",
    search: str = "",
    level: str = "",
    max_bytes: int = 256_000,
) -> dict[str, Any]:
    source = (source or "app").strip().lower()
    tail = max(1, min(int(tail), MAX_LINES))
    max_bytes = max(4096, min(int(max_bytes), MAX_BYTES))
    if source == "model":
        if not model:
            raise GuardError("source=model requires a model id")
        base = f"http://127.0.0.1:{compose.config.backend_port}"
        result = model_logs(
            base,
            model,
            token=compose.config.api_token,
            limit=tail,
            search=search,
            level=level,
        )
        if result.status == 404:
            raise NotFoundError(f"model is not running: {model}")
        if not result.ok:
            raise GuardError(f"could not read model logs: {result.error or result.status}")
        payload = result.data if isinstance(result.data, dict) else {}
        lines = [str(line) for line in payload.get("lines") or []]
        lines = _filter(lines, search=search, level=level)
        selected, truncated, size = _bounded(lines, tail=tail, max_bytes=max_bytes)
        return {
            "source": "model",
            "model": model,
            "lines": [redact_text(line) for line in selected],
            "truncated": truncated,
            "bytes": size,
        }
    if source == "build":
        files = _build_log_files(compose.config.data)
        if not files:
            # A build may still be running in the backend process. Point the
            # caller at app logs instead of pretending no evidence exists.
            app = read_logs(
                compose,
                source="app",
                service="backend",
                tail=tail,
                since=since,
                search=search,
                level=level,
                max_bytes=max_bytes,
            )
            app["source"] = "build"
            app["fallback"] = "no build log files found; returned backend app logs"
            return app
        chunks: list[str] = []
        used: list[str] = []
        for path in files:
            text = _tail_file(path, max_bytes=max_bytes // max(1, len(files)))
            if not text:
                continue
            used.append(str(path))
            chunks.append(f"===== {path} =====")
            chunks.extend(text.splitlines())
        lines = _filter(chunks, search=search, level=level)
        selected, truncated, size = _bounded(lines, tail=tail, max_bytes=max_bytes)
        return {
            "source": "build",
            "files": used,
            "lines": [redact_text(line) for line in selected],
            "truncated": truncated,
            "bytes": size,
        }
    if source == "control":
        raise GuardError("source=control is served by the journal command")
    if source != "app":
        raise GuardError("source must be app, build, model, or control")
    services = ["backend", "frontend"] if service in {"all", ""} else [service]
    if any(name not in {"backend", "frontend"} for name in services):
        raise GuardError("service must be backend, frontend, or all")
    chunks = []
    used = []
    for name in services:
        result = compose.logs(name, tail=tail, since=since)
        used.append(name)
        chunks.append(f"===== compose service: {name} =====")
        text = result.stdout or result.stderr
        chunks.extend(text.splitlines())
    lines = _filter(chunks, search=search, level=level)
    selected, truncated, size = _bounded(lines, tail=tail, max_bytes=max_bytes)
    return {
        "source": "app",
        "services": used,
        "lines": [redact_text(line) for line in selected],
        "truncated": truncated,
        "bytes": size,
    }
