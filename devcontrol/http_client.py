"""Tiny stdlib HTTP/SSE client for the development instance."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from .redact import redact_text


@dataclass
class HttpResult:
    status: int
    data: Any = None
    text: str = ""
    error: str = ""

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300 and not self.error


def _opener() -> urllib.request.OpenerDirector:
    # Never let an ambient proxy intercept loopback development traffic.
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def request_json(
    url: str,
    *,
    token: str = "",
    timeout: float = 4.0,
    method: str = "GET",
) -> HttpResult:
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers, method=method)
    try:
        with _opener().open(request, timeout=timeout) as response:
            body = response.read(2_000_000).decode("utf-8", errors="replace")
            return HttpResult(status=int(response.status), data=_parse(body), text=body)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(200_000).decode("utf-8", errors="replace")
        except OSError:
            pass
        return HttpResult(
            status=int(exc.code),
            data=_parse(body),
            text=body,
            error=redact_text(f"HTTP {exc.code}: {body[:500]}"),
        )
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        return HttpResult(status=0, error=redact_text(f"{type(exc).__name__}: {exc}"))


def request_text(
    url: str,
    *,
    token: str = "",
    timeout: float = 4.0,
    max_bytes: int = 1_000_000,
) -> HttpResult:
    # Vite's dev server answers 404 to a narrow text/plain Accept header even
    # though a browser request succeeds; use a browser-like accept for health.
    headers = {"Accept": "text/html,application/xhtml+xml,application/json;q=0.9,*/*;q=0.8"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with _opener().open(request, timeout=timeout) as response:
            body = response.read(max_bytes).decode("utf-8", errors="replace")
            return HttpResult(status=int(response.status), text=body)
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read(200_000).decode("utf-8", errors="replace")
        except OSError:
            pass
        return HttpResult(status=int(exc.code), text=body, error=f"HTTP {exc.code}")
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        return HttpResult(status=0, error=redact_text(f"{type(exc).__name__}: {exc}"))


def _parse(body: str) -> Any:
    try:
        return json.loads(body)
    except ValueError:
        return None


def task_snapshot(base_url: str, *, token: str, timeout: float = 3.0) -> list[dict[str, Any]]:
    """Read the initial task snapshot from the SSE stream and close it.

    The endpoint is a stream, not a polling API. Reading only the first
    ``task_snapshot`` event keeps this bounded and does not leave a subscriber
    behind in the application process.
    """

    url = base_url.rstrip("/") + "/api/events"
    headers = {"Accept": "text/event-stream"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with _opener().open(request, timeout=timeout) as response:
            event = ""
            data_lines: list[str] = []
            consumed = 0
            while consumed < 512_000:
                raw = response.readline()
                if not raw:
                    break
                consumed += len(raw)
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
                elif not line:
                    if event == "task_snapshot" and data_lines:
                        try:
                            payload = json.loads("\n".join(data_lines))
                        except ValueError:
                            return []
                        tasks = payload.get("tasks") if isinstance(payload, dict) else None
                        return [item for item in tasks or [] if isinstance(item, dict)]
                    event = ""
                    data_lines = []
            return []
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
        return []


def model_logs(
    base_url: str,
    model: str,
    *,
    token: str,
    limit: int = 500,
    search: str = "",
    level: str = "",
) -> HttpResult:
    encoded = urllib.parse.quote(model, safe="")
    query = {"limit": str(max(1, min(limit, 5000)))}
    if search:
        query["search"] = search
    if level:
        query["level"] = level
    url = f"{base_url.rstrip('/')}/api/proxy/models/{encoded}/logs?" + urllib.parse.urlencode(query)
    return request_json(url, token=token)
