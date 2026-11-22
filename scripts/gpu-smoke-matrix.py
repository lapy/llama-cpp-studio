#!/usr/bin/env python3
"""Manual GPU smoke-matrix runner for a Studio inference endpoint.

The runner is deliberately standard-library only and makes no hardware
assumption: it is meant to be pointed at a host the operator has already
started, usually over loopback or a private network. It reads a JSON matrix,
calls ``GET /v1/models`` once, calls chat completions for every model id and
declared alias, and writes one JSON report with status, latency and usage.

Example::

    python scripts/gpu-smoke-matrix.py --matrix gpu-matrix.json --output smoke.json

The key is read from the environment variable named by the matrix
(``key_env``), never from the matrix itself and never written to the report.
``--dry-run`` validates the matrix and prints the planned requests without any
network call, which is the safe first step on a new host.

Matrix example::

    {
      "base_url": "http://127.0.0.1:2000",
      "key_env": "STUDIO_INFERENCE_KEY",
      "prompt": "Reply with a short description of the sky.",
      "max_tokens": 32,
      "timeout_seconds": 120,
      "models": [
        "org/model-name",
        {"id": "org/other-model", "aliases": ["friendly-alias"], "prompt": "Say hello."}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
DEFAULT_KEY_ENV = "STUDIO_INFERENCE_KEY"
DEFAULT_PROMPT = "Reply with a short description of the sky."
DEFAULT_MAX_TOKENS = 32
DEFAULT_TIMEOUT_SECONDS = 120.0
MAX_MODELS = 200
MAX_ALIASES = 10
MAX_PROMPT_CHARS = 1000
MAX_BODY_BYTES = 1024 * 1024

# Ignore proxy environment variables: this runner talks to one explicit host.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class MatrixError(ValueError):
    """The matrix is malformed or outside the supported bounds."""


def _bounded_text(value: Any, field: str, *, limit: int) -> str:
    text = str(value or "").strip()
    if not text:
        raise MatrixError(f"{field} must be a non-empty string")
    if len(text) > limit:
        raise MatrixError(f"{field} must be at most {limit} characters")
    return text


def _bounded_int(value: Any, field: str, *, default: int, minimum: int, maximum: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatrixError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise MatrixError(f"{field} must be between {minimum} and {maximum}")
    return value


def _bounded_float(
    value: Any, field: str, *, default: float, minimum: float, maximum: float
) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixError(f"{field} must be a number")
    number = float(value)
    if not minimum <= number <= maximum:
        raise MatrixError(f"{field} must be between {minimum:g} and {maximum:g}")
    return number


def normalize_matrix(document: Any) -> dict[str, Any]:
    """Validate a decoded matrix and return the normalized form."""
    if not isinstance(document, dict):
        raise MatrixError("matrix must be a JSON object")

    base_url = _bounded_text(document.get("base_url"), "base_url", limit=2048).rstrip("/")
    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise MatrixError("base_url must be an http(s) URL with a host")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise MatrixError("base_url must not carry a query, fragment, or credentials")

    key_env = str(document.get("key_env") or DEFAULT_KEY_ENV).strip()
    if not key_env or len(key_env) > 128:
        raise MatrixError("key_env must be a non-empty environment variable name")

    prompt = _bounded_text(document.get("prompt", DEFAULT_PROMPT), "prompt", limit=MAX_PROMPT_CHARS)
    max_tokens = _bounded_int(
        document.get("max_tokens"),
        "max_tokens",
        default=DEFAULT_MAX_TOKENS,
        minimum=1,
        maximum=4096,
    )
    timeout_seconds = _bounded_float(
        document.get("timeout_seconds"),
        "timeout_seconds",
        default=DEFAULT_TIMEOUT_SECONDS,
        minimum=1.0,
        maximum=3600.0,
    )

    raw_models = document.get("models")
    if not isinstance(raw_models, list) or not raw_models:
        raise MatrixError("models must be a non-empty list")
    if len(raw_models) > MAX_MODELS:
        raise MatrixError(f"models must contain at most {MAX_MODELS} entries")

    targets: list[dict[str, Any]] = []
    for position, entry in enumerate(raw_models, start=1):
        if isinstance(entry, str):
            model_id = _bounded_text(entry, f"models[{position}]", limit=300)
            aliases: list[str] = []
            entry_prompt = prompt
        elif isinstance(entry, dict):
            model_id = _bounded_text(
                entry.get("id", entry.get("model")),
                f"models[{position}].id",
                limit=300,
            )
            raw_aliases = entry.get("aliases") or []
            if not isinstance(raw_aliases, list):
                raise MatrixError(f"models[{position}].aliases must be a list")
            if len(raw_aliases) > MAX_ALIASES:
                raise MatrixError(
                    f"models[{position}].aliases must contain at most {MAX_ALIASES} entries"
                )
            aliases = []
            for alias in raw_aliases:
                name = _bounded_text(alias, f"models[{position}].aliases", limit=300)
                if name != model_id and name not in aliases:
                    aliases.append(name)
            entry_prompt = prompt
            if entry.get("prompt") is not None:
                entry_prompt = _bounded_text(
                    entry.get("prompt"), f"models[{position}].prompt", limit=MAX_PROMPT_CHARS
                )
        else:
            raise MatrixError(f"models[{position}] must be a string or an object")
        targets.append({"model_id": model_id, "aliases": aliases, "prompt": entry_prompt})

    return {
        "base_url": base_url,
        "key_env": key_env,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "timeout_seconds": timeout_seconds,
        "targets": targets,
    }


def load_matrix(path: str | Path) -> dict[str, Any]:
    """Read and validate a matrix file."""
    try:
        # ``utf-8-sig`` accepts a Windows editor's BOM as well as plain UTF-8.
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise MatrixError(f"cannot read matrix: {exc}") from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MatrixError(f"matrix is not valid JSON: {exc}") from exc
    return normalize_matrix(document)


def _requested_names(target: dict[str, Any]) -> list[str]:
    return [target["model_id"], *target["aliases"]]


def dry_run_plan(matrix: dict[str, Any]) -> dict[str, Any]:
    """Describe the calls the runner would make, without any network access."""
    requests = []
    for target in matrix["targets"]:
        for name in _requested_names(target):
            requests.append(
                {
                    "method": "POST",
                    "path": "/v1/chat/completions",
                    "model": name,
                    "model_id": target["model_id"],
                    "alias": None if name == target["model_id"] else name,
                }
            )
    return {
        "dry_run": True,
        "base_url": matrix["base_url"],
        "key_env": matrix["key_env"],
        "models_endpoint": {"method": "GET", "path": "/v1/models"},
        "requests": requests,
        "request_count": len(requests),
    }


def _decode_json(raw: bytes) -> Any:
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except json.JSONDecodeError:
        return None


def _request_json(
    method: str,
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    api_key: str = "",
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """One bounded HTTP call. Never returns or reports the request headers."""
    headers = {
        "Accept": "application/json",
        "User-Agent": "studio-gpu-smoke-matrix/1",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    started = time.perf_counter()

    def elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 3)

    try:
        with _OPENER.open(request, timeout=timeout) as response:
            raw = response.read(MAX_BODY_BYTES)
            return {
                "status": int(getattr(response, "status", 200)),
                "body": _decode_json(raw),
                "latency_ms": elapsed_ms(),
                "error": None,
            }
    except urllib.error.HTTPError as exc:
        return {
            "status": int(exc.code),
            "body": None,
            "latency_ms": elapsed_ms(),
            "error": f"http_{exc.code}",
        }
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return {
            "status": None,
            "body": None,
            "latency_ms": elapsed_ms(),
            "error": type(exc).__name__,
        }


def _model_ids(body: Any) -> list[str]:
    data = body.get("data") if isinstance(body, dict) else None
    ids: list[str] = []
    for row in data if isinstance(data, list) else []:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            ids.append(row["id"])
    return ids[:500]


def _usage(body: Any) -> dict[str, float | int] | None:
    usage = body.get("usage") if isinstance(body, dict) else None
    if not isinstance(usage, dict):
        return None
    selected: dict[str, float | int] = {}
    for name, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        selected[str(name)[:64]] = value
        if len(selected) >= 16:
            break
    return selected or None


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_matrix(matrix: dict[str, Any]) -> dict[str, Any]:
    """Execute the matrix and return the JSON-serializable report."""
    base_url = matrix["base_url"]
    timeout = float(matrix["timeout_seconds"])
    api_key = os.getenv(matrix["key_env"], "").strip()

    models_response = _request_json(
        "GET", f"{base_url}/v1/models", api_key=api_key, timeout=timeout
    )
    advertised = _model_ids(models_response["body"])

    results = []
    for target in matrix["targets"]:
        for name in _requested_names(target):
            payload = {
                "model": name,
                "messages": [{"role": "user", "content": target["prompt"]}],
                "max_tokens": matrix["max_tokens"],
                "temperature": 0,
                "stream": False,
            }
            response = _request_json(
                "POST",
                f"{base_url}/v1/chat/completions",
                payload=payload,
                api_key=api_key,
                timeout=timeout,
            )
            results.append(
                {
                    "model_id": target["model_id"],
                    "requested_model": name,
                    "alias": None if name == target["model_id"] else name,
                    "listed": name in advertised,
                    "status": response["status"],
                    "latency_ms": response["latency_ms"],
                    "usage": _usage(response["body"]),
                    "error": response["error"],
                }
            )

    succeeded = sum(1 for row in results if row["status"] == 200)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_timestamp(),
        "base_url": base_url,
        "key_env": matrix["key_env"],
        "models_endpoint": {
            "status": models_response["status"],
            "latency_ms": models_response["latency_ms"],
            "model_count": len(advertised),
            "models": advertised,
            "error": models_response["error"],
        },
        "results": results,
        "summary": {
            "requests": len(results),
            "succeeded": succeeded,
            "failed": len(results) - succeeded,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gpu-smoke-matrix.py",
        description=(
            "Call GET /v1/models and chat completions for every model id and alias "
            "in a JSON matrix, then write one JSON report."
        ),
        epilog=(
            "Matrix keys: base_url, key_env, prompt, max_tokens, timeout_seconds, "
            "models. A model entry is either an id string or an object with id, "
            "aliases, and an optional prompt. The API key is read from the "
            "environment variable named by key_env and is never written to the "
            "report. Exit status is 0 only when every chat call and the models "
            "endpoint succeed, 1 on a failed call, and 2 on an invalid matrix."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--matrix",
        required=True,
        help="path to the JSON smoke matrix",
    )
    parser.add_argument(
        "--output",
        help="write the JSON report to this path instead of stdout",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the matrix and print the planned requests without any network call",
    )
    arguments = parser.parse_args(argv)

    try:
        matrix = load_matrix(arguments.matrix)
    except MatrixError as exc:
        print(f"gpu-smoke-matrix: {exc}", file=sys.stderr)
        return 2

    if arguments.dry_run:
        print(json.dumps(dry_run_plan(matrix), indent=2))
        return 0

    report = run_matrix(matrix)
    rendered = json.dumps(report, indent=2) + "\n"
    if arguments.output:
        destination = Path(arguments.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"wrote {destination}", file=sys.stderr)
    else:
        print(rendered, end="")
    healthy = report["models_endpoint"]["status"] == 200 and report["summary"]["failed"] == 0
    return 0 if healthy else 1


if __name__ == "__main__":
    raise SystemExit(main())
