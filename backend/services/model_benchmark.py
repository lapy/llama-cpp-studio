"""Bounded local inference benchmark with durable, revision-bound results."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import defaultdict
from typing import Any
from urllib.parse import quote

import httpx

from backend.data_store import DataStore, resolve_proxy_name
from backend.gpu_detector import get_gpu_info
from backend.proxy.llama_swap.client import get_llama_swap_client, shared_proxy_client


_locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


class BenchmarkError(ValueError):
    def __init__(self, code: str, detail: str, status_code: int = 400) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail
        self.status_code = status_code


async def run_model_benchmark(
    store: DataStore,
    model_id: str,
    *,
    prompt: str,
    max_tokens: int,
) -> dict:
    with store.exclusive_documents():
        model = store.get_model(model_id)
        if model is None:
            raise BenchmarkError("BENCHMARK_MODEL_NOT_FOUND", "Model not found.", 404)
        proxy_name = resolve_proxy_name(model)
        config_revision = store.document_revision("models.yaml")
        config_fingerprint = _fingerprint(model.get("config") or {})
    async with _locks[proxy_name]:
        await _require_running(proxy_name)
        stop_sampling = asyncio.Event()
        sample = {"peak_gpu_memory_bytes": None}
        sampler = asyncio.create_task(_sample_gpu_memory(stop_sampling, sample))
        started = time.perf_counter()
        first_token_at: float | None = None
        inference_finished_at: float | None = None
        completion_tokens: int | None = None
        output = ""
        timings: dict[str, Any] = {}
        client = shared_proxy_client(get_llama_swap_client().base_url)
        payload = {
            "model": proxy_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            async with client.stream(
                "POST",
                f"{get_llama_swap_client().base_url}/upstream/{quote(proxy_name, safe='')}/v1/chat/completions",
                json=payload,
                timeout=httpx.Timeout(120.0, connect=10.0),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    event = _sse_json(line)
                    if event is None:
                        continue
                    content = _content(event)
                    if content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                        output += content
                    usage = event.get("usage")
                    if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                        completion_tokens = int(usage["completion_tokens"])
                    event_timings = event.get("timings")
                    if isinstance(event_timings, dict):
                        timings.update(event_timings)
                inference_finished_at = time.perf_counter()
        except httpx.HTTPError as exc:
            raise BenchmarkError(
                "BENCHMARK_UPSTREAM_FAILED",
                "The running model did not complete the benchmark request.",
                502,
            ) from exc
        finally:
            stop_sampling.set()
            await sampler

        if first_token_at is None:
            raise BenchmarkError("BENCHMARK_EMPTY", "The model returned no generated text.", 502)
        finished = inference_finished_at or time.perf_counter()
        total_seconds = max(0.0, finished - started)
        generation_seconds = max(0.000001, finished - first_token_at)
        measured_rate = _number(timings.get("predicted_per_second"))
        if measured_rate is None and completion_tokens is not None:
            measured_rate = completion_tokens / generation_seconds
        result = {
            "id": str(uuid.uuid4()),
            "created_at": time.time(),
            "model_id": model_id,
            "proxy_name": proxy_name,
            "config_revision": config_revision,
            "config_fingerprint": config_fingerprint,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "time_to_first_token_ms": round((first_token_at - started) * 1000, 2),
            "total_seconds": round(total_seconds, 3),
            "completion_tokens": completion_tokens,
            "tokens_per_second": round(measured_rate, 3) if measured_rate is not None else None,
            "peak_observed_gpu_memory_bytes": sample["peak_gpu_memory_bytes"],
            "output_preview": output[:240],
        }
        _save_result(store, result)
        return result


def list_model_benchmarks(store: DataStore, model_id: str, *, limit: int = 20) -> list[dict]:
    rows = store._read_yaml("benchmarks.yaml").get("benchmarks") or []
    selected = [row for row in rows if isinstance(row, dict) and row.get("model_id") == model_id]
    selected.sort(key=lambda row: float(row.get("created_at") or 0), reverse=True)
    return selected[: max(1, min(int(limit), 100))]


async def _require_running(proxy_name: str) -> None:
    try:
        state = await get_llama_swap_client().get_running_models()
    except Exception as exc:
        raise BenchmarkError(
            "BENCHMARK_STATE_UNKNOWN",
            "Running model state could not be verified.",
            503,
        ) from exc
    rows = state.get("running") if isinstance(state, dict) else None
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("model") == proxy_name:
            if str(row.get("state") or "").lower() in {"running", "ready"}:
                return
    raise BenchmarkError(
        "BENCHMARK_MODEL_NOT_RUNNING",
        "Start the model and wait until it is running before benchmarking.",
        409,
    )


async def _sample_gpu_memory(stop: asyncio.Event, sample: dict) -> None:
    while True:
        try:
            info = await asyncio.wait_for(get_gpu_info(), timeout=2.0)
            if stop.is_set():
                return
            used = sum(
                int((gpu.get("memory") or {}).get("used") or 0)
                for gpu in info.get("gpus", [])
                if isinstance(gpu, dict)
            )
            if used:
                sample["peak_gpu_memory_bytes"] = max(sample["peak_gpu_memory_bytes"] or 0, used)
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), timeout=0.5)
            return
        except asyncio.TimeoutError:
            continue


def _sse_json(line: str) -> dict | None:
    text = str(line or "").strip()
    if not text.startswith("data:"):
        return None
    payload = text[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _content(event: dict) -> str:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    return str(delta.get("content") or "") if isinstance(delta, dict) else ""


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _fingerprint(config: dict) -> str:
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def _save_result(store: DataStore, result: dict) -> None:
    def mutate(document):
        rows = [row for row in document.get("benchmarks", []) if isinstance(row, dict)]
        rows.append(result)
        rows.sort(key=lambda row: float(row.get("created_at") or 0), reverse=True)
        document["schema_version"] = 1
        document["benchmarks"] = rows[:100]

    store._mutate("benchmarks.yaml", mutate)
