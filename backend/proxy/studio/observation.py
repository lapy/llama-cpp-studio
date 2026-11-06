"""Versioned, protocol-aware usage and timing observation for the Studio proxy.

One normalized record feeds request history, registry statistics, accounting
and benchmarks. Field provenance is explicit: an unknown cache count is not a
zero, a reservation is not measured consumption, and an engine timing metric is
not a proxy-observed decode rate.

The contract is additive. New fields are versioned with ``SCHEMA_VERSION`` so
historical rows remain readable as legacy observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

SCHEMA_VERSION = 1

PROTOCOL_UNKNOWN = "unknown"
PROTOCOL_OPENAI_CHAT = "openai.chat"
PROTOCOL_OPENAI_COMPLETIONS = "openai.completions"
PROTOCOL_OPENAI_EMBEDDINGS = "openai.embeddings"
PROTOCOL_OPENAI_RESPONSES = "openai.responses"
PROTOCOL_OPENAI_AUDIO = "openai.audio"
PROTOCOL_NATIVE_LLAMA = "native.llama"
PROTOCOL_NATIVE_SGLANG = "native.sglang"
PROTOCOL_NATIVE_AUDIO = "native.audio"

USAGE_AUTHORITATIVE = "authoritative"
USAGE_PARTIAL = "partial"
USAGE_MISSING = "missing"
USAGE_NOT_DISPATCHED = "not_dispatched"

# Formula and timing-origin descriptions exposed by the API and rendered as
# labels/tooltips. Keeping them here prevents the UI from inventing synonyms.
METRIC_SEMANTICS: dict[str, dict[str, str]] = {
    "ttft_ms": {
        "label": "Proxy TTFT",
        "definition": "Request start to first recognized generated-output event, including Studio overhead.",
        "origin": "proxy event observation; approximate",
    },
    "time_to_first_visible_text_ms": {
        "label": "Time to first visible text",
        "definition": "Request start to first non-empty user-visible text delta.",
        "origin": "proxy event observation; may follow reasoning or tool output",
    },
    "output_interval_ms": {
        "label": "Proxy output interval",
        "definition": "First generated-output event to last generated-output event.",
        "origin": "proxy event observation; excludes TTFT and terminal cleanup",
    },
    "proxy_decode_tokens_per_second": {
        "label": "Proxy decode rate (estimate)",
        "definition": "(output tokens - 1) / output interval, for a verified single sequence.",
        "origin": "proxy event observation; approximate when events bundle or omit tokens",
    },
    "request_effective_tokens_per_second": {
        "label": "Effective request throughput",
        "definition": "Output tokens / complete request duration, including TTFT.",
        "origin": "proxy wall clock",
    },
    "lifetime_output_tokens_per_second": {
        "label": "Lifetime wall-clock throughput",
        "definition": "Output tokens divided by wall time since this generation's stats window began.",
        "origin": "includes idle time by design; not generation speed",
    },
    "rolling_10s_output_tokens_per_second": {
        "label": "Rolling 10-second output throughput",
        "definition": "Completion-attributed output tokens in the trailing 10 seconds / 10 seconds.",
        "origin": "includes idle time by design",
    },
    "rolling_60s_output_tokens_per_second": {
        "label": "Rolling 60-second output throughput",
        "definition": "Completion-attributed output tokens in the trailing 60 seconds / 60 seconds.",
        "origin": "includes idle time by design",
    },
    "engine_decode_tokens_per_second": {
        "label": "Engine decode rate",
        "definition": "Engine-reported decode rate with its original numerator and timing semantics.",
        "origin": "engine clock and boundaries; not directly comparable to proxy rate",
    },
    "engine_prefill_tokens_per_second": {
        "label": "Engine prefill throughput",
        "definition": "Engine processed prompt tokens / engine prefill duration.",
        "origin": "engine clock; logical input / TTFT is not prefill speed",
    },
    "input_cache": {
        "label": "Input cache hit ratio",
        "definition": "Sum of cached input / sum of total input over the cache-known cohort.",
        "origin": "protocol cache subset; unknown cache reporting is excluded, never treated as zero",
    },
}

TIMING_ALLOWLIST = (
    "prompt_n",
    "prompt_ms",
    "prompt_per_second",
    "predicted_n",
    "predicted_ms",
    "predicted_per_second",
    "n_prompt_tokens",
    "n_gen_tokens",
    "t_prompt_processing",
    "t_token_generation",
)


def _int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 0 <= value <= 10**12 else None


def _first_int(container: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = _int(container.get(name))
        if value is not None:
            return value
    return None


def _detail(container: Mapping[str, Any], *names: str) -> Mapping[str, Any]:
    for name in names:
        value = container.get(name)
        if isinstance(value, Mapping):
            return value
    return {}


@dataclass(frozen=True)
class UsageCounts:
    """Protocol-normalized counts before finality and cache validation."""

    input_total: int | None = None
    input_cached: int | None = None
    input_cache_write: int | None = None
    input_processed: int | None = None
    output_total: int | None = None
    output_reasoning: int | None = None
    adapter: str = ""
    native_timing: Mapping[str, Any] = field(default_factory=dict)


def openai_usage_counts(usage: Mapping[str, Any], *, adapter: str = "openai") -> UsageCounts:
    """Normalize OpenAI chat/completions/embeddings usage."""
    input_total = _first_int(usage, "prompt_tokens", "input_tokens")
    output_total = _first_int(usage, "completion_tokens", "output_tokens")
    details = _detail(usage, "prompt_tokens_details", "input_tokens_details")
    cached = _first_int(details, "cached_tokens")
    cache_write = _first_int(details, "cache_write_tokens")
    output_details = _detail(usage, "completion_tokens_details", "output_tokens_details")
    reasoning = _first_int(output_details, "reasoning_tokens")
    return UsageCounts(
        input_total=input_total,
        input_cached=cached,
        input_cache_write=cache_write,
        output_total=output_total,
        output_reasoning=reasoning,
        adapter=adapter,
    )


def responses_usage_counts(usage: Mapping[str, Any]) -> UsageCounts:
    """Normalize the Responses API usage envelope."""
    counts = openai_usage_counts(usage, adapter="openai.responses")
    return UsageCounts(
        input_total=counts.input_total,
        input_cached=counts.input_cached,
        input_cache_write=counts.input_cache_write,
        output_total=counts.output_total,
        output_reasoning=counts.output_reasoning,
        adapter=counts.adapter,
    )


def native_llama_counts(payload: Mapping[str, Any]) -> UsageCounts:
    """Normalize native llama-family completion reporting.

    ``tokens_evaluated`` is logical prompt total; ``timings.prompt_n`` is
    processing work. ``tokens_cached`` is a cache subset. Never substitute one
    for the other.
    """
    input_total = _first_int(payload, "tokens_evaluated", "prompt_tokens")
    output_total = _first_int(payload, "tokens_predicted", "completion_tokens")
    cached = _first_int(payload, "tokens_cached")
    timings = payload.get("timings")
    timing_values: dict[str, Any] = {}
    processed = None
    if isinstance(timings, Mapping):
        timing_values = {
            name: timings[name]
            for name in TIMING_ALLOWLIST
            if isinstance(timings.get(name), (int, float))
            and not isinstance(timings.get(name), bool)
        }
        processed = _int(timings.get("prompt_n"))
    return UsageCounts(
        input_total=input_total,
        input_cached=cached,
        input_processed=processed,
        output_total=output_total,
        adapter="native.llama",
        native_timing=timing_values,
    )


def sglang_meta_counts(meta: Mapping[str, Any]) -> UsageCounts:
    """Normalize SGLang ``meta_info`` (native /generate path)."""
    input_total = _first_int(meta, "prompt_tokens", "input_tokens")
    output_total = _first_int(meta, "completion_tokens", "output_tokens")
    cached = _first_int(meta, "cached_tokens")
    if cached is None:
        details = _detail(meta, "prompt_tokens_details", "input_tokens_details")
        cached = _first_int(details, "cached_tokens")
    reasoning = _first_int(meta, "reasoning_tokens")
    timing_values: dict[str, Any] = {}
    for name in ("e2e_latency", "first_token_latency", "queue_time", "prefill_latency"):
        value = meta.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            timing_values[name] = value
    return UsageCounts(
        input_total=input_total,
        input_cached=cached,
        output_total=output_total,
        output_reasoning=reasoning,
        adapter="native.sglang",
        native_timing=timing_values,
    )


# ---------------------------------------------------------------------------
# Event-shape detection shared by SSE framing and nonstream lexical parsing
# ---------------------------------------------------------------------------


def _nonempty(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value)
    if isinstance(value, (list, tuple, dict)):
        return bool(value)
    return False


def output_event(payload: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return ``(generated_output, visible_text)`` for one protocol event."""
    generated = False
    visible = False
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, Mapping):
                continue
            delta = choice.get("delta")
            if isinstance(delta, Mapping):
                content = delta.get("content")
                if _nonempty(content):
                    generated = visible = True
                if _nonempty(delta.get("text")):
                    generated = visible = True
                if _nonempty(delta.get("tool_calls")):
                    generated = True
                if _nonempty(delta.get("function_call")):
                    generated = True
                if _nonempty(delta.get("reasoning_content")) or _nonempty(delta.get("reasoning")):
                    generated = True
            # Legacy completion streams carry the text on the choice itself.
            if _nonempty(choice.get("text")):
                generated = visible = True
    kind = payload.get("type")
    if isinstance(kind, str) and kind.startswith("response."):
        delta = payload.get("delta")
        if kind.endswith(".delta") and _nonempty(delta):
            generated = True
            if kind in (
                "response.output_text.delta",
                "response.refusal.delta",
            ):
                visible = True
    # Native llama completion and SGLang /generate streaming content. Responses
    # done events also carry a full `text`; they are not new token events.
    if not (isinstance(kind, str) and kind.startswith("response.")):
        for name in ("content", "text"):
            if _nonempty(payload.get(name)):
                generated = visible = True
    return generated, visible


def terminal_event(payload: Mapping[str, Any]) -> str | None:
    """Return a terminal kind when the event ends the protocol stream."""
    kind = payload.get("type")
    if isinstance(kind, str):
        if kind == "error" or kind.endswith(".error"):
            return "error"
        if kind == "response.completed":
            return "completed"
        if kind == "response.incomplete":
            return "incomplete"
        if kind == "response.failed":
            return "failed"
    status = payload.get("status")
    if payload.get("object") == "response" and isinstance(status, str):
        if status == "completed":
            return "completed"
        if status == "incomplete":
            return "incomplete"
        if status in ("failed", "cancelled"):
            return status
    if isinstance(payload.get("error"), Mapping):
        return "error"
    if payload.get("stop") is True:
        return "native_finished"
    meta = payload.get("meta_info")
    if isinstance(meta, Mapping) and meta.get("finish_reason") is not None:
        return "native_finished"
    if payload.get("finish_reason") is not None and payload.get("text") is not None:
        return "native_finished"
    return None


def protocol_for_path(path: str, *, sse: bool = False, content_type: str = "") -> str:
    """Classify the protocol from the upstream route and response shape."""
    lowered = (content_type or "").lower()
    if "/audio" in path or "audio" in lowered:
        return PROTOCOL_OPENAI_AUDIO
    if "/responses" in path:
        return PROTOCOL_OPENAI_RESPONSES
    if path.endswith("/embeddings"):
        return PROTOCOL_OPENAI_EMBEDDINGS
    if "/chat/completions" in path:
        return PROTOCOL_OPENAI_CHAT
    if path.endswith("/completions"):
        return PROTOCOL_OPENAI_COMPLETIONS
    if path.endswith("/completion") or "/generate" in path or "/native" in path:
        return PROTOCOL_NATIVE_LLAMA if path.endswith("/completion") else PROTOCOL_NATIVE_SGLANG
    return PROTOCOL_UNKNOWN


def normalize_cache(
    total: int | None, cached: int | None
) -> tuple[int | None, int | None, list[str]]:
    """Validate the cache subset and derive logical uncached input."""
    reasons: list[str] = []
    if total is None:
        return None, None, reasons
    if cached is None:
        reasons.append("cache_not_reported")
        return None, None, reasons
    if cached > total:
        reasons.append("cache_exceeds_total")
        return None, None, reasons
    return cached, total - cached, reasons


def _rate(numerator: int, seconds: float) -> float | None:
    if numerator <= 0 or seconds <= 0:
        return None
    return numerator / seconds


def finalize_observation(
    *,
    engine: str = "",
    engine_revision: str = "",
    engine_build: str = "",
    protocol: str = PROTOCOL_UNKNOWN,
    input_total: int | None = None,
    input_cached: int | None = None,
    input_cache_write: int | None = None,
    input_processed: int | None = None,
    output_total: int | None = None,
    output_reasoning: int | None = None,
    native_timing: Mapping[str, Any] | None = None,
    usage_reported: bool = False,
    usage_authoritative: bool = False,
    output_authoritative: bool | None = None,
    preflight_input: int | None = None,
    reserved_input: int | None = None,
    reserved_output: int | None = None,
    observed_input: int | None = None,
    observed_output: int | None = None,
    dispatched: bool = True,
    transport_complete: bool = False,
    structural_complete: bool = False,
    terminal_kind: str = "none",
    terminal_error: bool = False,
    error_after_usage: bool = False,
    reasons: Iterable[str] = (),
    first_output_ms: float | None = None,
    last_output_ms: float | None = None,
    first_visible_text_ms: float | None = None,
    terminal_ms: float | None = None,
    ttft_ms: float | None = None,
    first_byte_ms: float | None = None,
    latency_ms: float | None = None,
    output_event_count: int = 0,
    bundled_output_events: bool = False,
    sequence_scope: str = "single",
) -> dict[str, Any]:
    """Derive the normalized observation record persisted with a request.

    The function is pure and deterministic: callers pass bounded numeric
    evidence and monotonic-derived durations, and receive JSON-safe fields.
    """
    quality = list(dict.fromkeys(str(reason) for reason in reasons if reason))

    input_authoritative = bool(usage_authoritative and dispatched)
    output_final = bool(
        dispatched
        and usage_authoritative
        and (output_authoritative if output_authoritative is not None else True)
    )
    if dispatched and not output_final:
        quality.append(
            "output_blocked_by_incomplete_transport"
            if not transport_complete
            else "output_stale_after_usage"
        )
    cache_reasons: list[str] = []
    cached, uncached, cache_reasons = normalize_cache(input_total, input_cached)
    quality.extend(cache_reasons)

    if not dispatched:
        usage_status = USAGE_NOT_DISPATCHED
    elif (input_authoritative and input_total is not None) or (
        output_final and output_total is not None
    ):
        usage_status = USAGE_AUTHORITATIVE if output_final else USAGE_PARTIAL
    elif observed_input is not None or observed_output is not None or usage_reported:
        usage_status = USAGE_PARTIAL
    elif output_event_count > 0:
        usage_status = USAGE_PARTIAL
        quality.append("output_event_observed_without_usage")
    else:
        usage_status = USAGE_MISSING

    # Input: an authoritative total wins; otherwise a preflight engine count
    # remains useful evidence about dispatched input and is labelled as such.
    if input_authoritative and input_total is not None:
        input_tokens = input_total
        input_provenance = "engine_final"
    elif preflight_input is not None and dispatched:
        input_tokens = preflight_input
        input_provenance = "preflight"
        quality.append("input_from_preflight")
    else:
        input_tokens = None
        input_provenance = "unknown"
    output_tokens = output_total if output_final else None

    if input_authoritative and input_total is not None and input_cached is None:
        quality.append("cache_unknown")
    if not usage_reported and dispatched:
        quality.append("usage_not_reported")
    if terminal_error and error_after_usage:
        quality.append("terminal_error_after_usage")
    elif terminal_error:
        quality.append("terminal_error")

    output_interval_ms = None
    if first_output_ms is not None and last_output_ms is not None:
        output_interval_ms = max(0.0, last_output_ms - first_output_ms)

    decode_rate = None
    decode_quality = None
    if sequence_scope != "single":
        decode_quality = "incompatible_sequence_scope"
    elif output_tokens is None:
        decode_quality = "output_unknown"
    elif output_tokens <= 1:
        decode_quality = "insufficient_tokens"
    elif output_interval_ms is None or output_interval_ms <= 0:
        decode_quality = "unmeasurable_interval"
    elif output_event_count <= 1:
        decode_quality = "single_output_event"
    else:
        decode_rate = _rate(output_tokens - 1, output_interval_ms / 1000)
        if bundled_output_events:
            decode_quality = "bundled_events_approximate"
        else:
            decode_quality = "conventional_n_minus_one"
    if decode_quality in (
        "incompatible_sequence_scope",
        "single_output_event",
        "unmeasurable_interval",
    ):
        quality.append(decode_quality)

    effective_rate = None
    if output_tokens is not None and latency_ms is not None and latency_ms > 0:
        effective_rate = _rate(output_tokens, latency_ms / 1000)

    generation_ms = output_interval_ms
    return {
        "schema_version": SCHEMA_VERSION,
        "engine": engine,
        "engine_revision": engine_revision,
        "engine_build": engine_build,
        "protocol": protocol,
        "usage_status": usage_status,
        "usage_complete": bool(input_authoritative and output_final),
        "usage_source": (
            "engine"
            if (input_authoritative and input_total is not None)
            or (output_final and output_total is not None)
            else "preflight"
            if input_provenance == "preflight"
            else "unknown"
        ),
        "observation_reasons": quality,
        "input_tokens_total": input_total if input_authoritative else None,
        "input_tokens_cached": cached if input_authoritative else None,
        "input_tokens_cache_write": input_cache_write if input_authoritative else None,
        "input_tokens_uncached": uncached if input_authoritative else None,
        "input_tokens_processed": input_processed if input_authoritative else None,
        "input_tokens_preflight": preflight_input,
        "input_tokens_reserved": reserved_input,
        "input_tokens": input_tokens,
        "input_tokens_provenance": input_provenance,
        "output_tokens_total": output_total if output_final else None,
        "output_tokens_reasoning": output_reasoning if output_final else None,
        "output_tokens": output_tokens,
        "output_tokens_reserved": reserved_output,
        "observed_input_tokens": observed_input,
        "observed_output_tokens": observed_output,
        "native_timing": dict(native_timing or {}),
        "engine_metrics": dict(native_timing or {}),
        "cache_detail_complete": bool(
            input_authoritative and input_total is not None and cached is not None
        ),
        "transport_complete": bool(transport_complete),
        "structural_complete": bool(structural_complete),
        "terminal_kind": terminal_kind,
        "first_byte_ms": first_byte_ms,
        "ttft_ms": ttft_ms,
        "first_output_ms": first_output_ms,
        "last_output_ms": last_output_ms,
        "first_visible_text_ms": first_visible_text_ms,
        "time_to_first_visible_text_ms": first_visible_text_ms,
        "output_interval_ms": output_interval_ms,
        "generation_ms": generation_ms,
        "terminal_ms": terminal_ms,
        "proxy_decode_tokens_per_second": decode_rate,
        "decode_rate_quality": decode_quality,
        "output_tokens_per_second": decode_rate,
        "request_effective_tokens_per_second": effective_rate,
        "output_event_count": output_event_count,
        "bundled_output_events": bool(bundled_output_events),
        "sequence_scope": sequence_scope,
    }


def summarize_cache(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate cache coverage over a cache-known cohort.

    A request with unknown cache detail is never treated as a zero-cache
    measurement. The ratio is ``sum(cached) / sum(total)`` over the cohort.
    """
    known_total = 0
    cached_total = 0
    known = 0
    rows = list(rows)
    for row in rows:
        total = _int(row.get("input_tokens_total"))
        cached = _int(row.get("input_tokens_cached"))
        if total is None or cached is None or cached > total:
            continue
        known += 1
        known_total += total
        cached_total += cached
    return {
        "requests": len(rows),
        "cache_known_requests": known,
        "cache_unknown_requests": len(rows) - known,
        "input_tokens_cache_known": known_total,
        "input_tokens_cached": cached_total,
        "cache_hit_ratio": (cached_total / known_total if known_total and known else None),
        "cache_coverage": (known / len(rows)) if rows else None,
    }
