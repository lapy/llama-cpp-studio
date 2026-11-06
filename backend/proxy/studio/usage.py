"""Bounded incremental usage observation without changing forwarded bytes.

Framing (this module) is separate from protocol adapters (``observation.py``).
The observer recognizes OpenAI chat/completions/Responses, native llama.cpp and
SGLang event shapes, preserves total/cached/processed input counts, and keeps
transport completion, protocol terminal state and usage authority distinct.

The nonstream path is a bounded lexical scanner. It captures only selected
objects (usage/timings/meta_info) and native scalar counters, tracks container
closure, and never buffers a response body.
"""

from __future__ import annotations

import json
import re
import zlib

from backend.proxy.studio.observation import (
    UsageCounts,
    native_llama_counts,
    openai_usage_counts,
    output_event,
    responses_usage_counts,
    sglang_meta_counts,
    terminal_event,
)

_MAX_LINE = 262144
_MAX_CAPTURE = 65536
_MAX_DEPTH = 256
_NATIVE_SCALARS = {"tokens_evaluated", "tokens_predicted", "tokens_cached"}
_CAPTURE_PATHS = {"usage", "response.usage", "timings", "meta_info", "metrics"}
_NUMBER_RE = re.compile(rb"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")


class UsageObserver:
    def __init__(self, sse: bool = False):
        self.sse = sse
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.input_tokens_cached: int | None = None
        self.input_tokens_cache_write: int | None = None
        self.input_tokens_processed: int | None = None
        self.output_reasoning_tokens: int | None = None
        self.native_timing: dict = {}
        self.usage_adapter = ""
        self.usage_reported = False
        self.usage_events = 0
        self.usage_output_stale = False
        self.observation_gap = False
        self._outputs_since_usage = 0
        self.saw_token = False
        self.saw_text = False
        self.output_event_count = 0
        self.finished = False
        self.terminal_kind = "none"
        self.terminal_error = False
        self.error_after_usage = False
        self.reasons: list[str] = []
        # SSE framing
        self._line = bytearray()
        self._discard_line = False
        self._frame = bytearray()
        self._discard_frame = False
        self._event_name = ""
        # Nonstream lexical state
        self._stack: list[dict] = []
        self._in_string = False
        self._string_escape = False
        self._string_is_key = False
        self._string_buf = bytearray()
        self._string_unicode_remaining = 0
        self._pending_key: str | None = None
        self._in_primitive = False
        self._primitive_buf = bytearray()
        self._primitive_key: str | None = None
        self._top_scalars: dict[str, int] = {}
        self._capture: dict | None = None
        self._json_started = False
        self._json_root_closed = False
        self._json_roots = 0
        self._json_trailing = False
        self._json_too_deep = False
        self._json_grammar_error = False
        self._json_ended = False

    # -- normalized evidence -------------------------------------------------

    def _reason(self, reason: str) -> None:
        if reason not in self.reasons:
            self.reasons.append(reason)

    @property
    def structural_complete(self) -> bool:
        return bool(
            self._json_started
            and self._json_roots == 1
            and self._json_root_closed
            and not self._json_trailing
            and not self._json_too_deep
            and not self._json_grammar_error
        )

    @property
    def usage_authoritative(self) -> bool:
        """Whether the observed usage is the protocol's final accounting."""
        if not self.usage_reported or self.terminal_error:
            return False
        if self.sse:
            if not self.finished:
                return False
            return self.terminal_kind in (
                "done",
                "completed",
                "incomplete",
                "failed",
                "native_finished",
            )
        return self.structural_complete

    @property
    def output_authoritative(self) -> bool:
        """Output counts are final only when no output followed the snapshot."""
        return bool(
            self.usage_authoritative and not self.usage_output_stale and not self.observation_gap
        )

    def _apply_counts(self, counts: UsageCounts) -> None:
        """Replace the current snapshot; never merge fields across snapshots.

        A final snapshot that omits cache detail must not inherit an earlier
        partial snapshot's cached count. The last reported snapshot wins for
        every field it owns.
        """
        reported = any(
            value is not None
            for value in (
                counts.input_total,
                counts.input_cached,
                counts.input_cache_write,
                counts.input_processed,
                counts.output_total,
                counts.output_reasoning,
            )
        ) or bool(counts.native_timing)
        if not reported:
            return
        self.input_tokens = counts.input_total
        self.input_tokens_cached = counts.input_cached
        self.input_tokens_cache_write = counts.input_cache_write
        # Processed prompt work is engine timing evidence, not a usage field.
        # An ordinary usage object must not clear a timing value extracted
        # before it; serialization order cannot change the result.
        if counts.input_processed is not None:
            self.input_tokens_processed = counts.input_processed
        self.output_tokens = counts.output_total
        self.output_reasoning_tokens = counts.output_reasoning
        if counts.native_timing:
            self.native_timing.update(counts.native_timing)
        if counts.adapter:
            self.usage_adapter = counts.adapter
        self.usage_reported = True
        self.usage_events += 1
        # This snapshot is now the latest evidence. Output that follows it
        # makes its output counts stale for final settlement.
        self._outputs_since_usage = 0
        self.usage_output_stale = False
        # A usage snapshot after a gap is fresh evidence for the remaining
        # stream; only a gap that follows the last snapshot keeps output stale.
        self.observation_gap = False

    def _record_timing(self, values: dict) -> None:
        """Merge engine timing evidence without depending on field order."""
        for name in (
            "prompt_n",
            "prompt_ms",
            "prompt_per_second",
            "predicted_n",
            "predicted_ms",
            "predicted_per_second",
        ):
            item = values.get(name)
            if isinstance(item, (int, float)) and not isinstance(item, bool):
                self.native_timing[name] = item
        processed = values.get("prompt_n")
        if isinstance(processed, int) and not isinstance(processed, bool):
            self.input_tokens_processed = processed

    def _native_from_scalars(self) -> None:
        if not self._top_scalars:
            return
        counts = native_llama_counts(
            {
                **self._top_scalars,
                "timings": self.native_timing,
            }
        )
        self._apply_counts(counts)

    def _observe_output(self, payload: dict) -> None:
        generated, visible = output_event(payload)
        if generated:
            self.saw_token = True
            self.output_event_count += 1
            self._outputs_since_usage += 1
            if self.usage_events:
                self.usage_output_stale = True
        if visible:
            self.saw_text = True

    def _terminal(self, kind: str, *, event_name: str = "") -> None:
        if kind == "error" or event_name == "error":
            self.terminal_error = True
            self.terminal_kind = "error"
            if self.usage_events:
                self.error_after_usage = True
            self.finished = True
            return
        if kind in ("done", "completed", "incomplete", "failed", "native_finished"):
            self.terminal_kind = kind
            self.finished = True

    def _payload(self, payload, *, event_name: str = "") -> None:
        if not isinstance(payload, dict):
            return
        # Output is observed before usage so a snapshot in the same event owns
        # that output; output in a later event marks an earlier snapshot stale.
        self._observe_output(payload)
        # Usage snapshots are cumulative; the latest snapshot replaces the
        # previous one. Summing a continuous snapshot would double count.
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._apply_counts(openai_usage_counts(usage))
        response = payload.get("response")
        if isinstance(response, dict):
            response_usage = response.get("usage")
            if isinstance(response_usage, dict):
                self._apply_counts(responses_usage_counts(response_usage))
            meta = response.get("meta_info")
            if isinstance(meta, dict):
                self._apply_counts(sglang_meta_counts(meta))
        meta = payload.get("meta_info")
        if isinstance(meta, dict):
            self._apply_counts(sglang_meta_counts(meta))
        if isinstance(payload.get("timings"), dict):
            self._record_timing(payload["timings"])
        metrics = payload.get("metrics")
        if isinstance(metrics, dict):
            for name, value in list(metrics.items())[:32]:
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.native_timing[f"metrics.{name}"] = value
        if any(name in payload for name in _NATIVE_SCALARS):
            counts = native_llama_counts(payload)
            if counts.input_total is not None or counts.output_total is not None:
                self._apply_counts(counts)
        kind = terminal_event(payload)
        if kind is not None or event_name == "error":
            self._terminal(kind or "error", event_name=event_name)

    # -- SSE framing ---------------------------------------------------------

    def _event(self, line: bytes) -> None:
        line = line.rstrip(b"\r\n")
        if not line:
            if self._frame and not self._discard_frame:
                value = bytes(self._frame).strip()
                if value == b"[DONE]":
                    self._terminal("done")
                else:
                    try:
                        self._payload(json.loads(value), event_name=self._event_name)
                    except (ValueError, UnicodeDecodeError):
                        self.observation_gap = True
                        self._reason("malformed_event_json")
            elif self._discard_frame:
                self.observation_gap = True
                self._reason("oversized_event_discarded")
            self._frame.clear()
            self._discard_frame = False
            self._event_name = ""
        elif line.startswith(b"event:") and not self._discard_frame:
            self._event_name = line[6:].strip().decode("latin-1", "replace")[:64]
        elif line.startswith(b"data:") and not self._discard_frame:
            data = line[5:].lstrip(b" ")
            if len(self._frame) + len(data) + 1 > _MAX_LINE:
                self._frame.clear()
                self._discard_frame = True
                self.observation_gap = True
                self._reason("oversized_event_discarded")
            else:
                if self._frame:
                    self._frame.extend(b"\n")
                self._frame.extend(data)

    # -- nonstream lexical scanner ------------------------------------------

    def _path_for(self, key: str | None) -> str:
        """Path with array boundaries preserved.

        ``{"choices":[{"usage": ...}]}`` must not look like top-level
        ``usage``. Array segments carry a ``[]`` marker so only the declared
        capture paths (top-level usage/timings/meta_info/metrics and
        ``response.usage``) can match.
        """
        parts: list[str] = []
        for frame in self._stack:
            if frame["kind"] == "object":
                if frame.get("key"):
                    parts.append(str(frame["key"]))
            elif frame.get("key"):
                parts.append(f"{frame['key']}[]")
            else:
                parts.append("[]")
        if key:
            parts.append(key)
        return ".".join(parts)

    def _start_capture(self, path: str) -> None:
        self._capture = {
            "path": path,
            "buf": bytearray(b"{"),
            "depth": 1,
            "in_string": False,
            "escape": False,
        }

    def _finish_capture(self) -> None:
        capture = self._capture
        self._capture = None
        if not capture:
            return
        try:
            value = json.loads(capture["buf"])
        except (ValueError, UnicodeDecodeError):
            # A captured object that does not parse means the document is not
            # structurally valid; usage extracted before it cannot be final.
            self._grammar_error("malformed_json_object")
            return
        if not isinstance(value, dict):
            return
        path = capture["path"]
        if path == "usage":
            self._apply_counts(openai_usage_counts(value))
        elif path == "response.usage":
            self._apply_counts(responses_usage_counts(value))
        elif path == "meta_info":
            self._apply_counts(sglang_meta_counts(value))
        elif path == "timings":
            self._record_timing(value)
        elif path == "metrics":
            for name, item in list(value.items())[:32]:
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    self.native_timing[f"metrics.{name}"] = item

    def _grammar_error(self, reason: str = "malformed_json") -> None:
        self._json_grammar_error = True
        self._reason(reason)

    def _begin_value(self) -> bool:
        """Record the start of one JSON value; return False on invalid grammar."""
        if not self._json_started:
            self._json_started = True
        if not self._stack:
            if self._json_root_closed:
                self._json_root_closed = False
            return True
        top = self._stack[-1]
        if top["kind"] == "object":
            if not top["expect_value"] or top["expect_colon"]:
                self._grammar_error()
                return False
            self._pending_key = None
        elif not top["expect_value"]:
            self._grammar_error()
            return False
        self._pending_key = None
        top["expect_value"] = False
        top["allow_close"] = False
        return True

    def _close_container(self, frame: dict) -> None:
        if not self._stack:
            self._json_roots += 1
            self._json_root_closed = True

    def _finish_primitive(self) -> None:
        if not self._in_primitive:
            return
        text = bytes(self._primitive_buf)
        key = self._primitive_key
        self._in_primitive = False
        self._primitive_buf.clear()
        self._primitive_key = None
        if not text:
            self._grammar_error()
            return
        if text in (b"true", b"false", b"null") or _NUMBER_RE.fullmatch(text):
            pass
        else:
            self._grammar_error("invalid_json_literal")
            return
        if key is not None and len(self._stack) == 1:
            try:
                value = int(text)
            except (TypeError, ValueError):
                value = None
            if value is not None and 0 <= value <= 10**12:
                self._top_scalars[key] = value
        if not self._stack:
            self._json_roots += 1
            self._json_root_closed = True

    def _scan_byte(self, ch: int) -> None:
        if self._capture is not None:
            capture = self._capture
            capture["buf"].append(ch)
            if len(capture["buf"]) > _MAX_CAPTURE:
                self._capture = None
                self._grammar_error("usage_capture_overflow")
                return
            if capture["in_string"]:
                if capture["escape"]:
                    capture["escape"] = False
                elif ch == 92:
                    capture["escape"] = True
                elif ch == 34:
                    capture["in_string"] = False
            elif ch == 34:
                capture["in_string"] = True
            elif ch in (123, 91):
                capture["depth"] += 1
            elif ch in (125, 93):
                capture["depth"] -= 1
                if capture["depth"] == 0:
                    self._finish_capture()
            return

        if self._in_string:
            if self._string_unicode_remaining:
                if ch not in b"0123456789abcdefABCDEF":
                    self._grammar_error("invalid_escape")
                    self._string_unicode_remaining = 0
                else:
                    self._string_unicode_remaining -= 1
                return
            if self._string_escape:
                self._string_escape = False
                if ch == 117:  # u
                    self._string_unicode_remaining = 4
                elif ch not in b'"\\/bfnrt':
                    self._grammar_error("invalid_escape")
                return
            if ch == 92:
                self._string_escape = True
            elif ch < 0x20:
                self._grammar_error("invalid_string_control")
            elif ch == 34:
                self._in_string = False
                if self._string_is_key:
                    top = self._stack[-1]
                    top["expect_key"] = False
                    top["expect_colon"] = True
                    top["allow_close"] = False
                    self._pending_key = bytes(self._string_buf).decode("utf-8", "replace")[:64]
                # Value strings were recorded when they started.
            elif len(self._string_buf) < 64:
                self._string_buf.append(ch)
            return

        if self._in_primitive:
            if ch in b",}] \t\r\n":
                self._finish_primitive()
                if ch in b" \t\r\n":
                    return
                # Fall through so the delimiter updates the parent frame.
            else:
                if len(self._primitive_buf) < 64:
                    self._primitive_buf.append(ch)
                else:
                    self._grammar_error("json_token_too_long")
                return

        if ch in b" \t\r\n":
            return
        if self._json_root_closed:
            if ch in (123, 91):
                # Another JSON root is measurable but never one complete
                # document; structural completeness requires exactly one.
                self._json_root_closed = False
            else:
                self._json_trailing = True
                return
        if ch == 34:  # quote
            top = self._stack[-1] if self._stack else None
            key_role = bool(
                top
                and top["kind"] == "object"
                and top["expect_key"]
                and not top["expect_colon"]
                and not top["expect_value"]
            )
            if not key_role and not self._begin_value():
                return
            self._in_string = True
            self._string_escape = False
            self._string_unicode_remaining = 0
            self._string_is_key = key_role
            self._string_buf.clear()
            return
        if ch == 123:  # {
            key = self._pending_key
            if len(self._stack) >= _MAX_DEPTH:
                self._json_too_deep = True
                return
            if not self._begin_value():
                return
            path = self._path_for(key)
            if path in _CAPTURE_PATHS:
                # The captured object is consumed byte-for-byte here; do not
                # push a frame the main parser would later close twice.
                self._start_capture(path)
                return
            self._stack.append(
                {
                    "kind": "object",
                    "key": key,
                    "expect_key": True,
                    "expect_colon": False,
                    "expect_value": False,
                    "allow_close": True,
                }
            )
            return
        if ch == 91:  # [
            key = self._pending_key
            if len(self._stack) >= _MAX_DEPTH:
                self._json_too_deep = True
                return
            if not self._begin_value():
                return
            self._stack.append(
                {
                    "kind": "array",
                    "key": key,
                    "expect_key": False,
                    "expect_colon": False,
                    "expect_value": True,
                    "allow_close": True,
                }
            )
            return
        if ch in (125, 93):  # } ]
            if not self._stack:
                self._grammar_error("mismatched_json_container")
                return
            top = self._stack[-1]
            if ch == 125:
                if top["kind"] != "object":
                    self._grammar_error("mismatched_json_container")
                    return
                if top["expect_colon"] or top["expect_value"]:
                    self._grammar_error("missing_json_value")
                    return
                if top["expect_key"] and not top["allow_close"]:
                    self._grammar_error("trailing_json_comma")
                    return
            else:
                if top["kind"] != "array":
                    self._grammar_error("mismatched_json_container")
                    return
                if top["expect_value"] and not top["allow_close"]:
                    self._grammar_error("missing_json_value")
                    return
            self._stack.pop()
            self._close_container(top)
            return
        if ch == 58:  # :
            top = self._stack[-1] if self._stack else None
            if not top or top["kind"] != "object" or not top["expect_colon"]:
                self._grammar_error("invalid_json_colon")
                return
            top["expect_colon"] = False
            top["expect_value"] = True
            return
        if ch == 44:  # ,
            if not self._stack:
                self._grammar_error("invalid_json_comma")
                return
            top = self._stack[-1]
            if top["kind"] == "object":
                if top["expect_value"] or top["expect_colon"] or top["allow_close"]:
                    if top["allow_close"]:
                        self._grammar_error("trailing_json_comma")
                    else:
                        self._grammar_error("invalid_json_comma")
                    return
                top["expect_key"] = True
                top["allow_close"] = False
            else:
                if top["expect_value"] or top["allow_close"]:
                    self._grammar_error("invalid_json_comma")
                    return
                top["expect_value"] = True
                top["allow_close"] = False
            return
        # Primitive value start.
        top = self._stack[-1] if self._stack else None
        key = self._pending_key
        if not self._begin_value():
            return
        if (
            top is not None
            and top["kind"] == "object"
            and len(self._stack) == 1
            and key in _NATIVE_SCALARS
        ):
            self._primitive_key = key
        self._in_primitive = True
        self._primitive_buf = bytearray([ch])

    def feed(self, chunk: bytes):
        if self.sse:
            for part in chunk.splitlines(keepends=True):
                # A single SSE data line cannot grow without bound. Discard
                # oversized events, then resume at the next newline so final
                # usage remains observable even after multi-megabyte output.
                if not self._discard_line:
                    if len(self._line) + len(part) <= _MAX_LINE:
                        self._line.extend(part)
                    else:
                        self._line.clear()
                        self._discard_line = True
                        self._discard_frame = True
                if part.endswith(b"\n"):
                    if not self._discard_line:
                        self._event(bytes(self._line))
                    self._line.clear()
                    self._discard_line = False
            return
        for ch in chunk:
            self._scan_byte(ch)

    def finish(self):
        if self.sse:
            if self._line and not self._discard_line:
                self._event(bytes(self._line))
            self._line.clear()
            self._event(b"")
            if not self.finished:
                self._reason("missing_terminal_event")
            return
        if self._in_primitive:
            self._finish_primitive()
        self._native_from_scalars()
        self._json_ended = True
        if self._in_string:
            self._reason("truncated_json")
        if self._capture is not None:
            self._reason("truncated_object")
            self._capture = None
        if self._json_roots > 1:
            self._reason("multiple_json_roots")
        if self._json_started and (not self._json_root_closed or self._json_roots != 1):
            self._reason("truncated_json")
        if self._json_trailing:
            self._reason("trailing_json_data")
        if self._json_too_deep:
            self._reason("json_depth_limit")
        if self._json_grammar_error:
            self._reason("malformed_json")


class UsageContentDecoder:
    """Decode only the observation copy; forwarded response bytes stay raw.

    Compressed observation has a bounded expansion budget. Unsupported or
    excessive encodings remain unknown usage and retain quota reservations.
    """

    def __init__(self, encoding: str):
        self.enabled = encoding in ("", "identity", "gzip", "deflate")
        self.decoder = (
            zlib.decompressobj(16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS)
            if encoding in ("gzip", "deflate")
            else None
        )
        self.remaining = 64 * 1024 * 1024

    @property
    def complete(self):
        return self.enabled and (self.decoder is None or self.decoder.eof)

    def chunks(self, data: bytes):
        if not self.enabled:
            return
        if self.decoder is None:
            yield data
            return
        try:
            while data and self.enabled:
                decoded = self.decoder.decompress(data, min(262144, self.remaining + 1))
                data = self.decoder.unconsumed_tail
                self.remaining -= len(decoded)
                if self.remaining < 0:
                    self.enabled = False
                    return
                if decoded:
                    yield decoded
        except zlib.error:
            self.enabled = False
