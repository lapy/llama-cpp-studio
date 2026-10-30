"""Bounded incremental usage observation without changing forwarded bytes."""

from __future__ import annotations

import json
import zlib


class UsageObserver:
    def __init__(self, sse: bool = False):
        self.sse = sse
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.saw_token = False
        self.finished = False
        self._line = bytearray()
        self._discard_line = False
        self._frame = bytearray()
        self._discard_frame = False
        self._string = False
        self._escape = False
        self._word = bytearray()
        self._expect_usage = False
        self._capture: bytearray | None = None
        self._depth = 0
        self._json_depth = 0

    def _usage(self, payload):
        if not isinstance(payload, dict):
            return
        if payload.get("type") in ("response.completed", "response.failed", "response.incomplete"):
            self.finished = True
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            response = payload.get("response")
            usage = response.get("usage") if isinstance(response, dict) else None
        if isinstance(usage, dict):
            for names, attr in (
                (("prompt_tokens", "input_tokens"), "input_tokens"),
                (("completion_tokens", "output_tokens"), "output_tokens"),
            ):
                for name in names:
                    value = usage.get(name)
                    if (
                        isinstance(value, int)
                        and not isinstance(value, bool)
                        and 0 <= value <= 10**12
                    ):
                        setattr(self, attr, value)
                        break
        choices = payload.get("choices") or []
        if not isinstance(choices, list):
            choices = []
        self.saw_token |= any(
            isinstance(choice, dict)
            and isinstance(choice.get("delta"), dict)
            and bool(
                choice["delta"].get("content")
                or choice["delta"].get("tool_calls")
                or choice["delta"].get("reasoning_content")
            )
            for choice in choices
        )
        self.saw_token |= payload.get("type") in (
            "response.output_text.delta",
            "response.reasoning_text.delta",
        ) and bool(payload.get("delta"))

    def feed(self, chunk: bytes):
        if self.sse:
            for part in chunk.splitlines(keepends=True):
                # A single SSE data line cannot grow without bound. Discard
                # oversized events, then resume at the next newline so final
                # usage remains observable even after multi-megabyte output.
                if not self._discard_line:
                    if len(self._line) + len(part) <= 262144:
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
        # Scan JSON lexically and capture only usage objects, skipping all
        # content strings. This also handles arbitrarily large JSON responses.
        for ch in chunk:
            if not self._string:
                if ch in (123, 91):
                    self._json_depth += 1
                elif ch in (125, 93):
                    self._json_depth -= 1
            if self._capture is not None:
                self._capture.append(ch)
                if len(self._capture) > 65536:
                    self._capture = None
                    self._expect_usage = False
                    continue
                if not self._string:
                    if ch == 123:
                        self._depth += 1
                    elif ch == 125:
                        self._depth -= 1
                        if self._depth == 0:
                            try:
                                self._usage({"usage": json.loads(self._capture)})
                            except (ValueError, UnicodeDecodeError):
                                pass
                            self._capture = None
            if self._string:
                if self._escape:
                    self._escape = False
                elif ch == 92:
                    self._escape = True
                elif ch == 34:
                    self._string = False
                    self._expect_usage = bytes(self._word) == b"usage" and self._json_depth == 1
                elif len(self._word) < 32:
                    self._word.append(ch)
                continue
            if ch == 34:
                self._string = True
                self._word.clear()
            elif self._expect_usage:
                if ch == 123:
                    self._capture = bytearray(b"{")
                    self._depth = 1
                    self._expect_usage = False
                elif ch not in b" \t\r\n:":
                    self._expect_usage = False

    def _event(self, line: bytes):
        line = line.rstrip(b"\r\n")
        if not line:
            if self._frame and not self._discard_frame:
                if bytes(self._frame).strip() == b"[DONE]":
                    self.finished = True
                try:
                    self._usage(json.loads(self._frame))
                except (ValueError, UnicodeDecodeError):
                    pass
            self._frame.clear()
            self._discard_frame = False
        elif line.startswith(b"data:") and not self._discard_frame:
            data = line[5:].lstrip(b" ")
            if len(self._frame) + len(data) + 1 > 262144:
                self._frame.clear()
                self._discard_frame = True
            else:
                if self._frame:
                    self._frame.extend(b"\n")
                self._frame.extend(data)

    def finish(self):
        if self.sse:
            if self._line and not self._discard_line:
                self._event(bytes(self._line))
            self._line.clear()
            self._event(b"")


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
