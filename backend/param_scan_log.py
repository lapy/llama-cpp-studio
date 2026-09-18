"""Parse copyable parameter-scan logs back into raw ``--help`` text.

The notifications tray writes numbered ``H0001|…`` blocks. Regression tests
and later scans can feed that full log (or plain help) through the same unwrap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend.cli_help_parsers import RESERVED_FLAGS, help_flags_for_coverage

HELP_OUTPUT_HEADER_RE = re.compile(
    r"^===== HELP OUTPUT: (?P<title>.+) "
    r"\((?P<lines>\d+) lines, (?P<chars>\d+) chars\) =====$"
)
HELP_LINE_RE = re.compile(r"^H(\d+)\|(.*)$")
HELP_OUTPUT_END = "===== END HELP OUTPUT ====="
_BLOCK_STOP_PREFIXES = (
    "===== END HELP OUTPUT =====",
    "===== FLAG COVERAGE ",
    "===== CATALOG RESULT:",
    "===== END FLAG COVERAGE =====",
    "===== END CATALOG RESULT =====",
)
FLAG_COVERAGE_HEADER_RE = re.compile(
    r"^===== FLAG COVERAGE help=(?P<help>\d+) parsed=(?P<parsed>\d+) "
    r"missing=(?P<missing>\d+) extra=(?P<extra>\d+) =====$"
)


@dataclass(frozen=True)
class HelpCapture:
    title: str
    declared_lines: int
    declared_chars: int
    text: str
    complete: bool

    @property
    def line_count(self) -> int:
        if not self.text:
            return 0
        return len(self.text.splitlines())


@dataclass(frozen=True)
class FlagCoverage:
    help_flags: List[str]
    parsed_flags: List[str]
    missing: List[str]
    extra: List[str]
    reserved_missing: List[str] = field(default_factory=list)
    interesting_missing: List[str] = field(default_factory=list)


def _is_block_stop(line: str) -> bool:
    stripped = line.strip()
    if HELP_OUTPUT_HEADER_RE.match(stripped):
        return True
    return any(stripped.startswith(prefix) for prefix in _BLOCK_STOP_PREFIXES)


def _numbered_lines(lines: Sequence[str], start: int, end: int) -> Dict[int, str]:
    found: Dict[int, str] = {}
    for line in lines[start:end]:
        match = HELP_LINE_RE.match(line)
        if match:
            found[int(match.group(1))] = match.group(2)
    return found


def _all_numbered_lines(lines: Sequence[str]) -> Dict[int, str]:
    return _numbered_lines(lines, 0, len(lines))


def _join_numbered(found: Dict[int, str], declared_lines: int) -> Tuple[str, bool]:
    if declared_lines > 0 and all(index in found for index in range(1, declared_lines + 1)):
        return "\n".join(found[index] for index in range(1, declared_lines + 1)), True
    if not found:
        return "", False
    first, last = min(found), max(found)
    if first == 1 and last == len(found) and all(index in found for index in range(1, last + 1)):
        return "\n".join(found[index] for index in range(1, last + 1)), True
    return "\n".join(found[index] for index in sorted(found)), False


def extract_help_captures(text: str) -> List[HelpCapture]:
    """Return every ``HELP OUTPUT`` block, filling truncated H-line gaps when possible."""
    lines = (text or "").splitlines()
    headers = [
        (index, HELP_OUTPUT_HEADER_RE.match(line))
        for index, line in enumerate(lines)
        if HELP_OUTPUT_HEADER_RE.match(line)
    ]
    if not headers:
        return []

    captures: List[HelpCapture] = []
    for header_index, (start, match) in enumerate(headers):
        assert match is not None
        end = len(lines)
        for cursor in range(start + 1, len(lines)):
            if cursor != start and _is_block_stop(lines[cursor]):
                end = cursor
                break
        found = _numbered_lines(lines, start + 1, end)
        declared_lines = int(match.group("lines"))
        declared_chars = int(match.group("chars"))
        if declared_lines and len(found) < declared_lines:
            outside: Dict[int, str] = {}
            other_spans = []
            for other_index, (other_start, _other_match) in enumerate(headers):
                if other_index == header_index:
                    continue
                other_end = len(lines)
                for cursor in range(other_start + 1, len(lines)):
                    if cursor != other_start and _is_block_stop(lines[cursor]):
                        other_end = cursor
                        break
                other_spans.append((other_start, other_end))
            for line_no, line in enumerate(lines):
                if start <= line_no < end:
                    continue
                if any(other_start <= line_no < other_end for other_start, other_end in other_spans):
                    continue
                numbered = HELP_LINE_RE.match(line)
                if numbered:
                    index = int(numbered.group(1))
                    if index not in found:
                        outside[index] = numbered.group(2)
            found.update(outside)
        body, complete = _join_numbered(found, declared_lines)
        if not found:
            raw = [line for line in lines[start + 1 : end] if line != HELP_OUTPUT_END]
            body = "\n".join(raw)
            complete = bool(body.strip())
        captures.append(
            HelpCapture(
                title=match.group("title"),
                declared_lines=declared_lines,
                declared_chars=declared_chars,
                text=body,
                complete=complete,
            )
        )
    return captures


def extract_help_text(text: str) -> str:
    """Unwrap a scan log to raw help, or return plain ``--help`` text unchanged."""
    captures = extract_help_captures(text)
    if captures:
        return captures[0].text
    numbered = _all_numbered_lines((text or "").splitlines())
    if numbered:
        body, _complete = _join_numbered(numbered, max(numbered))
        return body
    return text or ""


def parse_flag_coverage_header(text: str) -> Optional[Dict[str, int]]:
    for line in (text or "").splitlines():
        match = FLAG_COVERAGE_HEADER_RE.match(line.strip())
        if match:
            return {key: int(value) for key, value in match.groupdict().items()}
    return None


def catalog_flags(sections: Sequence[dict]) -> List[str]:
    flags: List[str] = []
    seen = set()
    for section in sections or []:
        for row in section.get("params") or []:
            for flag in row.get("flags") or []:
                if (
                    isinstance(flag, str)
                    and flag.startswith("--")
                    and flag not in seen
                ):
                    seen.add(flag)
                    flags.append(flag)
    return flags


def compute_flag_coverage(
    help_text: str,
    sections: Sequence[dict],
    *,
    accounted_flags: Optional[Iterable[str]] = None,
) -> FlagCoverage:
    help_flags = help_flags_for_coverage(help_text or "")
    parsed_flags = catalog_flags(sections)
    accounted = set(accounted_flags or ())
    help_set = set(help_flags)
    parsed_set = set(parsed_flags)
    missing = [
        flag
        for flag in help_flags
        if flag not in parsed_set and flag not in accounted
    ]
    extra = [flag for flag in parsed_flags if flag not in help_set]
    reserved_missing = [flag for flag in missing if flag in RESERVED_FLAGS]
    interesting_missing = [flag for flag in missing if flag not in RESERVED_FLAGS]
    return FlagCoverage(
        help_flags=help_flags,
        parsed_flags=parsed_flags,
        missing=missing,
        extra=extra,
        reserved_missing=reserved_missing,
        interesting_missing=interesting_missing,
    )


def snapshot_param_rows(sections_or_params: Sequence[dict]) -> List[dict]:
    """Stable catalog subset for full-output snapshot comparison."""
    rows: List[dict] = []
    sources: List[tuple[Optional[str], dict]] = []
    if sections_or_params and "params" in (sections_or_params[0] or {}):
        for section in sections_or_params:
            sid = section.get("id")
            for param in section.get("params") or []:
                sources.append((sid, param))
    else:
        sources = [(param.get("section_id"), param) for param in sections_or_params or []]
    for section_id, param in sources:
        options = param.get("options") or []
        rows.append(
            {
                "key": param.get("key"),
                "primary_flag": param.get("primary_flag"),
                "negative_flag": param.get("negative_flag"),
                "flags": list(param.get("flags") or []),
                "value_kind": param.get("value_kind"),
                "type": param.get("type"),
                "scalar_type": param.get("scalar_type"),
                "default": param.get("default"),
                "section_id": param.get("section_id") or section_id,
                "options": [
                    str(opt.get("value"))
                    for opt in options
                    if isinstance(opt, dict) and opt.get("value") is not None
                ],
                "multiple": bool(param.get("multiple")),
                "reserved": bool(param.get("reserved")),
            }
        )
    rows.sort(key=lambda row: (row.get("section_id") or "", row.get("key") or ""))
    return rows


def coverage_issues(coverage: FlagCoverage) -> List[str]:
    issues: List[str] = []
    if coverage.interesting_missing:
        issues.append(
            "coverage missing from catalog: " + ",".join(coverage.interesting_missing)
        )
    if coverage.extra:
        issues.append("coverage extra parsed: " + ",".join(coverage.extra))
    return issues
