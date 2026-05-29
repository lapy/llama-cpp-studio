"""Exhaustive help-fixture audits: verify every CLI flag entry against parsed params."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from backend.cli_help_parsers import (
    LM_OPTION,
    LM_SECTION_HEADER,
    LONG_FLAG_RE,
    VLLM_CONFIG_GROUP_HEADER,
    VLLM_HELP_FOOTER,
    VLLM_OPTION,
    _extract_paren_default,
    _normalize_default_fragment,
)

_KNOWN_EMPTY_DESCRIPTION_FLAGS = frozenset(
    {
        "--dist-init-addr",
    }
)

_TYPE_BY_KIND = {
    "flag": "bool",
    "scalar": None,  # int | float | string
    "enum": "select",
    "csv_enum": "multiselect",
    "semicolon_enum": "multiselect",
    "repeatable": "list",
    "json_object": "json",
}


def _primary_from_flags(flags: Sequence[str]) -> str:
    positives = [f for f in flags if f.startswith("--") and not f.startswith("--no-")]
    if positives:
        return max(positives, key=len)
    return flags[-1]


def _section_id_from_label(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "options"


def _extract_default_from_text(text: str) -> Optional[str]:
    paren = _extract_paren_default(text)
    if paren is not None:
        return paren
    match = re.search(r"(?i)\bdefault(?:\s+to)?\s*[:=]\s*", text)
    if not match:
        return None
    tail = text[match.end() :].split("\n", 1)[0].strip()
    tail = re.split(r"\s+Type:\s*", tail, maxsplit=1)[0]
    return _normalize_default_fragment(tail)


def _coerce_default_for_compare(raw: str) -> Any:
    value = (raw or "").strip().strip(",")
    if not value:
        return None
    lower = value.lower()
    if lower in {"none", "null"}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?\d+\.\d+", value):
        return float(value)
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass
    return value


def _defaults_equivalent(expected_raw: Optional[str], parsed_default: Any) -> bool:
    if expected_raw is None:
        return True
    expected = _coerce_default_for_compare(expected_raw)
    if expected is None and parsed_default is None:
        return True
    if expected is None or parsed_default is None:
        return False
    if expected == parsed_default:
        return True
    if isinstance(expected, float) and isinstance(parsed_default, (int, float)):
        return float(expected) == float(parsed_default)
    if isinstance(expected, str) and isinstance(parsed_default, str):
        return expected.strip() == parsed_default.strip()
    return str(expected) == str(parsed_default)


def _enum_values_from_source_line(src_line: str) -> Optional[Set[str]]:
    match = re.search(r"\{([^}]+)\}", src_line)
    if not match:
        return None
    inner = match.group(1)
    if ":" in inner and '"' in inner:
        return None
    parts = [p.strip() for p in inner.split(",") if p.strip()]
    return set(parts) if parts else None


def _description_matches_source(source_desc: str, parsed_desc: str) -> bool:
    parsed = (parsed_desc or "").strip()
    source = (source_desc or "").strip()
    if not source:
        return bool(parsed)
    words = [w for w in re.findall(r"[A-Za-z]{4,}", source) if w.lower() not in {"type", "default"}]
    if not words:
        return bool(parsed)
    return any(word.lower() in parsed.lower() for word in words[:3])


def _check_type_consistency(param: dict, primary_flag: str, issues: List[str]) -> None:
    kind = param.get("value_kind")
    ui_type = param.get("type")
    expected = _TYPE_BY_KIND.get(kind)
    if expected and ui_type != expected:
        issues.append(f"{primary_flag}: type {ui_type!r} != expected {expected!r} for {kind}")
    if kind == "flag" and ui_type != "bool":
        issues.append(f"{primary_flag}: flag should have bool type, got {ui_type!r}")
    if kind in {"enum", "csv_enum", "semicolon_enum"} and not param.get("options"):
        issues.append(f"{primary_flag}: {kind} missing options")
    if kind == "json_object" and ui_type != "json":
        issues.append(f"{primary_flag}: json_object should have json type")


def extract_vllm_help_entries(text: str) -> List[dict]:
    lines = text.splitlines()
    entries: List[dict] = []
    section = "options"
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_no = i + 1
        if VLLM_HELP_FOOTER.match(stripped):
            break
        if stripped == "options:":
            section = "options"
            i += 1
            continue
        cg = VLLM_CONFIG_GROUP_HEADER.match(stripped)
        if cg:
            section = _section_id_from_label(cg.group(1))
            i += 1
            continue
        if stripped.startswith("positional arguments:"):
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if (
                    VLLM_HELP_FOOTER.match(nxt)
                    or nxt == "options:"
                    or VLLM_CONFIG_GROUP_HEADER.match(nxt)
                ):
                    break
                i += 1
            continue
        mo = VLLM_OPTION.match(line)
        if mo:
            flags = LONG_FLAG_RE.findall(mo.group(1))
            desc_lines: List[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.strip()
                if VLLM_HELP_FOOTER.match(ns):
                    break
                if (
                    VLLM_OPTION.match(nxt)
                    or ns == "options:"
                    or VLLM_CONFIG_GROUP_HEADER.match(ns)
                    or ns.startswith("positional arguments:")
                ):
                    break
                if not ns:
                    i += 1
                    break
                desc_lines.append(ns)
                i += 1
            entries.append(
                {
                    "line": line_no,
                    "section": section,
                    "primary": _primary_from_flags(flags),
                    "flags": flags,
                    "desc": " ".join(desc_lines),
                    "src_line": line.rstrip(),
                }
            )
            continue
        i += 1
    return entries


def extract_lmdeploy_help_entries(text: str) -> List[dict]:
    lines = text.splitlines()
    entries: List[dict] = []
    section = "options"
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_no = i + 1
        if stripped.startswith("positional arguments:"):
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if nxt == "options:" or (
                    LM_SECTION_HEADER.match(nxt) and "arguments" in nxt.lower()
                ):
                    break
                i += 1
            continue
        if stripped == "options:":
            section = "options"
            i += 1
            continue
        sh = LM_SECTION_HEADER.match(stripped)
        if sh and "arguments" in stripped.lower():
            section = _section_id_from_label(sh.group(1))
            i += 1
            continue
        mo = LM_OPTION.match(line)
        if mo:
            flags = LONG_FLAG_RE.findall(line)
            desc_lines: List[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.strip()
                if (
                    LM_OPTION.match(nxt)
                    or ns == "options:"
                    or (LM_SECTION_HEADER.match(ns) and "arguments" in ns.lower())
                ):
                    break
                if not ns:
                    i += 1
                    break
                desc_lines.append(ns)
                i += 1
            entries.append(
                {
                    "line": line_no,
                    "section": section,
                    "primary": _primary_from_flags(flags),
                    "flags": flags,
                    "desc": " ".join(desc_lines),
                    "src_line": line.rstrip(),
                }
            )
            continue
        i += 1
    return entries


def verify_all_help_params(
    entries: Sequence[dict],
    parsed: Sequence[dict],
    *,
    allow_empty_description: Optional[Iterable[str]] = None,
    skip_default_check: Optional[Iterable[str]] = None,
) -> List[str]:
    """Return a list of human-readable failures (empty list means all params verified)."""
    allow_empty = frozenset(allow_empty_description or ()) | _KNOWN_EMPTY_DESCRIPTION_FLAGS
    skip_defaults = frozenset(skip_default_check or ())

    by_primary = {p["primary_flag"]: p for p in parsed}
    first_section: Dict[str, str] = {}
    first_entry: Dict[str, dict] = {}
    for entry in entries:
        pf = entry["primary"]
        if pf not in first_section:
            first_section[pf] = entry["section"]
            first_entry[pf] = entry

    unique_primaries = sorted(first_section.keys())
    issues: List[str] = []

    if len(parsed) != len(unique_primaries):
        issues.append(
            f"param count parsed={len(parsed)} unique_flags={len(unique_primaries)}"
        )

    for pf in unique_primaries:
        param = by_primary.get(pf)
        entry = first_entry[pf]
        label = f"line {entry['line']} {pf}"

        if param is None:
            issues.append(f"{label}: missing from parsed output")
            continue

        exp_key = pf.lstrip("-").replace("-", "_")
        if param["key"] != exp_key:
            issues.append(f"{label}: key {param['key']!r} != {exp_key!r}")
        if param.get("primary_flag") != pf:
            issues.append(f"{label}: primary_flag {param.get('primary_flag')!r}")
        if pf not in (param.get("flags") or []):
            issues.append(f"{label}: primary flag not in flags {param.get('flags')!r}")

        for flag in entry["flags"]:
            if flag.startswith("--") and not flag.startswith("--no-"):
                if flag not in (param.get("flags") or []):
                    issues.append(f"{label}: missing alias {flag}")

        negatives = [f for f in entry["flags"] if f.startswith("--no-")]
        if negatives:
            if param.get("value_kind") != "flag":
                issues.append(
                    f"{label}: expected flag for paired --no- option, got {param.get('value_kind')}"
                )
            if param.get("negative_flag") not in negatives:
                issues.append(
                    f"{label}: negative_flag {param.get('negative_flag')!r} not in {negatives}"
                )

        if param.get("section_id") != first_section[pf]:
            issues.append(
                f"{label}: section {param.get('section_id')!r} != {first_section[pf]!r}"
            )

        if pf not in allow_empty:
            parsed_desc = (param.get("description") or "").strip()
            if not parsed_desc:
                issues.append(f"{label}: empty description")
            elif not _description_matches_source(entry.get("desc", ""), parsed_desc):
                if entry.get("desc") or re.search(
                    r"\s{2,}[A-Za-z]", entry.get("src_line", "")
                ):
                    issues.append(
                        f"{label}: description mismatch: {parsed_desc[:60]!r}"
                    )

        src_line = entry.get("src_line") or ""
        source_text = f"{src_line} {entry.get('desc', '')}"

        if re.search(r"\[[^\]]+\.\.\.\]", src_line):
            if param.get("value_kind") != "repeatable":
                issues.append(
                    f"{label}: expected repeatable for [... ...] spec, got {param.get('value_kind')}"
                )
        elif _enum_values_from_source_line(src_line) and not negatives:
            enum_values = _enum_values_from_source_line(src_line)
            kind = param.get("value_kind")
            if kind not in {"enum", "json_object"}:
                issues.append(
                    f"{label}: expected enum/json for {enum_values}, got {kind}"
                )
            elif kind == "enum":
                parsed_values = {
                    str(o.get("value")) for o in (param.get("options") or [])
                }
                missing = enum_values - parsed_values
                if missing:
                    issues.append(
                        f"{label}: enum missing values {sorted(missing)}"
                    )

        if pf not in skip_defaults:
            expected_default = _extract_default_from_text(source_text)
            if expected_default is not None and not _defaults_equivalent(
                expected_default, param.get("default")
            ):
                issues.append(
                    f"{label}: default expected {expected_default!r}, got {param.get('default')!r}"
                )

        if not param.get("flags"):
            issues.append(f"{label}: empty flags list")

        _check_type_consistency(param, pf, issues)

    parsed_primaries = {p["primary_flag"] for p in parsed}
    for pf in sorted(parsed_primaries - set(unique_primaries)):
        issues.append(f"extra parsed param not in fixture: {pf}")

    return issues
