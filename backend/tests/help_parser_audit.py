"""Exhaustive help-fixture audits: verify every CLI flag entry against parsed params."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from backend.cli_help_parsers import (
    CSV_ELLIPSIS_SPEC_RE,
    LM_OPTION,
    LM_SECTION_HEADER,
    LONG_FLAG_RE,
    SECTION_RULE_LLAMA,
    VLLM_CONFIG_GROUP_HEADER,
    VLLM_HELP_FOOTER,
    VLLM_OPTION,
    _AUDIO_BARE_KEYED_OPTION_SECTIONS,
    _AUDIO_BARE_OPTION_NAME_RE,
    _DASH_ENUM_ITEM_RE,
    _LLAMA_OPTION_MAX_INDENT,
    _RANGE_ELLIPSIS_RE,
    _REMOVED_ARGUMENT_RE,
    _audio_section_id,
    _extract_paren_default,
    _extract_value_spec,
    _is_llama_option_line,
    _normalize_default_fragment,
    _preferred_primary_flag,
    _raw_default,
    _split_audio_compact_flag_specs,
    _split_spec_and_description,
    _trim_llama_help_prologue,
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
    return _raw_default(text)


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
    if parsed_default is None and (
        "taken from" in str(expected_raw).lower()
        or str(expected_raw).lower().startswith("loaded from")
    ):
        return True
    if str(expected_raw).strip().lower() in {
        "no tools",
        "unset",
        "unused",
        "disabled",
        "search in path",
        "template default",
        "none, use host environment",
    } and parsed_default in (None, [], ""):
        return True
    if isinstance(parsed_default, bool):
        lower = str(expected_raw).strip().lower()
        if parsed_default and (
            "enabled" in lower or lower in {"true", "on", "yes"}
        ):
            return True
        if not parsed_default and (
            "disabled" in lower or lower in {"false", "off", "no"}
        ):
            return True
    if isinstance(parsed_default, (int, float)):
        numeric_head = re.match(
            r"^([-+]?\d+(?:\.\d+)?)", str(expected_raw).strip()
        )
        if numeric_head and float(numeric_head.group(1)) == float(parsed_default):
            return True
    if isinstance(parsed_default, list) and str(expected_raw).strip().lower() == "none":
        return parsed_default in (["none"], [])
    expected = _coerce_default_for_compare(expected_raw)
    if expected is None and parsed_default in (None, []):
        return True
    if expected is None or parsed_default is None:
        return False
    if expected == parsed_default:
        return True
    if isinstance(parsed_default, list) and isinstance(expected, str):
        sep = ";" if ";" in expected else ","
        parts = [part.strip() for part in expected.split(sep) if part.strip()]
        if parts == parsed_default:
            return True
        if parsed_default == [expected.strip()]:
            return True
    if isinstance(expected, float) and isinstance(parsed_default, (int, float)):
        return float(expected) == float(parsed_default)
    if isinstance(expected, str) and isinstance(parsed_default, str):
        return expected.strip() == parsed_default.strip()
    return str(expected) == str(parsed_default)


def _enum_values_from_source_line(src_line: str) -> Optional[Set[str]]:
    match = re.search(r"\{([^}]+)\}", src_line)
    if match:
        inner = match.group(1)
        if not (":" in inner and '"' in inner):
            parts = [p.strip() for p in inner.split(",") if p.strip()]
            if parts:
                return set(parts)
    for pattern in (r"\[([^\]|]+\|[^]]+)\]", r"<([^>|]+\|[^>]+)>"):
        match = re.search(pattern, src_line)
        if match and "..." not in match.group(1):
            parts = [p.strip() for p in match.group(1).split("|") if p.strip()]
            if parts and all(" " not in p for p in parts):
                return set(parts)
    return None


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


def extract_llama_help_entries(text: str) -> List[dict]:
    """Collect llama-server / ik_llama option blocks (skips removed arguments)."""
    text = _trim_llama_help_prologue(text)
    lines = text.splitlines()
    entries: List[dict] = []
    section = "general"
    i = 0
    has_dash_banners = any(SECTION_RULE_LLAMA.match(line.strip()) for line in lines)
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_no = i + 1
        sm = SECTION_RULE_LLAMA.match(stripped)
        if sm:
            section = _section_id_from_label(sm.group(1).strip())
            i += 1
            continue
        if not has_dash_banners:
            sh = LM_SECTION_HEADER.fullmatch(stripped)
            if sh:
                section = _section_id_from_label(sh.group(1).strip())
                i += 1
                continue
        if _is_llama_option_line(line):
            spec, inline_desc = _split_spec_and_description(line)
            flags = LONG_FLAG_RE.findall(spec)
            desc_lines: List[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if _is_llama_option_line(nxt):
                    break
                if not nxt.strip():
                    i += 1
                    continue
                desc_lines.append(nxt.strip())
                i += 1
            description = " ".join(x for x in desc_lines if x).strip()
            full_help = " ".join(x for x in (inline_desc, description) if x)
            if _REMOVED_ARGUMENT_RE.search(full_help):
                continue
            description = re.sub(
                r"\(env:\s*[^)]+\)", "", description, flags=re.IGNORECASE
            ).strip()
            if not flags:
                continue
            entries.append(
                {
                    "line": line_no,
                    "section": section,
                    "primary": _preferred_primary_flag(flags, spec),
                    "flags": flags,
                    "desc": description,
                    "src_line": line.rstrip(),
                }
            )
            continue
        i += 1
    return entries


def classify_llama_help_lines(text: str) -> List[dict]:
    """Label every fixture line: prologue, section, option, continuation, blank, or other."""
    lines = text.splitlines()
    first_section = next(
        (i for i, line in enumerate(lines) if SECTION_RULE_LLAMA.match(line.strip())),
        None,
    )
    has_dash_banners = first_section is not None
    rows: List[dict] = []
    in_option = False
    for line_no, line in enumerate(lines, start=1):
        if first_section is not None and line_no - 1 < first_section:
            role = "blank" if not line.strip() else "prologue"
            rows.append({"line": line_no, "role": role, "text": line})
            continue
        stripped = line.strip()
        leading = len(line) - len(line.lstrip(" "))
        if not stripped:
            rows.append({"line": line_no, "role": "blank", "text": line})
            in_option = False
            continue
        if SECTION_RULE_LLAMA.match(stripped) or (
            not has_dash_banners and LM_SECTION_HEADER.fullmatch(stripped)
        ):
            rows.append({"line": line_no, "role": "section", "text": stripped})
            in_option = False
            continue
        if _is_llama_option_line(line):
            rows.append({"line": line_no, "role": "option", "text": line.rstrip()})
            in_option = True
            continue
        if in_option or leading > _LLAMA_OPTION_MAX_INDENT:
            rows.append({"line": line_no, "role": "continuation", "text": stripped})
            in_option = True
            continue
        rows.append({"line": line_no, "role": "other", "text": stripped})
        in_option = False
    return rows


def infer_llama_expected_kind(spec: str, flags: Sequence[str], description: str) -> str:
    """Classify a llama-server option from help syntax, independent of parser internals."""
    flags = [f for f in flags if isinstance(f, str)]
    value_spec = _extract_value_spec(spec, flags)
    compact = re.sub(r"\s+", "", value_spec or "")
    desc = description or ""
    positives = [f for f in flags if f.startswith("--") and not f.startswith("--no-")]
    negatives = [f for f in flags if f.startswith("--no-")]
    vs = (value_spec or "").strip()

    if negatives and positives and not vs:
        return "flag"
    if _RANGE_ELLIPSIS_RE.fullmatch(compact):
        return "scalar"
    if _enum_values_from_source_line(spec):
        return "enum"
    if re.search(r"allowed\s+values:", desc, re.IGNORECASE):
        return "enum"
    dash_items = list(dict.fromkeys(_DASH_ENUM_ITEM_RE.findall(desc)))
    if len(dash_items) >= 2:
        return "enum"
    if "separated by ';'" in desc:
        return "semicolon_enum"
    if "comma-separated list of types" in desc.lower() or (
        CSV_ELLIPSIS_SPEC_RE.fullmatch(compact)
        and re.search(r"available\s+tools:", desc, re.IGNORECASE)
    ):
        return "csv_enum"
    vs_upper = vs.upper()
    if (
        re.fullmatch(r"JSON", vs_upper)
        or vs_upper.endswith(" JSON")
        or (
            ("json object" in desc.lower() or "valid json" in desc.lower())
            and not re.search(r"\b(FILE|FNAME|PATH)\b", vs_upper)
        )
    ):
        return "json_object"
    if "..." in vs and not CSV_ELLIPSIS_SPEC_RE.fullmatch(compact):
        if "comma-separated" not in desc.lower():
            return "repeatable"
    if negatives and positives:
        return "flag"
    if not vs:
        return "flag"
    return "scalar"


def verify_llama_help_line_by_line(text: str, parsed: Sequence[dict]) -> List[str]:
    """Fail if any help line is unclassified or any option is mis-typed vs help syntax."""
    issues: List[str] = []
    classified = classify_llama_help_lines(text)
    for row in classified:
        if row["role"] == "other":
            issues.append(
                f"line {row['line']}: unclassified help text: {row['text'][:80]!r}"
            )

    by_primary = {p["primary_flag"]: p for p in parsed}
    option_lines = [row for row in classified if row["role"] == "option"]
    removed = 0
    expected_count = 0
    for idx, row in enumerate(option_lines):
        spec, inline = _split_spec_and_description(row["text"])
        flags = LONG_FLAG_RE.findall(spec)
        desc_parts = [inline] if inline else []
        next_option_line = (
            option_lines[idx + 1]["line"] if idx + 1 < len(option_lines) else 10**9
        )
        for follow in classified:
            if follow["line"] <= row["line"]:
                continue
            if follow["line"] >= next_option_line:
                break
            if follow["role"] == "section":
                break
            if follow["role"] == "continuation":
                desc_parts.append(follow["text"])
        description = " ".join(desc_parts)
        if _REMOVED_ARGUMENT_RE.search(description):
            removed += 1
            continue
        if not flags:
            issues.append(
                f"line {row['line']}: option has no long flags: {row['text'][:80]!r}"
            )
            continue
        expected_count += 1
        primary = _preferred_primary_flag(flags, spec)
        expected_kind = infer_llama_expected_kind(spec, flags, description)
        param = by_primary.get(primary)
        if param is None:
            issues.append(f"line {row['line']} {primary}: not parsed")
            continue
        if param.get("value_kind") != expected_kind:
            issues.append(
                f"line {row['line']} {primary}: value_kind {param.get('value_kind')!r} "
                f"!= help syntax {expected_kind!r}"
            )
        for flag in flags:
            if flag not in (param.get("flags") or []):
                issues.append(f"line {row['line']} {primary}: missing alias {flag}")
        enum_values = _enum_values_from_source_line(spec)
        if enum_values and param.get("value_kind") == "enum":
            got = {str(o.get("value")) for o in (param.get("options") or [])}
            missing = enum_values - got
            if missing:
                issues.append(
                    f"line {row['line']} {primary}: enum missing {sorted(missing)}"
                )

    if len(parsed) != expected_count:
        issues.append(
            f"param count parsed={len(parsed)} live_options={expected_count} "
            f"removed={removed}"
        )
    return issues


def llama_help_expected_rows(parsed: Sequence[dict]) -> List[dict]:
    """Stable subset of parsed params for snapshot comparison."""
    rows: List[dict] = []
    for param in parsed:
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
                "section_id": param.get("section_id"),
                "options": [str(opt.get("value")) for opt in options if opt.get("value") is not None],
                "multiple": bool(param.get("multiple")),
                "reserved": bool(param.get("reserved")),
            }
        )
    rows.sort(key=lambda row: (row.get("section_id") or "", row.get("key") or ""))
    return rows


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
        positives = [
            f
            for f in entry["flags"]
            if f.startswith("--") and not f.startswith("--no-")
        ]
        if negatives and positives:
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


_AUDIO_USAGE_BRACKET_RE = re.compile(r"\[--([a-z0-9-]+)(?:\s+[^\]]*)?\]")
_AUDIO_SKIP_SECTIONS = frozenset({"endpoints", "tasks"})
_AUDIO_NESTED_OPTIONAL_FLAGS = frozenset({"--json"})
_AUDIO_INT_METAVARS = frozenset(
    {"n", "ms", "mb", "id", "port", "hz", "chars", "int", "seconds"}
)
_AUDIO_FLOAT_METAVARS = frozenset({"float", "double"})


def _audio_pipe_values(text: str) -> Optional[List[str]]:
    compact = str(text or "").strip().strip("<>[]{}")
    if "|" not in compact:
        return None
    parts = [part.strip() for part in compact.split("|") if part.strip()]
    if len(parts) < 2 or any(" " in part for part in parts):
        return None
    return parts


def _audio_default_from_text(text: str) -> Optional[str]:
    paren = _extract_paren_default(text)
    if paren is not None:
        return paren
    colon = _extract_default_from_text(text)
    if colon is not None:
        return colon
    match = re.search(
        r"(?:^|[,;]\s+|\s{2,})default(?:\s+to)?(?:\s*[:=]\s*|\s+)(?P<val>[^\s,;]+)",
        text or "",
        re.IGNORECASE,
    )
    if not match:
        return None
    return _normalize_default_fragment(match.group("val"))


def _audio_metavar(spec: str, primary: str) -> str:
    rest = spec
    if primary and primary in rest:
        rest = rest.split(primary, 1)[-1]
    rest = rest.strip().strip("[]")
    token = rest.split()[0] if rest.split() else ""
    return token.strip("<>[]{}\"'")


def extract_audio_cpp_help_entries(text: str, *, source: str = "cli") -> List[dict]:
    """Line-by-line inventory of audio.cpp help flags and bare keyed options."""
    lines = str(text or "").splitlines()
    entries: List[dict] = []
    section = "process_options" if source == "server" else "global"
    skip_body = False
    i = 0

    def _append_flag_entry(
        *,
        line_no: int,
        spec: str,
        desc: str,
        src_line: str,
        kind: str,
        section_id: str,
    ) -> None:
        flags = LONG_FLAG_RE.findall(spec)
        if not flags:
            return
        nested = [flag for flag in flags if flag in _AUDIO_NESTED_OPTIONAL_FLAGS]
        flags = [flag for flag in flags if flag not in _AUDIO_NESTED_OPTIONAL_FLAGS]
        if not flags:
            return
        primary = flags[0]
        if primary == "--no-ui":
            kind = "negative"
        pipe = _audio_pipe_values(_audio_metavar(spec, primary) if "|" in spec else spec)
        if pipe is None:
            pipe = _audio_pipe_values(spec.split(primary, 1)[-1] if primary in spec else spec)
        entries.append(
            {
                "line": line_no,
                "section": section_id,
                "kind": kind,
                "primary": primary,
                "flags": flags,
                "nested": nested,
                "desc": desc,
                "src_line": src_line,
                "spec": spec,
                "pipe": pipe,
                "metavar": _audio_metavar(spec, primary),
                "key": primary.lstrip("-").replace("-", "_"),
            }
        )

    usage_flags: List[tuple[int, str]] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        line_no = i + 1

        heading = LM_SECTION_HEADER.fullmatch(stripped)
        if heading and not stripped.startswith("--"):
            section = _audio_section_id(heading.group(1))
            skip_body = section in _AUDIO_SKIP_SECTIONS
            i += 1
            continue

        if skip_body:
            i += 1
            continue

        if "=" in stripped and not stripped.startswith("-") and not line[:1].isspace():
            i += 1
            continue

        bare_section = _AUDIO_BARE_KEYED_OPTION_SECTIONS.get(section)
        if (
            bare_section
            and stripped
            and not stripped.startswith("-")
            and line[:1].isspace()
        ):
            spec, desc = _split_spec_and_description(line)
            name = spec.split()[0] if spec.split() else ""
            if name and _AUDIO_BARE_OPTION_NAME_RE.match(name):
                value_spec = spec[len(name) :].strip()
                entries.append(
                    {
                        "line": line_no,
                        "section": section,
                        "kind": "bare",
                        "primary": f"--{bare_section.replace('_', '-')}",
                        "flags": [],
                        "nested": [],
                        "desc": desc,
                        "src_line": line.rstrip(),
                        "spec": spec,
                        "pipe": _audio_pipe_values(value_spec) or _audio_pipe_values(desc),
                        "metavar": value_spec.strip().split()[0].strip("<>[]{}")
                        if value_spec.strip()
                        else "",
                        "key": name,
                    }
                )
                i += 1
                continue

        if stripped.startswith("--") or (
            stripped.startswith("-") and "--" in stripped
        ):
            spec, inline_desc = _split_spec_and_description(line)
            desc_lines = [inline_desc] if inline_desc else []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                ns = nxt.strip()
                if ns.startswith("-") and "--" in ns:
                    break
                if LM_SECTION_HEADER.fullmatch(ns):
                    break
                if not ns:
                    i += 1
                    break
                if nxt and not nxt[0].isspace() and "=" in nxt:
                    break
                if (
                    _AUDIO_BARE_KEYED_OPTION_SECTIONS.get(section)
                    and nxt[:1].isspace()
                    and not ns.startswith("-")
                    and _AUDIO_BARE_OPTION_NAME_RE.match(ns.split()[0])
                ):
                    break
                desc_lines.append(ns)
                i += 1
            description = " ".join(desc_lines).strip()
            for sub_spec in _split_audio_compact_flag_specs(spec) or [spec]:
                _append_flag_entry(
                    line_no=line_no,
                    spec=sub_spec,
                    desc=description,
                    src_line=line.rstrip(),
                    kind="flag",
                    section_id=section,
                )
            continue

        if not skip_body:
            for match in _AUDIO_USAGE_BRACKET_RE.finditer(stripped):
                flag = f"--{match.group(1)}"
                if flag in _AUDIO_NESTED_OPTIONAL_FLAGS:
                    continue
                usage_flags.append((line_no, flag))
            if stripped.startswith(("audiocpp_server", "audiocpp_cli")):
                for flag in LONG_FLAG_RE.findall(stripped):
                    if flag in _AUDIO_NESTED_OPTIONAL_FLAGS:
                        continue
                    if f"[--{flag.lstrip('-')}" in stripped:
                        continue
                    usage_flags.append((line_no, flag))
        i += 1

    for line_no, flag in usage_flags:
        already = any(
            entry["primary"] == flag and entry["kind"] in {"flag", "negative", "usage"}
            for entry in entries
        )
        if already:
            continue
        entries.append(
            {
                "line": line_no,
                "section": "process_options" if source == "server" else "global",
                "kind": "usage",
                "primary": flag,
                "flags": [flag],
                "nested": [],
                "desc": "",
                "src_line": lines[line_no - 1].rstrip() if 0 < line_no <= len(lines) else "",
                "spec": flag,
                "pipe": None,
                "metavar": "",
                "key": flag.lstrip("-").replace("-", "_"),
            }
        )
    return entries


def verify_audio_cpp_help_params(
    entries: Sequence[dict],
    parsed: Sequence[dict],
) -> List[str]:
    """Verify every extracted audio.cpp help line against parsed params."""
    by_key = {str(param.get("key") or ""): param for param in parsed}
    by_flag: Dict[str, dict] = {}
    for param in parsed:
        primary = str(param.get("primary_flag") or "")
        if primary:
            by_flag.setdefault(primary, param)
        negative = str(param.get("negative_flag") or "")
        if negative:
            by_flag[negative] = param

    issues: List[str] = []
    claimed_keys: Set[str] = set()
    claimed_flags: Set[str] = set()
    first_flag_section: Dict[str, str] = {}

    for entry in entries:
        kind = entry.get("kind")
        label = f"line {entry['line']} {entry.get('primary') or entry.get('key')}"
        if kind == "bare":
            param = by_key.get(str(entry.get("key") or ""))
            if param is None:
                issues.append(f"{label}: bare option {entry.get('key')!r} missing")
                continue
            claimed_keys.add(param["key"])
            if param.get("section_id") != entry["section"]:
                issues.append(
                    f"{label}: section {param.get('section_id')!r} != {entry['section']!r}"
                )
        elif kind == "negative":
            param = by_flag.get(str(entry.get("primary") or ""))
            if param is None:
                issues.append(f"{label}: negative flag missing from parsed output")
                continue
            claimed_keys.add(param["key"])
            claimed_flags.add(str(param.get("primary_flag") or ""))
            if param.get("negative_flag") != entry["primary"]:
                issues.append(
                    f"{label}: negative_flag {param.get('negative_flag')!r}"
                )
            if param.get("value_kind") != "flag":
                issues.append(f"{label}: expected bool flag, got {param.get('value_kind')}")
            continue
        else:
            primary = str(entry.get("primary") or "")
            param = by_flag.get(primary) or by_key.get(str(entry.get("key") or ""))
            if param is None:
                issues.append(f"{label}: missing from parsed output")
                continue
            claimed_keys.add(param["key"])
            claimed_flags.add(primary)
            if primary not in first_flag_section:
                first_flag_section[primary] = entry["section"]
                if (
                    kind != "usage"
                    and param.get("section_id") != entry["section"]
                    and param.get("key") != "mode"
                ):
                    issues.append(
                        f"{label}: section {param.get('section_id')!r} != {entry['section']!r}"
                    )
            exp_key = primary.lstrip("-").replace("-", "_")
            if param.get("key") != exp_key and kind != "usage":
                issues.append(f"{label}: key {param.get('key')!r} != {exp_key!r}")
            if kind == "flag" and primary not in (param.get("flags") or []) and primary != param.get("negative_flag"):
                issues.append(f"{label}: {primary} not in flags {param.get('flags')!r}")

        if kind == "usage":
            continue

        source_text = f"{entry.get('src_line', '')} {entry.get('desc', '')}"
        pipe = entry.get("pipe")
        if pipe:
            values = {str(item).lower() for item in pipe}
            if values <= {"true", "false"}:
                if param.get("type") != "bool":
                    issues.append(f"{label}: expected bool for true|false, got {param.get('type')!r}")
            else:
                parsed_values = [
                    str(opt.get("value")) for opt in (param.get("options") or [])
                ]
                if parsed_values != list(pipe):
                    issues.append(
                        f"{label}: enum {parsed_values} != {list(pipe)}"
                    )
                if param.get("type") != "select":
                    issues.append(f"{label}: expected select, got {param.get('type')!r}")

        metavar = str(entry.get("metavar") or "").lower()
        if metavar in _AUDIO_INT_METAVARS and param.get("type") != "int":
            issues.append(f"{label}: metavar {metavar!r} expected int, got {param.get('type')!r}")
        if metavar in _AUDIO_FLOAT_METAVARS and param.get("type") != "float":
            issues.append(
                f"{label}: metavar {metavar!r} expected float, got {param.get('type')!r}"
            )
        if metavar in {"bool", "boolean"} and param.get("type") != "bool":
            issues.append(f"{label}: metavar bool expected bool type")

        expected_default = _audio_default_from_text(source_text)
        if (
            expected_default is not None
            and "|" not in str(entry.get("spec") or "").split(entry.get("primary") or "", 1)[-1]
            and not _defaults_equivalent(expected_default, param.get("default"))
        ):
            issues.append(
                f"{label}: default expected {expected_default!r}, got {param.get('default')!r}"
            )

        desc = (entry.get("desc") or "").strip()
        parsed_desc = (param.get("description") or "").strip()
        if desc and not _description_matches_source(desc, parsed_desc):
            issues.append(f"{label}: description mismatch: {parsed_desc[:80]!r}")

    parsed_keys = {str(param.get("key") or "") for param in parsed}
    extra = sorted(parsed_keys - claimed_keys)
    for key in extra:
        issues.append(f"extra parsed param not in help lines: {key}")
    return issues
