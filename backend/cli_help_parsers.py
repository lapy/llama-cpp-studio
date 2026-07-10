"""Parse --help text into structured parameter rows for the engine catalog."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

LONG_FLAG_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9-]*")

SECTION_RULE_LLAMA = re.compile(r"^[-=]{3,}\s*(.+?)\s*[-=]{3,}\s*$")

META_ENUM = re.compile(r"\{([^}]+)\}")
META_BRACKET = re.compile(r"\[([^\]]+)\]")
ALLOWED_VALUES_RE = re.compile(r"allowed\s+values:\s*([^\n(]+)", re.IGNORECASE)
ENV_PAREN_RE = re.compile(r"\s*\(env:\s*[^)]+\)", re.IGNORECASE)
DICT_KEYS_RE = re.compile(r"dict_keys\((\[[^\]]*\])\)")
CSV_ELLIPSIS_SPEC_RE = re.compile(r"^[^\s,]+(?:,[^\s,]+)+,?\.{3}$")
# ``<0...100>``-style numeric ranges (not “repeat this flag” ellipsis lists).
_RANGE_ELLIPSIS_RE = re.compile(r"^<\s*\d+\s*\.\.\.\s*\d+\s*>$", re.IGNORECASE)
_INLINE_OPTION_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_-]*$")
_VALUE_SPEC_PLACEHOLDER_RE = re.compile(
    r"^(?:N\d*|N|I|FNAME|TYPE|PATH|SEED|SAMPLERS|SEQUENCE|URL|FILE|HOST|PORT|PREFIX|JSON|SECONDS|INDEX|MiB\d*)(?:\.\.\.)?$",
    re.IGNORECASE,
)
_REMOVED_ARGUMENT_RE = re.compile(
    r"\b(?:the\s+)?argument\s+has\s+been\s+removed\b", re.IGNORECASE
)

LM_SECTION_HEADER = re.compile(r"^([A-Za-z][^:]{0,120}):\s*$")
LM_OPTION = re.compile(
    r"^\s+(?:(?:-[a-zA-Z0-9]+),?\s*)*(--[a-zA-Z0-9][a-zA-Z0-9_-]*)(?:\s+(.*))?$"
)

VLLM_CONFIG_GROUP_HEADER = re.compile(r"^([A-Z][A-Za-z0-9]*):\s*$")
VLLM_OPTION = re.compile(
    r"^\s+((?:--[a-zA-Z0-9][a-zA-Z0-9_-]*|-[a-zA-Z0-9]+)"
    r"(?:,\s*(?:--[a-zA-Z0-9][a-zA-Z0-9_-]*|-[a-zA-Z0-9]+))*)"
    r"(?:\s+(.*))?$"
)
VLLM_HELP_FOOTER = re.compile(r"^When passing JSON CLI arguments", re.IGNORECASE)

RESERVED_FLAGS = frozenset(
    {
        "--alias",
        "--help",
        "--hf-repo",
        "--mmproj",
        "--model",
        "--port",
        "--server-port",
        "--usage",
        "--version",
    }
)


def _inline_tail_looks_like_metavar(tail: str) -> bool:
    if not tail:
        return False
    first = tail.strip().split()[0].rstrip(",")
    if first.startswith("{") or first.startswith("["):
        return True
    return bool(_VALUE_SPEC_PLACEHOLDER_RE.match(first))


def _human_label(key: str) -> str:
    return " ".join(w.capitalize() for w in key.replace("_", " ").split())


def _snake_from_long_flag(flag: str) -> str:
    return flag.lstrip("-").replace("-", "_")


def _split_spec_and_description(line: str) -> Tuple[str, str]:
    body = line.strip()
    if not body:
        return "", ""
    parts = [part.strip() for part in re.split(r"\s{2,}", body) if part.strip()]
    if len(parts) == 1:
        return parts[0], ""
    if not LONG_FLAG_RE.search(parts[0]) and LONG_FLAG_RE.search(parts[1]):
        spec = " ".join(parts[:2]).strip()
        description = " ".join(parts[2:]).strip()
        return spec, description
    return parts[0].strip(), " ".join(parts[1:]).strip()


def _unique_flags(flags: List[str]) -> List[str]:
    return list(
        dict.fromkeys(f for f in flags if isinstance(f, str) and f.startswith("-"))
    )


def _flag_prefix_from_spec(spec: str) -> str:
    """Keep only the flag alias tokens, not metavar/enum text."""
    trimmed = (spec or "").strip()
    for marker in (" {", " ["):
        if marker in trimmed:
            trimmed = trimmed.split(marker, 1)[0]
    parts = trimmed.split()
    while parts:
        token = parts[-1].rstrip(",")
        if token.startswith("-") or _VALUE_SPEC_PLACEHOLDER_RE.match(token):
            break
        parts.pop()
    return " ".join(parts)


def _flags_from_help_spec(spec: str) -> List[str]:
    """Collect ``-h`` / ``--help`` style aliases from an argparse option spec line."""
    prefix = _flag_prefix_from_spec(spec)
    if not prefix:
        return []
    long_flags = LONG_FLAG_RE.findall(prefix)
    short_flags = [
        token
        for token in re.findall(r"-[a-zA-Z0-9]+", prefix)
        if not token.startswith("--")
    ]
    return _unique_flags(short_flags + long_flags)


def _select_positive_flag(flags: List[str]) -> Optional[str]:
    positives = [f for f in flags if not f.startswith("--no-")]
    if not positives:
        return None
    return max(enumerate(positives), key=lambda pair: (len(pair[1]), pair[0]))[1]


def _select_negative_flag(flags: List[str]) -> Optional[str]:
    negatives = [f for f in flags if f.startswith("--no-")]
    if not negatives:
        return None
    return max(enumerate(negatives), key=lambda pair: (len(pair[1]), pair[0]))[1]


def _is_prefix_alias_flag(shorter: str, longer: str) -> bool:
    """``--typical`` is a short alias of ``--typical-p`` on the same help line."""
    if shorter == longer:
        return False
    return longer.startswith(shorter + "-")


def _spec_positive_flags(spec: str) -> List[str]:
    return [
        f for f in LONG_FLAG_RE.findall(spec or "") if not f.startswith("--no-")
    ]


def _filter_prefix_alias_flags(spec_flags: List[str]) -> List[str]:
    return [
        flag
        for flag in spec_flags
        if not any(
            _is_prefix_alias_flag(flag, other)
            for other in spec_flags
            if other != flag
        )
    ]


def _preferred_primary_flag(flags: List[str], spec: str) -> str:
    """
    Pick the canonical ``--`` flag for config key / CLI emission when several
    aliases share one help entry.

    Drop short prefix aliases (``--typical`` vs ``--typical-p``), then prefer the
    longest remaining name (``--temperature`` over ``--temp``).
    """
    spec_flags = _spec_positive_flags(spec)
    filtered = _filter_prefix_alias_flags(spec_flags)
    if filtered:
        return max(filtered, key=len)
    positive = _select_positive_flag(flags)
    if positive:
        return positive
    return flags[-1] if flags else "--unknown"


def _flags_to_key(flags: List[str], spec: str = "") -> str:
    return _snake_from_long_flag(_preferred_primary_flag(flags, spec))


def _extract_value_spec(spec: str, flags: List[str]) -> str:
    value_spec = spec
    for flag in sorted(flags, key=len, reverse=True):
        value_spec = value_spec.replace(flag, " ")
    value_spec = re.sub(r"(?:^|[\s,])-[a-zA-Z0-9]+(?=[\s,]|$)", " ", value_spec)
    protected: List[str] = []

    def _protect(match: re.Match[str]) -> str:
        protected.append(match.group(0))
        return f"__META_{len(protected) - 1}__"

    value_spec = re.sub(r"\{[^}]+\}|\[[^\]]+\]", _protect, value_spec)
    value_spec = re.sub(r"\s*,\s*", " ", value_spec)
    value_spec = re.sub(r"\s+", " ", value_spec).strip()
    for idx, item in enumerate(protected):
        value_spec = value_spec.replace(f"__META_{idx}__", item)
    return value_spec


def _is_csv_enum_description(description: str) -> bool:
    desc = (description or "").lower()
    return any(
        marker in desc
        for marker in ("comma-separated", "comma separated", "comma separated list")
    )


def _is_semicolon_enum_description(description: str) -> bool:
    desc = description or ""
    return "separated by ';'" in desc or "separated by ';'" in desc.lower()


def _extract_inline_flag_options(spec: str, flags: List[str]) -> Optional[List[dict]]:
    """
    Options embedded on the flag line: ``--spec-type a,b,c`` (not ``N0,N1`` proportions).
    """
    if not flags or not spec:
        return None
    primary = _select_positive_flag(flags) or flags[-1]
    pos = spec.find(primary)
    if pos < 0:
        return None
    tail = spec[pos + len(primary) :].lstrip()
    if not tail or tail.startswith("{") or tail.startswith("["):
        return None
    if tail.startswith("<"):
        return None
    parts = [p.strip() for p in tail.split(",") if p.strip()]
    if len(parts) < 2:
        return None
    if any(_VALUE_SPEC_PLACEHOLDER_RE.match(p) for p in parts):
        return None
    if parts[-1].endswith("..."):
        return None
    if not all(_INLINE_OPTION_TOKEN_RE.match(p) for p in parts):
        return None
    return [{"value": p, "label": p} for p in parts]


def _options_from_delimited_default(raw_default: Optional[str], separator: str) -> Optional[List[dict]]:
    if not raw_default or separator not in raw_default:
        return None
    parts = [p.strip() for p in raw_default.split(separator) if p.strip()]
    if len(parts) < 2:
        return None
    if not all(_INLINE_OPTION_TOKEN_RE.match(p) for p in parts):
        return None
    return [{"value": p, "label": p} for p in parts]


def _brace_content_looks_like_json(inner: str) -> bool:
    """Help prose often includes JSON examples in ``{...}`` — not enum value sets."""
    body = (inner or "").strip()
    if not body:
        return False
    if '":"' in body or "':'" in body:
        return True
    if body.startswith('"') and ":" in body:
        return True
    if body.count(":") >= 2 and '"' in body:
        return True
    return False


def _is_json_object_param(value_spec: str, description: str) -> bool:
    """Flags whose value is a JSON object string (e.g. ``--chat-template-kwargs``)."""
    vs = (value_spec or "").strip().upper()
    desc = (description or "").lower()
    if re.fullmatch(r"JSON", vs) or vs.endswith(" JSON"):
        return True
    if "json object" in desc or "valid json" in desc:
        return True
    if re.search(r"\bjson\b", value_spec or "", re.IGNORECASE) and (
        "object" in desc or "e.g." in desc
    ):
        return True
    return False


def _parse_allowed_values_list(text: str) -> Optional[List[dict]]:
    """Parse ``allowed values: a, b, c`` including values continued on the next line(s)."""
    match = re.search(r"allowed\s+values:\s*", text, re.IGNORECASE)
    if not match:
        return None
    tail = text[match.end() :]
    tail = re.split(r"\(default:", tail, maxsplit=1, flags=re.IGNORECASE)[0]
    tail = re.split(r"\(env:", tail, maxsplit=1, flags=re.IGNORECASE)[0]
    tokens: List[str] = []
    for chunk in tail.split(","):
        for token in chunk.split():
            cleaned = token.strip().strip(".,;")
            if not cleaned:
                continue
            if not _INLINE_OPTION_TOKEN_RE.match(cleaned):
                continue
            tokens.append(cleaned)
    if not tokens:
        return None
    return [{"value": t, "label": t} for t in tokens]


def _extract_options(text: str) -> Optional[List[dict]]:
    allowed = _parse_allowed_values_list(text)
    if allowed:
        return allowed

    em = META_ENUM.search(text)
    if em:
        inner = em.group(1)
        if not _brace_content_looks_like_json(inner):
            parts = [x.strip() for x in inner.split(",") if x.strip()]
            if parts:
                return [{"value": p, "label": p} for p in parts]

    br = META_BRACKET.search(text)
    if br and "|" in br.group(1):
        parts = [x.strip() for x in br.group(1).split("|") if x.strip()]
        return [{"value": p, "label": p} for p in parts]

    av = ALLOWED_VALUES_RE.search(text)
    if av:
        parts = [x.strip() for x in av.group(1).split(",") if x.strip()]
        if parts:
            return [{"value": p, "label": p} for p in parts]

    dm = DICT_KEYS_RE.search(text)
    if dm:
        try:
            values = ast.literal_eval(dm.group(1))
            if isinstance(values, list):
                return [{"value": str(v), "label": str(v)} for v in values if str(v)]
        except Exception:
            return None
    return None


def _looks_like_single_metavar(value_spec: str) -> bool:
    token = (value_spec or "").strip().split()[0].rstrip(",")
    if not token or token.startswith("{") or token.startswith("["):
        return False
    if _VALUE_SPEC_PLACEHOLDER_RE.match(token):
        return True
    return bool(re.fullmatch(r"[A-Z][A-Z0-9_]*", token))


def _infer_multiple(value_spec: str, description: str) -> bool:
    desc = (description or "").lower()
    compact_spec = re.sub(r"\s+", "", value_spec or "")
    if any(marker in desc for marker in ("comma-separated", "comma separated", "csv")):
        return False
    if CSV_ELLIPSIS_SPEC_RE.fullmatch(compact_spec):
        return False
    hay = f"{value_spec} {description}".lower()
    if "..." in value_spec:
        if _RANGE_ELLIPSIS_RE.fullmatch(compact_spec):
            return False
        return True
    markers = (
        "repeatable",
        "repeated",
        "multiple times",
        "one or more",
        "list of",
        "path(s)",
        "paths)",
    )
    if not (value_spec or "").strip():
        markers = (
            "repeatable",
            "repeated",
            "multiple times",
            "one or more",
            "path(s)",
            "paths)",
        )
    elif (
        "list of" in hay
        and "," not in (value_spec or "")
        and "..." not in (value_spec or "")
    ):
        markers = tuple(m for m in markers if m != "list of")
    elif (value_spec or "").strip() and "..." not in (value_spec or ""):
        if _looks_like_single_metavar(value_spec):
            markers = tuple(m for m in markers if m != "one or more")
    return any(marker in hay for marker in markers)


def _normalize_default_fragment(raw: str) -> Optional[str]:
    """Trim and unwrap a single default token from llama-server help prose."""
    value = (raw or "").strip().strip(",")
    if not value:
        return None
    prose_tail = re.match(r"^(\S+)\s*,\s*[A-Za-z][A-Za-z-]*\s*$", value)
    if prose_tail:
        value = prose_tail.group(1)
    if re.fullmatch(r"""^['"].*['"]$""", value):
        value = value.strip("'\"")
    else:
        quoted = re.fullmatch(r"""['"]([^'"]*)['"].*""", value)
        if quoted:
            value = quoted.group(1).strip()
        else:
            value = value.rstrip(").;").strip()
    return value or None


def _extract_paren_default(text: str) -> Optional[str]:
    """Parse ``(default: …)`` including nested parentheses in the value."""
    match = re.search(r"\(default:\s*", text, re.IGNORECASE)
    if not match:
        return None
    value_start = match.end()
    depth = 1
    in_quote: Optional[str] = None
    i = value_start
    while i < len(text):
        ch = text[i]
        if in_quote:
            if ch == in_quote:
                in_quote = None
            i += 1
            continue
        if ch in "'\"":
            in_quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return _normalize_default_fragment(text[value_start:i])
        i += 1
    return _normalize_default_fragment(text[value_start:])


def _default_option_from_description(
    description: str, options: Optional[List[dict]]
) -> Optional[str]:
    if not options:
        return None
    desc = description or ""
    values = [str(opt.get("value")) for opt in options if opt.get("value") is not None]
    for value in values:
        escaped = re.escape(value)
        if re.search(rf"\b{escaped}\s*\(default\)", desc, re.IGNORECASE):
            return value
        if re.search(rf"\bdefaults?\s+(?:to\s+)?{escaped}\b", desc, re.IGNORECASE):
            return value
    return None


def _raw_default(text: str) -> Optional[str]:
    """
  Parse defaults from llama-server help.

  Handles ``(default: f16)``, ``default: 0.80``, and ``default: 'auto'`` without
  swallowing trailing ``(env: …)`` clauses.
    """
    if not text:
        return None

    paren = _extract_paren_default(text)
    if paren is not None:
        return paren

    match = re.search(r"(?i)\bdefault(?:\s+to)?\s*[:=]\s*", text)
    if not match:
        return None
    tail = text[match.end() :].strip()
    if not tail:
        return None
    tail = re.split(r"\s*\(env:", tail, maxsplit=1, flags=re.IGNORECASE)[0]
    tail = tail.split("\n", 1)[0].strip()
    tail = re.split(r"\.\s+[A-Z][a-z]", tail, maxsplit=1)[0]
    tail = re.split(r"\s+Type:\s*", tail, maxsplit=1)[0]
    if ")" in tail:
        before, _, after = tail.partition(")")
        if after.strip() and not after.strip().startswith(","):
            tail = before
    return _normalize_default_fragment(tail)


def _clean_description(description: str) -> str:
    """Drop ``(env: …)`` tails; keep human-readable help text for the UI."""
    cleaned = ENV_PAREN_RE.sub("", description or "").strip()
    return re.sub(r"\s{2,}", " ", cleaned).strip()


def _coerce_scalar_default(raw: str, scalar_type: str) -> Any:
    value = raw.strip().strip(",")
    if not value:
        return None
    lower = value.lower()
    if lower in {"none", "null", "disabled"}:
        return None
    if lower in {"true", "false"}:
        return lower == "true"
    if scalar_type == "int":
        match = re.match(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            return int(float(match.group(0)))
        return value
    if scalar_type == "float":
        match = re.match(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
        return value
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            return None
    return value


def _coerce_flag_default(raw: Optional[str]) -> Optional[bool]:
    if raw is None:
        return None
    lower = str(raw).strip().lower()
    if not lower:
        return None
    if lower in {"true", "on", "yes", "enabled", "enable"}:
        return True
    if lower in {"false", "off", "no", "disabled", "disable"}:
        return False
    if lower.startswith("enabled") or " enabled" in lower:
        return True
    if lower.startswith("disabled") or " disabled" in lower:
        return False
    return None


def _infer_scalar_type(
    value_spec: str,
    description: str,
    raw_default: Optional[str],
    options: Optional[List[dict]],
) -> str:
    text = f"{value_spec} {description}"
    tm = re.search(r"Type:\s*(\w+)", text)
    if tm:
        typ = tm.group(1).lower()
        if typ in {"int", "integer"}:
            return "int"
        if typ in {"float", "double"}:
            return "float"
        return "string"

    if re.search(r"\b(FLOAT|DOUBLE)\b", value_spec):
        return "float"
    if re.search(r"\b(INT|UINT|LONG|SHORT|PORT|SECONDS|INDEX)\b", value_spec):
        return "int"

    if raw_default:
        if re.fullmatch(r"[-+]?\d+\.\d+", raw_default.strip()):
            return "float"
        if re.fullmatch(r"[-+]?\d+", raw_default.strip()):
            return "int"

    if options:
        numeric = [opt.get("value") for opt in options]
        if numeric and all(re.fullmatch(r"[-+]?\d+", str(v)) for v in numeric):
            return "int"
    if re.search(r"\bN\b", value_spec):
        return "float" if raw_default and "." in raw_default else "int"
    return "string"


def _is_flag_only(value_spec: str, description: str) -> bool:
    del description
    vs = (value_spec or "").strip()
    if not vs:
        return True
    # e.g. ik_llama ``--embedding(s)`` → value spec ``(s)`` (optional plural in help text)
    if re.fullmatch(r"\([a-z]+\)", vs, flags=re.IGNORECASE):
        return True
    return False


def _ui_type(value_kind: str, scalar_type: str) -> str:
    if value_kind == "flag":
        return "bool"
    if value_kind == "json_object":
        return "json"
    if value_kind in {"csv_enum", "semicolon_enum"}:
        return "multiselect"
    if value_kind == "enum":
        return "select"
    if value_kind == "repeatable":
        return "list"
    return scalar_type


def _build_param_row(
    *,
    flags: List[str],
    spec: str,
    value_spec: str,
    description: str,
    section_id: Optional[str] = None,
    section_label: Optional[str] = None,
) -> Optional[dict]:
    flags = _unique_flags(flags)
    if not flags:
        return None
    if _REMOVED_ARGUMENT_RE.search(description or ""):
        return None

    positive_flag = _select_positive_flag(flags)
    primary_flag = _preferred_primary_flag(flags, spec)
    negative_flag = _select_negative_flag(flags) if positive_flag else None
    key = _flags_to_key(flags, spec)
    inline_options = _extract_inline_flag_options(spec, flags)
    options = inline_options or _extract_options(f"{value_spec} {description}")
    if negative_flag and _is_flag_only(value_spec, description):
        inline_options = None
        options = None
    csv_enum = bool(
        inline_options and _is_csv_enum_description(description)
    )
    raw_default = _raw_default(description)
    semicolon_enum = False
    if not options and _is_semicolon_enum_description(description):
        options = _options_from_delimited_default(raw_default, ";")
        semicolon_enum = bool(options)
    json_object = _is_json_object_param(value_spec, description)
    if json_object:
        options = None
        multiple = False
    multiple = (
        False
        if (csv_enum or semicolon_enum or json_object)
        else _infer_multiple(value_spec, description)
    )
    if inline_options and "..." not in (value_spec or ""):
        multiple = False
    elif (
        META_ENUM.search(value_spec or "")
        and "..." not in (value_spec or "")
        and not META_BRACKET.search(value_spec or "")
    ):
        multiple = False
    scalar_type = _infer_scalar_type(value_spec, description, raw_default, options)

    if json_object:
        value_kind = "json_object"
    elif csv_enum:
        value_kind = "csv_enum"
    elif semicolon_enum:
        value_kind = "semicolon_enum"
    elif multiple:
        value_kind = "repeatable"
    elif options:
        value_kind = "enum"
    elif _is_flag_only(value_spec, description):
        value_kind = "flag"
    else:
        value_kind = "scalar"

    default = None
    if raw_default is not None:
        if value_kind in {"csv_enum", "semicolon_enum"}:
            if isinstance(raw_default, str) and raw_default.strip():
                separator = ";" if value_kind == "semicolon_enum" else ","
                default = [
                    part.strip()
                    for part in raw_default.split(separator)
                    if part.strip()
                ]
        elif multiple:
            parsed = _coerce_scalar_default(raw_default, "string")
            default = parsed if isinstance(parsed, list) else None
        elif value_kind == "flag":
            default = _coerce_flag_default(raw_default)
        else:
            default = _coerce_scalar_default(raw_default, scalar_type)
    if default is None and value_kind == "enum":
        default = _default_option_from_description(description, options)

    reserved = any(flag in RESERVED_FLAGS for flag in flags)
    label = _human_label(key)
    display_description = _clean_description(description) or label

    return {
        "key": key,
        "label": label,
        "description": display_description,
        "flags": flags,
        "primary_flag": primary_flag,
        "negative_flag": negative_flag,
        "value_kind": value_kind,
        "scalar_type": scalar_type,
        "multiple": bool(multiple or value_kind in {"csv_enum", "semicolon_enum"}),
        "options": options,
        "default": default,
        "type": _ui_type(value_kind, scalar_type),
        "reserved": reserved,
        "section_id": section_id,
        "section_label": section_label,
    }


def _merge_param_rows(rows: List[dict]) -> List[dict]:
    by_key: Dict[str, dict] = {}
    for row in rows:
        key = row["key"]
        if key not in by_key:
            by_key[key] = dict(row)
            continue

        existing = by_key[key]
        existing["flags"] = list(
            dict.fromkeys((existing.get("flags") or []) + (row.get("flags") or []))
        )
        existing["primary_flag"] = existing.get("primary_flag") or row.get(
            "primary_flag"
        )
        existing["negative_flag"] = existing.get("negative_flag") or row.get(
            "negative_flag"
        )
        existing["reserved"] = bool(existing.get("reserved") or row.get("reserved"))
        if row.get("value_kind") == "json_object":
            existing["value_kind"] = "json_object"
            existing["type"] = "json"
            existing["multiple"] = False
            existing["options"] = None
        elif row.get("value_kind") in {"csv_enum", "semicolon_enum"}:
            existing["value_kind"] = row["value_kind"]
            existing["type"] = "multiselect"
            existing["multiple"] = True
            if row.get("options"):
                existing["options"] = row["options"]
            if row.get("default") is not None:
                existing["default"] = row["default"]
        elif row.get("multiple"):
            existing["multiple"] = True
            existing["value_kind"] = "repeatable"
            existing["type"] = "list"
        if row.get("options") and not existing.get("options"):
            existing["options"] = row["options"]
        if row.get("default") is not None and existing.get("default") is None:
            existing["default"] = row["default"]
        if len(row.get("description") or "") > len(existing.get("description") or ""):
            existing["description"] = row["description"]
        if existing.get("value_kind") == "scalar" and row.get("value_kind") in {
            "enum",
            "csv_enum",
            "semicolon_enum",
            "json_object",
        }:
            existing["value_kind"] = row["value_kind"]
            existing["type"] = row["type"]
            if row.get("value_kind") == "json_object":
                existing["options"] = None
                existing["multiple"] = False
            elif row.get("value_kind") == "enum" and row.get("options"):
                existing["options"] = row["options"]
                existing["multiple"] = False
        if existing.get("scalar_type") == "int" and row.get("scalar_type") == "float":
            existing["scalar_type"] = "float"
            if existing.get("value_kind") == "scalar":
                existing["type"] = "float"
    return list(by_key.values())


_CACHE_TYPE_KEY_RE = re.compile(
    r"^(?:cache_type_[kv](?:_draft)?|spec_draft_type_[kv])$"
)
_KV_CACHE_TYPE_TOKENS = frozenset(
    {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
)


def _is_cache_type_param(row: dict) -> bool:
    key = str(row.get("key") or "")
    if _CACHE_TYPE_KEY_RE.match(key):
        return True
    desc = (row.get("description") or "").lower()
    return "kv cache data type" in desc or "cache data type" in desc


def _enrich_cache_type_enum_options(params: List[dict]) -> None:
    """
    Some builds (e.g. ik_llama) document ``--cache-type-k-draft TYPE`` without an
    ``allowed values:`` line. Copy options from any sibling that has them, or use
  the standard KV-cache type set when we see a partial match.
    """
    reference: Optional[List[dict]] = None
    for row in params:
        opts = row.get("options")
        if not isinstance(opts, list) or not opts:
            continue
        values = {str(o.get("value")) for o in opts if o.get("value") is not None}
        if len(values & _KV_CACHE_TYPE_TOKENS) >= 3:
            reference = opts
            break
    if reference is None:
        reference = [{"value": v, "label": v} for v in sorted(_KV_CACHE_TYPE_TOKENS)]

    for row in params:
        if row.get("options"):
            continue
        if not _is_cache_type_param(row):
            continue
        row["options"] = list(reference)
        row["value_kind"] = "enum"
        row["type"] = "select"
        row["multiple"] = False


def parse_llama_server_help(text: str, engine: str) -> List[dict]:
    """
    Parse llama-server style help into param dicts (before section grouping).
    engine is kept for API compatibility; parsed metadata is engine-agnostic.
    """
    del engine
    lines = text.splitlines()
    raw: List[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if stripped.startswith("-") and "--" in line:
            block_first = line
            rest: List[str] = []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nstrip = nxt.lstrip()
                if nstrip.startswith("-") and "--" in nxt:
                    break
                if not nxt.strip():
                    i += 1
                    continue
                rest.append(nxt.strip())
                i += 1

            spec, desc = _split_spec_and_description(block_first)
            flags = LONG_FLAG_RE.findall(spec)
            description = " ".join(x for x in [desc, *rest] if x).strip()
            row = _build_param_row(
                flags=flags,
                spec=spec,
                value_spec=_extract_value_spec(spec, flags),
                description=description,
            )
            if row:
                raw.append(row)
            continue
        i += 1
    merged = _merge_param_rows(raw)
    _enrich_cache_type_enum_options(merged)
    return merged


def _attach_llama_sections(text: str, params: List[dict]) -> List[dict]:
    """Assign section_id/section_label using line order."""
    lines = text.splitlines()
    # Upstream llama-server uses ``----- section -----``; prose lines often end with ``:`` and would
    # falsely match ``LM_SECTION_HEADER``. ik_llama uses short ``foo:`` headers without dash banners.
    has_dash_banners = any(SECTION_RULE_LLAMA.match(L.strip()) for L in lines)
    use_colon_headers = not has_dash_banners
    section_id = "general"
    section_label = "General"
    flag_section: Dict[str, Tuple[str, str]] = {}
    for line in lines:
        stripped = line.strip()
        sm = SECTION_RULE_LLAMA.match(stripped)
        if sm:
            section_label = sm.group(1).strip()
            section_id = (
                re.sub(r"[^a-z0-9]+", "_", section_label.lower()).strip("_")
                or "general"
            )
            continue
        # ik_llama.cpp style: ``general:``, ``server:``, ``embedding:`` (no ``-----`` banner).
        sh = LM_SECTION_HEADER.fullmatch(stripped) if use_colon_headers else None
        if sh:
            section_label = sh.group(1).strip()
            section_id = (
                re.sub(r"[^a-z0-9]+", "_", section_label.lower()).strip("_")
                or "general"
            )
            continue
        if line.lstrip().startswith("-") and "--" in line:
            for flag in LONG_FLAG_RE.findall(line):
                flag_section[flag] = (section_id, section_label)

    for param in params:
        sid, slab = "general", "General"
        for flag in param.get("flags") or []:
            if flag in flag_section:
                sid, slab = flag_section[flag]
                break
        param["section_id"] = sid
        param["section_label"] = slab
    return params


def group_params_into_sections(params: List[dict]) -> List[dict]:
    seen: List[Tuple[str, str]] = []
    for param in params:
        sid = param.get("section_id") or "general"
        slab = param.get("section_label") or "General"
        if (sid, slab) not in seen:
            seen.append((sid, slab))

    sections: List[dict] = []
    for sid, slab in seen:
        plist = [
            {k: v for k, v in param.items() if k not in ("section_id", "section_label")}
            for param in params
            if (param.get("section_id") or "general") == sid
        ]
        plist.sort(key=lambda item: item.get("key") or "")
        sections.append({"id": sid, "label": slab, "params": plist})
    return sections


def _trim_llama_help_prologue(text: str) -> str:
    """Drop ggml/cuda log lines before the first ``----- section -----`` block (llama-server --help)."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if SECTION_RULE_LLAMA.match(line.strip()):
            return "\n".join(lines[i:])
    return text


def parse_llama_help_to_sections(text: str, engine: str) -> List[dict]:
    text = _trim_llama_help_prologue(text)
    params = parse_llama_server_help(text, engine)
    params = _attach_llama_sections(text, params)
    return group_params_into_sections(params)


_AUDIO_OPTION_TRANSPORT_FLAGS = frozenset(
    {"--load-option", "--session-option", "--request-option"}
)
_AUDIO_DOCUMENTATION_METAVARS = frozenset({"key=value"})


def _should_skip_audio_server_help_row(row: dict) -> bool:
    """Drop CLI transport/docs rows from ``audiocpp_server --help``."""
    primary = str(row.get("primary_flag") or "")
    if primary in _AUDIO_OPTION_TRANSPORT_FLAGS:
        return True
    key = str(row.get("key") or "")
    if key in _AUDIO_DOCUMENTATION_METAVARS or "=" in key:
        return True
    return False


_AUDIO_SCOPE_BY_SECTION = {
    "global": ("model", "server_config"),
    "supported_tasks": ("model", "server_config"),
    "available_input_options": ("request_option", "request_only"),
    "model_request_options": ("request_option", "key_value_option"),
    "model_session_options": ("session_option", "key_value_option"),
    "model_load_options": ("load_option", "key_value_option"),
    "common_output_options": ("request_option", "request_only"),
    "batch": ("request_option", "request_only"),
    "pipelines": ("request_option", "request_only"),
    "task_routing_and_media_roles": ("request_option", "request_only"),
    "common_generation": ("request_option", "request_only"),
    "inputs": ("request_option", "request_only"),
    "outputs": ("request_option", "request_only"),
    "streaming": ("request_option", "request_only"),
    "utility": ("request_option", "request_only"),
}
_AUDIO_PROCESS_KEYS = {
    "config",
    "host",
    "port",
    "backend",
    "device",
    "threads",
    "log",
    "log_file",
}
_AUDIO_MODEL_KEYS = {"family", "model", "task", "mode", "config", "weight"}
_AUDIO_STUDIO_RESERVED_KEYS = {"host", "port", "config", "model"}


def _audio_section_id(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label or "").lower()).strip("_") or "options"


def _audio_options_from_value_spec(value_spec: str) -> Optional[List[dict]]:
    value = str(value_spec or "").strip()
    value = value.strip("<>[]{}")
    if "|" not in value:
        return None
    parts = [part.strip() for part in value.split("|") if part.strip()]
    if len(parts) < 2 or any(" " in part for part in parts):
        return None
    return [{"value": part, "label": part} for part in parts]


def _audio_row_metadata(row: dict, section_id: str, source: str) -> dict:
    key = str(row.get("key") or "")
    if source == "server":
        scope, transport = "process", "server_flag"
    else:
        scope, transport = _AUDIO_SCOPE_BY_SECTION.get(
            section_id, ("request_option", "request_only")
        )
        if key in _AUDIO_MODEL_KEYS:
            scope, transport = "model", "server_config"
        elif key in {"backend", "device", "threads", "log", "log_file"}:
            scope, transport = "process", "server_config"
        elif key in {"load_option", "session_option", "request_option"}:
            scope = key
            transport = "key_value_option"

    row["scope"] = scope
    row["transport"] = transport
    row["emission"] = {
        "transport": transport,
        "flag": row.get("primary_flag"),
        **(
            {"option_key": row.get("option_key")}
            if row.get("option_key")
            else {}
        ),
    }
    row["read_only"] = scope == "request_option"
    row["reserved"] = bool(
        row.get("reserved")
        or key in _AUDIO_STUDIO_RESERVED_KEYS
        or key in {"load_option", "session_option", "request_option"}
    )
    row["applicability"] = {}
    if key == "backend":
        row["options"] = [
            {"value": value, "label": value}
            for value in (
                ["cpu", "cuda", "vulkan", "metal"]
                if source == "server"
                else ["cpu", "cuda", "vulkan", "metal", "best"]
            )
        ]
        row["value_kind"] = "enum"
        row["type"] = "select"
        row["scalar_type"] = "string"
    if key in {"device", "threads"}:
        row["minimum"] = 0 if key == "device" else 1
        row["scalar_type"] = "int"
        row["type"] = "int"
        row["value_kind"] = "scalar"
    return row


def parse_audio_cpp_help_to_sections(
    text: str, *, source: str = "cli"
) -> List[dict]:
    """Parse audio.cpp CLI/server and model-aware help.

    Upstream uses compact two-space-separated rows and model-specific headings
    such as ``Model load options``.  Request-only rows are catalogued for API
    guidance but explicitly marked read-only for startup configuration.
    """
    lines = str(text or "").splitlines()
    section_id = "process_options" if source == "server" else "global"
    section_label = "Process options" if source == "server" else "Global"
    raw: List[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        heading = LM_SECTION_HEADER.fullmatch(stripped)
        if heading and not stripped.startswith("--"):
            candidate = heading.group(1).strip()
            if candidate.lower() not in {"endpoints", "tasks"}:
                section_label = candidate
                section_id = _audio_section_id(candidate)
            i += 1
            continue

        if stripped.startswith("--") or (
            stripped.startswith("-") and "--" in stripped
        ):
            spec, inline_description = _split_spec_and_description(line)
            flags = _flags_from_help_spec(spec)
            if not flags:
                flags = LONG_FLAG_RE.findall(spec)
            description_lines = [inline_description] if inline_description else []
            i += 1
            while i < len(lines):
                next_line = lines[i]
                next_stripped = next_line.strip()
                if next_stripped.startswith("-") and "--" in next_stripped:
                    break
                if LM_SECTION_HEADER.fullmatch(next_stripped):
                    break
                if not next_stripped:
                    i += 1
                    break
                # A non-indented metadata line starts a different output block.
                if next_line and not next_line[0].isspace() and "=" in next_line:
                    break
                description_lines.append(next_stripped)
                i += 1

            value_spec = _extract_value_spec(spec, flags)
            description = " ".join(description_lines).strip()
            row = _build_param_row(
                flags=flags,
                spec=spec,
                value_spec=value_spec,
                description=description,
                section_id=section_id,
                section_label=section_label,
            )
            if row:
                option_transport_key = str(row.get("key") or "")
                if option_transport_key in {
                    "load_option",
                    "session_option",
                    "request_option",
                }:
                    remaining = value_spec.strip()
                    option_name, _, option_value_spec = remaining.partition(" ")
                    option_name = option_name.strip()
                    if option_name and not option_name.startswith("<"):
                        row["key"] = option_name
                        row["option_key"] = option_name
                        row["label"] = _human_label(option_name.replace(".", "_"))
                        row["value_spec"] = option_value_spec.strip()
                        row["scalar_type"] = _infer_scalar_type(
                            option_value_spec,
                            description,
                            _raw_default(description),
                            None,
                        )
                        row["value_kind"] = (
                            "flag"
                            if not option_value_spec.strip()
                            else "scalar"
                        )
                        row["type"] = _ui_type(
                            row["value_kind"], row["scalar_type"]
                        )
                        row["multiple"] = False
                pipe_options = _audio_options_from_value_spec(value_spec)
                if pipe_options:
                    row["options"] = pipe_options
                    row["value_kind"] = "enum"
                    row["type"] = "select"
                    row["multiple"] = False
                row = _audio_row_metadata(row, section_id, source)
                if option_transport_key in {
                    "load_option",
                    "session_option",
                    "request_option",
                }:
                    row["scope"] = option_transport_key
                    row["transport"] = "key_value_option"
                    row["read_only"] = option_transport_key == "request_option"
                    row["emission"] = {
                        "transport": "key_value_option",
                        "flag": row.get("primary_flag"),
                        "option_key": row.get("option_key"),
                    }
                if source != "server" or not _should_skip_audio_server_help_row(row):
                    raw.append(row)
            continue
        i += 1

    merged = _merge_param_rows(raw)
    if source == "server":
        merged = [
            row for row in merged if not _should_skip_audio_server_help_row(row)
        ]
        advertised = set(LONG_FLAG_RE.findall(text))
        process_specs = {
            "--config": ("<server.json>", "Generated server configuration path"),
            "--host": ("<ip>", "Listen host"),
            "--port": ("<port>", "Listen port"),
            "--backend": ("cpu|cuda|vulkan|metal", "Execution backend, default cuda"),
            "--device": ("<id>", "Backend device index"),
            "--threads": ("<n>", "Backend and OpenMP worker threads"),
            "--log": ("", "Enable framework logging"),
            "--log-file": ("<path>", "Write framework logs to a file"),
        }
        existing_keys = {str(row.get("key")) for row in merged}
        for flag, (value_spec, description) in process_specs.items():
            key = _snake_from_long_flag(flag)
            if flag not in advertised or key in existing_keys:
                continue
            spec = f"{flag} {value_spec}".strip()
            row = _build_param_row(
                flags=[flag],
                spec=spec,
                value_spec=value_spec,
                description=description,
                section_id="process_options",
                section_label="Process options",
            )
            if row:
                pipe_options = _audio_options_from_value_spec(value_spec)
                if pipe_options:
                    row["options"] = pipe_options
                    row["value_kind"] = "enum"
                    row["type"] = "select"
                    row["multiple"] = False
                merged.append(
                    _audio_row_metadata(row, "process_options", "server")
                )
    # ``_merge_param_rows`` keeps custom v2 metadata from the first occurrence;
    # restore it defensively for parsers fed duplicate model help rows.
    for row in merged:
        _audio_row_metadata(row, row.get("section_id") or section_id, source)
    return group_params_into_sections(merged)


def parse_audio_cpp_loader_list(text: str) -> List[str]:
    families: List[str] = []
    for line in str(text or "").splitlines():
        value = line.strip()
        if not value or value.startswith("registered_loaders="):
            continue
        if ":" in value:
            value = value.split(":", 1)[0].strip()
        if re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", value):
            families.append(value)
    return list(dict.fromkeys(families))


def parse_audio_cpp_loader_tasks(text: str) -> List[str]:
    """Extract task ids from ``family: task (modes)`` loader rows."""
    tasks: List[str] = []
    for line in str(text or "").splitlines():
        value = line.strip()
        if not value or ":" not in value:
            continue
        _, rest = value.split(":", 1)
        task = rest.strip().split("(", 1)[0].strip()
        if task and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*", task):
            tasks.append(task)
    return list(dict.fromkeys(tasks))


def parse_audio_cpp_inspection(text: str) -> dict:
    """Parse stable key/value output from ``audiocpp_cli --inspect``."""
    result: Dict[str, Any] = {
        "tasks": [],
        "languages": [],
        "configs": [],
        "weights": [],
        "capabilities": {},
    }
    for raw_line in str(text or "").splitlines():
        line = raw_line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key == "task":
            task_name, _, modes_value = value.partition(" modes=")
            result["tasks"].append(
                {
                    "task": task_name.strip(),
                    "modes": [
                        item.strip()
                        for item in modes_value.split(",")
                        if item.strip()
                    ],
                }
            )
        elif key in {"config", "weight"}:
            asset_id, _, path = value.partition(":")
            result[f"{key}s"].append({"id": asset_id, "path": path})
        elif key == "languages":
            result["languages"] = [
                item.strip() for item in value.split(",") if item.strip()
            ]
        elif key.startswith("supports_"):
            result["capabilities"][key] = value.lower() == "true"
        elif key in {"supported_tasks", "configs", "weights"}:
            try:
                result[f"{key}_count"] = int(value)
            except ValueError:
                result[f"{key}_count"] = 0
        else:
            result[key] = value
    result["task_names"] = [
        item["task"] for item in result["tasks"] if item.get("task")
    ]
    return result


def parse_lmdeploy_api_server_help(text: str) -> List[dict]:
    """Parse `lmdeploy serve api_server --help` output."""
    lines = text.splitlines()
    section_id = "options"
    section_label = "Options"
    raw: List[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        sh = LM_SECTION_HEADER.match(line.strip())
        if sh and "arguments" in line.lower():
            section_label = sh.group(1).strip()
            section_id = (
                re.sub(r"[^a-z0-9]+", "_", section_label.lower()).strip("_")
                or "options"
            )
            i += 1
            continue
        if line.strip() == "options:":
            section_label = "Options"
            section_id = "options"
            i += 1
            continue

        mo = LM_OPTION.match(line)
        if mo:
            spec, inline_desc = _split_spec_and_description(line)
            flags = _flags_from_help_spec(spec)
            desc_lines: List[str] = [inline_desc] if inline_desc else []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if (
                    LM_OPTION.match(nxt)
                    or LM_SECTION_HEADER.match(nxt.strip())
                    or nxt.strip() == "options:"
                ):
                    break
                if not nxt.strip():
                    i += 1
                    break
                desc_lines.append(nxt.strip())
                i += 1

            value_spec = _extract_value_spec(spec, flags)
            description = " ".join(desc_lines).strip()
            row = _build_param_row(
                flags=flags,
                spec=spec,
                value_spec=value_spec,
                description=description,
                section_id=section_id,
                section_label=section_label,
            )
            if row:
                raw.append(row)
            continue
        i += 1

    merged = _merge_param_rows(raw)
    for row in merged:
        row.setdefault("section_id", "options")
        row.setdefault("section_label", "Options")
    return merged


def lmdeploy_params_to_sections(params: List[dict]) -> List[dict]:
    for param in params:
        param.setdefault("section_id", "options")
        param.setdefault("section_label", "Options")
    return group_params_into_sections(params)


def _trim_vllm_serve_help_prologue(text: str) -> str:
    """Drop usage banner before the first ``options:`` or ConfigGroup section."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped == "options:" or VLLM_CONFIG_GROUP_HEADER.match(stripped):
            return "\n".join(lines[i:])
    return text


def _vllm_section_from_header(header: str) -> Tuple[str, str]:
    section_label = header.strip()
    section_id = (
        re.sub(r"[^a-z0-9]+", "_", section_label.lower()).strip("_") or "options"
    )
    return section_id, section_label


def parse_vllm_serve_help(text: str) -> List[dict]:
    """Parse ``vllm serve --help=all`` output (ConfigGroup sections + argparse layout)."""
    text = _trim_vllm_serve_help_prologue(text)
    lines = text.splitlines()
    section_id = "options"
    section_label = "Options"
    raw: List[dict] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if VLLM_HELP_FOOTER.match(stripped):
            break

        if stripped == "options:":
            section_id, section_label = _vllm_section_from_header("Options")
            i += 1
            continue

        cg = VLLM_CONFIG_GROUP_HEADER.match(stripped)
        if cg:
            section_id, section_label = _vllm_section_from_header(cg.group(1))
            i += 1
            continue

        if stripped.startswith("positional arguments:"):
            i += 1
            while i < len(lines):
                nxt = lines[i].strip()
                if VLLM_HELP_FOOTER.match(nxt):
                    break
                if nxt == "options:" or VLLM_CONFIG_GROUP_HEADER.match(nxt):
                    break
                i += 1
            continue

        mo = VLLM_OPTION.match(line)
        if mo:
            spec_part = mo.group(1).strip()
            inline_tail = (mo.group(2) or "").strip()
            flags = LONG_FLAG_RE.findall(spec_part)
            positive_flag = next(
                (f for f in flags if f.startswith("--") and not f.startswith("--no-")),
                flags[0] if flags else "",
            )
            line_for_spec = (
                f"{positive_flag} {inline_tail}".strip()
                if inline_tail
                else spec_part
            )
            spec, inline_desc = _split_spec_and_description(f"  {line_for_spec}")
            if inline_tail and not _inline_tail_looks_like_metavar(inline_tail):
                desc_lines: List[str] = [inline_tail.strip()]
            else:
                desc_lines = [inline_desc] if inline_desc else []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                nxt_stripped = nxt.strip()
                if VLLM_HELP_FOOTER.match(nxt_stripped):
                    break
                if (
                    VLLM_OPTION.match(nxt)
                    or nxt_stripped == "options:"
                    or VLLM_CONFIG_GROUP_HEADER.match(nxt_stripped)
                    or nxt_stripped.startswith("positional arguments:")
                ):
                    break
                if not nxt_stripped:
                    i += 1
                    break
                desc_lines.append(nxt_stripped)
                i += 1

            value_spec = _extract_value_spec(line_for_spec, flags)
            description = " ".join(desc_lines).strip()
            row = _build_param_row(
                flags=flags,
                spec=spec_part,
                value_spec=value_spec,
                description=description,
                section_id=section_id,
                section_label=section_label,
            )
            if row:
                raw.append(row)
            continue
        i += 1

    merged = _merge_param_rows(raw)
    for row in merged:
        row.setdefault("section_id", "options")
        row.setdefault("section_label", "Options")
    return merged


def vllm_params_to_sections(params: List[dict]) -> List[dict]:
    for param in params:
        param.setdefault("section_id", "options")
        param.setdefault("section_label", "Options")
    return group_params_into_sections(params)
