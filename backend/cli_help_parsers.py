"""Parse --help text into structured parameter rows for the engine catalog."""

from __future__ import annotations

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

LONG_FLAG_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9-]*")

SECTION_RULE_LLAMA = re.compile(r"^[-=]{3,}\s*(.+?)\s*[-=]{3,}\s*$")

META_ENUM = re.compile(r"\{([^}]+)\}")
META_BRACKET = re.compile(r"\[([^\]]+)\]")
DICT_KEYS_RE = re.compile(r"dict_keys\((\[[^\]]*\])\)")

LM_SECTION_HEADER = re.compile(r"^([A-Za-z][^:]{0,120}):\s*$")
LM_OPTION = re.compile(
    r"^\s+(?:(?:-[a-zA-Z0-9]+),?\s*)*(--[a-zA-Z0-9][a-zA-Z0-9_-]*)(?:\s+(.*))?$"
)

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
    return list(dict.fromkeys(f for f in flags if isinstance(f, str) and f.startswith("--")))


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


def _flags_to_key(flags: List[str]) -> str:
    primary = _select_positive_flag(flags)
    if primary:
        return _snake_from_long_flag(primary)
    chosen = flags[-1] if flags else "--unknown"
    key = _snake_from_long_flag(chosen)
    return key[3:] if key.startswith("no_") else key


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


def _extract_options(text: str) -> Optional[List[dict]]:
    em = META_ENUM.search(text)
    if em:
        parts = [x.strip() for x in em.group(1).split(",") if x.strip()]
        return [{"value": p, "label": p} for p in parts]

    br = META_BRACKET.search(text)
    if br and "|" in br.group(1):
        parts = [x.strip() for x in br.group(1).split("|") if x.strip()]
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


def _infer_multiple(value_spec: str, description: str) -> bool:
    hay = f"{value_spec} {description}".lower()
    if "..." in value_spec:
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
    return any(marker in hay for marker in markers)


def _raw_default(text: str) -> Optional[str]:
    match = re.search(r"(?i)\bdefault(?:\s+to)?\s*[:=]?\s*", text or "")
    if not match:
        return None
    tail = (text or "")[match.end() :].strip()
    if not tail:
        return None
    tail = re.split(r"\.\s+[A-Z][a-z]", tail, maxsplit=1)[0]
    tail = re.split(r"\s+Type:\s*", tail, maxsplit=1)[0]
    return tail.rstrip(").; ").strip()


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
        try:
            return int(float(value))
        except ValueError:
            return None
    if scalar_type == "float":
        try:
            return float(value)
        except ValueError:
            return None
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except Exception:
            return None
    return value


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
    if re.search(r"\b(INT|UINT|LONG|SHORT)\b", value_spec):
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
    return not bool((value_spec or "").strip())


def _ui_type(value_kind: str, scalar_type: str) -> str:
    if value_kind == "flag":
        return "bool"
    if value_kind == "enum":
        return "select"
    if value_kind == "repeatable":
        return "list"
    return scalar_type


def _build_param_row(
    *,
    flags: List[str],
    value_spec: str,
    description: str,
    section_id: Optional[str] = None,
    section_label: Optional[str] = None,
) -> Optional[dict]:
    flags = _unique_flags(flags)
    if not flags:
        return None

    primary_flag = _select_positive_flag(flags) or flags[-1]
    negative_flag = _select_negative_flag(flags)
    key = _flags_to_key(flags)
    options = _extract_options(f"{value_spec} {description}")
    multiple = _infer_multiple(value_spec, description)
    raw_default = _raw_default(description)
    scalar_type = _infer_scalar_type(value_spec, description, raw_default, options)

    if multiple:
        value_kind = "repeatable"
    elif options:
        value_kind = "enum"
    elif _is_flag_only(value_spec, description):
        value_kind = "flag"
    else:
        value_kind = "scalar"

    default = None
    if raw_default is not None:
        if multiple:
            parsed = _coerce_scalar_default(raw_default, "string")
            default = parsed if isinstance(parsed, list) else None
        elif value_kind == "flag":
            default = _coerce_scalar_default(raw_default, "string")
        else:
            default = _coerce_scalar_default(raw_default, scalar_type)

    reserved = any(flag in RESERVED_FLAGS for flag in flags)
    label = _human_label(key)

    return {
        "key": key,
        "label": label,
        "description": description or label,
        "flags": flags,
        "primary_flag": primary_flag,
        "negative_flag": negative_flag,
        "value_kind": value_kind,
        "scalar_type": scalar_type,
        "multiple": bool(multiple),
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
        existing["flags"] = list(dict.fromkeys((existing.get("flags") or []) + (row.get("flags") or [])))
        existing["primary_flag"] = existing.get("primary_flag") or row.get("primary_flag")
        existing["negative_flag"] = existing.get("negative_flag") or row.get("negative_flag")
        existing["reserved"] = bool(existing.get("reserved") or row.get("reserved"))
        if row.get("multiple"):
            existing["multiple"] = True
            existing["value_kind"] = "repeatable"
            existing["type"] = "list"
        if row.get("options") and not existing.get("options"):
            existing["options"] = row["options"]
        if row.get("default") is not None and existing.get("default") is None:
            existing["default"] = row["default"]
        if len(row.get("description") or "") > len(existing.get("description") or ""):
            existing["description"] = row["description"]
        if existing.get("value_kind") == "scalar" and row.get("value_kind") in {"enum", "flag"}:
            existing["value_kind"] = row["value_kind"]
            existing["type"] = row["type"]
        if existing.get("scalar_type") == "int" and row.get("scalar_type") == "float":
            existing["scalar_type"] = "float"
            if existing.get("value_kind") == "scalar":
                existing["type"] = "float"
    return list(by_key.values())


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
            row = _build_param_row(flags=flags, value_spec=_extract_value_spec(spec, flags), description=description)
            if row:
                raw.append(row)
            continue
        i += 1
    return _merge_param_rows(raw)


def _attach_llama_sections(text: str, params: List[dict]) -> List[dict]:
    """Assign section_id/section_label using line order."""
    lines = text.splitlines()
    section_id = "general"
    section_label = "General"
    flag_section: Dict[str, Tuple[str, str]] = {}
    for line in lines:
        sm = SECTION_RULE_LLAMA.match(line.strip())
        if sm:
            section_label = sm.group(1).strip()
            section_id = re.sub(r"[^a-z0-9]+", "_", section_label.lower()).strip("_") or "general"
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
            section_id = re.sub(r"[^a-z0-9]+", "_", section_label.lower()).strip("_") or "options"
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
            flags = LONG_FLAG_RE.findall(spec)
            desc_lines: List[str] = [inline_desc] if inline_desc else []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if LM_OPTION.match(nxt) or LM_SECTION_HEADER.match(nxt.strip()) or nxt.strip() == "options:":
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
