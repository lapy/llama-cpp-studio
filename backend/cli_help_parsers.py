"""Parse --help text into structured parameter rows for the engine catalog."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from backend.flag_config_key import choose_primary_flag, flag_to_config_key

LONG_FLAG_RE = re.compile(r"--[a-zA-Z0-9][a-zA-Z0-9-]*")

SECTION_RULE_LLAMA = re.compile(r"^[-=]{3,}\s*(.+?)\s*[-=]{3,}\s*$")

# Metavar / enum tail after flags on same line
META_ENUM = re.compile(r"\{([^}]+)\}")
META_BRACKET = re.compile(r"\[([^\]]+)\]")


def _human_label(key: str) -> str:
    return " ".join(w.capitalize() for w in key.replace("_", " ").split())


def _infer_type_llama(flag_line: str, description: str) -> Tuple[str, Optional[List[dict]], Optional[Any]]:
    em = META_ENUM.search(flag_line)
    if em:
        raw = em.group(1)
        parts = [x.strip() for x in raw.split(",") if x.strip()]
        opts = [{"value": p, "label": p} for p in parts]
        return "select", opts, None
    br = META_BRACKET.search(flag_line)
    if br and "|" in br.group(1):
        parts = [x.strip() for x in br.group(1).split("|") if x.strip()]
        opts = [{"value": p, "label": p} for p in parts]
        return "select", opts, None
    if re.search(r"\bN\b|\bINT\b|\bUINT\b", flag_line):
        return "int", None, None
    if re.search(r"\bFLOAT\b|\bDOUBLE\b", flag_line):
        return "float", None, None
    tmp = flag_line
    for m in LONG_FLAG_RE.findall(flag_line):
        tmp = tmp.replace(m, " ", 1)
    tmp = re.sub(r"^[\s,-]+|[\s,-]+$", "", tmp).strip()
    if tmp in ("N", "INT", "UINT"):
        return "int", None, None
    if tmp in ("FLOAT", "DOUBLE"):
        return "float", None, None
    if tmp == "":
        return "bool", None, None
    return "string", None, None


def _parse_llama_option_block(first: str, rest_lines: List[str], engine: str) -> Optional[dict]:
    flags = LONG_FLAG_RE.findall(first)
    if not flags:
        return None
    primary = choose_primary_flag(flags) or flags[0]
    key = flag_to_config_key(primary, engine)
    desc_parts: List[str] = []
    m = re.match(r"^(\s*)(.*)$", first)
    body = m.group(2) if m else first
    for f in flags:
        body = body.replace(f, " ", 1)
    body = re.sub(r"^[\s,]+", "", body)
    body = re.sub(r"[\s,]+$", "", body)
    if body:
        desc_parts.append(body.strip())
    for ln in rest_lines:
        t = ln.strip()
        if t:
            desc_parts.append(t)
    description = " ".join(desc_parts).strip()
    ptype, options, default = _infer_type_llama(first, description)
    label = _human_label(key)
    return {
        "key": key,
        "label": label,
        "type": ptype,
        "description": description or label,
        "flags": list(dict.fromkeys(flags)),
        "options": options,
        "default": default,
    }


def parse_llama_server_help(text: str, engine: str) -> List[dict]:
    """
    Parse llama-server style help into param dicts (before section grouping).
    engine is llama_cpp or ik_llama.
    """
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
                if nxt.strip() == "" and rest:
                    i += 1
                    continue
                if len(nxt) - len(nxt.lstrip()) >= 8 and nstrip and not (
                    nstrip.startswith("-") and "--" in nxt
                ):
                    rest.append(nxt)
                    i += 1
                    continue
                if not nxt.strip():
                    i += 1
                    continue
                if nstrip.startswith("-") and "--" in nxt:
                    break
                rest.append(nxt)
                i += 1
            row = _parse_llama_option_block(block_first, rest, engine)
            if row:
                raw.append(row)
            continue
        i += 1

    # Dedupe by config key (merge flags)
    by_key: Dict[str, dict] = {}
    for row in raw:
        k = row["key"]
        if k not in by_key:
            by_key[k] = dict(row)
            continue
        ex = by_key[k]
        ex["flags"] = list(dict.fromkeys((ex.get("flags") or []) + (row.get("flags") or [])))
        if len(row.get("description") or "") > len(ex.get("description") or ""):
            ex["description"] = row["description"]
    return list(by_key.values())


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
            flags = LONG_FLAG_RE.findall(line)
            for f in flags:
                flag_section[f] = (section_id, section_label)

    for p in params:
        sid, slab = "general", "General"
        for f in p.get("flags") or []:
            if f in flag_section:
                sid, slab = flag_section[f]
                break
        p["section_id"] = sid
        p["section_label"] = slab
    return params


def group_params_into_sections(params: List[dict]) -> List[dict]:
    seen: List[Tuple[str, str]] = []
    for p in params:
        sid = p.get("section_id") or "general"
        slab = p.get("section_label") or "General"
        if (sid, slab) not in seen:
            seen.append((sid, slab))
    sections: List[dict] = []
    for sid, slab in seen:
        plist = [
            {k: v for k, v in p.items() if k not in ("section_id", "section_label")}
            for p in params
            if (p.get("section_id") or "general") == sid
        ]
        plist.sort(key=lambda x: x.get("key") or "")
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


LM_SECTION_HEADER = re.compile(r"^([A-Za-z][^:]{0,120}):\s*$")
# argparse: "  --foo BAR" or "  -h, --help" (optional -x / -xy before --long)
LM_OPTION = re.compile(
    r"^\s+(?:(?:-[a-zA-Z0-9]+),?\s*)*(--[a-zA-Z0-9][a-zA-Z0-9_-]*)(?:\s+(.*))?$"
)


def _lm_type_from_tail(tail: str) -> Tuple[str, Optional[List[dict]], Optional[Any]]:
    t = tail or ""
    em = META_ENUM.search(t)
    if em:
        parts = [x.strip() for x in em.group(1).split(",") if x.strip()]
        opts = [{"value": p, "label": p} for p in parts]
        return "select", opts, None
    m = re.search(r"Type:\s*(\w+)", t)
    if m:
        typ = m.group(1).lower()
        if typ == "int":
            return "int", None, None
        if typ == "float":
            return "float", None, None
        if typ == "bool" or typ == "boolean":
            return "bool", None, None
        if typ == "str":
            return "string", None, None
        if typ == "loads":
            return "string", None, None
    if not tail or tail.strip() in ("", "Enable",):
        return "bool", None, None
    return "string", None, None


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
            long_flag = mo.group(1)
            tail = mo.group(2) or ""
            desc_lines = [tail] if tail.strip() else []
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if LM_OPTION.match(nxt) or LM_SECTION_HEADER.match(nxt.strip()):
                    break
                if nxt.strip() == "":
                    i += 1
                    break
                nind = len(nxt) - len(nxt.lstrip())
                # Wrapped help lines (argparse uses deep indent); check LM_OPTION first so "  --next" is not eaten.
                if nind >= 2:
                    desc_lines.append(nxt.strip())
                    i += 1
                    continue
                break
            description = " ".join(x for x in desc_lines if x).strip()
            ptype, options, default = _lm_type_from_tail(tail + " " + description)
            key = flag_to_config_key(long_flag, "lmdeploy")
            label = _human_label(key)
            raw.append(
                {
                    "key": key,
                    "label": label,
                    "type": ptype,
                    "description": description or label,
                    "flags": [long_flag],
                    "options": options,
                    "default": default,
                    "section_id": section_id,
                    "section_label": section_label,
                }
            )
            continue
        i += 1

    by_key: Dict[str, dict] = {}
    for row in raw:
        k = row["key"]
        if k not in by_key:
            by_key[k] = dict(row)
            continue
        ex = by_key[k]
        ex["flags"] = list(dict.fromkeys((ex.get("flags") or []) + (row.get("flags") or [])))
        slabs = [ex.get("section_label") or "", row.get("section_label") or ""]
        slabs = [s for s in slabs if s]
        if len(set(slabs)) > 1:
            ex["section_label"] = " / ".join(sorted(set(slabs)))
        elif slabs:
            ex["section_label"] = slabs[0]
    return list(by_key.values())


def lmdeploy_params_to_sections(params: List[dict]) -> List[dict]:
    for p in params:
        p.setdefault("section_id", "options")
        p.setdefault("section_label", "Options")
    return group_params_into_sections(params)
