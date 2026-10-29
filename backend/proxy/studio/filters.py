"""Request-body parameter filters with llama-swap setParams semantics.

``setParams`` values are always applied, keyed by the request's own model id
via ``setParamsByID`` first. Keys ending in ``?`` are defaults: applied only
when the request does not already carry the value. Paths use llama-swap's
GJSON-style dots with backslash escapes for literal dots. Adapted from
llama-swap's internal filter behavior (MIT); see THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import re
from typing import Any, Optional

_SPLIT_RE = re.compile(r"(?<!\\)\.")


def _path_parts(path: str) -> list[str]:
    return [part.replace("\\.", ".") for part in _SPLIT_RE.split(path) if part != ""]


def _get_at(root: Any, parts: list[str]) -> Any:
    current = root
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return None
    return current


def _set_at(root: dict, parts: list[str], value: Any, *, only_if_absent: bool) -> None:
    current = root
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            if only_if_absent and part in current:
                return
            child = {}
            current[part] = child
        current = child
    leaf = parts[-1]
    if only_if_absent and leaf in current:
        return
    current[leaf] = value


def apply_request_filters(payload: dict, filters: Optional[dict], requested_model: str) -> dict:
    if not isinstance(filters, dict) or not payload:
        return payload
    params: dict = {}
    plain = filters.get("setParams")
    if isinstance(plain, dict):
        params.update(plain)
    by_id = filters.get("setParamsByID")
    if isinstance(by_id, dict) and requested_model:
        candidate = by_id.get(requested_model)
        if isinstance(candidate, dict):
            # setParamsByID overrides global setParams for shared keys.
            params.update(candidate)
    if not params:
        return payload
    result = dict(payload)
    for path, value in params.items():
        soft = path.endswith("?")
        parts = _path_parts(path[:-1] if soft else path)
        if not parts:
            continue
        _set_at(result, parts, value, only_if_absent=soft)
    return result
