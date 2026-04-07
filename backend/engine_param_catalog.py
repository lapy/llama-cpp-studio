"""Persisted engine CLI parameter catalog (engine_params_catalog.yaml)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from backend.logging_config import get_logger

logger = get_logger(__name__)

CATALOG_FILENAME = "engine_params_catalog.yaml"
CSV_DESCRIPTION_MARKERS = ("comma-separated", "comma separated", "csv")


def _default_catalog_root() -> dict:
    return {"schema_version": 1, "engines": {}}


def read_catalog(store: Any) -> dict:
    """Read full catalog (thread-safe via DataStore._read_yaml)."""
    data = store._read_yaml(CATALOG_FILENAME)
    if not isinstance(data, dict):
        return _default_catalog_root()
    root = _default_catalog_root()
    root.update(data)
    eng = data.get("engines")
    root["engines"] = eng if isinstance(eng, dict) else {}
    return root


def get_version_entry(
    store: Any, engine: str, version: str
) -> Optional[dict]:
    """Return stored catalog entry for one engine version, or None."""
    data = read_catalog(store)
    eng = (data.get("engines") or {}).get(engine) or {}
    versions = eng.get("versions") or {}
    entry = versions.get(version)
    return entry if isinstance(entry, dict) else None


def upsert_version_entry(
    store: Any,
    engine: str,
    version: str,
    entry: dict,
) -> None:
    """Merge one version entry into the catalog YAML (atomic write, under lock)."""
    with store._lock:
        path = os.path.join(store._config_dir, CATALOG_FILENAME)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
                data = {}
        else:
            data = {}
        if not isinstance(data, dict):
            data = {}
        root = _default_catalog_root()
        root.update(data)
        root.setdefault("engines", {})
        root["engines"].setdefault(engine, {"versions": {}})
        root["engines"][engine].setdefault("versions", {})
        root["engines"][engine]["versions"][version] = dict(entry)
        store._write_yaml(path, root)


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def flags_from_entry(entry: Optional[dict]) -> List[str]:
    """Collect all long flags from a catalog entry."""
    if not entry or entry.get("scan_error"):
        return []
    out: List[str] = []
    for sec in entry.get("sections") or []:
        for p in sec.get("params") or []:
            for f in p.get("flags") or []:
                if isinstance(f, str) and f.startswith("--"):
                    out.append(f)
    return list(dict.fromkeys(out))


def _normalize_param_row(param: dict) -> dict:
    row = dict(param or {})
    flags = [str(f) for f in (row.get("flags") or []) if isinstance(f, str) and f.startswith("--")]
    flags = list(dict.fromkeys(flags))
    primary_flag = row.get("primary_flag")
    if not primary_flag and flags:
        positives = [f for f in flags if not f.startswith("--no-")]
        primary_flag = positives[-1] if positives else flags[-1]
    negative_flag = row.get("negative_flag")
    if not negative_flag:
        negatives = [f for f in flags if f.startswith("--no-")]
        negative_flag = negatives[-1] if negatives else None

    value_kind = row.get("value_kind")
    if not value_kind:
        if row.get("multiple") or row.get("type") == "list":
            value_kind = "repeatable"
        elif row.get("options"):
            value_kind = "enum"
        elif row.get("type") == "bool":
            value_kind = "flag"
        else:
            value_kind = "scalar"
    description = str(row.get("description") or "").lower()
    if value_kind == "repeatable" and any(marker in description for marker in CSV_DESCRIPTION_MARKERS):
        value_kind = "scalar"

    scalar_type = row.get("scalar_type")
    if not scalar_type:
        ui_type = row.get("type")
        if ui_type in {"int", "float", "string"}:
            scalar_type = ui_type
        else:
            scalar_type = "string"

    row["flags"] = flags
    row["primary_flag"] = primary_flag
    row["negative_flag"] = negative_flag
    row["value_kind"] = value_kind
    row["scalar_type"] = scalar_type
    row["multiple"] = bool(value_kind == "repeatable")
    if value_kind == "flag":
        row["type"] = "bool"
    elif value_kind == "enum":
        row["type"] = "select"
    elif value_kind == "repeatable":
        row["type"] = "list"
    else:
        row["type"] = scalar_type
    row["reserved"] = bool(row.get("reserved"))
    return row


def param_index_from_entry(entry: Optional[dict]) -> Dict[str, dict]:
    """Build config_key -> normalized param metadata from catalog."""
    out: Dict[str, dict] = {}
    if not entry or entry.get("scan_error"):
        return out
    for sec in entry.get("sections") or []:
        for raw in sec.get("params") or []:
            row = _normalize_param_row(raw)
            key = row.get("key")
            if not key:
                continue
            out[str(key)] = row
    return out


def _positive_flag_stem(flag: str) -> str:
    """Long-option name without leading dashes and without ``no-`` (so ``--no-embeddings`` → ``embeddings``)."""
    if not isinstance(flag, str):
        return ""
    s = flag.strip().lower()
    if not s.startswith("--"):
        return ""
    s = s[2:]
    if s.startswith("no-"):
        s = s[3:]
    return s


def _stem_is_embedding_mode_toggle(stem: str) -> bool:
    """
    True if the flag looks like the main “run as embedding server” switch.

    Matches ``--embedding``, ``--embeddings``, ``--pooling-embedding``, etc., without
    relying on the catalog config ``key`` (which may differ from the CLI name).
    """
    if not stem:
        return False
    if stem in ("embedding", "embeddings"):
        return True
    if stem.endswith("-embedding") or stem.endswith("-embeddings"):
        return True
    return False


def _row_embedding_mode_by_flags(row: dict) -> bool:
    ordered: List[str] = []
    pf = row.get("primary_flag")
    if pf:
        ordered.append(str(pf))
    for f in row.get("flags") or []:
        if isinstance(f, str):
            ordered.append(f)
    for f in ordered:
        if _stem_is_embedding_mode_toggle(_positive_flag_stem(f)):
            return True
    return False


def embedding_mode_config_key_from_entry(entry: Optional[dict]) -> Optional[str]:
    """
    Config key to set True for embedding mode for this engine version.

    Prefer boolean catalog params whose CLI flags look like ``--embeddings`` /
    ``--embedding`` (renames and compounds). If none match, fall back to the first
    boolean param whose config key starts with ``embedding`` (e.g. ``embeddings_only``).
    """
    if not entry or entry.get("scan_error"):
        return None
    index = param_index_from_entry(entry)
    by_flag: List[str] = []
    by_key_prefix: List[str] = []
    for key in sorted(index.keys()):
        row = index[key]
        if row.get("value_kind") != "flag":
            continue
        if _row_embedding_mode_by_flags(row):
            by_flag.append(key)
        elif str(key).startswith("embedding"):
            by_key_prefix.append(key)
    if by_flag:
        return by_flag[0]
    if by_key_prefix:
        return by_key_prefix[0]
    return None


def param_mapping_from_entry(entry: Optional[dict]) -> Dict[str, List[str]]:
    """Build config_key -> [cli flags] from catalog (primary flag first)."""
    m: Dict[str, List[str]] = {}
    for key, row in param_index_from_entry(entry).items():
        flags: List[str] = []
        if row.get("primary_flag"):
            flags.append(str(row["primary_flag"]))
        if row.get("negative_flag"):
            flags.append(str(row["negative_flag"]))
        for flag in row.get("flags") or []:
            if flag not in flags:
                flags.append(flag)
        if flags:
            m[key] = flags
    return m


def registry_payload_from_entry(
    engine: str,
    entry: Optional[dict],
    studio_sections: List[dict],
    *,
    has_active_engine: bool,
) -> dict:
    """API shape for param-registry: ordered ``sections`` (studio first, then CLI help groups)."""
    scan_error = (entry or {}).get("scan_error") if entry else None
    cli_sections = list((entry or {}).get("sections") or []) if entry else []
    scan_pending = bool(has_active_engine and entry is None)

    sections_out: List[dict] = []
    for sec in studio_sections:
        sec_copy = dict(sec)
        sec_copy["params"] = []
        for p in sec.get("params") or []:
            q = _normalize_param_row(p)
            q.setdefault("supported", True)
            sec_copy["params"].append(q)
        sections_out.append(sec_copy)
    for sec in cli_sections:
        sec_copy = dict(sec)
        sec_copy["params"] = []
        for p in sec.get("params") or []:
            q = _normalize_param_row(p)
            q.setdefault("supported", True)
            sec_copy["params"].append(q)
        sections_out.append(sec_copy)

    return {
        "schema_version": 1,
        "engine": engine,
        "sections": sections_out,
        "scan_error": scan_error,
        "scan_pending": bool(scan_pending and not scan_error),
        "scanned_at": (entry or {}).get("scanned_at") if entry else None,
    }
