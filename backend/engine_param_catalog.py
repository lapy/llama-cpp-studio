"""Persisted engine CLI parameter catalog (engine_params_catalog.yaml)."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from backend.logging_config import get_logger

logger = get_logger(__name__)

CATALOG_FILENAME = "engine_params_catalog.yaml"


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


def param_mapping_from_entry(entry: Optional[dict]) -> Dict[str, List[str]]:
    """Build config_key -> [cli flags] from catalog (primary flag first)."""
    m: Dict[str, List[str]] = {}
    if not entry or entry.get("scan_error"):
        return m
    for sec in entry.get("sections") or []:
        for p in sec.get("params") or []:
            key = p.get("key")
            flags = p.get("flags")
            if not key or not isinstance(flags, list) or not flags:
                continue
            m[str(key)] = [str(f) for f in flags if isinstance(f, str)]
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
            q = dict(p)
            q.setdefault("supported", True)
            sec_copy["params"].append(q)
        sections_out.append(sec_copy)
    for sec in cli_sections:
        sec_copy = dict(sec)
        sec_copy["params"] = []
        for p in sec.get("params") or []:
            q = dict(p)
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
