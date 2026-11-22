"""Persisted engine CLI parameter catalog (engine_params_catalog.yaml)."""

from __future__ import annotations

import copy
import os
import tempfile
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yaml

from backend.logging_config import get_logger

logger = get_logger(__name__)

CATALOG_FILENAME = "engine_params_catalog.yaml"
CSV_DESCRIPTION_MARKERS = ("comma-separated", "comma separated", "csv")
_YAML_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
_YAML_SAFE_DUMPER = getattr(yaml, "CSafeDumper", yaml.SafeDumper)
_CATALOG_CACHE: Dict[str, tuple[int, int, dict]] = {}
#: Serializes catalog writers. Deliberately not the store lock: the catalog is
#: megabytes of help text, and parsing/dumping it while holding the store-wide
#: lock blocked every other request (and the proxy) on the event loop.
_CATALOG_WRITE_LOCK = threading.RLock()


def _default_catalog_root() -> dict:
    return {"schema_version": 2, "engines": {}}


def _normalize_catalog_root(data: Any) -> dict:
    if not isinstance(data, dict):
        return _default_catalog_root()
    root = _default_catalog_root()
    root.update(data)
    root["schema_version"] = 2
    eng = data.get("engines")
    root["engines"] = eng if isinstance(eng, dict) else {}
    return root


def read_catalog(store: Any) -> dict:
    """Read full catalog, caching by file mtime so UI page loads do not reparse large YAML."""
    config_dir = getattr(store, "_config_dir", None)
    if not config_dir:
        return _normalize_catalog_root(store._read_yaml(CATALOG_FILENAME))

    path = os.path.join(store._config_dir, CATALOG_FILENAME)
    try:
        stat = os.stat(path)
        cache_key = os.path.abspath(path)
        cached = _CATALOG_CACHE.get(cache_key)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
    except OSError:
        stat = None
        cache_key = os.path.abspath(path)

    # Writers replace the file atomically (os.replace), so a plain read sees the
    # old or the new document. Do not take the store-wide lock here: holding it
    # across a multi-megabyte parse stalled every request on the event loop.
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.load(f, Loader=_YAML_SAFE_LOADER) or {}
    except FileNotFoundError:
        data = {}
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        data = {}
    root = _normalize_catalog_root(data)
    if stat is not None:
        _remember_catalog(path, root)
    return root


def _remember_catalog(path: str, root: dict) -> None:
    """Serve the document a writer just persisted from memory, never from a re-parse.

    ``engine_params_catalog.yaml`` grows with every scanned audio.cpp version
    (megabytes of help text). Re-parsing it on the event loop blocked the API
    and the proxy for seconds right after an install; writers now refresh this
    cache instead of dropping it, so a following ``/status`` poll is a dict
    lookup.
    """
    key = os.path.abspath(path)
    try:
        stat = os.stat(key)
    except OSError:
        _CATALOG_CACHE.pop(key, None)
        return
    _CATALOG_CACHE[key] = (stat.st_mtime_ns, stat.st_size, root)


def get_version_entry(store: Any, engine: str, version: str) -> Optional[dict]:
    """Return stored catalog entry for one engine version, or None."""
    data = read_catalog(store)
    eng = (data.get("engines") or {}).get(engine) or {}
    versions = eng.get("versions") or {}
    entry = versions.get(version)
    return entry if isinstance(entry, dict) else None


def get_model_profile_entry(
    store: Any, engine: str, version: str, fingerprint: str
) -> Optional[dict]:
    entry = get_version_entry(store, engine, version)
    profiles = (entry or {}).get("profiles") or {}
    profile = profiles.get(fingerprint)
    return profile if isinstance(profile, dict) else None


def _write_catalog_file(path: str, root: dict) -> None:
    """Atomically replace the catalog YAML without taking the store lock."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.dump(
                root,
                handle,
                Dumper=_YAML_SAFE_DUMPER,
                default_flow_style=False,
                sort_keys=False,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _catalog_for_write(store: Any) -> tuple[str, dict]:
    """A private, mutable copy of the catalog document to merge into."""
    path = os.path.join(str(store._config_dir), CATALOG_FILENAME)
    with _CATALOG_WRITE_LOCK:
        return path, copy.deepcopy(read_catalog(store))


def upsert_version_entry(
    store: Any,
    engine: str,
    version: str,
    entry: dict,
) -> None:
    """Merge one version entry into the catalog YAML (atomic write)."""
    with _CATALOG_WRITE_LOCK:
        path, root = _catalog_for_write(store)
        root.setdefault("engines", {}).setdefault(engine, {"versions": {}})
        root["engines"][engine].setdefault("versions", {})
        previous = root["engines"][engine]["versions"].get(version)
        merged = dict(previous) if isinstance(previous, dict) else {}
        merged.update(dict(entry))
        if isinstance(previous, dict) and "profiles" not in entry:
            merged["profiles"] = previous.get("profiles") or {}
        root["engines"][engine]["versions"][version] = merged
        _write_catalog_file(path, root)
        _remember_catalog(path, _normalize_catalog_root(root))


def upsert_model_profile_entry(
    store: Any,
    engine: str,
    version: str,
    fingerprint: str,
    profile: dict,
) -> None:
    """Persist one model-specific option profile without replacing version help."""
    with _CATALOG_WRITE_LOCK:
        path, root = _catalog_for_write(store)
        engine_row = root["engines"].setdefault(engine, {"versions": {}})
        versions = engine_row.setdefault("versions", {})
        version_row = versions.setdefault(version, {})
        profiles = version_row.setdefault("profiles", {})
        profiles[fingerprint] = dict(profile)
        _write_catalog_file(path, root)
        _remember_catalog(path, root)


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
    if value_kind == "repeatable" and any(
        marker in description for marker in CSV_DESCRIPTION_MARKERS
    ):
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
    row["multiple"] = bool(value_kind in ("repeatable", "space_list", "csv_enum", "semicolon_enum"))
    if value_kind == "flag":
        row["type"] = "bool"
    elif value_kind == "json_object":
        row["type"] = "json"
    elif value_kind in {"csv_enum", "semicolon_enum"}:
        row["type"] = "multiselect"
    elif value_kind == "enum":
        row["type"] = "select"
    elif value_kind in {"repeatable", "space_list"}:
        row["type"] = "list"
    else:
        row["type"] = scalar_type
    row["reserved"] = bool(row.get("reserved"))
    row["scope"] = str(row.get("scope") or "process")
    row["transport"] = str(row.get("transport") or "server_flag")
    emission = row.get("emission")
    if not isinstance(emission, dict):
        emission = {
            "transport": row["transport"],
            "flag": primary_flag,
        }
    row["emission"] = emission
    applicability = row.get("applicability")
    row["applicability"] = applicability if isinstance(applicability, dict) else {}
    row["required"] = bool(row.get("required"))
    row["nullable"] = not row["required"]
    row["read_only"] = bool(row.get("read_only"))
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


def scoped_param_index_from_entry(entry: Optional[dict]) -> Dict[str, Dict[str, dict]]:
    """Build ``scope -> key -> metadata`` for schema-v2 consumers."""
    out: Dict[str, Dict[str, dict]] = {}
    if not entry or entry.get("scan_error"):
        return out
    for section in entry.get("sections") or []:
        for raw in section.get("params") or []:
            row = _normalize_param_row(raw)
            key = str(row.get("key") or "")
            if not key:
                continue
            scope = str(row.get("scope") or "process")
            out.setdefault(scope, {})[key] = row
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
    *,
    has_active_engine: bool,
    profile: Optional[dict] = None,
    compatibility_warnings: Optional[List[str]] = None,
) -> dict:
    """API shape for param-registry: ordered CLI and model-profile sections."""
    scan_error = (entry or {}).get("scan_error") if entry else None
    cli_sections = list((entry or {}).get("sections") or []) if entry else []
    profile_sections = list((profile or {}).get("sections") or []) if profile else []
    scan_pending = bool(has_active_engine and entry is None)

    sections_out: List[dict] = []
    seen_params: set[tuple[str, str]] = set()
    for sec in [*profile_sections, *cli_sections]:
        sec_copy = dict(sec)
        sec_copy["params"] = []
        for p in sec.get("params") or []:
            q = _normalize_param_row(p)
            identity = (str(q.get("scope") or "process"), str(q.get("key") or ""))
            if identity in seen_params:
                continue
            seen_params.add(identity)
            q.setdefault("supported", True)
            sec_copy["params"].append(q)
        if sec_copy["params"]:
            sections_out.append(sec_copy)

    payload = {
        "schema_version": 2,
        "engine": engine,
        "sections": sections_out,
        "scan_error": scan_error,
        "scan_pending": bool(scan_pending and not scan_error),
        "scanned_at": (entry or {}).get("scanned_at") if entry else None,
        "profile_fingerprint": (profile or {}).get("fingerprint") if profile else None,
        "inspection": (profile or {}).get("inspection") if profile else None,
        "compatibility_warnings": list(compatibility_warnings or []),
    }
    if engine in {"sglang", "sglang_v100", "sglang_sxm2"}:
        # LLM and diffusion flags can share names but have different types,
        # defaults, and choices. Keep their catalogs separate all the way to
        # the form rather than merging e.g. the two quantization parameters.
        payload["runtime_profiles"] = {}
        diffusion = ((entry or {}).get("runtime_profiles") or {}).get("diffusion")
        if isinstance(diffusion, dict):
            diffusion_payload = registry_payload_from_entry(
                engine,
                {key: diffusion.get(key) for key in ("sections", "scan_error", "scanned_at")},
                has_active_engine=has_active_engine,
            )
            diffusion_payload.pop("runtime_profiles", None)
            payload["runtime_profiles"]["diffusion"] = diffusion_payload
    return payload
