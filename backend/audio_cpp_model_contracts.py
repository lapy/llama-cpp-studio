"""Typed audio.cpp model-spec contracts (options, capabilities, dependencies).

Upstream keeps install/layout JSON in ``model_specs/`` and migrates families to
typed contracts there (``schema_version: 1``). That typed tree is the long-term
source of truth.

TEMPORARY: ``model_specs_v1/`` is a migration preview used only for families
that are not yet typed in ``model_specs/``. Studio's pre-v1 adapter reads that
preview (plus layout packages/sources from ``model_specs/``) so peer options and
catalog metadata still work during the cutover. It also seeds known runtime peer
deps that loaders expose but specs have not declared yet. Shrink this adapter as
upstream migrates families and fills ``dependencies``; remove it once every
active family has ``schema_version: 1`` in ``model_specs/`` with complete peers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from backend.logging_config import get_logger

logger = get_logger(__name__)

# Temporary bridge for families not yet on typed model_specs/ schema_version 1.
# Delete once upstream finishes migration off model_specs_v1/.
TEMPORARY_PRE_V1_ADAPTER = "pre_v1_model_specs_v1"
TEMPORARY_PRE_V1_ADAPTER_NOTE = (
    "TEMPORARY audio.cpp adapter: reading model_specs_v1/ for families not yet "
    "migrated to typed model_specs/ (schema_version 1), and seeding runtime peer "
    "deps still missing from upstream dependencies. Shrink both as upstream "
    "migrates; do not invent peer graphs beyond known loader option keys."
)

# TEMPORARY: runtime peer deps that exist in loaders but are not yet declared in
# upstream model_specs*/dependencies. Drop each row as soon as upstream seeds it.
TEMPORARY_PEER_DEPENDENCY_SEEDS: Dict[str, List[Dict[str, Any]]] = {
    "vevo2": [
        {
            "kind": "external",
            "family": "whisper",
            "scope": "load",
            "option": "whisper_model_path",
            "required": False,
            "required_when": [],
        }
    ],
    "outetts": [
        {
            "kind": "model",
            "family": "qwen3_forced_aligner",
            "scope": "session",
            "option": "aligner_path",
            "required": False,
            "required_when": [],
        }
    ],
}

_SCOPE_TO_STUDIO = {
    "session": "session_option",
    "load": "load_option",
    "request": "request_option",
}


def _read_json(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Skipping model spec %s: %s", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def _is_typed_contract(payload: dict) -> bool:
    if payload.get("schema_version") is not None:
        return True
    if isinstance(payload.get("dependencies"), list):
        return True
    if isinstance(payload.get("options"), dict):
        return True
    if isinstance(payload.get("capabilities"), dict) and payload.get("category"):
        return True
    return False


def _model_path_alias(local_option: str) -> Optional[str]:
    """TEMPORARY: map preview ``foo_path`` locals to runtime ``foo_model_path``.

    Remove once live loaders and typed specs share the same public option keys.
    """
    name = str(local_option or "").strip()
    if not name or "." in name:
        return None
    if name.endswith("_model_path"):
        return None
    if name.endswith("_path"):
        return f"{name[:-5]}_model_path"
    return None


def public_option_key(
    family: str,
    option: str,
    *,
    known_keys: Optional[Set[str]] = None,
) -> str:
    """Resolve a dependency/option local name to the public runtime key."""
    family_key = str(family or "").strip()
    local = str(option or "").strip()
    if not local:
        return ""
    candidates: List[str] = []
    if "." in local:
        candidates.append(local)
    else:
        candidates.append(f"{family_key}.{local}" if family_key else local)
        alias = _model_path_alias(local)
        if alias and family_key:
            candidates.append(f"{family_key}.{alias}")
    known = {str(k).strip() for k in (known_keys or set()) if str(k).strip()}
    if known:
        for candidate in candidates:
            if candidate in known:
                return candidate
    # TEMPORARY: prefer historical *_model_path while preview specs still use *_path.
    if len(candidates) > 1:
        return candidates[-1]
    return candidates[0]


def normalize_dependency(
    family: str,
    raw: dict,
    *,
    known_keys: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    kind = str(raw.get("kind") or "model").strip() or "model"
    peer_family = str(raw.get("family") or "").strip()
    scope = str(raw.get("scope") or "session").strip() or "session"
    option = str(raw.get("option") or "").strip()
    if not option:
        return None
    option_key = public_option_key(family, option, known_keys=known_keys)
    required = bool(raw.get("required"))
    required_when: List[Dict[str, Any]] = []
    for row in raw.get("required_when") or []:
        if not isinstance(row, dict):
            continue
        condition = {
            "scope": str(row.get("scope") or "").strip() or "request",
            "option_key": str(row.get("option_key") or "").strip(),
            "equals": row.get("equals"),
        }
        if condition["option_key"]:
            required_when.append(condition)
    # Legacy preview / closed companions PR used required_for string lists.
    if not required_when:
        for item in raw.get("required_for") or []:
            key = str(item or "").strip()
            if key:
                required_when.append(
                    {"scope": "request", "option_key": key, "equals": True}
                )
    path = raw.get("path")
    out: Dict[str, Any] = {
        "kind": kind,
        "family": peer_family,
        "scope": scope,
        "option": option if "." not in option else option.rsplit(".", 1)[-1],
        "option_key": option_key,
        "required": required,
        "required_when": required_when,
    }
    if isinstance(path, str) and path.strip():
        out["path"] = path.strip()
    return out


def normalize_contract(
    payload: dict,
    *,
    family_hint: str = "",
    source: str = "model_specs",
    known_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    family = str(payload.get("family") or family_hint or "").strip().lower()
    dependencies: List[Dict[str, Any]] = []
    for row in payload.get("dependencies") or []:
        normalized = normalize_dependency(family, row, known_keys=known_keys)
        if normalized:
            dependencies.append(normalized)

    options = payload.get("options") if isinstance(payload.get("options"), dict) else {}
    capabilities = (
        payload.get("capabilities")
        if isinstance(payload.get("capabilities"), dict)
        else {}
    )
    tasks = [
        str(task).strip()
        for task in (payload.get("tasks") or [])
        if str(task).strip()
    ]
    modes = [
        str(mode).strip()
        for mode in (payload.get("modes") or [])
        if str(mode).strip()
    ]
    languages = [
        str(lang).strip()
        for lang in (payload.get("languages") or [])
        if str(lang).strip()
    ]
    schema_version = payload.get("schema_version")
    typed = bool(schema_version is not None)
    temporary = (not typed) or source == "model_specs_v1"
    return {
        "family": family,
        "display_name": str(payload.get("display_name") or family).strip(),
        "description": str(payload.get("description") or "").strip(),
        "category": str(payload.get("category") or "").strip(),
        "status": str(payload.get("status") or "").strip(),
        "schema_version": schema_version,
        "typed": typed,
        "temporary": temporary,
        "adapter": TEMPORARY_PRE_V1_ADAPTER if temporary else None,
        "source": source,
        "tasks": tasks,
        "modes": modes,
        "languages": languages,
        "capabilities": capabilities,
        "options": options,
        "dependencies": dependencies,
        "ui": payload.get("ui") if isinstance(payload.get("ui"), dict) else {},
        "packages": payload.get("packages")
        if isinstance(payload.get("packages"), list)
        else [],
    }


def _merge_layout(primary: Optional[dict], overlay: dict) -> dict:
    """Keep runtime package/layout fields from primary when overlaying v1 metadata."""
    if not primary:
        return dict(overlay)
    merged = dict(overlay)
    for key in ("sources", "packages", "package_defaults", "layouts"):
        if key in primary and key not in merged:
            merged[key] = primary[key]
        elif key in primary and not merged.get(key):
            merged[key] = primary[key]
    if primary.get("family") and not merged.get("family"):
        merged["family"] = primary["family"]
    return merged


def _dependency_already_covered(
    existing: Sequence[dict], candidate: dict
) -> bool:
    option_key = str(candidate.get("option_key") or "").strip()
    option = str(candidate.get("option") or "").strip()
    peer = str(candidate.get("family") or "").strip()
    for row in existing:
        if not isinstance(row, dict):
            continue
        if option_key and row.get("option_key") == option_key:
            return True
        if option and row.get("option") == option:
            return True
        # Same peer family already declared (any option) is enough to skip the seed.
        if peer and row.get("family") == peer:
            return True
        # Runtime accepts both outetts.aligner_path and *.aligner_model_path.
        row_key = str(row.get("option_key") or "")
        if option_key.endswith("_model_path") and row_key == option_key.replace(
            "_model_path", "_path"
        ):
            return True
        if option_key.endswith("_path") and not option_key.endswith("_model_path"):
            if row_key == f"{option_key[:-5]}_model_path":
                return True
    return False


def apply_temporary_peer_dependency_seeds(
    contract: Dict[str, Any],
    *,
    known_keys: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Fill runtime peer deps missing from upstream specs (TEMPORARY).

    Prefer declared ``dependencies`` from model specs. Only seed gaps, and mark
    each seeded row so it can be removed as upstream catches up.
    """
    family = str(contract.get("family") or "").strip().lower()
    seeds = TEMPORARY_PEER_DEPENDENCY_SEEDS.get(family) or []
    if not seeds:
        return contract

    existing = list(contract.get("dependencies") or [])
    added: List[Dict[str, Any]] = []
    for raw in seeds:
        normalized = normalize_dependency(family, raw, known_keys=known_keys)
        if not normalized:
            continue
        if _dependency_already_covered(existing + added, normalized):
            continue
        seeded = dict(normalized)
        seeded["temporary_seed"] = True
        added.append(seeded)

    if not added:
        return contract

    out = dict(contract)
    out["dependencies"] = existing + added
    out["temporary_peer_seeds"] = True
    # Typed specs stay typed; only untyped contracts get the pre-v1 adapter flag.
    if not out.get("typed"):
        out["temporary"] = True
        if not out.get("adapter"):
            out["adapter"] = TEMPORARY_PRE_V1_ADAPTER
    logger.debug(
        "TEMPORARY peer seeds applied for %s: %s",
        family,
        [row.get("option_key") for row in added],
    )
    return out


def load_family_contract(
    source_path: Optional[str],
    family: str,
    *,
    known_keys: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load the best available contract for a family.

    Preference order:
    1. Typed ``model_specs/<family>.json`` with ``schema_version`` (stable).
    2. TEMPORARY ``model_specs_v1/<family>.json`` overlay for unmigrated families.
    3. Rich but unversioned content already present in ``model_specs/`` (rare).
    4. TEMPORARY peer-seed stub when only Studio runtime seeds exist.
    """
    root = Path(str(source_path or ""))
    family_key = str(family or "").strip().lower()
    if not root.is_dir() or not family_key:
        return None
    primary_path = root / "model_specs" / f"{family_key}.json"
    preview_path = root / "model_specs_v1" / f"{family_key}.json"
    primary = _read_json(primary_path) if primary_path.is_file() else None
    preview = _read_json(preview_path) if preview_path.is_file() else None

    contract: Optional[Dict[str, Any]] = None
    if primary and primary.get("schema_version") is not None:
        contract = normalize_contract(
            primary,
            family_hint=family_key,
            source="model_specs",
            known_keys=known_keys,
        )
    elif preview and _is_typed_contract(preview):
        # TEMPORARY pre-v1 adapter: shrink as families move into typed model_specs/.
        logger.debug(
            "Using temporary pre-v1 model_specs_v1 adapter for family %s", family_key
        )
        contract = normalize_contract(
            _merge_layout(primary, preview),
            family_hint=family_key,
            source="model_specs_v1",
            known_keys=known_keys,
        )
    elif primary and _is_typed_contract(primary):
        contract = normalize_contract(
            primary,
            family_hint=family_key,
            source="model_specs",
            known_keys=known_keys,
        )
    elif family_key in TEMPORARY_PEER_DEPENDENCY_SEEDS:
        contract = {
            "family": family_key,
            "display_name": family_key,
            "description": "",
            "category": "",
            "status": "",
            "schema_version": None,
            "typed": False,
            "temporary": True,
            "adapter": TEMPORARY_PRE_V1_ADAPTER,
            "source": "temporary_peer_seed",
            "tasks": [],
            "modes": [],
            "languages": [],
            "capabilities": {},
            "options": {},
            "dependencies": [],
            "ui": {},
            "packages": [],
        }

    if contract is None:
        return None
    return apply_temporary_peer_dependency_seeds(contract, known_keys=known_keys)


def temporary_adapter_families(
    contracts: Dict[str, Dict[str, Any]],
) -> List[str]:
    """Families still served by the temporary pre-v1 / peer-seed adapter."""
    return sorted(
        family
        for family, contract in contracts.items()
        if (
            contract.get("temporary")
            or contract.get("temporary_peer_seeds")
            or contract.get("adapter") == TEMPORARY_PRE_V1_ADAPTER
        )
    )


def load_family_contracts(
    source_path: Optional[str],
    *,
    families: Optional[Sequence[str]] = None,
    known_keys: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    root = Path(str(source_path or ""))
    if not root.is_dir():
        return {}
    family_set = {
        str(item).strip().lower()
        for item in (families or [])
        if str(item).strip()
    }
    if not family_set:
        for directory in (root / "model_specs", root / "model_specs_v1"):
            if not directory.is_dir():
                continue
            for path in directory.glob("*.json"):
                family_set.add(path.stem.lower())
        family_set.update(TEMPORARY_PEER_DEPENDENCY_SEEDS.keys())
    out: Dict[str, Dict[str, Any]] = {}
    for family in sorted(family_set):
        contract = load_family_contract(
            str(root), family, known_keys=known_keys
        )
        if contract and contract.get("family"):
            out[contract["family"]] = contract
    return out


def family_dependencies_map(
    contracts: Dict[str, Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        family: list(contract.get("dependencies") or [])
        for family, contract in contracts.items()
        if contract.get("dependencies")
    }


def contracts_fingerprint(contracts: Dict[str, Dict[str, Any]]) -> str:
    payload = {
        family: {
            "source": contract.get("source"),
            "schema_version": contract.get("schema_version"),
            "tasks": contract.get("tasks") or [],
            "dependencies": contract.get("dependencies") or [],
            "options": contract.get("options") or {},
            "capabilities": contract.get("capabilities") or {},
        }
        for family, contract in sorted(contracts.items())
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dependency_sidecar_fields(
    family: str,
    dependencies: Sequence[dict],
    *,
    field_enrichment: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build Studio sidecar path fields from normalized dependencies.

    Deduplicates public option keys and ``*_path`` / ``*_model_path`` aliases so
    the UI shows one peer field per runtime slot.
    """
    family_key = str(family or "").strip().lower()
    enrichment = field_enrichment or {}
    fields: List[Dict[str, Any]] = []
    seen_keys: Set[str] = set()
    for dep in dependencies:
        if not isinstance(dep, dict):
            continue
        option_key = str(dep.get("option_key") or "").strip()
        if not option_key:
            continue
        alias_keys = _option_key_aliases(option_key)
        if any(key in seen_keys for key in alias_keys):
            continue
        seen_keys.update(alias_keys)
        scope = _SCOPE_TO_STUDIO.get(
            str(dep.get("scope") or "session").strip(), "session_option"
        )
        peer = str(dep.get("family") or "").strip()
        kind = str(dep.get("kind") or "model").strip()
        temporary_seed = bool(dep.get("temporary_seed"))
        label_bits = []
        if peer:
            label_bits.append(peer.replace("_", " ").title())
        label_bits.append("path")
        base = {
            "key": option_key,
            "label": " ".join(label_bits).strip() or option_key,
            "type": "path",
            "scope": scope,
            "description": _dependency_description(family_key, dep),
            "required": bool(dep.get("required")),
            "dependency": {
                "kind": kind,
                "family": peer,
                "option": str(dep.get("option") or "").strip(),
                "scope": str(dep.get("scope") or "session").strip() or "session",
                "required": bool(dep.get("required")),
                "required_when": list(dep.get("required_when") or []),
                "path": dep.get("path"),
                "temporary_seed": temporary_seed,
            },
        }
        if isinstance(dep.get("path"), str) and dep["path"].strip():
            base["placeholder"] = dep["path"].strip()
        overlay = enrichment.get(option_key) or enrichment.get(
            str(dep.get("option") or "")
        )
        if isinstance(overlay, dict):
            for key in ("label", "description", "placeholder", "type"):
                if overlay.get(key):
                    base[key] = overlay[key]
        fields.append(base)
    return fields


def _option_key_aliases(option_key: str) -> List[str]:
    key = str(option_key or "").strip()
    if not key:
        return []
    aliases = [key]
    if key.endswith("_model_path"):
        aliases.append(f"{key[:-11]}_path")
    elif key.endswith("_path") and not key.endswith("_model_path"):
        aliases.append(f"{key[:-5]}_model_path")
    return aliases


def _dependency_description(family: str, dep: dict) -> str:
    peer = str(dep.get("family") or "").strip()
    kind = str(dep.get("kind") or "model").strip()
    required = bool(dep.get("required"))
    when = dep.get("required_when") or []
    temporary_seed = bool(dep.get("temporary_seed"))
    if kind == "bundled_model":
        path = str(dep.get("path") or "").strip()
        base = f"Bundled {peer or 'model'} asset"
        if path:
            base = f"{base} ({path})"
    elif kind == "external":
        base = f"External {peer or 'dependency'} path"
    elif peer:
        base = f"Peer model family `{peer}`"
    else:
        base = "Runtime dependency path"
    if required:
        text = f"{base} required by {family}."
    elif when:
        keys = ", ".join(
            str(row.get("option_key") or "")
            for row in when
            if isinstance(row, dict) and row.get("option_key")
        )
        text = f"{base} needed when {keys}." if keys else f"Optional {base.lower()} for {family}."
    else:
        text = f"Optional {base.lower()} for {family}."
    if temporary_seed:
        text = (
            f"{text} Studio fills this peer until upstream declares it in "
            "model-spec dependencies."
        )
    return text


def _normalize_path_leaf(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if ":" in text and not text.startswith(("/", ".")):
        text = text.split(":", 1)[-1]
    return text.lstrip("./").lower()


def _basename_set(paths: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for path in paths:
        leaf = _normalize_path_leaf(path)
        if not leaf:
            continue
        out.add(leaf)
        out.add(Path(leaf).name.lower())
    return out


def path_leaves_from_spec(payload: dict) -> Set[str]:
    """Collect basename leaves used by package/family matching."""
    leaves: Set[str] = set()
    for source in payload.get("sources") or []:
        if not isinstance(source, dict):
            continue
        files = source.get("files") if isinstance(source.get("files"), dict) else {}
        for value in files.values():
            if isinstance(value, str):
                leaves.update(_basename_set([value]))
        optional = (
            source.get("optional_files")
            if isinstance(source.get("optional_files"), dict)
            else {}
        )
        for value in optional.values():
            if isinstance(value, str):
                leaves.update(_basename_set([value]))
        tensors = source.get("tensors") if isinstance(source.get("tensors"), dict) else {}
        for tensor in tensors.values():
            if isinstance(tensor, dict):
                prefix = tensor.get("prefix")
                if isinstance(prefix, str) and prefix:
                    leaves.add(prefix.lower())
            elif isinstance(tensor, str) and tensor:
                leaves.update(_basename_set([tensor]))
    for package in payload.get("packages") or []:
        if not isinstance(package, dict):
            continue
        for value in package.get("files") or []:
            if isinstance(value, str):
                leaves.update(_basename_set([value]))
        for value in package.get("required_files") or []:
            if isinstance(value, str):
                leaves.update(_basename_set([value]))
    return leaves


# Optional UI labels for dependency path fields (keyed by public option key).
DEPENDENCY_FIELD_ENRICHMENT: Dict[str, Dict[str, Any]] = {
    "qwen3_asr.forced_aligner_model_path": {
        "label": "Forced aligner model path",
        "placeholder": "/data/models/audio-cpp/qwen3_forced_aligner_0_6b/Qwen3-ForcedAligner-0.6B",
        "description": (
            "Path to an installed Qwen3 Forced Aligner bundle. Required for word "
            "timestamps (aligned ASR). Install package qwen3_forced_aligner_0_6b first."
        ),
    },
    "qwen3_asr.forced_aligner_path": {
        "label": "Forced aligner model path",
        "placeholder": "/data/models/audio-cpp/qwen3_forced_aligner_0_6b/Qwen3-ForcedAligner-0.6B",
        "description": (
            "Path to an installed Qwen3 Forced Aligner bundle. Required for word "
            "timestamps (aligned ASR)."
        ),
    },
    "qwen3_asr.vad_model_path": {
        "label": "VAD model path (timestamp chunking)",
        "placeholder": "assets/framework/models/silero_vad",
        "description": (
            "Optional VAD used when word timestamps are enabled and audio is long "
            "enough to need chunking."
        ),
    },
    "qwen3_asr.vad_path": {
        "label": "VAD model path (timestamp chunking)",
        "placeholder": "assets/framework/models/silero_vad",
        "description": (
            "Optional VAD used when word timestamps are enabled and audio is long "
            "enough to need chunking."
        ),
    },
    "vibevoice_asr.vad_model_path": {
        "label": "VAD model path",
        "placeholder": "assets/framework/models/silero_vad",
        "description": "Silero VAD used when audio_chunk_mode=vad.",
    },
    "vibevoice_asr.vad_path": {
        "label": "VAD model path",
        "placeholder": "assets/framework/models/silero_vad",
        "description": "Silero VAD used when audio_chunk_mode=vad.",
    },
    "miotts.codec_model_path": {
        "label": "MioCodec model path",
        "description": "Required MioCodec peer model for MioTTS.",
    },
    "miotts.codec_path": {
        "label": "MioCodec model path",
        "description": "Required MioCodec peer model for MioTTS.",
    },
    "miotts.best_of_n_asr_model_path": {
        "label": "Best-of-N ASR model path",
        "description": "Optional Qwen3 ASR peer used when best-of-N is enabled.",
    },
    "miotts.best_of_n_asr_path": {
        "label": "Best-of-N ASR model path",
        "description": "Optional Qwen3 ASR peer used when best-of-N is enabled.",
    },
    "vevo2.whisper_model_path": {
        "label": "Whisper model path",
        "placeholder": "/data/models/audio-cpp/whisper",
        "description": (
            "Optional external Whisper directory used by VeVo2 for VC / S2S / SVC "
            "tasks."
        ),
    },
    "outetts.aligner_model_path": {
        "label": "Aligner model path",
        "placeholder": "/data/models/audio-cpp/qwen3_forced_aligner_0_6b/Qwen3-ForcedAligner-0.6B",
        "description": (
            "Optional Qwen3 Forced Aligner peer for OuteTTS voice cloning when the "
            "package does not embed an aligner."
        ),
    },
    "outetts.aligner_path": {
        "label": "Aligner model path",
        "placeholder": "/data/models/audio-cpp/qwen3_forced_aligner_0_6b/Qwen3-ForcedAligner-0.6B",
        "description": (
            "Optional Qwen3 Forced Aligner peer for OuteTTS voice cloning when the "
            "package does not embed an aligner."
        ),
    },
}


def load_model_spec_path_leaves(source_path: Optional[str]) -> Dict[str, Set[str]]:
    """Map family -> path leaves from ``model_specs`` (and typed package files)."""
    root = Path(str(source_path or ""))
    specs_dir = root / "model_specs"
    if not specs_dir.is_dir():
        return {}
    out: Dict[str, Set[str]] = {}
    for path in sorted(specs_dir.glob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        family = str(payload.get("family") or path.stem).strip().lower()
        if not family:
            continue
        leaves = path_leaves_from_spec(payload)
        if leaves:
            out[family] = leaves
    return out


__all__ = [
    "DEPENDENCY_FIELD_ENRICHMENT",
    "TEMPORARY_PEER_DEPENDENCY_SEEDS",
    "TEMPORARY_PRE_V1_ADAPTER",
    "TEMPORARY_PRE_V1_ADAPTER_NOTE",
    "apply_temporary_peer_dependency_seeds",
    "contracts_fingerprint",
    "dependency_sidecar_fields",
    "family_dependencies_map",
    "load_family_contract",
    "load_family_contracts",
    "load_model_spec_path_leaves",
    "normalize_contract",
    "normalize_dependency",
    "path_leaves_from_spec",
    "public_option_key",
    "temporary_adapter_families",
]
