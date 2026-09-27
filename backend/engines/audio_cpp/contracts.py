"""Typed audio.cpp model-spec contracts (options, capabilities, dependencies)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from backend.logging_config import get_logger

logger = get_logger(__name__)

# Upstream deleted model_specs_v1/. Live model_specs/ still mix schema_version: 1
# with unversioned files that already carry options / capabilities. Load those
# as contracts; do not look for a preview tree.
TEMPORARY_PEER_DEPENDENCY_SEEDS: Dict[str, List[Dict[str, Any]]] = {
    "vevo2": [
        {
            "kind": "external",
            "family": "whisper",
            "scope": "load",
            "option": "whisper_model_path",
            "required": False,
        }
    ],
    "outetts": [
        {
            "kind": "model",
            "family": "qwen3_forced_aligner",
            "scope": "session",
            "option": "aligner_path",
            "required": False,
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
    candidate = local if "." in local else (
        f"{family_key}.{local}" if family_key else local
    )
    known = {str(k).strip() for k in (known_keys or set()) if str(k).strip()}
    if known and candidate in known:
        return candidate
    return candidate


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
    return {
        "family": family,
        "display_name": str(payload.get("display_name") or family).strip(),
        "description": str(payload.get("description") or "").strip(),
        "category": str(payload.get("category") or "").strip(),
        "status": str(payload.get("status") or "").strip(),
        "schema_version": schema_version,
        "typed": schema_version is not None,
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


def _looks_like_contract_spec(payload: Optional[dict]) -> bool:
    """True when model_specs JSON is usable as a Studio contract.

    schema_version: 1 is authoritative. Unversioned files still count when they
    already declare options, dependencies, or capabilities (post-0.7 trees no
    longer ship a separate model_specs_v1 preview).
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("schema_version") is not None:
        return True
    options = payload.get("options")
    if isinstance(options, dict) and any(
        isinstance(options.get(scope), list)
        for scope in ("request", "session", "load")
    ):
        return True
    dependencies = payload.get("dependencies")
    if isinstance(dependencies, list) and dependencies:
        return True
    capabilities = payload.get("capabilities")
    return isinstance(capabilities, dict) and bool(capabilities)


def _merge_peer_seeds(
    family: str,
    dependencies: List[Dict[str, Any]],
    *,
    known_keys: Optional[Set[str]] = None,
) -> List[Dict[str, Any]]:
    existing_options = {
        str(row.get("option") or "").strip()
        for row in dependencies
        if str(row.get("option") or "").strip()
    }
    existing_keys = {
        str(row.get("option_key") or "").strip()
        for row in dependencies
        if str(row.get("option_key") or "").strip()
    }
    for raw in TEMPORARY_PEER_DEPENDENCY_SEEDS.get(family) or []:
        seeded = normalize_dependency(family, raw, known_keys=known_keys)
        if not seeded:
            continue
        option = str(seeded.get("option") or "").strip()
        option_key = str(seeded.get("option_key") or "").strip()
        if option in existing_options or option_key in existing_keys:
            continue
        dependencies.append(seeded)
        if option:
            existing_options.add(option)
        if option_key:
            existing_keys.add(option_key)
    return dependencies


def load_family_contract(
    source_path: Optional[str],
    family: str,
    *,
    known_keys: Optional[Set[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Load a contract from ``model_specs/<family>.json`` (v1 or typed-shaped)."""
    root = Path(str(source_path or ""))
    family_key = str(family or "").strip().lower()
    if not root.is_dir() or not family_key:
        return None
    primary_path = root / "model_specs" / f"{family_key}.json"
    primary = _read_json(primary_path) if primary_path.is_file() else None
    if not _looks_like_contract_spec(primary):
        return None
    contract = normalize_contract(
        primary or {},
        family_hint=family_key,
        source="model_specs",
        known_keys=known_keys,
    )
    contract["dependencies"] = _merge_peer_seeds(
        family_key,
        list(contract.get("dependencies") or []),
        known_keys=known_keys,
    )
    return contract


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
        directory = root / "model_specs"
        if directory.is_dir():
            for path in directory.glob("*.json"):
                family_set.add(path.stem.lower())
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

    Deduplicates public option keys so the UI shows one peer field per runtime slot.
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
        if option_key in seen_keys:
            continue
        seen_keys.add(option_key)
        scope = _SCOPE_TO_STUDIO.get(
            str(dep.get("scope") or "session").strip(), "session_option"
        )
        peer = str(dep.get("family") or "").strip()
        kind = str(dep.get("kind") or "model").strip()
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


def _dependency_description(family: str, dep: dict) -> str:
    peer = str(dep.get("family") or "").strip()
    kind = str(dep.get("kind") or "model").strip()
    required = bool(dep.get("required"))
    when = dep.get("required_when") or []
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
]
