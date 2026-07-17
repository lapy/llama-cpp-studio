"""Runtime package discovery for audio.cpp (loaders + manager + model_specs)."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from backend.logging_config import get_logger

logger = get_logger(__name__)

_SUFFIX_RE = re.compile(
    r"(_(?:\d+(?:\.\d+)?[bBmM]|bf16|fp16|fp32|int8|q\d+|v\d+(?:\.\d+)*))+$",
    re.IGNORECASE,
)
_SIZE_TOKEN_RE = re.compile(
    r"^(?:\d+(?:\.\d+)?[bBmM]|bf16|fp16|fp32|int8|q\d+|v\d+(?:\.\d+)*)$",
    re.IGNORECASE,
)
_CONVERSION_TASKS = frozenset({"vc", "svc", "s2s"})
_SPEECH_TASKS = frozenset({"tts", "vdes", "clon"})
_ASR_TASKS = frozenset({"asr"})
_FORCE_TASKS_RUN_KEYS = frozenset(
    {
        "task-route",
        "task_route",
        "route",
        "source-audio",
        "source_audio",
        "target-voice",
        "target_voice",
        "prosody-ref",
        "prosody_ref",
    }
)


@dataclass
class DiscoveredPackage:
    package_id: str
    family: Optional[str]
    standalone: bool
    tasks: List[str] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    parent_package_id: Optional[str] = None
    match_score: float = 0.0
    match_reason: str = ""
    discovery_source: str = "heuristic"


@dataclass
class PackageDiscoveryIndex:
    families: Set[str] = field(default_factory=set)
    family_tasks: Dict[str, List[str]] = field(default_factory=dict)
    family_modes: Dict[str, List[str]] = field(default_factory=dict)
    packages: Dict[str, DiscoveredPackage] = field(default_factory=dict)
    spec_families: Set[str] = field(default_factory=set)

    def get(self, package_id: str) -> Optional[DiscoveredPackage]:
        return self.packages.get(str(package_id or "").strip())


def _normalize_path_leaf(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if ":" in text and not text.startswith(("/", ".")):
        # model_specs use roots like "model:config.yaml"
        text = text.split(":", 1)[-1]
    text = text.lstrip("./")
    return text.lower()


def _basename_set(paths: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for path in paths:
        leaf = _normalize_path_leaf(path)
        if not leaf:
            continue
        out.add(leaf)
        out.add(os.path.basename(leaf))
    return out


def _strip_package_stem(package_id: str) -> str:
    value = str(package_id or "").strip().lower()
    prev = None
    while prev != value:
        prev = value
        value = _SUFFIX_RE.sub("", value)
    return value.strip("_") or str(package_id or "").strip().lower()


def _token_set(value: str) -> Set[str]:
    parts = [p for p in re.split(r"[_\-.]+", str(value or "").lower()) if p]
    return {p for p in parts if not _SIZE_TOKEN_RE.match(p)}


def package_file_signature(package: dict) -> Set[str]:
    files: List[str] = []
    for item in package.get("required_files") or []:
        if isinstance(item, str):
            files.append(item)
    source = package.get("source") if isinstance(package.get("source"), dict) else {}
    placements = source.get("placements")
    if isinstance(placements, list):
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            for item in placement.get("required_files") or []:
                if isinstance(item, str):
                    files.append(item)
    # AST fallback: nested definition may still hold placements
    definition = source.get("definition")
    if isinstance(definition, dict):
        nested = definition.get("placements")
        if isinstance(nested, list):
            for placement in nested:
                if not isinstance(placement, dict):
                    continue
                for item in placement.get("required_files") or []:
                    if isinstance(item, str):
                        files.append(item)
                nested_source = placement.get("source")
                if isinstance(nested_source, dict):
                    for item in nested_source.get("required_files") or []:
                        if isinstance(item, str):
                            files.append(item)
    return _basename_set(files)


def load_model_specs(source_path: Optional[str]) -> Dict[str, Set[str]]:
    """Map family -> required path leaves from model_specs/*.json."""
    root = Path(str(source_path or ""))
    specs_dir = root / "model_specs"
    if not specs_dir.is_dir():
        return {}
    out: Dict[str, Set[str]] = {}
    for path in sorted(specs_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.debug("Skipping model_spec %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        family = str(payload.get("family") or path.stem).strip().lower()
        leaves: Set[str] = set()
        for source in payload.get("sources") or []:
            if not isinstance(source, dict):
                continue
            files = source.get("files") if isinstance(source.get("files"), dict) else {}
            for value in files.values():
                if isinstance(value, str):
                    leaves.update(_basename_set([value]))
            tensors = source.get("tensors") if isinstance(source.get("tensors"), dict) else {}
            for tensor in tensors.values():
                if isinstance(tensor, dict):
                    prefix = tensor.get("prefix")
                    if isinstance(prefix, str) and prefix:
                        leaves.add(prefix.lower())
        if family:
            out[family] = leaves
    return out


def _extract_placements_from_ast_source(source: dict) -> List[dict]:
    """Normalize AST CompositeSnapshotSource definition into placement dicts."""
    definition = source.get("definition") if isinstance(source, dict) else None
    if not isinstance(definition, dict):
        return []
    raw = definition.get("placements")
    if not isinstance(raw, list):
        return []
    placements: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        call = item.get("__call__")
        if call and call != "SnapshotPlacement":
            continue
        nested_source = item.get("source") if isinstance(item.get("source"), dict) else {}
        placements.append(
            {
                "repo_id": nested_source.get("repo_id") or item.get("repo_id"),
                "target_subdir": item.get("target_subdir") or "",
                "required_files": item.get("required_files")
                or nested_source.get("required_files")
                or [],
            }
        )
    return placements


def enrich_package_placements(package: dict) -> dict:
    """Ensure package.source.placements exists when only AST definition is present."""
    pkg = dict(package)
    source = dict(pkg.get("source") or {}) if isinstance(pkg.get("source"), dict) else {}
    if not source.get("placements"):
        placements = _extract_placements_from_ast_source(source)
        if placements:
            source["placements"] = placements
    pkg["source"] = source
    return pkg


def score_family_match(
    package: dict,
    family: str,
    *,
    spec_files: Optional[Set[str]] = None,
) -> Tuple[float, str]:
    package_id = str(package.get("id") or "").strip().lower()
    family_key = str(family or "").strip().lower()
    if not package_id or not family_key:
        return 0.0, "empty"

    score = 0.0
    reasons: List[str] = []

    if package_id == family_key:
        return 100.0, "exact_id"
    if package_id.startswith(family_key + "_"):
        score += 70.0
        reasons.append("id_prefix")

    stem = _strip_package_stem(package_id)
    if stem == family_key:
        score += 55.0
        reasons.append("stem_exact")
    elif stem.startswith(family_key + "_") or family_key.startswith(stem + "_"):
        score += 35.0
        reasons.append("stem_prefix")

    pkg_tokens = _token_set(package_id)
    fam_tokens = _token_set(family_key)
    if fam_tokens and fam_tokens.issubset(pkg_tokens):
        score += 25.0 + (5.0 * len(fam_tokens))
        reasons.append("token_subset")
    elif pkg_tokens & fam_tokens:
        overlap = len(pkg_tokens & fam_tokens)
        score += 8.0 * overlap
        reasons.append(f"token_overlap:{overlap}")

    # Prefer task-aligned families: …_tts package → *_tts loader over *_stt
    if "tts" in pkg_tokens and family_key.endswith("_tts"):
        score += 20.0
        reasons.append("tts_align")
    if "tts" in pkg_tokens and "stt" in fam_tokens:
        score -= 15.0
        reasons.append("stt_penalty")
    if ("asr" in pkg_tokens or "stt" in pkg_tokens) and (
        family_key.endswith("_asr") or family_key.endswith("_stt")
    ):
        score += 20.0
        reasons.append("asr_align")

    if spec_files:
        pkg_files = package_file_signature(package)
        if pkg_files:
            intersection = len(pkg_files & spec_files)
            union = len(pkg_files | spec_files) or 1
            coverage = intersection / max(len(spec_files), 1)
            jaccard = intersection / union
            layout = (coverage * 40.0) + (jaccard * 20.0)
            if layout > 0:
                score += layout
                reasons.append(f"layout:{coverage:.2f}")

    return score, ",".join(reasons) or "weak"


def match_package_family(
    package: dict,
    families: Sequence[str],
    specs: Dict[str, Set[str]],
    *,
    min_score: float = 40.0,
) -> Tuple[Optional[str], float, str]:
    best_family: Optional[str] = None
    best_score = 0.0
    best_reason = ""
    for family in families:
        score, reason = score_family_match(
            package, family, spec_files=specs.get(family)
        )
        if score > best_score:
            best_score = score
            best_family = family
            best_reason = reason
    if best_family and best_score >= min_score:
        return best_family, best_score, best_reason
    return None, best_score, best_reason or "no_match"


def _description_marks_dependency(description: str) -> bool:
    text = str(description or "").strip().lower()
    return text.startswith("subcomponent only.") or text.startswith("utility only.")


def _source_kind(package: dict) -> str:
    source = package.get("source") if isinstance(package.get("source"), dict) else {}
    return str(
        source.get("kind") or package.get("install_kind") or ""
    ).strip().lower()


def detect_standalone_graph(packages: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    """
    Return package_id -> {standalone: bool, parent_package_id: optional}.
    """
    enriched = [enrich_package_placements(p) for p in packages]
    by_id = {
        str(p.get("id") or "").strip(): p
        for p in enriched
        if str(p.get("id") or "").strip()
    }
    by_target: Dict[str, List[str]] = {}
    for pid, pkg in by_id.items():
        target = str(pkg.get("target_directory") or "").strip()
        if target:
            by_target.setdefault(target, []).append(pid)

    # Map repo_id / basename hints to package ids for placement edges
    repo_to_packages: Dict[str, List[str]] = {}
    for pid, pkg in by_id.items():
        source = pkg.get("source") if isinstance(pkg.get("source"), dict) else {}
        repo = source.get("repo_id")
        if isinstance(repo, str) and repo:
            repo_to_packages.setdefault(repo.lower(), []).append(pid)
        for repo_id in source.get("repo_ids") or []:
            if isinstance(repo_id, str) and repo_id:
                repo_to_packages.setdefault(repo_id.lower(), []).append(pid)

    result: Dict[str, Dict[str, Any]] = {
        pid: {"standalone": True, "parent_package_id": None} for pid in by_id
    }

    for pid, pkg in by_id.items():
        desc = str(pkg.get("description") or "")
        kind = _source_kind(pkg)
        if _description_marks_dependency(desc) or kind in {"utility"}:
            result[pid]["standalone"] = False
        operation = ""
        source = pkg.get("source") if isinstance(pkg.get("source"), dict) else {}
        if isinstance(source.get("operation_kind"), str):
            operation = source["operation_kind"]
        if "pytorch_to_safetensors" in operation or operation.startswith("utility"):
            result[pid]["standalone"] = False

    # Shared target_directory: utility/subcomponent siblings mark non-parent members
    for _target, ids in by_target.items():
        if len(ids) < 2:
            continue
        parents = []
        deps = []
        for pid in ids:
            pkg = by_id[pid]
            desc = str(pkg.get("description") or "")
            kind = _source_kind(pkg)
            is_dep = (
                _description_marks_dependency(desc)
                or kind in {"utility"}
                or "tokenizer" in pid.lower()
                or pid.lower().endswith("_model")
                or "audiovae" in pid.lower()
            )
            if is_dep:
                deps.append(pid)
            else:
                parents.append(pid)
        if parents and deps:
            parent = parents[0]
            for dep in deps:
                result[dep]["standalone"] = False
                result[dep]["parent_package_id"] = parent

    # Placement edges: parent placements referencing another package's repo
    for parent_id, pkg in by_id.items():
        if not result[parent_id]["standalone"]:
            continue
        source = pkg.get("source") if isinstance(pkg.get("source"), dict) else {}
        placements = source.get("placements") or []
        if not isinstance(placements, list):
            continue
        for placement in placements:
            if not isinstance(placement, dict):
                continue
            subdir = str(placement.get("target_subdir") or "")
            repo = str(placement.get("repo_id") or "").lower()
            # Non-root / sibling path suggests embedding a dependency
            looks_external = subdir.startswith("..") or (
                subdir
                and not subdir.startswith(".")
                and "/" not in subdir.rstrip("/")
                and subdir.lower()
                not in {
                    str(by_id[parent_id].get("target_directory") or "").lower()
                }
            )
            candidates: List[str] = []
            if repo:
                candidates.extend(repo_to_packages.get(repo, []))
            for other_id, other in by_id.items():
                if other_id == parent_id:
                    continue
                other_target = str(other.get("target_directory") or "")
                if other_target and (
                    other_target in subdir
                    or subdir.rstrip("/").endswith(other_target)
                ):
                    candidates.append(other_id)
            for dep_id in dict.fromkeys(candidates):
                if dep_id == parent_id:
                    continue
                # Only demote clear subcomponents / codecs / tokenizers
                dep_pkg = by_id.get(dep_id) or {}
                dep_desc = str(dep_pkg.get("description") or "")
                dep_id_l = dep_id.lower()
                if (
                    _description_marks_dependency(dep_desc)
                    or "tokenizer" in dep_id_l
                    or "codec" in dep_id_l
                    or "audiovae" in dep_id_l
                    or looks_external
                ):
                    result[dep_id]["standalone"] = False
                    if not result[dep_id]["parent_package_id"]:
                        result[dep_id]["parent_package_id"] = parent_id

    return result


def build_discovery_index(
    *,
    packages: Sequence[dict],
    families: Sequence[str],
    family_tasks: Optional[Dict[str, List[str]]] = None,
    family_modes: Optional[Dict[str, List[str]]] = None,
    source_path: Optional[str] = None,
) -> PackageDiscoveryIndex:
    from backend.feature_flags import audio_cpp_heuristic_discovery

    family_set = {str(f).strip().lower() for f in families if str(f).strip()}
    specs = load_model_specs(source_path)
    standalone_graph = detect_standalone_graph(packages)
    allow_heuristics = audio_cpp_heuristic_discovery()
    index = PackageDiscoveryIndex(
        families=family_set,
        family_tasks={k.lower(): list(v) for k, v in (family_tasks or {}).items()},
        family_modes={k.lower(): list(v) for k, v in (family_modes or {}).items()},
        spec_families=set(specs),
    )

    for raw in packages:
        package = enrich_package_placements(raw)
        package_id = str(package.get("id") or "").strip()
        if not package_id:
            continue

        pkg_family = str(package.get("family") or "").strip().lower() or None
        pkg_standalone = package.get("standalone")
        pkg_parent = package.get("parent_package_id")
        pkg_tasks = (
            package.get("tasks") if isinstance(package.get("tasks"), list) else None
        )
        pkg_modes = (
            package.get("modes") if isinstance(package.get("modes"), list) else None
        )

        discovery_source = "heuristic"
        if pkg_family:
            family, score, reason = pkg_family, 100.0, "package_json"
            discovery_source = "json"
        elif allow_heuristics:
            family, score, reason = match_package_family(
                package, sorted(family_set), specs
            )
            # Spec-only families (not yet in loaders) can still label deps for UX
            if not family and specs:
                family, score, reason = match_package_family(
                    package, sorted(specs.keys()), specs, min_score=55.0
                )
                if family and family not in family_set:
                    reason = f"{reason},spec_only"
        else:
            family, score, reason = None, 0.0, "heuristics_disabled"

        if isinstance(pkg_standalone, bool):
            graph = {
                "standalone": pkg_standalone,
                "parent_package_id": (
                    str(pkg_parent).strip() if pkg_parent else None
                ),
            }
            if discovery_source != "json":
                discovery_source = "json"
                if reason in {"", "heuristics_disabled", "no_match"}:
                    reason = "package_json_standalone"
        else:
            graph = standalone_graph.get(package_id) or {
                "standalone": True,
                "parent_package_id": None,
            }

        if pkg_tasks is not None:
            tasks = [str(t).strip() for t in pkg_tasks if str(t).strip()]
            discovery_source = "json"
        else:
            tasks = list(index.family_tasks.get(family or "", []))
        if pkg_modes is not None:
            modes = [str(m).strip() for m in pkg_modes if str(m).strip()]
            discovery_source = "json"
        else:
            modes = list(index.family_modes.get(family or "", []))

        index.packages[package_id] = DiscoveredPackage(
            package_id=package_id,
            family=family,
            standalone=bool(graph.get("standalone", True)),
            tasks=tasks,
            modes=modes,
            parent_package_id=graph.get("parent_package_id"),
            match_score=score,
            match_reason=reason,
            discovery_source=discovery_source,
        )
    return index


def resolve_api_endpoint(
    *,
    task: Optional[str] = None,
    inspection_tasks: Optional[Sequence[str]] = None,
    help_option_keys: Optional[Sequence[str]] = None,
    preferred_api_endpoint: Optional[str] = None,
) -> str:
    """Choose speech / transcriptions / tasks/run from inspect + help signals."""
    preferred = str(preferred_api_endpoint or "").strip()
    surface_map = {
        "speech": "/v1/audio/speech",
        "transcriptions": "/v1/audio/transcriptions",
        "transcription": "/v1/audio/transcriptions",
        "asr": "/v1/audio/transcriptions",
        "tasks": "/v1/tasks/run",
        "tasks/run": "/v1/tasks/run",
        "generic": "/v1/tasks/run",
    }
    if preferred in surface_map:
        return surface_map[preferred]
    if preferred in {
        "/v1/audio/speech",
        "/v1/audio/transcriptions",
        "/v1/tasks/run",
    }:
        return preferred

    task_key = str(task or "").strip().lower()
    tasks = {
        str(t).strip().lower()
        for t in (inspection_tasks or [])
        if str(t).strip()
    }
    if task_key:
        tasks.add(task_key)
    help_keys = {
        str(k).strip().lower().replace("_", "-")
        for k in (help_option_keys or [])
        if str(k).strip()
    }
    # Also accept underscore forms in the force set via normalization above
    help_force = bool(help_keys & {k.replace("_", "-") for k in _FORCE_TASKS_RUN_KEYS})

    has_conversion = bool(tasks & _CONVERSION_TASKS)
    has_speechish = bool(tasks & _SPEECH_TASKS)
    has_asr = bool(tasks & _ASR_TASKS)

    if has_asr and not has_speechish and not has_conversion:
        return "/v1/audio/transcriptions"
    if task_key in _ASR_TASKS and not has_speechish:
        return "/v1/audio/transcriptions"

    force_tasks_run = (
        has_conversion
        or help_force
        or (has_speechish and has_conversion)
    )
    if force_tasks_run or (tasks and not has_speechish and not has_asr):
        return "/v1/tasks/run"
    if has_speechish or task_key in _SPEECH_TASKS:
        return "/v1/audio/speech"
    return "/v1/tasks/run"


def resolve_defaults_key_for_endpoint(endpoint: str) -> str:
    if endpoint == "/v1/audio/transcriptions":
        return "transcription_defaults"
    if endpoint == "/v1/audio/speech":
        return "speech_defaults"
    return "task_defaults"


_INSTRUCTIONS_POLICIES = frozenset(
    {
        "caption_option",
        "text_prefix",
        "openai_instruct",
        "soft_tags",
        "none",
    }
)


def infer_instructions_policy(
    *,
    help_option_keys: Optional[Sequence[str]] = None,
    supports_style_condition: Optional[bool] = None,
    family: Optional[str] = None,
    docs_text: Optional[str] = None,
    inspection_policy: Optional[str] = None,
) -> str:
    """
    Return instructions_policy:
      caption_option | text_prefix | openai_instruct | soft_tags | none

    Prefer an explicit upstream ``instructions_policy`` from inspect/loaders JSON.
    """
    upstream = str(inspection_policy or "").strip().lower().replace("-", "_")
    if upstream in _INSTRUCTIONS_POLICIES:
        return upstream

    keys = {
        str(k).strip().lower().replace("-", "_")
        for k in (help_option_keys or [])
        if str(k).strip()
    }
    family_key = str(family or "").strip().lower()
    docs = str(docs_text or "").lower()

    if "caption" in keys or any(k.startswith("caption_") for k in keys):
        return "caption_option"
    if supports_style_condition and any(k.startswith("caption") for k in keys):
        return "caption_option"
    if "irodori" in family_key:
        return "caption_option"

    if family_key == "voxcpm2" or "voxcpm" in family_key or (
        "parentheses" in docs and "voxcpm" in docs
    ):
        # Common help still lists --instruct; docs say use text prefix
        return "text_prefix"

    if family_key == "omnivoice" or "omnivoice" in docs:
        return "soft_tags"

    if "qwen3_tts" in family_key or family_key.endswith("_voice_design"):
        return "openai_instruct"

    if "instruct" in keys or "instructions" in keys:
        return "openai_instruct"
    return "openai_instruct" if family_key else "none"


def load_optional_tts_docs(source_path: Optional[str]) -> str:
    root = Path(str(source_path or ""))
    for rel in ("docs/tts.md", "docs/TTS.md"):
        path = root / rel
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8")
            except OSError:
                return ""
    return ""


__all__ = [
    "DiscoveredPackage",
    "PackageDiscoveryIndex",
    "build_discovery_index",
    "detect_standalone_graph",
    "enrich_package_placements",
    "infer_instructions_policy",
    "load_model_specs",
    "load_optional_tts_docs",
    "match_package_family",
    "package_file_signature",
    "resolve_api_endpoint",
    "resolve_defaults_key_for_endpoint",
    "score_family_match",
]
