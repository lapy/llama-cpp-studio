"""Curated audio.cpp package catalog provider."""

from __future__ import annotations

import ast
import asyncio
import json
import os
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional

from backend.audio_cpp_discovery import build_discovery_index
from backend.data_store import get_store
from backend.feature_flags import audio_cpp_enabled
from backend.engine_param_catalog import get_version_entry
from backend.logging_config import get_logger
from backend.model_catalog.base import normalized_item


logger = get_logger(__name__)

# Matches upstream tools/model_manager.py POSTPROCESS_SNAPSHOT_PACKAGE_IDS:
# SnapshotSource packages that still require model_manager post-processing.
POSTPROCESS_SNAPSHOT_PACKAGE_IDS = frozenset({"voxcpm2"})

# Substring markers matched against collected Hugging Face repo ids.
# Prefer org/repo prefixes so new gated snapshots (e.g. Stable Audio variants) match.
_GATED_REPO_MARKERS = (
    "kyutai/pocket-tts",
    "stabilityai/",
)
_GATED_TEXT_MARKERS = (
    "gated",
    "requires access",
    "accept the license",
    "accept the conditions",
)

_METHOD_LABELS = {
    "direct": "Direct HF",
    "composite": "Assemble (model manager)",
    "converter": "Convert (model manager)",
    "bundled": "Bundled asset",
    "unavailable": "Unavailable",
}

_METHOD_HINTS = {
    "direct": "Downloads a ready Hugging Face snapshot into the framework layout.",
    "composite": "Uses audio.cpp model_manager.py to assemble multiple repos and/or post-process weights.",
    "converter": "Uses audio.cpp model_manager.py to download/convert archives or local checkpoints.",
    "bundled": "Ships inside the audio.cpp source tree (assets/framework/models); no Hugging Face download.",
    "unavailable": "This package cannot be installed by the active audio.cpp model manager.",
}


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _ast_value(node: Optional[ast.AST]) -> Any:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return [_ast_value(item) for item in node.elts]
    if isinstance(node, ast.Dict):
        return {
            str(_ast_value(key)): _ast_value(value)
            for key, value in zip(node.keys, node.values)
            if key is not None
        }
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _ast_value(node.operand)
        return -value if isinstance(value, (int, float)) else None
    if isinstance(node, ast.Call):
        return {
            "__call__": _call_name(node.func),
            "args": [_ast_value(arg) for arg in node.args],
            **{
                keyword.arg: _ast_value(keyword.value)
                for keyword in node.keywords
                if keyword.arg
            },
        }
    # Concatenated string literals are folded into Constant by Python's parser.
    return None


def _catalog_expression(tree: ast.Module) -> Optional[ast.AST]:
    for statement in tree.body:
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "CATALOG"
            for target in statement.targets
        ):
            return statement.value
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "CATALOG"
        ):
            return statement.value
    return None


def _collect_repo_ids(value: Any) -> List[str]:
    output: List[str] = []
    if isinstance(value, dict):
        repo = value.get("repo_id")
        if isinstance(repo, str):
            output.append(repo)
        for nested in value.values():
            output.extend(_collect_repo_ids(nested))
    elif isinstance(value, list):
        for nested in value:
            output.extend(_collect_repo_ids(nested))
    return list(dict.fromkeys(output))


def _placements_from_composite(source: dict) -> List[dict]:
    raw = source.get("placements")
    if not isinstance(raw, list):
        return []
    placements: List[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        nested = item.get("source") if isinstance(item.get("source"), dict) else {}
        placements.append(
            {
                "repo_id": nested.get("repo_id") or item.get("repo_id"),
                "target_subdir": item.get("target_subdir") or "",
                "required_files": list(
                    item.get("required_files")
                    or nested.get("required_files")
                    or []
                ),
            }
        )
    return placements


def _source_payload(source: Any) -> dict:
    source = source if isinstance(source, dict) else {}
    call = source.get("__call__")
    if call == "SnapshotSource":
        return {
            "kind": "huggingface_snapshot",
            "repo_id": source.get("repo_id"),
            "revision": source.get("revision") or "main",
            "include_prefixes": source.get("include_prefixes") or [],
            "exclude_prefixes": source.get("exclude_prefixes") or [],
            "required_files": source.get("required_files") or [],
        }
    if call == "CompositeSnapshotSource":
        return {
            "kind": "composite_snapshot",
            "repo_ids": _collect_repo_ids(source),
            "placements": _placements_from_composite(source),
            "definition": source,
        }
    if call == "ConverterSource":
        operation = source.get("kind") or ""
        utility = (
            str(operation).startswith("utility_")
            or operation in {"pytorch_to_safetensors"}
        )
        return {
            "kind": "utility" if utility else "composite",
            "operation_kind": operation,
            "description": source.get("description"),
            "url": source.get("url"),
            "definition": source,
        }
    if call == "UnsupportedSource":
        return {
            "kind": "unsupported",
            "reason": source.get("reason") or "Unsupported by this audio.cpp version",
        }
    return {"kind": "unknown", "definition": source}


def compute_upstream_install_kind(package_id: str, source: Optional[dict]) -> str:
    """Mirror upstream package_install_kind() for AST and incomplete JSON payloads."""
    source = source if isinstance(source, dict) else {}
    source_kind = str(source.get("kind") or "").strip().lower()
    package_key = str(package_id or "").strip().lower()
    if package_key in POSTPROCESS_SNAPSHOT_PACKAGE_IDS:
        return "composite"
    if source_kind == "bundled_asset":
        return "bundled"
    if source_kind == "huggingface_snapshot":
        return "snapshot"
    if source_kind == "composite_snapshot":
        return "composite"
    if source_kind == "utility":
        return "utility"
    if source_kind == "composite":
        return "composite"
    if source_kind == "unsupported":
        return "unsupported"
    return "unsupported"


def resolve_studio_install_method(package: dict) -> str:
    """Map a catalog package to Studio's install path: direct | composite | converter | bundled."""
    package = package if isinstance(package, dict) else {}
    source = package.get("source") if isinstance(package.get("source"), dict) else {}
    package_id = str(package.get("id") or "")
    install_kind = str(package.get("install_kind") or "").strip().lower()
    source_kind = str(source.get("kind") or "").strip().lower()
    # Prefer recomputed kind when AST mistakenly stored source.kind as install_kind.
    if (
        not install_kind
        or install_kind == source_kind
        or install_kind
        in {"huggingface_snapshot", "composite_snapshot", "bundled_asset", "unknown"}
    ):
        install_kind = compute_upstream_install_kind(package_id, source)

    if install_kind == "bundled" or source_kind == "bundled_asset":
        return "bundled"
    if install_kind == "snapshot":
        return "direct"
    if install_kind == "utility":
        return "converter"
    if install_kind == "composite":
        # ConverterSource (non-utility) uses source.kind=composite; assemble uses
        # composite_snapshot or a post-processed SnapshotSource.
        if source_kind in {"composite", "utility"}:
            return "converter"
        return "composite"
    if source_kind == "huggingface_snapshot":
        return "direct"
    if source_kind == "composite_snapshot":
        return "composite"
    if source_kind in {"composite", "utility"}:
        return "converter"
    return "unavailable"


def install_method_label(method: str) -> str:
    return _METHOD_LABELS.get(str(method or "").strip().lower(), "Unavailable")


def install_method_hint(method: str) -> str:
    return _METHOD_HINTS.get(str(method or "").strip().lower(), _METHOD_HINTS["unavailable"])


def package_is_gated(
    source: Optional[dict], *, description: Optional[str] = None
) -> bool:
    repos = _collect_repo_ids(source if isinstance(source, dict) else {})
    lowered = [repo.lower() for repo in repos]
    if any(
        any(marker in repo for marker in _GATED_REPO_MARKERS) for repo in lowered
    ):
        return True
    text = " ".join(
        filter(
            None,
            [
                str(description or ""),
                str((source or {}).get("description") or "")
                if isinstance(source, dict)
                else "",
                str((source or {}).get("reason") or "")
                if isinstance(source, dict)
                else "",
            ],
        )
    ).lower()
    return any(marker in text for marker in _GATED_TEXT_MARKERS)


def parse_model_manager_catalog(source_text: str) -> List[dict]:
    """Extract catalog constants without importing or executing upstream Python."""
    tree = ast.parse(source_text)
    expression = _catalog_expression(tree)
    if not isinstance(expression, (ast.List, ast.Tuple)):
        raise ValueError("audio.cpp model manager CATALOG was not found")
    packages: List[dict] = []
    for element in expression.elts:
        value = _ast_value(element)
        if not isinstance(value, dict) or value.get("__call__") != "ModelPackage":
            continue
        source = _source_payload(value.get("source"))
        package_id = value.get("id")
        if not isinstance(package_id, str):
            continue
        installable = source.get("kind") not in {"unsupported", "unknown"}
        package = {
            "id": package_id,
            "display_name": value.get("display_name") or package_id,
            "target_directory": value.get("target_directory") or package_id,
            "description": value.get("description") or "",
            "required_files": value.get("required_files") or [],
            "source": source,
            "installable": installable,
            "install_kind": compute_upstream_install_kind(package_id, source),
            "usage_examples": list(value.get("usage_examples") or []),
        }
        # Preserve optional discovery fields when present (JSON path or richer forks).
        for key in (
            "family",
            "standalone",
            "parent_package_id",
            "tasks",
            "modes",
            "gated",
            "languages",
            "size_bytes",
        ):
            if key in value and value.get(key) is not None:
                package[key] = value.get(key)
        packages.append(package)
    return packages


def _manager_python(active: dict) -> str:
    """Prefer the Studio helper venv (Torch) so ``list --json`` can run."""
    python_name = "python.exe" if os.name == "nt" else "python"
    python_subdir = "Scripts" if os.name == "nt" else "bin"
    candidates: List[str] = []
    helper_venv = str(active.get("helper_venv_path") or "").strip()
    if helper_venv:
        candidates.append(helper_venv)
    # Default location created by AudioModelInstaller.ensure_helper_environment.
    from backend.audio_cpp_manager import _data_root

    candidates.append(
        os.path.join(_data_root(), "audio-cpp", "tools", "model-manager-venv")
    )
    for venv in candidates:
        candidate = os.path.join(venv, python_subdir, python_name)
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


def discover_bundled_framework_packages(
    *,
    source_path: str,
    families: Iterable[str],
    existing_package_ids: Iterable[str],
) -> List[dict]:
    """Expose loader families that ship under assets/framework/models/."""
    root = os.path.join(str(source_path or ""), "assets", "framework", "models")
    if not os.path.isdir(root):
        return []
    family_set = {str(f).strip().lower() for f in families if str(f).strip()}
    existing = {str(pid).strip().lower() for pid in existing_package_ids if str(pid).strip()}
    packages: List[dict] = []
    try:
        entries = sorted(os.listdir(root))
    except OSError:
        return []
    for name in entries:
        family = str(name or "").strip().lower()
        if not family or family not in family_set or family in existing:
            continue
        model_dir = os.path.join(root, name)
        if not os.path.isdir(model_dir):
            continue
        required: List[str] = []
        try:
            for dirpath, _dirnames, filenames in os.walk(model_dir):
                for filename in filenames:
                    rel = os.path.relpath(os.path.join(dirpath, filename), model_dir)
                    required.append(rel.replace("\\", "/"))
        except OSError:
            required = []
        packages.append(
            {
                "id": family,
                "display_name": f"{family} (bundled)",
                "target_directory": family,
                "description": (
                    "Bundled with the audio.cpp source tree under "
                    "assets/framework/models. No Hugging Face download required."
                ),
                "required_files": required,
                "source": {
                    "kind": "bundled_asset",
                    "path": model_dir,
                    "relative_path": f"assets/framework/models/{name}",
                },
                "installable": True,
                "install_kind": "bundled",
                "family": family,
                "standalone": True,
                "usage_examples": [],
            }
        )
    return packages


class AudioCppCatalogProvider:
    id = "audio_cpp"

    def __init__(self, store=None):
        self.store = store or get_store()
        self.status: dict = {"available": False, "reason": None}

    def _manager_packages(self, active: dict) -> List[dict]:
        manager_path = str(active.get("model_manager_path") or "")
        if not manager_path or not os.path.isfile(manager_path):
            self.status = {
                "available": False,
                "reason": "The active audio.cpp version does not contain tools/model_manager.py.",
            }
            return []

        try:
            process = subprocess.run(
                [_manager_python(active), manager_path, "list", "--json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=120,
                cwd=os.path.dirname(manager_path),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            process = subprocess.CompletedProcess(
                args=[], returncode=1, stdout="", stderr=str(exc)
            )
        packages: List[dict] = []
        source = "model_manager_ast"
        if process.returncode == 0:
            try:
                payload = json.loads(process.stdout)
                if isinstance(payload, list):
                    packages = [item for item in payload if isinstance(item, dict)]
                    source = "model_manager_json"
            except json.JSONDecodeError:
                pass

        if not packages:
            # The manager imports optional Torch tooling before argparse.  Catalog
            # discovery must still work before the lazy helper environment exists,
            # so parse only literal package declarations without executing source.
            try:
                with open(manager_path, "r", encoding="utf-8") as handle:
                    packages = parse_model_manager_catalog(handle.read())
                source = "model_manager_ast"
            except Exception as exc:
                self.status = {
                    "available": False,
                    "reason": f"audio.cpp package catalog failed: {exc}",
                }
                return []

        version_entry = get_version_entry(
            self.store, "audio_cpp", str(active.get("version") or "")
        )
        caps = (version_entry or {}).get("capabilities") or {}
        families = [
            str(f).strip() for f in (caps.get("families") or []) if str(f).strip()
        ]
        bundled = discover_bundled_framework_packages(
            source_path=str(active.get("source_path") or ""),
            families=families,
            existing_package_ids=[
                str(pkg.get("id") or "") for pkg in packages if isinstance(pkg, dict)
            ],
        )
        if bundled:
            packages = [*packages, *bundled]

        self.status = {
            "available": bool(packages),
            "source": source,
            "reason": (
                None
                if packages
                else "No package declarations were found in model_manager.py."
            ),
            "bundled_packages": len(bundled),
        }
        if source == "model_manager_ast" and process.stderr:
            self.status["manager_warning"] = process.stderr.strip()[-1000:]
        return packages

    @staticmethod
    def _infer_metadata(package_id: str) -> dict:
        from backend.cli_help_parsers import infer_audio_cpp_family_tasks

        lowered = package_id.lower()
        tasks = infer_audio_cpp_family_tasks(lowered)
        if not tasks:
            # Package ids sometimes omit family suffixes present on loaders.
            if "parakeet" in lowered:
                tasks = ["asr"]
            elif any(token in lowered for token in ("seed_vc", "vevo")):
                tasks = ["vc"]
            elif any(token in lowered for token in ("roformer", "demucs")):
                tasks = ["sep"]
            elif any(
                token in lowered
                for token in ("stable_audio", "ace_step", "heartmula")
            ):
                tasks = ["gen"]
            elif "vad" in lowered:
                tasks = ["vad"]
            elif "diar" in lowered or "sortformer" in lowered:
                tasks = ["diar"]
        modes = ["offline"] if tasks else []
        return {"family": package_id, "tasks": tasks, "modes": modes}

    def _normalize_packages(self, packages: Iterable[dict], active: dict) -> List[dict]:
        version_entry = get_version_entry(
            self.store, "audio_cpp", str(active.get("version") or "")
        )
        caps = (version_entry or {}).get("capabilities") or {}
        scanned_families = [
            str(f).strip() for f in (caps.get("families") or []) if str(f).strip()
        ]
        family_tasks = caps.get("family_tasks") if isinstance(caps.get("family_tasks"), dict) else {}
        contract_grade = str(caps.get("contract_grade") or "").strip() or None
        package_list = list(packages)
        index = build_discovery_index(
            packages=package_list,
            families=scanned_families,
            family_tasks=family_tasks,
            family_modes=(
                caps.get("family_modes")
                if isinstance(caps.get("family_modes"), dict)
                else None
            ),
            source_path=str(active.get("source_path") or ""),
            contract_grade=contract_grade,
        )
        active_commit = active.get("source_commit") or active.get("version")
        output: List[dict] = []
        for package in package_list:
            package_id = str(package.get("id") or "").strip()
            if not package_id:
                continue
            discovered = index.get(package_id)
            inferred = self._infer_metadata(package_id)
            family = (
                (discovered.family if discovered and discovered.family else None)
                or str(inferred.get("family") or package_id)
            )
            standalone = True if discovered is None else bool(discovered.standalone)
            tasks = list(discovered.tasks) if discovered and discovered.tasks else list(
                inferred.get("tasks") or []
            )
            modes = list(discovered.modes) if discovered and discovered.modes else list(
                inferred.get("modes") or []
            )
            installable = bool(package.get("installable", True)) and standalone
            verified = bool(
                installable and scanned_families and family in set(scanned_families)
            )
            parent = discovered.parent_package_id if discovered else None
            if not standalone:
                unavailable = (
                    f"Dependency/subcomponent package; install parent "
                    f"'{parent}' instead."
                    if parent
                    else "Dependency/subcomponent package; install its parent runtime package."
                )
            elif not package.get("installable", True):
                unavailable = (
                    (package.get("source") or {}).get("reason")
                    or "The active audio.cpp version cannot install this package."
                )
            elif not scanned_families:
                unavailable = "Activate and capability-scan audio.cpp to verify loader compatibility."
            elif family not in set(scanned_families):
                unavailable = (
                    f"The active audio.cpp build does not advertise loader family '{family}'."
                )
            else:
                unavailable = None

            source = package.get("source") if isinstance(package.get("source"), dict) else {}
            source_kind = str(source.get("kind") or "unknown")
            install_kind = str(
                package.get("install_kind")
                or compute_upstream_install_kind(package_id, source)
            )
            method_package = {
                **package,
                "id": package_id,
                "source": source,
                "install_kind": install_kind,
            }
            method = resolve_studio_install_method(method_package)
            if isinstance(package.get("gated"), bool):
                gated = bool(package.get("gated"))
            else:
                gated = package_is_gated(
                    source, description=str(package.get("description") or "")
                )
            features = [
                *tasks,
                "streaming" if "streaming" in modes else "",
                "prepared-bundle",
                "model-manager" if method in {"composite", "converter"} else "",
            ]
            output.append(
                normalized_item(
                    provider="audio_cpp",
                    item_id=package_id,
                    display_name=package.get("display_name") or package_id,
                    description=package.get("description") or "",
                    source={
                        "provider": "audio_cpp",
                        "package_id": package_id,
                        "id": package_id,
                        "upstream": source,
                        "engine_commit": active_commit,
                    },
                    artifact_format="mixed",
                    package_kind="prepared_bundle",
                    tasks=tasks,
                    family=family,
                    modes=modes,
                    languages=[],
                    features=features,
                    compatible_engines=["audio_cpp"] if verified else [],
                    compatibility={
                        "audio_cpp": {
                            "verified": verified,
                            "evidence": [
                                f"audio.cpp model_manager.py at {active_commit}",
                                (
                                    f"active loader scan advertises {family}"
                                    if family in set(scanned_families)
                                    else f"active loader scan does not advertise {family}"
                                ),
                                (
                                    f"discovery match score={discovered.match_score:.1f}"
                                    f" ({discovered.match_reason})"
                                    if discovered
                                    else "discovery fallback heuristics"
                                ),
                                (
                                    f"discovery_source="
                                    f"{discovered.discovery_source if discovered else 'heuristic'}"
                                ),
                                f"install_method={method}",
                            ],
                        }
                    },
                    install_variants=[
                        {
                            "id": package_id,
                            "label": package.get("display_name") or package_id,
                            "method": method,
                            "method_label": install_method_label(method),
                            "method_hint": install_method_hint(method),
                            "uses_model_manager": method in {"composite", "converter"},
                            "install_kind": install_kind,
                            "installable": bool(installable and verified),
                            "required_files": package.get("required_files") or [],
                            "source": source,
                            "external_inputs_required": source_kind == "utility",
                            "external_inputs_optional": (
                                source_kind == "composite"
                                and source.get("operation_kind") == "demucs_reference"
                            ),
                            "operation_kind": source.get("operation_kind"),
                            "operation_description": source.get("description"),
                        }
                    ],
                    size_bytes=package.get("size_bytes"),
                    gated=gated,
                    release_status=(
                        "stable"
                        if contract_grade == "full"
                        and discovered
                        and discovered.discovery_source == "json"
                        else (
                            "stable"
                            if discovered and discovered.discovery_source == "json"
                            else "experimental"
                        )
                    ),
                    unavailable_reason=unavailable,
                    metadata={
                        "target_directory": package.get("target_directory"),
                        "usage_examples": package.get("usage_examples") or [],
                        "install_kind": install_kind,
                        "install_method": method,
                        "discovery": {
                            "family": family,
                            "standalone": standalone,
                            "parent_package_id": parent,
                            "match_score": discovered.match_score if discovered else None,
                            "match_reason": discovered.match_reason if discovered else None,
                            "discovery_source": (
                                discovered.discovery_source if discovered else "heuristic"
                            ),
                        },
                    },
                )
            )
        return output

    async def search(self, query: str, limit: int, filters: dict) -> List[dict]:
        if not audio_cpp_enabled():
            self.status = {
                "available": False,
                "reason": "The audio.cpp integration is disabled by AUDIO_CPP_ENABLED.",
            }
            return []
        active = self.store.get_active_engine_version("audio_cpp")
        if not active:
            self.status = {
                "available": False,
                "reason": "Install and activate audio.cpp to load its version-pinned package catalog.",
            }
            return []
        packages = await asyncio.to_thread(self._manager_packages, active)
        items = self._normalize_packages(packages, active)
        needle = str(query or "").strip().lower()
        if needle:
            items = [
                item
                for item in items
                if needle
                in " ".join(
                    [
                        str(item.get("provider_item_id") or ""),
                        str(item.get("display_name") or ""),
                        str(item.get("description") or ""),
                        str(item.get("family") or ""),
                        " ".join(item.get("tasks") or []),
                    ]
                ).lower()
            ]
        return items[: max(limit, 1)]

