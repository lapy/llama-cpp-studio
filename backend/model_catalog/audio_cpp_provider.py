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
        packages.append(
            {
                "id": package_id,
                "display_name": value.get("display_name") or package_id,
                "target_directory": value.get("target_directory") or package_id,
                "description": value.get("description") or "",
                "required_files": value.get("required_files") or [],
                "source": source,
                "installable": installable,
                "install_kind": source.get("kind"),
                "usage_examples": [],
            }
        )
    return packages


def _manager_python(active: dict) -> str:
    helper_venv = str(active.get("helper_venv_path") or "")
    if helper_venv:
        candidate = os.path.join(
            helper_venv, "Scripts" if os.name == "nt" else "bin", "python"
        )
        if os.path.isfile(candidate):
            return candidate
    return sys.executable


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
        if process.returncode == 0:
            try:
                payload = json.loads(process.stdout)
                if isinstance(payload, list):
                    self.status = {
                        "available": True,
                        "source": "model_manager_json",
                        "reason": None,
                    }
                    return [item for item in payload if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass

        # The manager imports optional Torch tooling before argparse.  Catalog
        # discovery must still work before the lazy helper environment exists,
        # so parse only literal package declarations without executing source.
        try:
            with open(manager_path, "r", encoding="utf-8") as handle:
                packages = parse_model_manager_catalog(handle.read())
            self.status = {
                "available": bool(packages),
                "source": "model_manager_ast",
                "reason": (
                    None
                    if packages
                    else "No package declarations were found in model_manager.py."
                ),
                "manager_warning": process.stderr.strip()[-1000:],
            }
            return packages
        except Exception as exc:
            self.status = {
                "available": False,
                "reason": f"audio.cpp package catalog failed: {exc}",
            }
            return []

    @staticmethod
    def _infer_metadata(package_id: str) -> dict:
        lowered = package_id.lower()
        if "asr" in lowered or "stt" in lowered or "parakeet" in lowered:
            return {"family": package_id, "tasks": ["asr"], "modes": ["offline"]}
        if "vad" in lowered:
            return {"family": package_id, "tasks": ["vad"], "modes": ["offline"]}
        if "diar" in lowered or "sortformer" in lowered:
            return {"family": package_id, "tasks": ["diar"], "modes": ["offline"]}
        if any(token in lowered for token in ("tts", "kokoro", "chatterbox", "voxcpm")):
            return {"family": package_id, "tasks": ["tts"], "modes": ["offline"]}
        if any(token in lowered for token in ("stable_audio", "ace_step", "heartmula")):
            return {"family": package_id, "tasks": ["gen"], "modes": ["offline"]}
        if any(token in lowered for token in ("roformer", "demucs")):
            return {"family": package_id, "tasks": ["sep"], "modes": ["offline"]}
        if any(token in lowered for token in ("voice", "seed_vc", "vevo")):
            return {"family": package_id, "tasks": ["vc"], "modes": ["offline"]}
        return {"family": package_id, "tasks": [], "modes": []}

    def _normalize_packages(self, packages: Iterable[dict], active: dict) -> List[dict]:
        version_entry = get_version_entry(
            self.store, "audio_cpp", str(active.get("version") or "")
        )
        caps = (version_entry or {}).get("capabilities") or {}
        scanned_families = [
            str(f).strip() for f in (caps.get("families") or []) if str(f).strip()
        ]
        family_tasks = caps.get("family_tasks") if isinstance(caps.get("family_tasks"), dict) else {}
        package_list = list(packages)
        index = build_discovery_index(
            packages=package_list,
            families=scanned_families,
            family_tasks=family_tasks,
            source_path=str(active.get("source_path") or ""),
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
            source_kind = str(source.get("kind") or package.get("install_kind") or "unknown")
            method = {
                "huggingface_snapshot": "direct",
                "composite_snapshot": "composite",
                "composite": "converter",
                "utility": "converter",
            }.get(source_kind, "unavailable")
            features = [
                *tasks,
                "streaming" if "streaming" in modes else "",
                "prepared-bundle",
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
                            ],
                        }
                    },
                    install_variants=[
                        {
                            "id": package_id,
                            "label": package.get("display_name") or package_id,
                            "method": method,
                            "installable": bool(installable and verified),
                            "required_files": package.get("required_files") or [],
                            "source": source,
                            "external_inputs_required": source_kind == "utility",
                            "external_inputs_optional": (
                                source_kind == "composite"
                                and source.get("operation_kind") == "demucs_reference"
                            ),
                        }
                    ],
                    size_bytes=package.get("size_bytes"),
                    gated=bool(package.get("gated")),
                    release_status=(
                        "stable"
                        if discovered and discovered.discovery_source == "json"
                        else "experimental"
                    ),
                    unavailable_reason=unavailable,
                    metadata={
                        "target_directory": package.get("target_directory"),
                        "usage_examples": package.get("usage_examples") or [],
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

