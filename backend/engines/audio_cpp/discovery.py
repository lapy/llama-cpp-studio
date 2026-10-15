"""Package identity resolution for audio.cpp catalogs.

Families, tasks, and modes come from engine-owned data only: the package
declaration itself (``model_manager_v2 list --json``, bundled/builtin
descriptors), the engine's ``model_specs`` contracts, and the version scan's
``--list-loaders --json`` capabilities. A package without a declared family
stays unclassified — no fuzzy name or layout scoring exists anywhere in
Studio (audio.cpp#852 supersedes those heuristics).

Request routing knowledge lives in
:mod:`backend.engines.audio_cpp.integration.contract_registry`; this module
never encodes endpoints.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from backend.logging_config import get_logger

logger = get_logger(__name__)


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
    discovery_source: str = "declared"


@dataclass
class PackageDiscoveryIndex:
    families: Set[str] = field(default_factory=set)
    family_tasks: Dict[str, List[str]] = field(default_factory=dict)
    family_modes: Dict[str, List[str]] = field(default_factory=dict)
    packages: Dict[str, DiscoveredPackage] = field(default_factory=dict)
    spec_families: Set[str] = field(default_factory=set)

    def get(self, package_id: str) -> Optional[DiscoveredPackage]:
        return self.packages.get(str(package_id or "").strip())


def build_discovery_index(
    *,
    packages: Sequence[dict],
    families: Sequence[str],
    family_tasks: Optional[Dict[str, List[str]]] = None,
    family_modes: Optional[Dict[str, List[str]]] = None,
    source_path: Optional[str] = None,
) -> PackageDiscoveryIndex:
    """Resolve each catalog package against engine-declared family data.

    Family resolution order: the package's own ``family`` field, then the
    ``packages`` list embedded in ``model_specs/<family>.json``. Tasks and
    modes fall back from the package to the family's loader/spec data.
    """
    family_set = {str(f).strip().lower() for f in families if str(f).strip()}
    from backend.engines.audio_cpp.contracts import (
        load_family_contracts,
        load_model_spec_path_leaves,
    )

    contracts = load_family_contracts(source_path) if source_path else {}
    package_families = {
        str(package.get("id")): family
        for family, contract in contracts.items()
        for package in contract.get("packages") or []
        if isinstance(package, dict) and package.get("id")
    }
    resolved_family_tasks = {
        str(k).strip().lower(): list(v)
        for k, v in (family_tasks or {}).items()
        if str(k).strip()
    }
    for family, contract in contracts.items():
        if not resolved_family_tasks.get(family):
            resolved_family_tasks[family] = list(contract.get("tasks") or [])
    index = PackageDiscoveryIndex(
        families=family_set,
        family_tasks=resolved_family_tasks,
        family_modes={
            **{family: list(contract.get("modes") or []) for family, contract in contracts.items()},
            **{k.lower(): list(v) for k, v in (family_modes or {}).items()},
        },
        spec_families=set(load_model_spec_path_leaves(source_path)) if source_path else set(),
    )

    known_ids = {
        str(package.get("id") or "").strip()
        for package in packages
        if isinstance(package, dict) and str(package.get("id") or "").strip()
    }

    for raw in packages:
        package = raw if isinstance(raw, dict) else {}
        package_id = str(package.get("id") or "").strip()
        if not package_id:
            continue

        pkg_family = str(
            package.get("family") or package_families.get(package_id) or ""
        ).strip().lower() or None
        family = pkg_family
        reason = "package_json" if family else "no_declared_family"

        standalone = package.get("standalone")
        parent = package.get("parent_package_id")
        if not isinstance(standalone, bool):
            description = str(package.get("description") or "").strip().lower()
            # Engine declarations mark dependency-only entries explicitly.
            standalone = not (
                description.startswith("subcomponent only.")
                or description.startswith("utility only.")
            )
        if not parent and not standalone:
            parent = _declared_parent_package_id(
                str(package.get("description") or ""), known_ids
            )

        pkg_tasks = package.get("tasks") if isinstance(package.get("tasks"), list) else None
        pkg_modes = package.get("modes") if isinstance(package.get("modes"), list) else None
        tasks = (
            [str(t).strip() for t in pkg_tasks if str(t).strip()]
            if pkg_tasks is not None
            else list(index.family_tasks.get(family or "", []))
        )
        modes = (
            [str(m).strip() for m in pkg_modes if str(m).strip()]
            if pkg_modes is not None
            else list(index.family_modes.get(family or "", []))
        )

        index.packages[package_id] = DiscoveredPackage(
            package_id=package_id,
            family=family,
            standalone=bool(standalone),
            tasks=tasks,
            modes=modes,
            parent_package_id=str(parent).strip() if parent else None,
            match_score=1000.0 if family else 0.0,
            match_reason=reason,
            discovery_source="json" if pkg_family else reason,
        )
    return index


def _declared_parent_package_id(description: str, known_ids: Set[str]) -> Optional[str]:
    """Parent runtime package named inside an engine subcomponent description.

    Engine declarations phrase dependency entries as e.g. ``Subcomponent only.
    Use moss_tts_nano_100m.`` — the parent id is matched only against exact ids
    already present in the same catalog, so this resolves declared text rather
    than scoring names.
    """
    if not description or not known_ids:
        return None
    for candidate in sorted(known_ids, key=len, reverse=True):
        if candidate in description:
            return candidate
    return None


__all__ = [
    "DiscoveredPackage",
    "PackageDiscoveryIndex",
    "build_discovery_index",
]
