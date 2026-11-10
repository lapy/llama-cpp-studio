#!/usr/bin/env python3
"""Engine-ID literal ratchet (config-driven engine refactoring, Phase 8
support — landed early per the plan's "introduce the ratchet lint early").

The committed inventory (docs/engine-parity-inventory.json) freezes today's
engine-ID literal sites outside backend/engines/. Migration code may delete
or shrink those literals freely; ADDING or GROWING a literal site outside the
engine home is a regression against the refactoring direction, so this gate
fails until the inventory is regenerated (making the growth a reviewed diff):

    python scripts/parity-inventory.py    # after intentional growth
    python scripts/engine-literals-ratchet.py

Exit code 1 lists each violation with its baseline count.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent

# Frozen AGGREGATE caps per area (R85). The per-file comparison below is
# defeated by regenerating the inventory; these caps are NOT — growth in an
# area requires editing this table (an explicit, reviewed diff), which is the
# plan's "architecture check against engine-ID dispatch outside definitions".
AREA_CAPS: Dict[str, int] = {
    "backend/": 11,
    "backend/audio/": 2,
    "backend/model_catalog/": 5,
    "backend/models/": 10,
    "backend/operations/": 3,
    # Converting every engine to declared profiles deleted the legacy venv
    # launch compilers and their binary/CUDA resolvers: 76 -> 38.
    # Capability declarations moved to backend/engines/: 28 -> 24.
    # sglang-sxm2 folded two SM70-fork checks into one declared set: 24 -> 23.
    "backend/proxy/": 23,
    # R105 tightened this 162 -> 152: the source-build routes now ask the spec
    # for the native family instead of six id tuples.
    # LMDeploy conversion tightened 152 -> 151: its compatibility module now
    # supplies only the URL prefix through the definition-derived router.
    "backend/routes/": 91,
    "backend/services/": 7,
    "frontend/e2e/": 22,
    "frontend/src/components/": 72,
    "frontend/src/composables/": 58,
    # R95 raised this 42 -> 44: the frontend descriptors now declare the
    # build-options catalog endpoint (one per catalog-bearing engine) instead of
    # the views branching on engine ids — reviewed diff.
    # 2026-11-06: 58 -> 74 for declared ui_kind/openStrategy/statusRefresh
    # metadata in the presentation overlay (data, not control flow); the
    # corresponding view literals fell 291 -> 265.
    # 2026-11-10: 74 -> 68. The panel family's ui_kind/openStrategy/
    # statusRefresh/panelFamily now live in one SGLANG_PANEL_BASE declaration
    # (12 -> 4 sites) when sglang-sxm2 joined; its identity fields add 2.
    "frontend/src/engines/": 68,
    "frontend/src/stores/": 323,
    "frontend/src/test-fixtures/": 43,
    # R90 raised this 6 -> 7: the audio-package predicate now lives once in
    # frontend/src/utils/audioModel.js (four duplicated checks across views and
    # components were removed); the one marker constant is the reviewed diff.
    "frontend/src/utils/": 7,
    # R88 raised this 369 -> 376: the status-regression fix replaced twenty dead
    # per-engine status reads with three labelled `statusForEngine('id')` calls
    # and the view test stub now keys the status map by engine id (reviewed diff).
    # R104 tightened this 376 -> 351: R90/R91/R101 moved the audio/text family
    # dispatch into descriptor data and the R88 stub fix removed dead reads, so
    # the measured total dropped; the cap now has NO slack.
    "frontend/src/views/": 265,
    # Every engine conversion tightened this as recipes/constants moved out of
    # the exporter into authoritative definitions; deleting the exporter after
    # the final conversion took it to 40.
    "scripts/": 40,
}


def area_for(path: str) -> str:
    """Coarse area for aggregate caps (directory-level, language-agnostic)."""
    parts = path.split("/")
    if path.startswith("backend/"):
        if len(parts) >= 3 and "." not in parts[1]:
            return f"backend/{parts[1]}/"
        return "backend/"
    if path.startswith("frontend/src/"):
        return (
            f"frontend/src/{parts[2]}/"
            if len(parts) >= 4 and "." not in parts[2]
            else "frontend/src/"
        )
    if path.startswith("frontend/"):
        return f"frontend/{parts[1]}/" if len(parts) >= 3 and "." not in parts[1] else "frontend/"
    if path.startswith("scripts/"):
        return "scripts/"
    return f"{parts[0]}/"


def area_totals(counts: dict) -> dict:
    totals: dict = {}
    for files in counts.values():
        for path, count in files.items():
            area = area_for(path)
            totals[area] = totals.get(area, 0) + count
    return totals


def compare_areas(totals: dict, caps: dict) -> list:
    violations = []
    for area, count in sorted(totals.items()):
        cap = caps.get(area)
        if cap is None:
            violations.append(f"new engine-ID literal area {area} ({count} occurrences)")
        elif count > cap:
            violations.append(f"engine-ID literal growth in {area}: cap {cap} -> {count}")
    return violations


def _load_parity_tool():
    spec = importlib.util.spec_from_file_location(
        "parity_inventory", ROOT / "scripts" / "parity-inventory.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_counts(path: Path) -> dict:
    inventory = json.loads(path.read_text(encoding="utf-8"))
    out: dict = {}
    for engine_id, buckets in inventory.get("engine_id_literals", {}).items():
        out[engine_id] = {e["file"]: e["count"] for e in buckets.get("outside_engines", [])}
    return out


def current_counts() -> dict:
    tool = _load_parity_tool()
    literals = tool.collect_engine_literals()
    out: dict = {}
    for engine_id, buckets in literals.items():
        out[engine_id] = {e["file"]: e["count"] for e in buckets.get("outside_engines", [])}
    return out


def compare(current: dict, baseline: dict) -> list:
    """Violations = new literal files or grown counts vs the frozen baseline.

    Shrinking is always allowed: retiring engine-ID control flow is the goal.
    """
    violations: list = []
    for engine_id, files in sorted(current.items()):
        base = baseline.get(engine_id, {})
        for file, count in sorted(files.items()):
            previous = base.get(file)
            if previous is None:
                violations.append(
                    f"{engine_id}: new engine-ID literal site {file} ({count} occurrences)"
                )
            elif count > previous:
                violations.append(f"{engine_id}: literal growth in {file}: {previous} -> {count}")
    return violations


def main() -> int:
    baseline_path = ROOT / "docs" / "engine-parity-inventory.json"
    if not baseline_path.exists():
        print("docs/engine-parity-inventory.json missing; run scripts/parity-inventory.py")
        return 1
    current = current_counts()
    violations = compare(current, baseline_counts(baseline_path))
    # R85: the aggregate caps are independent of the inventory — regenerating
    # the inventory cannot raise them.
    violations += compare_areas(area_totals(current), AREA_CAPS)
    if violations:
        print("engine-ID literals regressed against the frozen architecture caps:")
        for violation in violations:
            print(f"  - {violation}")
        print(
            "retire the literals instead; per-file growth may regenerate the inventory, "
            "aggregate growth requires editing AREA_CAPS (reviewed diff)"
        )
        return 1
    print("engine-ID literal ratchet: no growth beyond the committed inventory")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
