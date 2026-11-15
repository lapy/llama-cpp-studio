#!/usr/bin/env python3
"""Per-module backend coverage floors.

CI enforces one aggregate line-coverage floor in ``.coveragerc``. That single
number hides the riskiest modules: a large, barely-covered installer can be
propped up by well-covered stores. This gate pins reviewed minimums for the
high-risk modules in ``scripts/module-coverage-floors.json``.

The floors are measured from the real backend suite, with a small safety margin
below the observed percentage. Coverage.py writes a JSON report at
``test-results/backend-coverage.json``; each floor is matched to its report entry
by path suffix so absolute paths, Windows separators, and CI checkouts all work.

Usage::

    python scripts/check_module_coverage.py            # check (CI and npm)
    python scripts/check_module_coverage.py --update   # lower/raise to measured values

``--update`` only rewrites keys already present in the floor file. Adding a
module to the gate stays a deliberate review decision, mirroring the module size
ratchet. The aggregate ``fail_under`` gate in ``.coveragerc`` is untouched.
"""

from __future__ import annotations

import argparse
import json
import math
from decimal import Decimal
from pathlib import Path
from typing import Mapping

DEFAULT_REPORT = Path("test-results/backend-coverage.json")
DEFAULT_FLOORS = Path("scripts/module-coverage-floors.json")


def _normalized(path: str) -> str:
    """Normalize a report or floor path for suffix matching."""
    normalized = str(path).replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def load_floors(path: Path) -> dict[str, float]:
    """Load the ``{repo-relative module path: minimum lines percent}`` mapping."""
    floors = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(floors, dict):
        raise ValueError(f"{path} must contain a JSON object of module paths to floors")
    result: dict[str, float] = {}
    for module, floor in floors.items():
        if not isinstance(module, str) or not module:
            raise ValueError(f"{path} contains an invalid module path: {module!r}")
        if isinstance(floor, bool) or not isinstance(floor, (int, float)):
            raise ValueError(f"{path} floor for {module!r} must be a number")
        result[module] = float(floor)
    return result


def _line_percent(report_file: str, entry: object) -> float | None:
    """Return the line-coverage percentage from one coverage.py JSON entry.

    coverage.py 7.x reports the statement percentage as
    ``summary.percent_statements_covered`` (and the branch-combined percentage as
    ``summary.percent_covered``); some consumers nest it as
    ``summary.lines.percent``. Accept all three so the gate works with the real
    report and with simplified fixtures.
    """
    summary = entry.get("summary") if isinstance(entry, dict) else None
    if not isinstance(summary, dict):
        raise ValueError(f"coverage entry {report_file!r} has no summary object")

    lines = summary.get("lines")
    if isinstance(lines, dict) and lines.get("percent") is not None:
        return float(lines["percent"])
    for key in ("percent_statements_covered", "percent_covered"):
        if summary.get(key) is not None:
            return float(summary[key])
    covered = summary.get("covered_lines")
    statements = summary.get("num_statements")
    if isinstance(covered, (int, float)) and isinstance(statements, (int, float)) and statements:
        return 100.0 * covered / statements
    return None


def extract_percentages(report_path: Path) -> dict[str, float]:
    """Extract line-coverage percentages from a coverage.py JSON report."""
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    files = payload.get("files") if isinstance(payload, dict) else None
    if not isinstance(files, dict):
        raise ValueError(f"{report_path} is not a coverage.py JSON report (missing 'files')")
    percentages: dict[str, float] = {}
    for report_file, entry in files.items():
        percent = _line_percent(str(report_file), entry)
        if percent is None:
            # No measurable statements: treat the module as absent rather than as
            # a 100% pass, otherwise a parser change could silently pass the gate.
            continue
        percentages[str(report_file)] = percent
    return percentages


def match_percentage(module: str, percentages: Mapping[str, float]) -> float | None:
    """Find the report percentage for *module*, matching absolute paths by suffix.

    The shortest matching report path wins, so an exact relative entry beats an
    absolute path to the same file. Matching ignores separator style and case,
    which is harmless for repository module paths and necessary for Windows
    drive-letter casing.
    """
    target = _normalized(module).casefold()
    candidates = [
        report_file
        for report_file in percentages
        if _normalized(report_file).casefold() == target
        or _normalized(report_file).casefold().endswith("/" + target)
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda report_file: (len(_normalized(report_file)), report_file))
    return percentages[best]


def missing_modules(floors: Mapping[str, float], percentages: Mapping[str, float]) -> list[str]:
    """Floor keys that have no matching entry in the coverage report."""
    return [module for module in sorted(floors) if match_percentage(module, percentages) is None]


def compare(floors: Mapping[str, float], percentages: Mapping[str, float]) -> list[str]:
    """Return sorted, human-readable violations for missing modules and shortfalls."""
    violations: list[str] = []
    for module in sorted(floors):
        measured = match_percentage(module, percentages)
        if measured is None:
            violations.append(f"{module}: missing from coverage report")
        elif measured < floors[module]:
            violations.append(f"{module}: {measured:.1f}% < floor {floors[module]:.1f}%")
    return violations


def _one_decimal_floor(percent: float) -> float:
    """Round a measured percentage down to one decimal."""
    truncated = Decimal(str(percent)) * 10
    return float(math.floor(truncated) / 10)


def write_floors(path: Path, floors: Mapping[str, float]) -> None:
    """Write the mapping as sorted, 2-space JSON with a trailing newline."""
    ordered = {module: floors[module] for module in sorted(floors)}
    Path(path).write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8", newline="\n")


def update_floors(
    floors_path: Path, floors: Mapping[str, float], percentages: Mapping[str, float]
) -> dict[str, float]:
    """Recompute existing floors from measured percentages and rewrite *floors_path*.

    Keys already present are refreshed; report entries without a floor are never
    added. A floor key missing from the report is an error: silently keeping the
    old number would make the gate look healthy when it is not being measured.
    """
    missing = missing_modules(floors, percentages)
    if missing:
        raise ValueError(
            "cannot recompute floors for modules missing from the coverage report: "
            + ", ".join(missing)
        )
    updated = {
        module: _one_decimal_floor(match_percentage(module, percentages) or 0.0)
        for module in sorted(floors)
    }
    write_floors(floors_path, updated)
    return updated


def _format_violations(violations: list[str], total: int) -> str:
    lines = [f"module coverage floors failed ({len(violations)} of {total} modules):"]
    lines.extend(f"  - {violation}" for violation in violations)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Enforce per-module backend coverage floors from a coverage.py JSON report."
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
        help=f"coverage.py JSON report (default: {DEFAULT_REPORT})",
    )
    parser.add_argument(
        "--floors",
        default=str(DEFAULT_FLOORS),
        help=f"floor mapping to enforce (default: {DEFAULT_FLOORS})",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite existing floors to the report's current percentages",
    )
    args = parser.parse_args(argv)

    floors_path = Path(args.floors)
    report_path = Path(args.report)

    if not floors_path.exists():
        print(f"module coverage floor file {floors_path} is missing")
        return 1
    if not report_path.exists():
        print(
            f"coverage report {report_path} is missing; generate it with "
            f"pytest --cov-report=json:{report_path}"
        )
        return 1

    try:
        floors = load_floors(floors_path)
        percentages = extract_percentages(report_path)
    except (OSError, ValueError) as error:
        print(f"cannot read coverage inputs: {error}")
        return 1

    if args.update:
        try:
            updated = update_floors(floors_path, floors, percentages)
        except ValueError as error:
            print(f"{error}; floors were not rewritten")
            return 1
        changes = [
            f"  {module}: {floors[module]:.1f} -> {updated[module]:.1f}"
            for module in sorted(updated)
            if updated[module] != floors[module]
        ]
        if changes:
            print("updated module coverage floors:")
            print("\n".join(changes))
        print(f"module coverage floors updated ({len(updated)} modules)")
        return 0

    violations = compare(floors, percentages)
    if violations:
        print(_format_violations(violations, len(floors)))
        print(
            "raise the module's coverage; lower a floor only as a reviewed, "
            "deliberate change via --update"
        )
        return 1

    print(f"module coverage floors: {len(floors)} modules at or above their floors ({report_path})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
