#!/usr/bin/env python3
"""Module size ratchet.

The repository's largest modules have checked-in physical-line budgets in
``scripts/module-size-budgets.json``. This gate mirrors the engine-ID literal
ratchet (``scripts/engine-literals-ratchet.py``): it freezes a reviewed baseline
so growth becomes an explicit diff instead of silent drift.

Budgets are exact, with no slack, so extraction is reflected in the same change:

    python scripts/module-size-ratchet.py            # check (CI and `npm run lint`)
    python scripts/module-size-ratchet.py --update   # lower budgets after extraction

Growing a budgeted module fails the check. Extract a cohesive unit instead and
then lower that module's budget with ``--update``. Only budgeted paths are read,
so the check is fast and does not scan the repository; new files are not
discovered here, they are added to the budget file deliberately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGETS_PATH = ROOT / "scripts" / "module-size-budgets.json"


def load_budgets(path: Path) -> dict[str, int]:
    """Load the ``{repo-relative path: max physical lines}`` budget mapping."""
    budgets: dict[str, int] = json.loads(Path(path).read_text(encoding="utf-8"))
    return budgets


def measure(root: Path, rel_path: str) -> int:
    """Physical line count: newline bytes plus a final unterminated line."""
    data = (Path(root) / rel_path).read_bytes()
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def compare(budgets: dict[str, int], root: Path) -> list[str]:
    """Violations for budgeted modules that outgrew their budget or are stale."""
    violations: list[str] = []
    for rel_path in sorted(budgets):
        budget = budgets[rel_path]
        try:
            measured = measure(root, rel_path)
        except FileNotFoundError:
            violations.append(
                f"{rel_path}: budgeted module is missing; remove its budget or restore the file"
            )
            continue
        if measured > budget:
            violations.append(
                f"{rel_path}: grew from {budget} to {measured} lines; "
                "extract a module instead of growing it"
            )
        elif measured < budget:
            violations.append(
                f"{rel_path}: budget {budget} is stale (now {measured}); "
                "run python scripts/module-size-ratchet.py --update"
            )
    return violations


def write_budgets(path: Path, budgets: dict[str, int]) -> None:
    """Write the mapping as sorted, 2-space JSON with a trailing newline."""
    ordered = dict(sorted(budgets.items()))
    Path(path).write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8", newline="\n")


def update_budgets(path: Path, root: Path) -> dict[str, int]:
    """Measure every existing budgeted module in *root* and rewrite *path*.

    Only keys already present are measured and rewritten; adding or removing a
    budget stays a deliberate review decision.
    """
    budgets = load_budgets(path)
    updated = {rel_path: measure(root, rel_path) for rel_path in sorted(budgets)}
    write_budgets(path, updated)
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Enforce checked-in module size budgets.")
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite each existing budget to its module's current physical line count",
    )
    args = parser.parse_args(argv)

    if not BUDGETS_PATH.exists():
        print(f"{BUDGETS_PATH.relative_to(ROOT)} is missing")
        return 1

    if args.update:
        budgets = load_budgets(BUDGETS_PATH)
        try:
            updated = update_budgets(BUDGETS_PATH, ROOT)
        except FileNotFoundError as error:
            print(f"cannot measure missing budgeted module: {error.filename}")
            return 1
        changes = [
            f"  {rel_path}: {budgets[rel_path]} -> {updated[rel_path]}"
            for rel_path in sorted(updated)
            if updated[rel_path] != budgets[rel_path]
        ]
        if changes:
            print("updated module size budgets:")
            print("\n".join(changes))
        print(f"module size budgets updated ({len(updated)} modules)")
        return 0

    budgets = load_budgets(BUDGETS_PATH)
    violations = compare(budgets, ROOT)
    if violations:
        print(f"module size ratchet failed ({len(violations)} of {len(budgets)} modules):")
        for violation in violations:
            print(f"  - {violation}")
        print(
            "extract a module instead of growing a budgeted one; after extraction run "
            "python scripts/module-size-ratchet.py --update"
        )
        return 1
    print(f"module size ratchet: {len(budgets)} budgeted modules within their budgets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
