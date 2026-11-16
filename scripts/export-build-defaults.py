"""Export the engine build-option defaults the frontend falls back to.

The EnginesView build-settings panels need a default build config before
``GET /build-options`` answers (and when it fails). Those fallbacks used to be
hand-copied literals in the Vue SFC, explicitly "kept in sync with backend
defaults" — a drift risk. The source of truth is the backend build-option
catalogs, aggregated by ``backend.engines.build_defaults``; this script writes
them to a JSON module the frontend imports, and ``--check`` fails when the
checked-in file is stale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TARGET = ROOT / "frontend" / "src" / "engines" / "build-defaults.generated.json"

# `python scripts/export-build-defaults.py` must import the app package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _payload() -> dict:
    from backend.engines.build_defaults import export_payload

    return export_payload()


def _render() -> str:
    return json.dumps(_payload(), indent=2, sort_keys=True) + "\n"


def main(argv: list[str]) -> int:
    check = "--check" in argv
    rendered = _render()
    if check:
        current = TARGET.read_text(encoding="utf-8") if TARGET.exists() else ""
        if current != rendered:
            print(
                "frontend build defaults are stale; run scripts/export-build-defaults.py",
                file=sys.stderr,
            )
            return 1
        print("frontend build defaults current")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(rendered, encoding="utf-8")
    print(f"wrote {TARGET.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
