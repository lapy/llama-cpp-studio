"""Export the llama-family build-option defaults the frontend falls back to.

The EnginesView build-settings panel needs a default build config before
``GET /build-options`` answers (and when it fails). That fallback used to be a
hand-copied literal in the Vue SFC, explicitly "kept in sync with backend
defaults" — a drift risk. The source of truth is the backend's
``llama_cpp.build_options`` catalog; this script writes it to a JSON module the
frontend imports, and ``--check`` fails when the checked-in file is stale.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "frontend" / "src" / "engines" / "build-defaults.generated.json"


def _ordered_build_types() -> list:
    """Catalog order with the default first (the panel's historic UI order)."""
    from backend.engines.llama_cpp import build_options as llama_options

    default = llama_options.BUILD_TYPE_DEFAULT
    values = list(llama_options.BUILD_TYPE_VALUES)
    return [default] + [value for value in values if value != default]


def _payload() -> dict:
    from backend.engines.llama_cpp import build_options as llama_options

    return {
        "defaults": llama_options.default_build_settings(),
        "build_types": _ordered_build_types(),
    }


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
