"""R80 pins: the frontend build-default fallback is generated from the backend.

`frontend/src/engines/build-defaults.generated.json` is written by
`scripts/export-build-defaults.py` from the llama build-option catalog. This pin
fails when the checked-in file drifts from the backend defaults, so the Vue panel
can never silently show a stale fallback again.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.engines.llama_cpp import build_options as llama_options

ROOT = Path(__file__).resolve().parents[2]
GENERATED = ROOT / "frontend" / "src" / "engines" / "build-defaults.generated.json"


def _checked_in() -> dict:
    return json.loads(GENERATED.read_text(encoding="utf-8"))


def test_generated_frontend_defaults_match_the_backend_catalog():
    data = _checked_in()
    assert data["defaults"] == llama_options.default_build_settings()


def test_generated_build_types_are_default_first():
    data = _checked_in()
    assert data["build_types"] == ["Release", "Debug", "RelWithDebInfo", "MinSizeRel"]
    assert data["build_types"][0] == llama_options.BUILD_TYPE_DEFAULT
    assert set(data["build_types"]) == set(llama_options.BUILD_TYPE_VALUES)


def test_the_export_script_is_current():
    """The script's own --check agrees with the checked-in file."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "export_build_defaults", ROOT / "scripts" / "export-build-defaults.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._render() == GENERATED.read_text(encoding="utf-8")
