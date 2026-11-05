"""R102 pin: the legacy-route inventory is current.

Removing the legacy per-engine compatibility routers is an API-surface cutover;
the generated inventory is the blast-radius record, so it must never drift from
the code (same contract as the definitions / build-defaults checks).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_legacy_route_inventory_is_current():
    result = subprocess.run(
        [sys.executable, "scripts/legacy-route-inventory.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_inventory_lists_every_legacy_router_module():
    text = (ROOT / "docs/legacy-route-inventory.md").read_text(encoding="utf-8")
    for module in (
        "lmdeploy_versions.py",
        "onecat_vllm_versions.py",
        "sglang_versions.py",
        "unsloth_llama_versions.py",
        "vllm_versions.py",
    ):
        assert module in text, module


def test_no_legacy_router_is_the_only_surface_for_its_engine():
    """R103: every legacy mount's engine is reachable through a generic route."""
    text = (ROOT / "docs/legacy-route-inventory.md").read_text(encoding="utf-8")
    rows = [line for line in text.splitlines() if line.startswith("| `")]
    assert rows, "inventory has no data rows"
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        routing = cells[-2]
        assert routing in {"venv", "native"}, f"{cells[0]} routes only through its legacy prefix"

    # The generic router, the shared factory and llama/audio own surfaces are not
    # compatibility delegates and must not be listed.
    for generic in ("engine_versions.py", "llama_versions.py", "audio_cpp_versions.py"):
        assert f"`{generic}`" not in text, generic
