"""Tests for the engine-ID literal ratchet gate (refactoring Phase 8 seed)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_ratchet():
    spec = importlib.util.spec_from_file_location(
        "engine_literals_ratchet", ROOT / "scripts" / "engine-literals-ratchet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ratchet = _load_ratchet()


def test_live_tree_passes_the_committed_ratchet():
    baseline_path = ROOT / "docs" / "engine-parity-inventory.json"
    assert baseline_path.exists(), "run scripts/parity-inventory.py"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))["engine_id_literals"]
    current = {
        engine_id: {e["file"]: e["count"] for e in buckets.get("outside_engines", [])}
        for engine_id, buckets in baseline.items()
    }
    assert ratchet.compare(ratchet.current_counts(), current) == []


def test_new_literal_file_is_a_violation():
    violations = ratchet.compare(
        {"sglang": {"backend/routes/new_panel.py": 3}},
        {"sglang": {"backend/routes/engine_versions.py": 29}},
    )
    assert len(violations) == 1
    assert "new engine-ID literal site backend/routes/new_panel.py" in violations[0]


def test_literal_growth_is_a_violation():
    violations = ratchet.compare(
        {"vllm": {"frontend/src/stores/engines.js": 9}},
        {"vllm": {"frontend/src/stores/engines.js": 8}},
    )
    assert violations == ["vllm: literal growth in frontend/src/stores/engines.js: 8 -> 9"]


def test_shrinking_and_deleted_literals_pass():
    assert (
        ratchet.compare(
            {"vllm": {"frontend/src/stores/engines.js": 2}},
            {
                "vllm": {
                    "frontend/src/stores/engines.js": 8,
                    "frontend/src/views/EnginesView.vue": 4,
                }
            },
        )
        == []
    )


# --- R85: aggregate architecture caps (independent of the inventory) -----------


def test_area_totals_group_by_area():
    totals = ratchet.area_totals(
        {
            "sglang": {
                "backend/proxy/commands.py": 2,
                "backend/proxy/launch_spec.py": 3,
                "frontend/src/views/EnginesView.vue": 4,
                "scripts/parity-inventory.py": 1,
                "backend/main.py": 5,
            }
        }
    )
    assert totals == {
        "backend/proxy/": 5,
        "frontend/src/views/": 4,
        "scripts/": 1,
        "backend/": 5,
    }


def test_live_tree_is_within_the_area_caps():
    totals = ratchet.area_totals(ratchet.current_counts())
    assert ratchet.compare_areas(totals, ratchet.AREA_CAPS) == []


def test_area_growth_is_a_violation_even_after_inventory_regeneration():
    violations = ratchet.compare_areas(
        {"backend/proxy/": 77, "backend/new_area/": 1, "frontend/src/views/": 10},
        {"backend/proxy/": 76, "frontend/src/views/": 10},
    )
    assert violations == [
        "new engine-ID literal area backend/new_area/ (1 occurrences)",
        "engine-ID literal growth in backend/proxy/: cap 76 -> 77",
    ]
