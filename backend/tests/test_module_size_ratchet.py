"""Tests for the module size ratchet gate (scripts/module-size-ratchet.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BUDGETS_PATH = ROOT / "scripts" / "module-size-budgets.json"


def _load_ratchet():
    spec = importlib.util.spec_from_file_location(
        "module_size_ratchet", ROOT / "scripts" / "module-size-ratchet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ratchet = _load_ratchet()


def test_live_tree_is_within_budgets():
    assert BUDGETS_PATH.exists(), "scripts/module-size-budgets.json is missing"
    assert ratchet.compare(ratchet.load_budgets(BUDGETS_PATH), ROOT) == []


def test_measure_counts_physical_lines(tmp_path):
    (tmp_path / "terminated.py").write_bytes(b"one\ntwo\n")
    (tmp_path / "unterminated.py").write_bytes(b"one\ntwo")
    (tmp_path / "empty.py").write_bytes(b"")

    assert ratchet.measure(tmp_path, "terminated.py") == 2
    assert ratchet.measure(tmp_path, "unterminated.py") == 2
    assert ratchet.measure(tmp_path, "empty.py") == 0


def test_growth_is_a_violation(tmp_path):
    (tmp_path / "grown.py").write_text("one\ntwo\nthree\n", encoding="utf-8")

    violations = ratchet.compare({"grown.py": 2}, tmp_path)

    assert violations == [
        "grown.py: grew from 2 to 3 lines; extract a module instead of growing it"
    ]


def test_stale_budget_is_a_violation(tmp_path):
    (tmp_path / "shrunk.py").write_text("one\ntwo\n", encoding="utf-8")

    violations = ratchet.compare({"shrunk.py": 3}, tmp_path)

    assert violations == [
        "shrunk.py: budget 3 is stale (now 2); run python scripts/module-size-ratchet.py --update"
    ]


def test_update_is_idempotent(tmp_path):
    (tmp_path / "module.py").write_text("one\ntwo\n", encoding="utf-8")
    (tmp_path / "keep.py").write_text("one\ntwo\nthree\n", encoding="utf-8")
    (tmp_path / "unlisted.py").write_text("one\n", encoding="utf-8")
    budgets_path = tmp_path / "budgets.json"
    budgets_path.write_text(
        json.dumps({"keep.py": 99, "module.py": 99}, indent=2) + "\n", encoding="utf-8"
    )

    ratchet.update_budgets(budgets_path, tmp_path)
    first = budgets_path.read_bytes()
    ratchet.update_budgets(budgets_path, tmp_path)

    assert budgets_path.read_bytes() == first
    assert first.decode("utf-8") == '{\n  "keep.py": 3,\n  "module.py": 2\n}\n'
    assert b"\r\n" not in first
    assert json.loads(first) == {"keep.py": 3, "module.py": 2}
    assert "unlisted.py" not in json.loads(first)
    assert ratchet.compare(ratchet.load_budgets(budgets_path), tmp_path) == []
