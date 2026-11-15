"""Tests for the per-module coverage floor gate (scripts/check_module_coverage.py)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FLOORS_PATH = ROOT / "scripts" / "module-coverage-floors.json"
REPORT_PATH = ROOT / "test-results" / "backend-coverage.json"

REQUIRED_MODULES = {
    "backend/build_progress.py",
    "backend/cuda_installer.py",
    "backend/data_store.py",
    "backend/operations/cancellable.py",
    "backend/operations/supervisor.py",
    "backend/proxy/studio/manager.py",
    "backend/routes/llama_versions.py",
    "backend/routes/models.py",
    "backend/store_io.py",
}


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_module_coverage", ROOT / "scripts" / "check_module_coverage.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _write_report(path: Path, entries: dict[str, float]) -> Path:
    files = {
        report_file: {"summary": {"lines": {"percent": percent}}}
        for report_file, percent in entries.items()
    }
    path.write_text(json.dumps({"meta": {}, "files": files}, indent=2), encoding="utf-8")
    return path


def _write_floors(path: Path, floors: dict[str, float]) -> Path:
    path.write_text(json.dumps(floors, indent=2) + "\n", encoding="utf-8")
    return path


def test_compare_passes_at_or_above_the_floor():
    floors = {"backend/cuda_installer.py": 22.0, "backend/data_store.py": 83.4}
    percentages = {
        "C:\\Users\\ci\\repo\\backend\\cuda_installer.py": 22.0,
        "/home/ci/repo/backend/data_store.py": 83.41,
    }

    assert checker.compare(floors, percentages) == []


def test_compare_reports_shortfalls_with_one_decimal():
    violations = checker.compare(
        {"backend/cuda_installer.py": 22.0},
        {"C:\\Users\\ci\\repo\\backend\\cuda_installer.py": 21.74},
    )

    assert violations == ["backend/cuda_installer.py: 21.7% < floor 22.0%"]


def test_compare_reports_modules_missing_from_the_report():
    violations = checker.compare(
        {"backend/data_store.py": 83.0, "backend/store_io.py": 70.0},
        {"backend/data_store.py": 83.4},
    )

    assert violations == ["backend/store_io.py: missing from coverage report"]


def test_extract_percentages_matches_windows_absolute_paths(tmp_path):
    report = _write_report(
        tmp_path / "coverage.json",
        {"C:\\Users\\ci\\repo\\backend\\proxy\\studio\\manager.py": 74.25},
    )

    percentages = checker.extract_percentages(report)

    assert percentages == {"C:\\Users\\ci\\repo\\backend\\proxy\\studio\\manager.py": 74.25}
    assert checker.match_percentage("backend/proxy/studio/manager.py", percentages) == 74.25
    assert checker.match_percentage("backend/proxy/studio/manager.pyc", percentages) is None


def test_extract_percentages_reads_coverage_py_summary(tmp_path):
    report = tmp_path / "coverage.json"
    report.write_text(
        json.dumps(
            {
                "meta": {"format": 3, "version": "7.16.2"},
                "files": {
                    "backend\\cuda_installer.py": {
                        "summary": {
                            "covered_lines": 322,
                            "num_statements": 1308,
                            "percent_covered": 23.22,
                            "percent_statements_covered": 24.62,
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    percentages = checker.extract_percentages(report)

    # Line coverage (statements) wins over the branch-combined percentage.
    assert percentages == {"backend\\cuda_installer.py": 24.62}


def test_main_fails_with_a_clear_module_list(tmp_path, capsys):
    floors = _write_floors(
        tmp_path / "floors.json",
        {"backend/cuda_installer.py": 22.0, "backend/store_io.py": 70.0},
    )
    report = _write_report(
        tmp_path / "coverage.json",
        {
            "C:\\Users\\ci\\repo\\backend\\cuda_installer.py": 21.74,
            "C:\\Users\\ci\\repo\\backend\\store_io.py": 70.2,
        },
    )

    code = checker.main(["--report", str(report), "--floors", str(floors)])

    assert code == 1
    output = capsys.readouterr().out
    assert "module coverage floors failed (1 of 2 modules)" in output
    assert "backend/cuda_installer.py: 21.7% < floor 22.0%" in output


def test_main_fails_when_the_report_is_missing(tmp_path, capsys):
    floors = _write_floors(tmp_path / "floors.json", {"backend/store_io.py": 70.0})

    code = checker.main(["--report", str(tmp_path / "absent.json"), "--floors", str(floors)])

    assert code == 1
    output = capsys.readouterr().out
    assert str(tmp_path / "absent.json") in output
    assert "is missing" in output
    assert "pytest --cov-report=json:" in output


def test_update_rewrites_existing_keys_only_and_is_idempotent(tmp_path):
    floors = _write_floors(
        tmp_path / "floors.json",
        {"backend/store_io.py": 99.0, "backend/cuda_installer.py": 50.0},
    )
    report = _write_report(
        tmp_path / "coverage.json",
        {
            "backend/cuda_installer.py": 61.29,
            "C:\\Users\\ci\\repo\\backend\\store_io.py": 80.0,
            "backend/data_store.py": 83.4,
        },
    )

    first_code = checker.main(["--report", str(report), "--floors", str(floors), "--update"])
    first = floors.read_bytes()
    second_code = checker.main(["--report", str(report), "--floors", str(floors), "--update"])

    assert first_code == 0
    assert second_code == 0
    assert floors.read_bytes() == first
    assert json.loads(first) == {
        "backend/cuda_installer.py": 61.2,
        "backend/store_io.py": 80.0,
    }
    assert list(json.loads(first)) == ["backend/cuda_installer.py", "backend/store_io.py"]
    assert b"\r\n" not in first
    assert first.endswith(b"\n")


def test_update_refuses_to_drop_or_invent_keys(tmp_path):
    floors = _write_floors(
        tmp_path / "floors.json",
        {"backend/store_io.py": 99.0, "backend/cuda_installer.py": 50.0},
    )
    original = floors.read_bytes()
    report = _write_report(
        tmp_path / "coverage.json",
        {"backend/cuda_installer.py": 61.29, "backend/data_store.py": 83.4},
    )

    code = checker.main(["--report", str(report), "--floors", str(floors), "--update"])

    assert code == 1
    assert floors.read_bytes() == original


def test_committed_floors_cover_the_high_risk_modules():
    floors = checker.load_floors(FLOORS_PATH)

    assert REQUIRED_MODULES <= set(floors)
    assert list(floors) == sorted(floors)
    raw = json.loads(FLOORS_PATH.read_text(encoding="utf-8"))
    assert list(raw) == sorted(raw)
    for module, floor in floors.items():
        assert 0.0 <= floor <= 100.0, module
        assert round(floor, 1) == floor, f"{module} floor must have one decimal"


@pytest.mark.skipif(
    not REPORT_PATH.exists(),
    reason="no local coverage report; CI checks the freshly generated report separately",
)
def test_committed_floors_hold_against_the_current_report():
    floors = checker.load_floors(FLOORS_PATH)
    percentages = checker.extract_percentages(REPORT_PATH)

    assert checker.compare(floors, percentages) == []
