"""Line-by-line audits of audio.cpp --help fixtures against the parser."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from backend.cli_help_parsers import parse_audio_cpp_help_to_sections
from backend.tests.help_parser_audit import (
    extract_audio_cpp_help_entries,
    verify_audio_cpp_help_params,
)

_HERE = Path(__file__).resolve().parent
_FIXTURES = _HERE / "fixtures"
_REPO = _HERE.parents[1]
_LIVE_BIN = (
    _REPO
    / "data"
    / "audio-cpp"
    / "src"
    / "build"
    / "linux-cpu-release"
    / "bin"
)
_LIVE_SPECS = _REPO / "data" / "audio-cpp" / "src" / "model_specs"

_HELP_FIXTURES = (
    ("audio_cpp_server_help_live.txt", "server"),
    ("audio_cpp_server_help_sample.txt", "server"),
    ("audio_cpp_cli_help_live.txt", "cli"),
    ("audio_cpp_nemotron_help_live.txt", "cli"),
    ("audio_cpp_qwen3_asr_help_live.txt", "cli"),
    ("audio_cpp_model_help_sample.txt", "cli"),
)


def _flat_params(text: str, source: str) -> list[dict]:
    sections = parse_audio_cpp_help_to_sections(text, source=source)
    rows: list[dict] = []
    for section in sections:
        sid = section.get("id")
        slab = section.get("label")
        for param in section.get("params") or []:
            row = dict(param)
            row["section_id"] = sid
            row["section_label"] = slab
            rows.append(row)
    return rows


@pytest.mark.parametrize(("filename", "source"), _HELP_FIXTURES)
def test_audio_cpp_help_fixture_line_audit(filename: str, source: str):
    text = (_FIXTURES / filename).read_text(encoding="utf-8")
    entries = extract_audio_cpp_help_entries(text, source=source)
    parsed = _flat_params(text, source)
    issues = verify_audio_cpp_help_params(entries, parsed)
    assert entries, f"{filename}: extractor found no help entries"
    assert parsed, f"{filename}: parser found no params"
    assert not issues, f"{filename} help audit failed:\n" + "\n".join(issues)


@pytest.mark.parametrize(("filename", "source"), _HELP_FIXTURES)
def test_audio_cpp_help_lines_are_classified(filename: str, source: str):
    """Every flag/bare-option line must produce an extractor entry."""
    text = (_FIXTURES / filename).read_text(encoding="utf-8")
    entries = extract_audio_cpp_help_entries(text, source=source)
    covered = {entry["line"] for entry in entries if entry["kind"] != "usage"}
    skip_section = False
    uncovered: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.endswith(":") and not stripped.startswith("--"):
            heading = stripped[:-1].strip().lower()
            skip_section = heading in {"endpoints", "tasks"}
            continue
        if skip_section:
            continue
        if "=" in stripped and not stripped.startswith("-") and not line[:1].isspace():
            continue
        if stripped.startswith(("GET", "POST", "audiocpp_")):
            continue
        if stripped.startswith("[--"):
            continue
        looks_like_flag = stripped.startswith("--") or (
            stripped.startswith("-") and "--" in stripped
        )
        looks_like_bare = (
            line[:1].isspace()
            and not stripped.startswith("-")
            and stripped[0:1].isalpha()
            and ("<" in stripped or "." in stripped.split()[0])
        )
        if looks_like_flag and line_no not in covered:
            uncovered.append(f"{line_no}: {stripped}")
        elif looks_like_bare and " <" in f" {stripped}" and line_no not in covered:
            # Model request/session/load rows. Ignore prose such as
            # ``Shared CLI inputs and options are defaults for --batch-...``.
            first = stripped.split()[0]
            if (first[0].isalpha() and not first[0].isupper()) or "." in first:
                if first.lower() not in {"shared"}:
                    uncovered.append(f"{line_no}: {stripped}")
    assert not uncovered, f"{filename} unclassified help lines:\n" + "\n".join(uncovered)


def test_compact_task_mode_line_splits_into_two_params():
    text = """
  Supported tasks:
    --task asr --mode offline|streaming
"""
    parsed = _flat_params(text, "cli")
    index = {param["key"]: param for param in parsed}
    assert set(index) >= {"task", "mode"}
    assert "--mode" not in (index["task"].get("flags") or [])
    assert [opt["value"] for opt in index["mode"]["options"]] == ["offline", "streaming"]


def test_true_false_pipe_flags_are_bools():
    text = """
  Task routing and media roles:
    --use-pitch-shift true|false
    --do-sample true|false
"""
    parsed = _flat_params(text, "cli")
    index = {param["key"]: param for param in parsed}
    assert index["use_pitch_shift"]["type"] == "bool"
    assert index["do_sample"]["type"] == "bool"


def _capture_help(argv: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
        cwd=str(cwd),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return (proc.stdout or "") + (proc.stderr or "")


def test_server_help_fixture_matches_live_binary():
    binary = _LIVE_BIN / "audiocpp_server"
    if not binary.is_file():
        pytest.skip("audiocpp_server is not built locally")
    live = _capture_help([str(binary), "--help"], _LIVE_BIN.parent)
    fixture = (_FIXTURES / "audio_cpp_server_help_live.txt").read_text(encoding="utf-8")
    assert live == fixture


def test_cli_help_fixture_matches_live_binary():
    binary = _LIVE_BIN / "audiocpp_cli"
    if not binary.is_file():
        pytest.skip("audiocpp_cli is not built locally")
    live = _capture_help([str(binary), "--help"], _REPO / "data" / "audio-cpp" / "src")
    fixture = (_FIXTURES / "audio_cpp_cli_help_live.txt").read_text(encoding="utf-8")
    assert live == fixture


def test_nemotron_help_fixture_matches_live_binary():
    binary = _LIVE_BIN / "audiocpp_cli"
    model = _REPO / "data" / "models" / "audio-cpp" / "nemotron-3.5-asr-streaming-0.6b"
    if not binary.is_file() or not model.is_dir() or not _LIVE_SPECS.is_dir():
        pytest.skip("need local audiocpp_cli, specs, and nemotron bundle")
    live = _capture_help(
        [
            str(binary),
            "--model-spec-override",
            str(_LIVE_SPECS),
            "--model",
            str(model),
            "--family",
            "nemotron_asr",
            "--help",
        ],
        _REPO / "data" / "audio-cpp" / "src",
    )
    fixture = (_FIXTURES / "audio_cpp_nemotron_help_live.txt").read_text(encoding="utf-8")
    assert live == fixture


def test_qwen3_asr_help_fixture_matches_live_binary():
    binary = _LIVE_BIN / "audiocpp_cli"
    model = _REPO / "data" / "models" / "audio-cpp" / "Qwen3-ASR-0.6B"
    if not binary.is_file() or not model.is_dir() or not _LIVE_SPECS.is_dir():
        pytest.skip("need local audiocpp_cli, specs, and Qwen3-ASR bundle")
    live = _capture_help(
        [
            str(binary),
            "--model-spec-override",
            str(_LIVE_SPECS),
            "--model",
            str(model),
            "--family",
            "qwen3_asr",
            "--help",
        ],
        _REPO / "data" / "audio-cpp" / "src",
    )
    fixture = (_FIXTURES / "audio_cpp_qwen3_asr_help_live.txt").read_text(encoding="utf-8")
    assert live == fixture
