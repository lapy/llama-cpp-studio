"""Regression tests against full captured parameter-scan logs.

Fixtures are the copyable UI logs (``H0001|…`` help blocks, catalog, coverage),
not synthetic excerpts. Each engine is unwrapped, re-parsed, coverage-checked,
line-audited, and pinned to a catalog snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.cli_help_parsers import (
    LONG_FLAG_RE,
    _REMOVED_ARGUMENT_RE,
    _is_llama_option_line,
    parse_audio_cpp_help_to_sections,
    parse_audio_cpp_loaders_json,
    parse_llama_help_to_sections,
    parse_llama_server_help,
    parse_sglang_launch_server_help,
    parse_vllm_serve_help,
    sglang_params_to_sections,
    try_parse_json_payload,
    vllm_params_to_sections,
)
from backend.engine_param_scanner import _prefix_sections
from backend.param_scan_log import (
    compute_flag_coverage,
    extract_help_captures,
    extract_help_text,
    snapshot_param_rows,
)
from backend.tests.help_parser_audit import (
    extract_llama_help_entries,
    extract_lmdeploy_help_entries,
    extract_vllm_help_entries,
    verify_all_help_params,
    verify_llama_help_line_by_line,
    verify_vllm_help_line_by_line,
)

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "param_scans"
_EXPECTED = _FIXTURES / "expected"


def _read(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


def _expected(name: str) -> list:
    return json.loads((_EXPECTED / name).read_text(encoding="utf-8"))


def _flat(sections: list[dict]) -> list[dict]:
    return [param for section in sections for param in (section.get("params") or [])]


def _by_key(sections: list[dict]) -> dict:
    return {param["key"]: param for param in _flat(sections)}


def _assert_snapshot(actual_sections: list[dict], expected_name: str) -> None:
    actual = snapshot_param_rows(actual_sections)
    expected = _expected(expected_name)
    assert len(actual) == len(expected)
    by_actual = {row["key"]: row for row in actual}
    by_expected = {row["key"]: row for row in expected}
    assert set(by_actual) == set(by_expected)
    mismatches = [
        f"{key}: {by_actual[key]!r} != {by_expected[key]!r}"
        for key in sorted(by_expected)
        if by_actual[key] != by_expected[key]
    ]
    assert not mismatches, ";\n".join(mismatches[:40])


def _removed_option_flags(help_text: str) -> set[str]:
    flags: set[str] = set()
    pending: list[str] = []
    desc: list[str] = []

    def flush() -> None:
        if pending and _REMOVED_ARGUMENT_RE.search(" ".join(desc)):
            flags.update(pending)
        pending.clear()
        desc.clear()

    for line in help_text.splitlines():
        if _is_llama_option_line(line):
            flush()
            pending.extend(LONG_FLAG_RE.findall(line))
            desc.append(line)
            continue
        if pending:
            desc.append(line.strip())
    flush()
    return flags


def _parse_llama(help_text: str) -> list[dict]:
    return parse_llama_help_to_sections(help_text, "llama_cpp")


def test_extract_help_text_passthrough_and_gap_fill():
    raw = "options:\n  --port PORT  Listen port\n"
    assert extract_help_text(raw) == raw
    log = (
        "H0003|  --host HOST\n"
        "===== HELP OUTPUT: demo --help (3 lines, 40 chars) =====\n"
        "H0001|options:\n"
        "H0002|  --port PORT\n"
        "===== END HELP OUTPUT =====\n"
    )
    captures = extract_help_captures(log)
    assert len(captures) == 1
    assert captures[0].complete is True
    assert captures[0].text.splitlines() == ["options:", "  --port PORT", "  --host HOST"]


def test_sglang_v100_full_scan_log_regression():
    log = _read("sglang_v100.scan.txt")
    captures = extract_help_captures(log)
    assert len(captures) == 1
    help_text = captures[0].text
    assert captures[0].complete is True
    assert captures[0].line_count == captures[0].declared_lines == 1745
    assert "usage: sglang serve" in help_text

    raw = parse_sglang_launch_server_help(help_text)
    sections = sglang_params_to_sections(raw)
    coverage = compute_flag_coverage(help_text, sections)
    assert coverage.interesting_missing == []
    assert coverage.extra == []
    assert len(coverage.help_flags) == len(coverage.parsed_flags) == 413

    issues = verify_all_help_params(
        extract_lmdeploy_help_entries(help_text),
        raw,
        skip_default_check=("--stream-output",),
    )
    issues = [
        issue
        for issue in issues
        if not issue.startswith("param count ")
        and "extra parsed param not in fixture: --stream-output" not in issue
    ]
    assert not issues, ";\n".join(issues)

    _assert_snapshot(sections, "sglang_v100.json")
    by_key = _by_key(sections)
    assert by_key["tokenizer_backend"]["default"] is None
    assert by_key["pre_warm_nccl"]["default"] is None
    assert by_key["sampling_defaults"]["default"] == "model"
    assert by_key["asr_max_concurrent_sessions"]["default"] == 32
    assert by_key["disaggregation_transfer_backend"]["default"] == "mooncake"
    assert by_key["experts_shared_outer_loras"]["negative_flag"] == (
        "--no-experts-shared-outer-loras"
    )


def test_llama_cpp_full_scan_log_regression():
    log = _read("llama_cpp.scan.txt")
    captures = extract_help_captures(log)
    assert len(captures) == 1
    help_text = captures[0].text
    assert captures[0].complete is True
    assert captures[0].line_count == captures[0].declared_lines == 730
    assert help_text.startswith("----- common params -----")

    sections = _parse_llama(help_text)
    raw = parse_llama_server_help(help_text, "llama_cpp")
    from backend.cli_help_parsers import _attach_llama_sections

    attached = _attach_llama_sections(help_text, raw)
    coverage = compute_flag_coverage(
        help_text,
        sections,
        accounted_flags=_removed_option_flags(help_text),
    )
    assert coverage.extra == []
    assert coverage.interesting_missing == []

    issues = verify_all_help_params(extract_llama_help_entries(help_text), attached)
    assert not issues, ";\n".join(issues)
    line_issues = verify_llama_help_line_by_line(help_text, attached)
    assert not line_issues, ";\n".join(line_issues)

    _assert_snapshot(sections, "llama_cpp.json")
    by_key = _by_key(sections)
    assert by_key["temperature"]["default"] == 0.8
    assert by_key["port"]["reserved"] is True
    assert {section["id"] for section in sections} == {
        "common_params",
        "sampling_params",
        "speculative_params",
        "example_specific_params",
    }


def test_onecat_vllm_full_scan_log_regression():
    log = _read("onecat_vllm.scan.txt")
    captures = extract_help_captures(log)
    assert len(captures) == 1
    help_text = captures[0].text
    assert captures[0].complete is True
    assert captures[0].line_count == captures[0].declared_lines == 1724
    assert help_text.startswith("usage: vllm serve")

    raw = parse_vllm_serve_help(help_text)
    sections = vllm_params_to_sections(raw)
    coverage = compute_flag_coverage(help_text, sections)
    assert coverage.interesting_missing == []
    assert coverage.extra == []

    issues = verify_all_help_params(
        extract_vllm_help_entries(help_text),
        raw,
        skip_default_check=("--cpu-offload-params", "--offload-params"),
    )
    assert not issues, ";\n".join(issues)
    line_issues = verify_vllm_help_line_by_line(help_text, raw)
    assert not line_issues, ";\n".join(line_issues)

    _assert_snapshot(sections, "onecat_vllm.json")
    by_key = _by_key(sections)
    assert by_key["port"]["reserved"] is True
    assert by_key["disable_log_stats"]["value_kind"] == "flag"
    assert len(raw) == 262


def test_audio_cpp_full_scan_log_regression():
    log = _read("audio_cpp.scan.txt")
    captures = extract_help_captures(log)
    assert len(captures) == 3
    assert all(capture.complete for capture in captures)
    server = next(cap for cap in captures if cap.title.endswith("audiocpp_server --help"))
    cli = next(
        cap
        for cap in captures
        if cap.title.endswith("audiocpp_cli --help")
    )
    loaders = next(cap for cap in captures if "list-loaders" in cap.title)
    assert server.line_count == server.declared_lines == 61
    assert cli.line_count == cli.declared_lines == 124
    assert loaders.line_count == loaders.declared_lines == 1

    sections = [
        *_prefix_sections(
            parse_audio_cpp_help_to_sections(server.text, source="server"),
            "server",
        ),
        *_prefix_sections(
            parse_audio_cpp_help_to_sections(cli.text, source="cli"),
            "cli",
        ),
    ]
    coverage = compute_flag_coverage("\n".join([server.text, cli.text]), sections)
    assert coverage.interesting_missing == []
    assert coverage.extra == []

    _assert_snapshot(sections, "audio_cpp.json")
    by_key = _by_key(sections)
    assert by_key["backend"]["options"]
    assert by_key["task"]["value_kind"] == "enum"
    assert by_key["list_loaders"]["value_kind"] == "flag"
    assert "json" not in by_key

    payload = try_parse_json_payload(loaders.text)
    assert payload is not None
    meta = parse_audio_cpp_loaders_json(payload)
    assert meta.get("families") or payload.get("loaders")


@pytest.mark.parametrize(
    ("scan_name", "engine"),
    [
        ("sglang_v100.scan.txt", "sglang_v100"),
        ("llama_cpp.scan.txt", "llama_cpp"),
        ("onecat_vllm.scan.txt", "1cat_vllm"),
        ("audio_cpp.scan.txt", "audio_cpp"),
    ],
)
def test_full_scan_log_is_the_copyable_ui_format(scan_name: str, engine: str):
    log = _read(scan_name)
    assert log.startswith(f"=== Parameter scan: {engine} ===")
    assert "===== HELP OUTPUT:" in log
    assert "===== FLAG COVERAGE" in log
    assert "===== CATALOG RESULT:" in log
    assert extract_help_captures(log)
