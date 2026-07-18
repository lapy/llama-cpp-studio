"""Live autodetection against a real audiocpp_cli + prepared model.

Skipped unless AUDIO_CPP_CLI (and optionally AUDIO_CPP_MODEL /
AUDIO_CPP_FAMILY) point at a working install. Local Studio checkouts can also
auto-discover a CPU build under data/audio-cpp/src/build/ and a Nemotron bundle
under data/models/audio-cpp/.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from backend.cli_help_parsers import (
    parse_audio_cpp_help_to_sections,
    parse_audio_cpp_inspection,
)
from backend.engine_param_scanner import scan_audio_cpp_model_profile

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CLI = (
    _REPO_ROOT
    / "data"
    / "audio-cpp"
    / "src"
    / "build"
    / "linux-cpu-release"
    / "bin"
    / "audiocpp_cli"
)
_DEFAULT_SOURCE = _REPO_ROOT / "data" / "audio-cpp" / "src"
_DEFAULT_MODEL = (
    _REPO_ROOT
    / "data"
    / "models"
    / "audio-cpp"
    / "nemotron-3.5-asr-streaming-0.6b"
)
_DEFAULT_FAMILY = "nemotron_asr"


def _live_cli() -> Path | None:
    env = str(os.environ.get("AUDIO_CPP_CLI") or "").strip()
    if env:
        path = Path(env)
        return path if path.is_file() else None
    return _DEFAULT_CLI if _DEFAULT_CLI.is_file() else None


def _live_model() -> Path | None:
    env = str(os.environ.get("AUDIO_CPP_MODEL") or "").strip()
    if env:
        path = Path(env)
        return path if path.exists() else None
    return _DEFAULT_MODEL if _DEFAULT_MODEL.is_dir() else None


def _live_family() -> str:
    return str(os.environ.get("AUDIO_CPP_FAMILY") or _DEFAULT_FAMILY).strip()


def _live_source() -> str | None:
    env = str(os.environ.get("AUDIO_CPP_SOURCE") or "").strip()
    if env and Path(env).is_dir():
        return env
    if _DEFAULT_SOURCE.is_dir() and (_DEFAULT_SOURCE / "model_specs").is_dir():
        return str(_DEFAULT_SOURCE)
    return None


pytestmark = pytest.mark.skipif(
    _live_cli() is None or _live_model() is None,
    reason="Set AUDIO_CPP_CLI + AUDIO_CPP_MODEL (or use local data/audio-cpp build)",
)


class _Store:
    def __init__(self, tmp_path: Path):
        self._lock = threading.RLock()
        self._config_dir = str(tmp_path)

    def _write_yaml(self, path, data):
        import yaml

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)


def test_live_scan_audio_cpp_version_includes_server_process_options():
    cli = _live_cli()
    source = _live_source()
    if cli is None:
        pytest.skip("Need local audiocpp_cli")
    server = cli.parent / "audiocpp_server"
    if not server.is_file():
        pytest.skip("Need local audiocpp_server")

    from backend.engine_param_scanner import scan_audio_cpp_version

    entry = scan_audio_cpp_version(
        {
            "version": "live-cpu",
            "cli_binary_path": str(cli),
            "server_binary_path": str(server),
            "source_path": source,
        }
    )
    assert not entry.get("scan_error"), entry.get("scan_error")
    families = (entry.get("capabilities") or {}).get("families") or []
    assert len(families) >= 20
    process_keys = {
        param["key"]
        for section in entry.get("sections") or []
        for param in section.get("params") or []
        if param.get("scope") == "process"
    }
    assert {"host", "port", "backend", "threads"}.issubset(process_keys)


def test_live_audiocpp_help_advertises_session_options():
    cli = _live_cli()
    model = _live_model()
    family = _live_family()
    source = _live_source()
    assert cli is not None and model is not None

    import subprocess

    argv = [str(cli), "--model", str(model), "--family", family, "--help"]
    if source:
        argv[1:1] = ["--model-spec-override", str(Path(source) / "model_specs")]
    help_proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=source or str(cli.parent),
    )
    help_text = (help_proc.stdout or "") + (help_proc.stderr or "")
    assert help_proc.returncode == 0, help_text
    assert "Model session options:" in help_text

    sections = parse_audio_cpp_help_to_sections(help_text, source="cli")
    scoped = {
        (param["scope"], param["key"]): param
        for section in sections
        for param in section["params"]
    }
    session_keys = {key for scope, key in scoped if scope == "session_option"}
    assert session_keys, "expected session_option rows from live model help"
    # Nemotron advertises these; other families just need any session options.
    if family == "nemotron_asr":
        assert "nemotron_asr.weight_type" in session_keys
        weight = scoped[("session_option", "nemotron_asr.weight_type")]
        assert weight["type"] == "select"
        assert [opt["value"] for opt in weight["options"]] == [
            "native",
            "f32",
            "f16",
            "bf16",
            "q8_0",
        ]
        mem = scoped[("session_option", "nemotron_asr.mem_saver")]
        assert mem["type"] == "bool"
        assert scoped[("request_option", "language")]["read_only"] is True


def test_live_scan_fills_qwen3_asr_unadvertised_session_options(tmp_path):
    """Qwen3 ASR help omits session options; source discovery must fill them."""
    cli = _live_cli()
    source = _live_source()
    model_path = _REPO_ROOT / "data" / "models" / "audio-cpp" / "Qwen3-ASR-0.6B"
    if cli is None or source is None or not model_path.is_dir():
        pytest.skip("Need local audiocpp_cli, source tree, and Qwen3-ASR-0.6B")

    profile = scan_audio_cpp_model_profile(
        _Store(tmp_path),
        {
            "version": "live-qwen3",
            "cli_binary_path": str(cli),
            "server_binary_path": str(cli.parent / "audiocpp_server"),
            "source_path": source,
        },
        {
            "id": "audio-cpp--qwen3-asr-live",
            "family": "qwen3_asr",
            "task": "asr",
            "artifact": {"path": str(model_path), "package_kind": "prepared_bundle"},
            "config": {
                "engines": {"audio_cpp": {"family": "qwen3_asr", "task": "asr"}}
            },
        },
        force=True,
    )
    assert not profile.get("scan_error"), profile.get("scan_error")
    assert (profile.get("discovered_option_count") or 0) > 0
    session_keys = {
        param["key"]
        for section in profile.get("sections") or []
        for param in section.get("params") or []
        if param.get("scope") == "session_option"
    }
    assert "qwen3_asr.forced_aligner_model_path" in session_keys
    assert "qwen3_asr.vad_model_path" in session_keys
    assert "qwen3_asr.weight_type" in session_keys


def test_live_scan_audio_cpp_model_profile_persists_session_options(tmp_path):
    cli = _live_cli()
    model_path = _live_model()
    family = _live_family()
    source = _live_source()
    assert cli is not None and model_path is not None

    version_row = {
        "version": "live-test",
        "cli_binary_path": str(cli),
        "server_binary_path": str(cli.parent / "audiocpp_server"),
        "source_path": source,
    }
    model = {
        "id": f"audio-cpp--live-{family}",
        "family": family,
        "task": "asr",
        "artifact": {"path": str(model_path), "package_kind": "prepared_bundle"},
        "config": {"engines": {"audio_cpp": {"family": family, "task": "asr"}}},
    }
    profile = scan_audio_cpp_model_profile(
        _Store(tmp_path), version_row, model, force=True
    )
    assert not profile.get("scan_error"), profile.get("scan_error")
    inspection = profile.get("inspection") or {}
    assert inspection.get("family") == family

    scoped = {
        (param["scope"], param["key"]): param
        for section in profile.get("sections") or []
        for param in section.get("params") or []
    }
    session_keys = {key for scope, key in scoped if scope == "session_option"}
    assert session_keys
    if family == "nemotron_asr":
        assert "nemotron_asr.mem_saver" in session_keys
        assert scoped[("session_option", "nemotron_asr.mem_saver")]["type"] == "bool"

    # Inspection parse stays aligned with --inspect output.
    import subprocess

    argv = [
        str(cli),
        "--model",
        str(model_path),
        "--family",
        family,
        "--inspect",
    ]
    if source:
        argv[1:1] = ["--model-spec-override", str(Path(source) / "model_specs")]
    inspect_proc = subprocess.run(
        argv,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        cwd=source or str(cli.parent),
    )
    parsed = parse_audio_cpp_inspection(
        (inspect_proc.stdout or "") + (inspect_proc.stderr or "")
    )
    assert parsed.get("family") == family
    assert "asr" in (parsed.get("task_names") or [])
