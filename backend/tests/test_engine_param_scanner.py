"""engine_param_scanner: --help capture and empty-parse handling."""

import os

from backend import engine_param_scanner as scanner_mod
from backend.engine_param_scanner import (
    scan_audio_cpp_version,
    scan_engine_version,
    scan_llama_engine_version,
)
from backend.llama_server_exec import (
    llama_help_ld_library_path,
    resolve_llama_server_invocation_paths,
)


def test_scan_llama_cuda_only_stdout_reports_error(tmp_path, monkeypatch):
    fake = tmp_path / "llama-server"
    fake.write_bytes(b"\0")
    monkeypatch.setattr(
        "backend.engine_param_scanner._run_help_argv",
        lambda *a, **k: (
            "ggml_cuda_init: found 1 CUDA devices (Total VRAM: 8000 MiB):\n",
            "process exited with code 1",
        ),
    )
    row = {"binary_path": str(fake), "version": "t1"}
    entry = scan_llama_engine_version("llama_cpp", row)
    assert entry["scan_error"]
    assert sum(len(s.get("params") or []) for s in entry.get("sections") or []) == 0


def test_scan_llama_help_despite_nonzero_exit_succeeds(tmp_path, monkeypatch):
    fake = tmp_path / "llama-server"
    fake.write_bytes(b"\0")
    help_body = """
----- common params -----
-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context
"""
    monkeypatch.setattr(
        "backend.engine_param_scanner._run_help_argv",
        lambda *a, **k: (help_body, "process exited with code 1"),
    )
    row = {"binary_path": str(fake), "version": "t2"}
    entry = scan_llama_engine_version("llama_cpp", row)
    assert entry.get("scan_error") is None
    n = sum(len(s.get("params") or []) for s in entry.get("sections") or [])
    assert n >= 2


def test_run_help_argv_exit_127_lmdeploy_hint(monkeypatch):
    class _R:
        returncode = 127
        stdout = ""

    monkeypatch.setattr(scanner_mod.subprocess, "run", lambda *a, **k: _R())
    text, err = scanner_mod._run_help_argv(
        ["/data/lmdeploy/venv/bin/lmdeploy"], scan_engine="lmdeploy"
    )
    assert text == ""
    assert "127" in err
    assert "lmdeploy" in err.lower()


def test_run_help_argv_exit_127_llama_hint(monkeypatch):
    class _R:
        returncode = 127
        stdout = ""

    monkeypatch.setattr(scanner_mod.subprocess, "run", lambda *a, **k: _R())
    text, err = scanner_mod._run_help_argv(
        ["/app/llama-server"], scan_engine="llama_cpp"
    )
    assert text == ""
    assert "127" in err
    assert "llama.cpp" in err.lower() or "ggml" in err.lower()


def test_llama_scan_ld_path_includes_build_bin(tmp_path):
    root = tmp_path / "source-abc"
    root.mkdir()
    build_bin = root / "build" / "bin"
    build_bin.mkdir(parents=True)
    out = llama_help_ld_library_path(str(root))
    parts = out.split(os.pathsep)
    assert str(root) in parts
    assert str(build_bin) in parts


def test_scan_engine_version_uses_llama_swap_binary_for_active_row(
    tmp_path, monkeypatch
):
    """Active llama_cpp/ik_llama scan uses ``get_active_llama_swap_binary_path`` when it matches the engine."""
    swap_path = tmp_path / "swap" / "llama-server"
    swap_path.parent.mkdir(parents=True)
    swap_path.write_bytes(b"\0")

    row_path = tmp_path / "row" / "llama-server"
    row_path.parent.mkdir(parents=True)
    row_path.write_bytes(b"\0")

    captured = {}

    def spy(eng, row):
        captured["binary_path"] = row["binary_path"]
        return {
            "binary_path": row["binary_path"],
            "scanned_at": "t",
            "scan_error": None,
            "sections": [],
        }

    monkeypatch.setattr(scanner_mod, "scan_llama_engine_version", spy)
    monkeypatch.setattr(scanner_mod, "upsert_version_entry", lambda *a, **k: None)
    monkeypatch.setattr(scanner_mod, "_clear_llama_flags_cache", lambda: None)
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_llama_swap_binary_path",
        lambda _store: str(swap_path),
    )
    monkeypatch.setattr(
        "backend.llama_engine_resolve.infer_llama_engine_for_binary",
        lambda _store, _p: "llama_cpp",
    )

    class FakeStore:
        def get_active_engine_version(self, eng):
            if eng == "llama_cpp":
                return {"version": "v1", "binary_path": str(row_path)}
            return None

    scan_engine_version(
        FakeStore(), "llama_cpp", {"version": "v1", "binary_path": str(row_path)}
    )
    assert captured["binary_path"] == str(swap_path)


def test_scan_engine_version_keeps_row_path_for_non_active_version(
    tmp_path, monkeypatch
):
    swap_path = tmp_path / "swap" / "llama-server"
    swap_path.parent.mkdir(parents=True)
    swap_path.write_bytes(b"\0")
    row_path = tmp_path / "row" / "llama-server"
    row_path.parent.mkdir(parents=True)
    row_path.write_bytes(b"\0")

    captured = {}

    def spy(eng, row):
        captured["binary_path"] = row["binary_path"]
        return {
            "binary_path": row["binary_path"],
            "scanned_at": "t",
            "scan_error": None,
            "sections": [],
        }

    monkeypatch.setattr(scanner_mod, "scan_llama_engine_version", spy)
    monkeypatch.setattr(scanner_mod, "upsert_version_entry", lambda *a, **k: None)
    monkeypatch.setattr(scanner_mod, "_clear_llama_flags_cache", lambda: None)
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_llama_swap_binary_path",
        lambda _store: str(swap_path),
    )
    monkeypatch.setattr(
        "backend.llama_engine_resolve.infer_llama_engine_for_binary",
        lambda _store, _p: "llama_cpp",
    )

    class FakeStore:
        def get_active_engine_version(self, eng):
            if eng == "llama_cpp":
                return {"version": "v-active", "binary_path": str(row_path)}
            return None

    scan_engine_version(
        FakeStore(), "llama_cpp", {"version": "v-old", "binary_path": str(row_path)}
    )
    assert captured["binary_path"] == str(row_path)


def test_resolve_llama_server_prefers_build_bin_executable(tmp_path):
    """Same layout as llama-swap: stored path under .../bin/ but real binary under .../build/bin/."""
    install = tmp_path / "install"
    bindir = install / "bin"
    buildbin = install / "build" / "bin"
    bindir.mkdir(parents=True)
    buildbin.mkdir(parents=True)
    real = buildbin / "llama-server"
    real.write_bytes(b"#!/bin/true\n")
    real.chmod(0o755)
    stored = bindir / "llama-server"
    stored.write_text("placeholder\n")
    stored.chmod(0o644)
    exe, cwd = resolve_llama_server_invocation_paths(str(stored))
    assert exe == str(real)
    assert cwd == str(buildbin)


def test_audio_cpp_source_root_prefers_source_path_and_walks_parents(tmp_path):
    source = tmp_path / "audio-src"
    (source / "model_specs").mkdir(parents=True)
    build_bin = source / "build" / "linux-cpu-release" / "bin"
    build_bin.mkdir(parents=True)
    cli = build_bin / "audiocpp_cli"
    cli.write_text("x", encoding="utf-8")

    assert scanner_mod._audio_cpp_source_root(
        {"source_path": str(source)}, str(cli)
    ) == str(source)
    assert scanner_mod._audio_cpp_source_root({}, str(cli)) == str(source)
    assert scanner_mod._audio_cpp_model_spec_override({}, str(cli)) == str(
        source / "model_specs"
    )


def test_audio_scan_records_contract_fingerprint_and_drift(tmp_path, monkeypatch):
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_bytes(b"\0")
    cli.write_bytes(b"\0")
    server.chmod(0o755)
    cli.chmod(0o755)

    def fake_help(argv, **kwargs):
        if argv[0] == str(server):
            return (
                "Usage: audiocpp_server [options]\n"
                "  --host <host>  Bind host\n"
                "  --port <port>  Bind port\n",
                None,
            )
        if "--list-loaders" in argv:
            return ("demo_tts: tts (offline, streaming)\n", None)
        return (
            "Usage: audiocpp_cli [options]\n"
            "  --model <path>  Model path\n",
            None,
        )

    monkeypatch.setattr(scanner_mod, "_run_help_argv", fake_help)
    entry = scan_audio_cpp_version(
        {
            "version": "v1",
            "source_commit": "different-commit",
            "previous_contract_fingerprint": "0" * 64,
            "contract_fingerprint": "0" * 64,
            "server_binary_path": str(server),
            "cli_binary_path": str(cli),
        }
    )

    assert "compatibility_commit" not in entry
    assert len(entry["contract_fingerprint"]) == 64
    assert entry["contract_changed"] is True
    assert any("contract fingerprint changed" in row for row in entry["warnings"])
    assert entry["capabilities"]["families"] == ["demo_tts"]
    assert entry["capabilities"]["tasks"] == ["tts"]
    assert entry["capabilities"]["family_tasks"]["demo_tts"] == ["tts"]
    assert entry["capabilities"]["discovery_source"] == "text"


def test_audio_scan_prefers_list_loaders_json(tmp_path, monkeypatch):
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_bytes(b"\0")
    cli.write_bytes(b"\0")
    server.chmod(0o755)
    cli.chmod(0o755)
    calls = []

    def fake_help(argv, **kwargs):
        calls.append(list(argv))
        if argv[0] == str(server):
            return (
                "Usage: audiocpp_server [options]\n"
                "  --host <host>  Bind host\n"
                "  --port <port>  Bind port\n",
                None,
            )
        if "--list-loaders" in argv and "--json" in argv:
            return (
                '{"loaders":[{"family":"json_tts","tasks":[{"id":"tts","modes":["offline"]}],'
                '"instructions_policy":"openai_instruct"}]}',
                None,
            )
        if "--list-loaders" in argv:
            return ("should_not_use: tts (offline)\n", None)
        return (
            "Usage: audiocpp_cli [options]\n"
            "  --model <path>  Model path\n",
            None,
        )

    monkeypatch.setattr(scanner_mod, "_run_help_argv", fake_help)
    entry = scan_audio_cpp_version(
        {
            "version": "v1",
            "server_binary_path": str(server),
            "cli_binary_path": str(cli),
        }
    )
    assert entry["capabilities"]["families"] == ["json_tts"]
    assert entry["capabilities"]["discovery_source"] == "json"
    assert entry["capabilities"]["family_policies"]["json_tts"] == "openai_instruct"
    assert entry["capabilities"]["contract_grade"] in {"partial", "full"}
    assert any("--json" in call for call in calls if "--list-loaders" in call)


def test_probe_catalog_resolves_model_manager_from_source_path(tmp_path, monkeypatch):
    from backend.engine_param_scanner import _probe_catalog_contract

    tools = tmp_path / "tools"
    tools.mkdir()
    manager = tools / "model_manager.py"
    manager.write_text(
        "import json\n"
        "print(json.dumps([{"
        '"id":"demo","family":"demo","standalone":True,'
        '"tasks":["tts"],"gated":False'
        "}]))\n",
        encoding="utf-8",
    )

    class Result:
        returncode = 0
        stdout = (
            '[{"id":"demo","family":"demo","standalone":true,'
            '"tasks":["tts"],"gated":false}]'
        )
        stderr = ""

    monkeypatch.setattr(
        scanner_mod.subprocess,
        "run",
        lambda *a, **k: Result(),
    )
    probe = _probe_catalog_contract({"source_path": str(tmp_path)})
    assert probe["catalog_source"] == "json"
    assert probe["catalog_identity"] is True
    assert probe["catalog_package_count"] == 1


def test_grade_audio_cpp_contract_and_delta():
    from backend.engine_param_scanner import (
        compute_audio_cpp_capability_delta,
        grade_audio_cpp_contract,
    )

    assert (
        grade_audio_cpp_contract(
            loaders_source="json",
            catalog_source="json",
            catalog_identity=True,
            family_tasks={"omnivoice": ["tts"]},
        )
        == "full"
    )
    assert (
        grade_audio_cpp_contract(
            loaders_source="json",
            catalog_source="ast_fallback_needed",
            catalog_identity=False,
            family_tasks={"omnivoice": ["tts"]},
        )
        == "partial"
    )
    assert (
        grade_audio_cpp_contract(
            loaders_source="text",
            catalog_source="missing",
            catalog_identity=False,
            family_tasks={},
        )
        == "thin"
    )
    delta = compute_audio_cpp_capability_delta(
        {"capabilities": {"families": ["a"], "tasks": ["tts"], "contract_grade": "thin"}},
        {
            "capabilities": {
                "families": ["a", "b"],
                "tasks": ["tts", "asr"],
                "family_tasks": {"a": ["tts"], "b": []},
                "contract_grade": "full",
                "catalog_source": "json",
                "discovery_source": "json",
                "contract_warnings": ["demo warning"],
            }
        },
    )
    assert delta["added_families"] == ["b"]
    assert delta["contract_grade"] == "full"
    assert "b" in delta["families_without_tasks"]
    assert delta["warnings"] == ["demo warning"]


def test_model_profile_retries_when_cached_scan_error(tmp_path, monkeypatch):
    """Failed install-time scans must not permanently poison lazy loads."""
    from backend.engine_param_scanner import scan_audio_cpp_model_profile

    cli = tmp_path / "audiocpp_cli"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    cli.write_bytes(b"\0")
    cli.chmod(0o755)

    version_row = {
        "version": "v1",
        "cli_binary_path": str(cli),
        "server_binary_path": str(cli),
        "source_commit": "abc",
    }
    model = {
        "id": "audio-cpp--demo",
        "family": "demo_tts",
        "artifact": {"path": str(model_dir), "package_kind": "prepared_bundle"},
    }
    fingerprint = scanner_mod.audio_cpp_model_profile_fingerprint(version_row, model)
    calls = {"inspect": 0}

    class _Store:
        def __init__(self):
            self._lock = __import__("threading").RLock()
            self._config_dir = str(tmp_path)

    monkeypatch.setattr(
        scanner_mod,
        "get_model_profile_entry",
        lambda store, engine, version, fp: {
            "fingerprint": fingerprint,
            "scan_error": "empty audio.cpp inspection",
            "sections": [],
        },
    )

    def fake_inspect(argv, cli_path, cwd=None):
        calls["inspect"] += 1
        return (
            "family=demo_tts\n"
            "supported_tasks=1\n"
            "task=tts modes=offline\n",
            None,
        )

    def fake_help(argv, **kwargs):
        return (
            "Usage: audiocpp_cli\n"
            "Model session options:\n"
            "  demo_tts.weight_type <native|f32>\n",
            None,
        )

    monkeypatch.setattr(scanner_mod, "_run_audio_cpp_inspect", fake_inspect)
    monkeypatch.setattr(scanner_mod, "_run_help_argv", fake_help)
    monkeypatch.setattr(
        scanner_mod,
        "upsert_model_profile_entry",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        scanner_mod,
        "_audio_cpp_source_root",
        lambda *a, **k: None,
    )

    profile = scan_audio_cpp_model_profile(
        _Store(), version_row, model, force=False
    )
    assert calls["inspect"] == 1
    assert not profile.get("scan_error"), profile.get("scan_error")
    session_keys = {
        p["key"]
        for s in profile.get("sections") or []
        for p in s.get("params") or []
        if p.get("scope") == "session_option"
    }
    assert "demo_tts.weight_type" in session_keys
