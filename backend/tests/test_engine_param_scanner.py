"""engine_param_scanner: --help capture and empty-parse handling."""

import os

from backend import engine_param_scanner as scanner_mod
from backend.engine_param_scanner import scan_engine_version, scan_llama_engine_version
from backend.llama_server_exec import llama_help_ld_library_path, resolve_llama_server_invocation_paths


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
    text, err = scanner_mod._run_help_argv(["/app/llama-server"], scan_engine="llama_cpp")
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


def test_scan_engine_version_uses_llama_swap_binary_for_active_row(tmp_path, monkeypatch):
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

    scan_engine_version(FakeStore(), "llama_cpp", {"version": "v1", "binary_path": str(row_path)})
    assert captured["binary_path"] == str(swap_path)


def test_scan_engine_version_keeps_row_path_for_non_active_version(tmp_path, monkeypatch):
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

    scan_engine_version(FakeStore(), "llama_cpp", {"version": "v-old", "binary_path": str(row_path)})
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
