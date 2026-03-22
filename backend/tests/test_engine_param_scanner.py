"""engine_param_scanner: --help capture and empty-parse handling."""

from backend.engine_param_scanner import scan_llama_engine_version


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
