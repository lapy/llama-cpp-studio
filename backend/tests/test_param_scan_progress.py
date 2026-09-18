"""Parameter-scan progress sessions stream help output and extraction logs."""

from __future__ import annotations

import backend.progress_manager as pm_mod
from backend.param_scan_progress import (
    PARAM_SCAN_TASK_TYPE,
    ParamScanSession,
    get_param_scan_session,
    trace_extract,
)
from backend import engine_param_scanner as scanner_mod


def test_trace_extract_is_noop_without_session():
    trace_extract("accept", key="ctx_size")


def test_session_logs_help_and_catalog(monkeypatch):
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        session = ParamScanSession(engine="llama_cpp", version="v1")
        session.attach()
        assert get_param_scan_session() is session
        task = pm_mod.get_progress_manager().get_task(session.task_id)
        assert task["type"] == PARAM_SCAN_TASK_TYPE
        assert "llama.cpp" in task["description"]

        session.log_command(["/tmp/llama-server", "--help"], cwd="/tmp")
        session.log_capture(
            ["/tmp/llama-server", "--help"],
            "----- common -----\n-c, --ctx-size N  context size\n",
            None,
        )
        session.log_catalog(
            {
                "binary_path": "/tmp/llama-server",
                "scan_error": None,
                "sections": [
                    {
                        "id": "common",
                        "label": "common",
                        "params": [
                            {
                                "key": "ctx_size",
                                "flags": ["--ctx-size"],
                                "value_kind": "scalar",
                                "type": "int",
                                "default": 4096,
                                "reserved": False,
                            }
                        ],
                    }
                ],
            }
        )
        session.log_flag_coverage(
            "-c, --ctx-size N\n--orphan-flag X\n",
            [
                {
                    "params": [{"flags": ["--ctx-size"]}],
                }
            ],
        )
        session.finish_from_entry(
            {
                "scan_error": None,
                "sections": [{"params": [{}, {}]}],
            }
        )
        session.detach()

        task = pm_mod.get_progress_manager().get_task(session.task_id)
        assert task["status"] == "completed"
        assert "Indexed 2 CLI options" in task["message"]
        joined = "\n".join(session.lines)
        assert "HELP OUTPUT" in joined
        assert "H0002|-c, --ctx-size N  context size" in joined
        assert "EXTRACT accept" in joined
        assert "MISSING --orphan-flag" in joined
        assert get_param_scan_session() is None
    finally:
        pm_mod._progress_manager = None


def test_flag_coverage_ignores_wrapped_fragments_and_short_flags():
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        session = ParamScanSession(engine="sglang_v100", version="v1")
        session.attach()
        session.log_flag_coverage(
            "  --disable-radix-cache\n"
            "                        Disable --disable-\n"
            "                        radix cache. Use --attention-\n"
            "                        backend flashinfer.\n"
            "  -h, --help\n",
            [
                {
                    "params": [
                        {"flags": ["--disable-radix-cache"]},
                        {"flags": ["-h", "--help"]},
                    ]
                }
            ],
        )
        joined = "\n".join(session.lines)
        assert "MISSING --disable-" not in joined
        assert "MISSING --attention-" not in joined
        assert "EXTRA -h" not in joined
        assert "missing=0 extra=0" in joined
        session.detach()
    finally:
        pm_mod._progress_manager = None


def test_flag_coverage_ignores_removed_args_and_glob_fragments():
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        session = ParamScanSession(engine="llama_cpp", version="v1")
        session.attach()
        session.extract(
            "skip_row",
            reason="removed_argument",
            flags="--draft,--draft-n,--draft-max",
        )
        session.log_flag_coverage(
            "--draft N\n"
            "the argument has been removed.\n"
            "use the respective --spec-ngram-*-size-n\n"
            "--ctx-size N\n",
            [{"params": [{"flags": ["--ctx-size"]}]}],
        )
        joined = "\n".join(session.lines)
        assert "MISSING --draft" not in joined
        assert "MISSING --spec-ngram-" not in joined
        assert "MISSING --spec-ngram" not in joined
        assert "missing=0 extra=0" in joined
        session.detach()
    finally:
        pm_mod._progress_manager = None


def test_flag_coverage_ignores_vllm_footer_and_numactl_mentions():
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        session = ParamScanSession(engine="1cat_vllm", version="v1")
        session.attach()
        session.log_flag_coverage(
            "  --port PORT\n"
            "                        Use numactl --physcpubind and --membind.\n"
            "When passing JSON CLI arguments, the following sets of arguments are equivalent:\n"
            "   --json-arg '{\"key1\": \"value1\"}'\n",
            [{"params": [{"flags": ["--port"]}]}],
        )
        joined = "\n".join(session.lines)
        assert "MISSING --json-arg" not in joined
        assert "MISSING --physcpubind" not in joined
        assert "MISSING --membind" not in joined
        assert "missing=0 extra=0" in joined
        session.detach()
    finally:
        pm_mod._progress_manager = None


def test_flag_coverage_ignores_nested_optional_json_flag():
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        session = ParamScanSession(engine="audio_cpp", version="v1")
        session.attach()
        session.log_flag_coverage(
            "    --inspect\n"
            "    --list-loaders [--json]\n",
            [
                {
                    "params": [
                        {"flags": ["--inspect"]},
                        {"flags": ["--list-loaders"]},
                    ]
                }
            ],
        )
        joined = "\n".join(session.lines)
        assert "MISSING --json" not in joined
        assert "missing=0 extra=0" in joined
        session.detach()
    finally:
        pm_mod._progress_manager = None


def test_scan_engine_version_creates_progress_task(tmp_path, monkeypatch):
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        fake = tmp_path / "llama-server"
        fake.write_bytes(b"\0")
        help_body = """
----- common params -----
-h,    --help, --usage                  print usage and exit
-c,    --ctx-size N                     size of the prompt context (default: 4096)
"""
        monkeypatch.setattr(
            scanner_mod,
            "_run_help_argv",
            lambda *a, **k: (help_body, None),
        )
        monkeypatch.setattr(scanner_mod, "upsert_version_entry", lambda *a, **k: None)
        monkeypatch.setattr(scanner_mod, "_clear_llama_flags_cache", lambda: None)

        class Store:
            def get_active_engine_version(self, _engine):
                return None

        entry = scanner_mod.scan_engine_version(
            Store(),
            "llama_cpp",
            {"version": "v-scan", "binary_path": str(fake)},
        )
        assert entry.get("scan_error") is None
        tasks = list(pm_mod.get_progress_manager()._tasks.values())
        scan_tasks = [t for t in tasks if t["type"] == PARAM_SCAN_TASK_TYPE]
        assert len(scan_tasks) == 1
        task = scan_tasks[0]
        assert task["status"] == "completed"
        assert "Indexed" in task["message"]
        assert get_param_scan_session() is None
    finally:
        pm_mod._progress_manager = None


def test_parent_session_is_reused(tmp_path, monkeypatch):
    pm_mod._progress_manager = pm_mod.ProgressManager()
    try:
        fake = tmp_path / "llama-server"
        fake.write_bytes(b"\0")
        monkeypatch.setattr(
            scanner_mod,
            "_run_help_argv",
            lambda *a, **k: (
                "----- common params -----\n-c, --ctx-size N  size\n",
                None,
            ),
        )
        monkeypatch.setattr(scanner_mod, "upsert_version_entry", lambda *a, **k: None)
        monkeypatch.setattr(scanner_mod, "_clear_llama_flags_cache", lambda: None)

        class Store:
            def get_active_engine_version(self, _engine):
                return None

        parent = ParamScanSession(engine="llama_cpp", version="v-parent")
        parent.attach()
        scanner_mod.scan_engine_version(
            Store(),
            "llama_cpp",
            {"version": "v-parent", "binary_path": str(fake)},
        )
        assert parent._finished is False
        joined = "\n".join(parent.lines)
        assert "CATALOG RESULT" in joined
        assert "EXTRACT accept" in joined
        parent.complete("ok")
        parent.detach()
        tasks = [
            t
            for t in pm_mod.get_progress_manager()._tasks.values()
            if t["type"] == PARAM_SCAN_TASK_TYPE
        ]
        assert len(tasks) == 1
    finally:
        pm_mod._progress_manager = None
