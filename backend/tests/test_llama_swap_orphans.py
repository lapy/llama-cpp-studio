"""Coverage for engine-orphan detection/sweep and the exec-rendered swap cmds."""

import asyncio
import shlex
import signal

import pytest

import backend.proxy.llama_swap.orphans as orphans


VENV_PY = "/app/data/sglang-v100/x/venv/bin/python"
SITE = "/app/data/sglang-v100/x/venv/lib/python3.11/site-packages/torch/worker.py"


def _row(pid, ppid, pgid, name, cmdline, status="sleeping", create_time=1_000_000.0):
    return {
        "pid": pid,
        "ppid": ppid,
        "pgid": pgid,
        "name": name,
        "cmdline": shlex.split(cmdline) if cmdline else [],
        "create_time": create_time,
        "status": status,
    }


PS_ROWS = [
    # Leaked generation: launcher + renamed scheduler + tracker + plain worker,
    # all reparented to PID 1 and sharing the launcher's process group.
    _row(100, 1, 100, "python", f"{VENV_PY} -m sglang.launch_server --model-path m --port 3001"),
    _row(101, 1, 100, "sglang::schedul", "sglang::scheduler_TP0"),
    _row(102, 1, 100, "python", f"{VENV_PY} -c from multiprocessing.resource_tracker import main;main(46)"),
    _row(103, 1, 100, "python", f"{VENV_PY} {SITE}"),
    # Processes that must never be flagged.
    _row(200, 1, 200, "audiocpp_server", "/app/data/audio-cpp/build/bin/audiocpp_server --config x.json"),
    _row(300, 85, 100, "python", f"{VENV_PY} {SITE}"),  # supervised child of llama-swap
    _row(400, 1, 400, "sglang::schedul", "", status="zombie"),  # zombie remnant
    _row(500, 1, 500, "python", "python backend/main.py"),
    _row(600, 1, 600, "sh", "sh -c du -sh"),
]


class FakePsutil:
    # Mirrors psutil.PROC_ATTRS for the attributes orphans.py requests. psutil
    # has no "pgid" attribute and raises ValueError for unknown attrs, so the
    # fake must too -- an early version of the detector asked psutil for
    # "pgid" and the lenient fake hid the bug until the first real call.
    ALLOWED_ATTRS = {"pid", "ppid", "name", "cmdline", "create_time", "status"}

    def __init__(self, monkeypatch, rows):
        class FakeProc:
            def __init__(self, info):
                self.info = info

        import psutil

        pgid_map = {row["pid"]: row["pgid"] for row in rows}

        def fake_process_iter(attrs=None, ad_value=None):
            for attr in attrs or ():
                if attr not in self.ALLOWED_ATTRS:
                    raise ValueError(f"invalid attr name {attr!r}")
            return [FakeProc(row) for row in rows]

        def fake_getpgid(pid):
            if pid not in pgid_map:
                raise ProcessLookupError(pid)
            return pgid_map[pid]

        monkeypatch.setattr(psutil, "process_iter", fake_process_iter)
        monkeypatch.setattr(orphans.os, "getpgid", fake_getpgid)
        monkeypatch.setattr("backend.paths.studio_data_dir", lambda: "/app/data")


@pytest.fixture()
def fake_psutil(monkeypatch):
    FakePsutil(monkeypatch, PS_ROWS)
    return PS_ROWS


def test_detect_finds_reparented_engine_tree(fake_psutil):
    report = orphans.detect_engine_orphans()
    pids = {row["pid"] for row in report["orphans"]}
    assert pids == {100, 101, 102, 103}
    assert report["count"] == 4
    assert report["zombie_orphans"] == 1
    launcher = next(row for row in report["orphans"] if row["pid"] == 100)
    assert launcher["pgid"] == 100
    assert launcher["age_seconds"] is not None


def test_detect_ignores_audio_supervised_and_unrelated(fake_psutil):
    report = orphans.detect_engine_orphans()
    pids = {row["pid"] for row in report["orphans"]}
    assert 200 not in pids  # intentional detached audio sidecar
    assert 300 not in pids  # still supervised (ppid 85) even though pgid matches
    assert 500 not in pids  # unrelated python
    assert 600 not in pids


class FakeSignals:
    """Simulates a process group: group SIGTERMs only kill the leader."""

    def __init__(self, monkeypatch, members):
        self.pg = next(iter(members.keys()))
        self.members = set(members[self.pg])
        self.live = set(self.members)
        self.leader = self.pg
        self.calls = []
        monkeypatch.setattr(orphans, "_os_kill", self._kill)
        monkeypatch.setattr(orphans, "_os_killpg", self._killpg)

    def _kill(self, pid, sig):
        self.calls.append(("pid", pid, sig))
        if sig == 0:
            if pid not in self.live:
                raise ProcessLookupError
            return
        self.live.discard(pid)

    def _killpg(self, pgid, sig):
        self.calls.append(("pgid", pgid, sig))
        if pgid != self.pg:
            raise ProcessLookupError
        if sig == signal.SIGKILL:
            self.live.difference_update(self.members)
        elif sig == signal.SIGTERM:
            # Real kernels signal every member; emulate the worst case where
            # only the leader dies and workers must be escalated.
            self.live.discard(self.leader)


def test_sweep_dry_run_signals_nothing(fake_psutil, monkeypatch):
    fake = FakeSignals(monkeypatch, {100: {100, 101, 102, 103}})
    report = orphans.sweep_engine_orphans(dry_run=True)
    assert fake.calls == []
    assert len(report["candidates"]) == 4
    assert report["killed"] == []


def test_sweep_escalates_sigterm_then_sigkill(fake_psutil, monkeypatch):
    fake = FakeSignals(monkeypatch, {100: {100, 101, 102, 103}})
    report = orphans.sweep_engine_orphans(term_grace=0.05)
    assert ("pgid", 100, signal.SIGTERM) in fake.calls
    assert ("pgid", 100, signal.SIGKILL) in fake.calls
    assert fake.live == set()
    assert {row["pid"] for row in report["killed"]} == {100, 101, 102, 103}
    assert report["survivors"] == []


def test_sweep_never_signals_foreign_group_leaders(monkeypatch):
    # Only non-leader members reparented: killpg could hit the still-alive
    # group leader elsewhere, so the sweep must fall back to per-pid signals.
    rows = [row for row in PS_ROWS if row["pid"] in (101, 103)]
    FakePsutil(monkeypatch, rows)

    live = {101, 103}

    def killpg(pgid, sig):
        raise AssertionError("killpg must not be used when the group leader is missing")

    def kill(pid, sig):
        if sig == 0:
            if pid not in live:
                raise ProcessLookupError
            return
        live.discard(pid)

    monkeypatch.setattr(orphans, "_os_kill", kill)
    monkeypatch.setattr(orphans, "_os_killpg", killpg)
    report = orphans.sweep_engine_orphans(term_grace=0.05)
    assert live == set()
    assert len(report["killed"]) == 2


def test_sweep_reports_survivors(fake_psutil, monkeypatch):
    def kill(pgid, sig):
        pass  # nothing dies

    def _kill(pid, sig):
        if sig == 0:
            return
        # per-pid fallback also ineffective

    monkeypatch.setattr(orphans, "_os_kill", _kill)
    monkeypatch.setattr(orphans, "_os_killpg", kill)
    report = orphans.sweep_engine_orphans(term_grace=0.0, kill_grace=0.0)
    assert sorted(report["survivors"]) == [100, 101, 102, 103]
    assert report["killed"] == []


def test_bash_wrapper_uses_exec_after_cd():
    from backend.proxy.llama_swap.config import _render_bash_command

    rendered = _render_bash_command([VENV_PY, "-m", "sglang.launch_server"], cwd="/venv")
    assert rendered.startswith("bash -c ")
    inner = shlex.split(rendered)[2]
    assert inner.startswith("cd /venv && exec ")
    assert shlex.split(inner)[4:] == [VENV_PY, "-m", "sglang.launch_server"]

    inner = shlex.split(_render_bash_command(["llama-server", "--port", "${PORT}"]))[2]
    assert inner == "exec llama-server --port ${PORT}"


def test_bash_wrapper_inner_command_is_valid_bash(tmp_path):
    import subprocess

    from backend.proxy.llama_swap.config import _render_bash_command

    marker = "exec-ok"
    cmd = _render_bash_command(["printf", marker], cwd=str(tmp_path))
    # Run the rendered wrapper exactly as llama-swap would (it splits argv itself).
    result = subprocess.run(shlex.split(cmd), capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout == marker


def test_audio_cpp_cmd_renders_single_exec(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "backend.engines.audio_cpp.manager.get_audio_cpp_manager",
        lambda: type("M", (), {"server_configs_dir": str(tmp_path)})(),
    )
    from backend.proxy.llama_swap.config import _audio_cpp_swap_cmd

    cmd = _audio_cpp_swap_cmd(
        {"cmd_argv": ["/bin/audiocpp_server", "--config", "x.json"], "cmd_cwd": ""}
    )
    inner = shlex.split(cmd)[2]
    assert inner.count("exec") == 1
    assert inner == "exec /bin/audiocpp_server --config x.json"


def test_manager_sweep_and_report_spare_audio_sidecars(tmp_path, monkeypatch):
    from backend.proxy.llama_swap.manager import LlamaSwapManager

    manager = LlamaSwapManager(proxy_port=1, config_path=str(tmp_path / "c.yaml"))
    monkeypatch.setattr(
        "backend.proxy.llama_swap.manager.detect_engine_orphans",
        lambda: {
            "supported": True,
            "count": 2,
            "zombie_orphans": 0,
            "orphans": [
                {"pid": 7, "pgid": 7, "name": "audiocpp_server", "cmdline": "x", "age_seconds": 1},
                {"pid": 8, "pgid": 8, "name": "python", "cmdline": "y", "age_seconds": 1},
            ],
        },
    )
    monkeypatch.setattr(LlamaSwapManager, "_audio_sidecar_pids", staticmethod(lambda: {7}))
    report = asyncio.run(manager.engine_orphan_report())
    assert [row["pid"] for row in report["orphans"]] == [8]
    assert report["count"] == 1

    swept = {}

    def fake_sweep(*, dry_run=False):
        swept["dry_run"] = dry_run
        return {
            "killed": [{"pid": 8, "name": "python", "pgid": 8, "age_seconds": 1}],
            "survivors": [],
        }

    monkeypatch.setattr("backend.proxy.llama_swap.manager.sweep_engine_orphans", fake_sweep)
    result = asyncio.run(manager._sweep_engine_orphans(label="test"))
    assert [row["pid"] for row in result["killed"]] == [8]
    assert swept["dry_run"] is False
