"""Lifecycle-focused coverage for backend.llama_swap_manager."""

import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import backend.llama_swap_manager as llama_swap_manager


class FakeStdout:
    def __init__(self, lines):
        self._lines = iter(lines)

    def readline(self):
        return next(self._lines, "")


class FakeProcess:
    def __init__(self, polls=None, stdout=None):
        self._polls = iter(polls or [None])
        self._last_poll = None
        self.stdout = stdout
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def poll(self):
        try:
            self._last_poll = next(self._polls)
        except StopIteration:
            pass
        return self._last_poll

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self):
        self.wait_calls += 1
        return 0


class FakeTask:
    def __init__(self, coro=None, *, done=False, raise_cancelled=False, auto_close=True):
        self._done = done
        self._raise_cancelled = raise_cancelled
        self._coro = coro
        self.cancel_called = False
        if auto_close and coro is not None:
            coro.close()

    def done(self):
        return self._done

    def cancel(self):
        self.cancel_called = True
        if self._coro is not None:
            self._coro.close()

    def __await__(self):
        async def _wait():
            self._done = True
            if self._raise_cancelled:
                raise asyncio.CancelledError
            return None

        return _wait().__await__()


def _async_noop(*args, **kwargs):
    async def _run():
        return None

    return _run()


def test_start_proxy_returns_early_when_process_is_running(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = FakeProcess([None])
    observed = []

    async def unexpected():
        observed.append("called")

    monkeypatch.setattr(manager, "_do_start_proxy", unexpected)
    monkeypatch.setattr(manager, "_wait_for_proxy_ready", unexpected)

    asyncio.run(manager.start_proxy())

    assert observed == []


def test_start_proxy_starts_process_waits_ready_and_creates_monitor_once(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    observed = []
    created = []

    async def fake_start():
        observed.append("start")
        manager.process = FakeProcess([None])

    async def fake_wait():
        observed.append("wait")

    def fake_create_task(coro):
        created.append(coro)
        return FakeTask(coro)

    monkeypatch.setattr(manager, "_do_start_proxy", fake_start)
    monkeypatch.setattr(manager, "_wait_for_proxy_ready", fake_wait)
    monkeypatch.setattr(llama_swap_manager.asyncio, "create_task", fake_create_task)

    asyncio.run(manager.start_proxy())
    asyncio.run(manager.start_proxy())

    assert observed == ["start", "wait"]
    assert len(created) == 1
    assert manager.monitor_task is not None


def test_do_start_proxy_merges_cuda_env_and_launches_process(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(
        proxy_port=2345, config_path=str(tmp_path / "swap.yaml")
    )
    launched = {}
    created = []
    fake_process = FakeProcess([None], stdout=FakeStdout([]))

    async def fake_ensure():
        launched["ensured"] = True

    class FakeInstaller:
        def get_cuda_env(self):
            return {"CUDA_VISIBLE_DEVICES": "0"}

    def fake_popen(cmd, **kwargs):
        launched["cmd"] = cmd
        launched["kwargs"] = kwargs
        return fake_process

    def fake_create_task(coro):
        created.append(coro)
        return FakeTask(coro)

    monkeypatch.setattr(manager, "_ensure_config_file_for_proxy", fake_ensure)
    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer", lambda: FakeInstaller()
    )
    monkeypatch.setattr(llama_swap_manager.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(llama_swap_manager.os.path, "isdir", lambda path: False)
    monkeypatch.setattr(llama_swap_manager.asyncio, "create_task", fake_create_task)

    asyncio.run(manager._do_start_proxy())

    assert launched["ensured"] is True
    assert launched["cmd"] == [
        "llama-swap",
        "--config",
        manager.config_path,
        "--listen",
        "0.0.0.0:2345",
        "--watch-config",
    ]
    assert launched["kwargs"]["cwd"] is None
    assert launched["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert manager.process is fake_process
    assert len(created) == 1


def test_do_start_proxy_continues_when_cuda_env_lookup_fails(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    launched = {}

    def fake_popen(cmd, **kwargs):
        launched["env"] = kwargs["env"]
        return FakeProcess([None], stdout=FakeStdout([]))

    monkeypatch.setattr(manager, "_ensure_config_file_for_proxy", _async_noop)
    monkeypatch.setattr(
        "backend.cuda_installer.get_cuda_installer",
        lambda: (_ for _ in ()).throw(RuntimeError("no cuda")),
    )
    monkeypatch.setattr(llama_swap_manager.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(
        llama_swap_manager.asyncio, "create_task", lambda coro: FakeTask(coro)
    )

    asyncio.run(manager._do_start_proxy())

    assert isinstance(launched["env"], dict)
    assert manager.process is not None


def test_stream_llama_swap_logs_returns_without_process(tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    asyncio.run(manager._stream_llama_swap_logs())


def test_stream_llama_swap_logs_schedules_and_reads_output(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = FakeProcess(
        [None, None, None, 0],
        stdout=FakeStdout([" first line\n", "\n", "second line\n", ""]),
    )
    scheduled = []

    def fake_create_task(coro):
        scheduled.append(coro)
        return FakeTask(coro, auto_close=False)

    monkeypatch.setattr(llama_swap_manager.asyncio, "create_task", fake_create_task)

    asyncio.run(manager._stream_llama_swap_logs())
    assert len(scheduled) == 1

    asyncio.run(scheduled.pop())


def test_monitor_process_restarts_dead_proxy(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = FakeProcess([7])
    observed = []

    async def fake_start():
        observed.append("start")
        manager.process = FakeProcess([None])

    async def fake_wait():
        observed.append("ready")

    async def fake_sleep(delay):
        observed.append(("sleep", delay))
        if delay == 2:
            manager._should_restart = False

    monkeypatch.setattr(manager, "_do_start_proxy", fake_start)
    monkeypatch.setattr(manager, "_wait_for_proxy_ready", fake_wait)
    monkeypatch.setattr(llama_swap_manager.asyncio, "sleep", fake_sleep)

    asyncio.run(manager._monitor_process())

    assert observed == ["start", "ready", ("sleep", 2)]


def test_monitor_process_waits_before_retry_when_restart_fails(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = FakeProcess([9])
    observed = []

    async def broken_start():
        observed.append("start")
        raise RuntimeError("boom")

    async def fake_sleep(delay):
        observed.append(("sleep", delay))
        if delay == 2:
            manager._should_restart = False

    monkeypatch.setattr(manager, "_do_start_proxy", broken_start)
    monkeypatch.setattr(llama_swap_manager.asyncio, "sleep", fake_sleep)

    asyncio.run(manager._monitor_process())

    assert observed == ["start", ("sleep", 5), ("sleep", 2)]


def test_wait_for_proxy_ready_handles_connect_errors_then_succeeds(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    class FakeResponse:
        def __init__(self, status_code):
            self.status_code = status_code

    class FakeClient:
        def __init__(self):
            self.calls = 0

        async def get(self, url, timeout=1):
            self.calls += 1
            if self.calls == 1:
                raise httpx.ConnectError("down")
            return FakeResponse(200)

    class FakeLoop:
        def __init__(self, values):
            self._values = iter(values)

        def time(self):
            return next(self._values)

    loop = FakeLoop([0, 0, 0.2])
    monkeypatch.setattr(llama_swap_manager.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(llama_swap_manager.asyncio, "get_event_loop", lambda: loop)
    monkeypatch.setattr(llama_swap_manager.asyncio, "sleep", _async_noop)

    asyncio.run(manager._wait_for_proxy_ready(timeout=1))


def test_wait_for_proxy_ready_times_out(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    class FakeClient:
        async def get(self, url, timeout=1):
            raise httpx.ConnectError("down")

    class FakeLoop:
        def __init__(self, values):
            self._values = iter(values)

        def time(self):
            return next(self._values)

    loop = FakeLoop([0, 0, 0.6, 1.2])
    monkeypatch.setattr(llama_swap_manager.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(llama_swap_manager.asyncio, "get_event_loop", lambda: loop)
    monkeypatch.setattr(llama_swap_manager.asyncio, "sleep", _async_noop)

    with pytest.raises(Exception, match="did not become ready"):
        asyncio.run(manager._wait_for_proxy_ready(timeout=1))


def test_stop_proxy_handles_not_running_process(tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    asyncio.run(manager.stop_proxy())

    assert manager._should_restart is False
    assert manager.process is None


def test_stop_proxy_terminates_gracefully_and_clears_models(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = FakeProcess([None])
    manager.running_models = {"proxy": {"config": {}}}
    manager.monitor_task = FakeTask(done=False, raise_cancelled=True)

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(llama_swap_manager.asyncio, "to_thread", fake_to_thread)

    asyncio.run(manager.stop_proxy())

    assert manager.monitor_task.cancel_called is True
    assert manager.process is None
    assert manager.running_models == {}


def test_stop_proxy_kills_when_graceful_shutdown_times_out(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    process = FakeProcess([None])
    manager.process = process

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def fake_wait_for(awaitable, timeout):
        awaitable.close()
        raise asyncio.TimeoutError

    monkeypatch.setattr(llama_swap_manager.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(llama_swap_manager.asyncio, "wait_for", fake_wait_for)

    asyncio.run(manager.stop_proxy())

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 1


def test_restart_proxy_stops_running_process_before_starting_again(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.process = FakeProcess([None])
    observed = []

    async def fake_stop():
        observed.append(("stop", manager._should_restart))
        manager.process = None

    async def fake_start():
        observed.append(("start", manager._should_restart))

    monkeypatch.setattr(manager, "stop_proxy", fake_stop)
    monkeypatch.setattr(manager, "start_proxy", fake_start)

    asyncio.run(manager.restart_proxy())

    assert observed == [("stop", False), ("start", True)]


def test_register_model_accepts_dict_and_object_and_rejects_duplicates(tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    proxy = asyncio.run(
        manager.register_model(
            {"proxy_name": "proxy-a", "file_path": "/tmp/model.gguf", "display_name": "A"},
            {"ctx_size": 4096},
        )
    )
    assert proxy == "proxy-a"
    assert manager.running_models["proxy-a"]["model_path"] == "/tmp/model.gguf"

    class ObjModel:
        proxy_name = "proxy-b"
        file_path = "/tmp/other.gguf"
        name = "Other"
        display_name = None

    proxy_b = asyncio.run(manager.register_model(ObjModel(), {"threads": 8}))
    assert proxy_b == "proxy-b"

    with pytest.raises(ValueError, match="already registered"):
        asyncio.run(manager.register_model({"proxy_name": "proxy-a", "name": "dup"}, {}))

    with pytest.raises(ValueError, match="does not have a proxy_name"):
        asyncio.run(manager.register_model({"name": "missing"}, {}))


def test_detect_correct_binary_path_prefers_llama_server_and_has_fallback(tmp_path):
    version_dir = tmp_path / "version"
    build_bin = version_dir / "build" / "bin"
    build_bin.mkdir(parents=True)
    server_path = build_bin / "server"
    server_path.write_text("", encoding="utf-8")
    server_path.chmod(0o755)

    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    detected = manager._detect_correct_binary_path(str(version_dir))
    assert detected.endswith("server")

    llama_server_path = build_bin / "llama-server"
    llama_server_path.write_text("", encoding="utf-8")
    llama_server_path.chmod(0o755)
    detected_preferred = manager._detect_correct_binary_path(str(version_dir))
    assert detected_preferred.endswith("llama-server")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    fallback = manager._detect_correct_binary_path(str(empty_dir))
    assert fallback.endswith("build/bin/llama-server")


def test_ensure_correct_binary_path_updates_store_when_needed(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    saved = {}

    class FakeStore:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {"version": "v1", "binary_path": "versions/v1/server"}
            return None

        def _read_yaml(self, name):
            return {
                "llama_cpp": {
                    "versions": [{"version": "v1", "binary_path": "versions/v1/server"}]
                }
            }

        def _save_yaml(self, name, data):
            saved["name"] = name
            saved["data"] = data

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: FakeStore())
    monkeypatch.setattr(
        manager,
        "_detect_correct_binary_path",
        lambda version_dir: "/app/versions/v1/build/bin/llama-server",
    )

    asyncio.run(manager._ensure_correct_binary_path())

    assert saved["name"] == "engines.yaml"
    assert (
        saved["data"]["llama_cpp"]["versions"][0]["binary_path"]
        == "versions/v1/build/bin/llama-server"
    )


def test_ensure_correct_binary_path_noops_when_already_correct_or_no_active(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    class AlreadyCorrectStore:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {
                    "version": "v1",
                    "binary_path": "versions/v1/build/bin/llama-server",
                }
            return None

        def _read_yaml(self, name):  # pragma: no cover - should not be called
            raise AssertionError("unexpected read")

        def _save_yaml(self, name, data):  # pragma: no cover - should not be called
            raise AssertionError("unexpected save")

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: AlreadyCorrectStore())
    monkeypatch.setattr(
        manager,
        "_detect_correct_binary_path",
        lambda version_dir: "/app/versions/v1/build/bin/llama-server",
    )
    asyncio.run(manager._ensure_correct_binary_path())

    class NoActiveStore:
        def get_active_engine_version(self, engine):
            return None

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: NoActiveStore())
    asyncio.run(manager._ensure_correct_binary_path())


def test_regenerate_config_with_active_version_skips_missing_active_or_binary(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    monkeypatch.setattr(manager, "_ensure_correct_binary_path", _async_noop)

    class NoActiveStore:
        def get_active_engine_version(self, engine):
            return None

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: NoActiveStore())
    assert asyncio.run(manager.regenerate_config_with_active_version()) is None

    class MissingBinaryStore:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {"version": "v1", "binary_path": "versions/v1/build/bin/llama-server"}
            return None

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: MissingBinaryStore())
    assert asyncio.run(manager.regenerate_config_with_active_version()) is None


def test_regenerate_config_with_active_version_syncs_writes_and_tolerates_start_failures(
    monkeypatch, tmp_path
):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")
    observed = []

    class ActiveStore:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {"version": "v1", "binary_path": str(binary)}
            return None

    async def fake_sync():
        observed.append("sync")

    async def fake_write(path):
        observed.append(("write", path))

    async def broken_start():
        observed.append("start")
        raise RuntimeError("proxy unavailable")

    monkeypatch.setattr(manager, "_ensure_correct_binary_path", _async_noop)
    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: ActiveStore())
    monkeypatch.setattr(manager, "sync_running_models", fake_sync)
    monkeypatch.setattr(manager, "_write_config", fake_write)
    monkeypatch.setattr(manager, "start_proxy", broken_start)

    asyncio.run(manager.regenerate_config_with_active_version())
    asyncio.run(manager.regenerate_config_with_active_version(sync_running=False))

    assert observed == [
        "sync",
        ("write", str(binary)),
        "start",
        ("write", str(binary)),
        "start",
    ]


def test_unregister_model_success_and_external_model_paths(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.running_models = {"proxy-a": {"config": {}}}
    unloaded = []
    synced = []

    class FakeClient:
        async def unload_model(self, proxy_name):
            unloaded.append(proxy_name)
            return {"ok": True}

    async def fake_sync():
        synced.append("sync")

    monkeypatch.setattr("backend.llama_swap_client.LlamaSwapClient", FakeClient)
    monkeypatch.setattr(manager, "sync_running_models", fake_sync)

    asyncio.run(manager.unregister_model("proxy-a"))
    assert unloaded == ["proxy-a"]
    assert synced == ["sync"]
    assert "proxy-a" not in manager.running_models

    asyncio.run(manager.unregister_model("externally-loaded"))
    assert unloaded == ["proxy-a", "externally-loaded"]
    assert synced == ["sync", "sync"]


def test_unregister_model_raises_when_unload_fails(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    class BrokenClient:
        async def unload_model(self, proxy_name):
            raise RuntimeError("nope")

    monkeypatch.setattr("backend.llama_swap_client.LlamaSwapClient", BrokenClient)

    with pytest.raises(RuntimeError, match="nope"):
        asyncio.run(manager.unregister_model("broken"))


def test_user_apply_regenerate_config_continues_when_unload_all_fails(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.running_models = {"proxy": {"config": {}}}
    observed = {}

    class BrokenClient:
        async def unload_all_models(self):
            observed["unload_attempted"] = True
            raise RuntimeError("down")

    async def fake_regenerate(*, sync_running=True):
        observed["sync_running"] = sync_running

    monkeypatch.setattr("backend.llama_swap_client.LlamaSwapClient", BrokenClient)
    monkeypatch.setattr(manager, "regenerate_config_with_active_version", fake_regenerate)

    asyncio.run(manager.user_apply_regenerate_config())

    assert observed == {"unload_attempted": True, "sync_running": False}
    assert manager.running_models == {}
