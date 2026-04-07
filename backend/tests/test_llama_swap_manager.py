"""llama-swap manager apply-path regressions."""

import asyncio
from pathlib import Path

import backend.llama_swap_manager as llama_swap_manager


def test_flag_pair_helpers_and_json_norm():
    pairs = llama_swap_manager._flag_argv_to_pairs(
        ["--ctx-size", "4096", "plain", "--jinja", "--top-k", "40"]
    )
    assert pairs == [("--ctx-size", "4096"), ("--jinja", None), ("--top-k", "40")]
    assert llama_swap_manager._pairs_to_argv(pairs) == [
        "--ctx-size",
        "4096",
        "--jinja",
        "--top-k",
        "40",
    ]

    class Unserializable:
        pass

    assert "Unserializable" in llama_swap_manager._json_norm(Unserializable())


def test_normalize_bash_command_handles_no_marker_and_invalid_tail():
    plain = "python -m something"
    assert llama_swap_manager._normalize_bash_c_cmd_after_port_marker(plain) == plain

    invalid = "bash -c 'cmd --port ${PORT} \"unterminated'"
    assert llama_swap_manager._normalize_bash_c_cmd_after_port_marker(invalid) == invalid


def test_norm_config_text_and_diff_summary_limit():
    assert llama_swap_manager._norm_config_text("a\r\nb\r\n") == "a\nb"
    assert llama_swap_manager._norm_config_text("") == ""

    disk = {"models": {f"m{i}": {"cmd": "old"} for i in range(20)}}
    desired = {"models": {f"m{i}": {"cmd": "new"} for i in range(20)}}
    lines = llama_swap_manager.summarize_llama_swap_yaml_diff(
        llama_swap_manager.yaml.safe_dump(disk),
        llama_swap_manager.yaml.safe_dump(desired),
    )
    assert lines[-1].startswith("…and ")


def test_get_swap_config_stale_state_respects_applicability(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")

    class FakeStore:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {"version": "v1", "binary_path": str(binary)}
            return None

    monkeypatch.setattr("backend.data_store.get_store", lambda: FakeStore())

    manager.mark_swap_config_stale()
    assert manager.get_swap_config_stale_state() == {"applicable": True, "stale": True}

    binary.unlink()
    assert manager.get_swap_config_stale_state() == {"applicable": False, "stale": False}


def test_ensure_config_file_for_proxy_writes_stub_once(tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "llama-swap.yaml"))

    asyncio.run(manager._ensure_config_file_for_proxy())
    content = Path(manager.config_path).read_text(encoding="utf-8")
    assert "models: {}" in content

    Path(manager.config_path).write_text("custom: true\n", encoding="utf-8")
    asyncio.run(manager._ensure_config_file_for_proxy())
    assert Path(manager.config_path).read_text(encoding="utf-8") == "custom: true\n"


def test_get_config_pending_state_clears_stale_when_equal(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    Path(manager.config_path).write_text("models: {}\n", encoding="utf-8")
    manager.mark_swap_config_stale()

    async def fake_compute():
        return "models: {}\n"

    monkeypatch.setattr(manager, "compute_desired_config_content", fake_compute)

    state = asyncio.run(manager.get_config_pending_state())
    assert state == {"applicable": True, "pending": False, "changes": []}
    assert manager.get_swap_config_stale_state()["stale"] is False


def test_get_config_pending_state_returns_reason_on_compute_error(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    async def boom():
        raise ValueError("broken")

    monkeypatch.setattr(manager, "compute_desired_config_content", boom)
    state = asyncio.run(manager.get_config_pending_state())
    assert state["applicable"] is False
    assert "broken" in state["reason"]


def test_write_config_writes_yaml_and_clears_stale(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))
    manager.running_models = {"proxy": {"config": {}}}
    manager.mark_swap_config_stale()

    class Store:
        def list_models(self):
            return [{"id": "m1"}]

        def get_active_engine_version(self, engine):
            return None

    monkeypatch.setattr(llama_swap_manager, "get_store", lambda: Store())
    monkeypatch.setattr("backend.llama_swap_config.any_active_gguf_runtime_in_db", lambda: True)
    monkeypatch.setattr(
        llama_swap_manager,
        "generate_llama_swap_config",
        lambda running_models, all_models=None: "models: {}\n",
    )

    asyncio.run(manager._write_config())

    assert Path(manager.config_path).read_text(encoding="utf-8") == "models: {}\n"
    assert manager.get_swap_config_stale_state()["stale"] is False


def test_sync_running_models_updates_and_handles_failures(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    class FakeClient:
        async def get_running_models(self):
            return {"running": [{"model": "proxy-a", "state": "ready"}]}

    monkeypatch.setattr("backend.llama_swap_client.LlamaSwapClient", FakeClient)
    asyncio.run(manager.sync_running_models())
    assert "proxy-a" in manager.running_models

    class BrokenClient:
        async def get_running_models(self):
            raise RuntimeError("down")

    monkeypatch.setattr("backend.llama_swap_client.LlamaSwapClient", BrokenClient)
    asyncio.run(manager.sync_running_models())
    assert "proxy-a" in manager.running_models


def test_compute_desired_config_content_handles_missing_and_present_active_binary(monkeypatch, tmp_path):
    manager = llama_swap_manager.LlamaSwapManager(config_path=str(tmp_path / "swap.yaml"))

    async def noop():
        return None

    monkeypatch.setattr(manager, "_ensure_correct_binary_path", noop)

    class NoActiveStore:
        def get_active_engine_version(self, engine):
            return None

    monkeypatch.setattr("backend.data_store.get_store", lambda: NoActiveStore())
    assert asyncio.run(manager.compute_desired_config_content()) is None

    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")

    class ActiveStore:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {"binary_path": str(binary)}
            return None

        def list_models(self):
            return [{"id": "m1"}]

    async def fake_sync():
        manager.running_models = {"proxy": {"config": {}}}

    monkeypatch.setattr("backend.data_store.get_store", lambda: ActiveStore())
    monkeypatch.setattr(manager, "sync_running_models", fake_sync)
    monkeypatch.setattr(
        llama_swap_manager,
        "generate_llama_swap_config",
        lambda running_models, all_models=None: "models: {}\n",
    )

    assert asyncio.run(manager.compute_desired_config_content()) == "models: {}\n"


def test_user_apply_regenerate_config_skips_post_unload_sync(monkeypatch):
    import backend.llama_swap_client as llama_swap_client
    from backend.llama_swap_manager import LlamaSwapManager

    manager = LlamaSwapManager(config_path="/tmp/llama-swap-test.yaml")
    manager.running_models = {"already-loaded": {"config": {}}}
    observed = {}

    class FakeClient:
        async def unload_all_models(self):
            observed["unloaded"] = True

    async def fake_regenerate(*, sync_running=True):
        observed["sync_running"] = sync_running

    monkeypatch.setattr(llama_swap_client, "LlamaSwapClient", FakeClient)
    monkeypatch.setattr(manager, "regenerate_config_with_active_version", fake_regenerate)

    asyncio.run(manager.user_apply_regenerate_config())

    assert observed == {
        "unloaded": True,
        "sync_running": False,
    }
    assert manager.running_models == {}
