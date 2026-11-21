"""Activation must rescan only the profiles that actually changed."""

from __future__ import annotations

import pytest

from backend.engines.audio_cpp.activation import plan_profile_rescans


class FakeStore:
    def __init__(self, models):
        self._models = models

    def list_models(self):
        return self._models


def _model(model_id: str, family: str, task: str = "tts") -> dict:
    return {
        "id": model_id,
        "compatible_engines": ["audio_cpp"],
        "config": {
            "engine": "audio_cpp",
            "engines": {"audio_cpp": {"family": family, "task": task}},
        },
    }


def _row(tmp_path, **extra):
    server = tmp_path / "audiocpp_server"
    cli = tmp_path / "audiocpp_cli"
    server.write_text("x")
    cli.write_text("x")
    return {
        "version": "v1",
        "server_binary_path": str(server),
        "cli_binary_path": str(cli),
        **extra,
    }


def test_contract_changed_forces_every_model(tmp_path):
    store = FakeStore([_model("a", "qwen3_tts"), _model("b", "omnivoice")])
    plan = plan_profile_rescans(store, _row(tmp_path), {}, contract_changed=True)
    assert {(m["id"], force) for m, force in plan} == {("a", True), ("b", True)}


def test_only_models_intersecting_delta_are_rescanned(tmp_path, monkeypatch):
    store = FakeStore([_model("a", "qwen3_tts"), _model("b", "omnivoice", task="asr")])
    monkeypatch.setattr(
        "backend.engines.audio_cpp.activation._profile_cache_is_fresh",
        lambda *a, **k: True,
    )
    delta = {
        "added_families": ["omnivoice"],
        "removed_families": [],
        "added_tasks": [],
        "removed_tasks": [],
    }
    plan = plan_profile_rescans(store, _row(tmp_path), delta, contract_changed=False)
    assert [(m["id"], force) for m, force in plan] == [("b", True)]


def test_cold_cache_fills_without_force(tmp_path, monkeypatch):
    store = FakeStore([_model("a", "qwen3_tts")])
    monkeypatch.setattr(
        "backend.engines.audio_cpp.activation._profile_cache_is_fresh",
        lambda *a, **k: False,
    )
    plan = plan_profile_rescans(store, _row(tmp_path), {}, contract_changed=False)
    assert [(m["id"], force) for m, force in plan] == [("a", False)]


def test_fresh_cache_and_no_delta_skips_every_model(tmp_path, monkeypatch):
    store = FakeStore([_model("a", "qwen3_tts"), _model("b", "omnivoice")])
    monkeypatch.setattr(
        "backend.engines.audio_cpp.activation._profile_cache_is_fresh",
        lambda *a, **k: True,
    )
    assert plan_profile_rescans(store, _row(tmp_path), {}, contract_changed=False) == []


def test_non_audio_models_are_ignored(tmp_path, monkeypatch):
    store = FakeStore([{"id": "llama-1", "compatible_engines": ["llama_cpp"]}])
    assert plan_profile_rescans(store, _row(tmp_path), {}, contract_changed=True) == []


@pytest.mark.asyncio
async def test_rescan_runs_with_bounded_concurrency(tmp_path, monkeypatch):
    import asyncio

    from backend.engines.audio_cpp import activation as act_mod

    active = {"current": 0, "peak": 0}
    barrier = asyncio.Event()

    def fake_profile(store_obj, row, model, force=False):
        active["current"] += 1
        active["peak"] = max(active["peak"], active["current"])
        try:
            import time

            time.sleep(0.02)
        finally:
            active["current"] -= 1
        return {"fingerprint": f"fp-{model.get('id')}", "scan_error": None}

    monkeypatch.setattr("backend.engines.scan.scanner.scan_audio_cpp_model_profile", fake_profile)
    store = FakeStore([_model(f"m{i}", "qwen3_tts") for i in range(8)])
    plan = [(m, True) for m in store.list_models()]
    results = await act_mod.rescan_model_profiles(store, _row(tmp_path), plan)
    assert len(results) == 8
    assert all(row["ok"] for row in results)
    assert active["peak"] <= act_mod.RESCAN_CONCURRENCY


# --- activation gate: old build vs unloadable binary -------------------------


def _version_store(tmp_path):
    from backend import data_store

    return data_store.DataStore(config_dir=str(tmp_path / "config"))


def _script(path, body: str):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _registered_row(store, cli, version="v1"):
    row = {
        "version": version,
        "cli_binary_path": str(cli),
        "server_binary_path": str(cli),
    }
    store.add_engine_version("audio_cpp", dict(row))
    return row


def test_activation_advises_an_update_for_an_old_but_runnable_build(tmp_path):
    from fastapi import HTTPException

    from backend.engines.audio_cpp.activation import assert_activation_supported

    store = _version_store(tmp_path)
    cli = _script(tmp_path / "audiocpp_cli", 'echo "Usage: audiocpp_cli [options]"\n')
    row = _registered_row(store, cli)

    with pytest.raises(HTTPException) as excinfo:
        assert_activation_supported(store, row)

    detail = str(excinfo.value.detail)
    assert "--list-loaders --json" in detail
    assert "v0.9.1" in detail
    assert "Update audio.cpp" in detail
    assert "cannot run on this host" not in detail


def test_activation_reports_a_binary_that_cannot_execute(tmp_path):
    from fastapi import HTTPException

    from backend.engines.audio_cpp.activation import assert_activation_supported

    store = _version_store(tmp_path)
    cli = _script(
        tmp_path / "audiocpp_cli",
        'echo "$0: /lib/x86_64-linux-gnu/libc.so.6: version GLIBC_2.43 not found" >&2\nexit 1\n',
    )
    row = _registered_row(store, cli)

    with pytest.raises(HTTPException) as excinfo:
        assert_activation_supported(store, row)

    detail = str(excinfo.value.detail)
    assert "cannot run on this host" in detail
    assert "GLIBC_2.43" in detail
    assert "host glibc:" in detail
    assert "build this release from source" in detail
    assert "Update audio.cpp" not in detail


def test_loader_probe_failure_carries_the_loader_excerpt(tmp_path):
    from backend.engines.scan.scanner import (
        _run_audio_cpp_loaders,
        audio_cpp_probe_failure_is_execution,
    )

    cli = _script(
        tmp_path / "audiocpp_cli",
        'echo "$0: /lib/x86_64-linux-gnu/libc.so.6: version GLIBC_2.43 not found" >&2\nexit 1\n',
    )

    payload, error = _run_audio_cpp_loaders(str(cli))

    assert payload is None
    assert "process exited with code 1" in error
    assert "GLIBC_2.43 not found" in error
    assert audio_cpp_probe_failure_is_execution(error) is True


def test_loader_probe_failure_for_an_old_build_is_not_an_exec_failure(tmp_path):
    from backend.engines.scan.scanner import (
        _run_audio_cpp_loaders,
        audio_cpp_probe_failure_is_execution,
    )

    cli = _script(tmp_path / "audiocpp_cli", 'echo "Usage: audiocpp_cli [options]"\n')

    payload, error = _run_audio_cpp_loaders(str(cli))

    assert payload is None
    assert "Usage: audiocpp_cli" in error
    assert audio_cpp_probe_failure_is_execution(error) is False
