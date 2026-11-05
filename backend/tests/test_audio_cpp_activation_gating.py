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
        "config": {"engine": "audio_cpp", "engines": {"audio_cpp": {"family": family, "task": task}}},
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
    delta = {"added_families": ["omnivoice"], "removed_families": [], "added_tasks": [], "removed_tasks": []}
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

    monkeypatch.setattr(
        "backend.engines.scan.scanner.scan_audio_cpp_model_profile", fake_profile
    )
    store = FakeStore([_model(f"m{i}", "qwen3_tts") for i in range(8)])
    plan = [(m, True) for m in store.list_models()]
    results = await act_mod.rescan_model_profiles(store, _row(tmp_path), plan)
    assert len(results) == 8
    assert all(row["ok"] for row in results)
    assert active["peak"] <= act_mod.RESCAN_CONCURRENCY
