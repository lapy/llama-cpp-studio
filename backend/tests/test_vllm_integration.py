"""Contracts for first-class vanilla vLLM support."""

from pathlib import Path

import pytest

import backend.llama_swap_config as llama_swap_config
from backend.engine_registry import ENGINE_REGISTRY
from backend.model_schema import compatible_engines_for_record
from backend.venv_install_settings import default_install_settings
from backend.vllm_manager import VllmManager


def test_vllm_registry_and_defaults():
    spec = ENGINE_REGISTRY["vllm"]
    assert spec.runtime_kind == "vllm"
    assert spec.scanner_kind == "vllm_help"
    assert "vllm" in compatible_engines_for_record({"format": "safetensors"})
    settings = default_install_settings("vllm")
    assert settings["source_repo"] == "https://github.com/vllm-project/vllm.git"
    assert settings["source_branch"] == "main"


@pytest.mark.asyncio
async def test_vllm_source_install_targets_checkout_root(tmp_path, monkeypatch):
    manager = VllmManager(
        base_dir=str(tmp_path / "vllm"), log_path=str(tmp_path / "vllm.log")
    )
    manager._prepare_versioned_paths("source")
    checkout = Path(manager._base_dir) / "source"
    checkout.mkdir(parents=True)
    calls = []

    async def fake_pip(args, operation, **kwargs):
        calls.append((args, operation, kwargs))
        return 0

    monkeypatch.setattr(manager, "_run_pip", fake_pip)
    env = {"CUDA_HOME": "/studio/cuda/current"}
    await manager._install_source_checkout(str(checkout), env)

    assert calls[1][0] == ["install", "-v", "-e", str(checkout)]
    assert calls[1][2]["cwd"] == str(checkout)
    assert calls[1][2]["env"] == env


def test_vllm_preview_uses_openai_server(monkeypatch):
    monkeypatch.setattr(
        llama_swap_config, "_resolve_sglang_bin", lambda engine: "/opt/vllm/bin/python"
    )
    monkeypatch.setattr(llama_swap_config, "_active_engine_param_index", lambda engine: {})
    monkeypatch.setattr(
        llama_swap_config,
        "_resolve_sglang_cuda_env",
        lambda engine: {"CUDA_HOME": "/studio/cuda/current"},
    )
    preview = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "id": "model",
            "huggingface_id": "org/model",
            "format": "safetensors",
            "config": {"engine": "vllm", "engines": {"vllm": {}}},
        }
    )
    assert preview["ok"] is True
    assert "-m vllm.entrypoints.openai.api_server" in preview["cmd"]
    assert "--model org/model" in preview["cmd"]
    assert "--port ${PORT}" in preview["cmd"]
    assert preview["env"] == ["CUDA_HOME=/studio/cuda/current"]


def test_vllm_routes_and_settings(client, monkeypatch, tmp_path):
    from backend import data_store

    store = data_store.DataStore(config_dir=str(tmp_path / "config"))
    monkeypatch.setattr(data_store, "_store", store)
    response = client.get("/api/vllm/build-settings")
    assert response.status_code == 200
    assert response.json()["source_branch"] == "main"
    response = client.put(
        "/api/vllm/build-settings",
        json={"source_repo": "https://github.com/example/vllm.git", "source_branch": "dev"},
    )
    assert response.status_code == 200
    assert store.get_engine_build_settings("vllm")["source_branch"] == "dev"
