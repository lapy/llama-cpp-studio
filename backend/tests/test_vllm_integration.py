"""Contracts for first-class vanilla vLLM support."""

import stat

import backend.proxy.commands as runtime_commands
from backend.engines.registry import ENGINE_REGISTRY
from backend.models.schema import compatible_engines_for_record
from backend.venv_install_settings import default_install_settings


def test_vllm_registry_and_defaults():
    spec = ENGINE_REGISTRY["vllm"]
    assert spec.runtime_kind == "vllm"
    assert spec.scanner_kind == "vllm_help"
    assert "vllm" in compatible_engines_for_record({"format": "safetensors"})
    settings = default_install_settings("vllm")
    assert settings["source_repo"] == "https://github.com/vllm-project/vllm.git"
    assert settings["source_branch"] == "main"


# NOTE: the former `test_vllm_source_install_targets_checkout_root` drove the
# legacy _install_source_checkout helper directly. That helper was removed in
# the Phase 3 cutover (its argv/cwd/env contract — editable install of the
# checkout root inside the CUDA+ccache env — is now executed by the shared
# pip.install_package primitive and pinned end-to-end by
# test_vllm_legacy_plan_shadow.py).


def test_vllm_preview_uses_openai_server(monkeypatch, tmp_path):
    # Phase 6: the preview compiles through the exported profile, so the
    # active install row + Studio-managed CUDA env drive resolution (the
    # legacy resolver symbols are no longer consulted).
    install = tmp_path / "vllm"
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    python_bin = bin_dir / "python"
    python_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    python_bin.chmod(python_bin.stat().st_mode | stat.S_IXUSR)

    import backend.cuda_installer as cuda_installer
    import backend.data_store as data_store

    class _Store:
        def get_active_engine_version(self, engine_id):
            return {"version": "v1", "venv_path": str(install)}

    class _Cuda:
        def get_cuda_env(self, version):
            return {"CUDA_HOME": "/studio/cuda/current"}

    monkeypatch.setattr(data_store, "get_store", lambda: _Store())
    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: _Cuda())
    monkeypatch.setattr(runtime_commands, "_active_engine_param_index", lambda engine: {})
    preview = runtime_commands.preview_model_command(
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
    assert "CUDA_HOME=/studio/cuda/current" in preview["env"]


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
