"""audio.cpp sidecars and capability-driven llama-swap runtime dispatch."""

from pathlib import Path
from types import SimpleNamespace

import yaml

import backend.audio_cpp_runtime as audio_runtime
import backend.llama_swap_config as swap_config
from backend.llama_swap_manager import LlamaSwapManager


class _Store:
    def __init__(self, active):
        self.active = active

    def get_active_engine_version(self, engine):
        return self.active if engine == "audio_cpp" else None


def _fixture(tmp_path):
    binary = tmp_path / "build" / "bin" / "audiocpp_server"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    binary.chmod(0o755)
    model_root = tmp_path / "models" / "demo"
    model_root.mkdir(parents=True)
    active = {
        "version": "v1",
        "server_binary_path": str(binary),
        "cli_binary_path": str(binary.parent / "audiocpp_cli"),
        "source_path": str(tmp_path),
        "build_config": {"backend": "cuda"},
    }
    config = {
        "engine": "audio_cpp",
        "family": "demo_tts",
        "task": "tts",
        "mode": "streaming",
        "backend": "cuda",
        "device": 1,
        "threads": 6,
        "lazy_load": False,
        "load_options": {"language": "en"},
        "session_options": {"temperature": 0.8},
        "model_alias": "assistant-voice",
        "swap_env": {"CUDA_VISIBLE_DEVICES": "1"},
    }
    model = {
        "id": "audio-cpp--demo",
        "proxy_name": "audio-demo",
        "artifact": {
            "package_kind": "prepared_bundle",
            "path": str(model_root),
        },
        "compatible_engines": ["audio_cpp"],
        "config": {
            "engine": "audio_cpp",
            "engines": {"audio_cpp": dict(config)},
        },
    }
    return _Store(active), model, config


def test_audio_runtime_preview_is_pure_and_uses_stable_model_id(
    tmp_path, monkeypatch
):
    store, model, config = _fixture(tmp_path)
    sidecar_root = tmp_path / "sidecars"
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(sidecar_root))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )

    assert not sidecar_root.exists()
    assert runtime["sidecar"]["lazy_load"] is False
    assert runtime["sidecar"]["models"] == [
        {
            "id": "audio-demo",
            "family": "demo_tts",
            "path": model["artifact"]["path"],
            "task": "tts",
            "mode": "streaming",
            "load_options": {"language": "en"},
            "session_options": {"temperature": 0.8},
        }
    ]
    assert runtime["use_model_name"] == "audio-demo"
    assert runtime["generic_task_path"] == "/upstream/audio-demo/v1/tasks/run"
    assert "${PORT}" in runtime["cmd_argv"]
    assert "CUDA_VISIBLE_DEVICES=1" in runtime["env"]


def test_llama_swap_config_dispatches_audio_and_collects_sidecar(
    tmp_path, monkeypatch
):
    store, model, _ = _fixture(tmp_path)
    sidecar_root = tmp_path / "sidecars"
    monkeypatch.setattr(swap_config.data_store, "get_store", lambda: store)
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(sidecar_root))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)
    sidecars = {}

    rendered = swap_config.generate_llama_swap_config(
        {}, [model], sidecar_payloads=sidecars
    )
    document = yaml.safe_load(rendered)
    block = document["models"]["audio-demo"]

    assert block["useModelName"] == "audio-demo"
    assert block["aliases"] == ["assistant-voice"]
    assert "--config ${studio_audio_config}" in block["cmd"]
    assert len(sidecars) == 1
    sidecar = next(iter(sidecars.values()))
    assert sidecar["models"][0]["id"] == "audio-demo"


def test_sidecar_apply_is_atomic_and_removes_only_generated_orphans(
    tmp_path, monkeypatch
):
    root = tmp_path / "servers"
    root.mkdir()
    orphan = root / "orphan.json"
    orphan.write_text("{}", encoding="utf-8")
    unrelated = root / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    manager_stub = SimpleNamespace(server_configs_dir=str(root))
    monkeypatch.setattr(
        "backend.audio_cpp_manager.get_audio_cpp_manager",
        lambda: manager_stub,
    )
    target = root / "active.json"
    payload = {"models": [{"id": "audio-demo"}]}

    LlamaSwapManager._write_audio_sidecars({str(target): payload})
    LlamaSwapManager._remove_orphan_audio_sidecars({str(target)})

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == payload
    assert not orphan.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_any_active_runtime_accepts_audio_only_installation(tmp_path, monkeypatch):
    store, _, _ = _fixture(tmp_path)
    Path(store.active["cli_binary_path"]).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(swap_config.data_store, "get_store", lambda: store)

    assert swap_config.any_active_runtime_in_db() is True

