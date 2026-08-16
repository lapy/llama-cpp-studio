"""audio.cpp sidecars and capability-driven llama-swap runtime dispatch."""

from pathlib import Path
from types import SimpleNamespace

import yaml

import backend.audio_cpp_runtime as audio_runtime
import backend.llama_swap_config as swap_config
from backend import reference_audio
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
    assert "${PORT}" in runtime["cmd_argv"]
    assert "CUDA_VISIBLE_DEVICES=1" in runtime["env"]


def test_audio_runtime_accepts_gguf_file_model_path(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    gguf = tmp_path / "models" / "demo" / "model.gguf"
    # Replace directory artifact with a bare GGUF file path.
    import shutil

    shutil.rmtree(model["artifact"]["path"])
    gguf.parent.mkdir(parents=True)
    gguf.write_bytes(b"gguf")
    model["artifact"] = {
        "package_kind": "prepared_bundle",
        "path": str(gguf),
        "runtime_path": str(gguf),
        "layout": "gguf_file",
    }
    model["local_path"] = str(gguf)
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )
    # Prefer the package directory when it hosts model.gguf.
    assert runtime["sidecar"]["models"][0]["path"] == str(gguf.parent.resolve())


def test_generate_swap_skips_broken_audio_model(tmp_path, monkeypatch):
    """One broken audio.cpp model must not abort the whole llama-swap YAML."""
    _store, model, config = _fixture(tmp_path)
    model["config"] = {
        "engine": "audio_cpp",
        "engines": {"audio_cpp": dict(config)},
    }
    model["llama_swap_id"] = "audio-demo"

    def boom(*_args, **_kwargs):
        raise ValueError("boom")

    monkeypatch.setattr(
        "backend.audio_cpp_runtime.build_audio_cpp_runtime", boom
    )
    monkeypatch.setattr(
        swap_config, "get_active_binary_path_for_engine", lambda *_args: ""
    )

    yaml_str = swap_config.generate_llama_swap_config({}, [model])
    parsed = yaml.safe_load(yaml_str)
    assert "audio-demo" not in (parsed.get("models") or {})


def test_audio_runtime_injects_model_specs_override_from_source_tree(
    tmp_path, monkeypatch
):
    store, model, config = _fixture(tmp_path)
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "demo_tts.json").write_text('{"family":"demo_tts"}', encoding="utf-8")
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )

    assert runtime["sidecar"]["model_spec_override"] == str(specs)
    assert runtime["sidecar"]["models"][0]["model_spec_override"] == str(specs)
    assert "--model-spec-override" in runtime["cmd_argv"]
    assert str(specs) in runtime["cmd_argv"]
    assert runtime["cmd_cwd"] == str(tmp_path)

    cmd = swap_config._audio_cpp_swap_cmd(runtime)
    assert cmd.startswith("bash -c ")
    assert f"cd {tmp_path}" in cmd or f"cd '{tmp_path}'" in cmd or f'cd "{tmp_path}"' in cmd
    assert "--model-spec-override" in cmd


def test_audio_runtime_writes_normalized_voice_presets(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    refs = tmp_path / "models" / "demo" / "refs"
    refs.mkdir(parents=True)
    wav = refs / "voice.wav"
    wav.write_bytes(b"RIFF")
    config["voice_presets"] = {
        "assistant": {
            "voice_ref": "refs/voice.wav",
            "reference_text": "Hello there.",
        }
    }
    config["default_voice_preset"] = "assistant"
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )
    sidecar_model = runtime["sidecar"]["models"][0]
    assert sidecar_model["default_voice_preset"] == "assistant"
    assert sidecar_model["voice_presets"]["assistant"]["reference_text"] == "Hello there."
    assert sidecar_model["voice_presets"]["assistant"]["voice_ref"] == str(wav.resolve())


def test_audio_runtime_resolves_voice_refs_from_data_reference_root(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    data_root = tmp_path / "data"
    monkeypatch.setattr(reference_audio, "_data_root", lambda: str(data_root))
    refs = (
        data_root
        / "models"
        / "audio-cpp"
        / "reference-audio"
        / reference_audio._safe_storage_key(model["id"])
        / "refs"
    )
    refs.mkdir(parents=True)
    wav = refs / "voice.wav"
    wav.write_bytes(b"RIFF")
    config["voice_presets"] = {
        "assistant": {
            "voice_ref": "refs/voice.wav",
            "reference_text": "Hello there.",
        }
    }
    config["default_voice_preset"] = "assistant"
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )

    sidecar_model = runtime["sidecar"]["models"][0]
    assert sidecar_model["voice_presets"]["assistant"]["voice_ref"] == str(wav.resolve())


def test_audio_runtime_promotes_unique_pocket_tts_preset(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    config["family"] = "pocket_tts"
    config["voice_presets"] = {"alba": {"voice_id": "alba"}}
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )
    assert runtime["sidecar"]["models"][0]["default_voice_preset"] == "alba"


def test_audio_runtime_seeds_pocket_tts_session_voice_from_packaged_ids(
    tmp_path, monkeypatch
):
    store, model, config = _fixture(tmp_path)
    config["family"] = "pocket_tts"
    embeddings = tmp_path / "models" / "demo" / "embeddings"
    embeddings.mkdir()
    (embeddings / "cosette.safetensors").write_bytes(b"x")
    (embeddings / "alba.safetensors").write_bytes(b"x")
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )
    sidecar_model = runtime["sidecar"]["models"][0]
    assert sidecar_model["default_voice_preset"] == "alba"
    assert sidecar_model["voice_presets"]["alba"]["voice_id"] == "alba"


def test_audio_runtime_writes_speech_defaults_as_llama_swap_set_params(
    tmp_path, monkeypatch
):
    store, model, config = _fixture(tmp_path)
    config["speech_defaults"] = {
        "instructions": "female, young adult, moderate pitch",
        "temperature": 0.8,
    }
    monkeypatch.setattr(
        swap_config, "get_active_binary_path_for_engine", lambda *args: ""
    )
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    block = swap_config._llama_swap_yaml_model_block_for_config(
        cmd="audiocpp_server --config ${studio_audio_config}",
        env_list=[],
        model_id="audio-demo",
        config=config,
        use_model_name="audio-demo",
    )

    assert block["filters"]["setParams"] == {
        "instructions": "female, young adult, moderate pitch",
        "temperature": 0.8,
    }


def test_audio_runtime_writes_transcription_defaults_as_llama_swap_set_params(tmp_path):
    import backend.llama_swap_config as swap_config

    config = {
        "engine": "audio_cpp",
        "family": "qwen3_asr",
        "task": "asr",
        "transcription_defaults": {
            "language": "en",
            "prompt": "Transcribe clearly.",
        },
    }
    block = swap_config._llama_swap_yaml_model_block_for_config(
        cmd="audiocpp_server --config ${studio_audio_config}",
        env_list=[],
        model_id="audio-asr",
        config=config,
        use_model_name="audio-asr",
    )
    assert block["filters"]["setParams"] == {
        "language": "en",
        "options": {"text": "Transcribe clearly."},
    }


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
