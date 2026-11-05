"""audio.cpp sidecars and capability-driven Studio proxy runtime dispatch."""

from pathlib import Path
from types import SimpleNamespace

import yaml

import backend.engines.audio_cpp.runtime as audio_runtime
import backend.proxy.commands as swap_config
from backend import reference_audio


class _Store:
    def __init__(self, active):
        self.active = active

    def get_active_engine_version(self, engine):
        return self.active if engine == "audio_cpp" else None


def test_server_flags_are_not_shadowed_by_cli_help(monkeypatch):
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: {
        "sections": [
            {"params": [{"key": "future_flag", "scope": "process", "transport": "server_flag",
                         "primary_flag": "--future-flag", "value_kind": "scalar"}]},
            {"params": [{"key": "future_flag", "scope": "request_option", "transport": "request_only",
                         "primary_flag": "--future-flag"}]},
        ],
    })
    assert audio_runtime._server_flag_tokens(None, {"version": "v1"}, {"future_flag": 12}) == [
        "--future-flag", "12",
    ]


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
        "runtime_env": {"CUDA_VISIBLE_DEVICES": "1"},
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


def test_audio_runtime_leaves_unset_process_defaults_to_engine(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    for key in ("threads", "device", "lazy_load"):
        config.pop(key)
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *a, **k: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *a: None)
    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    assert not {"threads", "device", "lazy_load"} & runtime["sidecar"].keys()


def test_audio_runtime_forces_ui_off_despite_stale_saved_setting(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    config.update(ui=True, ui_management=True)
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *a, **k: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *a: {"sections": [{"params": [
        {"key": "ui", "primary_flag": "--ui", "negative_flag": "--no-ui", "scope": "process", "transport": "server_flag", "value_kind": "flag"},
        {"key": "ui_management", "primary_flag": "--ui-management", "scope": "process", "transport": "server_flag", "value_kind": "flag"},
    ]}]})
    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    assert runtime["sidecar"]["ui"] is False
    assert runtime["sidecar"]["ui_management"] is False
    assert "--ui" not in runtime["cmd_argv"]
    assert "--ui-management" not in runtime["cmd_argv"]


def test_audio_runtime_allows_ui_on_only_for_ui_enabled_build(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    store.active["build_config"]["build_server_frontends"] = True
    config["ui"] = True
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *a, **k: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *a: {"sections": [{"params": [
        {"key": "ui", "primary_flag": "--ui", "negative_flag": "--no-ui", "scope": "process", "transport": "server_flag", "value_kind": "flag"},
    ]}]})
    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    assert "--ui" in runtime["cmd_argv"]


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


def test_builtin_runtime_passes_engine_id_verbatim_and_skips_package_mutation(tmp_path, monkeypatch):
    from backend.engines.audio_cpp.artifact import build_builtin_artifact_descriptor

    store, model, config = _fixture(tmp_path)
    config.update(family="builtin_audio_utils", task="s2s", mode="offline")
    model.update(family="builtin_audio_utils", artifact=build_builtin_artifact_descriptor("rnnoise"))
    model["config"]["engines"]["audio_cpp"] = dict(config)
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)
    monkeypatch.setattr(audio_runtime, "colocate_packaged_embeddings",
                        lambda *_args: (_ for _ in ()).throw(AssertionError("builtin is not a package")))
    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "utility")
    row = runtime["sidecar"]["models"][0]
    assert row["path"] == "rnnoise"
    assert row["family"] == "builtin_audio_utils"
    assert row["task"] == "s2s"
    assert runtime["cmd_cwd"] == str(tmp_path)




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


def test_audio_runtime_config_model_spec_override_wins(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    discovered = tmp_path / "model_specs"
    discovered.mkdir()
    (discovered / "demo_tts.json").write_text('{"family":"demo_tts"}', encoding="utf-8")
    custom = tmp_path / "custom-specs"
    custom.mkdir()
    config["model_spec_override"] = str(custom)
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    runtime = audio_runtime.build_audio_cpp_runtime(
        store, model, config, "audio-demo"
    )

    assert runtime["sidecar"]["model_spec_override"] == str(custom)
    assert runtime["sidecar"]["models"][0]["model_spec_override"] == str(custom)
    argv = runtime["cmd_argv"]
    assert argv[argv.index("--model-spec-override") + 1] == str(custom)
    assert str(discovered) not in argv


def _assert_audio_swap_cmd(cmd, *, source_root, specs):
    assert 'audiocpp_server' in cmd
    assert '--model-spec-override' in cmd
    assert str(specs) in cmd
    assert '--port ${PORT}' in cmd
    assert 'bash -c' not in cmd


def test_preview_native_proxy_command_uses_audio_cpp_swap_cmd(tmp_path, monkeypatch):
    store, model, _config = _fixture(tmp_path)
    specs = tmp_path / "model_specs"
    specs.mkdir()
    (specs / "demo_tts.json").write_text('{"family":"demo_tts"}', encoding="utf-8")
    monkeypatch.setattr(swap_config.data_store, "get_store", lambda: store)
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(
        audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)

    preview = swap_config.preview_model_command(model)
    assert preview["ok"] is True
    _assert_audio_swap_cmd(preview["cmd"], source_root=tmp_path, specs=specs)
    assert preview["sidecar"]["model_spec_override"] == str(specs)


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


def _write_default_voice_spec(tmp_path, family, voice):
    specs = tmp_path / "model_specs"
    specs.mkdir(exist_ok=True)
    (specs / f"{family}.json").write_text(
        '{"family":"%s","schema_version":1,"ui":{"default_voice":"%s","builtin_voices":["%s"]}}'
        % (family, voice, voice),
        encoding="utf-8",
    )


def test_audio_runtime_promotes_unique_pocket_tts_preset(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    _write_default_voice_spec(tmp_path, "pocket_tts", "alba")
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
    _write_default_voice_spec(tmp_path, "pocket_tts", "alba")
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










def test_any_active_runtime_accepts_audio_only_installation(tmp_path, monkeypatch):
    store, _, _ = _fixture(tmp_path)
    Path(store.active["cli_binary_path"]).write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(swap_config.data_store, "get_store", lambda: store)

    assert swap_config.any_active_runtime_in_db() is True


def test_conversion_runtime_does_not_seed_spec_speech_voice(tmp_path, monkeypatch):
    store, model, config = _fixture(tmp_path)
    _write_default_voice_spec(tmp_path, "future_multi", "speaker")
    config.update(family="future_multi", task="vc")
    monkeypatch.setattr(audio_runtime, "_sidecar_root", lambda: str(tmp_path / "sidecars"))
    monkeypatch.setattr(audio_runtime, "validate_audio_model_config", lambda *args, **kwargs: {})
    monkeypatch.setattr(audio_runtime, "get_version_entry", lambda *args: None)
    runtime = audio_runtime.build_audio_cpp_runtime(store, model, config, "audio-demo")
    row = runtime["sidecar"]["models"][0]
    assert "default_voice_preset" not in row
    assert "voice_presets" not in row
