"""Parser-driven command generation and shell escaping."""

from __future__ import annotations

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

import backend.llama_swap_config as llama_swap_config


def _raise(exc):
    raise exc


def test_emit_structured_tokens_uses_catalog_metadata():
    param_index = {
        "backend": {
            "primary_flag": "--backend",
            "value_kind": "enum",
        },
        "jinja": {
            "primary_flag": "--jinja",
            "value_kind": "flag",
        },
        "kv_offload": {
            "primary_flag": "--kv-offload",
            "negative_flag": "--no-kv-offload",
            "value_kind": "flag",
        },
        "stop": {
            "primary_flag": "--stop",
            "value_kind": "repeatable",
        },
        "temperature": {
            "primary_flag": "--temperature",
            "value_kind": "scalar",
            "default": 0.8,
        },
    }
    config = {
        "temperature": 0.7,
        "jinja": True,
        "kv_offload": False,
        "stop": ["END", "User:"],
        "backend": "pytorch",
        "custom_args": "--chat-template chatml --flag-only",
    }

    tokens = llama_swap_config._emit_structured_tokens(
        config,
        engine="llama_cpp",
        param_index=param_index,
    )

    assert tokens == [
        "--backend",
        "pytorch",
        "--jinja",
        "--no-kv-offload",
        "--stop",
        "END",
        "--stop",
        "User:",
        "--temperature",
        "0.7",
        "--chat-template",
        "chatml",
        "--flag-only",
    ]


def test_emit_structured_tokens_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown structured config keys for llama_cpp: temp"):
        llama_swap_config._emit_structured_tokens(
            {"temp": 0.7},
            engine="llama_cpp",
            param_index={"temperature": {"primary_flag": "--temperature", "value_kind": "scalar"}},
        )


def test_emit_param_tokens_covers_defaults_repeatable_and_missing_primary():
    assert llama_swap_config._emit_param_tokens(
        "temperature",
        0.8,
        {"primary_flag": "--temperature", "value_kind": "scalar", "default": 0.8},
    ) == []
    assert llama_swap_config._emit_param_tokens(
        "stop",
        ["END", "", None],
        {"primary_flag": "--stop", "value_kind": "repeatable"},
    ) == ["--stop", "END"]
    assert llama_swap_config._emit_param_tokens(
        "flag_only",
        False,
        {"primary_flag": "--enable-x", "value_kind": "flag"},
    ) == []
    with pytest.raises(ValueError, match="missing primary_flag metadata"):
        llama_swap_config._emit_param_tokens(
            "temperature",
            0.7,
            {"value_kind": "scalar"},
        )


def test_split_custom_args_rejects_invalid_shell_syntax():
    with pytest.raises(ValueError, match="Invalid custom_args shell syntax"):
        llama_swap_config._split_custom_args("--chat-template 'unterminated")


def test_supported_flags_prefer_catalog_and_cache(monkeypatch):
    llama_swap_config.clear_supported_flags_cache()
    monkeypatch.setattr(llama_swap_config, "infer_engine_id_for_binary", lambda path: "llama_cpp")
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_entry",
        lambda engine: {"sections": [{"params": [{"flags": ["--threads", "--ctx-size"]}]}]},
    )
    monkeypatch.setattr(llama_swap_config, "_abs_binary_path", lambda path: path)

    flags = llama_swap_config.supported_flags_for_llama_binary("/tmp/llama-server")

    assert flags == {"--threads", "--ctx-size"}
    assert llama_swap_config._supported_flags_cache["/tmp/llama-server"] == {"--threads", "--ctx-size"}


def test_runtime_helper_fallbacks_and_wrappers(monkeypatch):
    monkeypatch.setattr("backend.data_store.get_store", lambda: object())
    monkeypatch.setattr(
        llama_swap_config,
        "infer_llama_engine_for_binary",
        lambda store, path: _raise(RuntimeError("boom")),
    )
    assert llama_swap_config.infer_engine_id_for_binary("/tmp/llama-server") == "llama_cpp"

    class Store:
        def get_active_engine_version(self, engine):
            if engine == "llama_cpp":
                return {"version": "v1"}
            return {}

    monkeypatch.setattr("backend.data_store.get_store", lambda: Store())
    monkeypatch.setattr("backend.engine_param_catalog.get_version_entry", lambda store, engine, version: {"sections": []})
    monkeypatch.setattr("backend.engine_param_catalog.param_mapping_from_entry", lambda entry: {"temperature": ["--temperature"]})
    monkeypatch.setattr("backend.engine_param_catalog.param_index_from_entry", lambda entry: {"temperature": {"primary_flag": "--temperature"}})

    assert llama_swap_config._active_engine_entry("llama_cpp") == {"sections": []}
    assert llama_swap_config.resolve_llama_param_mapping_from_engine("llama_cpp") == {"temperature": ["--temperature"]}
    assert llama_swap_config.get_param_mapping(False) == {"temperature": ["--temperature"]}
    assert llama_swap_config._active_engine_param_index("llama_cpp") == {"temperature": {"primary_flag": "--temperature"}}


def test_get_supported_flags_parses_help_and_uses_cache(monkeypatch, tmp_path):
    llama_swap_config.clear_supported_flags_cache()
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")

    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(binary), str(tmp_path)),
    )
    monkeypatch.setattr(llama_swap_config, "llama_help_ld_library_path", lambda cwd: "/fake/lib")
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        return SimpleNamespace(stdout="--threads\n--ctx-size\n", returncode=0)

    monkeypatch.setattr(llama_swap_config.subprocess, "run", fake_run)

    first = llama_swap_config.get_supported_flags(str(binary))
    second = llama_swap_config.get_supported_flags(str(binary))

    assert first == {"--threads", "--ctx-size"}
    assert second == first
    assert calls["count"] == 1


def test_get_supported_flags_handles_warning_branches(monkeypatch, tmp_path):
    llama_swap_config.clear_supported_flags_cache()
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(binary), str(tmp_path)),
    )
    monkeypatch.setattr(llama_swap_config, "llama_help_ld_library_path", lambda cwd: "/fake/lib")

    monkeypatch.setattr(
        llama_swap_config.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="", returncode=1),
    )
    assert llama_swap_config.get_supported_flags(str(binary)) == set()

    llama_swap_config.clear_supported_flags_cache()
    monkeypatch.setattr(
        llama_swap_config.subprocess,
        "run",
        lambda *args, **kwargs: _raise(subprocess.TimeoutExpired(cmd="llama-server --help", timeout=30)),
    )
    assert llama_swap_config.get_supported_flags(str(binary)) == set()


def test_is_flag_supported_uses_mapping_and_fallback(monkeypatch):
    monkeypatch.setattr(
        llama_swap_config,
        "supported_flags_for_llama_binary",
        lambda path: {"--temperature", "--ctx-size"},
    )

    assert llama_swap_config.is_flag_supported(
        "temperature",
        "--temperature",
        "/tmp/llama-server",
        {"temperature": ["--temp", "--temperature"]},
    ) is True
    assert llama_swap_config.is_flag_supported(
        "threads",
        "--threads",
        "/tmp/llama-server",
        {"temperature": ["--temp", "--temperature"]},
    ) is False


def test_is_ik_llama_cpp_uses_flag_fallback(monkeypatch):
    monkeypatch.setattr(llama_swap_config, "get_supported_flags", lambda path: {"--smart-expert-reduction"})

    assert llama_swap_config.is_ik_llama_cpp("/tmp/llama-server") is True
    assert llama_swap_config.is_ik_llama_cpp(None) is False


def test_is_ik_llama_cpp_uses_store_matches(monkeypatch):
    class Store:
        def get_active_engine_version(self, engine):
            if engine == "ik_llama":
                return {"binary_path": "/tmp/ik-llama"}
            if engine == "llama_cpp":
                return {"binary_path": "/tmp/llama"}
            return None

    monkeypatch.setattr("backend.data_store.get_store", lambda: Store())
    monkeypatch.setattr(llama_swap_config, "_abs_binary_path", lambda path: path)
    assert llama_swap_config.is_ik_llama_cpp("/tmp/ik-llama") is True
    assert llama_swap_config.is_ik_llama_cpp("/tmp/llama") is False


def test_render_bash_command_round_trips_shell_escaping():
    argv = [
        "python3",
        "-c",
        "import json, sys; print(json.dumps(sys.argv[1:]))",
        "has spaces",
        "quote's here",
        "*.gguf",
        '{"yaml": "a: b"}',
        "--literal-leading-dash",
        "${PORT}",
    ]
    cmd = llama_swap_config._render_bash_command(argv)
    env = dict(os.environ)
    env["PORT"] = "4317"
    result = subprocess.run(
        ["bash", "-lc", cmd],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    assert json.loads(result.stdout.strip()) == [
        "has spaces",
        "quote's here",
        "*.gguf",
        '{"yaml": "a: b"}',
        "--literal-leading-dash",
        "4317",
    ]


def test_resolve_llama_model_source_and_mmproj(monkeypatch, tmp_path):
    missing_model = tmp_path / "missing.gguf"
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_gguf_model_path_for_quant",
        lambda hf_id, quant: str(missing_model),
    )

    model_path, hf_repo_arg, hf_id = llama_swap_config._resolve_llama_model_source(
        {"huggingface_id": "org/model", "quantization": "Q4_K_M"}
    )
    assert model_path is None
    assert hf_repo_arg == "org/model:q4_k_m"
    assert hf_id == "org/model"

    fallback_path, fallback_repo, _ = llama_swap_config._resolve_llama_model_source(
        {"huggingface_id": None, "quantization": None},
        fallback_model_path="models/local.gguf",
    )
    assert fallback_path == "/app/models/local.gguf"
    assert fallback_repo is None

    monkeypatch.setattr(
        "backend.huggingface.resolve_cached_model_path",
        lambda hf_id, filename: "cache/mmproj.gguf",
    )
    orig_exists = llama_swap_config.os.path.exists
    monkeypatch.setattr(
        llama_swap_config.os.path,
        "exists",
        lambda path: True if path == "cache/mmproj.gguf" else orig_exists(path),
    )
    mmproj = llama_swap_config._resolve_mmproj_path(
        {"mmproj_filename": "mmproj.gguf"},
        "org/model",
        None,
    )
    assert mmproj == "/app/cache/mmproj.gguf"
    assert llama_swap_config._resolve_mmproj_path({"mmproj_filename": "mmproj.gguf"}, "org/model", "repo:quant") is None


def test_runtime_path_helpers_and_model_attr(monkeypatch, tmp_path):
    cuda_root = tmp_path / "cuda"
    (cuda_root / "lib64").mkdir(parents=True)

    class Installer:
        def _get_cuda_path(self):
            return str(cuda_root)

    monkeypatch.setattr("backend.cuda_installer.get_cuda_installer", lambda: Installer())
    assert llama_swap_config._resolve_cuda_library_path("/build") == f"{cuda_root / 'lib64'}:/build"

    venv_dir = tmp_path / "venv"
    lmdeploy_bin = venv_dir / "bin" / "lmdeploy"
    lmdeploy_bin.parent.mkdir(parents=True)
    lmdeploy_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    lmdeploy_bin.chmod(0o755)

    class Store:
        def get_active_engine_version(self, engine):
            return {"venv_path": str(venv_dir)} if engine == "lmdeploy" else None

    monkeypatch.setattr("backend.data_store.get_store", lambda: Store())
    assert llama_swap_config._resolve_lmdeploy_bin() == str(lmdeploy_bin)

    class Obj:
        value = 7

    assert llama_swap_config._model_attr({"value": 3}, "value") == 3
    assert llama_swap_config._model_attr(Obj(), "value") == 7


def test_any_active_gguf_runtime_in_db_handles_errors(monkeypatch):
    monkeypatch.setattr("backend.data_store.get_store", lambda: _raise(RuntimeError("boom")))
    assert llama_swap_config.any_active_gguf_runtime_in_db() is False


def test_preview_llama_swap_command_uses_catalog_metadata(monkeypatch, tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary_path.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_text("gguf", encoding="utf-8")

    monkeypatch.setattr(
        llama_swap_config,
        "resolve_gguf_model_path_for_quant",
        lambda hf_id, quant: str(model_path),
    )
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_binary_path_for_engine",
        lambda store, eng: str(binary_path),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(binary_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_resolve_cuda_library_path",
        lambda build_dir: "/fake/lib",
    )
    monkeypatch.setattr(
        llama_swap_config,
        "infer_engine_id_for_binary",
        lambda path: "llama_cpp",
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_param_index",
        lambda engine: {
            "jinja": {"primary_flag": "--jinja", "value_kind": "flag"},
            "stop": {"primary_flag": "--stop", "value_kind": "repeatable"},
            "temperature": {
                "primary_flag": "--temperature",
                "value_kind": "scalar",
                "default": 0.8,
            },
        },
    )

    preview = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "id": "m1",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {
                "engine": "llama_cpp",
                "engines": {
                    "llama_cpp": {
                        "temperature": 0.7,
                        "jinja": True,
                        "stop": ["END HERE"],
                    }
                },
            },
        }
    )

    assert preview["ok"] is True
    assert "--model" in preview["cmd"]
    assert "--alias org-model.q4_k_m" in preview["cmd"]
    assert "--jinja" in preview["cmd"]
    assert "--temperature 0.7" in preview["cmd"]
    assert "--stop" in preview["cmd"]
    assert "END HERE" in preview["cmd"]
    assert preview["use_model_name"] is None


def test_preview_lmdeploy_command_uses_catalog_metadata(monkeypatch):
    monkeypatch.setattr(
        llama_swap_config,
        "_resolve_lmdeploy_bin",
        lambda: "/opt/lmdeploy/bin/lmdeploy",
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_param_index",
        lambda engine: {
            "allow_credentials": {
                "primary_flag": "--allow-credentials",
                "value_kind": "flag",
            },
            "backend": {
                "primary_flag": "--backend",
                "value_kind": "enum",
            },
            "tp": {
                "primary_flag": "--tp",
                "value_kind": "scalar",
            },
        },
    )

    preview = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "id": "repo-model",
            "huggingface_id": "org/repo-model",
            "format": "safetensors",
            "config": {
                "engine": "lmdeploy",
                "engines": {
                    "lmdeploy": {
                        "allow_credentials": True,
                        "backend": "pytorch",
                        "tp": 2,
                    }
                },
            },
        }
    )

    assert preview["ok"] is True
    assert "serve api_server org/repo-model --server-port ${PORT}" in preview["cmd"]
    assert "--allow-credentials" in preview["cmd"]
    assert "--backend pytorch" in preview["cmd"]
    assert "--tp 2" in preview["cmd"]
    assert preview["use_model_name"] == "org/repo-model"


def test_generate_llama_swap_config_builds_groups_for_catalog_driven_models(monkeypatch, tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary_path.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_text("gguf", encoding="utf-8")

    import backend.data_store as data_store

    monkeypatch.setattr(
        data_store,
        "resolve_proxy_name",
        lambda model: model.get("proxy_name")
        or f"{model.get('huggingface_id', '').replace('/', '-')}.{str(model.get('quantization', '')).lower()}".strip("."),
    )
    monkeypatch.setattr(
        data_store,
        "normalize_proxy_alias",
        lambda alias: (alias or "").strip().lower(),
    )
    monkeypatch.setattr(
        data_store,
        "generate_proxy_name",
        lambda hf_id, quant=None: f"{hf_id.replace('/', '-')}.{str(quant or '').lower()}".strip("."),
    )
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_binary_path_for_engine",
        lambda store, eng: str(binary_path),
    )
    monkeypatch.setattr(llama_swap_config, "resolve_gguf_model_path_for_quant", lambda hf_id, quant: str(model_path))
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(binary_path), str(tmp_path)),
    )
    monkeypatch.setattr(llama_swap_config, "_resolve_cuda_library_path", lambda cwd: "/fake/lib")
    monkeypatch.setattr(llama_swap_config, "infer_engine_id_for_binary", lambda path: "llama_cpp")
    monkeypatch.setattr(llama_swap_config, "_resolve_lmdeploy_bin", lambda: "/opt/lmdeploy/bin/lmdeploy")

    def fake_param_index(engine):
        if engine == "lmdeploy":
            return {"tp": {"primary_flag": "--tp", "value_kind": "scalar"}}
        return {"temperature": {"primary_flag": "--temperature", "value_kind": "scalar"}}

    monkeypatch.setattr(llama_swap_config, "_active_engine_param_index", fake_param_index)

    all_models = [
        {
            "id": "gguf-1",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 0.7}}},
        },
        {
            "id": "st-1",
            "huggingface_id": "org/repo-model",
            "format": "safetensors",
            "config": {"engine": "lmdeploy", "engines": {"lmdeploy": {"tp": 2}}},
        },
    ]
    running_overlay = {
        "org-model.q4_k_m": {
            "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {"temperature": 0.9}}},
            "model_path": "runtime/model.gguf",
        }
    }

    yaml_str = llama_swap_config.generate_llama_swap_config(running_overlay, all_models=all_models)
    doc = json.loads(json.dumps(llama_swap_config.yaml.safe_load(yaml_str)))

    assert set(doc["models"].keys()) == {"org-model.q4_k_m", "org-repo-model"}
    assert "--temperature 0.9" in doc["models"]["org-model.q4_k_m"]["cmd"]
    assert "serve api_server org/repo-model --server-port ${PORT} --tp 2" in doc["models"]["org-repo-model"]["cmd"]
    assert doc["groups"]["concurrent_models"]["members"] == ["org-model.q4_k_m", "org-repo-model"]


def test_generate_running_overlay_empty_config_keeps_catalog_ik_llama_binary(monkeypatch, tmp_path):
    """sync_running_models uses config: {}; merged config must still use ik_llama from the DB model."""
    llama_bin = tmp_path / "llama-server"
    llama_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    llama_bin.chmod(0o755)
    ik_bin = tmp_path / "ik-server"
    ik_bin.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    ik_bin.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_text("gguf", encoding="utf-8")

    import backend.data_store as data_store

    monkeypatch.setattr(
        data_store,
        "resolve_proxy_name",
        lambda model: model.get("proxy_name")
        or f"{model.get('huggingface_id', '').replace('/', '-')}.{str(model.get('quantization', '')).lower()}".strip("."),
    )
    monkeypatch.setattr(data_store, "normalize_proxy_alias", lambda alias: (alias or "").strip().lower())
    monkeypatch.setattr(
        data_store,
        "generate_proxy_name",
        lambda hf_id, quant=None: f"{hf_id.replace('/', '-')}.{str(quant or '').lower()}".strip("."),
    )
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_binary_path_for_engine",
        lambda store, eng: str(ik_bin) if eng == "ik_llama" else str(llama_bin),
    )
    monkeypatch.setattr(llama_swap_config, "resolve_gguf_model_path_for_quant", lambda hf_id, quant: str(model_path))

    def inv_paths(path):
        sub = "ik-build" if "ik-server" in str(path) else "ll-build"
        d = tmp_path / sub
        d.mkdir(exist_ok=True)
        return (str(path), str(d))

    monkeypatch.setattr(llama_swap_config, "resolve_llama_server_invocation_paths", inv_paths)
    monkeypatch.setattr(llama_swap_config, "_resolve_cuda_library_path", lambda cwd: "/fake/lib")
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_param_index",
        lambda engine: {"temperature": {"primary_flag": "--temperature", "value_kind": "scalar"}},
    )

    all_models = [
        {
            "id": "gguf-ik",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {
                "engine": "ik_llama",
                "engines": {"ik_llama": {"temperature": 0.5}},
            },
        }
    ]
    running_overlay = {"org-model.q4_k_m": {"config": {}, "model_path": str(model_path)}}

    yaml_str = llama_swap_config.generate_llama_swap_config(running_overlay, all_models=all_models)
    doc = json.loads(json.dumps(llama_swap_config.yaml.safe_load(yaml_str)))
    cmd = doc["models"]["org-model.q4_k_m"]["cmd"]
    assert "./ik-server" in cmd
    assert "ik-build" in cmd
    assert "./llama-server" not in cmd


def test_preview_handles_missing_proxy_and_missing_runtime(monkeypatch):
    monkeypatch.setattr("backend.data_store.resolve_proxy_name", lambda model: "")
    no_proxy = llama_swap_config.preview_llama_swap_command_for_model({"config": {}})
    assert no_proxy["ok"] is False
    assert no_proxy["error"] == "Model has no proxy name"

    monkeypatch.setattr("backend.data_store.resolve_proxy_name", lambda model: "proxy")
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_binary_path_for_engine",
        lambda store, eng: None,
    )
    no_runtime = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "config": {"engine": "llama_cpp", "engines": {"llama_cpp": {}}},
        }
    )
    assert no_runtime["ok"] is False
    assert "No active llama-server binary configured" in no_runtime["error"]


def test_preview_surfaces_metadata_errors(monkeypatch, tmp_path):
    binary_path = tmp_path / "llama-server"
    binary_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    binary_path.chmod(0o755)
    model_path = tmp_path / "model.gguf"
    model_path.write_text("gguf", encoding="utf-8")

    monkeypatch.setattr(
        llama_swap_config,
        "resolve_gguf_model_path_for_quant",
        lambda hf_id, quant: str(model_path),
    )
    monkeypatch.setattr(
        "backend.llama_engine_resolve.get_active_binary_path_for_engine",
        lambda store, eng: str(binary_path),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "resolve_llama_server_invocation_paths",
        lambda path: (str(binary_path), str(tmp_path)),
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_resolve_cuda_library_path",
        lambda build_dir: "/fake/lib",
    )
    monkeypatch.setattr(
        llama_swap_config,
        "infer_engine_id_for_binary",
        lambda path: "llama_cpp",
    )
    monkeypatch.setattr(
        llama_swap_config,
        "_active_engine_param_index",
        lambda engine: {
            "temperature": {
                "primary_flag": "--temperature",
                "value_kind": "scalar",
            }
        },
    )

    preview = llama_swap_config.preview_llama_swap_command_for_model(
        {
            "id": "m1",
            "huggingface_id": "org/model",
            "quantization": "Q4_K_M",
            "format": "gguf",
            "config": {
                "engine": "llama_cpp",
                "engines": {
                    "llama_cpp": {
                        "temp": 0.7,
                    }
                },
            },
        }
    )

    assert preview["ok"] is False
    assert "Unknown structured config keys for llama_cpp: temp" in preview["error"]
