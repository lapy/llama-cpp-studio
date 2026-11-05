"""1Cat-vLLM launch is compiled from its authoritative definition profile.

`_compile_module(... engine="1cat_vllm")` used to be the Python shape; it is
gone. The definition's `runtime_profiles.default` is now the only contract:
venv interpreter, `-m vllm.entrypoints.openai.api_server`, `--model` / `--port`,
venv-root cwd, no managed-CUDA env. These pins freeze the observable spec. The
engine id is derived from the shipped registry entry with this module profile
and no CUDA env — no engine-id literal is hardcoded here.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from backend.engines.registry import ENGINE_REGISTRY, SHIPPED_ENGINE_IDS
from backend.proxy import launch_spec, profile_launch

MODULE = "vllm.entrypoints.openai.api_server"


def _profiled_engine() -> str:
    """The module-based engine WITHOUT a managed-CUDA env (1Cat-vLLM)."""
    candidates = [
        engine_id
        for engine_id in SHIPPED_ENGINE_IDS
        if MODULE in ENGINE_REGISTRY[engine_id].runtime_profile.get("argv_prefix", [])
        and not (ENGINE_REGISTRY[engine_id].runtime_profile.get("env") or {}).get("cuda")
    ]
    assert len(candidates) == 1, candidates
    return candidates[0]


def _make_bin(path, name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    exe = path / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


class _Store:
    def __init__(self, row):
        self._row = row

    def get_active_engine_version(self, engine_id):
        return dict(self._row)


def _setup(tmp_path, monkeypatch, *, with_tokens=None, with_venv=True):
    engine = _profiled_engine()
    install = tmp_path / "installs" / "v1"
    exe = _make_bin(install / "bin", "python") if with_venv else str(install / "bin" / "python")
    row = {"version": "v1", "venv_path": str(install)}

    import backend.proxy.commands as commands

    monkeypatch.setattr(commands, "_active_engine_param_index", lambda engine: {})
    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: dict(row))
    if with_tokens is not None:
        monkeypatch.setattr(
            commands, "_emit_structured_tokens", lambda config, **kwargs: list(with_tokens)
        )
    return engine, install, exe


def _compiled(engine, model, config):
    spec = profile_launch.compile_profile_launch(model, config, "model-stable", engine)
    assert spec is not None
    return spec


def test_default_config_compiles_the_declared_profile(tmp_path, monkeypatch):
    engine, install, exe = _setup(tmp_path, monkeypatch)
    model = {"huggingface_id": "org/model"}
    spec = _compiled(engine, model, {})

    assert spec.executable == exe
    assert spec.argv == [
        "-m",
        MODULE,
        "--model",
        "org/model",
        "--port",
        dict(launch_spec.PORT_PLACEHOLDER),
    ]
    assert spec.cwd == str(install)
    assert spec.env.set["PATH"].startswith(str(install / "bin"))
    assert "CUDA_HOME" not in spec.env.set
    assert spec.file_identities.keys() == {"executable"}


def test_structured_tokens_are_appended(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch, with_tokens=["--tp", "2"])
    model = {"huggingface_id": "org/model"}
    spec = _compiled(engine, model, {"tp": 2})
    assert spec.argv[-2:] == ["--tp", "2"]


def test_dispatch_uses_the_declared_profile(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch)
    model = {"huggingface_id": "org/model"}
    assert launch_spec._compile_launch(model, {}, "model-stable", engine) == _compiled(
        engine, model, {}
    )


def test_port_flag_is_studio_owned(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch, with_tokens=["--port", "1"])
    model = {"huggingface_id": "org/model"}
    with pytest.raises(launch_spec.LaunchCompileError):
        profile_launch.compile_profile_launch(model, {}, "model-stable", engine)


def test_missing_huggingface_id_fails(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch)
    with pytest.raises(launch_spec.LaunchCompileError):
        profile_launch.compile_profile_launch({"local_path": "/m"}, {}, "model-stable", engine)


def test_missing_environment_fails(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch, with_venv=False)
    model = {"huggingface_id": "org/model"}
    with pytest.raises(launch_spec.LaunchCompileError):
        profile_launch.compile_profile_launch(model, {}, "model-stable", engine)
