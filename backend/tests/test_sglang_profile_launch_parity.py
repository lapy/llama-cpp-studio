"""SGLang launch is compiled from its authoritative definition profile.

The Python `_compile_module` branch is gone: the definition's
`runtime_profiles.default` (including the `sglang_runtime` switch) IS the
contract. These pins freeze the observable spec: venv python,
`-m sglang.launch_server`, `--model-path` / `--port`, venv-root cwd, managed
CUDA env; the diffusion arm swaps the argv prefix, re-indexes parameters from
the scanned diffusion catalog and raises the typed unavailable error.
"""

from __future__ import annotations

import stat

import pytest

from backend.engines.registry import ENGINE_REGISTRY, SHIPPED_ENGINE_IDS
from backend.proxy import launch_spec, profile_launch

MODULE = "sglang.launch_server"
CUDA_ENV = {"CUDA_HOME": "/cuda", "PATH": "/cuda/bin", "LD_LIBRARY_PATH": "/cuda/lib"}
DIFFUSION_CATALOG = {
    "runtime_profiles": {
        "diffusion": {
            "sections": [{"params": [{"key": "steps", "flags": ["--steps"], "type": "int"}]}]
        }
    }
}


def _profiled_engine() -> str:
    candidates = [
        engine_id
        for engine_id in SHIPPED_ENGINE_IDS
        if MODULE in ENGINE_REGISTRY[engine_id].runtime_profile.get("argv_prefix", [])
        and not (ENGINE_REGISTRY[engine_id].runtime_profile.get("env") or {}).get("required")
    ]
    assert len(candidates) == 1, candidates
    return candidates[0]


def _make_bin(path, name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    exe = path / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


class _Cuda:
    def get_cuda_env(self, version):
        return dict(CUDA_ENV)


def _setup(tmp_path, monkeypatch, *, entry=None):
    engine = _profiled_engine()
    install = tmp_path / "installs" / "v1"
    exe = _make_bin(install / "bin", "python")
    row = {"version": "v1", "venv_path": str(install), "cuda_version": "12.8"}

    import backend.cuda_installer as cuda_installer
    import backend.proxy.commands as commands

    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: dict(row))
    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: _Cuda())
    monkeypatch.setattr(
        commands, "_active_engine_param_index", lambda engine: {"cfg": {"key": "cfg"}}
    )
    monkeypatch.setattr(
        commands,
        "_active_engine_entry",
        lambda engine: dict(entry) if entry is not None else {},
    )
    monkeypatch.setattr(
        commands,
        "_emit_structured_tokens",
        lambda config, **kwargs: [f"--{key}" for key in sorted(config) if key != "sglang_runtime"],
    )
    return engine, install, exe


def _compiled(engine, model, config):
    spec = profile_launch.compile_profile_launch(model, config, "model-stable", engine)
    assert spec is not None
    return spec


def test_llm_runtime_compiles_the_declared_profile(tmp_path, monkeypatch):
    engine, install, exe = _setup(tmp_path, monkeypatch)
    model = {"huggingface_id": "org/model"}
    spec = _compiled(engine, model, {})

    assert spec.executable == exe
    assert spec.argv == [
        "-m",
        MODULE,
        "--model-path",
        "org/model",
        "--port",
        dict(launch_spec.PORT_PLACEHOLDER),
    ]
    assert spec.cwd == str(install)
    assert spec.env.set["CUDA_HOME"] == "/cuda"


def test_diffusion_runtime_compiles_the_declared_profile(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch, entry=DIFFUSION_CATALOG)
    model = {"huggingface_id": "org/model"}
    config = {"sglang_runtime": "diffusion", "cfg": 1, "steps": 2}
    spec = _compiled(engine, model, config)

    assert spec.argv[:5] == [
        "-c",
        "from sglang.cli.main import main; main()",
        "serve",
        "--model-type",
        "diffusion",
    ]
    assert "--steps" in spec.argv
    assert "--cfg" not in spec.argv


def test_diffusion_unavailable_error_is_typed(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(
        tmp_path, monkeypatch, entry={"runtime_profiles": {"diffusion": {"scan_error": "boom"}}}
    )
    model = {"huggingface_id": "org/model"}
    config = {"sglang_runtime": "diffusion"}
    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(model, config, "model-stable", engine)
    assert str(excinfo.value) == (
        "SGLang diffusion is unavailable: boom. Rescan engine parameters after "
        "installing the fork's diffusion dependencies."
    )


def test_dispatch_uses_the_declared_profile(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch)
    model = {"huggingface_id": "org/model"}
    assert launch_spec._compile_launch(model, {}, "model-stable", engine) == _compiled(
        engine, model, {}
    )
