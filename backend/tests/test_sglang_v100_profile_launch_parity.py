"""SGLang V100 launch is compiled from its authoritative definition profile.

The Python `_compile_module` branch is gone. The declared profile owns: managed
CUDA resolution with a 12.8 fallback, three fixed env defaults
(FLASHINFER_DISABLE_VERSION_CHECK, TORCH_CUDA_ARCH_LIST=7.0,
CUDA_DEVICE_ORDER=PCI_BUS_ID) and the hard CUDA_HOME requirement with its
pinned error message.
"""

from __future__ import annotations

import stat

import pytest

from backend.engines.registry import ENGINE_REGISTRY, SHIPPED_ENGINE_IDS
from backend.proxy import launch_spec, profile_launch

MODULE = "sglang.launch_server"
CUDA_ENV = {"CUDA_HOME": "/cuda", "PATH": "/cuda/bin", "LD_LIBRARY_PATH": "/cuda/lib"}


def _profiled_engine() -> str:
    candidates = [
        engine_id
        for engine_id in SHIPPED_ENGINE_IDS
        if MODULE in ENGINE_REGISTRY[engine_id].runtime_profile.get("argv_prefix", [])
        and (ENGINE_REGISTRY[engine_id].runtime_profile.get("env") or {}).get("required")
    ]
    # Sibling SM70 forks (sglang-sxm2) declare the same launch_server profile;
    # this suite pins sglang_v100's parity contract, so select it by id.
    assert "sglang_v100" in candidates, candidates
    return "sglang_v100"


def _make_bin(path, name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    exe = path / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


class _Cuda:
    def __init__(self, env):
        self._env = env
        self.seen: list = []

    def get_cuda_env(self, version):
        self.seen.append(version)
        return dict(self._env)


def _setup(tmp_path, monkeypatch, *, cuda_env=None, with_version=True):
    engine = _profiled_engine()
    install = tmp_path / "installs" / "v1"
    exe = _make_bin(install / "bin", "python")
    row = {"version": "v1", "venv_path": str(install)}
    if with_version:
        row["cuda_version"] = "12.8"

    import backend.cuda_installer as cuda_installer
    import backend.proxy.commands as commands

    cuda = _Cuda(CUDA_ENV if cuda_env is None else cuda_env)
    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: dict(row))
    monkeypatch.setattr(cuda_installer, "get_cuda_installer", lambda: cuda)
    monkeypatch.setattr(commands, "_active_engine_param_index", lambda engine: {})
    monkeypatch.setattr(commands, "_active_engine_entry", lambda engine: {})
    monkeypatch.setattr(commands, "_emit_structured_tokens", lambda config, **kwargs: [])
    return engine, install, exe, cuda


def _compiled(engine, model, config):
    spec = profile_launch.compile_profile_launch(model, config, "model-stable", engine)
    assert spec is not None
    return spec


def test_default_config_compiles_the_declared_profile(tmp_path, monkeypatch):
    engine, install, exe, _cuda = _setup(tmp_path, monkeypatch)
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
    assert spec.env.set["TORCH_CUDA_ARCH_LIST"] == "7.0"
    assert spec.env.set["FLASHINFER_DISABLE_VERSION_CHECK"] == "1"
    assert spec.env.set["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_cuda_version_falls_back_to_12_8(tmp_path, monkeypatch):
    engine, _install, _exe, cuda = _setup(tmp_path, monkeypatch, with_version=False)
    model = {"huggingface_id": "org/model"}
    _compiled(engine, model, {})
    assert cuda.seen and all(version == "12.8" for version in cuda.seen)


def test_missing_cuda_home_error_is_pinned(tmp_path, monkeypatch):
    engine, _install, _exe, _cuda = _setup(tmp_path, monkeypatch, cuda_env={})
    monkeypatch.delenv("CUDA_HOME", raising=False)
    model = {"huggingface_id": "org/model"}
    with pytest.raises(launch_spec.LaunchCompileError) as excinfo:
        profile_launch.compile_profile_launch(model, {}, "model-stable", engine)
    assert str(excinfo.value) == "SGLang V100 requires Studio-managed CUDA 12.8"
    assert excinfo.value.field == "CUDA_HOME"


def test_dispatch_uses_the_declared_profile(tmp_path, monkeypatch):
    engine, _install, _exe, _cuda = _setup(tmp_path, monkeypatch)
    model = {"huggingface_id": "org/model"}
    assert launch_spec._compile_launch(model, {}, "model-stable", engine) == _compiled(
        engine, model, {}
    )
