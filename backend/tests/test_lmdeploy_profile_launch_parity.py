"""LMDeploy launch is compiled from its authoritative definition profile.

LMDeploy no longer has a Python launch compiler: ``_compile_launch`` consults
the definition's ``runtime_profiles.default`` and that output IS the contract.
These pins freeze the observable spec (executable, argv including the typed
port placeholder, cwd, env composition, install identity and file identities)
so a YAML edit that changes launch behavior is a reviewed diff.

The engine id is derived from the shipped registry entry that declares this
port binding — no engine-id literal is hardcoded here.
"""

from __future__ import annotations

import stat

import pytest

from backend.engines.registry import ENGINE_REGISTRY, SHIPPED_ENGINE_IDS
from backend.proxy import launch_spec, profile_launch


def _profiled_engine() -> str:
    candidates = [
        engine_id
        for engine_id in SHIPPED_ENGINE_IDS
        if ENGINE_REGISTRY[engine_id].runtime_profile
        and ENGINE_REGISTRY[engine_id].runtime_profile.get("port_binding", {}).get("flag")
        == "--server-port"
    ]
    assert len(candidates) == 1, candidates
    return candidates[0]


def _make_bin(path, name: str) -> str:
    path.mkdir(parents=True, exist_ok=True)
    exe = path / name
    exe.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def _setup(tmp_path, monkeypatch, *, with_tokens=None):
    engine = _profiled_engine()
    install = tmp_path / "installs" / "v1"
    exe = _make_bin(install / "bin", "lmdeploy")
    row = {"version": "v1", "venv_path": str(install), "install_dir": str(install)}

    import backend.proxy.commands as commands

    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: dict(row))
    if with_tokens is not None:
        monkeypatch.setattr(
            commands,
            "_emit_structured_tokens",
            lambda config, **kwargs: list(with_tokens),
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
        "serve",
        "api_server",
        "org/model",
        "--server-port",
        dict(launch_spec.PORT_PLACEHOLDER),
    ]
    assert spec.cwd is None
    assert spec.env.set["PATH"].startswith(str(install / "bin"))
    assert spec.engine_install_id
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
    engine, _install, _exe = _setup(tmp_path, monkeypatch, with_tokens=["--server-port", "1"])
    model = {"huggingface_id": "org/model"}
    with pytest.raises(launch_spec.LaunchCompileError):
        profile_launch.compile_profile_launch(model, {}, "model-stable", engine)


def test_cpu_contract_is_enforced(tmp_path, monkeypatch):
    engine, _install, _exe = _setup(tmp_path, monkeypatch)
    model = {"huggingface_id": "org/model"}
    with pytest.raises(launch_spec.LaunchCompileError):
        profile_launch.compile_profile_launch(model, {"gpu_mode": "cpu"}, "model-stable", engine)


def test_relative_venv_path_normalizes_to_the_app_root(tmp_path, monkeypatch):
    """Relative venv paths resolve against the configured app root.

    The production default is ``/app``; the test points it at ``tmp_path`` so
    CI runners (where ``/app`` does not exist) exercise the same normalization.
    """
    engine = _profiled_engine()
    rel = "pytest-lmdeploy-rel"
    app_dir = tmp_path / rel
    exe = _make_bin(app_dir / "bin", "lmdeploy")
    row = {"version": "v1", "venv_path": rel}
    monkeypatch.setattr(profile_launch, "APP_ROOT", str(tmp_path))
    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: dict(row))
    model = {"huggingface_id": "org/model"}
    profiled = _compiled(engine, model, {})
    assert profiled.executable == exe


def test_non_executable_binary_is_rejected(tmp_path, monkeypatch):
    engine = _profiled_engine()
    install = tmp_path / "installs" / "v1"
    bin_dir = install / "bin"
    bin_dir.mkdir(parents=True)
    plain = bin_dir / "lmdeploy"
    plain.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    plain.chmod(0o644)  # exists, NOT executable
    row = {"version": "v1", "venv_path": str(install)}
    monkeypatch.setattr(launch_spec, "_active_row", lambda _engine: dict(row))
    model = {"huggingface_id": "org/model"}
    with pytest.raises(launch_spec.LaunchCompileError):
        profile_launch.compile_profile_launch(model, {}, "model-stable", engine)
