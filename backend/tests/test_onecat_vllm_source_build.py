"""Source-build toolchain helpers for 1Cat-vLLM (Rust bootstrap, build env).

NOTE: the requirement-file candidate discovery that used to live here as
``_source_requirement_files`` moved into the shared ``pip.install_requirements``
primitive (candidate order comes from the 1cat_vllm recipe). Its contract —
nested ``requirements/build/cuda.txt`` wins over the flat layouts, missing
files are skipped — is re-pinned in
``backend/tests/test_onecat_legacy_plan_shadow.py`` through the executed plan.
"""

import os
from pathlib import Path

import pytest

from backend.engines.python import rust_toolchain
from backend.engines.python.rust_toolchain import (
    compose_rust_env,
    discover_rust_bin_dir,
    ensure_rust,
)
from backend.tests.venv_fixtures import OneCatVllmManager


def _manager(tmp_path: Path) -> OneCatVllmManager:
    return OneCatVllmManager(
        log_path=str(tmp_path / "onecat.log"),
        state_path=str(tmp_path / "onecat_state.json"),
        base_dir=str(tmp_path / "1cat-vllm"),
    )


def _candidates(tmp_path: Path) -> list:
    return [str(tmp_path / "tools" / "rust" / "cargo" / "bin")]


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def _isolate_rust_search(manager: OneCatVllmManager, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "no-rust-on-path"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CARGO_HOME", raising=False)
    monkeypatch.delenv("RUSTUP_HOME", raising=False)


def test_discover_rust_bin_dir_finds_managed_toolchain(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)
    cargo_bin = tmp_path / "tools" / "rust" / "cargo" / "bin"
    _write_executable(cargo_bin / "rustc")
    _write_executable(cargo_bin / "cargo")

    found = discover_rust_bin_dir(
        {"PATH": str(tmp_path / "no-rust-on-path")},
        manager._managed_tools_dir("rust"),
        candidates=_candidates(tmp_path),
    )
    assert found is not None
    assert Path(found).resolve() == cargo_bin.resolve()


def test_build_env_moves_readonly_rustup_home(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)
    rustup = tmp_path / "usr-local-rustup"
    cargo = tmp_path / "usr-local-cargo"
    rustup.mkdir()
    cargo.mkdir()
    rustup.chmod(0o555)
    cargo.chmod(0o555)
    monkeypatch.setenv("RUSTUP_HOME", str(rustup))
    monkeypatch.setenv("CARGO_HOME", str(cargo))
    try:
        env = compose_rust_env(
            manager,
            {
                "PATH": str(tmp_path / "no-rust-on-path"),
                "HOME": str(tmp_path / "home"),
                "RUSTUP_HOME": str(rustup),
                "CARGO_HOME": str(cargo),
            },
            manager._managed_tools_dir("rust"),
            candidates=_candidates(tmp_path),
        )
    finally:
        rustup.chmod(0o755)
        cargo.chmod(0o755)

    managed = tmp_path / "tools" / "rust"
    assert Path(env["RUSTUP_HOME"]).resolve() == (managed / "rustup").resolve()
    assert Path(env["CARGO_HOME"]).resolve() == (managed / "cargo").resolve()
    assert os.access(env["RUSTUP_HOME"], os.W_OK)
    assert os.access(env["CARGO_HOME"], os.W_OK)


def test_build_env_prepends_managed_cargo_bin(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)
    cargo_bin = tmp_path / "tools" / "rust" / "cargo" / "bin"
    _write_executable(cargo_bin / "rustc")
    _write_executable(cargo_bin / "cargo")

    env = compose_rust_env(
        manager,
        {"PATH": str(tmp_path / "no-rust-on-path"), "HOME": str(tmp_path / "home")},
        manager._managed_tools_dir("rust"),
        candidates=_candidates(tmp_path),
    )
    path_parts = [Path(p).resolve() for p in env["PATH"].split(os.pathsep) if p]
    assert cargo_bin.resolve() in path_parts
    assert Path(env["CARGO_HOME"]).resolve() == (tmp_path / "tools" / "rust" / "cargo").resolve()


@pytest.mark.asyncio
async def test_ensure_rust_skips_install_when_present(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)
    cargo_bin = tmp_path / "tools" / "rust" / "cargo" / "bin"
    _write_executable(cargo_bin / "rustc")
    _write_executable(cargo_bin / "cargo")

    calls = []

    async def fake_run_logged(*args, **kwargs):
        calls.append(args)
        return 0

    async def fake_broadcast(*args, **kwargs):
        return None

    monkeypatch.setattr(manager, "_run_logged", fake_run_logged)
    monkeypatch.setattr(manager, "_broadcast_log_line", fake_broadcast)

    env = compose_rust_env(
        manager,
        {"PATH": str(tmp_path / "no-rust-on-path"), "HOME": str(tmp_path / "home")},
        manager._managed_tools_dir("rust"),
        candidates=_candidates(tmp_path),
    )
    await ensure_rust(
        manager, env, manager._managed_tools_dir("rust"), candidates=_candidates(tmp_path)
    )
    assert calls == []
    assert str(cargo_bin.resolve()) in env["PATH"]


@pytest.mark.asyncio
async def test_ensure_rust_bootstraps_when_missing(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)

    async def fake_run_logged(argv, operation, **kwargs):
        if argv and argv[0] == "curl":
            installer = Path(argv[-1])
            installer.parent.mkdir(parents=True, exist_ok=True)
            installer.write_text("#!/bin/sh\n")
        elif argv and argv[0] == "sh":
            cargo_bin = tmp_path / "tools" / "rust" / "cargo" / "bin"
            _write_executable(cargo_bin / "rustc")
            _write_executable(cargo_bin / "cargo")
        return 0

    async def fake_broadcast(*args, **kwargs):
        return None

    monkeypatch.setattr(manager, "_run_logged", fake_run_logged)
    monkeypatch.setattr(manager, "_broadcast_log_line", fake_broadcast)

    env = {"PATH": str(tmp_path / "no-rust-on-path"), "HOME": str(tmp_path / "home")}
    await ensure_rust(
        manager, env, manager._managed_tools_dir("rust"), candidates=_candidates(tmp_path)
    )
    cargo_bin = (tmp_path / "tools" / "rust" / "cargo" / "bin").resolve()
    assert cargo_bin in [Path(p).resolve() for p in env["PATH"].split(os.pathsep) if p]
    found = discover_rust_bin_dir(
        env, manager._managed_tools_dir("rust"), candidates=_candidates(tmp_path)
    )
    assert found is not None
    assert Path(found).resolve() == cargo_bin


@pytest.mark.asyncio
async def test_ensure_rust_raises_when_install_fails(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)

    async def fake_run_logged(*args, **kwargs):
        return 1

    async def fake_broadcast(*args, **kwargs):
        return None

    monkeypatch.setattr(manager, "_run_logged", fake_run_logged)
    monkeypatch.setattr(manager, "_broadcast_log_line", fake_broadcast)

    env = {"PATH": str(tmp_path / "no-rust-on-path"), "HOME": str(tmp_path / "home")}
    with pytest.raises(RuntimeError, match="rustup"):
        await ensure_rust(
            manager, env, manager._managed_tools_dir("rust"), candidates=_candidates(tmp_path)
        )
