"""Source-build toolchain helpers for 1Cat-vLLM (Rust + requirement files)."""

import os
from pathlib import Path

import pytest

from backend.onecat_vllm_manager import OneCatVllmManager


def _manager(tmp_path: Path) -> OneCatVllmManager:
    return OneCatVllmManager(
        log_path=str(tmp_path / "onecat.log"),
        state_path=str(tmp_path / "onecat_state.json"),
        base_dir=str(tmp_path / "1cat-vllm"),
    )


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)


def _isolate_rust_search(manager: OneCatVllmManager, monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "no-rust-on-path"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CARGO_HOME", raising=False)
    monkeypatch.delenv("RUSTUP_HOME", raising=False)
    monkeypatch.setattr(
        manager,
        "_rust_bin_candidates",
        lambda env: [str(tmp_path / "tools" / "rust" / "cargo" / "bin")],
    )


def test_source_requirement_files_prefers_nested_build_cuda(tmp_path: Path):
    clone = tmp_path / "src"
    nested = clone / "requirements" / "build" / "cuda.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("setuptools-rust>=1.9.0\n")
    (clone / "requirements" / "cuda.txt").write_text("torch==2.10.0\n")
    (clone / "requirements" / "common.txt").write_text("numpy\n")

    files = OneCatVllmManager._source_requirement_files(str(clone))
    assert files[0] == str(nested)
    assert any(p.endswith("requirements/cuda.txt") for p in files)
    assert not any(p.endswith("requirements/build.txt") for p in files)


def test_source_requirement_files_falls_back_to_flat_build_txt(tmp_path: Path):
    clone = tmp_path / "src"
    req = clone / "requirements"
    req.mkdir(parents=True)
    (req / "build.txt").write_text("cmake\n")
    (req / "cuda.txt").write_text("torch\n")

    files = OneCatVllmManager._source_requirement_files(str(clone))
    assert [Path(p).name for p in files] == ["build.txt", "cuda.txt"]


def test_discover_rust_bin_dir_finds_managed_toolchain(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)
    cargo_bin = tmp_path / "tools" / "rust" / "cargo" / "bin"
    _write_executable(cargo_bin / "rustc")
    _write_executable(cargo_bin / "cargo")

    found = manager._discover_rust_bin_dir({"PATH": str(tmp_path / "no-rust-on-path")})
    assert found is not None
    assert Path(found).resolve() == cargo_bin.resolve()


def test_build_env_prepends_managed_cargo_bin(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path)
    _isolate_rust_search(manager, monkeypatch, tmp_path)
    cargo_bin = tmp_path / "tools" / "rust" / "cargo" / "bin"
    _write_executable(cargo_bin / "rustc")
    _write_executable(cargo_bin / "cargo")

    env = manager._build_env()
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

    env = manager._build_env()
    await manager._ensure_rust(env)
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
    await manager._ensure_rust(env)
    cargo_bin = (tmp_path / "tools" / "rust" / "cargo" / "bin").resolve()
    assert cargo_bin in [Path(p).resolve() for p in env["PATH"].split(os.pathsep) if p]
    found = manager._discover_rust_bin_dir(env)
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
        await manager._ensure_rust(env)
