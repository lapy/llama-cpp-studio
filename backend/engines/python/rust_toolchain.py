"""Reusable Rust toolchain provisioning for source builds.

A definition that declares ``build.prepare_fork``/``toolchain.ensure`` with
``kind: rust`` gets this mechanism: discover an existing cargo+rustc, retarget
rustup/cargo homes that live on a read-only image prefix into Studio's data
root, and (when still missing) install rustup into that managed root.

Inputs are explicit: the host (for logged commands/progress), the environment
to mutate, the managed root path, and the rustup installer URL. Output is the
same environment with cargo/rustc on PATH.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, List, Optional

RUSTUP_INIT_URL = "https://sh.rustup.rs"


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _dir_writable(path: str) -> bool:
    return bool(path) and os.path.isdir(path) and os.access(path, os.W_OK)


def _bin_dir_has_rust(bin_dir: str) -> bool:
    if not bin_dir or not os.path.isdir(bin_dir):
        return False
    rustc = os.path.join(bin_dir, _exe("rustc"))
    cargo = os.path.join(bin_dir, _exe("cargo"))
    return all(os.path.isfile(path) and os.access(path, os.X_OK) for path in (rustc, cargo))


def _apply_bin_dir(host, env: Dict[str, str], bin_dir: str) -> None:
    host._prepend_path(env, bin_dir)
    cargo_home = os.path.dirname(os.path.abspath(bin_dir))
    if _dir_writable(cargo_home):
        env.setdefault("CARGO_HOME", cargo_home)
    rustup_home = os.path.join(os.path.dirname(cargo_home), "rustup")
    if _dir_writable(rustup_home):
        env.setdefault("RUSTUP_HOME", rustup_home)


def _retarget_unwritable_homes(managed_root: str, env: Dict[str, str]) -> None:
    current_rustup = env.get("RUSTUP_HOME") or ""
    current_cargo = env.get("CARGO_HOME") or ""
    if current_rustup and os.path.isdir(current_rustup) and not _dir_writable(current_rustup):
        rustup_home = os.path.join(managed_root, "rustup")
        os.makedirs(rustup_home, exist_ok=True)
        env["RUSTUP_HOME"] = rustup_home
    if current_cargo and os.path.isdir(current_cargo) and not _dir_writable(current_cargo):
        cargo_home = os.path.join(managed_root, "cargo")
        os.makedirs(cargo_home, exist_ok=True)
        env["CARGO_HOME"] = cargo_home


def discover_rust_bin_dir(
    env: Optional[Dict[str, str]],
    managed_root: str,
    *,
    candidates: Optional[List[str]] = None,
) -> Optional[str]:
    """Return a directory that contains both ``rustc`` and ``cargo``."""
    env = env if env is not None else os.environ
    which_rustc = shutil.which(_exe("rustc"), path=env.get("PATH", ""))
    if which_rustc:
        which_dir = os.path.dirname(os.path.abspath(which_rustc))
        if _bin_dir_has_rust(which_dir):
            return which_dir
    home = env.get("HOME") or os.path.expanduser("~")
    cargo_home = env.get("CARGO_HOME") or ""
    if candidates is None:
        candidates = [
            os.path.join(managed_root, "cargo", "bin"),
            os.path.join(cargo_home, "bin") if cargo_home else "",
            os.path.join(home, ".cargo", "bin"),
            "/usr/local/cargo/bin",
            "/root/.cargo/bin",
        ]
    for candidate in candidates:
        if _bin_dir_has_rust(candidate):
            return candidate
    return None


def compose_rust_env(
    host,
    env: Dict[str, str],
    managed_root: str,
    *,
    candidates: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Place a discovered toolchain on PATH, or point rustup at the managed root."""
    _retarget_unwritable_homes(managed_root, env)
    rust_bin = discover_rust_bin_dir(env, managed_root, candidates=candidates)
    if rust_bin:
        _apply_bin_dir(host, env, rust_bin)
        return env
    cargo_home = os.path.join(managed_root, "cargo")
    rustup_home = os.path.join(managed_root, "rustup")
    env.setdefault("CARGO_HOME", cargo_home)
    env.setdefault("RUSTUP_HOME", rustup_home)
    host._prepend_path(env, os.path.join(cargo_home, "bin"))
    return env


async def _rustc_usable(host, env: Dict[str, str]) -> bool:
    try:
        return await host._run_logged(["rustc", "--version"], "rust_check", env=env) == 0
    except Exception:  # noqa: BLE001 — a missing shim is just "not usable"
        return False


async def ensure_rust(
    host,
    env: Dict[str, str],
    managed_root: str,
    *,
    installer_url: str = RUSTUP_INIT_URL,
    candidates: Optional[List[str]] = None,
) -> None:
    """Ensure cargo/rustc are available, installing rustup into ``managed_root``."""
    _retarget_unwritable_homes(managed_root, env)
    rust_bin = discover_rust_bin_dir(env, managed_root, candidates=candidates)
    if rust_bin:
        _apply_bin_dir(host, env, rust_bin)
        # rustup shims can exist without a default toolchain (a fresh or
        # half-installed rustup). Select stable once so cargo/rustc work.
        rustup_present = bool(shutil.which("rustup", path=env.get("PATH") or None))
        if rustup_present and not await _rustc_usable(host, env):
            code = await host._run_logged(["rustup", "default", "stable"], "install_rust", env=env)
            if code != 0:
                raise RuntimeError(
                    "Rustup is present but has no usable toolchain; install a stable "
                    "Rust toolchain (rustup default stable) and retry."
                )
            rust_bin = discover_rust_bin_dir(env, managed_root, candidates=candidates) or rust_bin
            _apply_bin_dir(host, env, rust_bin)
        await host._broadcast_log_line(f"Using Rust toolchain at {rust_bin}")
        return

    cargo_home = os.path.join(managed_root, "cargo")
    rustup_home = os.path.join(managed_root, "rustup")
    os.makedirs(cargo_home, exist_ok=True)
    os.makedirs(rustup_home, exist_ok=True)
    env["CARGO_HOME"] = cargo_home
    env["RUSTUP_HOME"] = rustup_home
    host._prepend_path(env, os.path.join(cargo_home, "bin"))
    await host._broadcast_log_line(
        "Rust toolchain not found; installing rustup (required to build the 1Cat-vLLM wheel)."
    )
    installer = os.path.join(managed_root, "rustup-init.sh")
    code = await host._run_logged(
        ["curl", "--proto", "=https", "--tlsv1.2", "-sSfL", installer_url, "-o", installer],
        "install_rust",
        env=env,
    )
    if code != 0 or not os.path.isfile(installer):
        raise RuntimeError(
            "Failed to download rustup. 1Cat-vLLM source builds need "
            "cargo/rustc (https://rustup.rs)."
        )
    os.chmod(installer, 0o755)
    install_code = await host._run_logged(
        [
            "sh",
            installer,
            "-y",
            "--profile",
            "minimal",
            "--default-toolchain",
            "stable",
            "--no-modify-path",
        ],
        "install_rust",
        env=env,
    )
    rust_bin = discover_rust_bin_dir(env, managed_root)
    if install_code != 0 or not rust_bin:
        raise RuntimeError(
            "Rust toolchain is required to build 1Cat-vLLM from source "
            "(cargo/rustc not found). Install rustup from https://rustup.rs "
            "and retry."
        )
    _apply_bin_dir(host, env, rust_bin)
    await host._broadcast_log_line(f"Rust toolchain installed at {rust_bin}")
