"""Run engine binaries --help and persist parsed params into the catalog."""

from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, Optional, Tuple

from backend.cli_help_parsers import (
    lmdeploy_params_to_sections,
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
)
from backend.engine_param_catalog import iso_now, upsert_version_entry
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _clear_llama_flags_cache() -> None:
    try:
        from backend.llama_swap_config import clear_supported_flags_cache

        clear_supported_flags_cache()
    except Exception:
        pass

HELP_TIMEOUT = 90


def _abs_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.join("/app", p.lstrip("/"))


def _run_help_argv(
    argv: list,
    *,
    cwd: Optional[str] = None,
    extra_env: Optional[dict] = None,
) -> Tuple[str, Optional[str]]:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    try:
        r = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=HELP_TIMEOUT,
            cwd=cwd or None,
            env=env,
        )
        text = r.stdout or ""
        if not text.strip():
            return "", "empty help output"
        if r.returncode != 0:
            # Caller may still parse stdout when --help printed despite non-zero exit.
            return text, f"process exited with code {r.returncode}"
        return text, None
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except FileNotFoundError:
        return "", "binary not found"
    except Exception as e:
        return "", str(e)


def scan_llama_engine_version(engine: str, version_row: dict) -> dict:
    """engine: llama_cpp | ik_llama"""
    binary_path = version_row.get("binary_path")
    if not binary_path:
        return _error_entry("", "missing binary_path")
    path = _abs_path(binary_path)
    if not os.path.isfile(path):
        return _error_entry(path, f"binary not found: {path}")

    binary_dir = os.path.dirname(path)
    working_dir = binary_dir
    if "/bin/" in binary_dir and "/build/bin/" not in binary_dir:
        working_dir = binary_dir.replace("/bin/", "/build/bin/")
    env_ld = binary_dir
    if "/bin/" in env_ld and "/build/bin/" not in env_ld:
        env_ld = env_ld.replace("/bin/", "/build/bin/")

    text, run_err = _run_help_argv(
        [path, "--help"],
        cwd=working_dir if os.path.isdir(working_dir) else None,
        extra_env={"LD_LIBRARY_PATH": env_ld},
    )
    if not text.strip():
        return _error_entry(path, run_err or "empty help output")
    try:
        sections = parse_llama_help_to_sections(text, engine)
    except Exception as e:
        logger.exception("llama help parse failed")
        return _error_entry(path, f"parse error: {e}")

    n_params = sum(len(s.get("params") or []) for s in sections)
    if n_params == 0:
        msg = run_err or (
            "No CLI flags parsed from --help. If you only see GPU/CUDA lines, the binary may have exited "
            "before usage text was printed; try running it with --help in a shell."
        )
        return _error_entry(path, msg)

    return {
        "binary_path": path,
        "scanned_at": iso_now(),
        "scan_error": None,
        "sections": sections,
    }


def scan_lmdeploy_version(version_row: dict) -> dict:
    venv = version_row.get("venv_path")
    if not venv:
        return _error_entry("", "missing venv_path")
    vdir = _abs_path(venv)
    if not os.path.isdir(vdir):
        return _error_entry(vdir, f"venv not found: {vdir}")
    lmdeploy_bin = os.path.join(vdir, "bin", "lmdeploy")
    if not os.path.isfile(lmdeploy_bin) or not os.access(lmdeploy_bin, os.X_OK):
        return _error_entry(lmdeploy_bin, "lmdeploy binary missing or not executable")

    text, run_err = _run_help_argv(
        [lmdeploy_bin, "serve", "api_server", "--help"],
        cwd=vdir,
        extra_env={"VIRTUAL_ENV": vdir, "PATH": f"{os.path.join(vdir, 'bin')}:{os.environ.get('PATH', '')}"},
    )
    if not text.strip():
        return _error_entry(lmdeploy_bin, run_err or "empty help output")
    try:
        raw = parse_lmdeploy_api_server_help(text)
        sections = lmdeploy_params_to_sections(raw)
    except Exception as e:
        logger.exception("lmdeploy help parse failed")
        return _error_entry(lmdeploy_bin, f"parse error: {e}")

    n_params = sum(len(s.get("params") or []) for s in sections)
    if n_params == 0:
        msg = run_err or "No CLI flags parsed from lmdeploy serve api_server --help."
        return _error_entry(lmdeploy_bin, msg)

    return {
        "binary_path": lmdeploy_bin,
        "scanned_at": iso_now(),
        "scan_error": None,
        "sections": sections,
    }


def _error_entry(binary_path: str, message: str) -> dict:
    return {
        "binary_path": binary_path or None,
        "scanned_at": iso_now(),
        "scan_error": message,
        "sections": [],
    }


def scan_engine_version(store: Any, engine: str, version_row: dict) -> dict:
    """Parse --help and write catalog for this version row. Returns entry dict."""
    ver = version_row.get("version")
    if not ver:
        entry = _error_entry("", "version row missing version id")
        return entry

    if engine in ("llama_cpp", "ik_llama"):
        entry = scan_llama_engine_version(engine, version_row)
    elif engine == "lmdeploy":
        entry = scan_lmdeploy_version(version_row)
    else:
        entry = _error_entry("", f"unknown engine {engine}")

    try:
        upsert_version_entry(store, engine, str(ver), entry)
        _clear_llama_flags_cache()
    except Exception as e:
        logger.error("Failed to write param catalog: %s", e)
    return entry


def resolve_version_row(store: Any, engine: str, version: Optional[str]) -> Optional[dict]:
    """Pick version dict from engines.yaml."""
    versions = store.get_engine_versions(engine)
    if version:
        for v in versions:
            if v.get("version") == version:
                return v
        return None
    active = store.get_active_engine_version(engine)
    return active
