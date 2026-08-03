"""Shared audio.cpp inspect argv / environment construction.

Installer (async) and profile scanner (sync) must probe packages the same way:
full library env, optional ``--model-spec-override``, prefer ``--inspect --json``.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.cli_help_parsers import parse_audio_cpp_inspection, try_parse_json_payload
from backend.runtime_env import audio_cpp_library_dirs, build_swap_process_env


def audio_cpp_inspect_env(cli_path: str, *, source_path: str = "") -> Dict[str, str]:
    return build_swap_process_env(
        {},
        library_dirs=audio_cpp_library_dirs(cli_path, source_path or None),
        include_cuda=True,
    )


def build_audio_cpp_inspect_argv(
    cli_path: str,
    model_path: str,
    *,
    family: Optional[str] = None,
    model_spec_override: Optional[str] = None,
    load_options: Optional[Dict[str, Any]] = None,
) -> List[str]:
    argv: List[str] = [cli_path, "--model", model_path]
    family_name = str(family or "").strip()
    if family_name:
        argv.extend(["--family", family_name])
    override = str(model_spec_override or "").strip()
    if override:
        argv.extend(["--model-spec-override", override])
    for key, value in (load_options or {}).items():
        if value is not None and str(value) != "":
            argv.extend(["--load-option", f"{key}={value}"])
    return argv


def inspect_command_variants(base_argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (json_argv, text_argv) for inspect probes."""
    base = list(base_argv)
    return [*base, "--inspect", "--json"], [*base, "--inspect"]


def parse_audio_inspect_text(text: str) -> dict:
    return parse_audio_cpp_inspection(text or "")


def select_inspect_payload(json_text: str, text_fallback: str = "") -> Tuple[str, dict]:
    """Prefer JSON inspect payload when parseable."""
    if try_parse_json_payload(json_text) is not None:
        parsed = parse_audio_inspect_text(json_text)
        if parsed.get("family") or parsed.get("task_names") or parsed.get("tasks"):
            return json_text, parsed
    parsed = parse_audio_inspect_text(text_fallback or json_text)
    return (text_fallback or json_text), parsed
