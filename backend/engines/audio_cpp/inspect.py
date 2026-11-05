"""Shared audio.cpp inspect argv / environment construction.

Installer (async) and profile scanner (sync) must probe packages the same way:
full library env, optional ``--model-spec-override``, prefer ``--inspect --json``.
The probe FLAGS are engine data (spec ``protocol.probe``), not constants here.
"""

from __future__ import annotations

from backend.engines.audio_cpp.identity import default_audio_engine_id

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend.engines.scan.help_parsers import parse_audio_cpp_inspection, try_parse_json_payload
from backend.runtime_env import audio_cpp_library_dirs, build_runtime_process_env


def probe_flags() -> Dict[str, str]:
    """Inspect-probe flags declared by the engine definition."""
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(default_audio_engine_id())
    protocol = dict(spec.protocol) if spec is not None else {}
    flags = dict(protocol.get("probe") or {})
    if not flags:
        raise RuntimeError("audio.cpp inspect probe flags are unavailable")
    return flags


def audio_cpp_inspect_env(cli_path: str, *, source_path: str = "") -> Dict[str, str]:
    return build_runtime_process_env(
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
    flags = probe_flags()
    argv: List[str] = [cli_path, flags["model_flag"], model_path]
    family_name = str(family or "").strip()
    if family_name:
        argv.extend([flags["family_flag"], family_name])
    override = str(model_spec_override or "").strip()
    if override:
        argv.extend([flags["spec_override_flag"], override])
    for key, value in (load_options or {}).items():
        if value is not None and str(value) != "":
            argv.extend([flags["load_option_flag"], f"{key}={value}"])
    return argv


def inspect_command_variants(base_argv: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Return (json_argv, text_argv) for inspect probes."""
    flags = probe_flags()
    base = list(base_argv)
    return (
        [*base, flags["inspect_flag"], flags["json_flag"]],
        [*base, flags["inspect_flag"]],
    )


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
