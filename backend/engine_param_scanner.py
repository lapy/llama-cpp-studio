"""Run engine binaries --help and persist parsed params into the catalog."""

from __future__ import annotations

import os
import hashlib
import json
import subprocess
from typing import Any, Optional, Tuple

from backend.cli_help_parsers import (
    lmdeploy_params_to_sections,
    parse_audio_cpp_help_to_sections,
    parse_audio_cpp_inspection,
    parse_audio_cpp_loader_list,
    parse_audio_cpp_loader_tasks,
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
    parse_vllm_serve_help,
    vllm_params_to_sections,
)
from backend.engine_param_catalog import (
    get_model_profile_entry,
    iso_now,
    upsert_model_profile_entry,
    upsert_version_entry,
)
from backend.logging_config import get_logger

logger = get_logger(__name__)


def _clear_llama_flags_cache() -> None:
    try:
        from backend.llama_swap_config import clear_supported_flags_cache

        clear_supported_flags_cache()
    except Exception:
        pass


HELP_TIMEOUT = 90


def _help_subprocess_failure_message(
    returncode: int,
    argv0: str,
    *,
    empty_stdout: bool,
    scan_engine: Optional[str] = None,
) -> str:
    """Human-readable scan failure (126/127 often mean exec/loader/shebang issues)."""
    exe = argv0 or "(unknown)"
    tail = (
        " (no stdout from --help; output may be missing or only on stderr)"
        if empty_stdout
        else ""
    )
    if returncode == 127:
        head = (
            f"process exited with code 127{tail}: the program could not be run (POSIX 127 — often "
            f"“not found” at exec or in a wrapper). Executable: {exe}. "
        )
        if scan_engine == "lmdeploy":
            return head + (
                "For LMDeploy: `lmdeploy` is usually a script; fix the venv shebang Python or a stale `venv_path`."
            )
        if scan_engine in ("llama_cpp", "ik_llama"):
            return head + (
                "For llama.cpp / ik_llama: wrong arch or libc (e.g. glibc binary on musl), missing shared "
                "libraries (CUDA/GGML — `.so` search path), bad `binary_path`, or a wrapper with a broken "
                "shebang. Run `file` on the binary and the same `--help` in the API container; ensure "
                "`LD_LIBRARY_PATH` includes the directory with ggml/llama shared libs (often `build/bin` next to the build)."
            )
        return head + "Check shebang, dynamic linker, and PATH/LD_LIBRARY_PATH."
    if returncode == 126:
        return (
            f"process exited with code 126{tail}: cannot execute (permission denied or not a valid executable). "
            f"Executable: {exe}"
        )
    return f"process exited with code {returncode}{tail}"


def _abs_path(p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.join("/app", p.lstrip("/"))


def _abs_audio_path(path: str) -> str:
    if not path or os.path.isabs(path):
        return path
    if os.path.isdir("/app/data"):
        return os.path.normpath(os.path.join("/app", path))
    return os.path.abspath(path)


def _run_help_argv(
    argv: list,
    *,
    cwd: Optional[str] = None,
    extra_env: Optional[dict] = None,
    scan_engine: Optional[str] = None,
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
        argv0 = argv[0] if argv else ""
        if not text.strip():
            if r.returncode != 0:
                return "", _help_subprocess_failure_message(
                    r.returncode, argv0, empty_stdout=True, scan_engine=scan_engine
                )
            return "", "empty help output"
        if r.returncode != 0:
            # Caller may still parse stdout when --help printed despite non-zero exit.
            return text, _help_subprocess_failure_message(
                r.returncode, argv0, empty_stdout=False, scan_engine=scan_engine
            )
        return text, None
    except subprocess.TimeoutExpired:
        return "", "timeout"
    except FileNotFoundError:
        return "", "binary not found"
    except Exception as e:
        return "", str(e)


def scan_llama_engine_version(engine: str, version_row: dict) -> dict:
    """engine: llama_cpp | ik_llama"""
    from backend.llama_server_exec import (
        llama_help_ld_library_path,
        resolve_llama_server_invocation_paths,
    )

    binary_path = version_row.get("binary_path")
    if not binary_path:
        return _error_entry("", "missing binary_path")
    path = _abs_path(binary_path)
    exec_path, work_cwd = resolve_llama_server_invocation_paths(path)
    if not os.path.isfile(exec_path):
        return _error_entry(exec_path, f"binary not found: {exec_path}")

    ld_path = llama_help_ld_library_path(work_cwd)

    text, run_err = _run_help_argv(
        [exec_path, "--help"],
        cwd=work_cwd if os.path.isdir(work_cwd) else None,
        extra_env={"LD_LIBRARY_PATH": ld_path},
        scan_engine=engine,
    )
    if not text.strip():
        return _error_entry(exec_path, run_err or "empty help output")
    try:
        sections = parse_llama_help_to_sections(text, engine)
    except Exception as e:
        logger.exception("llama help parse failed")
        return _error_entry(exec_path, f"parse error: {e}")

    n_params = sum(len(s.get("params") or []) for s in sections)
    if n_params == 0:
        msg = run_err or (
            "No CLI flags parsed from --help. If you only see GPU/CUDA lines, the binary may have exited "
            "before usage text was printed; try running it with --help in a shell."
        )
        return _error_entry(exec_path, msg)

    return {
        "binary_path": exec_path,
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
        extra_env={
            "VIRTUAL_ENV": vdir,
            "PATH": f"{os.path.join(vdir, 'bin')}:{os.environ.get('PATH', '')}",
        },
        scan_engine="lmdeploy",
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


def scan_onecat_vllm_version(version_row: dict) -> dict:
    """Scan ``vllm serve --help=all`` for 1Cat-vLLM."""
    venv = version_row.get("venv_path")
    if not venv:
        return _error_entry("", "missing venv_path")
    vdir = _abs_path(venv)
    if not os.path.isdir(vdir):
        return _error_entry(vdir, f"venv not found: {vdir}")
    python_bin = os.path.join(vdir, "bin", "python")
    if not os.path.isfile(python_bin) or not os.access(python_bin, os.X_OK):
        return _error_entry(python_bin, "venv python missing or not executable")

    vllm_bin = os.path.join(vdir, "bin", "vllm")
    if os.path.isfile(vllm_bin) and os.access(vllm_bin, os.X_OK):
        help_argv = [vllm_bin, "serve", "--help=all"]
        scan_binary = vllm_bin
    else:
        help_argv = [python_bin, "-m", "vllm", "serve", "--help=all"]
        scan_binary = python_bin

    text, run_err = _run_help_argv(
        help_argv,
        # Run from the venv (not a source checkout) so the installed package + its
        # CUDA extensions are imported, per the 1Cat-vLLM runtime notes.
        cwd=vdir,
        extra_env={
            "VIRTUAL_ENV": vdir,
            "PATH": f"{os.path.join(vdir, 'bin')}:{os.environ.get('PATH', '')}",
        },
        scan_engine="1cat_vllm",
    )
    if not text.strip():
        return _error_entry(scan_binary, run_err or "empty help output")
    try:
        raw = parse_vllm_serve_help(text)
        sections = vllm_params_to_sections(raw)
    except Exception as e:
        logger.exception("1Cat-vLLM help parse failed")
        return _error_entry(scan_binary, f"parse error: {e}")

    n_params = sum(len(s.get("params") or []) for s in sections)
    if n_params == 0:
        msg = run_err or "No CLI flags parsed from vllm serve --help=all."
        return _error_entry(scan_binary, msg)

    return {
        "binary_path": scan_binary,
        "scanned_at": iso_now(),
        "scan_error": None,
        "sections": sections,
    }


def _prefix_sections(sections: list, prefix: str) -> list:
    out = []
    for section in sections:
        row = dict(section)
        row["id"] = f"{prefix}_{section.get('id') or 'options'}"
        out.append(row)
    return out


def _audio_env(binary_path: str) -> dict:
    binary_dir = os.path.dirname(binary_path)
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    return {
        "LD_LIBRARY_PATH": os.pathsep.join(
            item for item in (binary_dir, existing) if item
        )
    }


def scan_audio_cpp_version(version_row: dict) -> dict:
    server_path = _abs_audio_path(str(version_row.get("server_binary_path") or ""))
    cli_path = _abs_audio_path(str(version_row.get("cli_binary_path") or ""))
    for name, path in (("server", server_path), ("CLI", cli_path)):
        if not path:
            return _error_entry("", f"missing audio.cpp {name} binary path")
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            return _error_entry(path, f"audio.cpp {name} binary missing or not executable")

    server_text, server_error = _run_help_argv(
        [server_path, "--help"],
        cwd=os.path.dirname(server_path),
        extra_env=_audio_env(server_path),
        scan_engine="audio_cpp",
    )
    cli_text, cli_error = _run_help_argv(
        [cli_path, "--help"],
        cwd=os.path.dirname(cli_path),
        extra_env=_audio_env(cli_path),
        scan_engine="audio_cpp",
    )
    loaders_text, loaders_error = _run_help_argv(
        [cli_path, "--list-loaders"],
        cwd=os.path.dirname(cli_path),
        extra_env=_audio_env(cli_path),
        scan_engine="audio_cpp",
    )
    if not server_text.strip():
        return _error_entry(server_path, server_error or "empty audiocpp_server help")
    if not cli_text.strip():
        return _error_entry(cli_path, cli_error or "empty audiocpp_cli help")
    if not loaders_text.strip():
        return _error_entry(cli_path, loaders_error or "empty audio.cpp loader list")

    try:
        server_sections = parse_audio_cpp_help_to_sections(
            server_text, source="server"
        )
        cli_sections = parse_audio_cpp_help_to_sections(cli_text, source="cli")
        families = parse_audio_cpp_loader_list(loaders_text)
    except Exception as exc:
        logger.exception("audio.cpp help parse failed")
        return _error_entry(cli_path, f"parse error: {exc}")

    sections = [
        *_prefix_sections(server_sections, "server"),
        *_prefix_sections(cli_sections, "cli"),
    ]
    param_count = sum(len(section.get("params") or []) for section in sections)
    if param_count == 0:
        return _error_entry(cli_path, "No audio.cpp options were parsed")
    if not families:
        return _error_entry(
            cli_path,
            "No audio.cpp loader families were parsed; refusing an unverified capability scan",
        )

    task_param = next(
        (
            param
            for section in cli_sections
            for param in section.get("params") or []
            if param.get("key") == "task"
        ),
        {},
    )
    tasks = [
        str(option.get("value"))
        for option in (task_param.get("options") or [])
        if option.get("value")
    ]
    if not tasks:
        tasks = parse_audio_cpp_loader_tasks(loaders_text)
    from backend.audio_cpp_manager import AUDIO_CPP_COMPATIBILITY_COMMIT

    source_commit = str(version_row.get("source_commit") or "")
    compatibility_warning = (
        None
        if not source_commit or source_commit == AUDIO_CPP_COMPATIBILITY_COMMIT
        else (
            f"Installed audio.cpp commit {source_commit[:12]} differs from the "
            f"tested parser contract {AUDIO_CPP_COMPATIBILITY_COMMIT[:12]}"
        )
    )
    return {
        "binary_path": server_path,
        "server_binary_path": server_path,
        "cli_binary_path": cli_path,
        "scanned_at": iso_now(),
        "scan_error": None,
        "sections": sections,
        "profiles": {},
        "capabilities": {
            "families": families,
            "tasks": tasks,
            "input_modalities": ["audio", "text"],
            "output_modalities": ["audio", "text", "segments", "events"],
            "endpoints": [
                "/health",
                "/v1/models",
                "/v1/audio/speech",
                "/v1/audio/transcriptions",
                "/v1/audio/voices",
                "/v1/tasks/run",
            ],
        },
        "contract_fingerprint": hashlib.sha256(
            (server_text + "\n" + cli_text + "\n" + loaders_text).encode("utf-8")
        ).hexdigest(),
        "compatibility_commit": AUDIO_CPP_COMPATIBILITY_COMMIT,
        "warnings": [
            message
            for message in (
                server_error,
                cli_error,
                loaders_error,
                compatibility_warning,
            )
            if message
        ],
    }


def audio_cpp_model_profile_fingerprint(version_row: dict, model: dict) -> str:
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    path = (
        artifact.get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    )
    resolved = _abs_audio_path(str(path)) if path else ""
    stat_payload: dict = {}
    try:
        stat = os.stat(resolved)
        stat_payload = {
            "mtime_ns": stat.st_mtime_ns,
            "size": stat.st_size,
        }
    except OSError:
        pass
    manifest = model.get("manifest") if isinstance(model.get("manifest"), dict) else {}
    payload = {
        "version": version_row.get("source_commit") or version_row.get("version"),
        "path": os.path.realpath(resolved) if resolved else "",
        "stat": stat_payload,
        "family": model.get("family"),
        "tasks": model.get("tasks") or [],
        "manifest_fingerprint": manifest.get("capability_fingerprint")
        or manifest.get("fingerprint"),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _audio_model_path(model: dict) -> str:
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    raw = (
        artifact.get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    )
    return _abs_audio_path(str(raw)) if raw else ""


def _audio_profile_identity_section(inspection: dict, model: dict) -> dict:
    task_options = [
        {"value": task["task"], "label": task["task"]}
        for task in inspection.get("tasks") or []
        if task.get("task")
    ]
    modes = []
    for task in inspection.get("tasks") or []:
        for mode in task.get("modes") or []:
            if mode not in modes:
                modes.append(mode)
    family = inspection.get("family") or model.get("family")
    params = [
        {
            "key": "family",
            "label": "Family",
            "description": "Inspected audio.cpp loader family",
            "type": "select",
            "value_kind": "enum",
            "scalar_type": "string",
            "options": [{"value": family, "label": family}] if family else [],
            "default": family,
            "scope": "model",
            "transport": "server_config",
            "emission": {"transport": "server_config", "key": "family"},
            "required": True,
            "reserved": False,
        },
        {
            "key": "task",
            "label": "Task",
            "description": "Task exposed by the inspected package",
            "type": "select",
            "value_kind": "enum",
            "scalar_type": "string",
            "options": task_options,
            "default": (model.get("task") or (task_options[0]["value"] if task_options else None)),
            "scope": "model",
            "transport": "server_config",
            "emission": {"transport": "server_config", "key": "task"},
            "required": True,
            "reserved": False,
        },
        {
            "key": "mode",
            "label": "Mode",
            "description": "Offline or streaming session mode",
            "type": "select",
            "value_kind": "enum",
            "scalar_type": "string",
            "options": [{"value": mode, "label": mode} for mode in modes],
            "default": model.get("mode") or ("offline" if "offline" in modes else (modes[0] if modes else None)),
            "scope": "model",
            "transport": "server_config",
            "emission": {"transport": "server_config", "key": "mode"},
            "required": True,
            "reserved": False,
        },
    ]
    for key, label in (("configs", "Config"), ("weights", "Weight")):
        singular = key[:-1]
        assets = inspection.get(key) or []
        if not assets:
            continue
        params.append(
            {
                "key": singular,
                "label": label,
                "description": f"Discovered {singular} asset",
                "type": "select",
                "value_kind": "enum",
                "scalar_type": "string",
                "options": [
                    {
                        "value": asset.get("id"),
                        "label": asset.get("id"),
                        "path": asset.get("path"),
                    }
                    for asset in assets
                ],
                "default": assets[0].get("id") if len(assets) == 1 else None,
                "scope": "model",
                "transport": "server_config",
                "emission": {"transport": "server_config", "key": singular},
                "required": False,
                "reserved": False,
                "asset_selector": True,
            }
        )
    return {"id": "model_identity", "label": "Model identity", "params": params}


def scan_audio_cpp_model_profile(
    store: Any,
    version_row: dict,
    model: dict,
    *,
    force: bool = False,
) -> dict:
    version = str(version_row.get("version") or "")
    fingerprint = audio_cpp_model_profile_fingerprint(version_row, model)
    if not force:
        cached = get_model_profile_entry(
            store, "audio_cpp", version, fingerprint
        )
        if cached:
            return cached

    cli_path = _abs_audio_path(str(version_row.get("cli_binary_path") or ""))
    model_path = _audio_model_path(model)
    if not cli_path or not os.path.isfile(cli_path):
        return _error_entry(cli_path, "audio.cpp CLI binary unavailable")
    if not model_path or not os.path.exists(model_path):
        return _error_entry(model_path, "prepared audio.cpp model path unavailable")

    family = str(model.get("family") or "").strip()
    base_argv = [cli_path, "--model", model_path]
    if family:
        base_argv.extend(["--family", family])
    config = model.get("config") if isinstance(model.get("config"), dict) else {}
    engine_config = ((config.get("engines") or {}).get("audio_cpp") or {}) if isinstance(config, dict) else {}
    for key, value in (engine_config.get("load_options") or {}).items():
        if value is not None and str(value) != "":
            base_argv.extend(["--load-option", f"{key}={value}"])

    inspect_text, inspect_error = _run_help_argv(
        [*base_argv, "--inspect"],
        cwd=os.path.dirname(cli_path),
        extra_env=_audio_env(cli_path),
        scan_engine="audio_cpp",
    )
    if not inspect_text.strip() or inspect_error:
        profile = {
            **_error_entry(cli_path, inspect_error or "empty audio.cpp inspection"),
            "fingerprint": fingerprint,
        }
        upsert_model_profile_entry(
            store, "audio_cpp", version, fingerprint, profile
        )
        return profile

    help_text, help_error = _run_help_argv(
        [*base_argv, "--help"],
        cwd=os.path.dirname(cli_path),
        extra_env=_audio_env(cli_path),
        scan_engine="audio_cpp",
    )
    if not help_text.strip():
        profile = {
            **_error_entry(cli_path, help_error or "empty model-aware help"),
            "fingerprint": fingerprint,
        }
        upsert_model_profile_entry(
            store, "audio_cpp", version, fingerprint, profile
        )
        return profile

    try:
        inspection = parse_audio_cpp_inspection(inspect_text)
        sections = parse_audio_cpp_help_to_sections(help_text, source="cli")
        applicability = {
            "families": [inspection.get("family")] if inspection.get("family") else [],
            "tasks": inspection.get("task_names") or [],
            "modes": {
                task.get("task"): task.get("modes") or []
                for task in inspection.get("tasks") or []
            },
        }
        for section in sections:
            for param in section.get("params") or []:
                param["applicability"] = applicability
        sections.insert(0, _audio_profile_identity_section(inspection, model))
        profile = {
            "fingerprint": fingerprint,
            "model_id": model.get("id"),
            "model_path": model_path,
            "engine_version": version,
            "scanned_at": iso_now(),
            "scan_error": help_error,
            "inspection": inspection,
            "sections": sections,
            "applicability": applicability,
        }
    except Exception as exc:
        logger.exception("audio.cpp model profile parse failed")
        profile = {
            **_error_entry(cli_path, f"model profile parse error: {exc}"),
            "fingerprint": fingerprint,
        }
    upsert_model_profile_entry(store, "audio_cpp", version, fingerprint, profile)
    return profile


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
        row = dict(version_row)
        active = store.get_active_engine_version(engine)
        if active and active.get("version") == ver and active.get("binary_path"):
            try:
                from backend.llama_engine_resolve import (
                    get_active_llama_swap_binary_path,
                    infer_llama_engine_for_binary,
                )

                swap_bin = get_active_llama_swap_binary_path(store)
                if (
                    swap_bin
                    and infer_llama_engine_for_binary(store, swap_bin) == engine
                ):
                    row["binary_path"] = swap_bin
            except Exception as e:
                logger.debug("Active llama-swap binary override skipped: %s", e)
        entry = scan_llama_engine_version(engine, row)
    elif engine == "lmdeploy":
        entry = scan_lmdeploy_version(version_row)
    elif engine == "1cat_vllm":
        entry = scan_onecat_vllm_version(version_row)
    elif engine == "audio_cpp":
        entry = scan_audio_cpp_version(version_row)
    else:
        entry = _error_entry("", f"unknown engine {engine}")

    try:
        upsert_version_entry(store, engine, str(ver), entry)
        _clear_llama_flags_cache()
    except Exception as e:
        logger.error("Failed to write param catalog: %s", e)
    return entry


def resolve_version_row(
    store: Any, engine: str, version: Optional[str]
) -> Optional[dict]:
    """Pick version dict from engines.yaml."""
    versions = store.get_engine_versions(engine)
    if version:
        for v in versions:
            if v.get("version") == version:
                return v
        return None
    active = store.get_active_engine_version(engine)
    return active
