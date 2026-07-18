"""Run engine binaries --help and persist parsed params into the catalog."""

from __future__ import annotations

import os
import hashlib
import json
import subprocess
from typing import Any, List, Optional, Tuple

from backend.cli_help_parsers import (
    lmdeploy_params_to_sections,
    parse_audio_cpp_help_to_sections,
    parse_audio_cpp_inspection,
    parse_audio_cpp_loader_family_tasks,
    parse_audio_cpp_loader_list,
    parse_audio_cpp_loader_tasks,
    parse_audio_cpp_loaders_json,
    parse_llama_help_to_sections,
    parse_lmdeploy_api_server_help,
    parse_vllm_serve_help,
    try_parse_json_payload,
    vllm_params_to_sections,
)
from backend.engine_param_catalog import (
    get_model_profile_entry,
    get_version_entry,
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
    from backend.runtime_env import audio_cpp_library_dirs, build_swap_process_env

    return build_swap_process_env(
        {},
        library_dirs=audio_cpp_library_dirs(binary_path),
        include_cuda=True,
    )


def _audio_cpp_source_root(
    version_row: Optional[dict], cli_path: str
) -> Optional[str]:
    """Locate the audio.cpp checkout that contains ``model_specs/``.

    Non-deployment builds resolve package specs relative to the source tree, not
    the binary directory. Studio records ``source_path`` on version rows; when
    missing we walk parents of the CLI path.
    """
    candidates: List[str] = []
    if isinstance(version_row, dict):
        source = str(version_row.get("source_path") or "").strip()
        if source:
            candidates.append(os.path.abspath(source))
    cur = os.path.dirname(os.path.abspath(cli_path or ""))
    for _ in range(8):
        if not cur:
            break
        candidates.append(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    seen = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        if os.path.isdir(os.path.join(candidate, "model_specs")):
            return candidate
    return None


def _audio_cpp_workdir(version_row: Optional[dict], cli_path: str) -> str:
    root = _audio_cpp_source_root(version_row, cli_path)
    if root:
        return root
    return os.path.dirname(os.path.abspath(cli_path or "")) or "."


def _audio_cpp_model_spec_override(
    version_row: Optional[dict], cli_path: str
) -> Optional[str]:
    root = _audio_cpp_source_root(version_row, cli_path)
    if not root:
        return None
    return os.path.join(root, "model_specs")


def grade_audio_cpp_contract(
    *,
    loaders_source: str,
    catalog_source: str,
    catalog_identity: bool,
    family_tasks: Optional[dict] = None,
) -> str:
    """Return ``full`` | ``partial`` | ``thin`` for the active audio.cpp pin."""
    loaders_rich = str(loaders_source or "") == "json" and bool(family_tasks)
    catalog_rich = str(catalog_source or "") == "json" and bool(catalog_identity)
    if loaders_rich and catalog_rich:
        return "full"
    if loaders_rich or catalog_rich or str(loaders_source or "") == "json":
        return "partial"
    return "thin"


def _resolve_model_manager_path(version_row: dict) -> str:
    """Prefer the version row path; fall back to ``source_path/tools/model_manager.py``."""
    manager_path = str(version_row.get("model_manager_path") or "").strip()
    if manager_path and os.path.isfile(manager_path):
        return manager_path
    source_path = str(version_row.get("source_path") or "").strip()
    if source_path:
        candidate = os.path.join(source_path, "tools", "model_manager.py")
        if os.path.isfile(candidate):
            return candidate
    return ""


def _probe_catalog_contract(version_row: dict) -> dict:
    """Best-effort ``model_manager list --json`` identity probe (no AST fallback)."""
    manager_path = _resolve_model_manager_path(version_row)
    if not manager_path:
        return {
            "catalog_source": "missing",
            "catalog_identity": False,
            "catalog_package_count": 0,
            "warning": None,
        }
    helper_venv = str(version_row.get("helper_venv_path") or "").strip()
    python_name = "python.exe" if os.name == "nt" else "python"
    python_subdir = "Scripts" if os.name == "nt" else "bin"
    python_bin = None
    if helper_venv:
        candidate = os.path.join(helper_venv, python_subdir, python_name)
        if os.path.isfile(candidate):
            python_bin = candidate
    if not python_bin:
        from backend.audio_cpp_manager import _data_root

        default_venv = os.path.join(
            _data_root(), "audio-cpp", "tools", "model-manager-venv"
        )
        candidate = os.path.join(default_venv, python_subdir, python_name)
        if os.path.isfile(candidate):
            python_bin = candidate
    if not python_bin:
        import sys

        python_bin = sys.executable
    try:
        process = subprocess.run(
            [python_bin, manager_path, "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=60,
            cwd=os.path.dirname(manager_path),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {
            "catalog_source": "missing",
            "catalog_identity": False,
            "catalog_package_count": 0,
            "warning": f"catalog probe failed: {exc}",
        }
    if process.returncode != 0:
        return {
            "catalog_source": "ast_fallback_needed",
            "catalog_identity": False,
            "catalog_package_count": 0,
            "warning": (process.stderr or "").strip()[-400:] or "model_manager list --json failed",
        }
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError:
        return {
            "catalog_source": "ast_fallback_needed",
            "catalog_identity": False,
            "catalog_package_count": 0,
            "warning": "model_manager list --json returned non-JSON",
        }
    if not isinstance(payload, list) or not payload:
        return {
            "catalog_source": "json",
            "catalog_identity": False,
            "catalog_package_count": 0,
            "warning": "model_manager catalog JSON was empty",
        }
    identity_keys = ("family", "standalone", "tasks", "gated")
    sample = next((row for row in payload if isinstance(row, dict)), {})
    identity = all(key in sample for key in identity_keys)
    return {
        "catalog_source": "json",
        "catalog_identity": identity,
        "catalog_package_count": len(payload),
        "warning": (
            None
            if identity
            else "model_manager JSON lacks family/standalone/tasks/gated identity fields"
        ),
    }


def compute_audio_cpp_capability_delta(
    previous: Optional[dict], current: Optional[dict]
) -> dict:
    prev_caps = (previous or {}).get("capabilities") or {}
    curr_caps = (current or {}).get("capabilities") or {}
    prev_families = set(prev_caps.get("families") or [])
    curr_families = set(curr_caps.get("families") or [])
    prev_tasks = set(prev_caps.get("tasks") or [])
    curr_tasks = set(curr_caps.get("tasks") or [])
    curr_family_tasks = (
        curr_caps.get("family_tasks")
        if isinstance(curr_caps.get("family_tasks"), dict)
        else {}
    )
    empty_task_families = sorted(
        family
        for family in curr_families
        if not (curr_family_tasks.get(family) or curr_family_tasks.get(str(family).lower()))
    )
    return {
        "added_families": sorted(curr_families - prev_families),
        "removed_families": sorted(prev_families - curr_families),
        "added_tasks": sorted(curr_tasks - prev_tasks),
        "removed_tasks": sorted(prev_tasks - curr_tasks),
        "contract_grade": curr_caps.get("contract_grade"),
        "previous_contract_grade": prev_caps.get("contract_grade"),
        "catalog_source": curr_caps.get("catalog_source"),
        "discovery_source": curr_caps.get("discovery_source"),
        "families_without_tasks": empty_task_families,
        "heuristic_fallbacks_used": curr_caps.get("contract_grade") in {None, "thin", "partial"},
        "warnings": list(curr_caps.get("contract_warnings") or []),
    }


def _run_audio_cpp_loaders(
    cli_path: str, *, cwd: Optional[str] = None
) -> Tuple[str, Optional[str], str]:
    """Prefer ``--list-loaders --json``; fall back to text listing."""
    workdir = cwd or os.path.dirname(cli_path)
    env = _audio_env(cli_path)
    json_text, json_error = _run_help_argv(
        [cli_path, "--list-loaders", "--json"],
        cwd=workdir,
        extra_env=env,
        scan_engine="audio_cpp",
    )
    if try_parse_json_payload(json_text) is not None:
        return json_text, json_error, "json"
    text, error = _run_help_argv(
        [cli_path, "--list-loaders"],
        cwd=workdir,
        extra_env=env,
        scan_engine="audio_cpp",
    )
    return text, error or json_error, "text"


def _run_audio_cpp_inspect(
    base_argv: list, cli_path: str, *, cwd: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Prefer ``--inspect --json``; fall back to key=value text."""
    workdir = cwd or os.path.dirname(cli_path)
    env = _audio_env(cli_path)
    json_text, json_error = _run_help_argv(
        [*base_argv, "--inspect", "--json"],
        cwd=workdir,
        extra_env=env,
        scan_engine="audio_cpp",
    )
    if try_parse_json_payload(json_text) is not None:
        return json_text, json_error
    text, error = _run_help_argv(
        [*base_argv, "--inspect"],
        cwd=workdir,
        extra_env=env,
        scan_engine="audio_cpp",
    )
    return text, error or json_error


def scan_audio_cpp_version(version_row: dict) -> dict:
    server_path = _abs_audio_path(str(version_row.get("server_binary_path") or ""))
    cli_path = _abs_audio_path(str(version_row.get("cli_binary_path") or ""))
    for name, path in (("server", server_path), ("CLI", cli_path)):
        if not path:
            return _error_entry("", f"missing audio.cpp {name} binary path")
        if not os.path.isfile(path) or not os.access(path, os.X_OK):
            return _error_entry(path, f"audio.cpp {name} binary missing or not executable")

    workdir = _audio_cpp_workdir(version_row, cli_path)
    server_text, server_error = _run_help_argv(
        [server_path, "--help"],
        cwd=workdir,
        extra_env=_audio_env(server_path),
        scan_engine="audio_cpp",
    )
    cli_text, cli_error = _run_help_argv(
        [cli_path, "--help"],
        cwd=workdir,
        extra_env=_audio_env(cli_path),
        scan_engine="audio_cpp",
    )
    loaders_text, loaders_error, loaders_source = _run_audio_cpp_loaders(
        cli_path, cwd=workdir
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
        loaders_payload = try_parse_json_payload(loaders_text)
        loaders_meta = (
            parse_audio_cpp_loaders_json(loaders_payload)
            if loaders_payload is not None
            else {}
        )
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
    family_task_map = parse_audio_cpp_loader_family_tasks(loaders_text)
    catalog_probe = _probe_catalog_contract(version_row)
    contract_grade = grade_audio_cpp_contract(
        loaders_source=loaders_source,
        catalog_source=str(catalog_probe.get("catalog_source") or ""),
        catalog_identity=bool(catalog_probe.get("catalog_identity")),
        family_tasks=family_task_map,
    )
    contract_warnings = [
        message
        for message in (
            (
                None
                if loaders_source == "json"
                else "Loader catalog is text-only; prefer audiocpp_cli --list-loaders --json"
            ),
            catalog_probe.get("warning"),
            (
                None
                if contract_grade == "full"
                else f"audio.cpp contract grade is '{contract_grade}' (heuristics may apply)"
            ),
        )
        if message
    ]
    fingerprint = hashlib.sha256(
        (server_text + "\n" + cli_text + "\n" + loaders_text).encode("utf-8")
    ).hexdigest()
    previous_fp = str(version_row.get("contract_fingerprint") or "").strip()
    contract_changed = bool(previous_fp and previous_fp != fingerprint)
    capabilities = {
        "families": families,
        "tasks": tasks,
        "family_tasks": family_task_map,
        "discovery_source": loaders_source,
        "catalog_source": catalog_probe.get("catalog_source"),
        "catalog_identity": bool(catalog_probe.get("catalog_identity")),
        "catalog_package_count": int(catalog_probe.get("catalog_package_count") or 0),
        "contract_grade": contract_grade,
        "contract_warnings": contract_warnings,
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
    }
    if loaders_meta.get("family_modes"):
        capabilities["family_modes"] = loaders_meta["family_modes"]
    if loaders_meta.get("family_policies"):
        capabilities["family_policies"] = loaders_meta["family_policies"]
    if loaders_meta.get("family_endpoints"):
        capabilities["family_endpoints"] = loaders_meta["family_endpoints"]
    return {
        "binary_path": server_path,
        "server_binary_path": server_path,
        "cli_binary_path": cli_path,
        "scanned_at": iso_now(),
        "scan_error": None,
        "sections": sections,
        "profiles": {},
        "capabilities": capabilities,
        "contract_fingerprint": fingerprint,
        "previous_contract_fingerprint": previous_fp or None,
        "contract_changed": contract_changed,
        "warnings": [
            message
            for message in (
                server_error,
                cli_error,
                loaders_error,
                (
                    "audio.cpp CLI/help contract fingerprint changed since the last scan"
                    if contract_changed
                    else None
                ),
                *contract_warnings,
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
        # Failed scans are persisted for UI visibility but must not stick forever —
        # retry on the next lazy load so a flaky install-time inspect can recover.
        if cached and not cached.get("scan_error"):
            return cached

    cli_path = _abs_audio_path(str(version_row.get("cli_binary_path") or ""))
    model_path = _audio_model_path(model)
    if not cli_path or not os.path.isfile(cli_path):
        profile = {
            **_error_entry(cli_path, "audio.cpp CLI binary unavailable"),
            "fingerprint": fingerprint,
        }
        upsert_model_profile_entry(store, "audio_cpp", version, fingerprint, profile)
        return profile
    if not model_path or not os.path.exists(model_path):
        profile = {
            **_error_entry(model_path, "prepared audio.cpp model path unavailable"),
            "fingerprint": fingerprint,
        }
        upsert_model_profile_entry(store, "audio_cpp", version, fingerprint, profile)
        return profile

    workdir = _audio_cpp_workdir(version_row, cli_path)
    family = str(model.get("family") or "").strip()
    base_argv = [cli_path, "--model", model_path]
    if family:
        base_argv.extend(["--family", family])
    spec_override = _audio_cpp_model_spec_override(version_row, cli_path)
    if spec_override:
        # Prefer explicit override so scans work even when cwd is wrong.
        base_argv.extend(["--model-spec-override", spec_override])
    config = model.get("config") if isinstance(model.get("config"), dict) else {}
    engine_config = ((config.get("engines") or {}).get("audio_cpp") or {}) if isinstance(config, dict) else {}
    for key, value in (engine_config.get("load_options") or {}).items():
        if value is not None and str(value) != "":
            base_argv.extend(["--load-option", f"{key}={value}"])
    # User/config override wins over discovered source model_specs.
    config_spec = str(engine_config.get("model_spec_override") or "").strip()
    if config_spec:
        if "--model-spec-override" in base_argv:
            idx = base_argv.index("--model-spec-override")
            base_argv[idx + 1] = config_spec
        else:
            base_argv.extend(["--model-spec-override", config_spec])

    inspect_text, inspect_error = _run_audio_cpp_inspect(
        base_argv, cli_path, cwd=workdir
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
        cwd=workdir,
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
        family_name = str(
            inspection.get("family") or family or model.get("family") or ""
        ).strip()
        discovered: List[dict] = []
        discovery_root = _audio_cpp_source_root(version_row, cli_path)
        if discovery_root and family_name:
            try:
                from backend.audio_cpp_option_discovery import (
                    discover_family_options,
                    merge_discovered_options_into_sections,
                )

                discovered = discover_family_options(discovery_root, family_name)
                if discovered:
                    # Keep identity-free help sections, merge gaps, then re-add identity.
                    sections = merge_discovered_options_into_sections(
                        sections, discovered
                    )
            except Exception:
                logger.exception(
                    "audio.cpp source option discovery failed for family=%s",
                    family_name,
                )
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
            "discovered_option_count": len(discovered),
            "discovery_source_root": discovery_root,
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
        previous = get_version_entry(store, engine, str(ver)) or {}
        row = dict(version_row)
        if previous.get("contract_fingerprint"):
            row["contract_fingerprint"] = previous.get("contract_fingerprint")
        entry = scan_audio_cpp_version(row)
        if isinstance(entry, dict) and not entry.get("scan_error"):
            entry["capability_delta"] = compute_audio_cpp_capability_delta(
                previous, entry
            )
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
