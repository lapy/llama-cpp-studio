"""Pure runtime adapter for one audio.cpp model behind llama-swap."""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Dict, List

from backend.audio_model_config import validate_audio_model_config
from backend.engine_param_catalog import get_version_entry, param_index_from_entry
from backend.engine_param_scanner import _audio_cpp_model_spec_override
from backend.feature_flags import audio_cpp_enabled
from backend.model_config import normalize_model_config
from backend.audio_voice_presets import (
    normalize_default_voice_preset,
    normalize_voice_presets,
)
from backend.reference_audio import reference_audio_storage_root
from backend.runtime_env import audio_cpp_library_dirs, build_swap_process_env


_STUDIO_FLAGS = {
    "--config",
    "--host",
    "--port",
    "--model",
    "--backend",
    "--device",
    "--threads",
    "--lazy-load",
    "--model-spec-override",
    "--model-spec",
}


def _sidecar_root() -> str:
    data_root = "/app/data" if os.path.isdir("/app/data") else os.path.abspath("data")
    return os.path.join(data_root, "config", "audio-cpp", "servers")


def _safe_sidecar_name(stable_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", stable_id).strip("-._")[:72]
    digest = hashlib.sha256(stable_id.encode("utf-8")).hexdigest()[:12]
    return f"{slug or 'audio-model'}-{digest}.json"


def _artifact_model_path(model: dict) -> str:
    artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
    path = str(
        artifact.get("path")
        or model.get("local_path")
        or model.get("model_path")
        or ""
    )
    return os.path.abspath(path) if path else ""


def _clean_options(value: Any) -> dict:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): nested
        for key, nested in value.items()
        if str(key) and nested is not None and nested != ""
    }


def _custom_args(raw: Any) -> List[str]:
    import shlex

    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        tokens = shlex.split(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid custom_args shell syntax: {exc}") from exc
    for token in tokens:
        if token.split("=", 1)[0] in _STUDIO_FLAGS:
            raise ValueError(
                f"{token.split('=', 1)[0]} is Studio-owned for audio.cpp"
            )
    return tokens


def _flag_tokens(key: str, value: Any, row: dict) -> List[str]:
    flag = row.get("primary_flag")
    negative = row.get("negative_flag")
    kind = row.get("value_kind") or "scalar"
    if kind == "flag":
        if value is True and flag:
            return [str(flag)]
        if value is False and negative:
            return [str(negative)]
        return []
    if value is None or value == "" or value == [] or not flag:
        return []
    if kind == "repeatable":
        output: List[str] = []
        for item in value if isinstance(value, list) else [value]:
            output.extend([str(flag), str(item)])
        return output
    return [str(flag), str(value)]


def _server_flag_tokens(store: Any, active: dict, config: dict) -> List[str]:
    entry = get_version_entry(store, "audio_cpp", str(active.get("version") or ""))
    index = param_index_from_entry(entry)
    output: List[str] = []
    for key, row in index.items():
        if row.get("reserved"):
            continue
        if str(row.get("scope") or "process") != "process":
            continue
        if str(row.get("transport") or "server_flag") != "server_flag":
            continue
        flag = str(row.get("primary_flag") or "")
        if flag in _STUDIO_FLAGS or key in {
            "config",
            "host",
            "port",
            "backend",
            "device",
            "threads",
            "lazy_load",
        }:
            continue
        output.extend(_flag_tokens(key, config.get(key), row))
    output.extend(_custom_args(config.get("custom_args")))
    return output


def _runtime_env(active: dict, config: dict) -> List[str]:
    raw = config.get("swap_env")
    user_env = {
        str(key): str(value)
        for key, value in (raw.items() if isinstance(raw, dict) else [])
        if str(key)
        and not str(key).startswith("LLAMA_STUDIO_")
        and value is not None
        and value != ""
    }
    server_binary = str(active.get("server_binary_path") or "")
    source_path = str(active.get("source_path") or "")
    build_backend = str(
        (active.get("build_config") or {}).get("backend") or "cpu"
    ).lower()
    include_cuda = build_backend == "cuda"
    env = build_swap_process_env(
        user_env,
        library_dirs=audio_cpp_library_dirs(server_binary, source_path),
        include_cuda=include_cuda,
    )
    return [f"{key}={env[key]}" for key in sorted(env)]


def build_audio_cpp_runtime(
    store: Any,
    model: dict,
    config: dict,
    stable_id: str,
) -> Dict[str, Any]:
    """Return command, environment, and generated sidecar content without writing."""

    if not audio_cpp_enabled():
        raise ValueError(
            "The audio.cpp integration is disabled by AUDIO_CPP_ENABLED"
        )
    active = store.get_active_engine_version("audio_cpp")
    if not active or not active.get("server_binary_path"):
        raise ValueError("No active audio.cpp server binary configured")
    server_binary = os.path.abspath(str(active["server_binary_path"]))
    if not os.path.isfile(server_binary):
        raise ValueError(f"audio.cpp server binary not found at: {server_binary}")
    model_path = _artifact_model_path(model)
    if not model_path or not os.path.isdir(model_path):
        raise ValueError("Prepared audio.cpp model directory does not exist")

    validate_audio_model_config(
        store,
        model,
        normalize_model_config(model.get("config")),
        allow_scan=False,
    )

    sidecar_path = os.path.join(
        _sidecar_root(),
        _safe_sidecar_name(stable_id),
    )
    lazy_load = bool(config.get("lazy_load", False))
    # Source builds are not AUDIOCPP_DEPLOYMENT_BUILD — package specs live under
    # the checkout's model_specs/. Mirror CLI inspect and inject that path so
    # llama-swap launches do not depend on process cwd.
    spec_override = str(config.get("model_spec_override") or "").strip()
    if not spec_override:
        discovered = _audio_cpp_model_spec_override(
            active,
            str(active.get("cli_binary_path") or server_binary),
        )
        if discovered and os.path.isdir(discovered):
            spec_override = discovered
    model_row: Dict[str, Any] = {
        "id": stable_id,
        "family": str(config["family"]),
        "path": model_path,
        "task": str(config["task"]),
        "mode": str(config["mode"]),
        "load_options": _clean_options(config.get("load_options")),
        "session_options": _clean_options(config.get("session_options")),
    }
    for key in ("config", "weight"):
        if config.get(key) not in (None, ""):
            model_row[key] = config[key]
    if spec_override:
        model_row["model_spec_override"] = spec_override
    if "model_lazy" in config:
        model_row["lazy"] = bool(config["model_lazy"])
    reference_root = reference_audio_storage_root(
        model_path,
        storage_key=model.get("id"),
    )
    presets = normalize_voice_presets(
        config.get("voice_presets"),
        model_root=model_path,
        reference_root=reference_root,
    )
    if presets:
        model_row["voice_presets"] = presets
    default_preset = normalize_default_voice_preset(
        config.get("default_voice_preset"),
        model_root=model_path,
        reference_root=reference_root,
        voice_presets=presets,
    )
    if default_preset is not None:
        model_row["default_voice_preset"] = default_preset

    sidecar = {
        "host": "127.0.0.1",
        "port": 8080,
        "backend": str(config.get("backend") or "cpu"),
        "device": int(config.get("device") or 0),
        "threads": max(1, int(config.get("threads") or 1)),
        "lazy_load": lazy_load,
        "models": [model_row],
    }
    if spec_override:
        sidecar["model_spec_override"] = spec_override
    argv = [
        server_binary,
        "--config",
        "${studio_audio_config}",
        "--host",
        "127.0.0.1",
        "--port",
        "${PORT}",
        *_server_flag_tokens(store, active, config),
    ]
    if spec_override:
        argv.extend(["--model-spec-override", spec_override])
    return {
        "cmd_argv": argv,
        "env": _runtime_env(active, config),
        "macros": {"studio_audio_config": sidecar_path},
        "sidecar_path": sidecar_path,
        "sidecar": sidecar,
        "use_model_name": stable_id,
        "generic_task_path": f"/upstream/{stable_id}/v1/tasks/run",
    }
