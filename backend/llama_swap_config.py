import os
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Set

import yaml

from backend.huggingface import resolve_gguf_model_path_for_quant
from backend.llama_engine_resolve import (
    abs_llama_binary_path as _abs_binary_path,
    get_active_llama_swap_binary_path,
    infer_llama_engine_for_binary,
)
from backend.llama_server_exec import (
    llama_help_ld_library_path,
    resolve_llama_server_invocation_paths,
)
from backend.logging_config import get_logger
from backend.model_config import effective_model_config_from_raw

logger = get_logger(__name__)

_supported_flags_cache: Dict[str, Set[str]] = {}

_ALLOWED_NONCANONICAL_KEYS = frozenset({"custom_args", "engine", "engines", "model_alias"})


def clear_supported_flags_cache() -> None:
    _supported_flags_cache.clear()


def infer_engine_id_for_binary(binary_path: str) -> str:
    """Resolve llama_cpp vs ik_llama from engines.yaml active rows."""
    try:
        from backend.data_store import get_store

        return infer_llama_engine_for_binary(get_store(), binary_path)
    except Exception as e:
        logger.debug("infer_engine_id_for_binary: %s", e)
    return "llama_cpp"


def _active_engine_entry(engine: str) -> Optional[dict]:
    from backend.data_store import get_store
    from backend.engine_param_catalog import get_version_entry

    store = get_store()
    active = store.get_active_engine_version(engine)
    if not active or not active.get("version"):
        return None
    return get_version_entry(store, engine, active["version"])


def _active_engine_param_index(engine: str) -> Dict[str, dict]:
    from backend.engine_param_catalog import param_index_from_entry

    return param_index_from_entry(_active_engine_entry(engine))


def resolve_llama_param_mapping_from_engine(engine: str) -> Dict[str, list]:
    from backend.engine_param_catalog import param_mapping_from_entry

    return param_mapping_from_entry(_active_engine_entry(engine))


def supported_flags_for_llama_binary(binary_path: str) -> Set[str]:
    """Prefer catalog flags for the active engine version; else parse --help."""
    eng = infer_engine_id_for_binary(binary_path)
    try:
        from backend.engine_param_catalog import flags_from_entry

        entry = _active_engine_entry(eng)
        flags = flags_from_entry(entry)
        if flags:
            norm = os.path.abspath(_abs_binary_path(binary_path))
            _supported_flags_cache[norm] = set(flags)
            return set(flags)
    except Exception as e:
        logger.debug("supported_flags_for_llama_binary catalog: %s", e)
    return get_supported_flags(binary_path)


def get_supported_flags(llama_server_path: str) -> Set[str]:
    """
    Get the set of supported flags for a llama-server binary by parsing --help output.
    Results are cached per binary path.
    """
    path = llama_server_path
    if not os.path.isabs(path):
        path = _abs_binary_path(path)
    exec_path, work_cwd = resolve_llama_server_invocation_paths(path)
    normalized_path = os.path.abspath(exec_path) if os.path.exists(exec_path) else exec_path
    if normalized_path in _supported_flags_cache:
        return _supported_flags_cache[normalized_path]

    supported_flags: Set[str] = set()
    try:
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = llama_help_ld_library_path(work_cwd)
        result = subprocess.run(
            [exec_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            cwd=work_cwd,
            env=env,
        )
        help_text = result.stdout or ""
        if not help_text.strip():
            logger.warning("No stdout from %s --help (exit %s)", exec_path, result.returncode)
        elif result.returncode != 0 and "--" not in help_text:
            logger.warning(
                "Unexpected --help exit %s from %s; stderr/stdout may be incomplete",
                result.returncode,
                exec_path,
            )
        else:
            supported_flags = set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", help_text))
    except subprocess.TimeoutExpired:
        logger.warning("Timeout getting help from %s", exec_path)
    except FileNotFoundError:
        logger.warning("Binary not found: %s", exec_path)
    except Exception as e:
        logger.warning("Error checking flags for %s: %s", exec_path, e)

    _supported_flags_cache[normalized_path] = supported_flags
    return supported_flags


def is_flag_supported(
    config_key: str,
    flag_name: str,
    llama_server_path: str,
    param_mapping: Dict[str, list],
) -> bool:
    """Check whether any known flag variant for a config key is supported by the binary."""
    if config_key in param_mapping:
        supported_flags = supported_flags_for_llama_binary(llama_server_path)
        for option in param_mapping[config_key]:
            if option in supported_flags:
                return True
        if supported_flags:
            return False
    return flag_name in supported_flags_for_llama_binary(llama_server_path)


def is_ik_llama_cpp(llama_server_path: Optional[str]) -> bool:
    """Detect ik_llama.cpp using active engine rows in engines.yaml, then flag heuristics."""
    if not llama_server_path:
        return False
    try:
        from backend.data_store import get_store

        store = get_store()
        norm = os.path.abspath(_abs_binary_path(llama_server_path))
        active_ik = store.get_active_engine_version("ik_llama")
        if active_ik and active_ik.get("binary_path"):
            if os.path.abspath(_abs_binary_path(active_ik["binary_path"])) == norm:
                return True
        active_llama = store.get_active_engine_version("llama_cpp")
        if active_llama and active_llama.get("binary_path"):
            if os.path.abspath(_abs_binary_path(active_llama["binary_path"])) == norm:
                return False
    except Exception as e:
        logger.debug("is_ik_llama_cpp store check: %s", e)

    try:
        supported_flags = get_supported_flags(llama_server_path)
        ik_specific_flags = {
            "--mla-use",
            "--smart-expert-reduction",
            "--attention-max-batch",
            "--no-fused-moe",
        }
        if supported_flags & ik_specific_flags:
            return True
    except Exception as e:
        logger.debug("is_ik_llama_cpp flag check: %s", e)
    return False


def get_param_mapping(is_ik: bool) -> Dict[str, list]:
    engine = "ik_llama" if is_ik else "llama_cpp"
    return resolve_llama_param_mapping_from_engine(engine)


def get_active_binary_path_from_db() -> Optional[str]:
    try:
        from backend.data_store import get_store

        return get_active_llama_swap_binary_path(get_store())
    except Exception as e:
        logger.error("Error getting binary path from data store: %s", e)
        return None


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value == "":
        return True
    if isinstance(value, float) and value != value:
        return True
    return False


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return None


def _quote_shell_token(token: str) -> str:
    return token if token == "${PORT}" else shlex.quote(token)


def _shell_join(tokens: List[str]) -> str:
    return " ".join(_quote_shell_token(str(token)) for token in tokens)


def _render_bash_command(
    argv: List[str],
    *,
    cwd: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
) -> str:
    parts: List[str] = []
    if cwd:
        parts.append(f"cd {_quote_shell_token(cwd)} &&")
    cmd_tokens = list(argv)
    if env:
        env_tokens = ["env"] + [f"{key}={value}" for key, value in env.items()]
        cmd_tokens = env_tokens + cmd_tokens
    parts.append(_shell_join(cmd_tokens))
    inner_cmd = " ".join(parts)
    return f"bash -c {shlex.quote(inner_cmd)}"


def _emit_param_tokens(key: str, value: Any, meta: dict) -> List[str]:
    primary_flag = meta.get("primary_flag")
    negative_flag = meta.get("negative_flag")
    value_kind = meta.get("value_kind") or "scalar"
    default = meta.get("default")

    if value_kind == "flag":
        explicit = _coerce_bool(value)
        if explicit is True and primary_flag:
            return [str(primary_flag)]
        if explicit is False and negative_flag:
            return [str(negative_flag)]
        return []

    if _is_empty_value(value):
        return []

    if value_kind == "repeatable":
        items = value if isinstance(value, list) else [value]
        out: List[str] = []
        for item in items:
            if _is_empty_value(item):
                continue
            out.extend([str(primary_flag), str(item)])
        return out

    if default is not None and value == default:
        return []

    if not primary_flag:
        raise ValueError(f"Parameter {key} is missing primary_flag metadata")
    return [str(primary_flag), str(value)]


def _split_custom_args(raw: str) -> List[str]:
    try:
        return shlex.split(raw)
    except ValueError as e:
        raise ValueError(f"Invalid custom_args shell syntax: {e}") from e


def _emit_structured_tokens(
    config: Dict[str, Any],
    *,
    engine: str,
    param_index: Dict[str, dict],
) -> List[str]:
    unknown_keys: List[str] = []
    tokens: List[str] = []

    for key, value in sorted((config or {}).items(), key=lambda item: item[0]):
        if key in _ALLOWED_NONCANONICAL_KEYS or _is_empty_value(value):
            continue
        meta = param_index.get(key)
        if not meta:
            unknown_keys.append(key)
            continue
        if meta.get("reserved"):
            continue
        tokens.extend(_emit_param_tokens(key, value, meta))

    if unknown_keys:
        joined = ", ".join(sorted(unknown_keys))
        raise ValueError(
            f"Unknown structured config keys for {engine}: {joined}. "
            "Rescan CLI parameters for the active engine and resave the config."
        )

    raw_custom_args = config.get("custom_args")
    if isinstance(raw_custom_args, str) and raw_custom_args.strip():
        tokens.extend(_split_custom_args(raw_custom_args.strip()))

    return tokens


def _resolve_cuda_library_path(build_dir: str) -> str:
    library_path = build_dir
    try:
        from backend.cuda_installer import get_cuda_installer

        cuda_installer = get_cuda_installer()
        cuda_path = cuda_installer._get_cuda_path()
        if cuda_path:
            cuda_lib = os.path.join(cuda_path, "lib64")
            if os.path.exists(cuda_lib):
                library_path = f"{cuda_lib}:{library_path}"
    except Exception as e:
        logger.debug("Could not get CUDA library path: %s", e)
    return library_path


def _resolve_lmdeploy_bin() -> Optional[str]:
    try:
        from backend.data_store import get_store

        store = get_store()
        active = store.get_active_engine_version("lmdeploy")
        venv = active.get("venv_path") if active else None
        if not venv:
            return None
        if not os.path.isabs(venv):
            venv = os.path.join("/app", venv)
        if not os.path.isdir(venv):
            return None
        candidate = os.path.join(venv, "bin", "lmdeploy")
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    except Exception as e:
        logger.debug("Could not resolve LMDeploy binary: %s", e)
    return None


def _model_attr(model: Any, key: str, default: Any = None) -> Any:
    if isinstance(model, dict):
        return model.get(key, default)
    return getattr(model, key, default)


def _resolve_llama_model_source(
    model: Any,
    *,
    fallback_model_path: Optional[str] = None,
) -> tuple[Optional[str], Optional[str], Optional[str]]:
    hf_id = _model_attr(model, "huggingface_id")
    quantization = _model_attr(model, "quantization")
    hf_repo_arg = None
    model_path = None

    if hf_id and quantization:
        resolved = resolve_gguf_model_path_for_quant(hf_id, str(quantization))
        if resolved and os.path.exists(resolved):
            model_path = resolved
        else:
            hf_repo_arg = f"{hf_id}:{str(quantization).lower()}"
    elif fallback_model_path:
        model_path = fallback_model_path

    if model_path and not os.path.isabs(model_path):
        model_path = f"/app/{model_path}"

    return model_path, hf_repo_arg, hf_id


def _resolve_mmproj_path(model: Any, hf_id: Optional[str], hf_repo_arg: Optional[str]) -> Optional[str]:
    if hf_repo_arg:
        return None
    mmproj_filename = _model_attr(model, "mmproj_filename")
    if not mmproj_filename or not hf_id:
        return None
    try:
        from backend.huggingface import resolve_cached_model_path

        mmproj_path = resolve_cached_model_path(hf_id, mmproj_filename)
        if mmproj_path and os.path.exists(mmproj_path):
            if not os.path.isabs(mmproj_path):
                return f"/app/{mmproj_path}"
            return mmproj_path
    except Exception as e:
        logger.debug("resolve mmproj failed for %s: %s", hf_id, e)
    return None


def _build_llama_command(
    *,
    model: Any,
    config: Dict[str, Any],
    proxy_model_name: str,
    llama_server_path: str,
    param_index: Dict[str, dict],
    fallback_model_path: Optional[str] = None,
) -> str:
    model_path, hf_repo_arg, hf_id = _resolve_llama_model_source(model, fallback_model_path=fallback_model_path)
    if not model_path and not hf_repo_arg:
        raise ValueError("Model path could not be resolved from HF metadata or runtime overlay")

    _, work_cwd = resolve_llama_server_invocation_paths(llama_server_path)
    binary_name = os.path.basename(llama_server_path)
    library_path = _resolve_cuda_library_path(work_cwd)

    argv: List[str] = [f"./{binary_name}"]
    if hf_repo_arg:
        argv.extend(["--hf-repo", hf_repo_arg])
    else:
        argv.extend(["--model", str(model_path)])

    argv.extend(["--port", "${PORT}", "--alias", proxy_model_name])
    mmproj_path = _resolve_mmproj_path(model, hf_id, hf_repo_arg)
    if mmproj_path:
        argv.extend(["--mmproj", mmproj_path])

    argv.extend(_emit_structured_tokens(config, engine=infer_engine_id_for_binary(llama_server_path), param_index=param_index))
    return _render_bash_command(
        argv,
        cwd=work_cwd,
        env={"LD_LIBRARY_PATH": library_path},
    )


def _build_lmdeploy_command(
    *,
    model: Any,
    config: Dict[str, Any],
    lmdeploy_bin: str,
    param_index: Dict[str, dict],
) -> str:
    hf_id = _model_attr(model, "huggingface_id")
    if not hf_id:
        raise ValueError("LMDeploy model must have huggingface_id")

    argv: List[str] = [lmdeploy_bin, "serve", "api_server", hf_id, "--server-port", "${PORT}"]
    argv.extend(_emit_structured_tokens(config, engine="lmdeploy", param_index=param_index))
    return _render_bash_command(argv)


def generate_llama_swap_config(
    models: Dict[str, Dict[str, Any]],
    llama_server_path: Optional[str] = None,
    all_models: list = None,
) -> str:
    """
    Generate the YAML configuration for llama-swap.
    """
    config_data = {
        "healthCheckTimeout": 600,
        "logTimeFormat": "2006-01-02 15:04:05",
        "sendLoadingState": True,
        "models": {},
    }

    from backend.data_store import (
        generate_proxy_name as _generate_proxy_name,
        normalize_proxy_alias as _normalize_proxy_alias,
        resolve_proxy_name as _resolve_proxy_name,
    )

    resolved_llama_server_path: Optional[str] = None
    resolved_llama_engine: Optional[str] = None
    llama_param_index: Optional[Dict[str, dict]] = None

    def _ensure_llama_runtime() -> tuple[str, str, Dict[str, dict]]:
        nonlocal resolved_llama_server_path, resolved_llama_engine, llama_param_index
        if resolved_llama_server_path is None:
            resolved_llama_server_path = llama_server_path or get_active_binary_path_from_db()
            if not resolved_llama_server_path:
                raise ValueError("No active llama-server binary configured")
            if not os.path.isabs(resolved_llama_server_path):
                resolved_llama_server_path = os.path.join("/app", resolved_llama_server_path)
            if not os.path.exists(resolved_llama_server_path):
                raise ValueError(f"llama-server binary not found at: {resolved_llama_server_path}")
            resolved_llama_engine = infer_engine_id_for_binary(resolved_llama_server_path)
            llama_param_index = _active_engine_param_index(resolved_llama_engine)
        return resolved_llama_server_path, resolved_llama_engine or "llama_cpp", llama_param_index or {}

    lmdeploy_bin = _resolve_lmdeploy_bin()
    lmdeploy_param_index = _active_engine_param_index("lmdeploy")

    all_models_by_proxy: Dict[str, Any] = {}

    if all_models:
        for model in all_models:
            proxy_model_name = _resolve_proxy_name(model)
            if not proxy_model_name:
                logger.warning(
                    "Model '%s' does not have a proxy name set, skipping",
                    _model_attr(model, "display_name") or _model_attr(model, "name"),
                )
                continue

            all_models_by_proxy[proxy_model_name] = model
            config = effective_model_config_from_raw(_model_attr(model, "config"))
            engine = config.get("engine")

            if engine == "lmdeploy":
                if not lmdeploy_bin:
                    logger.warning("LMDeploy binary unavailable; skipping %s", proxy_model_name)
                    continue
                try:
                    cmd = _build_lmdeploy_command(
                        model=model,
                        config=config,
                        lmdeploy_bin=lmdeploy_bin,
                        param_index=lmdeploy_param_index,
                    )
                    config_data["models"][proxy_model_name] = {
                        "cmd": cmd,
                        "useModelName": _model_attr(model, "huggingface_id"),
                    }
                except Exception as e:
                    logger.warning("Failed to build LMDeploy cmd for %s: %s", proxy_model_name, e)
                continue

            try:
                runtime_path, _, param_index = _ensure_llama_runtime()
                cmd = _build_llama_command(
                    model=model,
                    config=config,
                    proxy_model_name=proxy_model_name,
                    llama_server_path=runtime_path,
                    param_index=param_index,
                )
                config_data["models"][proxy_model_name] = {"cmd": cmd}
            except Exception as e:
                logger.warning("Failed to build llama cmd for %s: %s", proxy_model_name, e)

    def _overlay_model_for_running_key(proxy_key: str) -> Any:
        model = all_models_by_proxy.get(proxy_key)
        if model is not None:
            return model
        if not all_models:
            return None
        for candidate in all_models:
            if _resolve_proxy_name(candidate) == proxy_key:
                return candidate
            proxy_name = _normalize_proxy_alias(_model_attr(candidate, "proxy_name"))
            if proxy_name and proxy_name == proxy_key:
                return candidate
            generated = _generate_proxy_name(
                _model_attr(candidate, "huggingface_id", ""),
                _model_attr(candidate, "quantization"),
            )
            if generated and generated == proxy_key:
                return candidate
        return None

    for proxy_model_name, model_data in models.items():
        overlay_model = _overlay_model_for_running_key(proxy_model_name)
        overlay_config = effective_model_config_from_raw(model_data.get("config"))
        alias = _normalize_proxy_alias(overlay_config.get("model_alias"))
        resolved_proxy_model_name = _resolve_proxy_name(overlay_model) if overlay_model else alias or proxy_model_name

        if overlay_config.get("engine") == "lmdeploy" and overlay_model and lmdeploy_bin:
            try:
                cmd = _build_lmdeploy_command(
                    model=overlay_model,
                    config=overlay_config,
                    lmdeploy_bin=lmdeploy_bin,
                    param_index=lmdeploy_param_index,
                )
                config_data["models"].pop(proxy_model_name, None)
                config_data["models"][resolved_proxy_model_name] = {
                    "cmd": cmd,
                    "useModelName": _model_attr(overlay_model, "huggingface_id"),
                }
            except Exception as e:
                logger.warning("Failed to build LMDeploy overlay cmd for %s: %s", resolved_proxy_model_name, e)
            continue

        fallback_model_path = model_data.get("model_path")
        model_for_command = overlay_model or model_data
        try:
            runtime_path, _, param_index = _ensure_llama_runtime()
            cmd = _build_llama_command(
                model=model_for_command,
                config=overlay_config,
                proxy_model_name=resolved_proxy_model_name,
                llama_server_path=runtime_path,
                param_index=param_index,
                fallback_model_path=fallback_model_path,
            )
            config_data["models"].pop(proxy_model_name, None)
            config_data["models"][resolved_proxy_model_name] = {"cmd": cmd}
        except Exception as e:
            logger.warning("Failed to build llama overlay cmd for %s: %s", resolved_proxy_model_name, e)

    if config_data["models"]:
        config_data["groups"] = {
            "concurrent_models": {
                "swap": False,
                "exclusive": False,
                "members": sorted(config_data["models"].keys()),
            }
        }

    return yaml.dump(config_data, sort_keys=False, indent=2)


def preview_llama_swap_command_for_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the llama-swap ``cmd`` string for one model and surface metadata
    validation errors directly instead of silently skipping the model.
    """
    from backend.data_store import resolve_proxy_name

    proxy = resolve_proxy_name(model)
    if not proxy:
        return {
            "ok": False,
            "error": "Model has no proxy name",
            "cmd": None,
            "proxy_name": None,
        }

    config = effective_model_config_from_raw(model.get("config"))
    engine = config.get("engine")

    try:
        if engine == "lmdeploy":
            lmdeploy_bin = _resolve_lmdeploy_bin()
            if not lmdeploy_bin:
                raise ValueError("LMDeploy binary unavailable")
            cmd = _build_lmdeploy_command(
                model=model,
                config=config,
                lmdeploy_bin=lmdeploy_bin,
                param_index=_active_engine_param_index("lmdeploy"),
            )
            return {
                "ok": True,
                "cmd": cmd,
                "proxy_name": proxy,
                "use_model_name": _model_attr(model, "huggingface_id"),
            }

        llama_server_path = get_active_binary_path_from_db()
        if not llama_server_path:
            raise ValueError("No active llama-server binary configured")
        if not os.path.isabs(llama_server_path):
            llama_server_path = os.path.join("/app", llama_server_path)
        if not os.path.exists(llama_server_path):
            raise ValueError(f"llama-server binary not found at: {llama_server_path}")

        resolved_engine = infer_engine_id_for_binary(llama_server_path)
        cmd = _build_llama_command(
            model=model,
            config=config,
            proxy_model_name=proxy,
            llama_server_path=llama_server_path,
            param_index=_active_engine_param_index(resolved_engine),
        )
        return {
            "ok": True,
            "cmd": cmd,
            "proxy_name": proxy,
            "use_model_name": None,
        }
    except Exception as e:
        logger.warning("preview_llama_swap_command_for_model failed: %s", e)
        return {"ok": False, "error": str(e), "cmd": None, "proxy_name": proxy}
