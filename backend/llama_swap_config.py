import os
import re
import shlex
import subprocess
from typing import Any, Dict, List, Optional, Set

import yaml

from backend import data_store
from backend import engine_param_catalog
from backend.huggingface import resolve_gguf_model_path_for_quant
from backend.llama_engine_resolve import (
    abs_llama_binary_path as _abs_binary_path,
    get_active_binary_path_for_engine,
    infer_llama_engine_for_binary,
)
from backend.llama_server_exec import (
    llama_help_ld_library_path,
    resolve_llama_server_invocation_paths,
)
from backend.logging_config import get_logger
from backend.model_config import effective_model_config_from_raw, merge_model_config_put

logger = get_logger(__name__)

_supported_flags_cache: Dict[str, Set[str]] = {}

_ALLOWED_NONCANONICAL_KEYS = frozenset(
    {"custom_args", "engine", "engines", "model_alias", "swap_env"}
)

_SWAP_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_LLAMA_SWAP_MACRO_TOKEN_RE = re.compile(r"^\$\{[A-Za-z0-9_-]+\}$")
_UNQUOTED_CMD_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:=+,@%${}-]+$")

# User ``swap_env`` keys prefixed with ``LLAMA_STUDIO_`` are reserved (ignored).
_STUDIO_ENV_PREFIX = "LLAMA_STUDIO_"

_MODEL_MACRO_MODEL_PATH = "studio_model_path"
_MODEL_MACRO_HF_REPO = "studio_hf_repo"
_MODEL_MACRO_MMPROJ_PATH = "studio_mmproj_path"


def clear_supported_flags_cache() -> None:
    _supported_flags_cache.clear()


def infer_engine_id_for_binary(binary_path: str) -> str:
    """Resolve llama_cpp vs ik_llama from engines.yaml active rows."""
    try:
        return infer_llama_engine_for_binary(data_store.get_store(), binary_path)
    except Exception as e:
        logger.debug("infer_engine_id_for_binary: %s", e)
    return "llama_cpp"


def _active_engine_entry(engine: str) -> Optional[dict]:
    store = data_store.get_store()
    active = store.get_active_engine_version(engine)
    if not active or not active.get("version"):
        return None
    return engine_param_catalog.get_version_entry(store, engine, active["version"])


def _active_engine_param_index(engine: str) -> Dict[str, dict]:
    return engine_param_catalog.param_index_from_entry(_active_engine_entry(engine))


def resolve_llama_param_mapping_from_engine(engine: str) -> Dict[str, list]:
    return engine_param_catalog.param_mapping_from_entry(_active_engine_entry(engine))


def supported_flags_for_llama_binary(binary_path: str) -> Set[str]:
    """Prefer catalog flags for the active engine version; else parse --help."""
    eng = infer_engine_id_for_binary(binary_path)
    try:
        entry = _active_engine_entry(eng)
        flags = engine_param_catalog.flags_from_entry(entry)
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
    normalized_path = (
        os.path.abspath(exec_path) if os.path.exists(exec_path) else exec_path
    )
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
            logger.warning(
                "No stdout from %s --help (exit %s)", exec_path, result.returncode
            )
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
        store = data_store.get_store()
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


def any_active_gguf_runtime_in_db() -> bool:
    """True if at least one active llama_cpp or ik_llama binary exists on disk."""
    try:
        store = data_store.get_store()
        return bool(
            get_active_binary_path_for_engine(store, "llama_cpp")
            or get_active_binary_path_for_engine(store, "ik_llama")
        )
    except Exception as e:
        logger.debug("any_active_gguf_runtime_in_db: %s", e)
        return False


def _gguf_engine_id_for_config(config: Dict[str, Any]) -> str:
    """Model config engine for GGUF llama-server commands (lmdeploy excluded by caller)."""
    return "ik_llama" if (config or {}).get("engine") == "ik_llama" else "llama_cpp"


def _effective_config_for_running_overlay(
    overlay_model: Any, model_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge catalog model config with the in-memory / llama-swap overlay row.

    After ``sync_running_models``, overlay rows often have ``config: {}``; we must still use
    the engine (and params) from the database model so YAML targets the right binary.
    """
    raw_runtime = (model_data or {}).get("config")
    if not isinstance(raw_runtime, dict):
        raw_runtime = {}
    if overlay_model is not None:
        return effective_model_config_from_raw(
            merge_model_config_put(_model_attr(overlay_model, "config"), raw_runtime)
        )
    return effective_model_config_from_raw(raw_runtime if raw_runtime else None)


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
    if _LLAMA_SWAP_MACRO_TOKEN_RE.match(token) or _UNQUOTED_CMD_TOKEN_RE.match(token):
        return token
    return shlex.quote(token)


def _shell_join(tokens: List[str]) -> str:
    return " ".join(_quote_shell_token(str(token)) for token in tokens)


def _macro_ref(name: str) -> str:
    return f"${{{name}}}"


def _gguf_macro_names(gguf_engine: str) -> tuple[str, str]:
    slug = gguf_engine.replace("-", "_")
    return f"studio_gguf_bin_{slug}", f"studio_gguf_ld_{slug}"


def _register_gguf_engine_macros(
    registry: Dict[str, Dict[str, Any]],
    *,
    engine: str,
    abs_bin: str,
    ld: str,
) -> str:
    """
    Register shared GGUF engine binary macro; validate ``ld`` (library path base) is consistent
    per engine. ``ld`` is emitted as literal ``LD_LIBRARY_PATH`` in ``env``, not as a macro.
    """
    bin_macro, ld_macro = _gguf_macro_names(engine)
    norm_bin = os.path.abspath(abs_bin) if os.path.exists(abs_bin) else abs_bin
    ld = (ld or "").strip()
    if engine in registry:
        ex = registry[engine]
        if ex["abs_bin"] != norm_bin or ex["ld"] != ld:
            raise ValueError(
                f"llama-swap config: inconsistent binary or library path for engine {engine!r}"
            )
        return ex["bin_macro"]
    registry[engine] = {
        "bin_macro": bin_macro,
        "ld_macro": ld_macro,
        "abs_bin": norm_bin,
        "ld": ld,
    }
    return bin_macro


def _macros_dict_for_yaml(registry: Dict[str, Dict[str, Any]]) -> Dict[str, str]:
    """Global GGUF macros: engine binary path only (``LD_LIBRARY_PATH`` is literal ``env``)."""
    out: Dict[str, str] = {}
    for eng in sorted(registry.keys()):
        inf = registry[eng]
        out[inf["bin_macro"]] = inf["abs_bin"]
    return out


def _llama_swap_model_macros(
    *,
    model_path: Optional[str],
    hf_repo_arg: Optional[str],
    mmproj_path: Optional[str],
) -> Dict[str, str]:
    """Per-model llama-swap macros used by GGUF ``cmd`` values."""
    out: Dict[str, str] = {}
    if hf_repo_arg:
        out[_MODEL_MACRO_HF_REPO] = str(hf_repo_arg)
    elif model_path:
        out[_MODEL_MACRO_MODEL_PATH] = str(model_path)
    if mmproj_path:
        out[_MODEL_MACRO_MMPROJ_PATH] = str(mmproj_path)
    return out


def _merge_macro_maps(*maps: Optional[Dict[str, str]]) -> Optional[Dict[str, str]]:
    out: Dict[str, str] = {}
    for mapping in maps:
        if mapping:
            out.update(mapping)
    return out or None


def _normalize_swap_env(config: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Parse per-model ``swap_env`` (llama-swap YAML ``env``) from effective config."""
    raw = (config or {}).get("swap_env")
    if not isinstance(raw, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        name = key.strip()
        if not name or not _SWAP_ENV_KEY_RE.match(name):
            logger.debug("swap_env: skipping invalid variable name %r", key)
            continue
        if _is_empty_value(value):
            continue
        out[name] = str(value).strip() if isinstance(value, str) else str(value)
    return out


def _gguf_user_env_and_ld_suffix(user_env: Dict[str, str]) -> tuple[Dict[str, str], Optional[str]]:
    """
    Drop reserved ``LLAMA_STUDIO_*`` keys; split out ``LD_LIBRARY_PATH`` (merged into GGUF ``env``).
    """
    merged = {
        k: v
        for k, v in user_env.items()
        if not str(k).startswith(_STUDIO_ENV_PREFIX)
    }
    user_ld = merged.pop("LD_LIBRARY_PATH", None)
    if user_ld and str(user_ld).strip():
        return merged, str(user_ld).strip()
    return merged, None


def _gguf_swap_env_lines(
    user_merged: Dict[str, str],
    *,
    resolved_ld_library_path: str,
    user_ld_suffix: Optional[str],
) -> List[str]:
    """
    GGUF ``env``: user ``swap_env`` (minus reserved keys and split ``LD_LIBRARY_PATH``) plus
    merged ``LD_LIBRARY_PATH`` as a **literal** (llama-swap does not expand ``${…}`` in ``env``).
    Model / mmproj / hf-repo paths live only in per-model ``macros``, not here.
    """
    merged = dict(user_merged)
    ld = (resolved_ld_library_path or "").strip()
    if user_ld_suffix:
        ld = f"{ld}:{user_ld_suffix}" if ld else user_ld_suffix
    if ld:
        merged["LD_LIBRARY_PATH"] = ld
    return [f"{k}={merged[k]}" for k in sorted(merged.keys())]


def _lmdeploy_swap_env_list(user_env: Dict[str, str]) -> List[str]:
    if not user_env:
        return []
    return [f"{k}={user_env[k]}" for k in sorted(user_env.keys())]


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
        store = data_store.get_store()
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


def _resolve_mmproj_path(
    model: Any, hf_id: Optional[str], hf_repo_arg: Optional[str]
) -> Optional[str]:
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


def _build_llama_swap_gguf_cmd(
    *,
    bin_macro: str,
    proxy_model_name: str,
    model_macros: Dict[str, str],
    structured_argv: List[str],
) -> str:
    """
    Single-line ``cmd`` for llama-swap: no ``bash -c``, no ``env`` prefix.

    Shared engine binary uses a top-level macro (e.g. ``${studio_gguf_bin_llama_cpp}``).
    Per-model model / hf-repo / mmproj paths use per-model ``macros`` (``${studio_model_path}``, …).
    ``LD_LIBRARY_PATH`` is **not** in ``cmd``; it is a literal entry in YAML ``env`` (llama-swap
    does not expand ``${…}`` inside ``env`` values).
    """
    parts: List[str] = []
    parts.append(_macro_ref(bin_macro))
    if _MODEL_MACRO_HF_REPO in model_macros:
        parts.extend(["--hf-repo", _macro_ref(_MODEL_MACRO_HF_REPO)])
    else:
        parts.extend(["--model", _macro_ref(_MODEL_MACRO_MODEL_PATH)])
    parts.extend(["--port", "${PORT}", "--alias", proxy_model_name])
    if _MODEL_MACRO_MMPROJ_PATH in model_macros:
        parts.extend(["--mmproj", _macro_ref(_MODEL_MACRO_MMPROJ_PATH)])
    if structured_argv:
        parts.extend(structured_argv)
    return _shell_join(parts)


def _build_llama_command(
    *,
    model: Any,
    config: Dict[str, Any],
    proxy_model_name: str,
    llama_server_path: str,
    param_index: Dict[str, dict],
    fallback_model_path: Optional[str] = None,
    engine_for_params: Optional[str] = None,
    gguf_macro_registry: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple[str, List[str], Dict[str, str]]:
    model_path, hf_repo_arg, hf_id = _resolve_llama_model_source(
        model, fallback_model_path=fallback_model_path
    )
    if not model_path and not hf_repo_arg:
        raise ValueError(
            "Model path could not be resolved from HF metadata or runtime overlay"
        )

    exec_path, work_cwd = resolve_llama_server_invocation_paths(llama_server_path)
    library_path = _resolve_cuda_library_path(work_cwd)
    mmproj_path = _resolve_mmproj_path(model, hf_id, hf_repo_arg)

    structured_engine = (
        engine_for_params
        if engine_for_params in ("llama_cpp", "ik_llama")
        else infer_engine_id_for_binary(llama_server_path)
    )
    structured_argv = _emit_structured_tokens(
        config, engine=structured_engine, param_index=param_index
    )

    model_macros = _llama_swap_model_macros(
        model_path=model_path,
        hf_repo_arg=hf_repo_arg,
        mmproj_path=mmproj_path,
    )

    user_merged, user_ld = _gguf_user_env_and_ld_suffix(_normalize_swap_env(config))
    if gguf_macro_registry is not None:
        bin_macro = _register_gguf_engine_macros(
            gguf_macro_registry,
            engine=structured_engine,
            abs_bin=exec_path,
            ld=library_path,
        )
    else:
        bin_macro = _gguf_macro_names(structured_engine)[0]

    cmd = _build_llama_swap_gguf_cmd(
        bin_macro=bin_macro,
        proxy_model_name=proxy_model_name,
        model_macros=model_macros,
        structured_argv=structured_argv,
    )
    env_list = _gguf_swap_env_lines(
        user_merged,
        resolved_ld_library_path=library_path,
        user_ld_suffix=user_ld,
    )
    return cmd, env_list, model_macros


def _build_lmdeploy_command(
    *,
    model: Any,
    config: Dict[str, Any],
    lmdeploy_bin: str,
    param_index: Dict[str, dict],
) -> tuple[str, List[str]]:
    hf_id = _model_attr(model, "huggingface_id")
    if not hf_id:
        raise ValueError("LMDeploy model must have huggingface_id")

    argv: List[str] = [
        lmdeploy_bin,
        "serve",
        "api_server",
        hf_id,
        "--server-port",
        "${PORT}",
    ]
    argv.extend(
        _emit_structured_tokens(config, engine="lmdeploy", param_index=param_index)
    )
    env_list = _lmdeploy_swap_env_list(_normalize_swap_env(config))
    return _render_bash_command(argv), env_list


def _llama_swap_yaml_model_block(
    *,
    cmd: str,
    env_list: List[str],
    use_model_name: Optional[str] = None,
    model_macros: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    block: Dict[str, Any] = {"cmd": cmd}
    if env_list:
        block["env"] = env_list
    if model_macros:
        block["macros"] = model_macros
    if use_model_name:
        block["useModelName"] = use_model_name
    return block


def generate_llama_swap_config(
    running_models: Dict[str, Dict[str, Any]],
    all_models: Optional[List[Any]] = None,
) -> str:
    """
    Generate llama-swap YAML. GGUF entries use the active binary for each model's engine
    (``llama_cpp`` or ``ik_llama``) from the engines store.
    """
    config_data = {
        "healthCheckTimeout": 600,
        "logTimeFormat": "2006-01-02 15:04:05",
        "sendLoadingState": True,
        "models": {},
    }

    gguf_macro_registry: Dict[str, Dict[str, Any]] = {}

    _llama_runtime_cache: Dict[str, tuple[str, Dict[str, dict]]] = {}

    def _ensure_llama_runtime_for_engine(
        gguf_engine: str,
    ) -> tuple[str, Dict[str, dict]]:
        if gguf_engine not in _llama_runtime_cache:
            path = get_active_binary_path_for_engine(
                data_store.get_store(), gguf_engine
            )
            if not path:
                raise ValueError(
                    f"No active {gguf_engine} llama-server binary configured"
                )
            if not os.path.isabs(path):
                path = os.path.join("/app", path)
            if not os.path.exists(path):
                raise ValueError(f"llama-server binary not found at: {path}")
            _llama_runtime_cache[gguf_engine] = (
                path,
                _active_engine_param_index(gguf_engine),
            )
        return _llama_runtime_cache[gguf_engine]

    lmdeploy_bin = _resolve_lmdeploy_bin()
    lmdeploy_param_index = _active_engine_param_index("lmdeploy")

    all_models_by_proxy: Dict[str, Any] = {}

    if all_models:
        for model in all_models:
            proxy_model_name = data_store.resolve_proxy_name(model)
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
                    logger.warning(
                        "LMDeploy binary unavailable; skipping %s", proxy_model_name
                    )
                    continue
                try:
                    cmd, env_list = _build_lmdeploy_command(
                        model=model,
                        config=config,
                        lmdeploy_bin=lmdeploy_bin,
                        param_index=lmdeploy_param_index,
                    )
                    config_data["models"][proxy_model_name] = _llama_swap_yaml_model_block(
                        cmd=cmd,
                        env_list=env_list,
                        use_model_name=_model_attr(model, "huggingface_id"),
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to build LMDeploy cmd for %s: %s", proxy_model_name, e
                    )
                continue

            gguf_engine = _gguf_engine_id_for_config(config)
            try:
                runtime_path, param_index = _ensure_llama_runtime_for_engine(
                    gguf_engine
                )
                cmd, env_list, model_macros = _build_llama_command(
                    model=model,
                    config=config,
                    proxy_model_name=proxy_model_name,
                    llama_server_path=runtime_path,
                    param_index=param_index,
                    engine_for_params=gguf_engine,
                    gguf_macro_registry=gguf_macro_registry,
                )
                config_data["models"][proxy_model_name] = _llama_swap_yaml_model_block(
                    cmd=cmd, env_list=env_list, model_macros=model_macros
                )
            except Exception as e:
                logger.warning(
                    "Failed to build llama cmd for %s: %s", proxy_model_name, e
                )

    def _overlay_model_for_running_key(proxy_key: str) -> Any:
        model = all_models_by_proxy.get(proxy_key)
        if model is not None:
            return model
        if not all_models:
            return None
        for candidate in all_models:
            if data_store.resolve_proxy_name(candidate) == proxy_key:
                return candidate
            proxy_name = data_store.normalize_proxy_alias(
                _model_attr(candidate, "proxy_name")
            )
            if proxy_name and proxy_name == proxy_key:
                return candidate
            generated = data_store.generate_proxy_name(
                _model_attr(candidate, "huggingface_id", ""),
                _model_attr(candidate, "quantization"),
            )
            if generated and generated == proxy_key:
                return candidate
        return None

    for proxy_model_name, model_data in running_models.items():
        overlay_model = _overlay_model_for_running_key(proxy_model_name)
        overlay_config = _effective_config_for_running_overlay(
            overlay_model, model_data
        )
        alias = data_store.normalize_proxy_alias(overlay_config.get("model_alias"))
        resolved_proxy_model_name = (
            data_store.resolve_proxy_name(overlay_model)
            if overlay_model
            else alias or proxy_model_name
        )

        if (
            overlay_config.get("engine") == "lmdeploy"
            and overlay_model
            and lmdeploy_bin
        ):
            try:
                cmd, env_list = _build_lmdeploy_command(
                    model=overlay_model,
                    config=overlay_config,
                    lmdeploy_bin=lmdeploy_bin,
                    param_index=lmdeploy_param_index,
                )
                config_data["models"].pop(proxy_model_name, None)
                config_data["models"][resolved_proxy_model_name] = (
                    _llama_swap_yaml_model_block(
                        cmd=cmd,
                        env_list=env_list,
                        use_model_name=_model_attr(overlay_model, "huggingface_id"),
                    )
                )
            except Exception as e:
                logger.warning(
                    "Failed to build LMDeploy overlay cmd for %s: %s",
                    resolved_proxy_model_name,
                    e,
                )
            continue

        fallback_model_path = model_data.get("model_path")
        model_for_command = overlay_model or model_data
        gguf_engine = _gguf_engine_id_for_config(overlay_config)
        try:
            runtime_path, param_index = _ensure_llama_runtime_for_engine(gguf_engine)
            cmd, env_list, model_macros = _build_llama_command(
                model=model_for_command,
                config=overlay_config,
                proxy_model_name=resolved_proxy_model_name,
                llama_server_path=runtime_path,
                param_index=param_index,
                fallback_model_path=fallback_model_path,
                engine_for_params=gguf_engine,
                gguf_macro_registry=gguf_macro_registry,
            )
            config_data["models"].pop(proxy_model_name, None)
            config_data["models"][resolved_proxy_model_name] = _llama_swap_yaml_model_block(
                cmd=cmd, env_list=env_list, model_macros=model_macros
            )
        except Exception as e:
            logger.warning(
                "Failed to build llama overlay cmd for %s: %s",
                resolved_proxy_model_name,
                e,
            )

    if config_data["models"]:
        config_data["groups"] = {
            "concurrent_models": {
                "swap": False,
                "exclusive": False,
                "members": sorted(config_data["models"].keys()),
            }
        }

    if gguf_macro_registry:
        reordered: Dict[str, Any] = {}
        for key, val in config_data.items():
            if key == "models":
                reordered["macros"] = _macros_dict_for_yaml(gguf_macro_registry)
            reordered[key] = val
        config_data = reordered

    # Wide ``width`` keeps long ``cmd`` lines on one physical line (valid YAML; avoids
    # awkward wraps that look like broken quoting).
    return yaml.dump(config_data, sort_keys=False, indent=2, width=4096)


def preview_llama_swap_command_for_model(model: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the llama-swap ``cmd`` string for one model and surface metadata
    validation errors directly instead of silently skipping the model.
    """
    proxy = data_store.resolve_proxy_name(model)
    if not proxy:
        return {
            "ok": False,
            "error": "Model has no proxy name",
            "cmd": None,
            "env": None,
            "macros": None,
            "proxy_name": None,
        }

    config = effective_model_config_from_raw(model.get("config"))
    engine = config.get("engine")

    try:
        if engine == "lmdeploy":
            lmdeploy_bin = _resolve_lmdeploy_bin()
            if not lmdeploy_bin:
                raise ValueError("LMDeploy binary unavailable")
            cmd, env_list = _build_lmdeploy_command(
                model=model,
                config=config,
                lmdeploy_bin=lmdeploy_bin,
                param_index=_active_engine_param_index("lmdeploy"),
            )
            return {
                "ok": True,
                "cmd": cmd,
                "env": env_list if env_list else None,
                "macros": None,
                "proxy_name": proxy,
                "use_model_name": _model_attr(model, "huggingface_id"),
            }

        gguf_engine = _gguf_engine_id_for_config(config)
        llama_server_path = get_active_binary_path_for_engine(
            data_store.get_store(), gguf_engine
        )
        if not llama_server_path:
            raise ValueError(f"No active {gguf_engine} llama-server binary configured")
        if not os.path.isabs(llama_server_path):
            llama_server_path = os.path.join("/app", llama_server_path)
        if not os.path.exists(llama_server_path):
            raise ValueError(f"llama-server binary not found at: {llama_server_path}")

        gguf_macro_registry: Dict[str, Dict[str, Any]] = {}
        cmd, env_list, model_macros = _build_llama_command(
            model=model,
            config=config,
            proxy_model_name=proxy,
            llama_server_path=llama_server_path,
            param_index=_active_engine_param_index(gguf_engine),
            engine_for_params=gguf_engine,
            gguf_macro_registry=gguf_macro_registry,
        )
        return {
            "ok": True,
            "cmd": cmd,
            "env": env_list if env_list else None,
            "macros": _merge_macro_maps(
                _macros_dict_for_yaml(gguf_macro_registry), model_macros
            ),
            "proxy_name": proxy,
            "use_model_name": None,
        }
    except Exception as e:
        logger.warning("preview_llama_swap_command_for_model failed: %s", e)
        return {
            "ok": False,
            "error": str(e),
            "cmd": None,
            "env": None,
            "macros": None,
            "proxy_name": proxy,
        }
