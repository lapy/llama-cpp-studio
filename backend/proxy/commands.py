import asyncio


import json


import os


import re


import shlex


import subprocess


from typing import Any, Dict, List, Optional, Set


from backend import data_store


from backend.engines import params as engine_param_catalog


from backend.engines.registry import ENGINE_REGISTRY, GGUF_ENGINE_IDS, get_engine_spec


from backend.models.hub import resolve_gguf_model_path


from backend.engines.llama_cpp.resolve import (
    abs_llama_binary_path as _abs_binary_path,
    get_active_binary_path_for_engine,
    infer_llama_engine_for_binary,
)


from backend.engines.llama_cpp.server_exec import (
    llama_help_ld_library_path,
    resolve_llama_server_invocation_paths,
)


from backend.logging_config import get_logger


from backend.models.config import effective_model_config_from_raw, merge_model_config_put


from backend.proxy.client import MODEL_START_BUDGET_SECONDS


logger = get_logger(__name__)


def resolve_gguf_model_path_for_quant(_huggingface_id: str, _quantization: str) -> Optional[str]:
    """Compatibility seam for old tests/plugins; manifests are no longer consulted."""
    return None


_supported_flags_cache: Dict[str, Set[str]] = {}


_cuda_library_path_cache: Dict[str, str] = {}


_ALLOWED_NONCANONICAL_KEYS = frozenset(
    {
        "custom_args",
        "engine",
        "engines",
        "model_alias",
        "set_params_by_id",
        "swap_aliases",
        "runtime_env",
        "runtime_env_unset",
        "gpu_mode",
        "gpu_devices",
        "sglang_runtime",
    }
)


_RUNTIME_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_RUNTIME_TOKEN_RE = re.compile(r"^\$\{[A-Za-z0-9_-]+\}$")


CAPCOMPAT_ENGINE_IDS = frozenset(
    engine_id
    for engine_id, spec in ENGINE_REGISTRY.items()
    if dict(spec.launch_policy).get("capcompat") is True
)


_MODEL_MODALITIES = ("text", "audio", "image", "video")


def _runtime_context_key(engine: Optional[str]) -> tuple:
    """The model-config key carrying this engine's runtime context (R5).

    Read from the engine's declared ``launch_policy``; no engine-id set here.
    """
    if not engine:
        return ()
    from backend.engines.registry import get_engine_spec

    spec = get_engine_spec(str(engine))
    if spec is None:
        return ()
    key = str(dict(spec.launch_policy).get("runtime_context_key") or "")
    return (key,) if key else ()


_PIPELINE_MODEL_MODALITIES = {
    "image-text-to-text": (("text", "image"), ("text",)),
    "visual-question-answering": (("text", "image"), ("text",)),
    "image-to-text": (("image",), ("text",)),
    "document-question-answering": (("text", "image"), ("text",)),
    "video-text-to-text": (("text", "video"), ("text",)),
    "text-to-image": (("text",), ("image",)),
    "image-to-image": (("image",), ("image",)),
    "text-to-video": (("text",), ("video",)),
}


_MMPROJ_ARG_RE = re.compile(r"(?:^|\s)--mmproj(?:\s|=|$)")


_UNQUOTED_CMD_TOKEN_RE = re.compile(r"^[A-Za-z0-9_./:=+,@%${}-]+$")


_STUDIO_ENV_PREFIX = "LLAMA_STUDIO_"


_MODEL_MACRO_MODEL_PATH = "studio_model_path"


_MODEL_MACRO_HF_REPO = "studio_hf_repo"


_MODEL_MACRO_MMPROJ_PATH = "studio_mmproj_path"


_MODEL_MACRO_DRAFT_PATH = "studio_draft_path"


def model_capabilities(
    engine: Optional[str],
    model: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Return model capability metadata, or None when it is unknown.

    With a model or config, Studio writes the modalities, reranker flag, and
    context it already knows. Probe-supported engines leave every unknown field
    unset so capcompat can fill it (tools, and context when the launch config
    does not set a window). Engines Studio proxy cannot read keep ``disableAuto``
    and fall back to the stored context length.
    """
    if model is None and config is None:
        if engine in CAPCOMPAT_ENGINE_IDS:
            return None
        return {"disableAuto": True}

    inputs, outputs, reranker = _capability_modalities(engine, model, config)
    block: Dict[str, Any] = {}
    if inputs:
        block["in"] = inputs
    if outputs:
        block["out"] = outputs
    if reranker:
        block["reranker"] = True
    context = _capability_context(engine, model, config)
    if context:
        block["context"] = context
    if engine not in CAPCOMPAT_ENGINE_IDS:
        block["disableAuto"] = True
    return block or None


def _capability_modalities(
    engine: Optional[str],
    model: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> tuple[List[str], List[str], bool]:
    raw_inputs, raw_outputs = _record_modalities(model, config)
    if not raw_inputs and not raw_outputs:
        pipeline = str(_model_attr(model, "pipeline_tag") or "").strip().lower()
        mapped = _PIPELINE_MODEL_MODALITIES.get(pipeline)
        if mapped:
            raw_inputs, raw_outputs = list(mapped[0]), list(mapped[1])
    reranker = _reranker_model(model, config, raw_outputs)
    if (
        not raw_inputs
        and not raw_outputs
        and _text_generation_engine(engine)
        and not _embedding_model(model, config)
        and not reranker
    ):
        raw_inputs, raw_outputs = ["text"], ["text"]
    if _has_vision_projector(model, config):
        if "text" not in raw_inputs:
            raw_inputs = ["text", *raw_inputs]
        if "image" not in raw_inputs:
            raw_inputs.append("image")
    if _embedding_model(model, config):
        raw_outputs = [item for item in raw_outputs if item != "text"]
        if "text" not in raw_inputs:
            raw_inputs = ["text", *raw_inputs]
    if reranker and not raw_inputs:
        raw_inputs = ["text"]
    return (
        [item for item in _MODEL_MODALITIES if item in raw_inputs],
        [item for item in _MODEL_MODALITIES if item in raw_outputs],
        reranker,
    )


def _record_modalities(
    model: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> tuple[List[str], List[str]]:
    from backend.models.schema import normalize_model_record

    record = dict(model or {})
    task = (config or {}).get("task")
    if task:
        record["task"] = task
        record["tasks"] = [task]
        record.pop("input_modalities", None)
        record.pop("output_modalities", None)
    normalized = normalize_model_record(record)
    return (
        list(normalized.get("input_modalities") or []),
        list(normalized.get("output_modalities") or []),
    )


def _text_generation_engine(engine: Optional[str]) -> bool:
    spec = get_engine_spec(engine)
    return bool(spec and "text-generation" in spec.tasks)


def _embedding_model(model: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]]) -> bool:
    if (config or {}).get("embedding"):
        return True
    if not model:
        return False
    from backend.services.model_metadata import model_is_embedding

    return model_is_embedding(model)


def _reranker_model(
    model: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
    outputs: List[str],
) -> bool:
    from backend.models.schema import canonical_task

    tasks = list(_model_attr(model, "tasks") or [])
    for value in (
        _model_attr(model, "task"),
        _model_attr(model, "pipeline_tag"),
        (config or {}).get("task"),
    ):
        if value:
            tasks.append(value)
    normalized = {canonical_task(item) for item in tasks}
    if "reranking" in normalized or "text-ranking" in normalized:
        return True
    return "score" in outputs


def _has_vision_projector(
    model: Optional[Dict[str, Any]], config: Optional[Dict[str, Any]]
) -> bool:
    if _model_attr(model, "mmproj_filename"):
        return True
    files = _model_attr(model, "files") or []
    if isinstance(files, list):
        for entry in files:
            if isinstance(entry, dict) and entry.get("role") == "mmproj":
                return True
            if (
                isinstance(entry, str)
                and "mmproj" in entry.lower()
                and entry.lower().endswith(".gguf")
            ):
                return True
    custom = str((config or {}).get("custom_args") or "")
    return bool(_MMPROJ_ARG_RE.search(custom))


def _capability_context(
    engine: Optional[str],
    model: Optional[Dict[str, Any]],
    config: Optional[Dict[str, Any]],
) -> Optional[int]:
    from backend.utils.coercion import coerce_positive_int

    for key in _runtime_context_key(engine):
        context = coerce_positive_int((config or {}).get(key))
        if context:
            return context
    if engine in CAPCOMPAT_ENGINE_IDS:
        return None
    for key in ("max_context_length", "context_length"):
        context = coerce_positive_int(_model_attr(model, key))
        if context:
            return context
    return None


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


def any_active_runtime_in_db() -> bool:
    """True when any registered engine has a runnable active installation."""
    try:
        store = data_store.get_store()
        from backend.engines.registry import engine_specs
        from backend.feature_flags import audio_cpp_enabled

        for spec in engine_specs():
            if spec.id == "audio_cpp" and not audio_cpp_enabled():
                continue
            active = store.get_active_engine_version(spec.id)
            if not active:
                continue
            paths = []
            for field in spec.active_path_fields:
                raw = str(active.get(field) or "")
                if not raw:
                    paths = []
                    break
                candidates = (
                    [raw]
                    if os.path.isabs(raw)
                    else [os.path.abspath(raw), os.path.join("/app", raw)]
                )
                existing = next((path for path in candidates if os.path.exists(path)), None)
                if not existing:
                    paths = []
                    break
                paths.append(existing)
            if paths and len(paths) == len(spec.active_path_fields):
                return True
        return False
    except Exception as e:
        logger.debug("any_active_runtime_in_db: %s", e)
        return False


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
    if _RUNTIME_TOKEN_RE.match(token) or _UNQUOTED_CMD_TOKEN_RE.match(token):
        return token
    return shlex.quote(token)


def _shell_join(tokens: List[str]) -> str:
    return " ".join(_quote_shell_token(str(token)) for token in tokens)


def _coerce_json_scalar_string(text: str) -> Any:
    """
    Parse UI / YAML string values into JSON scalars when obvious (``true``, ``false``, numbers).
    Unquoted words like ``medium`` stay strings.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text
    if isinstance(parsed, (dict, list)):
        return text
    return parsed


def _json_safe_filter_value(value: Any) -> Any:
    """Keep JSON-serializable request-default values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value != value:
            return None
        return value
    if isinstance(value, str):
        return _coerce_json_scalar_string(value)
    if isinstance(value, list):
        cleaned = [_json_safe_filter_value(v) for v in value]
        return [v for v in cleaned if v is not None]
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            key = str(k).strip()
            if not key:
                continue
            cv = _json_safe_filter_value(v)
            if cv is None:
                continue
            if isinstance(cv, list) and not cv:
                continue
            if isinstance(cv, dict) and not cv:
                continue
            out[key] = cv
        return out
    return None


def request_filters_and_aliases(
    *,
    stable_id: str,
    config: Dict[str, Any],
    model: Optional[Dict[str, Any]] = None,
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    """Collect the real model alias and request-default filters."""
    from backend.audio.voice_presets import audio_request_defaults_to_request_params

    filters = None
    routing_name = data_store.resolve_routing_name_from_config(config, stable_id)
    overrides = {}
    for item in config.get("set_params_by_id") or []:
        if not isinstance(item, dict) or not isinstance(item.get("params"), dict):
            continue
        suffix = str(item.get("sub_id") or "").strip()
        name = f"{routing_name}:{suffix}" if suffix else routing_name
        cleaned = _json_safe_filter_value(item["params"])
        if cleaned:
            overrides[name] = cleaned
    if overrides:
        filters = {"requestAliases": overrides}
    request_kwargs: Dict[str, Any] = {}
    if model and str(config.get("engine") or "") == "audio_cpp":
        from backend.reference_audio import reference_audio_storage_root

        artifact = model.get("artifact") if isinstance(model.get("artifact"), dict) else {}
        model_root = str(artifact.get("path") or model.get("local_path") or "")
        if model_root:
            request_kwargs["model_root"] = model_root
            request_kwargs["reference_root"] = reference_audio_storage_root(
                model_root,
                storage_key=model.get("id"),
            )
    request_params = audio_request_defaults_to_request_params(config, **request_kwargs)
    if request_params:
        cleaned = _json_safe_filter_value(request_params)
        if isinstance(cleaned, dict) and cleaned:
            if filters is None:
                filters = {}
            # set-if-undefined filters let each request override
            # Studio defaults, including individual nested engine options.
            cleaned = _soft_default_params(cleaned)
            existing = filters.get("setParams")
            if isinstance(existing, dict) and existing:
                filters["setParams"] = {**cleaned, **existing}
            else:
                filters["setParams"] = cleaned
    aliases: List[str] = []
    seen = {stable_id}
    for alias in data_store.collect_model_aliases(config, stable_id):
        if alias not in seen:
            aliases.append(alias)
            seen.add(alias)
    return filters, aliases


def _soft_default_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a JSON defaults object to Studio proxy soft filter paths.

    Escape literal option names such as ``family.preserve_punctuation``;
    Studio proxy uses GJSON/SJSON paths, where an unescaped dot means nesting.
    """
    out: Dict[str, Any] = {}

    def visit(value: Any, path: str) -> None:
        if isinstance(value, dict) and value:
            for key, child in value.items():
                escaped = re.sub(r"([\\.*?#|!:@])", r"\\\1", str(key))
                visit(child, f"{path}.{escaped}" if path else escaped)
        else:
            out[f"{path}?"] = value

    visit(params, "")
    return out


def _emit_param_tokens(key: str, value: Any, meta: dict) -> List[str]:
    primary_flag = meta.get("primary_flag")
    negative_flag = meta.get("negative_flag")
    value_kind = meta.get("value_kind") or "scalar"
    default = meta.get("default")
    list_metadata = (
        meta.get("type") == "list"
        or meta.get("nargs") in ("+", "*")
        or "..." in str(meta.get("value_spec") or meta.get("metavar") or "")
    )
    if list_metadata and value_kind not in {"repeatable", "csv_enum", "semicolon_enum"}:
        value_kind = "space_list"

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

    if value_kind == "space_list":
        items = value if isinstance(value, list) else shlex.split(str(value))
        values = [str(item) for item in items if not _is_empty_value(item)]
        return [str(primary_flag), *values] if values else []

    if value_kind == "json_object":
        if isinstance(value, dict):
            payload = json.dumps(value, separators=(",", ":"), ensure_ascii=False)
        elif isinstance(value, str):
            payload = value.strip()
            if not payload:
                return []
        else:
            return []
        if default is not None:
            default_payload = (
                json.dumps(default, separators=(",", ":"), ensure_ascii=False)
                if isinstance(default, dict)
                else str(default).strip()
            )
            if payload == default_payload:
                return []
        if not primary_flag:
            raise ValueError(f"Parameter {key} is missing primary_flag metadata")
        return [str(primary_flag), payload]

    if value_kind in {"csv_enum", "semicolon_enum"}:
        separator = ";" if value_kind == "semicolon_enum" else ","
        items = (
            value if isinstance(value, list) else ([value] if not _is_empty_value(value) else [])
        )
        parts = [str(item).strip() for item in items if not _is_empty_value(item)]
        if not parts:
            return []
        joined = separator.join(parts)
        default_joined = None
        if default is not None:
            if isinstance(default, list):
                default_joined = separator.join(str(x) for x in default if not _is_empty_value(x))
            else:
                default_joined = str(default)
        if default_joined is not None and joined == default_joined:
            return []
        if not primary_flag:
            raise ValueError(f"Parameter {key} is missing primary_flag metadata")
        return [str(primary_flag), joined]

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
    cache_key = build_dir or ""
    if cache_key in _cuda_library_path_cache:
        return _cuda_library_path_cache[cache_key]
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
    _cuda_library_path_cache[cache_key] = library_path
    return library_path


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
        resolved = resolve_gguf_model_path(model)
        if not _model_attr(model, "files"):
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


def _resolve_companion_path(
    model: Any,
    hf_id: Optional[str],
    hf_repo_arg: Optional[str],
    field: str,
) -> Optional[str]:
    if hf_repo_arg:
        return None
    filename = _model_attr(model, field)
    if not filename or not hf_id:
        return None
    try:
        from backend.models.hub import resolve_cached_model_path

        path = resolve_cached_model_path(hf_id, filename)
        if path and os.path.exists(path):
            if not os.path.isabs(path):
                return f"/app/{path}"
            return path
    except Exception as e:
        logger.debug("resolve %s failed for %s: %s", field, hf_id, e)
    return None


def _resolve_mmproj_path(
    model: Any, hf_id: Optional[str], hf_repo_arg: Optional[str]
) -> Optional[str]:
    return _resolve_companion_path(model, hf_id, hf_repo_arg, "mmproj_filename")


def _resolve_draft_companion(
    model: Any, hf_id: Optional[str], hf_repo_arg: Optional[str]
) -> tuple[Optional[str], Optional[str]]:
    """Return ``(draft_path, spec_type)`` for DSpark, DFlash or MTP companions.

    DSpark takes precedence, then DFlash, then MTP (all share ``--model-draft``).
    """
    dspark_path = _resolve_companion_path(model, hf_id, hf_repo_arg, "dspark_filename")
    if dspark_path:
        return dspark_path, "draft-dspark"
    dflash_path = _resolve_companion_path(model, hf_id, hf_repo_arg, "dflash_filename")
    if dflash_path:
        return dflash_path, "draft-dflash"
    mtp_path = _resolve_companion_path(model, hf_id, hf_repo_arg, "mtp_filename")
    if mtp_path:
        return mtp_path, "draft-mtp"
    return None, None


async def preview_model_command_async(model):
    return await asyncio.to_thread(preview_model_command, model)


def preview_model_command(model):
    from backend.proxy.launch_spec import compile_model_runtime, annotate_preview, redact_env_value

    try:
        compiled = compile_model_runtime(model)
        return annotate_preview(
            model,
            {
                "ok": True,
                "cmd": compiled.engine_command,
                "env": [
                    f"{key}={redact_env_value(key, value)}"
                    for key, value in sorted(compiled.launch.env.set.items())
                ],
                "filters": compiled.proxy.filters,
                "aliases": compiled.proxy.aliases,
                "proxy_name": compiled.proxy.model_id,
                "proxy_model_id": compiled.proxy.model_id,
                "routing_name": data_store.resolve_routing_name(model),
                "sidecar": compiled.launch.artifacts.get("server.json"),
            },
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "cmd": None, "env": None}
