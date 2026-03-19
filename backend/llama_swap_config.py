import yaml
import os
import subprocess
import re
import json
import shlex
from typing import Dict, Any, Set, Optional
from backend.logging_config import get_logger
from backend.huggingface import resolve_gguf_model_path_for_quant

logger = get_logger(__name__)

# Cache for supported flags per binary path
_supported_flags_cache: Dict[str, Set[str]] = {}


def _quote_arg_if_needed(arg: str) -> str:
    """
    Escape an argument if it contains spaces, special characters, or is a complex value.
    Since the command is already wrapped in single quotes in YAML, we escape special
    characters with backslashes rather than using quotes.

    Args:
        arg: The argument value to potentially escape

    Returns:
        The argument, escaped if necessary
    """
    if not isinstance(arg, str):
        arg = str(arg)

    # Check if escaping is needed
    needs_escaping = False

    # Always escape if it contains spaces, quotes, or shell special characters
    if any(
        char in arg
        for char in [
            " ",
            "\t",
            "\n",
            "|",
            "&",
            ";",
            "(",
            ")",
            "<",
            ">",
            "*",
            "?",
            "[",
            "]",
            "{",
            "}",
            "$",
            "`",
            "\\",
        ]
    ):
        needs_escaping = True

    # Escape regex patterns (common in --override-tensor)
    if re.search(r"[.*+?^${}|()\[\]\\]", arg):
        needs_escaping = True

    # Escape if it starts with a dash (could be confused with a flag)
    if arg.startswith("-") and not arg.startswith("--"):
        needs_escaping = True

    if not needs_escaping:
        return arg

    # Escape special characters for use within single-quoted string context
    # Since the command is wrapped in single quotes, we need to break out of them
    # Pattern: '...'\''escaped_value'\''...'
    # This breaks out of single quotes ('), adds escaped quote (\'), adds value,
    # adds escaped quote (\'), then continues with single quote (')

    # Within single quotes in bash, all characters are literal except single quotes.
    # To include a single quote, we break out: '...'\''...'
    # So we only need to escape single quotes in the value itself.

    # Escape any single quotes in the value by breaking out of quotes
    # Replace ' with '\'' (break out, add quote, continue)
    escaped = arg.replace("'", "'\\''")

    # Wrap in quote-break pattern: '\''escaped_value'\''
    # This ensures the value is properly inserted even if it contains quotes.
    # For values without quotes, this still works correctly.
    return "'\\''" + escaped + "'\\''"


def _coerce_model_config(config_value: Optional[Any]) -> Dict[str, Any]:
    if not config_value:
        return {}
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse stored model config while generating llama-swap config"
            )
            return {}
    return {}


def get_supported_flags(llama_server_path: str) -> Set[str]:
    """
    Get the set of supported flags for a llama-server binary by parsing --help output.
    Results are cached per binary path.

    Args:
        llama_server_path: Path to the llama-server binary

    Returns:
        Set of supported flag names (e.g., {"--typical-p", "--min-p", "--tfs"})
    """
    # Normalize path for caching
    normalized_path = (
        os.path.abspath(llama_server_path)
        if os.path.exists(llama_server_path)
        else llama_server_path
    )

    # Check cache first
    if normalized_path in _supported_flags_cache:
        return _supported_flags_cache[normalized_path]

    supported_flags = set()

    try:
        # Get the binary directory for LD_LIBRARY_PATH
        binary_dir = os.path.dirname(llama_server_path)
        working_dir = binary_dir

        # Fix path if needed
        if "/bin/" in binary_dir and "/build/bin/" not in binary_dir:
            binary_dir = binary_dir.replace("/bin/", "/build/bin/")
            working_dir = binary_dir

        # Run --help command
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = binary_dir

        result = subprocess.run(
            [llama_server_path, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
            cwd=working_dir,
            env=env,
        )

        if result.returncode == 0:
            # Parse the help output to find flags
            help_text = result.stdout

            # Extract all flags (both short and long form)
            # Pattern matches: -x, --flag, --flag-name, etc.
            # Matches flags at start of line, after whitespace, or after commas
            flag_pattern = r"(?:^|\s|,)(-{1,2}[a-zA-Z0-9][a-zA-Z0-9\-]*)"
            matches = re.findall(flag_pattern, help_text)

            for flag_match in matches:
                # Remove any leading whitespace or comma from the match
                flag = flag_match.lstrip(" ,\t")
                # Only add long form flags (--flag-name) to the set
                if flag.startswith("--"):
                    # Remove any trailing commas or whitespace
                    flag = flag.rstrip(" ,\t")
                    if flag:
                        supported_flags.add(flag)
                elif flag.startswith("-") and len(flag) == 2:
                    # Short flag, try to find long form in param_mapping
                    pass  # We'll handle this via param_mapping

            logger.debug(
                f"Found {len(supported_flags)} flags in {llama_server_path} help output"
            )
        else:
            logger.warning(
                f"Failed to get help from {llama_server_path}: {result.stdout}"
            )

    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout getting help from {llama_server_path}")
    except FileNotFoundError:
        logger.warning(f"Binary not found: {llama_server_path}")
    except Exception as e:
        logger.warning(f"Error checking flags for {llama_server_path}: {e}")

    # Cache the result (even if empty)
    _supported_flags_cache[normalized_path] = supported_flags

    return supported_flags


def is_flag_supported(
    config_key: str,
    flag_name: str,
    llama_server_path: str,
    param_mapping: Dict[str, list],
) -> bool:
    """
    Check if a flag is supported by the llama-server binary.

    Args:
        config_key: The config key (e.g., "typical_p")
        flag_name: The flag name to check (e.g., "--typical-p")
        llama_server_path: Path to the llama-server binary
        param_mapping: Dictionary mapping config keys to flag options

    Returns:
        True if the flag is supported, False otherwise
    """
    # If we have a param_mapping entry, check if the flag is in the mapping
    if config_key in param_mapping:
        flag_options = param_mapping[config_key]
        supported_flags = get_supported_flags(llama_server_path)

        # Check if any of the flag variants are supported
        for option in flag_options:
            if option in supported_flags:
                return True

        # If no flags found in help output, assume not supported (safer)
        # But only if we got help output (not empty due to error)
        if supported_flags:
            return False

    # If flag not in param_mapping or mapping check failed, assume supported
    # (fallback to original behavior for known flags)
    return True


def is_ik_llama_cpp(llama_server_path: Optional[str]) -> bool:
    """
    Detect if the binary is ik_llama.cpp by checking for ik-specific flags.
    Falls back to checking database repository_source if flag detection fails.

    Args:
        llama_server_path: Path to the llama-server binary

    Returns:
        True if ik_llama.cpp, False otherwise
    """
    if not llama_server_path:
        return False

    try:
        supported_flags = get_supported_flags(llama_server_path)
        # ik_llama.cpp has specific flags that don't exist in standard llama.cpp
        # Check for --mla-use, --smart-expert-reduction, or --attention-max-batch
        ik_specific_flags = [
            "--mla-use",
            "--smart-expert-reduction",
            "--attention-max-batch",
            "--no-fused-moe",
        ]
        if any(flag in supported_flags for flag in ik_specific_flags):
            logger.debug(f"Detected ik_llama.cpp via flag check: {llama_server_path}")
            return True
    except Exception as e:
        logger.debug(f"Error detecting ik_llama.cpp via flags: {e}")

    # Fallback: Check store for repository_source
    try:
        from backend.data_store import get_store
        store = get_store()
        active_version = store.get_active_engine_version("ik_llama") or store.get_active_engine_version("llama_cpp")
        if active_version and active_version.get("repository_source"):
            is_ik = active_version.get("repository_source") == "ik_llama.cpp"
            if is_ik:
                logger.debug(
                    f"Detected ik_llama.cpp via store repository_source: {active_version.get('repository_source')}"
                )
            return is_ik
    except Exception as e:
        logger.debug(f"Error checking store for ik_llama.cpp: {e}")

    return False


def get_param_mapping(is_ik: bool) -> Dict[str, list]:
    """
    Get the parameter mapping based on the llama.cpp version.

    Args:
        is_ik: True if ik_llama.cpp, False for standard llama.cpp

    Returns:
        Dictionary mapping config keys to flag options
    """
    base_mapping = {
        "ctx_size": ["-c", "--ctx-size"],
        "n_predict": ["-n", "--n-predict"],
        "threads": ["-t", "--threads"],
        "n_gpu_layers": ["-ngl", "--n-gpu-layers"],
        "batch_size": ["-b", "--batch-size"],
        "ubatch_size": ["-ub", "--ubatch-size"],
        "temp": ["--temp"],
        "temperature": ["--temp"],
        "top_k": ["--top-k"],
        "top_p": ["--top-p"],
        "min_p": ["--min-p"],
        "typical_p": ["--typical"],
        "tfs_z": [],  # Flag not supported in this version
        "repeat_penalty": ["--repeat-penalty"],
        "presence_penalty": ["--presence-penalty"],
        "frequency_penalty": ["--frequency-penalty"],
        "mirostat": ["--mirostat"],
        "seed": ["--seed"],
        "threads_batch": ["--threads-batch"],
        "parallel": ["--parallel"],
        "rope_freq_base": ["--rope-freq-base"],
        "rope_freq_scale": ["--rope-freq-scale"],
        "flash_attn": ["--flash-attn"],
        "yarn_ext_factor": ["--yarn-ext-factor"],
        "yarn_attn_factor": ["--yarn-attn-factor"],
        "rope_scaling": ["--rope-scaling"],
        "tensor_split": ["--tensor-split"],
        "main_gpu": ["--main-gpu"],
        "split_mode": ["-sm", "--split-mode"],
        "no_mmap": ["--no-mmap"],
        "mlock": ["--mlock"],
        "low_vram": ["--low-vram"],
        "logits_all": ["--logits-all"],
        "embedding": ["--embedding"],
        "cont_batching": ["--cont-batching"],
        "no_kv_offload": ["--no-kv-offload"],
        "cache_type_k": ["--cache-type-k"],
        "cache_type_v": ["--cache-type-v"],
        "grammar": ["--grammar"],
        "json_schema": ["--json-schema"],
        "yaml": ["--yaml"],
        "jinja": ["--jinja"],
        "moe_offload_pattern": [],  # Handled specially
        "moe_offload_custom": [],  # Custom MoE pattern (override-tensor), handled specially
        "cpu_moe": ["--cpu-moe"],
        "n_cpu_moe": ["--n-cpu-moe"],
        "override_tensor": ["-ot", "--override-tensor"],
        "host": ["--host"],
        "port": ["--port"],
    }

    # Version-specific mappings
    if is_ik:
        # ik_llama.cpp uses --mirostat-ent instead of --mirostat-tau
        base_mapping.update(
            {
                "mirostat_tau": [
                    "--mirostat-ent"
                ],  # ik_llama.cpp uses --mirostat-ent (tau parameter)
                "mirostat_eta": [
                    "--mirostat-lr"
                ],  # ik_llama.cpp uses --mirostat-lr (eta/learning rate parameter)
                # ik_llama.cpp specific flags
                "mla_attn": [
                    "-mla",
                    "--mla-use",
                ],  # MLA attention (--mla-use in ik_llama.cpp)
                "attn_max_batch": [
                    "-amb",
                    "--attention-max-batch",
                ],  # Attention max batch
                "fused_moe": [
                    "-fmoe",
                    "--fused-moe",
                ],  # Fused MoE (enabled by default, use --no-fused-moe to disable)
                "smart_expert_reduction": [
                    "-ser",
                    "--smart-expert-reduction",
                ],  # Smart expert reduction
            }
        )
    else:
        # Standard llama.cpp
        base_mapping.update(
            {
                "mirostat_tau": ["--mirostat-tau"],
                "mirostat_eta": ["--mirostat-eta"],
            }
        )

    return base_mapping


def get_active_binary_path_from_db() -> Optional[str]:
    """
    Gets the active llama-server binary path from the data store.

    Returns:
        Absolute path to the llama-server binary, or None if not found.
    """
    try:
        from backend.data_store import get_store

        store = get_store()
        for engine in ("llama_cpp", "ik_llama"):
            active_version = store.get_active_engine_version(engine)
            if not active_version or not active_version.get("binary_path"):
                continue
            binary_path = active_version["binary_path"]
            if not os.path.isabs(binary_path):
                binary_path = os.path.join("/app", binary_path)
            if os.path.exists(binary_path):
                return binary_path
            abs_path = os.path.abspath(binary_path)
            if os.path.exists(abs_path):
                return abs_path
        logger.warning("No active llama-cpp version found in data store")
        return None
    except Exception as e:
        logger.error(f"Error getting binary path from data store: {e}")
        return None


def _build_lmdeploy_cmd(
    model: Any,
    config: Dict[str, Any],
    lmdeploy_bin: str,
    _model_attr: Any,
) -> str:
    """Build lmdeploy serve api_server command for llama-swap config."""
    config = dict(config or {})
    hf_id = _model_attr(model, "huggingface_id")
    if not hf_id:
        raise ValueError("LMDeploy model must have huggingface_id")
    cmd_parts = [lmdeploy_bin, "serve", "api_server", hf_id]
    cmd_parts.extend(["--server-port", "${PORT}"])
    cmd_parts.extend(["--backend", "turbomind"])
    if config.get("session_len") is not None:
        cmd_parts.extend(["--session-len", str(config["session_len"])])
    if config.get("max_batch_size") is not None:
        cmd_parts.extend(["--max-batch-size", str(config["max_batch_size"])])
    if config.get("tensor_parallel") is not None:
        cmd_parts.extend(["--tp", str(config["tensor_parallel"])])
    if config.get("dtype"):
        cmd_parts.extend(["--dtype", str(config["dtype"])])
    if config.get("quant_policy") is not None:
        cmd_parts.extend(["--quant-policy", str(config["quant_policy"])])
    if config.get("enable_prefix_caching"):
        cmd_parts.append("--enable-prefix-caching")
    if config.get("chat_template"):
        cmd_parts.extend(["--chat-template", str(config["chat_template"])])
    if config.get("tool_call_parser"):
        cmd_parts.extend(["--tool-call-parser", _quote_arg_if_needed(str(config["tool_call_parser"]))])
    if config.get("reasoning_parser"):
        cmd_parts.extend(["--reasoning-parser", _quote_arg_if_needed(str(config["reasoning_parser"]))])
    # Escape single quotes in the command for bash -c '...'
    inner_cmd = " ".join(cmd_parts)
    inner_cmd = inner_cmd.replace("'", "'\\''")
    return f"bash -c '{inner_cmd}'"


def generate_llama_swap_config(
    models: Dict[str, Dict[str, Any]],
    llama_server_path: Optional[str] = None,
    all_models: list = None,
) -> str:
    """
    Generates the YAML configuration for llama-swap.

    Args:
        models: A dictionary where keys are proxy model names (e.g., "llama-3-8b-q4")
                and values are dictionaries containing 'model_path' and 'config' (llama.cpp params).
        llama_server_path: The absolute path to the llama-server binary. If None, will be retrieved from database.
        all_models: Optional list of all models from database to include in config.

    Returns:
        A string containing the YAML configuration.
    """
    # Get binary path from database if not provided
    if not llama_server_path:
        llama_server_path = get_active_binary_path_from_db()
        if not llama_server_path:
            raise ValueError(
                "No llama-server binary path provided and none found in database"
            )

    # Ensure absolute path
    if not os.path.isabs(llama_server_path):
        llama_server_path = os.path.join("/app", llama_server_path)

    if not os.path.exists(llama_server_path):
        raise ValueError(f"llama-server binary not found at: {llama_server_path}")

    # Detect ik_llama.cpp once at the start
    is_ik = is_ik_llama_cpp(llama_server_path)
    if is_ik:
        logger.info(
            f"Detected ik_llama.cpp binary at {llama_server_path}, using ik-specific parameter mappings"
        )
    else:
        logger.debug(
            f"Using standard llama.cpp parameter mappings for {llama_server_path}"
        )

    # Global llama-swap configuration
    config_data = {
        # Give large models plenty of time to load (default is 120s)
        "healthCheckTimeout": 600,
        # Add timestamps to llama-swap logs (v173+)
        "logTimeFormat": "2006-01-02 15:04:05",
        # Stream loading progress in API response during model swap (v171+)
        "sendLoadingState": True,
        "models": {},
    }

    def _model_attr(m: Any, key: str, default: Any = None) -> Any:
        """Get attribute from model (dict or object)."""
        if isinstance(m, dict):
            return m.get(key, default)
        return getattr(m, key, default)

    from backend.data_store import (
        generate_proxy_name as _generate_proxy_name,
        normalize_proxy_alias as _normalize_proxy_alias,
        resolve_proxy_name as _resolve_proxy_name,
    )

    # Resolve LMDeploy binary and build proxy->model map for overlay (used for both all_models and running overlay)
    lmdeploy_bin = None
    all_models_by_proxy: Dict[str, Any] = {}
    all_models_by_legacy_proxy: Dict[str, Any] = {}
    try:
        from backend.data_store import get_store as _get_store
        store = _get_store()
        # Prefer the active versioned LMDeploy engine, same pattern as llama_cpp.
        active_lmdeploy = store.get_active_engine_version("lmdeploy")
        venv = active_lmdeploy.get("venv_path") if active_lmdeploy else None
        # Fallback to legacy single-status layout if no active version is found.
        if not venv:
            legacy_status = store.get_lmdeploy_status()
            if legacy_status.get("installed"):
                venv = legacy_status.get("venv_path")
        if venv:
            # Ensure the venv path still exists before resolving the binary.
            if not os.path.isabs(venv):
                venv = os.path.join("/app", venv)
            if os.path.isdir(venv):
                candidate = os.path.join(venv, "bin", "lmdeploy")
                if not os.path.isabs(candidate):
                    candidate = os.path.join("/app", candidate)
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    lmdeploy_bin = candidate
                else:
                    logger.debug(
                        f"LMDeploy binary not found or not executable at {candidate}; "
                        "LMDeploy engine entries will be skipped in llama-swap config"
                    )
            else:
                logger.debug(
                    f"LMDeploy venv_path does not exist at {venv}; "
                    "LMDeploy engine entries will be skipped in llama-swap config"
                )
    except Exception as e:
        logger.debug(f"Could not resolve LMDeploy binary: {e}")

    # First, add all models from the data store (if provided)
    if all_models:
        for model in all_models:
            proxy_model_name = _resolve_proxy_name(model)
            if not proxy_model_name:
                logger.warning(
                    f"Model '{_model_attr(model, 'display_name') or _model_attr(model, 'name')}' does not have a proxy_name set, skipping"
                )
                continue
            all_models_by_proxy[proxy_model_name] = model
            legacy_proxy_name = _normalize_proxy_alias(_model_attr(model, "proxy_name"))
            if legacy_proxy_name and legacy_proxy_name != proxy_model_name:
                all_models_by_legacy_proxy[legacy_proxy_name] = model
            generated_proxy_name = _generate_proxy_name(
                _model_attr(model, "huggingface_id", ""),
                _model_attr(model, "quantization"),
            )
            if generated_proxy_name and generated_proxy_name != proxy_model_name:
                all_models_by_legacy_proxy[generated_proxy_name] = model

            # Engine is stored in `model["config"]["engine"]` by the UI.
            config = _coerce_model_config(_model_attr(model, "config"))
            engine = config.get("engine")
            # LMDeploy-backed models are detected strictly by engine, not by format.
            is_lmdeploy = engine == "lmdeploy"
            if is_lmdeploy and lmdeploy_bin:
                try:
                    cmd_with_env = _build_lmdeploy_cmd(model, config, lmdeploy_bin, _model_attr)
                    config_data["models"][proxy_model_name] = {"cmd": cmd_with_env}
                except Exception as e:
                    logger.warning(f"Failed to build LMDeploy cmd for {proxy_model_name}: {e}")
                continue

            hf_id = _model_attr(model, "huggingface_id")
            quantization = _model_attr(model, "quantization")

            # Prefer existing on-disk path when we have hf_id+quant to avoid
            # llama.cpp re-downloading and duplicating storage. Fall back to --hf-repo if not found.
            hf_repo_arg = None
            model_path = None
            if hf_id and quantization:
                resolved = resolve_gguf_model_path_for_quant(hf_id, str(quantization))
                if resolved and os.path.exists(resolved):
                    model_path = resolved
                else:
                    hf_repo_arg = f"{hf_id}:{str(quantization).lower()}"
            if not model_path:
                legacy = _model_attr(model, "file_path")
                if legacy:
                    model_path = legacy if os.path.isabs(legacy) else f"/app/{legacy}"

            # If we don't have either an HF repo+quant or a legacy path, skip.
            if not hf_repo_arg and not model_path:
                logger.warning(
                    f"Model '{proxy_model_name}' path could not be resolved (hf_id={hf_id}), skipping"
                )
                continue

            # Ensure absolute path when we are in local-path mode.
            if model_path and not os.path.isabs(model_path):
                model_path = f"/app/{model_path}"

            # Get the working directory and build directory for LD_LIBRARY_PATH
            working_dir = os.path.dirname(llama_server_path)
            build_dir = os.path.dirname(llama_server_path)
            binary_name = os.path.basename(llama_server_path)

            # Convert paths to absolute paths
            if not os.path.isabs(working_dir):
                working_dir = f"/app/{working_dir}"
            if not os.path.isabs(build_dir):
                build_dir = f"/app/{build_dir}"

            # Fix: The shared libraries are in the build/bin directory, not bin directory
            if "/bin/" in build_dir:
                build_dir = build_dir.replace("/bin/", "/build/bin/")

            # Fix: Use the llama-server from build/bin directory, not bin directory
            if "/bin/" in working_dir:
                working_dir = working_dir.replace("/bin/", "/build/bin/")

            # Parse existing config if available
            if proxy_model_name and config.get("jinja") is not None:
                logger.debug(
                    f"Model {proxy_model_name}: jinja={config.get('jinja')} (type: {type(config.get('jinja'))})"
                )

            # Build llama.cpp command arguments (excluding the base launcher).
            # We keep the first 3 entries in cmd_args unused; only cmd_args[3:]
            # (starting from "--port") are appended to the final command string.
            cmd_args = [
                None,
                None,
                None,
                "--port",
                "${PORT}",
            ]

            # If the user provided a model_alias in config, propagate it to llama.cpp
            # via --alias so that /v1/models exposes this name. Value is quoted when needed.
            alias_for_api = config.get("model_alias")
            if isinstance(alias_for_api, str) and alias_for_api.strip():
                cmd_args.extend(["--alias", _quote_arg_if_needed(alias_for_api.strip())])

            # Vision: if model has mmproj (multimodal projector) and we're using a
            # local model path, add --mmproj so vision is available. When using
            # --hf-repo, llama.cpp will auto-download mmproj if available.
            mmproj_filename = _model_attr(model, "mmproj_filename")
            if mmproj_filename and hf_id and not hf_repo_arg:
                from backend.huggingface import resolve_cached_model_path

                mmproj_path = resolve_cached_model_path(hf_id, mmproj_filename)
                if mmproj_path and os.path.exists(mmproj_path):
                    if not os.path.isabs(mmproj_path):
                        mmproj_path = f"/app/{mmproj_path}"
                    quoted_mmproj = _quote_arg_if_needed(mmproj_path)
                    cmd_args.extend(["--mmproj", quoted_mmproj])

            # Default values to skip (these cause errors if flag isn't supported)
            default_values = {
                "typical_p": 1.0,
                "min_p": 0.0,
                "tfs_z": 1.0,
                "presence_penalty": 0.0,
                "frequency_penalty": 0.0,
                "mirostat": 0,
                "seed": -1,
                "rope_freq_scale": 1.0,
                "yarn_ext_factor": 1.0,
                "yarn_attn_factor": 1.0,
            }

            # Use the already-detected ik_llama.cpp status
            param_mapping = get_param_mapping(is_ik)

            # Emit standard key/value flags
            # Track if --temp has been added to avoid duplicates (temp and temperature both map to --temp)
            temp_flag_added = False

            for key, value in config.items():
                if key == "jinja":
                    logger.debug(
                        f"Processing jinja for {proxy_model_name}: value={value} (type={type(value)})"
                    )

                # Skip temperature if temp is already set (they both map to --temp)
                if (
                    key == "temperature"
                    and "temp" in config
                    and config.get("temp") is not None
                ):
                    logger.debug(
                        f"Skipping temperature (using temp instead) for {proxy_model_name}"
                    )
                    continue

                # Skip cpu_moe and n_cpu_moe here - they're handled specially below with moe_offload_pattern
                if key in ("cpu_moe", "n_cpu_moe"):
                    continue

                if key in param_mapping and value is not None:
                    # Skip if param_mapping has an empty list (unsupported flag)
                    flag_options = param_mapping[key]
                    if not flag_options:
                        logger.debug(
                            f"Skipping unsupported flag: {key} (empty mapping)"
                        )
                        continue

                    # Check if flag is supported by the binary (runtime check)
                    # Only skip if we have valid help output and flag is not supported
                    supported_flags = get_supported_flags(llama_server_path)
                    if (
                        supported_flags
                    ):  # Only validate if we successfully parsed help output
                        flag_supported = is_flag_supported(
                            key, flag_options[0], llama_server_path, param_mapping
                        )
                        if not flag_supported:
                            if key == "jinja":
                                logger.warning(
                                    f"Flag --jinja not supported according to help output for {proxy_model_name}"
                                )
                            logger.debug(
                                f"Skipping unsupported flag: {key} for {llama_server_path}"
                            )
                            continue

                    # Skip default values (which cause errors if flag isn't supported)
                    if key in default_values:
                        if (
                            isinstance(value, (int, float))
                            and abs(value - default_values[key]) < 1e-6
                        ):
                            continue

                    if isinstance(value, bool):
                        flag_name = flag_options[0]
                        if key == "flash_attn":
                            if value:
                                cmd_args.extend([flag_name, "on"])
                        elif value:
                            cmd_args.append(flag_name)
                    elif isinstance(value, str) and value.strip() == "":
                        continue
                    else:
                        # Check if this is --temp flag and already added
                        if flag_options[0] == "--temp":
                            if temp_flag_added:
                                logger.debug(
                                    f"Skipping duplicate --temp flag for {proxy_model_name}"
                                )
                                continue
                            temp_flag_added = True
                        # Quote complex values (grammar, json_schema, yaml, override_tensor, etc.)
                        value_str = str(value)
                        if key in (
                            "grammar",
                            "json_schema",
                            "yaml",
                            "override_tensor",
                        ) or flag_options[0] in (
                            "--grammar",
                            "--json-schema",
                            "--yaml",
                            "--override-tensor",
                            "-ot",
                        ):
                            value_str = _quote_arg_if_needed(value_str)
                        cmd_args.extend([flag_options[0], value_str])

            # Special handling: MoE offload flags
            # Check for direct cpu_moe or n_cpu_moe parameters first (these take precedence)
            cpu_moe_direct = config.get("cpu_moe")
            n_cpu_moe_direct = config.get("n_cpu_moe")
            moe_pattern = config.get("moe_offload_pattern")

            if n_cpu_moe_direct is not None:
                # Direct n_cpu_moe parameter takes precedence
                if isinstance(n_cpu_moe_direct, (int, float)) and n_cpu_moe_direct > 0:
                    cmd_args.extend(["--n-cpu-moe", str(int(n_cpu_moe_direct))])
            elif cpu_moe_direct is True or moe_pattern == "cpu":
                # Direct cpu_moe boolean flag OR pattern "cpu" maps to --cpu-moe flag
                cmd_args.append("--cpu-moe")
            elif moe_pattern and moe_pattern != "none":
                # Fall back to moe_offload_pattern for other patterns
                if isinstance(moe_pattern, (int, float)) and moe_pattern > 0:
                    # Numeric pattern maps to --n-cpu-moe
                    cmd_args.extend(["--n-cpu-moe", str(int(moe_pattern))])
                elif (
                    isinstance(moe_pattern, str)
                    and moe_pattern != "cpu"
                    and moe_pattern != "n_layers"
                ):
                    # Try to parse as number for backward compatibility
                    try:
                        n_layers = int(moe_pattern)
                        if n_layers > 0:
                            cmd_args.extend(["--n-cpu-moe", str(n_layers)])
                    except ValueError:
                        pass  # Invalid pattern, skip (other patterns use moe_offload_custom)

            # Special handling: Custom MoE offload pattern (override-tensor)
            moe_custom = config.get("moe_offload_custom")
            if moe_custom and isinstance(moe_custom, str) and moe_custom.strip():
                # Custom pattern uses --override-tensor flag
                # Format: e.g., ".ffn_.*_exps.=CPU" or "exps=CPU"
                # Quote complex regex patterns
                quoted_value = _quote_arg_if_needed(moe_custom.strip())
                cmd_args.extend(["--override-tensor", quoted_value])

            # Special handling: stop words list (multiple --stop flags)
            if isinstance(config.get("stop"), list):
                for s in config.get("stop"):
                    if isinstance(s, str) and s.strip():
                        cmd_args.extend(["--stop", s])

            # Free-form custom args appended verbatim at the end
            if (
                isinstance(config.get("customArgs"), str)
                and config.get("customArgs").strip()
            ):
                cmd_args.append(config.get("customArgs").strip())

            # Handle host parameter (use config value or default to 0.0.0.0)
            host_value = config.get("host", "0.0.0.0")
            if "--host" not in cmd_args:
                cmd_args.extend(["--host", str(host_value)])

            # Ensure LD_LIBRARY_PATH points to the directory containing the shared libraries
            # The shared libraries are in the same directory as the binary
            library_path = build_dir
            
            # Add CUDA library path if CUDA is installed
            try:
                from backend.cuda_installer import get_cuda_installer
                cuda_installer = get_cuda_installer()
                cuda_path = cuda_installer._get_cuda_path()
                if cuda_path:
                    cuda_lib = os.path.join(cuda_path, "lib64")
                    if os.path.exists(cuda_lib):
                        # Prepend CUDA lib path to library_path
                        library_path = f"{cuda_lib}:{library_path}"
                        logger.debug(f"Added CUDA library path to LD_LIBRARY_PATH: {cuda_lib}")
            except Exception as e:
                logger.debug(f"Could not get CUDA library path: {e}")

            # Create the command with proper shell syntax for environment variables.
            # Use --model when we have a local path (resolved or legacy); otherwise --hf-repo.
            if hf_repo_arg:
                launcher = f"./{binary_name} --hf-repo {hf_repo_arg}"
            else:
                quoted_model_path = _quote_arg_if_needed(model_path)
                launcher = f"./{binary_name} --model {quoted_model_path}"

            cmd_with_env = (
                f"bash -c 'cd {working_dir} && LD_LIBRARY_PATH={library_path} {launcher} "
                + " ".join(cmd_args[3:])
                + "'"
            )

            config_data["models"][proxy_model_name] = {"cmd": cmd_with_env}

    # Then, add/update with running models (these take precedence for active models)
    for proxy_model_name, model_data in models.items():
        overlay_model = all_models_by_proxy.get(proxy_model_name) or all_models_by_legacy_proxy.get(proxy_model_name)
        resolved_proxy_model_name = (
            _resolve_proxy_name(overlay_model)
            if overlay_model
            else _normalize_proxy_alias(model_data.get("config", {}).get("model_alias")) or proxy_model_name
        )
        overlay_config = _coerce_model_config(_model_attr(overlay_model, "config")) if overlay_model else {}
        engine = overlay_config.get("engine")
        # For overlay models, also rely solely on the engine flag to detect LMDeploy.
        is_lmdeploy_overlay = engine == "lmdeploy" and lmdeploy_bin and overlay_model
        if is_lmdeploy_overlay:
            config = _coerce_model_config(model_data.get("config"))
            try:
                cmd_with_env = _build_lmdeploy_cmd(overlay_model, config, lmdeploy_bin, _model_attr)
                config_data["models"].pop(proxy_model_name, None)
                config_data["models"][resolved_proxy_model_name] = {"cmd": cmd_with_env}
            except Exception as e:
                logger.warning(f"Failed to build LMDeploy overlay cmd for {resolved_proxy_model_name}: {e}")
            continue

        model_path = model_data["model_path"]
        llama_cpp_config = model_data["config"]

        # Build llama.cpp command arguments (using full path to llama-server).
        # For overlay models, prefer existing on-disk path when available to avoid duplicate storage.
        hf_id_overlay = _model_attr(overlay_model, "huggingface_id") if overlay_model else None
        quantization_overlay = _model_attr(overlay_model, "quantization") if overlay_model else None
        hf_repo_arg_overlay = None
        if hf_id_overlay and quantization_overlay:
            resolved = resolve_gguf_model_path_for_quant(hf_id_overlay, str(quantization_overlay))
            if resolved and os.path.exists(resolved):
                model_path = resolved
            else:
                hf_repo_arg_overlay = f"{hf_id_overlay}:{str(quantization_overlay).lower()}"

        # Quote model path if it contains spaces or special characters (local-path mode).
        quoted_model_path = _quote_arg_if_needed(model_path)
        cmd_args = [
            None,
            None,
            None,
            "--port",
            "${PORT}",
        ]
        # Propagate model_alias from the live llama_cpp_config if present so that
        # llama.cpp exposes this name via /v1/models. Value is quoted when needed.
        alias_for_api_overlay = llama_cpp_config.get("model_alias")
        if isinstance(alias_for_api_overlay, str) and alias_for_api_overlay.strip():
            cmd_args.extend(["--alias", _quote_arg_if_needed(alias_for_api_overlay.strip())])
        # Vision: add --mmproj if model has mmproj_filename
        if overlay_model and not hf_repo_arg_overlay:
            mmproj_fn = _model_attr(overlay_model, "mmproj_filename")
            if mmproj_fn and hf_id_overlay:
                from backend.huggingface import resolve_cached_model_path

                mmproj_path = resolve_cached_model_path(hf_id_overlay, mmproj_fn)
                if mmproj_path and os.path.exists(mmproj_path):
                    if not os.path.isabs(mmproj_path):
                        mmproj_path = f"/app/{mmproj_path}"
                    cmd_args.extend(["--mmproj", _quote_arg_if_needed(mmproj_path)])

        # Default values to skip (these cause errors if flag isn't supported)
        default_values = {
            "typical_p": 1.0,
            "min_p": 0.0,
            "tfs_z": 1.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "mirostat": 0,
            "seed": -1,
            "rope_freq_scale": 1.0,
            "yarn_ext_factor": 1.0,
            "yarn_attn_factor": 1.0,
        }

        # Use the already-detected ik_llama.cpp status
        param_mapping = get_param_mapping(is_ik)

        # Emit standard key/value flags
        # Track if --temp has been added to avoid duplicates (temp and temperature both map to --temp)
        temp_flag_added = False

        for key, value in llama_cpp_config.items():
            # Skip temperature if temp is already set (they both map to --temp)
            if (
                key == "temperature"
                and "temp" in llama_cpp_config
                and llama_cpp_config.get("temp") is not None
            ):
                logger.debug(f"Skipping temperature (using temp instead)")
                continue

            # Skip cpu_moe and n_cpu_moe here - they're handled specially below with moe_offload_pattern
            if key in ("cpu_moe", "n_cpu_moe"):
                continue

            if key in param_mapping and value is not None:
                # Skip if param_mapping has an empty list (unsupported flag)
                flag_options = param_mapping[key]
                if not flag_options:
                    logger.debug(f"Skipping unsupported flag: {key} (empty mapping)")
                    continue

                # Check if flag is supported by the binary (runtime check)
                # Only skip if we have valid help output and flag is not supported
                supported_flags = get_supported_flags(llama_server_path)
                if (
                    supported_flags
                ):  # Only validate if we successfully parsed help output
                    flag_supported = is_flag_supported(
                        key, flag_options[0], llama_server_path, param_mapping
                    )
                    if not flag_supported:
                        logger.debug(
                            f"Skipping unsupported flag: {key} for {llama_server_path}"
                        )
                        continue

                # Skip default values (which cause errors if flag isn't supported)
                if key in default_values:
                    if (
                        isinstance(value, (int, float))
                        and abs(value - default_values[key]) < 1e-6
                    ):
                        continue

                if isinstance(value, bool):
                    flag_name = flag_options[0]
                    if key == "flash_attn":
                        if value:
                            cmd_args.extend([flag_name, "on"])
                    elif value:
                        cmd_args.append(flag_name)
                elif isinstance(value, str) and value.strip() == "":
                    continue
                else:
                    # Check if this is --temp flag and already added
                    if flag_options[0] == "--temp":
                        if temp_flag_added:
                            logger.debug(f"Skipping duplicate --temp flag")
                            continue
                        temp_flag_added = True
                    # Quote complex values (grammar, json_schema, yaml, override_tensor, etc.)
                    value_str = str(value)
                    if key in (
                        "grammar",
                        "json_schema",
                        "yaml",
                        "override_tensor",
                    ) or flag_options[0] in (
                        "--grammar",
                        "--json-schema",
                        "--yaml",
                        "--override-tensor",
                        "-ot",
                    ):
                        value_str = _quote_arg_if_needed(value_str)
                    cmd_args.extend([flag_options[0], value_str])

        # Special handling: MoE offload flags
        # Check for direct cpu_moe or n_cpu_moe parameters first (these take precedence)
        cpu_moe_direct = llama_cpp_config.get("cpu_moe")
        n_cpu_moe_direct = llama_cpp_config.get("n_cpu_moe")
        moe_pattern = llama_cpp_config.get("moe_offload_pattern")

        if n_cpu_moe_direct is not None:
            # Direct n_cpu_moe parameter takes precedence
            if isinstance(n_cpu_moe_direct, (int, float)) and n_cpu_moe_direct > 0:
                cmd_args.extend(["--n-cpu-moe", str(int(n_cpu_moe_direct))])
        elif cpu_moe_direct is True or moe_pattern == "cpu":
            # Direct cpu_moe boolean flag OR pattern "cpu" maps to --cpu-moe flag
            cmd_args.append("--cpu-moe")
        elif moe_pattern and moe_pattern != "none":
            # Fall back to moe_offload_pattern for other patterns
            if isinstance(moe_pattern, (int, float)) and moe_pattern > 0:
                # Numeric pattern maps to --n-cpu-moe
                cmd_args.extend(["--n-cpu-moe", str(int(moe_pattern))])
            elif (
                isinstance(moe_pattern, str)
                and moe_pattern != "cpu"
                and moe_pattern != "n_layers"
            ):
                # Try to parse as number for backward compatibility
                try:
                    n_layers = int(moe_pattern)
                    if n_layers > 0:
                        cmd_args.extend(["--n-cpu-moe", str(n_layers)])
                except ValueError:
                    pass  # Invalid pattern, skip (other patterns use moe_offload_custom)

        # Special handling: Custom MoE offload pattern (override-tensor)
        moe_custom = llama_cpp_config.get("moe_offload_custom")
        if moe_custom and isinstance(moe_custom, str) and moe_custom.strip():
            # Custom pattern uses --override-tensor flag
            # Format: e.g., ".ffn_.*_exps.=CPU" or "exps=CPU"
            # Quote complex regex patterns
            quoted_value = _quote_arg_if_needed(moe_custom.strip())
            cmd_args.extend(["--override-tensor", quoted_value])

        # Special handling: stop words list (multiple --stop flags)
        if isinstance(llama_cpp_config.get("stop"), list):
            for s in llama_cpp_config.get("stop"):
                if isinstance(s, str) and s.strip():
                    cmd_args.extend(["--stop", s])

        # Free-form custom args appended verbatim at the end
        if (
            isinstance(llama_cpp_config.get("customArgs"), str)
            and llama_cpp_config.get("customArgs").strip()
        ):
            cmd_args.append(llama_cpp_config.get("customArgs").strip())

        # Handle host parameter (use config value or default to 0.0.0.0)
        host_value = llama_cpp_config.get("host", "0.0.0.0")
        if "--host" not in cmd_args:
            cmd_args.extend(["--host", str(host_value)])

        # Convert model path to absolute path
        if not os.path.isabs(model_path):
            model_path = f"/app/{model_path}"

        # Get the working directory and build directory for LD_LIBRARY_PATH
        working_dir = os.path.dirname(llama_server_path)
        build_dir = os.path.dirname(llama_server_path)
        binary_name = os.path.basename(llama_server_path)

        # Convert paths to absolute paths
        if not os.path.isabs(working_dir):
            working_dir = f"/app/{working_dir}"
        if not os.path.isabs(build_dir):
            build_dir = f"/app/{build_dir}"

        # Fix: The shared libraries are in the build/bin directory, not bin directory
        if "/bin/" in build_dir and "/build/bin/" not in build_dir:
            build_dir = build_dir.replace("/bin/", "/build/bin/")

        # Fix: Use the llama-server from build/bin directory, not bin directory
        if "/bin/" in working_dir and "/build/bin/" not in working_dir:
            working_dir = working_dir.replace("/bin/", "/build/bin/")

        # Ensure LD_LIBRARY_PATH points to the directory containing the shared libraries
        # The shared libraries are in the same directory as the binary
        library_path = build_dir

        # Create the command with proper shell syntax for environment variables.
        if hf_repo_arg_overlay:
            launcher = f"./{binary_name} --hf-repo {hf_repo_arg_overlay}"
        else:
            launcher = f"./{binary_name} --model {quoted_model_path}"

        cmd_with_env = (
            f"bash -c 'cd {working_dir} && LD_LIBRARY_PATH={library_path} {launcher} "
            + " ".join(cmd_args[3:])
            + "'"
        )

        config_data["models"].pop(proxy_model_name, None)
        config_data["models"][resolved_proxy_model_name] = {"cmd": cmd_with_env}

    # Add groups configuration to allow multiple models to run simultaneously
    # Note: This means models won't be unloaded when new ones start - user must manage memory
    if config_data["models"]:
        config_data["groups"] = {
            "concurrent_models": {
                "swap": False,  # Allow multiple models to run at the same time
                "exclusive": False,  # Don't unload other groups when this group runs
                "members": list(config_data["models"].keys()),
            }
        }

    return yaml.dump(config_data, sort_keys=False, indent=2)
