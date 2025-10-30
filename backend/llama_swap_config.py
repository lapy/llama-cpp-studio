import yaml
import os
import subprocess
import re
from typing import Dict, Any, Set, Optional
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Cache for supported flags per binary path
_supported_flags_cache: Dict[str, Set[str]] = {}

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
    normalized_path = os.path.abspath(llama_server_path) if os.path.exists(llama_server_path) else llama_server_path
    
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
            env=env
        )
        
        if result.returncode == 0:
            # Parse the help output to find flags
            help_text = result.stdout
            
            # Extract all flags (both short and long form)
            # Pattern matches: -x, --flag, --flag-name, etc.
            # Matches flags at start of line, after whitespace, or after commas
            flag_pattern = r'(?:^|\s|,)(-{1,2}[a-zA-Z0-9][a-zA-Z0-9\-]*)'
            matches = re.findall(flag_pattern, help_text)
            
            for flag_match in matches:
                # Remove any leading whitespace or comma from the match
                flag = flag_match.lstrip(' ,\t')
                # Only add long form flags (--flag-name) to the set
                if flag.startswith('--'):
                    # Remove any trailing commas or whitespace
                    flag = flag.rstrip(' ,\t')
                    if flag:
                        supported_flags.add(flag)
                elif flag.startswith('-') and len(flag) == 2:
                    # Short flag, try to find long form in param_mapping
                    pass  # We'll handle this via param_mapping
            
            logger.debug(f"Found {len(supported_flags)} flags in {llama_server_path} help output")
        else:
            logger.warning(f"Failed to get help from {llama_server_path}: {result.stdout}")
            
    except subprocess.TimeoutExpired:
        logger.warning(f"Timeout getting help from {llama_server_path}")
    except FileNotFoundError:
        logger.warning(f"Binary not found: {llama_server_path}")
    except Exception as e:
        logger.warning(f"Error checking flags for {llama_server_path}: {e}")
    
    # Cache the result (even if empty)
    _supported_flags_cache[normalized_path] = supported_flags
    
    return supported_flags

def is_flag_supported(config_key: str, flag_name: str, llama_server_path: str, param_mapping: Dict[str, list]) -> bool:
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

def get_active_binary_path_from_db() -> Optional[str]:
    """
    Gets the active llama-server binary path from the database.
    
    Returns:
        Absolute path to the llama-server binary, or None if not found.
    """
    try:
        from backend.database import SessionLocal, LlamaVersion
        db = SessionLocal()
        try:
            active_version = db.query(LlamaVersion).filter(LlamaVersion.is_active == True).first()
            if not active_version or not active_version.binary_path:
                logger.warning("No active llama-cpp version found in database")
                return None
            
            # Convert to absolute path
            binary_path = active_version.binary_path
            if not os.path.isabs(binary_path):
                binary_path = os.path.join("/app", binary_path)
            
            # Verify the path exists
            if os.path.exists(binary_path):
                return binary_path
            else:
                logger.warning(f"Binary path from database does not exist: {binary_path}")
                return None
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Error getting binary path from database: {e}")
        return None

def generate_llama_swap_config(models: Dict[str, Dict[str, Any]], llama_server_path: Optional[str] = None, all_models: list = None) -> str:
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
            raise ValueError("No llama-server binary path provided and none found in database")
    
    # Ensure absolute path
    if not os.path.isabs(llama_server_path):
        llama_server_path = os.path.join("/app", llama_server_path)
    
    if not os.path.exists(llama_server_path):
        raise ValueError(f"llama-server binary not found at: {llama_server_path}")
    config_data = {
        "models": {}
    }

    # First, add all models from the database (if provided)
    if all_models:
        for model in all_models:
            # Use the centralized proxy name from the database
            if not model.proxy_name:
                logger.warning(f"Model '{model.name}' does not have a proxy_name set, skipping")
                continue
                
            proxy_model_name = model.proxy_name
            model_path = model.file_path
            
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
            if "/bin/" in build_dir:
                build_dir = build_dir.replace("/bin/", "/build/bin/")
            
            # Fix: Use the llama-server from build/bin directory, not bin directory
            if "/bin/" in working_dir:
                working_dir = working_dir.replace("/bin/", "/build/bin/")
            
            # Parse existing config if available
            config = {}
            if model.config:
                import json
                try:
                    config = json.loads(model.config)
                    if proxy_model_name and "jinja" in config:
                        logger.debug(f"Model {proxy_model_name}: jinja={config.get('jinja')} (type: {type(config.get('jinja'))})")
                except:
                    config = {}
            
            # Build llama.cpp command arguments
            cmd_args = [llama_server_path, "--model", model_path, "--port", "${PORT}"]
            
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
                "yarn_attn_factor": 1.0
            }
            
            param_mapping = {
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
                "mirostat_tau": ["--mirostat-tau"],
                "mirostat_eta": ["--mirostat-eta"],
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
                # MoE flags (moe_offload_pattern: 'none' = no flag, 'cpu' = --cpu-moe, number = --n-cpu-moe N)
                "moe_offload_pattern": [],  # Handled specially below
                "moe_offload_custom": []  # Custom MoE pattern, handled specially
            }

            # Emit standard key/value flags
            for key, value in config.items():
                if key == "jinja":
                    logger.debug(f"Processing jinja for {proxy_model_name}: value={value} (type={type(value)})")
                
                if key in param_mapping and value is not None:
                    # Skip if param_mapping has an empty list (unsupported flag)
                    flag_options = param_mapping[key]
                    if not flag_options:
                        logger.debug(f"Skipping unsupported flag: {key} (empty mapping)")
                        continue
                    
                    # Check if flag is supported by the binary (runtime check)
                    # Only skip if we have valid help output and flag is not supported
                    supported_flags = get_supported_flags(llama_server_path)
                    if supported_flags:  # Only validate if we successfully parsed help output
                        flag_supported = is_flag_supported(key, flag_options[0], llama_server_path, param_mapping)
                        if not flag_supported:
                            if key == "jinja":
                                logger.warning(f"Flag --jinja not supported according to help output for {proxy_model_name}")
                            logger.debug(f"Skipping unsupported flag: {key} for {llama_server_path}")
                            continue
                    
                    # Skip default values (which cause errors if flag isn't supported)
                    if key in default_values:
                        if isinstance(value, (int, float)) and abs(value - default_values[key]) < 1e-6:
                            continue
                    
                    if isinstance(value, bool):
                        if value:
                            flag_name = flag_options[0]
                            cmd_args.append(flag_name)
                    elif isinstance(value, str) and value.strip() == "":
                        continue
                    else:
                        cmd_args.extend([flag_options[0], str(value)])

            # Special handling: MoE offload flags
            moe_pattern = config.get("moe_offload_pattern")
            if moe_pattern and moe_pattern != "none":
                if moe_pattern == "cpu":
                    cmd_args.append("--cpu-moe")
                elif isinstance(moe_pattern, (int, float)) and moe_pattern > 0:
                    cmd_args.extend(["--n-cpu-moe", str(int(moe_pattern))])
                elif isinstance(moe_pattern, str):
                    try:
                        n_layers = int(moe_pattern)
                        if n_layers > 0:
                            cmd_args.extend(["--n-cpu-moe", str(n_layers)])
                    except ValueError:
                        pass  # Invalid pattern, skip
            
            # Special handling: stop words list (multiple --stop flags)
            if isinstance(config.get("stop"), list):
                for s in config.get("stop"):
                    if isinstance(s, str) and s.strip():
                        cmd_args.extend(["--stop", s])

            # Free-form custom args appended verbatim at the end
            if isinstance(config.get("customArgs"), str) and config.get("customArgs").strip():
                cmd_args.append(config.get("customArgs").strip())

            # Ensure --host 0.0.0.0 is always present for llama-server
            if "--host" not in cmd_args and "0.0.0.0" not in cmd_args:
                cmd_args.extend(["--host", "0.0.0.0"])

            # Ensure LD_LIBRARY_PATH points to the directory containing the shared libraries
            # The shared libraries are in the same directory as the binary
            library_path = build_dir
            
            # Create the command with proper shell syntax for environment variables
            cmd_with_env = f"bash -c 'cd {working_dir} && LD_LIBRARY_PATH={library_path} ./{binary_name} --model {model_path} " + " ".join(cmd_args[3:]) + "'"  # Skip llama_server_path, --model, model_path, --port
            
            config_data["models"][proxy_model_name] = {
                "cmd": cmd_with_env
            }

    # Then, add/update with running models (these take precedence for active models)
    for proxy_model_name, model_data in models.items():
        model_path = model_data["model_path"]
        llama_cpp_config = model_data["config"]

        # Build llama.cpp command arguments (using full path to llama-server)
        cmd_args = [llama_server_path, "--model", model_path, "--port", "${PORT}"]

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
            "yarn_attn_factor": 1.0
        }
        
        param_mapping = {
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
            "mirostat_tau": ["--mirostat-tau"],
            "mirostat_eta": ["--mirostat-eta"],
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
            # MoE flags (moe_offload_pattern: 'none' = no flag, 'cpu' = --cpu-moe, number = --n-cpu-moe N)
            "moe_offload_pattern": [],  # Handled specially below
            "moe_offload_custom": []  # Custom MoE pattern, handled specially
        }

        # Emit standard key/value flags
        for key, value in llama_cpp_config.items():
            if key in param_mapping and value is not None:
                # Skip if param_mapping has an empty list (unsupported flag)
                flag_options = param_mapping[key]
                if not flag_options:
                    logger.debug(f"Skipping unsupported flag: {key} (empty mapping)")
                    continue
                
                # Check if flag is supported by the binary (runtime check)
                # Only skip if we have valid help output and flag is not supported
                supported_flags = get_supported_flags(llama_server_path)
                if supported_flags:  # Only validate if we successfully parsed help output
                    flag_supported = is_flag_supported(key, flag_options[0], llama_server_path, param_mapping)
                    if not flag_supported:
                        logger.debug(f"Skipping unsupported flag: {key} for {llama_server_path}")
                        continue
                
                # Skip default values (which cause errors if flag isn't supported)
                if key in default_values:
                    if isinstance(value, (int, float)) and abs(value - default_values[key]) < 1e-6:
                        continue
                
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(flag_options[0])
                elif isinstance(value, str) and value.strip() == "":
                    continue
                else:
                    cmd_args.extend([flag_options[0], str(value)])

        # Special handling: MoE offload flags
        moe_pattern = llama_cpp_config.get("moe_offload_pattern")
        if moe_pattern and moe_pattern != "none":
            if moe_pattern == "cpu":
                cmd_args.append("--cpu-moe")
            elif isinstance(moe_pattern, (int, float)) and moe_pattern > 0:
                cmd_args.extend(["--n-cpu-moe", str(int(moe_pattern))])
            elif isinstance(moe_pattern, str):
                try:
                    n_layers = int(moe_pattern)
                    if n_layers > 0:
                        cmd_args.extend(["--n-cpu-moe", str(n_layers)])
                except ValueError:
                    pass  # Invalid pattern, skip
        
        # Special handling: stop words list (multiple --stop flags)
        if isinstance(llama_cpp_config.get("stop"), list):
            for s in llama_cpp_config.get("stop"):
                if isinstance(s, str) and s.strip():
                    cmd_args.extend(["--stop", s])

        # Free-form custom args appended verbatim at the end
        if isinstance(llama_cpp_config.get("customArgs"), str) and llama_cpp_config.get("customArgs").strip():
            cmd_args.append(llama_cpp_config.get("customArgs").strip())

        # Ensure --host 0.0.0.0 is always present for llama-server
        if "--host" not in cmd_args and "0.0.0.0" not in cmd_args:
            cmd_args.extend(["--host", "0.0.0.0"])

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
        
        # Create the command with proper shell syntax for environment variables
        cmd_with_env = f"bash -c 'cd {working_dir} && LD_LIBRARY_PATH={library_path} ./{binary_name} --model {model_path} " + " ".join(cmd_args[3:]) + "'"  # Skip llama_server_path, --model, model_path, --port
        
        config_data["models"][proxy_model_name] = {
            "cmd": cmd_with_env
        }

    # Add groups configuration to allow multiple models to run simultaneously
    if config_data["models"]:
        config_data["groups"] = {
            "concurrent_models": {
                "swap": False,  # Allow multiple models to run at the same time
                "exclusive": False,  # Don't unload other groups when this group runs
                "members": list(config_data["models"].keys())
            }
        }

    return yaml.dump(config_data, sort_keys=False, indent=2)