import yaml
import os
from typing import Dict, Any
from backend.logging_config import get_logger

logger = get_logger(__name__)

def generate_llama_swap_config(models: Dict[str, Dict[str, Any]], llama_server_path: str, all_models: list = None) -> str:
    """
    Generates the YAML configuration for llama-swap.

    Args:
        models: A dictionary where keys are proxy model names (e.g., "llama-3-8b-q4")
                and values are dictionaries containing 'model_path' and 'config' (llama.cpp params).
        llama_server_path: The absolute path to the llama-server binary.
        all_models: Optional list of all models from database to include in config.

    Returns:
        A string containing the YAML configuration.
    """
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
                except:
                    config = {}
            
            # Build llama.cpp command arguments
            cmd_args = [llama_server_path, "--model", model_path, "--port", "${PORT}"]
            
            # Add other llama.cpp parameters from the config
            param_mapping = {
                "ctx_size": ["-c", "--ctx-size"],
                "n_predict": ["-n", "--n-predict"],
                "threads": ["-t", "--threads"],
                "n_gpu_layers": ["-ngl", "--n-gpu-layers"],
                "batch_size": ["-b", "--batch-size"],
                "ubatch_size": ["-ub", "--ubatch-size"],
                "temp": ["--temp"],
                "top_k": ["--top-k"],
                "top_p": ["--top-p"],
                "repeat_penalty": ["--repeat-penalty"],
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
                "yaml": ["--yaml"]
            }

            for key, value in config.items():
                if key in param_mapping and value is not None:
                    if isinstance(value, bool):
                        if value:
                            cmd_args.append(param_mapping[key][0])
                    elif isinstance(value, str) and value.strip() == "":
                        continue
                    else:
                        cmd_args.extend([param_mapping[key][0], str(value)])

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

        # Add other llama.cpp parameters from the config
        param_mapping = {
            "ctx_size": ["-c", "--ctx-size"],
            "n_predict": ["-n", "--n-predict"],
            "threads": ["-t", "--threads"],
            "n_gpu_layers": ["-ngl", "--n-gpu-layers"],
            "batch_size": ["-b", "--batch-size"],
            "ubatch_size": ["-ub", "--ubatch-size"],
            "temp": ["--temp"],
            "top_k": ["--top-k"],
            "top_p": ["--top-p"],
            "repeat_penalty": ["--repeat-penalty"],
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
            "yaml": ["--yaml"]
        }

        for key, value in llama_cpp_config.items():
            if key in param_mapping and value is not None:
                if isinstance(value, bool):
                    if value:
                        cmd_args.append(param_mapping[key][0])
                elif isinstance(value, str) and value.strip() == "":
                    continue
                else:
                    cmd_args.extend([param_mapping[key][0], str(value)])

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