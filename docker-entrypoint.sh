#!/bin/bash
# Don't use set -e here, as we want to continue even if CUDA setup fails
# (CUDA might not be installed yet)

# Source the CUDA environment setup script if it exists
# This allows the same logic to be used at startup and runtime
if [ -f "/app/setup-cuda-env.sh" ]; then
    source /app/setup-cuda-env.sh
    if [ -n "$CUDA_HOME" ]; then
        echo "CUDA environment configured:"
        echo "  CUDA_HOME=$CUDA_HOME"
        echo "  CUDA_VERSION=$(basename "$CUDA_HOME" | sed 's/cuda-//')"
        echo "  PATH includes: ${CUDA_HOME}/bin"
        echo "  LD_LIBRARY_PATH includes: ${CUDA_HOME}/lib64"
    fi
else
    # Fallback: use inline function if helper script not available
    setup_cuda_env() {
        local data_root="/app/data"
        local cuda_install_dir="${data_root}/cuda"
        
        if [ ! -d "$cuda_install_dir" ]; then
            return 0
        fi
        
        # First, check for the current symlink (preferred method)
        local latest_cuda_path=""
        local current_symlink="${cuda_install_dir}/current"
        
        if [ -e "$current_symlink" ]; then
            # Resolve the symlink to get the actual path
            local resolved_path=$(readlink -f "$current_symlink" 2>/dev/null || realpath "$current_symlink" 2>/dev/null)
            if [ -n "$resolved_path" ] && [ -d "$resolved_path" ] && [ -f "${resolved_path}/bin/nvcc" ]; then
                latest_cuda_path="$resolved_path"
            fi
        fi
        
        # Fallback: Find the latest CUDA installation by scanning directories
        if [ -z "$latest_cuda_path" ]; then
            while IFS= read -r cuda_dir; do
                if [ -d "$cuda_dir" ] && [ -f "${cuda_dir}/bin/nvcc" ]; then
                    latest_cuda_path="$cuda_dir"
                    break
                fi
            done < <(find "$cuda_install_dir" -maxdepth 1 -type d -name "cuda-*" ! -name current 2>/dev/null | sort -V -r)
        fi
        
        if [ -z "$latest_cuda_path" ]; then
            return 0
        fi
        
        export CUDA_HOME="$latest_cuda_path"
        export CUDA_PATH="$latest_cuda_path"
        
        if [ -d "${latest_cuda_path}/bin" ]; then
            if [[ ":$PATH:" != *":${latest_cuda_path}/bin:"* ]]; then
                export PATH="${latest_cuda_path}/bin:$PATH"
            fi
        fi
        
        if [ -d "${latest_cuda_path}/lib64" ]; then
            if [[ ":$LD_LIBRARY_PATH:" != *":${latest_cuda_path}/lib64:"* ]]; then
                export LD_LIBRARY_PATH="${latest_cuda_path}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            fi
        fi
    }
    setup_cuda_env
fi

# Execute the main command
exec "$@"
