#!/bin/bash
# Don't use set -e here, as we want to continue even if CUDA setup fails
# (CUDA might not be installed yet)

# Check and warn about /app/data permissions
# Volume mounts may have incorrect ownership/permissions
if [ -d "/app/data" ]; then
    # Check if we can write to the data directory
    if [ ! -w "/app/data" ]; then
        echo "WARNING: /app/data directory is not writable by current user ($(id -u))"
        echo "This will cause configuration and model write errors."
        echo "To fix, run on the host: sudo chown -R $(id -u):$(id -g) <volume-path>"
    fi
fi

# Source the CUDA environment setup script if it exists
# This allows the same logic to be used at startup and runtime
if [ -f "/app/setup-cuda-env.sh" ]; then
    source /app/setup-cuda-env.sh
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

        # NCCL (optional, installed into ${CUDA_HOME} by this project's CUDA installer when available)
        if [ -d "${latest_cuda_path}/include" ]; then
            export NCCL_ROOT="$latest_cuda_path"
            export NCCL_HOME="$latest_cuda_path"
            export NCCL_INCLUDE_DIR="${latest_cuda_path}/include"

            if [[ ":${CPATH:-}:" != *":${latest_cuda_path}/include:"* ]]; then
                export CPATH="${latest_cuda_path}/include${CPATH:+:$CPATH}"
            fi
        fi

        if [ -d "${latest_cuda_path}/lib64" ]; then
            export NCCL_LIB_DIR="${latest_cuda_path}/lib64"

            if [[ ":${LIBRARY_PATH:-}:" != *":${latest_cuda_path}/lib64:"* ]]; then
                export LIBRARY_PATH="${latest_cuda_path}/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
            fi

            if [[ ":${CMAKE_LIBRARY_PATH:-}:" != *":${latest_cuda_path}/lib64:"* ]]; then
                export CMAKE_LIBRARY_PATH="${latest_cuda_path}/lib64${CMAKE_LIBRARY_PATH:+:$CMAKE_LIBRARY_PATH}"
            fi
        fi

        if [[ ":${CMAKE_PREFIX_PATH:-}:" != *":${latest_cuda_path}:"* ]]; then
            export CMAKE_PREFIX_PATH="${latest_cuda_path}${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
        fi
    }
    setup_cuda_env
fi

if [ -n "$CUDA_HOME" ]; then
    echo "CUDA environment configured:"
    echo "  CUDA_HOME=$CUDA_HOME"
    echo "  CUDA_VERSION=$(basename "$CUDA_HOME" | sed 's/cuda-//')"
    echo "  PATH includes: ${CUDA_HOME}/bin"
    echo "  LD_LIBRARY_PATH includes: ${CUDA_HOME}/lib64"
fi

# Execute the main command
exec "$@"
