#!/bin/bash
# Helper script to setup CUDA environment variables from data directory
# Can be sourced: source setup-cuda-env.sh
# Or executed: ./setup-cuda-env.sh (will print export commands)

# Function to setup CUDA environment variables from data directory
setup_cuda_env() {
    local data_root="/app/data"
    local cuda_install_dir="${data_root}/cuda"
    
    # Check if CUDA install directory exists
    if [ ! -d "$cuda_install_dir" ]; then
        return 1
    fi
    
    # First, check for the current symlink (preferred method)
    local latest_cuda_path=""
    local latest_version=""
    local current_symlink="${cuda_install_dir}/current"
    
    if [ -e "$current_symlink" ]; then
        # Resolve the symlink to get the actual path
        local resolved_path=$(readlink -f "$current_symlink" 2>/dev/null || realpath "$current_symlink" 2>/dev/null)
        if [ -n "$resolved_path" ] && [ -d "$resolved_path" ] && [ -f "${resolved_path}/bin/nvcc" ]; then
            latest_cuda_path="$resolved_path"
            latest_version=$(basename "$resolved_path" | sed 's/cuda-//')
        fi
    fi
    
    # Fallback: Find the latest CUDA installation by scanning directories
    if [ -z "$latest_cuda_path" ]; then
        # Check for installed CUDA versions (sorted by version, newest first)
        while IFS= read -r cuda_dir; do
            if [ -d "$cuda_dir" ] && [ -f "${cuda_dir}/bin/nvcc" ]; then
                latest_cuda_path="$cuda_dir"
                latest_version=$(basename "$cuda_dir" | sed 's/cuda-//')
                break
            fi
        done < <(find "$cuda_install_dir" -maxdepth 1 -type d -name "cuda-*" ! -name current 2>/dev/null | sort -V -r)
    fi
    
    # If no CUDA installation found, return
    if [ -z "$latest_cuda_path" ]; then
        return 1
    fi
    
    # Set CUDA environment variables
    export CUDA_HOME="$latest_cuda_path"
    export CUDA_PATH="$latest_cuda_path"
    
    # Add CUDA bin to PATH if it exists and is not already in PATH
    if [ -d "${latest_cuda_path}/bin" ]; then
        if [[ ":$PATH:" != *":${latest_cuda_path}/bin:"* ]]; then
            export PATH="${latest_cuda_path}/bin:$PATH"
        fi
    fi
    
    # Add CUDA lib64 to LD_LIBRARY_PATH if it exists and is not already in LD_LIBRARY_PATH
    if [ -d "${latest_cuda_path}/lib64" ]; then
        if [[ ":$LD_LIBRARY_PATH:" != *":${latest_cuda_path}/lib64:"* ]]; then
            export LD_LIBRARY_PATH="${latest_cuda_path}/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    fi

    # NCCL is installed by this project's CUDA installer into ${CUDA_HOME} when available.
    # Export NCCL + compile-time include/library paths so builds (e.g. lmdeploy from source)
    # can find nccl.h / libnccl during compilation.
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
    
    # Check for TensorRT (optional)
    if [ -f "${latest_cuda_path}/lib64/libnvinfer.so" ] || \
       [ -f "${latest_cuda_path}/lib64/libnvinfer.so.10" ] || \
       [ -f "${latest_cuda_path}/lib64/libnvinfer.so.8" ]; then
        export TENSORRT_PATH="$latest_cuda_path"
        export TENSORRT_ROOT="$latest_cuda_path"
    fi
    
    return 0
}

# If sourced, setup environment directly
if [ "${BASH_SOURCE[0]}" != "${0}" ]; then
    # Script is being sourced
    setup_cuda_env
# If executed, print export commands that can be eval'd
else
    # Script is being executed
    if setup_cuda_env; then
        echo "export CUDA_HOME=\"$CUDA_HOME\""
        echo "export CUDA_PATH=\"$CUDA_PATH\""
        echo "export PATH=\"$PATH\""
        echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
        if [ -n "${NCCL_ROOT:-}" ]; then
            echo "export NCCL_ROOT=\"$NCCL_ROOT\""
            echo "export NCCL_HOME=\"$NCCL_HOME\""
        fi
        if [ -n "${CPATH:-}" ]; then
            echo "export CPATH=\"$CPATH\""
        fi
        if [ -n "${LIBRARY_PATH:-}" ]; then
            echo "export LIBRARY_PATH=\"$LIBRARY_PATH\""
        fi
        if [ -n "${CMAKE_PREFIX_PATH:-}" ]; then
            echo "export CMAKE_PREFIX_PATH=\"$CMAKE_PREFIX_PATH\""
        fi
        if [ -n "${CMAKE_LIBRARY_PATH:-}" ]; then
            echo "export CMAKE_LIBRARY_PATH=\"$CMAKE_LIBRARY_PATH\""
        fi
        if [ -n "$TENSORRT_PATH" ]; then
            echo "export TENSORRT_PATH=\"$TENSORRT_PATH\""
            echo "export TENSORRT_ROOT=\"$TENSORRT_ROOT\""
        fi
    else
        exit 1
    fi
fi
