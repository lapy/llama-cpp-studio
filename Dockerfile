################################################################################
# llama.cpp Studio - Multi-stage Docker build
################################################################################
ARG BASE_IMAGE=ubuntu:22.04

################################################################################
# Stage 1: Frontend Builder
# Purpose: Compile Vue.js frontend with Vite
################################################################################
FROM node:20-slim AS frontend-builder

WORKDIR /build

# Copy package files and install dependencies (including devDependencies for build tools)
COPY package*.json ./
RUN if [ -f package-lock.json ] || [ -f npm-shrinkwrap.json ]; then \
        npm ci; \
    else \
        npm install; \
    fi

# Copy frontend source (vite.config.js expects files at /build root, not /build/frontend)
COPY frontend/ ./
RUN npm run build

################################################################################
# Stage 2: Python Builder
# Purpose: Create isolated venv with all Python dependencies
################################################################################
FROM ${BASE_IMAGE} AS python-builder

ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies required for python wheels and runtime compilation (llama.cpp)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    libopenblas-dev \
    git \
    curl \
    wget \
    ca-certificates \
    pciutils \
    lsb-release \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Rust (required for tokenizers)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | \
    sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH="/root/.cargo/bin:${PATH}"

# Create venv and install Python packages
ENV VENV_PATH=/opt/venv
RUN python3 -m venv ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

################################################################################
# Stage 3: Runtime
# Purpose: Minimal production image with compiled artifacts
################################################################################
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=all \
    VENV_PATH=/opt/venv \
    PYTHONPATH=/app

# Install runtime dependencies (retain build toolchain for llama.cpp builds and GPU detection)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    pkg-config \
    ninja-build \
    curl \
    wget \
    ca-certificates \
    # Core libs for Python packages
    libssl3 \
    libffi8 \
    libcurl4 \
    libopenblas0 \
    # GPU acceleration support
    libvulkan1 \
    vulkan-tools \
    mesa-vulkan-drivers \
    ocl-icd-libopencl1 \
    libnuma1 \
    pciutils \
    usbutils \
    lshw \
    # Optional: ROCm (fails gracefully if unavailable)
    && (apt-get install -y --no-install-recommends rocminfo rocm-smi || echo "ROCm unavailable") \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install CUDA runtime libraries (for NVIDIA GPU support)
# These are needed when llama-server binaries are built with CUDA support
# The NVIDIA Container Toolkit should provide these, but we install as fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean || true

# Try to install CUDA runtime from NVIDIA repository (fails gracefully if not available)
# This provides libcudart.so.12 and other CUDA runtime libraries
RUN ( \
    wget -qO - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | gpg --dearmor -o /usr/share/keyrings/cuda.gpg 2>/dev/null || true \
    && echo "deb [signed-by=/usr/share/keyrings/cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list 2>/dev/null || true \
    && apt-get update \
    && apt-get install -y --no-install-recommends cuda-runtime-12-8 2>/dev/null || \
    (echo "CUDA runtime installation skipped (may be provided by NVIDIA Container Toolkit)" && true) \
    ) && rm -rf /var/lib/apt/lists/* && apt-get clean || true

# Install llama-swap binary
ARG LLAMA_SWAP_VERSION=168
RUN wget -q https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_amd64.tar.gz -O /tmp/llama-swap.tar.gz && \
    tar -xzf /tmp/llama-swap.tar.gz -C /tmp && \
    mv /tmp/llama-swap /usr/local/bin/llama-swap && \
    chmod +x /usr/local/bin/llama-swap && \
    rm -rf /tmp/* && \
    llama-swap --version

# Copy Python venv from builder
COPY --from=python-builder ${VENV_PATH} ${VENV_PATH}
ENV PATH="${VENV_PATH}/bin:${PATH}"

# Set up application directory
WORKDIR /app

# Copy application code (excluding data via .dockerignore)
COPY backend/ ./backend/
COPY migrate_db.py ./
COPY --from=frontend-builder /build/dist ./frontend/dist
COPY frontend/public ./frontend/public

# Create python symlink for compatibility
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Create non-root user and data directory structure
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /app/data/{models,configs,logs,llama-cpp,temp} && \
    chown -R appuser:appuser /app

# Expose API port
EXPOSE 8080

# Declare volume for persistent data
VOLUME ["/app/data"]

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

# Start application
CMD ["python", "backend/main.py"]
