# llama.cpp Docker Manager
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=all

# Install system dependencies including Vulkan and ROCm support
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    cmake \
    make \
    gcc \
    g++ \
    pkg-config \
    libssl-dev \
    libffi-dev \
    libcurl4-openssl-dev \
    unzip \
    libopenblas-dev \
    libvulkan-dev \
    vulkan-tools \
    vulkan-validationlayers-dev \
    mesa-vulkan-drivers \
    mesa-common-dev \
    ocl-icd-opencl-dev \
    opencl-headers \
    pciutils \
    libnuma-dev \
    wget \
    ca-certificates \
    gnupg2 \
    glslang-tools \
    && rm -rf /var/lib/apt/lists/*

# Try to install ROCm packages (optional, may fail if not available)
RUN apt-get update && (apt-get install -y rocminfo rocm-dev rocm-device-libs || echo "ROCm packages not available, continuing without them") && rm -rf /var/lib/apt/lists/*

# Install llama-swap proxy for multi-model serving
# Pin llama-swap version; download and install
ARG LLAMA_SWAP_VERSION=168
RUN wget -q https://github.com/mostlygeek/llama-swap/releases/download/v${LLAMA_SWAP_VERSION}/llama-swap_${LLAMA_SWAP_VERSION}_linux_amd64.tar.gz -O /tmp/llama-swap.tar.gz && \
    tar -xzf /tmp/llama-swap.tar.gz -C /tmp && \
    mv /tmp/llama-swap /usr/local/bin/llama-swap && \
    chmod +x /usr/local/bin/llama-swap && \
    rm /tmp/llama-swap.tar.gz && \
    llama-swap --version || (echo "Failed to install llama-swap" && exit 1)

# Install latest Node.js (required for modern ES modules)
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs

# Set working directory
WORKDIR /app

# Set Python path
ENV PYTHONPATH=/app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Control whether to build frontend (can be disabled to unblock backend build)
ARG BUILD_FRONTEND=true

# Copy frontend files first
COPY frontend/ ./frontend/
COPY package.json ./frontend/

# Build frontend conditionally
WORKDIR /app/frontend
RUN if [ "$BUILD_FRONTEND" = "true" ]; then npm install && npm run build; else echo "Skipping frontend build"; fi
WORKDIR /app

# Copy remaining application code
COPY backend/ ./backend/
COPY *.py ./
COPY *.md ./

# Create python symlink for NVIDIA base image compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Create non-root user first
RUN useradd -ms /bin/bash appuser

# Create data directory structure with proper ownership
RUN mkdir -p /app/data/{models,configs,logs,llama-cpp} && \
    chown -R appuser:appuser /app

# Expose port
EXPOSE 8080

# Set volume mount point
VOLUME ["/app/data"]

# Start the application
USER appuser
CMD ["python", "backend/main.py"]
