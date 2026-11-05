#!/bin/bash
set -euo pipefail

cd /workspace/active
export PYTHONPATH=/workspace/active
export HOME=/workspace
export STUDIO_DATA_DIR=/data
export STUDIO_ACCESS_MODE=local
export STUDIO_BIND_HOST=127.0.0.1
export STUDIO_PROXY_LISTEN=127.0.0.1
export HF_HOME=/data/hf-cache
export HUGGINGFACE_HUB_CACHE=/data/hf-cache/hub
export WATCHFILES_FORCE_POLLING=true

exec /opt/venv/bin/python -m uvicorn backend.main:app \
  --host 127.0.0.1 --port 8081 --reload \
  --reload-dir backend \
  --reload-include '*.py' --reload-include '*.yaml'
