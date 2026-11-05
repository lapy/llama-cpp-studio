#!/bin/bash
set -euo pipefail

cd /workspace/active
export HOME=/workspace
export VITE_DEV_BACKEND_URL=http://127.0.0.1:8081

exec /usr/local/bin/npm run dev -- --host 0.0.0.0 --port 5173 --strictPort
