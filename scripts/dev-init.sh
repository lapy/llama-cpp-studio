#!/bin/bash
# One-time / idempotent setup for the llama-cpp-studio dev container.
# Runs at every container start (see docker-compose.dev.yml); cheap when done.
set -uo pipefail

echo "== dev-init: python dev dependencies =="
if pip install -q -r /app/requirements-dev.lock; then
    echo "python deps: OK ($(python --version), $(pip --version | cut -d' ' -f1-2))"
else
    echo "python deps: FAILED" >&2
fi

echo "== dev-init: node modules =="
if [ -d /app/node_modules/vite ]; then
    echo "node modules: present"
else
    echo "node modules: running npm ci (first start only; stored in named volume)"
    (cd /app && npm ci --no-audit --no-fund) || echo "npm ci: FAILED" >&2
fi

echo "== dev-init: git hooks (best-effort) =="
(cd /app && node scripts/install-git-hooks.mjs >/dev/null 2>&1) || true

echo "== dev-init: done =="
