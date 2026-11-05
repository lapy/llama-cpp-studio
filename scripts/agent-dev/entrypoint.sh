#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  init)
    if [[ -z "${DEV_UID:-}" || -z "${DEV_GID:-}" ]]; then
      echo "DEV_UID and DEV_GID are required" >&2
      exit 2
    fi
    mkdir -p /app/node_modules /app/data/home /app/data/config /app/data/logs \
      /app/data/temp /app/data/models /app/data/ccache /app/data/hf-cache
    # Only isolated dev volumes are initialized; production is never mounted.
    chown "${DEV_UID}:${DEV_GID}" /app/node_modules /app/data
    # Do not recurse into model/cache trees: they can contain many gigabytes.
    # New files are written by the container's own UID, so top-level ownership
    # is sufficient after the first init.
    chown "${DEV_UID}:${DEV_GID}" \
      /app/data/home /app/data/config /app/data/logs /app/data/temp \
      /app/data/models /app/data/ccache /app/data/hf-cache

    # Keep the image venv aligned when an agent changes dev lockfiles. This is
    # a host-side development concern, so only the isolated /opt/venv is touched.
    if [[ -f /app/requirements-dev.lock ]]; then
      fingerprint=$(sha256sum /app/requirements-dev.lock | cut -d' ' -f1)
      if [[ ! -f /opt/venv/.agent-requirements-sha ]] \
        || [[ $(cat /opt/venv/.agent-requirements-sha) != "$fingerprint" ]]; then
        /opt/venv/bin/python -m pip install --no-cache-dir -r /app/requirements-dev.lock
        printf '%s\n' "$fingerprint" > /opt/venv/.agent-requirements-sha
      fi
    fi

    if [[ -f /app/package-lock.json ]]; then
      fingerprint=$(sha256sum /app/package-lock.json | cut -d' ' -f1)
      if [[ ! -f /app/node_modules/.agent-lock-sha ]] \
        || [[ $(cat /app/node_modules/.agent-lock-sha) != "$fingerprint" ]]; then
        npm ci --no-audit --no-fund --ignore-scripts
        # esbuild's install script selects/validates its platform binary.
        npm rebuild esbuild
        printf '%s\n' "$fingerprint" > /app/node_modules/.agent-lock-sha
        chown -R "${DEV_UID}:${DEV_GID}" /app/node_modules
      fi
    fi
    ;;
  backend)
    exec /usr/local/bin/docker-entrypoint.sh /opt/venv/bin/python -m uvicorn backend.main:app \
      --host 0.0.0.0 --port 8081 --reload --reload-dir /app/backend \
      --reload-include '*.py' --reload-include '*.yaml'
    ;;
  frontend)
    exec npm run dev -- --host 0.0.0.0 --port 5173 --strictPort
    ;;
  *)
    echo 'Expected init, backend, or frontend' >&2
    exit 2
    ;;
esac
