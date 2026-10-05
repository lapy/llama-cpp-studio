#!/usr/bin/env bash
# Measure navigation of the image that was just smoked. This does not rebuild
# the image and does not replace the compressed asset-size budget.
set -euo pipefail

IMAGE="${1:-${IMAGE:-}}"
if [ -z "$IMAGE" ]; then
  echo "usage: measure-production-navigation.sh IMAGE" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAME="studio-nav-$$"
DATA="$(mktemp -d)"
STARTED=0

cleanup() {
  local status=$?
  if [ "$STARTED" = 1 ]; then
    if [ "$status" -ne 0 ]; then
      docker logs "$NAME" >&2 || true
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  rm -rf "$DATA"
  exit "$status"
}
trap cleanup EXIT

mkdir -p "$DATA/config"
cat > "$DATA/config/models.yaml" <<'EOF'
schema_version: 3
models:
  - id: org/demo
    huggingface_id: org/demo
    display_name: Demo
    base_model_name: Demo
    format: safetensors
    config:
      engine: llama_cpp
      engines:
        llama_cpp:
          ctx_size: 4096
EOF
# The image runs as appuser. A host temp directory is mode 0700 and may belong
# to a different uid, so hand the tree to that user and keep it removable.
chmod -R a+rwX "$DATA"
docker run --rm --user 0 --entrypoint chown \
  -v "$DATA:/data" \
  "$IMAGE" \
  -R appuser:appuser /data
chmod -R a+rwX "$DATA"

docker run -d --name "$NAME" \
  -e STUDIO_ACCESS_MODE=local \
  -e CUDA_VISIBLE_DEVICES= \
  -v "$DATA:/app/data" \
  -p 127.0.0.1::8080 \
  "$IMAGE" >/dev/null
STARTED=1

PORT="$(docker port "$NAME" 8080/tcp | awk -F: 'NR==1 { print $NF }')"
if [ -z "$PORT" ]; then
  echo "container did not publish port 8080" >&2
  exit 1
fi

python3 - "$PORT" <<'PY'
import json
import sys
import time
import urllib.request

port = sys.argv[1]
base = f"http://127.0.0.1:{port}"
deadline = time.time() + 90
while time.time() < deadline:
    try:
        with urllib.request.urlopen(base + "/api/live", timeout=5) as response:
            live = json.loads(response.read().decode())
        if live.get("live") is True:
            raise SystemExit(0)
    except Exception:
        time.sleep(2)
raise SystemExit("navigation server did not become live")
PY

STUDIO_NAVIGATION_URL="http://127.0.0.1:${PORT}" \
  node "$ROOT/frontend/e2e/production-navigation.mjs"
echo "production navigation ok"
