#!/usr/bin/env bash
# Measure navigation of the image that was just smoked. This does not rebuild
# the image and does not replace the compressed asset-size budget.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/container-fixture.sh"

IMAGE="${1:-${IMAGE:-}}"
if [ -z "$IMAGE" ]; then
  echo "usage: measure-production-navigation.sh IMAGE" >&2
  exit 2
fi

NAME="studio-nav-$$"
DATA="$(mktemp -d)"
EVIDENCE="$(studio_evidence_dir)"
mkdir -p "$EVIDENCE" "$DATA/config"
STARTED=0

cleanup() {
  local status=$?
  local docker_cmd
  docker_cmd="$(docker_bin)"
  if [ "$STARTED" = 1 ]; then
    if [ "$status" -ne 0 ]; then
      "$docker_cmd" logs "$NAME" > "$EVIDENCE/navigation-container.log" 2>&1 || true
      cat "$EVIDENCE/navigation-container.log" >&2 || true
    fi
    "$docker_cmd" rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  restore_fixture_to_host "$IMAGE" "$DATA" || true
  rm -rf "$DATA"
  if [ -d "$DATA" ]; then
    echo "fixture data remained at $DATA" | tee "$EVIDENCE/navigation-cleanup.txt" >&2
    status=1
  elif [ "$STARTED" = 1 ] && "$docker_cmd" ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container $NAME remained" | tee "$EVIDENCE/navigation-cleanup.txt" >&2
    status=1
  else
    echo "removed container and fixture data" > "$EVIDENCE/navigation-cleanup.txt"
  fi
  exit "$status"
}
trap cleanup EXIT

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

prepare_fixture_for_image "$IMAGE" "$DATA" "$EVIDENCE" "config/models.yaml"
if [ ! -f "$EVIDENCE/runtime-versions.txt" ]; then
  {
    echo "host_node=$(node -v 2>/dev/null || echo unavailable)"
    echo "container_python=$("$(docker_bin)" run --rm --entrypoint python "$IMAGE" -V)"
    echo "container_identity=$("$(docker_bin)" run --rm --entrypoint id "$IMAGE")"
  } > "$EVIDENCE/runtime-versions.txt"
fi
echo "navigation_host_node=$(node -v 2>/dev/null || echo unavailable)" >> "$EVIDENCE/runtime-versions.txt"

"$(docker_bin)" run -d --name "$NAME" \
  -e STUDIO_ACCESS_MODE=local \
  -e CUDA_VISIBLE_DEVICES= \
  -v "$DATA:/app/data" \
  -p 127.0.0.1::8080 \
  "$IMAGE" >/dev/null
STARTED=1

PORT="$("$(docker_bin)" port "$NAME" 8080/tcp | awk -F: 'NR==1 { print $NF }')"
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

"$(docker_bin)" exec "$NAME" python -c 'open("/app/data/config/models.yaml").read(); open("/app/data/.running-write-probe","w").write("ok")'

STUDIO_NAVIGATION_URL="http://127.0.0.1:${PORT}" \
  node "$ROOT/frontend/e2e/production-navigation.mjs" | tee "$EVIDENCE/navigation-samples.txt"
echo "production navigation ok" | tee -a "$EVIDENCE/navigation-samples.txt"
