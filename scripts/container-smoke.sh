#!/usr/bin/env bash
# Smoke the image that is about to be published. The caller passes the local
# image name produced by the build that will be pushed; this script does not
# rebuild it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/container-fixture.sh"

IMAGE="${1:-${IMAGE:-}}"
if [ -z "$IMAGE" ]; then
  echo "usage: container-smoke.sh IMAGE" >&2
  exit 2
fi

NAME="studio-smoke-$$"
DATA="$(mktemp -d)"
EVIDENCE="$(studio_evidence_dir)"
mkdir -p "$EVIDENCE"
STARTED=0
LOG_DUMPED=0

dump_logs() {
  if [ "$LOG_DUMPED" = 1 ]; then
    return
  fi
  LOG_DUMPED=1
  docker_cmd="$(docker_bin)"
  "$docker_cmd" logs "$NAME" > "$EVIDENCE/smoke-container.log" 2>&1 || true
  cat "$EVIDENCE/smoke-container.log" >&2 || true
}

cleanup() {
  local status=$?
  local docker_cmd
  docker_cmd="$(docker_bin)"
  if [ "$status" -ne 0 ]; then
    dump_logs
  fi
  if [ "$STARTED" = 1 ]; then
    "$docker_cmd" rm -f "$NAME" >/dev/null 2>&1 || true
  fi
  restore_fixture_to_host "$IMAGE" "$DATA" || true
  rm -rf "$DATA"
  if [ -d "$DATA" ]; then
    echo "fixture data remained at $DATA" | tee "$EVIDENCE/smoke-cleanup.txt" >&2
    status=1
  elif "$docker_cmd" ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "container $NAME remained" | tee "$EVIDENCE/smoke-cleanup.txt" >&2
    status=1
  else
    echo "removed container and fixture data" > "$EVIDENCE/smoke-cleanup.txt"
  fi
  exit "$status"
}
trap cleanup EXIT

record_image_identity "$IMAGE" "$EVIDENCE"
{
  echo "host_node=$(node -v 2>/dev/null || echo unavailable)"
  echo "container_python=$("$(docker_bin)" run --rm --entrypoint python "$IMAGE" -V)"
  echo "container_identity=$("$(docker_bin)" run --rm --entrypoint id "$IMAGE")"
} > "$EVIDENCE/runtime-versions.txt"

prepare_fixture_for_image "$IMAGE" "$DATA" "$EVIDENCE"

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

python3 - "$PORT" <<'PY' | tee "$EVIDENCE/smoke-probe.txt"
import json
import re
import sys
import time
import urllib.error
import urllib.request

port = sys.argv[1]
base = f"http://127.0.0.1:{port}"
deadline = time.time() + 90
live = ready = None
last_error = ""


def fetch(path):
    with urllib.request.urlopen(base + path, timeout=5) as response:
        body = response.read()
        return response.status, body


while time.time() < deadline:
    try:
        status, body = fetch("/api/live")
        live = json.loads(body.decode())
        status, body = fetch("/api/ready")
        ready = json.loads(body.decode())
        if (
            status == 200
            and live.get("live") is True
            and ready.get("ready") is True
            and ready.get("proxy_required") is False
        ):
            break
    except Exception as exc:
        last_error = str(exc)
        live = ready = None
    time.sleep(2)
else:
    raise SystemExit(
        f"startup deadline exceeded: live={live!r} ready={ready!r} error={last_error}"
    )

status, body = fetch("/")
document = body.decode("utf-8", "replace")
if status != 200 or "llama.cpp Studio" not in document:
    raise SystemExit("studio document was not served")
refs = re.findall(r'(?:src|href)="([^"]+\.(?:js|css))"', document)
if not refs:
    raise SystemExit("studio document has no script or stylesheet")
ref = refs[0]
asset_path = ref if ref.startswith("/") else "/" + ref
status, asset = fetch(asset_path)
if status != 200 or not asset:
    raise SystemExit(f"asset {ref} was not served")
print(f"startup ok; served {ref} ({len(asset)} bytes)")
PY

"$(docker_bin)" exec "$NAME" python -c 'open("/app/data/.running-write-probe","w").write("ok")'

set +e
"$(docker_bin)" stop --time 20 "$NAME"
stop_status=$?
set -e
if [ "$stop_status" -ne 0 ]; then
  echo "docker stop exited $stop_status" >&2
  exit 1
fi

read -r state exit_code < <("$(docker_bin)" inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$NAME")
if [ "$state" != "exited" ] || [ "$exit_code" != "0" ]; then
  echo "shutdown state is ${state} exit ${exit_code}" >&2
  exit 1
fi
echo "shutdown ok" | tee -a "$EVIDENCE/smoke-probe.txt"
