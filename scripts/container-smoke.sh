#!/usr/bin/env bash
# Smoke the image that is about to be published. The caller passes the local
# image name produced by the build that will be pushed; this script does not
# rebuild it.
set -euo pipefail

IMAGE="${1:-${IMAGE:-}}"
if [ -z "$IMAGE" ]; then
  echo "usage: container-smoke.sh IMAGE" >&2
  exit 2
fi

NAME="studio-smoke-$$"
VOLUME="studio-smoke-$$"
LOG_DUMPED=0

dump_logs() {
  if [ "$LOG_DUMPED" = 1 ]; then
    return
  fi
  LOG_DUMPED=1
  docker logs "$NAME" >&2 || true
}

cleanup() {
  local status=$?
  if [ "$status" -ne 0 ]; then
    dump_logs
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker volume rm -f "$VOLUME" >/dev/null 2>&1 || true
  exit "$status"
}
trap cleanup EXIT

docker volume create "$VOLUME" >/dev/null
docker run -d --name "$NAME" \
  -e STUDIO_ACCESS_MODE=local \
  -e CUDA_VISIBLE_DEVICES= \
  -v "$VOLUME:/app/data" \
  -p 127.0.0.1::8080 \
  "$IMAGE" >/dev/null

PORT="$(docker port "$NAME" 8080/tcp | awk -F: 'NR==1 { print $NF }')"
if [ -z "$PORT" ]; then
  echo "container did not publish port 8080" >&2
  exit 1
fi

python3 - "$PORT" <<'PY'
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

set +e
docker stop --time 20 "$NAME"
stop_status=$?
set -e
if [ "$stop_status" -ne 0 ]; then
  echo "docker stop exited $stop_status" >&2
  exit 1
fi

read -r state exit_code < <(docker inspect --format '{{.State.Status}} {{.State.ExitCode}}' "$NAME")
if [ "$state" != "exited" ] || [ "$exit_code" != "0" ]; then
  echo "shutdown state is ${state} exit ${exit_code}" >&2
  exit 1
fi
echo "shutdown ok"
