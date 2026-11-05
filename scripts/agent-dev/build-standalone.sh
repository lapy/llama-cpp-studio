#!/usr/bin/env bash
# Build the self-contained standalone image in two stages:
#   1. the runtime base from the repository Dockerfile
#   2. the standalone layer (source snapshot, SSH, supervisor, dev deps)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_IMAGE="${STUDIO_STANDALONE_BASE:-llama-cpp-studio:standalone-base}"
IMAGE="${STUDIO_STANDALONE_IMAGE:-llama-cpp-studio:standalone}"
LICENSE="${VITE_PRIMEUI_LICENSE:-dev-placeholder-not-for-distribution}"

docker build \
  -f "$ROOT/Dockerfile" \
  --target runtime \
  --build-arg "VITE_PRIMEUI_LICENSE=$LICENSE" \
  -t "$BASE_IMAGE" \
  "$ROOT"

docker build \
  -f "$ROOT/scripts/agent-dev/standalone.Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  -t "$IMAGE" \
  "$ROOT"

echo "Built $IMAGE (base $BASE_IMAGE)"
