#!/usr/bin/env bash
# Permission decisions only. This does not start a container and is not
# container execution evidence.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$ROOT/container-fixture.sh"

[ "$(container_mismatch_uid 1000 1001)" = "1000" ]
[ "$(container_mismatch_uid 1000 1000)" = "65534" ]
[ "$(container_mismatch_uid 65534 65534)" = "65533" ]
[ "$(fixture_dir_mode)" = "0750" ]
[ "$(fixture_file_mode)" = "0640" ]
echo "container fixture permission logic ok"
