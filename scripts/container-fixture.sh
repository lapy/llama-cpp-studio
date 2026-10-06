#!/usr/bin/env bash
# Helpers for the image that is about to be published.
# scripts/container-fixture-test.sh checks the uid and mode decisions only.
# That check does not start a container and is not container execution evidence.

container_mismatch_uid() {
  local host_uid="$1"
  local container_uid="$2"
  if [ "$host_uid" != "$container_uid" ]; then
    printf '%s\n' "$host_uid"
    return 0
  fi
  if [ "$container_uid" != "65534" ]; then
    printf '%s\n' "65534"
    return 0
  fi
  printf '%s\n' "65533"
}

fixture_dir_mode() {
  printf '%s\n' "0750"
}

fixture_file_mode() {
  printf '%s\n' "0640"
}

docker_bin() {
  printf '%s\n' "${STUDIO_DOCKER:-docker}"
}

studio_evidence_dir() {
  if [ -n "${STUDIO_EVIDENCE_DIR:-}" ]; then
    printf '%s\n' "$STUDIO_EVIDENCE_DIR"
    return 0
  fi
  local root
  root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  printf '%s\n' "$root/artifacts/container-validation"
}

prepare_fixture_for_image() {
  local image="$1"
  local data="$2"
  local evidence="$3"
  local required_file="${4:-}"
  local docker host_uid host_gid container_uid container_gid mismatch
  docker="$(docker_bin)"
  host_uid="$(id -u)"
  host_gid="$(id -g)"
  container_uid="$("$docker" run --rm --entrypoint id "$image" -u)"
  container_gid="$("$docker" run --rm --entrypoint id "$image" -g)"
  container_uid="${container_uid//$'\r'/}"
  container_gid="${container_gid//$'\r'/}"
  mismatch="$(container_mismatch_uid "$host_uid" "$container_uid")"

  if [ "$mismatch" != "$host_uid" ]; then
    "$docker" run --rm --user 0 --entrypoint chown \
      -v "$data:/data" \
      "$image" \
      -R "${mismatch}:${mismatch}" /data
  fi
  "$docker" run --rm --user 0 --entrypoint chmod \
    -v "$data:/data" \
    "$image" \
    -R go-rwx /data

  if "$docker" run --rm --entrypoint sh \
    -v "$data:/data" \
    "$image" \
    -c 'touch /data/.uid-mismatch-probe'; then
    echo "container user $container_uid could write the fixture before ownership alignment" >&2
    return 1
  fi

  "$docker" run --rm --user 0 --entrypoint chown \
    -v "$data:/data" \
    "$image" \
    -R "${container_uid}:${container_gid}" /data
  "$docker" run --rm --user 0 --entrypoint python \
    -v "$data:/data" \
    "$image" \
    -c 'import os, sys
root = "/data"
dir_mode = 0o750
file_mode = 0o640
for dirpath, dirnames, filenames in os.walk(root):
    os.chmod(dirpath, dir_mode)
    for name in filenames:
        os.chmod(os.path.join(dirpath, name), file_mode)
'

  local read_check="touch /data/.uid-write-probe && rm -f /data/.uid-write-probe"
  if [ -n "$required_file" ]; then
    read_check="test -r /data/${required_file} && ${read_check}"
  fi
  if ! "$docker" run --rm --entrypoint sh \
    -v "$data:/data" \
    "$image" \
    -c "$read_check"; then
    echo "container user $container_uid could not use the fixture after ownership alignment" >&2
    return 1
  fi

  mkdir -p "$evidence"
  local label="fresh"
  if [ -n "$required_file" ]; then
    label="seeded"
  fi
  cat > "$evidence/uid-mismatch-${label}.txt" <<EOF
host_uid=${host_uid}
host_gid=${host_gid}
container_uid=${container_uid}
container_gid=${container_gid}
mismatch_uid=${mismatch}
dir_mode=$(fixture_dir_mode)
file_mode=$(fixture_file_mode)
required_file=${required_file:-<fresh>}
EOF
}

restore_fixture_to_host() {
  local image="$1"
  local data="$2"
  local docker
  docker="$(docker_bin)"
  if [ ! -d "$data" ]; then
    return 0
  fi
  "$docker" run --rm --user 0 --entrypoint chown \
    -v "$data:/data" \
    "$image" \
    -R "$(id -u):$(id -g)" /data \
    || "$docker" run --rm --user 0 --entrypoint rm \
      -v "$data:/data" \
      "$image" \
      -rf /data
}

record_image_identity() {
  local image="$1"
  local evidence="$2"
  local docker
  docker="$(docker_bin)"
  mkdir -p "$evidence"
  "$docker" image inspect --format 'id={{.Id}}
created={{.Created}}
repo_digest={{if .RepoDigests}}{{index .RepoDigests 0}}{{else}}<none>{{end}}' \
    "$image" > "$evidence/image-identity.txt"
}
