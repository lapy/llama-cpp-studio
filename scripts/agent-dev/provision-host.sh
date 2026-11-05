#!/usr/bin/env bash
# Provision a dedicated Unix account, SSH key, checkout, and control state for
# the isolated Studio development stack. Run as root on the development host.
#
# This script never touches the live Studio user, compose project, data root,
# or ports. It only creates the requested development account and checkout.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: sudo ./provision-host.sh --ssh-key-file FILE [options]

  --user NAME           dedicated Unix user (default: studioagent)
  --ssh-key-file FILE   public key to authorize for that user (required)
  --key-label LABEL     comment/label for the key (default: studio-dev-agent)
  --repo-url URL        clone URL when the checkout does not exist
  --ref REF             checkout ref after clone (default: main)
  --checkout PATH       checkout path (default: <user-home>/llama-cpp-studio-dev)
  --no-docker-group     do not add the user to the docker group
  --no-clone            only create the user and authorize the key
  -h, --help            show this help

The account needs Docker access to run the isolated compose project. Membership
in the docker group is root-equivalent on the host; use a dedicated host.
EOF
}

USER_NAME="studioagent"
SSH_KEY_FILE=""
KEY_LABEL="studio-dev-agent"
REPO_URL="${STUDIO_DEV_REPO_URL:-}"
REF="${STUDIO_DEV_REF:-main}"
CHECKOUT=""
DOCKER_GROUP=1
NO_CLONE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --user) USER_NAME="${2:?}"; shift 2 ;;
    --ssh-key-file) SSH_KEY_FILE="${2:?}"; shift 2 ;;
    --key-label) KEY_LABEL="${2:?}"; shift 2 ;;
    --repo-url) REPO_URL="${2:?}"; shift 2 ;;
    --ref) REF="${2:?}"; shift 2 ;;
    --checkout) CHECKOUT="${2:?}"; shift 2 ;;
    --no-docker-group) DOCKER_GROUP=0; shift ;;
    --no-clone) NO_CLONE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  echo "Run this script as root (sudo)." >&2
  exit 1
fi
if [[ -z "$SSH_KEY_FILE" || ! -f "$SSH_KEY_FILE" ]]; then
  echo "--ssh-key-file is required and must exist." >&2
  exit 2
fi
if [[ ! "$USER_NAME" =~ ^[a-z_][a-z0-9_-]{0,31}$ ]]; then
  echo "Invalid user name: $USER_NAME" >&2
  exit 2
fi

KEY_TEXT="$(tr -d '\r' < "$SSH_KEY_FILE" | sed -e 's/[[:space:]]*$//' | head -n 1)"
if [[ ! "$KEY_TEXT" =~ ^(ssh-(rsa|ed25519)|ecdsa-sha2-) ]]; then
  echo "The key file does not look like an OpenSSH public key." >&2
  exit 2
fi
KEY_MATERIAL="$(awk '{print $1" "$2}' <<< "$KEY_TEXT")"

if ! id -u "$USER_NAME" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "$USER_NAME"
fi
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
USER_GROUP="$(id -gn "$USER_NAME")"
if [[ -z "$USER_HOME" || ! -d "$USER_HOME" ]]; then
  echo "Could not resolve the home directory for $USER_NAME." >&2
  exit 1
fi
if [[ -z "$CHECKOUT" ]]; then
  CHECKOUT="$USER_HOME/llama-cpp-studio-dev"
fi

install -d -m 700 -o "$USER_NAME" -g "$USER_GROUP" "$USER_HOME/.ssh"
AUTH_KEYS="$USER_HOME/.ssh/authorized_keys"
touch "$AUTH_KEYS"
chown "$USER_NAME:$USER_GROUP" "$AUTH_KEYS"
chmod 600 "$AUTH_KEYS"
if ! grep -Fq "$KEY_MATERIAL" "$AUTH_KEYS" 2>/dev/null; then
  # Keep PTY allocation available for interactive debugging; disable only
  # forwarding. The shell still enforces the Unix account boundary.
  printf '%s no-port-forwarding,no-agent-forwarding,no-X11-forwarding %s\n' \
    "$KEY_TEXT" "$KEY_LABEL" >> "$AUTH_KEYS"
  echo "Authorized key for $USER_NAME ($KEY_LABEL)."
else
  echo "Key is already authorized for $USER_NAME."
fi

if [[ "$DOCKER_GROUP" -eq 1 ]]; then
  if getent group docker >/dev/null 2>&1; then
    usermod -aG docker "$USER_NAME"
    echo "Added $USER_NAME to the docker group."
  else
    echo "WARNING: docker group does not exist; install Docker before starting the stack." >&2
  fi
fi

if [[ "$NO_CLONE" -eq 1 ]]; then
  echo "Checkout provisioning skipped (--no-clone)."
  exit 0
fi

if [[ ! -d "$CHECKOUT/.git" ]]; then
  if [[ -z "$REPO_URL" ]]; then
    if command -v git >/dev/null 2>&1; then
      REPO_URL="$(git -C "$(dirname "$0")/../.." remote get-url origin 2>/dev/null || true)"
    fi
  fi
  if [[ -z "$REPO_URL" ]]; then
    echo "Checkout is missing and no --repo-url was supplied." >&2
    exit 2
  fi
  if [[ -d "$CHECKOUT" && -n "$(ls -A "$CHECKOUT" 2>/dev/null)" ]]; then
    echo "Refusing to clone into non-empty directory: $CHECKOUT" >&2
    exit 2
  fi
  runuser -u "$USER_NAME" -- git clone "$REPO_URL" "$CHECKOUT"
  if [[ -n "$REF" ]]; then
    runuser -u "$USER_NAME" -- git -C "$CHECKOUT" checkout "$REF"
  fi
  echo "Cloned $REPO_URL to $CHECKOUT."
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "WARNING: docker is not installed on this host." >&2
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "WARNING: python3 is not installed on this host; install Python 3.11+ and re-run bootstrap." >&2
  exit 1
fi

if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)'; then
  runuser -u "$USER_NAME" -- env HOME="$USER_HOME" bash -lc \
    "cd '$CHECKOUT' && python3 -m devcontrol bootstrap --repo '$CHECKOUT'"
else
  echo "WARNING: Python 3.11+ is required for devcontrol; bootstrap was not run." >&2
  exit 1
fi

cat <<EOF

Provisioned $USER_NAME with checkout $CHECKOUT.

From an agent machine:
  ssh $USER_NAME@<host> 'agent-dev inspect-instance --json'
  ssh $USER_NAME@<host> 'agent-dev stack build && agent-dev stack up'

The control wrappers are installed under $USER_HOME/.local/bin by bootstrap.
The live Studio compose project, data root, and ports were not touched.
EOF
