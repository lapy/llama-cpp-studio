#!/bin/bash
set -euo pipefail

/usr/local/bin/studio-init-workspace.sh

mkdir -p /run/sshd /state/logs
chown agent:agent /state /state/logs

# Prefer an explicit key injected through the environment (no host file
# mount). Otherwise leave the persistent /state/authorized_keys in place;
# operators can `docker cp` a public key into it after the container starts.
if [[ -n "${STUDIO_AUTHORIZED_KEYS_TEXT:-}" ]]; then
  printf '%s\n' "$STUDIO_AUTHORIZED_KEYS_TEXT" > /state/authorized_keys
fi
touch /state/authorized_keys
chmod 600 /state/authorized_keys
chown agent:agent /state/authorized_keys

exec /opt/venv/bin/supervisord -c /etc/supervisor/studio.conf
