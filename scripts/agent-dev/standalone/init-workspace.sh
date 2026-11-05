#!/bin/bash
set -euo pipefail

mkdir -p /workspace /data /state/logs /workspace/worktrees /run/sshd

# The volumes may be created by Docker as root; the SSH user owns them.
chown agent:agent /workspace /data /state
chown agent:agent /state/logs
chown agent:agent /workspace/worktrees

if [[ ! -d /workspace/repo/.git ]]; then
  rm -rf /workspace/repo
  if [[ -n "${STUDIO_REPO_URL:-}" ]]; then
    if [[ -n "${STUDIO_REPO_REF:-}" ]]; then
      runuser -u agent -- git clone --branch "$STUDIO_REPO_REF" "$STUDIO_REPO_URL" /workspace/repo
    else
      runuser -u agent -- git clone "$STUDIO_REPO_URL" /workspace/repo
    fi
  else
    mkdir -p /workspace/repo
    cp -a /opt/studio-src/. /workspace/repo/
    # The shared node_modules volume is linked in below; do not duplicate the
    # image's install into the checkout.
    rm -rf /workspace/repo/node_modules
  fi
fi

chown -R agent:agent /workspace/repo

if [[ ! -d /workspace/node_modules ]]; then
  if [[ -d /opt/studio-src/node_modules ]]; then
    cp -a /opt/studio-src/node_modules /workspace/node_modules
  else
    (cd /workspace/repo && runuser -u agent -- npm ci --no-audit --no-fund)
    cp -a /workspace/repo/node_modules /workspace/node_modules || true
  fi
fi
chown -R agent:agent /workspace/node_modules

for directory in /workspace/repo /workspace/worktrees/*; do
  [[ -d "$directory" ]] || continue
  if [[ ! -e "$directory/node_modules" ]]; then
    ln -s /workspace/node_modules "$directory/node_modules"
    chown -h agent:agent "$directory/node_modules"
  fi
done

if [[ -d /workspace/repo/.git ]]; then
  if ! grep -qx 'node_modules' /workspace/repo/.git/info/exclude 2>/dev/null; then
    echo 'node_modules' >> /workspace/repo/.git/info/exclude
  fi
  chown agent:agent /workspace/repo/.git/info/exclude
fi

if [[ ! -L /workspace/active ]]; then
  ln -sfn /workspace/repo /workspace/active
fi
chown -h agent:agent /workspace/active

if [[ ! -f /state/config.json ]]; then
  runuser -u agent -- env HOME=/workspace /usr/local/bin/agent-dev bootstrap --no-wrappers
fi

# sshd reads this as root. The entrypoint writes an environment-injected key.
if [[ ! -s /state/authorized_keys ]]; then
  echo "studio: /state/authorized_keys is empty; copy an agent public key into it" >&2
fi
