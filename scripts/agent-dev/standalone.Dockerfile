# Self-contained development stack: application processes, control service,
# MCP, SSH, state, and checkouts all live in this one container.
#
# It intentionally has no Docker socket, no host bind mounts for code/state, and
# no host OS account or package installation. Build the base runtime first with
# scripts/agent-dev/build-standalone.sh.
ARG BASE_IMAGE=llama-cpp-studio:standalone-base
FROM ${BASE_IMAGE}

USER root
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-server \
    && rm -rf /var/lib/apt/lists/*

# Supervisor runs as PID 1 and owns only the three local programs.
RUN /opt/venv/bin/python -m pip install --no-cache-dir supervisor==4.3.0

# appuser already occupies UID 1001 in the runtime image; use 1002. Key-only
# SSH still needs an unlocked password field; '*' is not a login password and
# is not treated as a locked account by OpenSSH.
RUN groupadd -g 1002 agent \
    && useradd -m -u 1002 -g 1002 -d /workspace -s /bin/bash agent \
    && usermod -p '*' agent

# Immutable source snapshot used only when the workspace volume is empty.
COPY . /opt/studio-src
RUN cd /opt/studio-src \
    && npm ci --no-audit --no-fund \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements-dev.lock \
    && git init -q -b main \
    && git config user.name "studio-standalone" \
    && git config user.email "studio-standalone@localhost" \
    && git add -A \
    && git commit -qm "studio standalone source snapshot"
RUN git config --system --add safe.directory '*'

COPY scripts/agent-dev/standalone/sshd_config /etc/ssh/sshd_config
COPY scripts/agent-dev/standalone/supervisord.conf /etc/supervisor/studio.conf
COPY scripts/agent-dev/standalone/entrypoint.sh /usr/local/bin/studio-entrypoint
COPY scripts/agent-dev/standalone/init-workspace.sh /usr/local/bin/studio-init-workspace.sh
COPY scripts/agent-dev/standalone/run-backend.sh /usr/local/bin/studio-run-backend
COPY scripts/agent-dev/standalone/run-frontend.sh /usr/local/bin/studio-run-frontend
COPY scripts/agent-dev/standalone/agent-dev /usr/local/bin/agent-dev
RUN chmod 0755 \
    /usr/local/bin/studio-entrypoint \
    /usr/local/bin/studio-init-workspace.sh \
    /usr/local/bin/studio-run-backend \
    /usr/local/bin/studio-run-frontend \
    /usr/local/bin/agent-dev \
    && mkdir -p /workspace /data /state/logs /models /run/sshd \
    && chown agent:agent /workspace /data /state /models \
    && ssh-keygen -A

VOLUME ["/workspace", "/data", "/state"]
EXPOSE 2222 5173

# Override the production image's /api/live:8080 healthcheck, which this
# container does not serve.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl -fsS http://127.0.0.1:5173/ || exit 1

ENTRYPOINT ["/usr/local/bin/studio-entrypoint"]
