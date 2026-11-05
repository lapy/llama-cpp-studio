# Host-side control plane in a container.
#
# It only needs Python, git, the Docker CLI, and the Compose plugin. The
# development checkout and control state are bind-mounted at their *host* paths
# so nested Compose bind mounts resolve on the Docker host, not inside
# this container.
FROM python:3.12-slim

ARG DOCKER_VERSION=27.5.1
ARG COMPOSE_VERSION=2.32.4
ARG DOCKER_ARCH=x86_64

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        git \
        openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Static Docker CLI + Compose plugin; no daemon is installed in this image.
RUN curl -fsSL "https://download.docker.com/linux/static/stable/${DOCKER_ARCH}/docker-${DOCKER_VERSION}.tgz" \
        -o /tmp/docker.tgz \
    && tar -xzf /tmp/docker.tgz -C /tmp \
    && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker /tmp/docker.tgz \
    && mkdir -p /usr/local/lib/docker/cli-plugins \
    && curl -fsSL "https://github.com/docker/compose/releases/download/v${COMPOSE_VERSION}/docker-compose-linux-${DOCKER_ARCH}" \
        -o /usr/local/lib/docker/cli-plugins/docker-compose \
    && chmod 0755 /usr/local/lib/docker/cli-plugins/docker-compose \
    && docker --version \
    && docker compose version

# The checkout is often owned by the host appdata user while this container
# runs as root. Git's dubious-ownership check would otherwise reject it.
RUN git config --system --add safe.directory '*'

WORKDIR /workspace
CMD ["sleep", "infinity"]
