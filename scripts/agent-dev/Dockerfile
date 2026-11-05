# Dev-tools layer on top of the runtime image built from the repository
# Dockerfile. devcontrol builds the base image first and passes it here.
ARG BASE_IMAGE=llama-cpp-studio:agent-dev-base
FROM ${BASE_IMAGE}

USER root
WORKDIR /app

COPY requirements-dev.lock /tmp/studio-agent-requirements.lock
RUN /opt/venv/bin/python -m pip install --no-cache-dir -r /tmp/studio-agent-requirements.lock \
    && sha256sum /tmp/studio-agent-requirements.lock | cut -d' ' -f1 \
        > /opt/venv/.agent-requirements-sha \
    && rm /tmp/studio-agent-requirements.lock

USER appuser
