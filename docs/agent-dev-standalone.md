# Self-contained development image

This is the "everything inside one container" option:

- The application backend, Vite dev server, control service, MCP server, and SSH
  server all run inside one image.
- There is **no Docker socket**, no host Docker Compose project, no host Python,
  no host Unix account, and no host checkout/state bind mount.
- Code, state, data, and worktrees live in named volumes.
- `apply_patch`, `run_checks`, `reload_service`, `capture_diagnostics`, and
  `rollback_changes` only affect processes and files inside that container.
- Agents connect over SSH on port `2222`; JSON CLI and MCP run in the same SSH
  session.

The Docker host only needs Docker; the agent public key is injected into the
container.

## Build

The standalone image needs the runtime base image first:

```bash
scripts/agent-dev/build-standalone.sh
```

Override tags if needed:

```bash
STUDIO_STANDALONE_IMAGE=my-studio:standalone \
STUDIO_STANDALONE_BASE=my-studio:standalone-base \
  scripts/agent-dev/build-standalone.sh
```

The build runs `npm ci` and installs `requirements-dev.lock` into the image's
`/opt/venv`. The final image contains an immutable source snapshot used only
when the workspace volume is empty.

## First run

Create the SSH key (on the client or the host; it is not mounted):

```bash
mkdir -p /srv/studio-dev
ssh-keygen -t ed25519 -N '' -f /srv/studio-dev/agent_key
```

Start it:

```bash
docker compose -f scripts/agent-dev/standalone.compose.yaml up -d
docker cp /srv/studio-dev/agent_key.pub \
  studio-dev:/state/authorized_keys
```

First start copies the image snapshot into the `workspace` volume, links the
shared `node_modules`, initializes `devcontrol`, and starts `sshd`, Uvicorn, and
Vite under `supervisord`.

For automation, set `STUDIO_AUTHORIZED_KEYS_TEXT` to the public key instead of
using `docker cp`; the entrypoint writes it to `/state/authorized_keys` on every
start.

If you want to share a host model directory, add the optional read-only overlay:

```bash
STUDIO_MODELS_DIR=/srv/llama-cpp-studio-models \
  docker compose \
  -f scripts/agent-dev/standalone.compose.yaml \
  -f scripts/agent-dev/standalone.models.yaml \
  up -d
```

Connect:

```bash
ssh -p 2222 -i /srv/studio-dev/agent_key agent@docker-host
agent-dev inspect-instance --json
agent-dev stack status
```

The frontend is published on the **Docker host's loopback** by default. Tunnel
through the host SSH account, not the container's SSH account:

```bash
ssh root@docker-host -L 15173:127.0.0.1:15173
```

Then open `http://127.0.0.1:15173`. Vite proxies `/api` to the backend inside the
container. The `agent@...:2222` SSH account is for CLI/MCP only; it cannot reach
the host's loopback port. Alternatively set `STUDIO_FRONTEND_BIND=0.0.0.0` and
open the Vite port directly on the LAN.

## MCP

The MCP server is another command in the same SSH session:

```json
{
  "mcpServers": {
    "studio-dev": {
      "command": "ssh",
      "args": [
        "-p", "2222",
        "-i", "/srv/studio-dev/agent_key",
        "agent@docker-host",
        "DEV_ACTOR=alice agent-dev mcp"
      ]
    }
  }
}
```

Use a distinct key per agent, or add multiple lines to `authorized_keys`; the
operation journal records `DEV_ACTOR` when the client sets it.

## What lives where

| Container path | Type | Contents |
| --- | --- | --- |
| `/workspace/repo` | named volume `workspace` | checkout, git refs, checkpoints |
| `/workspace/worktrees` | same volume | per-agent worktrees |
| `/workspace/active` | symlink | checkout currently bound to the processes |
| `/data` | named volume `data` | isolated Studio configuration, operation history, models, caches |
| `/state` | named volume `state` | devcontrol config, journal, checkpoints, logs, supervisor socket |
| `/models` | named volume or optional read-only bind | model files |
| `/state/authorized_keys` | in the `state` volume | agent SSH public keys |

The container has no mount of the live Studio data directory, no host checkout or
state, no SSH key mount, and no Docker socket. The only optional host path is the
read-only model directory when the models overlay is explicitly enabled.

## Updating

- **Code:** apply patches and use worktrees inside `/workspace`; restart with
  `agent-dev reload-service backend`.
- **Image update:** rebuild the image, then recreate the container. The
  `workspace` volume is preserved, so existing patches/checkpoints remain. To
  take a fresh source snapshot and discard the volume:

  ```bash
  docker compose -f scripts/agent-dev/standalone.compose.yaml down -v
  docker compose -f scripts/agent-dev/standalone.compose.yaml up -d
  ```

  This removes the container's workspace/data/state volumes. Do not run it if
  you need the journal or unmerged patches.

## Cloning a repository instead of the snapshot

Set `STUDIO_REPO_URL` (and optionally `STUDIO_REPO_REF`) in the compose
environment. On first start the entrypoint clones it into `/workspace/repo`
instead of copying `/opt/studio-src`. The clone runs inside the container; no
host git credentials are involved unless you provide them to the container.

## GPU (optional)

The default compose file is CPU-only. To pass the host NVIDIA runtime into the
single container:

```bash
docker compose \
  -f scripts/agent-dev/standalone.compose.yaml \
  -f scripts/agent-dev/standalone.gpu.yaml \
  up -d
```

The GPU override uses `gpus: all` and lets `NVIDIA_VISIBLE_DEVICES` select the
device inside the container. Set `STUDIO_CUDA_VISIBLE_DEVICES` to the CUDA
ordinals you want the application to see. `CUDA_DEVICE_ORDER=PCI_BUS_ID` keeps
those ordinals aligned with `nvidia-smi` on mixed-generation hosts. Verify both
`nvidia-smi` and a CUDA driver query in the container before running engine
tests. Like any GPU container, this is not a security boundary around the
device.

## Security boundary

The container is limited to itself in the sense that it has no Docker socket,
no host account, and no host code/state mounts. It is still a Linux container on
the Docker host:

- SSH public-key access controls who can enter the container.
- The container can read `/models` if mounted and can use a GPU if explicitly
  enabled.
- The host can inspect and remove the container/volumes as usual.
- Do not add `/var/run/docker.sock` to this compose file; doing so changes the
  trust model completely.
