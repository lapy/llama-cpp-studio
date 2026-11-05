# Remote development stack for agents

This describes the isolated development stack controlled by `devcontrol`. It is
separate from the live Studio instance and from the local `docker-compose.dev.yml`
container:

| Concern | Live instance | Agent development stack |
| --- | --- | --- |
| Compose project | `llama-cpp-studio` | `llama-cpp-studio-dev` |
| Checkout | deployment checkout | `~/llama-cpp-studio-dev` plus per-agent worktrees |
| Data root | deployment `data/` | `~/llama-cpp-studio-dev-data` |
| Ports | usually `8080`, `2000` | `18081`, `15173`, `12000` by default |
| Access | production credentials | a dedicated dev management token |
| Code | built artifact | bind-mounted source with Uvicorn/Vite reload |

`devcontrol` lives outside `backend/`. The production image only copies
`backend/`, `frontend/public`, and the built frontend, so this control plane is
not importable from a production build.

## What is implemented

- A separate git checkout and data/state roots, with a Compose project name that
  cannot be the production project.
- A source-mounted dev container built from the repository `Dockerfile` plus a
  dev-tools layer, with Python `--reload`, Vite HMR, and a named `node_modules`
  volume.
- Optional GPU selection through `DEV_GPU`/`NVIDIA_VISIBLE_DEVICES` and an
  optional Compose override.
- SSH access for agents through a dedicated Unix account and per-agent keys.
- A JSON CLI (`python -m devcontrol`, or the `agent-dev` wrapper) and a stdio MCP
  server using the same guarded service.
- An optional containerized controller for hosts without a suitable host Python.
- Operation history, checkpoints, worktree isolation, an instance guard, and
  bounded log/diagnostic access.
- Guarded `inspect_instance`, `read_logs`, `apply_patch`, `run_checks`,
  `reload_service`, `capture_diagnostics`, and `rollback_changes` actions.

The control plane is standard-library only. The MCP transport is implemented
directly as JSON-RPC 2.0 over newline-delimited stdio; no host-side MCP package
is required.

## Host prerequisites

For a single-container option with no Docker socket and no host OS changes, see
[Self-contained development image](agent-dev-standalone.md).

- Linux host with Docker Engine and Compose v2. `docker compose version` must
  work for the development user.
- Git and Python 3.11 or newer on the host.
- NVIDIA Container Toolkit and a working `nvidia-smi` only when GPU work is
  selected.
- SSH server and a dedicated account. Do not reuse the live Studio account.

Docker group membership is root-equivalent on the host. Use a dedicated
development machine or a dedicated account, and treat every agent with access to
that account as trusted to control Docker on that host. The control plane's
guards prevent accidental cross-project changes; they are not a hard security
boundary against a malicious Docker user.

## Provision the host

From a checkout on the development host, as root:

```bash
sudo bash scripts/agent-dev/provision-host.sh \
  --user studioagent \
  --ssh-key-file /tmp/agent-alice.pub \
  --repo-url git@github.com:OWNER/llama-cpp-studio.git \
  --ref main
```

The script creates the user, authorizes the key with forwarding restrictions,
adds the user to the `docker` group when available, clones the checkout, and
runs `bootstrap` as that user. It does not modify the live account, Compose
project, data root, or ports.

Each agent should get its own key and label:

```bash
sudo bash scripts/agent-dev/provision-host.sh --user studioagent \
  --ssh-key-file /tmp/agent-bob.pub --key-label bob --no-clone
```

The generated key line disables port, agent, and X11 forwarding but keeps PTY
allocation. Agents get an interactive shell and SFTP so they can edit files and
run commands.

### Manual bootstrap

If the host already has a checkout:

```bash
cd ~/llama-cpp-studio-dev
python3 -m devcontrol bootstrap --repo "$PWD" --repo-url git@github.com:OWNER/llama-cpp-studio.git
```

`bootstrap` is idempotent. It creates the state root (`~/.local/state/llama-cpp-studio-dev`
by default), a separate data root, worktree root, shared-model root, the dev
management token, an initial checkpoint, and `agent-dev`/`agent-dev-mcp` wrappers
in `~/.local/bin`. It never rewrites the live configuration. Re-run it after
changing paths or with `--reset-token` to rotate the token.

## Build and start

```bash
agent-dev stack build          # builds llama-cpp-studio:agent-dev-base, then the agent dev layer
agent-dev stack up             # starts the isolated project
agent-dev inspect-instance --json
```

`stack up` refuses to start if the dev image is missing, and validates that the
Compose project, image tags, paths, and ports are not the live/default values.
On first start the `init` service installs `requirements-dev.lock` into
`/opt/venv`, runs `npm ci` into the named volume, and fixes ownership of only the
isolated volumes.

Open the frontend at `http://127.0.0.1:15173` (Vite) and the management API at
`http://127.0.0.1:18081`. The native proxy is on `127.0.0.1:12000`.

### GPU selection

```bash
agent-dev stack up --gpu 0
agent-dev inspect-instance --json      # gpu.requested and gpu.devices
agent-dev stack up --gpu 1,2
agent-dev stack up --gpu ""            # disable GPU for the next start
```

The Compose override reserves the NVIDIA driver capability, while
`NVIDIA_VISIBLE_DEVICES` limits the container to the selected host devices. The
control service keeps `CUDA_VISIBLE_DEVICES` remapped to the container's
zero-based visible set. GPU work is still shared physical hardware: coordinate
long runs with the live instance and other agents.

## Worktrees and coordination

Each agent should use its own worktree:

```bash
agent-dev worktree create alice --base HEAD
agent-dev worktree list --json
agent-dev worktree activate alice
```

Only one worktree is mounted into the running backend/frontend at a time.
`activate` re-runs `init`, recreates the two application containers with the new
bind mount, waits for health, and records the operation. The previous active
worktree is restored if startup fails.

Mutating operations take an instance lock. A concurrent operation returns
`OPERATION_BUSY` with the holder's operation ID and actor:

```json
{
  "ok": false,
  "operation_id": "op-20261105T105000-ab12cd34ef",
  "revision": "8b1d0f...",
  "error": {
    "code": "OPERATION_BUSY",
    "message": "another operation is active on the shared development instance",
    "detail": {
      "holder": {
        "operation_id": "op-20261105T104959-...",
        "action": "run_checks",
        "worktree": "main",
        "actor": "bob@host"
      }
    },
    "exit_code": 3
  }
}
```

Read-only operations (`inspect_instance`, `read_logs`, `capture_diagnostics`,
worktree list) do not take the instance lock and can run while another agent is
working. Editing an inactive worktree uses a per-worktree lock, so different
agents can prepare patches in parallel; they serialize only when activating or
touching the active worktree.

## JSON contract

Every action prints one JSON object. Successful actions always include:

- `operation_id`: unique ID in `operations.jsonl`.
- `revision`: the active checkout's commit, or
  `<commit>+dirty.<tree-id>` when the working tree is not committed.
- `revision_detail.commit`, `.tree`, `.branch`, `.dirty`, `.changed_files`.
- `action`, `worktree`, `ok`, and `status`.

The dirty tree ID is computed from the full working tree, so it proves which
uncommitted source the service saw. Agents should record the revision returned
by `apply_patch`/`inspect_instance` rather than assuming the commit is enough.

Errors are JSON too and carry `operation_id`, `revision` when available, and an
`error` object with `code`, `message`, `detail`, and `exit_code`. Exit codes are
`0` success, `1` failure, `2` usage, and `3` guard/busy refusal.

## CLI reference

```bash
agent-dev inspect-instance --json
agent-dev read-logs --source app --service backend --tail 200 --level error
agent-dev read-logs --source build --tail 300
agent-dev read-logs --source model --model llama-3 --tail 500

agent-dev apply-patch --patch /tmp/fix.diff
agent-dev apply-patch --worktree alice --stdin < /tmp/fix.diff
agent-dev apply-patch --check-only --patch /tmp/fix.diff

agent-dev run-checks --check ruff
agent-dev run-checks --check backend-tests --test backend/tests/test_engine_contracts.py
agent-dev run-checks --check frontend-tests --frontend-target frontend/src/stores/engines.test.js

agent-dev reload-service backend
agent-dev capture-diagnostics --reason "model load exits 1" --note "only on V100"
agent-dev rollback-changes --worktree alice
agent-dev rollback-changes --worktree alice --checkpoint cp-20261105T105000-...
agent-dev journal --limit 25 --json
```

`--json` is the default and may appear before or after the subcommand.

### Patches and checkpoints

`apply_patch` accepts a UTF-8 unified diff up to 2 MiB. It rejects absolute
paths, parent traversal, `.git` paths, binary patches, and patches that do not
apply cleanly. Before applying, it creates a checkpoint of the worktree
(including non-ignored untracked files). `--no-checkpoint` disables that, but is
not recommended for agents.

`rollback_changes` restores the checkpoint tree without moving the branch pointer.
It brings back files that existed at checkpoint time and removes non-ignored files
created later; ignored `node_modules`/data directories are preserved. If no
checkpoint is named, the latest checkpoint for that worktree is used.

### Checks

Checks are an allowlist. They run inside the development containers, not on the
host:

| Name | What it runs |
| --- | --- |
| `backend-tests` | `python -m pytest backend/tests -q --tb=short` |
| `backend-tests-fast` | the same with `-x` |
| `ruff` | `python -m ruff check backend scripts devcontrol` |
| `frontend-tests` | `npm run test:frontend` |
| `frontend-lint` | `npm run lint:frontend` |
| `typecheck` | `npm run typecheck:api` |
| `style` | `npm run style:check` |
| `format` | `npm run format:check` |
| `openapi` | `npm run openapi:check` |
| `lint` | `npm run lint` (aggregate static checks) |

`--test` targets must stay under `backend/tests`; `--frontend-target` targets must
stay under `frontend/src` or `frontend/e2e`. There is no arbitrary command
option. Checks run only against the active worktree, so activate a worktree
before testing it.

### Logs

- `source=app` reads bounded `docker compose logs` for `backend`, `frontend`, or
  both.
- `source=build` reads recent `*.log` files under the isolated data root
  (llama.cpp, audio.cpp, and install logs). If none exist, it returns backend app
  logs with an explicit `fallback` field.
- `source=model` reads the application's retained model log buffer through
  `/api/proxy/models/{id}/logs`; the model must be running.

All paths are bounded by line count and bytes and pass through best-effort secret
redaction.

### Diagnostics

`capture_diagnostics` writes a redacted bundle under
`~/.local/state/llama-cpp-studio-dev/diagnostics/<operation_id>/` and a
`.tar.gz` beside it. It includes inspection output, Compose state, app/build and
running-model log tails, the git diff, untracked files, recent operations, the server diagnostics
bundle when reachable, and `nvidia-smi` when GPU selection is active. The
response returns both paths, the file list, and the archive size.

## MCP

The MCP server speaks JSON-RPC 2.0 over stdio, so it can be started by an SSH
client without opening another network port:

```json
{
  "mcpServers": {
    "studio-dev": {
      "command": "ssh",
      "args": [
        "-i", "~/.ssh/studio-agent",
        "studioagent@dev-host",
        "agent-dev-mcp"
      ]
    }
  }
}
```

Tools exposed: `inspect_instance`, `read_logs`, `apply_patch`, `run_checks`,
`reload_service`, `capture_diagnostics`, `rollback_changes`, plus
`checkpoint_changes`, `list_worktrees`, `create_worktree`, and
`activate_worktree` for multi-agent setup. Tool calls return MCP
`content`/`structuredContent`; `isError` is true when the underlying operation
is refused or fails.

If an MCP client requires a direct command rather than an SSH wrapper, install a
small launcher on the host or use:

```json
{"command":"ssh","args":["studioagent@dev-host","cd ~/llama-cpp-studio-dev && python3 -m devcontrol mcp"]}
```

## Security and isolation notes

- The live compose project, data root, and default ports are not used. The
  control service validates its project name and image tags and refuses
  `8080`/`2000`/`5173` for its own ports.
- The dev management token is generated separately and stored in mode-0600
  `dev.env`; it is not the live token. Redaction is best effort, not a guarantee
  that application logs contain no sensitive data.
- The control plane does not expose a network listener. SSH credentials and the
  SSH server are the outer boundary.
- `DEV_DATA` must be a dedicated directory. The validator rejects data roots
  inside the checkout, but it cannot prove a path is not the live instance's data
  directory; check the generated `config.json` before the first `stack up`.
- Shared models are mounted read-only at `/app/shared-models`. The application
  does not automatically index that directory; use the normal model-import flow
  or symlink/copy into the isolated model root as appropriate.
- `reload_service` restarts the selected container. Ordinary edits are picked up
  by Uvicorn autoreload and Vite HMR; use `reload_service` when a clean process
  restart is needed.
- The production image does not include `devcontrol`; do not add a production
  entry point for it.

## Verification

Run the control-plane tests from a checkout with the dev dependencies installed:

```bash
python -m pytest devcontrol/tests -q
python -m ruff check devcontrol scripts/agent-dev
```

The suite uses temporary repositories and does not require Docker. The Compose
stack itself still needs a real Docker host for `stack build`, `stack up`, and
`run_checks`.

## Troubleshooting

- `OPERATION_BUSY`: inspect `detail.holder`, wait for that operation, or use
  `journal` to see the operation's action.
- `CONFIG_INVALID`: run `bootstrap` and check the generated paths/ports.
- `image is missing`: run `agent-dev stack build`; the base image must build
  from the repository `Dockerfile` before the dev layer.
- Backend health fails after `activate`: check `read-logs --source app --service backend`;
  the operation restores the previous active worktree when container startup
  itself fails.
- GPU selection fails: verify `nvidia-smi -L` on the host and the NVIDIA
  Container Toolkit; clear `DEV_GPU` to fall back to CPU.
- Docker permission denied: add the dedicated user to the `docker` group and
  start a fresh login session.
