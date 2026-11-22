# Single-operator deployment guide

Studio is designed for one technical operator on one host. This guide states the
supported boundary and the operational practices that follow from it.

## Scope

Supported:

- One Studio management service (one container or one local process) and one
  native proxy on the same host.
- One operator who owns the data directory, engine installations, models,
  inference keys and backups.
- Local-only access, or remote access through an explicit token/session and a
  TLS-terminating reverse proxy.
- Multiple models and engines on that host, subject to the GPU/memory limits
  described below.

Not supported in this release:

- Multi-tenant RBAC, SSO/OIDC, per-user audit identity.
- High availability, clustering, or multiple Studio writers against one data
  root.
- Distributed quota accounting or multi-host GPU scheduling.
- Automatic model eviction to make room for another model.

If several people need to share inference, keep Studio on a trusted host and
front it with a reverse proxy that authenticates users; give each client a
separate inference key. The management token remains the operator credential.

## Deployment modes

| Mode | Default surface | Use |
| --- | --- | --- |
| Docker CPU | UI/API `8080`, inference `2000` on loopback | Evaluation, GGUF/audio CPU work |
| Docker NVIDIA | same, with the NVIDIA runtime | GPU inference |
| Local dev | `npm run dev:all`: Vite `5173`, API `8081`, proxy `2000` | Development |
| Remote access | management and inference bound explicitly; TLS at a reverse proxy | Trusted network / VPN |

`STUDIO_BIND_HOST` controls management binding; `STUDIO_PROXY_LISTEN` controls
the inference listener; `proxy_port` controls the inference port. The
client-facing `public_inference_url` is a display preference and does not rebind
the listener.

## Security boundary

- Management and inference authorization are separate. An inference key grants
  only its scopes and model policy; it cannot call `/api/*` management routes.
- Local mode trusts loopback and the container bridge. Do not expose the
  management port to an untrusted network.
- Remote mode requires `STUDIO_API_TOKEN` (or the mode-0600 token file) and uses
  a browser session plus CSRF for UI mutations. Rotate the token by restarting
  with a new value; active sessions then need a new login.
- Terminate TLS at a reverse proxy. Forward only the management and inference
  ports you intend to expose. Do not put tokens or credentials in URLs.
- Inference keys are bearer credentials. They are shown once on creation or
  rotation; store them in the client secret store, not in shell history or
  screenshots. Revoke a key before deleting its row.
- The diagnostics bundle and configuration history may contain local paths and
  operational state. Treat the data directory and every backup as private.

## GPU and toolkit practice

- Use the engine matrix in the README and the per-engine docs. CUDA 12.8/12.9 is
  required for Volta/SM70 paths; CUDA 13 drops SM70. GCC 15 is rejected by
  nvcc for the CUDA 12 builds.
- One model can use the whole host. Studio bounds the number of loaded models
  and can warn or refuse a start based on free VRAM when GPU admission is
  enabled; it does not yet evict a healthy model or schedule across other
  applications on the host.
- Do not run a large model load while another process is using the same GPUs.
  Check `nvidia-smi` before a cold start; Studio cannot account for external
  GPU memory.
- Model load failures are reported as engine/runtime failures with the engine
  log. A successful software test does not prove throughput or every
  engine/model/device combination.

## Data, storage and backup

- `STUDIO_DATA_DIR` (default `./data`, container `/app/data`) is the complete
  operational state: configuration, models, engine installs, CUDA toolkits,
  caches, logs, history and the accounting database.
- A **portable backup** (System → Backup & restore) contains only portable
  configuration: public inference URL, proxy port, model settings and
  configuration templates. It excludes weights, credentials, installs,
  processes, history, launch generations and the accounting ledger.
- A **full operational backup** must copy the data volume and a consistent
  SQLite backup of the accounting database. Stop Studio or use filesystem/SQLite
  snapshot semantics; copying a live SQLite file without care can produce an
  inconsistent backup.
- The Storage section reports model payloads, Hugging Face cache, engine
  installs, CUDA toolkits, JIT caches, history/logs and the accounting DB. Only
  Studio-owned, unreferenced items are cleanable there. Deleting a running or
  retained model remains the model/engine surface's decision.
- Prefer deleting models in the Models workspace so launch references and
  retired generations are handled first, then reclaim orphaned cache entries in
  Storage.

## Operating routine

1. Start Studio, open System → Hardware and confirm the expected GPU, driver and
   toolkit.
2. Install or select one engine version; activate it.
3. Search for a model, download it, review its configuration, then start it.
4. Check Proxy → Overview and the model's logs for readiness. Use Model Library
   → Connect to copy a client configuration; create an inference key if the
   client is not on loopback.
5. Before upgrading an engine or CUDA toolkit, stop the models that use it and
   take a portable backup plus a data-volume backup.
6. Review System → Storage monthly, System → Change history for accidental
   edits, and the diagnostics bundle when a probe or proxy state is unclear.
7. On an unclean shutdown, start Studio and inspect Needs attention, operations
   and the proxy's recovery state before retrying a mutation.

## Incident checklist

| Symptom | First checks |
| --- | --- |
| Model will not start | Engine log, selected engine version, configured GPU devices, free VRAM, CUDA/toolkit path, admission message |
| Proxy unreachable | Proxy lifecycle state, listener/port, bind address, firewall, reverse proxy |
| 401/403 from inference | Inference key scope, model restriction, expiry, rate/token budget; management token is not an inference key |
| UI shows stale or unknown state | Freshness indicator, backend `/api/status`, SSE/live-updates health, diagnostics bundle |
| Install/build fails | Install log, host compiler/CUDA selection, disk space, cancellation/operation state |
| Disk full | System → Storage categories; delete unreferenced models/caches; move the data root only with Studio stopped |
| Settings save looks stuck | Persistence queue state, retry/reconcile action; do not edit YAML by hand |

## Validation

Run the release gates in `docs/current-engineering-status.md`. Hardware claims
must name the engine, model, GPU, driver and toolkit actually used. CPU fixtures,
mocked installers and browser fixtures do not establish GPU behavior.
