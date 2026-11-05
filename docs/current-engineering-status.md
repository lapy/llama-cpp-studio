# Studio engineering reference

Studio runs a modular Python management service, Vue workspace, native inference dataplane, and supervised engine processes. The reference below describes the implementation boundaries and validation tools.

## Source map

| Area                                                     | Principal implementation                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------- |
| Application lifecycle, API registration, static frontend | `backend/main.py`                                                         |
| Engine capability registry                               | `backend/engines/registry.py`                                             |
| Engine-specific installers, scanners, activation         | `backend/engines/` and engine route modules                               |
| Model acquisition, files, configuration, actions         | `backend/routes/models.py`, `backend/model_catalog/`, `backend/services/` |
| Model configuration normalization and templates          | `backend/models/config.py`, `backend/model_config_templates.py`           |
| Native proxy/routing/process ownership                   | `backend/proxy/studio/manager.py`, `dataplane.py`, `registry.py`          |
| Streaming HTTP transport and usage observation           | `backend/proxy/studio/transport.py`, `usage.py`                           |
| Quotas, request metadata, durable reservations           | `backend/proxy/studio/accounting.py`                                      |
| Launch compiler and command emission                     | `backend/proxy/launch_spec.py`, `commands.py`                             |
| Immutable launch generations and receipts                | `backend/proxy/manifests.py`                                              |
| Runtime planning, selective/batch apply, rollback        | `backend/services/model_runtime_apply.py`, `backend/routes/runtime.py`    |
| Management and inference authorization                   | `backend/access_policy.py`, `backend/inference_auth.py`                   |
| Serialized document persistence                          | `backend/data_store.py`, `backend/store_io.py`                            |
| Portable restore and private history                     | `backend/config_backup.py`, `backend/config_history.py`                   |
| Operations, admission, progress, reconciliation          | `backend/operations/`                                                     |
| Audio integration contract and inspection                | `backend/engines/audio_cpp/integration/`, `backend/audio/`                |
| Workspace views and feature components                   | `frontend/src/views/`, `frontend/src/components/`                         |
| Shared API contracts and generated schema                | `frontend/src/api/`                                                       |
| Isolated agent development control                       | `devcontrol/`, `scripts/agent-dev/` (not in production images)            |

## Configuration persistence

Configuration mutation holds thread/process serialization across read, validation, mutation, and write. Atomic replacement and bounded write admission establish document-level behavior. Corrupt/unreadable state produces a failure; mutation does not silently create a replacement configuration.

Persistence responses distinguish an uncommitted write, a committed replacement with an acknowledgement/durability failure, and an unknown outcome. Frontend persistence notices retain the draft and expose the appropriate retry, refresh, or reconciliation action. A response timeout alone does not establish whether the filesystem changes.

Portable restore spans settings, templates, and model configuration documents with a durable journal and revision-bound preview. Startup/application reconciliation follows the journal. Private history records bounded configuration snapshots and supplies redacted diffs plus revision-checked restoration of a supported item. Restore changes saved configuration; it does not imply process launch or launch publication.

YAML contains configuration and operation records. SQLite stores native proxy accounting, telemetry, policy reservations, and observation-window state. Portable exports and operational-volume/database backups serve different purposes.

## Runtime and operation guarantees

The compiler builds a structured launch specification with an executable, ordered arguments, working directory, environment set/unset operations, artifact digests, and relevant file identities. A launch revision identifies immutable launch content. Desired configuration, published revision, and verified running revision remain separate.

Per-model application validates review expectations, journals the selected operation, and affects that model. Sequential batch apply uses the same service and stops after a failed outcome. Native route snapshots support live default/group changes while accepted requests retain their target. Request presets resolve to a real process and use JSON parameter filters.

The process registry shares cold starts, tracks startup/drain/ready states, preserves unrelated running processes during selective apply, monitors unexpected exits, retains bounded logs, and owns process-group termination. A cancelled cold-start waiter does not cancel work needed by another waiter. Unused starts can be cancelled when their final waiter leaves.

Durable operation admission and resource fences prevent automatic duplicate side effects when work is busy or uncertain. A confirmation binds an operation/state to the deliberate retry. Startup reconciliation classifies interrupted or uncertain work; it does not equate a missing response with absence of an effect.

## Authorization and privacy boundaries

Management authorization checks local/remote access policy, bearer tokens, or browser sessions/CSRF as applicable. Native proxy administration uses Studio's internal service credential. Inference keys grant scoped inference/catalog access and optional model/policy restrictions; an inference credential does not grant management operations.

Request observations contain bounded metadata and usage rather than prompt/completion bodies or authorization secrets. The local benchmark feature separately records its explicit prompt and output preview. Private configuration history can contain sensitive local state despite displayed redaction. Data-directory access and full-backup handling must reflect those differences.

Quota reservations commit before dispatch, count against concurrent admissions, and survive process interruption. Batched telemetry has a different durability profile. The local accounting database is a single-volume service component, not a distributed multi-host quota store.

## API and version contracts

`VERSION` supplies the application version. `scripts/check-version.py` checks consistency with package and backend metadata. `scripts/export-openapi.py` exports a deterministic management schema; the checked-in document is `frontend/src/api/openapi.json`.

The TypeScript boundary in `frontend/src/api/configuration.ts` covers portable backup/restore and history contracts. Benchmark routes expose their Pydantic request/response contract through OpenAPI. Other APIs use their route/service contracts and frontend request implementations. The management schema does not enumerate the separate native inference proxy's forwarded engine APIs.

## Reproducible checks

From an activated Python 3.12 environment with the locked development dependencies and Node.js 24+:

```bash
npm run test:frontend
python -m pytest backend/tests -q
npm run lint
npm run typecheck:api
npm run style:check
npm run format:check
npm run openapi:check
npm run build
npm run test:browser
node frontend/e2e/cold-models-budget.mjs
```

The frontend production build requires the configured PrimeUI license value. Scoped Python/Prettier format checks follow the package/CI command definitions. ESLint and Stylelint cover the configured frontend sources; Ruff covers backend and scripts.

Additional tools include:

```bash
python -m scripts.proxy_conformance --report conformance.json
python -m scripts.proxy_benchmark --fixture --accounting quota --requests 256 --concurrency 32
pip-audit -r requirements.lock
npm audit --omit=dev --audit-level=high
```

CI configuration in `.github/workflows/` defines unit/static/schema checks, audits, browser/navigation jobs, container smoke tests, image scans, SBOM generation, publication, and attestation. Job conditions and repository secrets determine which release/publication steps run.

## What validation establishes

| Check                                 | Scope                                                                                                                                               |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frontend unit tests                   | View/composable behavior, payloads, persistence states, parameter parsing, and UI state transitions.                                                |
| Backend unit/integration tests        | Configuration, launch compilation, alias/routing semantics, process lifecycle, action admission, restore/history, and API behavior.                 |
| Real CPU process/socket fixtures      | Startup coordination, HTTP/SSE streaming, cancellation, headers, compression, multipart ordering, and process cleanup.                              |
| Browser checks                        | Navigation, responsive layout, role/name/focus behavior, accessibility assertions, configuration save/reload, proxy controls, and request examples. |
| Production build and asset checks     | Bundle generation, configured contracts, and the declared asset budget.                                                                             |
| Container checks                      | Image startup, mounted data/user handling, navigation, shutdown, and image-specific behavior.                                                       |
| Device/model smoke and workload tests | Compatibility and resource/performance behavior of the actual engine, model, driver, toolkit, and hardware.                                         |

CPU fixtures do not measure GPU throughput. Browser assertions do not replace a manual screen-reader session. Container/image checks require Docker. A successful build does not certify every engine/model/device combination. Operational rollout uses the checks appropriate to the actual deployment.
