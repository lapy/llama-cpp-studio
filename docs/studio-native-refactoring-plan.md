# Studio feature implementation map

This reference maps current features to the source and validation that implement them. The [feature guide](features.md) supplies user workflows, the [API reference](api-reference.md) supplies request contracts, and the [engineering reference](current-engineering-status.md) supplies system boundaries.

## Workspaces and visual structure

| Feature                                                                       | Implementation                                                                                           | Validation                                                                   |
| ----------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| Studio orbital identity, compact header, workspace navigation, theme controls | `frontend/src/components/layout/AppHeader.vue`, `WorkspaceNavigator.vue`, shared layout/theme components | Header/navigation unit tests and browser layout checks.                      |
| Models card/table library, filters, runtime/file/action columns               | `frontend/src/views/ModelLibrary.vue`, model components                                                  | Library unit tests and browser table/filter workflows.                       |
| Model configuration, field help, commands, engine sections, templates         | `frontend/src/views/ModelConfig.vue`, configuration components and stores                                | Configuration/runtime/profile tests and browser save/import workflows.       |
| CLI command import with full list arity                                       | `frontend/src/utils/parseCliCommand.js`, `ParseCommandDialog.vue`, `backend/proxy/commands.py`           | Parser regressions, native argument emission tests, browser import/save.     |
| Audio task workspace                                                          | `frontend/src/views/AudioWorkspace.vue`, audio composables/components                                    | Workspace/request/profile tests and browser journeys.                        |
| Engine installs/versions/build configuration                                  | `frontend/src/views/EnginesView.vue`, `backend/engines/`, engine routes                                  | Installer/activation/scanner/API tests and task-operation coverage.          |
| Proxy overview, keys, policies, routing, requests and logs                    | `frontend/src/views/ProxyView.vue`, `backend/routes/proxy_admin.py`, inference-key routes                | Proxy unit/API tests and browser policy/routing workflows.                   |
| System hardware and CUDA controls                                             | `frontend/src/views/SystemView.vue`, `components/system/HardwarePanel.vue`, hardware/CUDA services       | System tests, probe/API tests, responsive navigation/control browser checks. |
| Backup, history, diagnostics, benchmark tools                                 | System components and matching backend routes/services; benchmark API/service                            | Persistence, restore/history, benchmark, and browser workflows.              |

## Native inference features

| Feature                                                  | Implementation                                                               | Validation                                                                                           |
| -------------------------------------------------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Real-model IDs and ordinary aliases                      | `backend/data_store.py`, launch compiler, native manager                     | Alias normalization, collision, routing, and acquisition tests.                                      |
| Named request presets and JSON overrides                 | `RequestAliasesEditor.vue`, `backend/proxy/commands.py`, native `filters.py` | Boolean/nested value, save/reload/removal, default/preset resolution, and request-isolation tests.   |
| Live single-/multi-target routing, round-robin selection | Native `RoutingSnapshot` and manager resolver; Proxy editor/settings route   | Route publication/active-stream preservation tests and browser multi-target save.                    |
| Request filters and model-selection headers              | Native dataplane/filter interfaces                                           | Header/filter contract tests.                                                                        |
| Desired/published/running revisions                      | Launch compiler, manifests, receipts, apply service                          | Compiler, pointer, receipt, journal, and plan tests.                                                 |
| Per-model and sequential batch apply                     | Model runtime routes, `/api/runtime/apply-launch`, apply service             | Revision/idempotency, sequential-stop, rollback, and endpoint tests.                                 |
| Process supervision and lifecycle                        | Native `registry.py`/`manager.py`                                            | Shared starts, cancellation, unexpected exits, drain, restart, process cleanup, and socket fixtures. |
| Stream/backpressure/upstream transport                   | Native `dataplane.py`, `transport.py`, `usage.py`                            | Real CPU fixture JSON/SSE, status, compression, headers, multipart, and cancellation conformance.    |
| Access keys/scopes/model restrictions                    | `backend/inference_auth.py`, native identity/admission                       | Authentication, scoped catalog/routing, policy, and administration boundaries.                       |
| Budgets/reservations/telemetry/statistics                | Native `accounting.py`, request supervisor/usage observers                   | Atomic admission, reservations, UTC boundaries, restart/reconciliation, and history tests.           |
| Engine UIs and concrete passthrough                      | Native path resolver/UI transport, audio proxy routes                        | Path/auth/cookie/header and audio transport tests.                                                   |

## Persistence and metadata

| Feature                                                 | Implementation                                                             | Validation                                                                                           |
| ------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Atomic serialized configuration mutation                | `backend/data_store.py`, `store_io.py`                                     | Corruption, concurrent mutation, write checkpoints, commitment/outcome tests.                        |
| Portable configuration backup/preview/restore           | `backend/config_backup.py`, backup route and typed frontend boundary       | Validation, mapping, conflict, revision, journal, reconciliation, and UI recovery tests.             |
| Private configuration history and selected-item restore | `backend/config_history.py`, history route/components                      | Redacted diffs, retention, revision checks, scope and restore tests.                                 |
| Durable operations and resource admission               | `backend/operations/`                                                      | State-machine, side-effect fence, busy/confirmation, cancellation, and startup reconciliation tests. |
| Engine/audio capability discovery                       | Registry, scan modules, audio integration composer, model specs/inspection | Help/JSON fixtures, precedence, profile cache invalidation, contract drift, and activation tests.    |
| Generated API contract                                  | `scripts/export-openapi.py`, `frontend/src/api/openapi.json`               | Deterministic schema export/check and API type checks.                                               |

Hardware monitoring, build support, inference capability, and measured performance are separate claims. CPU fixtures verify software contracts; the actual installed engine/model and target hardware establish their device-specific behavior.
