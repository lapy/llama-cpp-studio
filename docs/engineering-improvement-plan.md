# Engineering review and improvement plan

Reviewed 2026-09-27 against the initial clean working tree at `f919a61`.

**Assessment:** This is a capable single-machine inference control plane with substantial regression coverage. Its main engineering risk is consistency across configuration files, long-running operations, subprocesses, and browser state. Prioritize correctness and recovery before adding more engines or undertaking broad structural changes. Keep a modular monolith suited to the single-machine product.

The existing engine registry, normalized model records, catalog providers, shared lifecycle helpers, and parser fixtures are useful foundations. Preserve them. Source builds, Python environments, GPU configuration, model downloads, and proxy activation have different failure modes; their common orchestration should be explicit while engine-specific behavior stays in adapters.

**Validation and scope**

- Backend: **1,483 passed, 15 skipped**, one duplicate OpenAPI operation-ID warning; Python 3.12.3. Executed in a temporary copy with a separate data directory.
- Frontend: **367 passed across 40 files**. The passing run nevertheless emitted repeated socket errors against `127.0.0.1:3000`.
- Production frontend build passed. Main JavaScript output: **1,229.51 kB minified / 305.11 kB gzip**; CSS: 138.18 kB. Build output was directed to `/tmp`.
- Targeted temporary probes reproduced lost concurrent storage updates, overwrite after YAML corruption, mutation of queued SSE events, event-loop blocking, and loss of the existing proxy configuration on replacement failure.
- No live GPU builds, model downloads, real inference, Docker build, dependency vulnerability audit, or browser end-to-end workflow was performed. Existing dependencies were used; this does not establish clean-install reproducibility.
- Concurrent edits appeared in audio conversion/proxy code and related files during the review. Those edits were left untouched and are outside the initial snapshot's test results. This document is the only repository file created by the review.

**Priority findings**

1. **P1 — Storage updates can lose data; corruption becomes an empty store.**

   Evidence: `backend/data_store.py:241` catches read failures and returns `{}`. Individual reads and writes hold a lock, but mutation methods release it between reading and saving (`:292`, `:346`, `:470`). Two concurrent `update_settings` calls, synchronized after reading, retained only one new key. Replacing a temporary settings file with malformed YAML and then updating it silently replaced the damaged document with just the new setting.

   Hold a lock across the entire read/validate/mutate/write operation. Distinguish absent files from corrupt or unreadable files; refuse mutation on corruption and preserve recovery material. Add unique temporary files, flush/fsync where durability is required, backups, and schema validation/migrations. Explicitly enforce one backend writer or use interprocess coordination; an instance-local `RLock` cannot protect multiple workers. Do not switch storage engines as a substitute for defining transaction boundaries. A transactional embedded store is a later option if durable job/config coordination requires it.

   Acceptance: concurrent independent mutations all survive; corrupt input remains untouched; injected write failures preserve the last valid document; duplicate model/version identifiers are rejected atomically.

2. **P1 — The default deployment exposes privileged management operations without authentication.**

   Evidence: `backend/main.py:187` and its router registrations provide no shared authentication boundary. Compose publishes `8080:8080` and `2000:2000`; development Uvicorn and Vite bind all interfaces. Management routes can build user-selected source repositories, import local bundles, delete models, and change credentials. CORS does not authenticate direct HTTP clients. Risk depends on host/network reachability; no remote exploitation was attempted.

   Define local-only and remote-access modes. Publish the management port on loopback by default and provide an explicit remote configuration. Require authentication for remote management, with an appropriate browser session and CSRF policy. Audit the separately published inference/proxy port, including its admin routes, rather than assuming Studio authentication covers it. Keep credentials in restricted files or environment injection and redact them from diagnostics.

   Acceptance: an unauthenticated remote client cannot invoke management mutations; the documented local workflow still works; Studio and proxy administrative surfaces have explicit, tested access policies.

3. **P1 — Slow update checks can stall the whole API.**

   Evidence: async handlers in `backend/routes/llama_versions.py:512` and `:570` invoke synchronous HTTP helpers. `backend/llama_github_refs.py:19`, `:60`, and the direct request at `backend/routes/llama_versions.py:597` omit timeouts. A stub taking 150 ms delayed a scheduled 10 ms event-loop heartbeat to 151 ms.

   Use a lifecycle-managed asynchronous HTTP client with explicit connection/read deadlines, connection limits, and bounded retries for safe operations. Move unavoidable synchronous SDK, scanning, filesystem, and subprocess work off the event loop with bounded concurrency. Audit route-to-helper call paths, not just `async def` declarations. Close the client created in `backend/llama_swap_manager.py:612` on every exit path.

   Acceptance: delayed or unavailable upstream services do not prevent status, SSE, or cancellation requests from responding; network operations terminate within a documented deadline; shutdown closes all clients.

4. **P1 — Proxy configuration replacement can remove the last working configuration.**

   Evidence: `backend/llama_swap_manager.py:369` removes the current file before `os.rename`. An injected rename failure left the config path absent. The apply flow at `:906` unloads models before generating and validating the replacement, catches all unload errors, and then continues. Audio sidecars and the main YAML are published separately.

   Render and validate a complete candidate before disrupting running models. Publish with atomic replacement without unlinking the current file first. Serialize application, associate candidates with a configuration revision, and keep the previous valid generation for rollback. Stage sidecars so old and new YAML cannot reference incompatible generations. Classify unload errors and verify the proxy has accepted the intended configuration before reporting success.

   Acceptance: render/write/reload failures preserve a usable previous generation; overlapping apply requests are serialized; a concurrent edit remains pending; success reflects proxy acceptance, not merely a file write.

5. **P1 — Progress transport can consume unbounded memory and leave the UI stale.**

   Evidence: `backend/progress_manager.py:14` retains tasks indefinitely; `:253` creates an unbounded queue for each subscriber and sends only active tasks at connection time. `_broadcast` queues references to mutable task dictionaries. A probe created then completed a task before consumption; the queued `task_created` event already said `completed`. There is only an initial heartbeat. The frontend merges task updates and reconnects without replacing its task snapshot (`frontend/src/stores/progress.js:214`, `:225`), so completion during a disconnect can leave an old task appearing to run indefinitely. Its retry timer is not cancelled by `disconnect()`.

   Snapshot event payloads, bound queues and task retention, coalesce replaceable progress updates, and define an overflow policy that triggers explicit resynchronization. Add periodic heartbeats and either sequenced replay or an authoritative task snapshot on reconnect, including terminal outcomes. Keep the event loop as the queue owner and marshal worker-thread callbacks safely. Cancel pending frontend reconnect timers during teardown.

   Acceptance: slow clients have bounded memory cost; event payloads retain their emission-time values; reconnect after completion or restart produces correct task state; disconnect prevents later automatic reconnection.

6. **P2 — Job ownership and recovery are fragmented.**

   Evidence: builds and audio installs use detached `asyncio.create_task` calls (`backend/routes/llama_versions.py:1439`, `backend/routes/model_catalog.py:194`), downloads use FastAPI background tasks, and Python installers use `CancellableOperationManager`. Task status is primarily in memory. The generic cancellation helper terminates the immediate child and cancels its coroutine without waiting or escalating (`backend/operation_cancel.py:10`). App shutdown explicitly manages the proxy but has no common job supervisor.

   There is already partial recovery: `repair_stale_building_versions` marks abandoned builds broken when versions are listed. Extend this foundation into startup reconciliation for every operation type, rather than claiming recovery is wholly absent.

   Introduce one in-process supervisor with retained task handles, durable status, resource locks, operation IDs, and explicit states: queued, running, cancelling, cancelled, succeeded, failed, interrupted. Manage subprocess groups with bounded terminate/wait/kill behavior appropriate to the platform. Reconcile partial installs and downloads at startup. Resume only operations proven resumable; otherwise explain interruption and offer cleanup/retry. No distributed queue is needed for the current deployment model.

   Acceptance: restart during each supported operation yields a truthful recoverable state; cancellation leaves no owned descendants running; duplicate requests cannot operate concurrently on the same installation directory.

7. **P2 — Engine expansion multiplies changes across oversized modules and weak API contracts.**

   Evidence: `EnginesView.vue` is 4,306 lines, `ModelSearch.vue` 3,993, and `ModelConfig.vue` 3,848. `llama_manager.py` is 2,981 lines; the models and llama-version routers each exceed 1,900 lines. `VllmManager` inherits from the concrete `SglangManager`. Several endpoints accept unrestricted dictionaries. The suite reports duplicate OpenAPI operation IDs for the multi-method audio passthrough route.

   Extract cohesive workflows incrementally: engine installation, model acquisition, runtime configuration, and proxy application. Keep routes responsible for validation and HTTP mapping; services own use cases; adapters own engine-specific commands and discovery. Extend `EngineSpec` through explicit adapter registration instead of spreading engine-name conditionals. Reuse Python installation behavior through a neutral shared implementation or composition.

   Define Pydantic request/response envelopes and typed task events. Preserve extensible engine options in a deliberate map validated against discovered capabilities. Add a shared frontend API client and generated or checked types at boundaries, then adopt TypeScript incrementally where useful. Split views into feature components and composables without forcing engine-specific controls into an overly generic form.

   Acceptance: one engine workflow is migrated end to end with unchanged behavior; OpenAPI operation IDs are unique; adding a comparable engine no longer requires editing unrelated workflow implementations.

8. **P2 — Passing unit tests do not yet establish release or lifecycle reliability.**

   Evidence: `backend/tests/conftest.py:16` returns `TestClient(app)` without entering a context manager, despite its lifespan claim. The shared client fixture also lacks a globally isolated store. SSE smoke coverage is skipped because the synchronous client blocks. Frontend tests pass while making unexpected socket attempts. Python requirements are entirely unpinned. Test CI and Docker publishing are independent workflows; publishing has no dependency on successful tests for the same commit. The Docker workflow builds the frontend, but regular frontend test CI does not. Fork PR image builds also need a deliberate policy for the required license secret.

   Isolate state by default, enter the lifespan explicitly in lifecycle tests, and fail unit tests on unexpected network access. Add async transport tests and a small browser suite for install/configure/apply/start/stop, cancellation, and reconnect. Use tiny fake engine executables to exercise actual process creation and termination on CPU CI. Maintain a separate opt-in live compatibility lane for real engines and GPUs.

   Lock the application Python dependency graph and declare supported Python/Node versions. Gate published images on tests for the same revision; verify locked installs, production builds, and container startup. Add formatting/lint/type checks incrementally and make schema warnings actionable. Preserve the existing extensive parser and domain regression fixtures.

   Acceptance: tests cannot write to a developer's real data directory; lifecycle tests actually execute startup/shutdown; no unexpected network errors are tolerated; failing tests prevent release publication; a clean locked install passes.

**Architecture and performance follow-through**

The intended dependency direction is HTTP/UI → application workflows → domain contracts → storage/process/network adapters. Construct shared clients, store, supervisor, and adapters in application lifespan and inject them into workflows. Avoid import-time resource creation and global singleton resets as the testing mechanism.

Frontend performance has a measured opportunity: all major views are eagerly imported in `frontend/src/router/index.js:1`, producing the 1.23 MB entry bundle. Lazy-load routes and heavyweight feature panels. Restore content-hashed assets instead of `Date.now()` names in `frontend/vite.config.mjs`; serve immutable caching for hashed assets and revalidation for HTML. Remove custom HTML query-string rewriting in `backend/main.py`. Measure entry size and route loading before setting a budget; an initial goal is at least a 30% reduction in entry JavaScript without slower common navigation.

Add structured operation/task/model/engine identifiers to logs and a small set of metrics: operation duration/failure, event-loop delay, queue depth, cancellation latency, and config revision. Separate cheap liveness from readiness and detailed status. Today `/api/status` returns HTTP 200 even when proxy health is false, while Docker uses it as its health check. Preserve healthy first-run behavior when no engine is installed, but expose required runtime failures clearly.

**Delivery order and completion criteria**

The effort estimates below are rough engineer-weeks for an engineer familiar with the repository, including focused regression tests; they are not delivery commitments. Complete the first phase before broad refactoring. CI work can proceed alongside subsequent phases.

| Phase | Scope | Estimate | Exit criteria |
| --- | --- | --- | --- |
| 1: Protect state and access | Transactional YAML mutations and corruption handling; safe proxy replacement; network deadlines; loopback defaults and remote access policy | 2–3 | Reproduced data-loss and replacement failures have regression tests; slow upstream cannot stall API; management exposure is explicit |
| 2: Own operations | Common supervisor, process-tree cancellation, durable outcomes, startup reconciliation, bounded SSE and reconnect snapshots | 2–3 | Restart/cancel/reconnect scenarios pass for builds, downloads, and installs; memory is bounded |
| 3: Make releases trustworthy | Isolated lifecycle tests, unexpected-network failures, fake-engine integration, browser smoke, dependency locks, publish gates | 1–2 | A release is built from a passing, reproducible revision; startup and critical workflows are exercised |
| 4: Reduce change cost | Migrate one engine workflow to service/adapter boundaries; typed API/events; split the largest views; route splitting and caching | 2–3 | Pilot removes duplicated orchestration, preserves behavior, and reduces initial bundle size; remaining migrations have a repeatable pattern |

Start with separate, reviewable changes for storage correctness, proxy atomic replacement, and update-check timeouts. Each should include the corresponding failure-injection regression. Then improve operation ownership and transport together, since the browser needs an authoritative job state to recover correctly. Use the existing tests to protect behavior while extracting architecture; avoid a repository-wide rewrite.
