**Project quality assessment and improvement plan — 4 October 2026**

Reviewed commit `2323179`, starting with a clean working tree. This assessment covers the current implementation; the September assessments are historical context, not a list of still-open defects.

**Assessment**

Studio is a capable local inference control plane with substantial domain knowledge and regression coverage. Its strongest qualities are engine integration, explicit operation management, and safeguards around configuration persistence. Its main weaknesses are inconsistent runtime truth in the UI, remaining synchronous work in async request paths, and the cost of maintaining several very large workflow modules. The next investment should be reliability and simplification of everyday workflows, followed by targeted performance and structural improvements.

| Area | Assessment | Evidence |
| --- | --- | --- |
| Product capability | Strong for experienced local-inference users | Nine engine variants, model discovery, runtime configuration, audio workflows, task cancellation, launch manifests, and client handoff |
| Code quality | Strong domain coverage; uneven modularity and contracts | Extensive tests and useful service/adapter boundaries coexist with 4,000-line Vue views, large routers, global singletons, and dictionary-based API contracts |
| UX | Functional and substantially improved; operational state still needs attention | Draft guards, recovery states, filtering, task history, and responsive layouts are present; readiness guidance and proxy failure handling can contradict reality |
| Performance | Promising for small libraries; several confirmed avoidable costs | Fast local build and route splitting, but blocking metadata calls, leaked polling, whole-document persistence, and uncompressed static delivery |
| Release confidence | Good unit/component foundation; incomplete integration evidence | Passing suites and publication test gates; no committed browser suite or container-startup verification in the reviewed workflows |

**What was verified**

- Backend: **1,606 passed, 7 skipped**, 27.14 seconds, using `.venv/bin/python -m pytest backend/tests -q --tb=short`.
- Frontend: **388 passed across 42 files**, 29.91 seconds, using `npm run test:frontend`.
- Production build: passed in **1.06 seconds**, using `npm run build` with the existing local configuration.
- Chromium inspection of the production frontend with controlled API fixtures: empty/populated/failed library, Search, Engines, and Audio; desktop width 1440 and mobile width 390. Inspected screenshots and exercised navigation during an in-flight catalog poll. No uncaught page exceptions or horizontal overflow occurred in these sampled screens.
- Isolated Python probes exercised a delayed metadata lookup, catalog sizes of 10/100/1,000 models, operation histories of 100/1,000/10,000 records, proxy outage handling, and static asset response headers. All storage probes used temporary data roots.

These are local observations, not production SLOs. The installed environment uses Python 3.12.3 and Node 20.20.2; the repository declares Node 24+, so the frontend results do not establish a clean supported-version installation. No real model downloads, engine installations, GPU inference, Docker image build/startup, fresh dependency installation, full accessibility audit, or live remote deployment were performed. Configuration and populated audio workflows were reviewed in source/tests rather than exercised end to end in the browser. The browser fixtures deliberately closed SSE responses, so their “Reconnecting” footer is not a finding against the application.

**Preserve the improvements already implemented**

The current code includes transactional YAML mutation with atomic replacement/backups and corruption refusal; supervised operations with startup reconciliation and process cleanup; bounded SSE queues and reconnect snapshots; proxy candidate validation and launch-apply recovery; loopback defaults and an explicit remote authentication policy; lazy routes and content-hashed assets; unsaved-edit protection and draft restoration; persistent request-error states; library filters/list layout; and an Activity panel.

Both Docker publication test jobs gate the image build. Dependencies are pinned/locked. These should not be proposed again as missing foundations. Continue extending the existing design rather than undertaking a broad rewrite or immediately replacing YAML with a database.

**Priority 1: remove remaining event-loop blocking**

Confirmed in [model routes](../backend/routes/models.py): `get_search_file_sizes()` calls the synchronous `get_accurate_file_sizes()` directly. That reaches Hugging Face SDK I/O through [hub helpers](../backend/models/hub.py). `get_quantization_sizes_from_hf()` is declared async but also performs synchronous SDK calls, and the route's fallback executes sequential `requests.head(..., timeout=10)` calls inside an async handler.

A controlled 300 ms blocking metadata stub produced a **340 ms maximum gap** in a concurrent 10 ms heartbeat. This demonstrates that a slow lookup can delay unrelated API requests and progress delivery. It is not a measurement of Hugging Face network latency.

Move blocking SDK work into a bounded worker path, use async HTTP for HEAD fallbacks, set operation deadlines, and bound fallback concurrency. Audit actual call chains rather than treating every synchronous helper as a defect; several existing paths already offload correctly. Share the existing async client where appropriate.

Acceptance: with an upstream lookup delayed by two seconds, unrelated liveness requests and SSE heartbeats remain responsive; concurrency is bounded; timeout/cancellation outcomes are visible. Start with a CPU-only regression test against these two metadata routes.

**Priority 1: make runtime status truthful and recoverable**

The [proxy client](../backend/proxy/llama_swap/client.py) returns an empty list on a running-model lookup failure. The [model-list route](../backend/routes/models.py) also catches lookup errors and returns ordinary model records with `is_active: false` and no status-quality indicator. A probe confirmed this result. The library consequently cannot distinguish an unreachable proxy from a verified stopped model.

The [app shell](../frontend/src/App.vue) refreshes system status on startup, while its visibility/task callbacks refresh the stale-config flag. Normal library polling does not refresh that system-status snapshot. A green header can therefore outlive the health observation it represents. In [ModelConfig](../frontend/src/views/ModelConfig.vue), “Pending changes” derives from the deployment-wide stale flag, and “In use” means no pending flag, even for a stopped model. Those are different concepts from this model's saved, published, and running generations.

Introduce explicit runtime states including unknown/unreachable, last successful observation time, and model-specific published/running revision information. Preserve the last known state during transient outages with a stale indication. Refresh shared health through one owner. Add recovery for management-session expiry instead of allowing unrelated requests to fail indefinitely after a backend restart.

Acceptance: disconnecting the proxy never silently changes a running model to a verified stopped state; a header cannot claim current health from an old observation; changing model A does not label unchanged model B pending; stopped-but-published settings are described accurately.

**Priority 1: fix poll ownership and request lifetime**

In [ModelLibrary](../frontend/src/views/ModelLibrary.vue), `queueCatalogPoll()` schedules another timer after awaiting `refreshCatalogs()`. Unmount clears the current timer but does not prevent an already-running callback from scheduling its successor. Browser reproduction: hold the second catalog request, navigate to Audio, release it, and observe a subsequent `/api/models` **and `/api/models/safetensors`** pair from the departed library. Audio's own initial models fetch was accounted for separately.

Use explicit disposal or a shared query owner, cancel view-owned requests where safe, and stop rescheduling after unmount. Pause routine polling in hidden tabs and retain fast polling only while a model is transitioning. The library displays the unified `/api/models` result yet also polls the safetensors inventory every five seconds; remove that duplicate request where its data is unused. The [shared client](../frontend/src/api/client.js) currently supplies credential/CSRF behavior but no common deadline or response recovery policy.

Acceptance: navigation during a delayed request leaves no library timer running; hidden tabs produce no routine library polls; returning to the view refreshes once; stalled control-plane reads eventually expose Retry. Do not automatically retry mutations whose outcome is uncertain.

**Priority 2: make first-run guidance accurate and compact**

[SetupChecklist](../frontend/src/components/common/SetupChecklist.vue) derives readiness from whether any hard-coded version array is nonempty. It omits `unslothLlamaVersions`, does not require an active/runnable version, and does not initiate loading those arrays when opening Models directly. “Review configuration” becomes complete simply because a model exists.

In the populated browser fixture, Studio showed a running model while still directing the user to install an engine. At 390 px wide, the token warning and six-step checklist pushed the model card below the initial 1,000 px viewport. The responsive layout fits, but the information hierarchy delays the daily task.

Drive guidance from the existing engine descriptors (`runnable`, `active_version`, capabilities), fetched independently of visiting Engines. Show one current next action, let users expand the full checklist, and collapse onboarding after verified success. Make token setup contextual to gated access. Group installed/runnable engines first, distinguish “Manage” from “Activate” for an already-active version, and use task/hardware compatibility to narrow engine choices while retaining manual selection.

Acceptance: direct entry to Models works for every supported engine, including Unsloth; inactive/broken installs are not considered ready; downloading a model does not imply its configuration has been reviewed; on a small screen, returning users reach model controls without scrolling through onboarding.

**Priority 2: reduce storage and catalog costs before they accumulate**

[DataStore](../backend/data_store.py) reparses YAML on reads and rewrites an entire document with backup/fsync on mutations. [Operation supervision](../backend/operations/supervisor.py) persists through those synchronous calls. The in-memory progress history has a retention cap, but durable operation rows are only removed through explicit dismissal; automatic in-memory eviction does not prune the durable document.

Local synthetic results, with proxy network work stubbed out:

| Probe | Size | Observed duration |
| --- | --- | --- |
| Model-list handler, 15 iterations | 10 models | 0.68 ms median |
| Model-list handler, 15 iterations | 100 models | 6.36 ms median |
| Model-list handler, 15 iterations | 1,000 models | 157.87 ms median |
| Durable operation upsert, one sample | 100 existing rows | 16.91 ms |
| Durable operation upsert, one sample | 1,000 existing rows | 134.18 ms |
| Durable operation upsert, one sample | 10,000 existing rows | 1,370.63 ms |

The 1,000-model response was about **1.16 MB of JSON**, including per-model configuration. These handler timings exclude HTTP serialization/transport and are stress probes, not representative claims about a typical user's library. Operation samples include local filesystem costs and need repeated benchmarking before setting final budgets.

Add an explicit age/count policy for terminal operation records, preserving active work and recoverable failures. Move expensive persistence off the event loop without weakening serialization/durability. Cache parsed documents with revision/invalidation semantics, return compact library summaries, and fetch configuration on demand. Reuse proxy HTTP connections; its current client creates fresh `httpx.AsyncClient` instances for routine calls. Add pagination or rendering virtualization only if larger-library browser measurements justify it.

Acceptance: history retention is automatic and survives restart; a 1,000-model fixture does not create long event-loop stalls; catalog responses avoid shipping detailed configs on every status poll; concurrent edits and failed writes retain their existing safety guarantees.

**Priority 2: finish delivery performance and client handoff**

The production entry chunk is **464.28 kB raw / 110.96 kB estimated gzip**, and HTML preloads another **130.90 kB raw / 48.10 kB estimated gzip** shared chunk before route dependencies. ModelConfig's route chunk is another 238.85 kB raw. Lazy loading already exists; do not count it as future work.

An ASGI request advertising gzip/Brotli received the entry file with `Content-Length: 464289` and no `Content-Encoding`. Immutable caching is correctly present. Serve negotiated precompressed static assets, or compress them at the deployment boundary, while preserving immediate SSE delivery. Lazy-load heavy engine/configuration dialogs and establish a measured first-route budget before further splitting. Build-output gzip estimates do not mean users currently receive compressed assets.

[ConnectDialog](../frontend/src/components/common/ConnectDialog.vue) always constructs a chat-completions request and derives its scheme/host from the management page. That does not cover embeddings, alternate external inference URLs, or an HTTPS management frontend whose proxy port serves HTTP. These are source-confirmed assumptions; live deployment failures were not exercised. Use capability-specific request examples, a configured public inference URL, bounded test requests, and a management-origin test relay where appropriate.

Acceptance: static compression is verified over HTTP; cold Models transfer falls from its recorded baseline without slower ordinary navigation; text and embedding models get valid examples; remote endpoint construction is tested separately from local defaults.

**Priority 2: reduce change cost with one complete extraction at a time**

Current sizes include EnginesView **4,549 lines**, ModelConfig **4,423**, ModelSearch **4,264**, the native engine manager **3,082**, model routes **2,189**, and llama-version routes **1,928**. Line counts alone are not defects, but these files mix rendering, transport, workflow state, validation, and engine-specific branching. The checklist's missing engine is a concrete example of duplicated capability knowledge drifting.

First extract library runtime querying and engine readiness, then one engine installation workflow from view through API/service/adapter. Keep engine-specific controls explicit. Define checked request/response models for status, catalog summaries, launch plans, and task events; generate frontend types from OpenAPI or introduce boundary checks, then adopt TypeScript incrementally. The backend has an engine registry and the frontend has a shared client already; consolidate usage rather than adding competing abstractions.

Acceptance: adding an engine uses shared descriptors for readiness and selection; the migrated workflow has one owner for requests/state; contract changes fail CI; behavior remains covered while files are decomposed. Do not split modules merely to satisfy a line-count threshold.

**Priority 1–2: turn test breadth into workflow and release confidence**

The suites are valuable, especially parser, engine, cancellation, and persistence regressions. Frontend component tests frequently replace routers, stores, and PrimeVue controls, so passing tests do not establish real browser behavior—the polling race illustrates the gap. The reviewed workflows lack browser journeys and container boot/readiness verification. Production frontend builds are conditional on a license secret, leaving fork PRs without that check. Lint/type tools are listed, but the package scripts and CI do not enforce lint/type gates.

Add a small browser suite for direct-entry readiness, save/leave/restore, apply/start/stop, failed/stale data, reconnect, and navigation during active requests. Pair it with actual fake-engine subprocess tests and a container smoke test for empty startup, readiness, shutdown, and configured network boundaries. Keep GPU/live-engine compatibility in a separate opt-in lane. Give fork PRs an explicit build-validation policy; keep licensed production verification mandatory for release revisions. Ratchet lint/type coverage around changed boundaries instead of requiring repository-wide cleanup first.

Initial acceptance: a release candidate passes the supported Node/Python matrix, browser journeys, and container smoke; tests cover the newly reproduced failures; no production build is silently omitted from release validation. Browser checks should include keyboard navigation, focus after dialogs, accessible control names, and mobile screenshots rather than inferring accessibility from markup alone.

**Delivery plan**

Effort is an approximate range for one engineer familiar with the project, including focused regression coverage. Sequence by dependencies and observed failures; revise after the first milestone.

| Milestone | Work | Effort | Completion evidence |
| --- | --- | --- | --- |
| 1. Remove observable reliability defects | Offload metadata I/O; repair poll teardown; represent unreachable runtime state; add corresponding regressions | 4–6 days | Slow upstream leaves control plane responsive; navigation leak reproduced before/fixed after; outages preserve truthful UI state |
| 2. Make the core journey dependable | Descriptor-based readiness; compact onboarding; per-model saved/published/running labels; capability-aware Connect; initial browser journeys | 4–7 days | Direct entry and install/configure/apply/start/connect paths work under fixture success, delay, failure, and recovery |
| 3. Control recurring performance costs | Durable history retention; bounded/offloaded persistence; compact cached catalog; shared proxy transport; static compression and budgets | 4–7 days | Repeatable benchmarks at agreed library/history sizes; measured event-loop and transfer improvements without persistence regressions |
| 4. Make improvements sustainable | Extract one engine workflow; checked API/event boundaries; incremental lint/type checks; supported-version/container/release validation | 6–10 days | End-to-end migrated workflow and release gates pass; further engine migrations follow an established pattern |

Start the browser and container harness work during milestone 1 so later changes can use it. The full plan is roughly **four to six engineering weeks**, subject to integration complexity. The first useful deliverable is a small set of fixes for the confirmed blocking-I/O, polling, and runtime-state problems, not a feature expansion.

**Evidence artifacts from this review**

Raw local artifacts are retained under `/tmp`: `studio-assessment-backend.log`, `studio-assessment-frontend.log`, `studio-assessment-build.log`, `studio-assessment-probes.py`, `studio-assessment-probes.log`, `studio-assessment-browser.cjs`, and `studio-assessment-browser.json`, plus `studio-*.png` screenshots. These temporary paths are not durable repository dependencies. The important measurements, limitations, and reproduction descriptions are recorded above. No application source or user model/configuration data was changed by this review.
