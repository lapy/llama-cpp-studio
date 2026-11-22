# Single-operator hardening release plan

Status: **implemented — release candidate**, 2026-11-22. Branch:
`feat/single-operator-hardening`.

## Product boundary

Studio is a **single-operator, single-host inference workspace**. One technical
operator owns the machine, the models, the engines, the gateway and the data
directory. The release optimizes for operational safety, truthful state, quick
recovery and a coherent UI.

Non-goals for this release: multi-tenant RBAC/SSO, HA/clustering, distributed
quota storage, multi-host scheduling, a full Strata model-preparation stack, and
a general plugin SDK.

## Delivery status

| ID | Delivered | Evidence / caveat |
| --- | --- | --- |
| WS1 | Proxy settings (`off`/`warn`/`enforce`, margin, per-model estimates), pre-spawn per-device guard, reservations, 409 `INSUFFICIENT_VRAM`, Proxy card and activity/state fields. | Fake-GPU tests and route tests; no real NVIDIA admission run (shared V100s were occupied during the release). |
| WS2 | `/api/storage/inventory`, preview/apply cleanup with containment and reference guards, System → Storage panel. | Tmp-tree backend tests, route tests, store/component tests. |
| WS3 | Generic native lifecycle for `native_host` definitions; inert Strata registration retired with a planned-integration contract. | Temp native definition + stub host tests; no legacy route added. |
| WS4 | `/api/metrics`, bounded JSONL mutation audit, benchmark warmup/repeats and aggregates, GPU smoke-matrix script. | Route/middleware/benchmark tests; no live GPU scrape. |
| WS5 | UI foundation and all page migrations; guided first-model flow; Names & routing summary; shared sections/tabs/toolbars/states; mobile section-spacing fix. | Chromium and Firefox journeys, axe standards run, 320/390/1440 screenshot review. |
| WS6 | Single-operator deployment guide, README/features/API docs, generated artifacts, release evidence. | See the release evidence section below. |

## Release evidence

- Backend: full suite green on the dev box.
- Frontend: unit suite, production build, Chromium and Firefox browser suites, and the accessibility standards journey green.
- Gates: OpenAPI, engine parity, legacy routes, build defaults, version, Python/frontend lint, formatting, style, API typecheck, module-size and engine-literal ratchets green.
- Hardware: ReVolta install/discovery and the storage/GPU-admission code paths were exercised with fixtures; real multi-model GPU admission and Strata remain unverified by design (no dedicated host, shared GPUs unavailable).

## Workstreams

| ID | Workstream | Owner surface | Branch |
| --- | --- | --- | --- |
| WS1 | GPU-aware admission and truthful GPU states | proxy backend, ProxyView | `ws/gpu-admission` |
| WS2 | Storage governance | storage service/routes, System → Storage | `ws/storage` |
| WS3 | Generic native engine lifecycle; retire inert Strata registration | engines/routes/tests | `ws/engine-native` |
| WS4 | Metrics, audit, benchmark, GPU validation tooling | routes/middleware/scripts | `ws/observability` |
| WS5 | UI/UX alignment foundation and page migrations | styles, shared components, views | `ws/ui-foundation`, then `ws/ui-pages-*` |
| WS6 | Single-operator docs and release evidence | docs/README | lead |

Workstreams are implemented in separate git worktrees from this branch. The lead
integrates them one at a time, resolves shared-file conflicts, runs the full
gates, and records evidence. No workstream edits another workstream's owned new
files.

## WS1 — GPU-aware admission

Goal: a start that cannot fit should be refused with a truthful reason before a
process is spawned; every admission decision must be visible.

Contract (additive):

- Proxy settings (live-applied, no restart):
  - `vram_admission`: `off` (default) | `warn` | `enforce`.
  - `vram_safety_margin_mib`: int 0..1048576, default 1024, per selected GPU.
  - `vram_estimates`: map proxy model id -> MiB 1..4194304, max 500 entries.
- Estimate precedence: settings override (high) > last uncontended observed
  ready-time used-memory delta (medium) > model file-size weights-only lower
  bound (low) > unknown.
- Enforce rejects only a known deficit: override or observed estimate over free
  memory minus margin and in-flight reservations. Derived-only deficits warn.
  Unknown/probe failure admits with `unknown` decision. NVIDIA only in this
  release; CPU mode skips.
- Device set comes from the compiled LaunchSpec (`CUDA_VISIBLE_DEVICES`), not
  engine-id dispatch. Numeric ids map to probe indexes; UUIDs map through the
  cached GPU inventory; unmappable ids make the check unknown.
- New `VramAdmissionError` maps to HTTP 409 with
  `{error, code: "INSUFFICIENT_VRAM", detail}` on inference cold start and
  preload/reload. Existing 429/502 bodies stay byte-compatible.
- Additive observation: `/api/proxy.vram` (mode, probe, last_admission,
  estimates_configured), catalog rows `vram_estimate`, running rows `vram`,
  activity types `vram_warning|vram_rejected|vram_unverified`.
- UI: Proxy settings card (mode, margin, per-model overrides, probe status,
  last admission), Overview warning, Events severity mapping, running detail.
  No new model-config field.
- Docs: proxy operations, features, API reference, runtime-auto-fit boundary.

Acceptance: fake-GPU unit tests for estimate precedence, reservation release,
warn/enforce/unknown, probe failure, CPU/UUID selection, registry no-spawn on
denial, API 409, live apply; ProxyView component tests; no synchronous probe on
the event loop.

## WS2 — Storage governance

Goal: one place to see disk use and safely reclaim what Studio owns.

Contract (additive):

- `GET /api/storage/inventory?refresh=false` -> filesystem, scan quality,
  seven disjoint categories (`models`, `hf_cache`, `engine_installs`,
  `cuda_toolkits`, `jit_caches`, `history_backups`, `accounting_db`), units with
  bytes, `cleanable`, `blocked {code,message}`, `manage_path`, `outside_data_root`.
- `POST /api/storage/cleanup/preview` with `category`, `unit_ids`,
  `keep_history` -> plan id (sha256), reclaimable bytes, units, refused units,
  confirmation phrase. The API never accepts a path.
- `POST /api/storage/cleanup/apply` with the plan id and `confirm: true` ->
  completed/partial/refused with freed bytes and per-unit failures. Stale plan
  409; scope caps 409; in-flight GC 409; confirmation required 400.
- Cleanable: orphan HF repo dirs, model payload dirs with no catalog row, JIT
  cache roots, CUDA installer cache, history pruning. Report-only with deep
  links: catalog models, engine installs, CUDA toolkits, accounting DB. No
  second destructive path for what another surface owns.
- Guards: realpath containment, no symlink following, never delete a root,
  running model / retained launch reference / active download or build refusal.
  Blocking scan/delete via `run_blocking`; 10 s inventory TTL cache; bounded
  scan and node caps.
- UI: System gains a fifth section `Storage` (`/system#storage`) using the
  shared page/section patterns: filesystem summary, category table/cards with
  share bars, per-unit disclosure, blocked reasons and manage links, confirm
  dialog with phrase, partial-failure reporting, accessible live status.
- Docs: features System/Storage, API reference routes and workflow, engineering
  map row.

Acceptance: inventory disjointness/containment tests, cleanup refusals and
digest, preview→apply with tmp trees, route tests for 409/400, store/component
tests, no real user data or downloads.

## WS3 — Generic native engine lifecycle; retire inert Strata

Goal: a native engine definition with `native_host` and a lifecycle block gets
generic `/api/engines/{id}` verbs without a legacy prefix. Remove the Strata
registration until a real host exists (documented in `docs/strata-engine.md` as
planned, not shipped).

Contract:

- `build_definition_lifecycle` resolves a native installer from
  `get_native_installer(engine_id)` when `definition.native_host` is set; the
  venv `GenericVenvInstaller` remains for python-venv definitions.
- Verb map: install (release), status, cancel, check-updates when declared,
  build-settings, remove; `install-source` only when a source recipe exists,
  otherwise 405.
- `register_definition_lifecycles` registers the native record with the same
  one-record rule and startup verification.
- Tests: temp native definition + stub host proves generic install/status/cancel
  and 405s; existing venv tests unchanged.
- Remove `definitions/strata.yaml`, `docs/strata-engine.md` runtime claims,
  README/features rows, engine counts/sets/tests, data dirs and parity entries.
  Keep a short planned-integration note in `docs/strata-engine.md` (no
  registered engine) so the work is not lost.
- Update engine onboarding docs to describe the generic native path.

Acceptance: test suite/counts updated, no inert registry engine, no new legacy
route, no engine-ID literal growth, parity inventory regenerated.

## WS4 — Observability, audit, benchmark, validation

- `GET /api/metrics` (management auth): Prometheus text with app info/uptime,
  proxy request counters/gauges, loaded/starting model counts, operation
  counts, persistence queue, free disk, GPU free/total memory. Values read from
  existing services; unknown is omitted, not zero. No blocking probes on the
  loop; use cached observations.
- Audit: one mutation middleware appends JSONL under
  `<data>/logs/audit.jsonl` for non-GET `/api/*` requests with timestamp,
  method, path (no query values), status, duration, actor kind
  (`local`, `token`, `session`), and request id. Rotate at 5 MiB / keep 5;
  never log bodies/headers/secrets. Tests with TestClient and a temp data dir.
- Benchmark: additive `warmup` (0-3) and `repeats` (1-5) request fields; keep
  existing response fields; add per-run timings and aggregate
  min/p50/p95/max; document that GPU memory is sampled and results are not
  normalized across hardware.
- `scripts/gpu-smoke-matrix.py`: stdlib-only runner that reads a JSON matrix
  (base URL, key env, model IDs, prompts), calls `/v1/models` and chat
  completions, records model/alias, status, latency, usage if present, and
  writes JSON. No CI hardware assumption; docs explain manual use.

Acceptance: route/middleware/benchmark tests, script `--help`/dry-run, docs.

## WS5 — UI/UX alignment

Every page must look and behave like one product. The existing
`docs/frontend-standards.md` is the accessibility contract; this workstream adds
the visual and structural consistency contract.

### Shared rules

1. **Page shell.** Every routed page uses `page-shell` (or `--wide` for
   table-heavy workspaces), one `PageHeader` with title, description, optional
   meta and actions. One primary action per page.
2. **Sections.** All page sections use one `PageSection` pattern: heading,
   optional description, optional actions, consistent padding/gap/border.
   No page-local `system-card`/`proxy-card`/`ev-section` lookalikes after
   migration; styles become shared classes.
3. **State.** Loading uses `LoadingState`; empty uses `EmptyState` with a next
   action; stale/error banners use the shared `state-banner` with retry; no
   silent blank panels.
4. **Controls.** PrimeVue components with shared sizes; one primary button per
   section; icon-only buttons have aria-labels; labels are sentence case and
   consistent (`Save`, `Save settings`, `Retry`, `Refresh`).
5. **Tables and toolbars.** One toolbar pattern (search, filters, actions),
   one table density, action columns labelled, empty/loading inside the table
   shell, horizontal scroll contained.
6. **Forms.** Shared field spacing, label/hint/error placement, two-column at
   desktop and one at mobile, consistent save bar and unsaved-change prompt.
7. **Dialogs.** Shared header/body/footer structure, width scale, focus return,
   destructive confirmation and one-time secret treatment.
8. **Type and color.** Page h1, section h2, sub h3; tokens only; both themes;
   no hardcoded colors; consistent icon sizes and empty-state illustrations.
9. **Responsive.** No horizontal overflow at 320/390 px; actions remain
   reachable; tables scroll locally; touch targets meet the standards.
10. **Usefulness.** Every page answers: what is this, what is the primary
    action, what state am I in, what do I do next. Empty/failed states always
    offer the next step.

### Page checklist

| Page | Required outcome |
| --- | --- |
| Models | Clear library/table switch, toolbar, first-model CTA, running/attention states, consistent row actions |
| Search | Filter/toolbar consistency, result cards/table, empty/loading/error, import action |
| ModelConfig | Section nav (Engine / Runtime / Names / Launch), Names & routing summary, consistent save/review bar |
| Engines | Card overview + version table consistency, install/build dialogs, filters, empty states |
| Proxy | Tabs, overview/settings/keys, shared sections, GPU admission card, request table |
| System | Shared section tabs, Hardware/Storage/Diagnostics/Backup/History visual parity |
| Audio | Model/context header, task tabs/panels, result states, voice/reference controls |
| NotFound | Shared shell, meaningful heading, route back to Models |

### Implementation sequence

1. `ws/ui-foundation`: add shared `PageSection` (or equivalent classes),
   typography/spacing tokens, `SectionTabs`/toolbar/state primitives, section
   nav; migrate NotFound and the shell as reference; write
   `docs/ui-alignment.md` with the audit and checklist.
2. Feature agents (WS1/WS2) build their panels with the shared patterns.
3. `ws/ui-pages-models`, `ws/ui-pages-engines`, `ws/ui-pages-proxy-audio`
   migrate their pages after the feature branches land, using the reference
   page and screenshot review.
4. Run `frontend/e2e/ui-refactor.spec.mjs` for refreshed screenshots, compare
   every page at desktop/mobile and both themes, and record intentional changes.

Acceptance: unit tests, browser journeys, a11y, no page-local duplicate section
styles, no horizontal overflow, and a screenshot review table in the PR.

## WS6 — Documentation and evidence

- `docs/single-operator-deployment.md`: boundary, local vs remote, TLS/reverse
  proxy, token/session handling, backup/full-data distinction, GPU/toolkit
  matrix, storage budgets, recovery.
- README/features updates for every user-visible change.
- Final evidence table: full suites, build, browser/a11y, ratchets, hardware
  validation or exact blockage.

## Integration protocol

1. Each workstream branches from `feat/single-operator-hardening` in its own
   worktree and commits conventional commits.
2. The lead merges/cherry-picks in order WS3 → WS1 → WS2 → WS4 → WS5, resolving
   shared files (`backend/main.py`, docs, tests sets).
3. After each merge: targeted tests, then full suite on the dev box.
4. Regenerate OpenAPI, parity/legacy inventories, module budgets and literal
   caps once at the end; never hand-edit generated artifacts.
5. Only lower module budgets after extraction. No new legacy route.
6. Record unverified hardware claims explicitly.

## Deferred

- RBAC/SSO/OIDC, audit user identity beyond actor kind, HA/multi-node.
- AMD VRAM admission, MIG identity, peak/calibrated VRAM, automatic eviction.
- Full Strata Terminal host/model preparation.
- Full SDK generation and inference OpenAPI; a versioned public SDK.
- Multi-GPU benchmark normalization.
