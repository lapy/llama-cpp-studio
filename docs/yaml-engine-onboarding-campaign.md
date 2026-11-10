# YAML Engine Onboarding — Campaign Plan (decisions signed off 2026-11-02)

> **Historical evidence — not the release entry point.** Start at the
> [delivery map](delivery-map.md); the [change inventory](change-inventory.md)
> maps this work to the review series.
>
> **Retired 2026-11-10:** the `yamlcraft` fixture persona and its harness tests
> were removed once the YAML definitions were authoritative. Kept as the design
> record.

Goal: deliver the plan's central promise — *"Add an engine by adding one
validated YAML definition, without adding an installer class, runtime
compiler branch, API router, frontend component, or engine-ID allowlist"*
(`config-driven-engine-refactoring-plan.md`, Objective and scope) — and
PROVE it with one unfamiliar fixture engine before any further migration
accumulates. Companion: `engine-refactor-handoff.md` (shipped state),
`engine-surface-contract-matrix.md` §8 (tranche log). The plan document
itself stays untouched.

## 1. Ground truth (survey of 2026-11-02)

Already generic enough: definition schema v2 (49-primitive vocabulary,
typed conditions, recipes), loader/validator, async executor with the
cleanup guarantee, 20 shared primitive handlers, the generic lifecycle
router (8/9 engines), `/api/engines` descriptors with server-win picker
behavior, engine-ID-agnostic action admission.

Exactly FOUR code touchpoints block YAML-only onboarding:

| # | Blocker | Location |
| --- | --- | --- |
| B1 | Startup refuses any YAML engine absent from (or divergent from) the Python registry; exporter separately rejects "unexpected" YAML files | `definitions/startup_check.py:38-50`; `scripts/export-engine-definitions.py:1682-1685` |
| B2 | `LIFECYCLE` registration is Python-call-site only — installer class, update source, url prefix, build-settings defaults all code-constructed | `routes/engine_versions.py:88-90`, `engines/adapters.py:40-46`, `venv_install_settings.py:9-51` |
| B3 | Recipe handlers assume an `EngineInstaller`-shaped object (`_run_pip`/`_ensure_venv`); no generic implementation exists for a data-only engine | `definitions/primitives.py:169,223,531-533` |
| B4 | Frontend `registry.js` is a second hardcoded identity registry; the server descriptor payload feeds only picker options — routing/status/verbs throw for unknown ids | `frontend/src/engines/registry.js:42-7,211-224,231-236` |

Also open (from the plan, not blockers for the control-plane proof but
part of the full proof): profiles compilation is shadow-only (launch
still runs `launch_spec.py` branches), and install snapshots/definition
digests do not exist yet (plan "Immutable definitions": record digest +
recipe version + resolved inputs per installation; legacy compat map).

## 2. Definition of done — the fixture engine `yamlcraft`

A hand-authored `backend/engines/definitions/yamlcraft.yaml` with
**zero other source edits** must produce:

| Gate | Requirement | Verifier |
| --- | --- | --- |
| G1 startup | app boots with `yamlcraft` definitions-authoritative; the nine migrated engines keep byte-identical `to_api_dict()` | existing equality pins pass UNCHANGED; new startup tests |
| G2 API | `/api/engines/yamlcraft/{status,check-updates,build-settings,install,install-source,cancel}` behave like a venv engine (same contracts, 405 matrix where unimplemented) | generic-router pins, parameterized over the fixture |
| G3 install | recipe executes end-to-end in the offline test harness using ONLY the 20 shared primitives (+ at most one newly shared generic primitive); admission/confirmation free via LIFECYCLE | install-equivalence tests |
| G4 snapshots | every install row records definition digest + recipe version + resolved inputs; existing rows resolve through the explicit legacy mapping; editing YAML cannot reinterpret an old install | new pin set |
| G5 UI | picker card + dialog + model-compat presence come from the server descriptor alone; no new engine-ID literal anywhere | architecture guard test: `yamlcraft` must appear in ZERO non-test, non-YAML source files (grep ratchet) |
| G6 preservation | all existing gates stay green across every slice: pytest (2168+), vitest (593+), ratchet, ruff 44, prettier, openapi, boot smoke | full battery per slice |

FULL proof per plan step 9 additionally requires scan/**launch**/model
compatibility through YAML — that depends on the runtime/discovery
(profiles) migration, which is deliberately OUT of this campaign
(§5). Campaign ships the control-plane + UI proof; launch proof is the
next campaign's step 1, with G1–G5 as its precondition.

## 3. Slices (minimal, ordered, each characterization-first)

- **S1 — Definition tiers + digest groundwork** (behavior-neutral).
  Loader gains `tier: generated | authoritative`; exporter accepts
  hand-authored authoritative files (`--check` learns the tier rule);
  canonical digest (sha256 of normalized definition) computed at load;
  install rows gain `definition_digest` / `recipe_version` columns
  written for NEW installs, absent = legacy mapping. Baseline: exporter
  byte-equality and startup pins unchanged.
  **DONE 2026-11-02** — shipped as planned except one deviation,
  deliberate: install rows are scan-derived per engine (no write choke
  point), so digests live in a separate additive ledger
  `install_snapshots.yaml` instead of row fields (row fields would have
  altered response payloads = drift). `definition_tier()` lives beside
  `verify_engine_definitions` in `startup_check.py`.
- **S2 — Startup acceptance inversion** (the big behavior move).
  `ENGINE_REGISTRY` becomes a derived view: authoritative definitions
  are LOADED into it (same `to_api_dict()` shape), generated nine keep
  the existing equality guarantee as the migration bridge. Refusal text
  pinned RED-first, then moved deliberately (drift checks become
  generated-tier-only; authoritative engines validate against schema +
  capability vocabularies instead). This is the tranche where
  "definitions mirror registry" flips to "registry derives from
  definitions".
  **DONE 2026-11-02** — legacy refusal bytes recorded pre-inversion
  (see docstring of `test_engine_definitions_authoritative.py`);
  `register_engine_spec` mutates the registry + derived id-sets IN
  PLACE (shared objects, so from-import holders see registrations —
  the sets are now mutable `set`s, documented in `registry.py`).
  Same-id conflicts refuse via the unchanged generated-drift path
  (message bytes pinned). Exporter `--check` gained
  `--definitions-dir` (test-only, refused without `--check`) and the
  authoritative-file rule.
- **S3 — Generic data-driven lifecycle + installer.** One
  `GenericVenvInstaller` implementing the `_run_pip`-shaped surface
  against shared primitives (this is the plan's "one reusable
  primitive", NOT an engine class); `LIFECYCLE` gains a builder that
  materializes entries from definition fields — schema additions:
  `update_source` (kind/package/repo/branch), reuse of `settings_defaults`
  for build-settings verbs + `version_setting_key`. Generated nine stay
  on their current Python registrations untouched (cutover later).
  **DONE 2026-11-02** — shipped as an optional v2 `lifecycle` definition
  section (strict keys), `backend/engines/python/generic.py`
  (`GenericVenvInstaller`, pip-wheel persona: venv.create / pip.*/vcs.*/
  fs.remove_tree + generic publish handlers; anything else fails loudly
  via the executor's unimplemented-primitive refusal),
  `build_definition_lifecycle` + `register_definition_lifecycles`
  (called at startup after verification; Python registration always
  wins), and `register_engine_install_settings` so `_saved()`/settings
  verbs work from lifecycle data (`default_repo_url`/`default_branch`/
  `version_setting_key`). Deviations: no legacy-prefix routers for
  authoritative engines (generic `/api/engines/{id}` surface only —
  decided boundary, documented), and `GENERATED_ENGINE_IDS` (import-time
  snapshot) became the tier anchor because S2 registration made
  membership-based tiering stale.
- **S4 — The `yamlcraft` fixture + control-plane proof.** Hand-authored
  authoritative YAML (persona decision §4), generic-router pins
  parameterized to include it, install-equivalence run in the offline
  harness, G5 architecture-guard test wired into the ratchet script.
  **DONE 2026-11-02** — `backend/engines/definitions/yamlcraft.yaml`
  ships; startup registers it live (smoke: `10 engines (9 generated
  registry-equivalent, 1 authoritative)` +
  `Registered YAML-onboarded engines: yamlcraft` + live
  `GET /api/engines/yamlcraft/status → 200`). The proof suite is
  `test_engine_definitions_fixture.py` (G1–G5 incl. offline install
  through the real executor + ledger digest equality). Deviations:
  the G5 guard is a pytest architecture test scanning all non-test
  sources (not a ratchet-script hook — same enforcement, better
  locality); the fixture deliberately claims artifact format
  `fixture_only` so it NEVER leaks into safetensors/gguf compatibility
  inference or the model catalog (discovered as a real behavior leak
  in full-suite runs and fixed — the full-run order-dependence also
  drove order-proof updates to five existing pins); launch proof
  remains out of scope by decision.
- **S5 — Descriptor-driven frontend lookup.** Server descriptor payload
  (already fetched by the store) gains the routing/capability fields
  (`genericApi` generic flags, versions key, statusRef, managerKey,
  dialog family); `registry.js` merges server-backed entries over the
  local table; local table demotes to offline fallback. Existing
  registry pins byte-pinned UNCHANGED for the nine; `yamlcraft` UI
  presence with zero frontend edits is the acceptance act.
  **DONE 2026-11-03 — data & action layer ONLY (see S6 for the card
  grid; an earlier stamp here overclaimed UI completion and was
  corrected after review).** Additive-overlay implementation. The payload
  gains `generic_lifecycle: true` + `generic_verbs` for AUTHORITATIVE
  engines ONLY (smoke: exactly one occurrence in the live payload —
  the nine descriptors are byte-preserved on the wire, pinned
  backend-side); `registry.js` gained `applyServerDescriptors` /
  `serverBackedDescriptor` / `serverCancelEndpoint` helpers; the store
  polls status, partitions version rows, and routes EVERY venv verb
  (install/install-source/remove/build-settings/check-updates) through
  the generic router for server-backed ids — with dynamic
  `serverStatuses`/`serverVersions` maps and no named refs. Labels
  (toasts, delete dialogs) and task-cancel routing resolve from the
  live payload via a refreshed module-level overlay. Proof:
  `frontend/src/stores/engines.server.test.js` (8 tests) +
  descriptor pin in `test_engine_definitions_fixture.py`. All nine-
  engine pins untouched by construction (static tables unchanged;
  `toHaveLength(9)` still green). Deviations/boundaries disclosed at
  the time: (1) the EnginesView card grid remained a curated 9-card
  template — REVIEW REJECTED THAT BOUNDARY ("marking frontend
  onboarding done is premature"), so it moved to slice S6 below rather
  than staying a documented gap; (2) ModelConfig already merged
  server-first — Crafty appears there labeled by the server and is
  correctly hidden from real models (`fixture_only` matches no
  artifact format — a control-plane engine must not pose as a model
  runtime); (3) ModelConfig's non-safetensors venv-coercion list and
  `engineMatchesFilters` special cases stay engine-id-hardcoded (they
  concern generated engines' quirks, not onboarding).
- **S6 — Dynamic engine card + install/manage journey (added after
  review; the campaign is NOT complete until launch/discovery — see
  scope note above).** A tenth engine must appear on the Engines page
  with a working card and dialog, zero view-code edits. **DONE
  2026-11-03.** `GenericEnginePanel.vue` renders one complete
  install/manage journey for ANY `generic_lifecycle` descriptor: card
  (label, counts, installed/busy badge, CTA) + manage dialog with
  release install (check-updates prefill), source install (repo/branch
  prefilled from the engine's DECLARED defaults via the settings verb),
  live operation state with Cancel, version list from the dynamic
  partition, generic settings save, and confirmed engine-wide Remove —
  all through the store's server-backing-aware verb helpers. The nine
  generated cards are untouched (the panel is a payload-driven section
  appended to the grid); EnginesView gained one import, one computed,
  the v-for, and a task-completion refresh branch for server-backed
  managers. Proof: `GenericEnginePanel.test.js` (9 tests: render,
  states, every verb call + payload, confirm-gated remove, disabled
  while busy) and backend `test_engine_definitions_journey.py` (the
  FULL HTTP journey through the real generic router: settings
  roundtrip, offline install via recipe + executor, status-to-ready
  poll, ready version row, digest ledger row, 400/404 contracts) — so
  the card's exact calls are proven against the live router offline.
  **S6.1 contract fixes (review, 2026-11-03):** the first cut read a
  NESTED operation object that the backend never sends — real
  `_stateful_status` bytes are `operation: str|null` (the operation
  NAME) plus a separate `progress_task_id`; the panel now reads the
  real contract (busy detection, Cancel wiring) and the invented test
  shape was replaced with the true one, plus a guard test asserting a
  nested object does NOT masquerade as a running operation. All
  action controls are now gated on advertised `generic_verbs`
  (status-only lifecycle renders zero actions, incl. no settings
  fetch); the journey's `_ensure_venv` stub was made sync to match
  the base class (the async stub produced unawaited coroutines —
  journey now passes under `-W error::RuntimeWarning`), and it
  asserts the mid-flight contract (string operation +
  `progress_task_id`) whenever the stubbed install is caught running.
  Lesson recorded: the S6 panel test passed while BOTH sides invented
  the same wrong shape — future panel/contract tests must quote bytes
  from a live or journey-captured response, not hand-written fixtures.
  Still open after S6: runtime launch/discovery (out of this
  campaign's scope by decision), snapshot-guided retries, specialized
  installer bodies, audio migration, end-to-end inference.

Recommended campaign order: **S1 → S2 → S3 → S4** proves the backend
promise; **S5** completes the UI half; ship S1–S4 first, then re-decide
on S5 vs the runtime campaign (Q4).

## 4. Key design decisions (owner sign-off, 2026-11-02)

- **Q1 fixture persona — pip wheel engine (APPROVED).** PyPI-style
  install: `venv.create` + `pip.*` + `publish.finalize` over shared
  primitives only; zero native machinery.
- **Q2 inversion strategy — registry becomes a derived view (APPROVED).**
  Authoritative definitions load INTO `ENGINE_REGISTRY` with identical
  `to_api_dict()` output; one identity system; the nine engines stay
  guarded by the existing byte-equality pins.
- **Q3 snapshot scope — install-row digests only (APPROVED).** Launch
  metadata digests land with the runtime campaign.
- **Q4 campaign order — S5 before the runtime campaign (APPROVED).**
  Ship S1–S4 (backend proof), then S5 descriptor-fed UI; runtime/
  discovery migration follows as its own campaign with G1–G5 as
  preconditions.

## 5. Non-goals / deliberate exclusions (this campaign)

- Runtime/discovery migration (profiles → `launch_spec.py`, readiness
  policies, native launcher manifests) — next campaign per plan step 6.
- Specialized primitive extraction for cmake/CUDA/patch machinery — the
  nine generated engines keep their Python-owned recipes until the plan
  step 5 cutover (`cutover allow-list` guard test stays as the fence);
  `yamlcraft` proves the onboarding path WITHOUT that work.
- audio.cpp adapter/services; hardware build & inference parity (stated
  verification boundary: not covered by the suites).
- Marking the nine "migrated" — plan step 5 forbids it while engine
  classes own recipes; this campaign changes that for NEW engines only.

## 6. Risks

- **S2 is the risk center**: startup refusal is pinned as a safety
  feature; the inversion must keep refusing broken definitions
  (schema/vocabulary/duplicate-id checks stay live) while accepting
  authoritative ones. Mitigation: RED-first pins on the refusal strings,
  generated-tier drift checks preserved, boot smoke per slice.
- **S3 hidden installer surface**: shared primitives calling
  `installer._run_pip` means GenericVenvInstaller must cover everything
  the fixture recipe's primitives touch — enumerate against
  SHARED_PRIMITIVES call sites first; the guard "unclaimed primitive
  raises" stays intact.
- **S5 byte-parity**: registry.js feeds many call-sites (fan-out,
  cancel maps, VERSION_ROWS) — merge must reproduce the nine descriptors
  exactly; existing pins run unchanged or the tranche doesn't ship.

Estimated effort (tranche-sized): S1 small · S2 medium · S3 large ·
S4 medium · S5 medium.
