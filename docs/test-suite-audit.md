# Test-suite audit

> **Historical evidence — not the release entry point.** Start at the
> [delivery map](delivery-map.md); current verification is recorded in
> [verification evidence](review-verification-evidence.md).

Reviewed 2026-11-03 against the current, uncommitted workspace.

Scope: the whole test inventory, including earlier audio, model-router,
native-proxy, persistence, and UI refactors, as well as the current engine
refactor. This is an inventory and source audit, not a full execution or a
line-by-line review of every assertion. The findings below describe the audit
baseline; the authorized cleanup is recorded next.

## Cleanup applied

- Deleted 11 obsolete or structural test files. Task classifiers from the five
  family-only files moved into the shared task-profile suite; the functional
  source-sealing case moved into the shared installer-policy suite.
- Removed repeated missing-contract family matrices, the ambient installation
  assertion, the skipped SSE case, the shadow-mode definition-import allowlist,
  and fixed shipped-fixture counts. Installed-contract and installer failure
  scenarios remain covered.
- Replaced the historical family catalog with 12 task contracts, covering task
  aliases without enumerating every engine-owned family name.
- Consolidated the removed audio gateway's route boundary and route-uniqueness
  checks into API behavior tests, replacing source-file/decorator checks.
- Deleted the unused proxy parity normalizer along with its test.
- Trimmed decorative/deleted-branding and duplicate header-link assertions.
  Broad theme accessibility remains in the standards suite; responsive-layout
  checks no longer repeat every theme, and the configuration save-action case
  exercises its specific mobile interaction instead of another page matrix.
- Removed deleted paths from explicit npm test/format commands.

The suites classified as protecting current behavior below were retained.
Compatibility-route and legacy launch-comparison tests remain while their
supported routes and compiler implementations still exist.

Cleanup verification: full backend suite **1,945 passed, 8 skipped**; full
frontend unit suite **611 passed**. Python/frontend lint, the refreshed parity
inventory and literal ratchet passed. The edited browser suites collect 19
Chromium cases, but execution could not start because the installed Playwright
version's Chromium binary is absent. No browser behavior pass is claimed.

## Assessment

There are confirmed leftovers from earlier refactors. Several suites preserve
the migration process or an old source layout rather than current behavior.
There is also meaningful older coverage of current persistence, process,
streaming, authorization, and model contracts. The audit does not support
the claim that most tests are useless, or a defensible percentage to delete.

Current inventory (test files, excluding fixtures and configuration):

| Runner | Files | Source lines |
| --- | ---: | ---: |
| Backend pytest | 155 | 46,380 |
| Frontend Vitest | 64 | 13,938 |
| Browser Playwright | 9 | 2,475 |
| Total | 228 | 62,793 |

These are file/line counts, not collected case counts. Parameterization,
property tests and three browser projects make reported case counts a poor
measure of distinct behavior.

## Confirmed cleanup candidates outside the current engine refactor

### 1. Removed audio WebUI implementation checks

`backend/tests/test_audio_cpp_ui_removed.py` asserts that three old modules
cannot be imported and three source files do not exist. Those checks preserve
the record of a deletion, not current audio behavior. The same file separately
checks route registration, redirect behavior and OpenAPI exposure.

Recommendation: remove the file/import absence checks. If the removed gateway
is an intentional unsupported/security boundary, retain one precise route
contract in the current routing suite. Do not retain four different ways of
proving that old source code was deleted.

### 2. Models-router extraction guardrails

`backend/tests/test_models_routes_architecture_contract.py` reads
`routes/models.py` and counts literal decorator strings; it also forbids
function names and the substring `download_lock`. These are leftovers from
moving downloads into services. Formatting or a valid implementation move can
fail them without changing the product.

Recommendation: retire the source-layout checks. If duplicate routes remain a
real concern, verify uniqueness on the assembled application route table,
once, across all routers. Preserve the scheduling, duplicate-download and
error scenarios in `test_models_route_bundles.py` and related behavior suites.

### 3. Audio family-catalog tests adapted into empty-result checks

The ASR, TTS, ALIGN, ANALYSIS, GEN, SEP and VC profile suites still enumerate
historical family catalogs. Tests named `profile_exists_for_documented_family`
now assert `None`; tests named `field_groups_are_well_formed` frequently assert
`[]`. Their implementations now obtain metadata from installed engine
contracts, rather than a Studio-owned family catalog.

These helpers are not universally constant or necessarily obsolete APIs:
supplying an installed contract can produce real metadata. The redundant
part is repeating the same missing-contract behavior for historical family
names, and presenting it as coverage of those families.

Recommendation: replace these matrices with one missing-contract case and
representative installed-contract cases. Consolidate task-name classifiers
into one parameterized suite. Keep the installed-spec, request-default,
voice-preset and model-specific sidecar cases; do not delete these files
wholesale. `test_audio_task_profiles.py`, `test_audio_cpp_spec_fields.py` and
the installed-spec cases already provide better owners for much of this
coverage.

### 4. A local-installation assertion masquerading as a unit test

`test_audio_cpp_tracking.py::test_persisted_engine_pin_uses_v2_manager_and_commit`
reads repository-local `data/config/engines.yaml`. It returns successfully
without asserting anything when that file or its audio versions are absent;
when present, it constrains a developer's installed manager, commit and
tracking ref. This is leftover validation of a particular migrated setup.

Recommendation: remove the ambient-state test. If a state migration still
needs support, test it using explicit old-state fixtures in a temporary store.

### 5. Permanently disabled SSE smoke test

`test_app_smoke.py::test_sse_events_route` is unconditionally skipped because
the synchronous client buffers an endless stream. It provides no coverage.

Recommendation: remove the dormant test. Retain or implement a bounded
streaming-client check of the actual event endpoint if existing progress/SSE
coverage does not already exercise that contract. A passing or skipped test
count is not evidence of event delivery.

### 6. Old proxy parity normalizer

`test_proxy_parity_harness.py` tests only `scripts.proxy_parity.normalize`.
The script's response observer uses `canonical_response`; source search found
no callers of `normalize` other than its recursion and this test.

Recommendation: remove the unused helper and its test together after checking
whether this script is an externally supported interface. Do not remove the
whole script: `scripts.proxy_conformance.py` actively imports its `observe`
function, which supports current native-proxy conformance testing.

### 7. Native UI/branding migration browser assertions

`frontend/e2e/studio-native.spec.mjs` mixes useful current navigation and
command-import journeys with assertions such as exactly eight decorative
orbit spans and absence of old `Virtual models` copy. `AppHeader.test.js`
has separate tests that both assert the same `/proxy` management link; one
test title still says it checks the configured UI port although it does not.

Recommendation: trim branding/deleted-copy assertions and combine redundant
link checks. Keep health-state transitions, accessible names, navigation,
parameter-list import and field-help behavior.

### 8. Accumulated browser accessibility/layout passes

`ui-refactor.spec.mjs`, `standards.spec.mjs` and `accessibility.spec.mjs`
exercise overlapping themes, pages, layout and dialog accessibility.
`ui-refactor.spec.mjs` alone nests two themes, four widths and seven pages
inside one test; Playwright repeats suites across three browser engines.

Recommendation: give one suite ownership of broad page/theme accessibility,
one a small responsive layout matrix, and keep focused interaction checks
for keyboard/focus/restore dialogs. Do not delete accessibility coverage as
obsolete: these are current behaviors, and browser semantics are not replaced
by happy-dom unit tests. Exact redundancy and runtime savings require a
per-scenario browser matrix, not filename matching.

## Current engine-refactor accumulation (additional to the older leftovers)

- `*_definition_parity.py`: repeated mocking of the executor and runner to
  compare materialized plans, handler-key maps and context-key maps. Consolidate
  common-host behavior; remove assertions that merely mirror the implementation.
- Inheritance/source-inspection assertions should not be replicated per engine.
  Behavior must survive a different valid class structure.
- `test_engine_definitions_equivalence.py` retains a shadow-mode import-reader
  allowlist and fixed shipped-fixture IDs. Retire the transitional import gate
  and exact counts; retain actual schema, descriptor and supported-primitive
  validation while two sources of definitions coexist.
- Legacy-vs-profile launch comparison suites are temporary cutover evidence.
  Once a legacy compiler is removed, retain a small independent expected
  launch contract instead of keeping the obsolete compiler solely as an oracle.
- Legacy route equality still has a purpose while compatibility endpoints are
  supported. Consolidate representative delegation contracts; remove them with
  the routes, not merely because the new route exists.
- Legacy/shadow installer suites contain useful process argv, environment,
  source-reuse, cleanup and failure scenarios. Move these into shared-host and
  engine-policy tests rather than discarding them with the old harness.

## Older tests that should not be classified as obsolete

- `test_engineering_hardening.py`: corrupt-file preservation, atomic writes,
  duplicate IDs, cancellation ownership and real process cleanup still protect
  current behavior. Some store/concurrency cases overlap with newer suites;
  consolidate ownership, keeping distinct race/failure scenarios.
- `test_proxy_transport.py`, `test_proxy_supervision.py`,
  `test_proxy_accounting.py`, `test_proxy_controls.py`,
  `test_proxy_live_routing.py`, `test_studio_proxy.py`: these exercise the native
  runtime. The word proxy is not evidence that they test the removed external
  proxy. Keep admission, streaming, process ownership and access controls.
- `test_audio_swap_filters.py`: the filename is historical, but the tests call
  current `backend.proxy.commands.request_filters_and_aliases`. Rename or
  reorganize; do not delete current request-default behavior.
- `test_persistence_failure.py`, `test_action_recovery.py`,
  `test_operation_state_machine.py`: persistence failure and interrupted-action
  semantics remain distinct from simple store CRUD tests.
- CLI recorded-help and inspect fixtures: preserve version-specific parsing
  and discovery behavior. Fixture reads are not source-inspection anti-patterns.
- Clipboard unit tests and real browser clipboard/focus checks cover different
  platform boundaries; overlap in feature name does not establish redundancy.
- Live audio tests gated on local binaries/models are optional integration
  coverage, not the same problem as an unconditionally disabled test. Give them
  an explicit integration lane instead of pretending ordinary CI covers them.

## Cleanup order and acceptance

1. Remove no-op/local-state/deleted-source checks and the unused normalizer.
2. Consolidate family-catalog empty-result matrices and architecture text checks.
3. Move valuable legacy installer cases into shared behavior suites; retire
   duplicate migration scaffolding and legacy oracles with each completed cutover.
4. Assign browser scenario ownership and reduce repeated page/theme/width passes.
5. Run the affected suites and the full suite after consolidation. Compare the
   distinct failure scenarios retained, not just green totals. Use targeted
   fault injection for the highest-risk removed duplicates where useful.

No deletion quota is justified yet. The appropriate deliverable is fewer
tests with clear ownership of current contracts, not an arbitrary percentage
reduction. This audit did not run tests because its claims concern source and
coverage ownership; previous green test totals do not determine test value.
