# Config-Driven Engine Refactor — Uncommitted-Tree Handoff

> **Historical evidence — not the release entry point.** Start at the
> [delivery map](delivery-map.md); the [change inventory](change-inventory.md)
> maps this work to the review series.

Generated as the final verification + handoff sweep of the tranche series
(40+ tranches). Companion documents:
`docs/config-driven-engine-refactoring-plan.md` (the plan — never modified
by any tranche) and `docs/engine-surface-contract-matrix.md` (the
contract record, Â§8 carries the full tranche log). Everything below is
UNCOMMITTED by design, for review.

> **Status rebaseline (2026-11-04).** This section is a historical tranche log.
> The requirement-level status of record is
> `docs/engine-refactor-requirement-matrix.md`, which re-checks every original
> requirement against the current tree. In particular: the R98 frontend claim
> was false (real result then: exit 1, 14 unhandled errors) and is superseded by
> R99/R100 plus this pass's gate-runner fix; the R70 "Phase 7 complete" claim
> was withdrawn; "Phase 6 done" covers only the five pip-engine compilers
> (native GGUF launch remains a decision, not completed work); and the registry
> now loads definition-owned engines at import, with the exporter tier removed
> entirely.
>
> **Slice 6 (same date):** the V100 and 1Cat engine-specific installer
> classes/mixins are gone. Upstream rewrites are versioned transform assets
> applied by `vcs.apply_transform`; the fork build environment is declared
> through `build.prepare_fork`; 1Cat wheel selection, Rust provisioning,
> requirement fallback and wheel building are declared primitives. All nine
> shipped engines are definition-owned (the exporter tier has been removed). A
> YAML-onboarded engine renders through the
> payload-driven `GenericEnginePanel` with no local frontend descriptor.
> Native llama-family source rebuilds, audio.cpp rebuilds and Unsloth release
> retries now resolve the frozen install snapshot and refuse a changed
> definition through the shared `backend/engines/install_snapshots.py`; every
> shipped engine records digest/resolved inputs before retrying.
>
> **Final status (2026-11-04):** the requirement-level status of record and the
> ownership/compatibility inventory are `docs/engine-refactor-requirement-matrix.md`
> and `docs/engine-refactor-final-audit.md`. Completed scope: all nine shipped
> engines are definition-owned; V100/1Cat subclasses and the exporter tier are
> gone; reusable primitives cover transforms, fork builds, wheel/Rust/release
> assets and snapshot retries; audio services and extension routing resolve
> their engine from declared capabilities; compatibility URLs are thin
> delegates; the YAML-only harness proves install→discovery→launch→request→stop/
> remove for fixtures and for a shipped vLLM engine. Native source builds,
> Unsloth installs and native GGUF launches now run on the shared host; GGUF
> launch/inference and a CUDA source build were exercised on a remote V100.
> Native source builds use the shared `CmakeSourceInstaller` with a
> `source_build` definition policy; the `nativecraft` fixture proves a third
> native fork is configured by data. Transitional: SGLang runtime-selector copy
> and audio-package branches (requirement 9). Verification: backend 2156 passed
> / 8 skipped; frontend gate CLEAN (652); all static gates current.

## 1. What was built (scoreboard)

- **Backend dual-surface router**: generic `/api/engines/{engine_id}/{verb}`
  (`backend/routes/engine_lifecycle.py`) over the shared `LIFECYCLE`
  records (`backend/routes/engine_versions.py`). Serves 8 of 9 engines —
  the five python-venv engines via shared lifecycle callables, the three
  native llama-family engines via per-verb `verbs` override maps that
  delegate to their legacy route functions (identical by construction).
  Unimplemented verbs answer a pinned 405 contract; audio.cpp remains on
  its own router by design. `response_model=` is never used on the generic
  router (OpenAPI-only `responses=`) to avoid byte drift.
- **Engine definitions + shadows**: declarative engine definitions
  (`backend/engines/definitions/`, exported by
  `scripts/export-engine-definitions.py`) with legacy-plan shadow tests
  (one `test_*_legacy_plan_shadow.py` per family) proving release/plan
  equivalence; primitives library + executor equivalence suites.
- **Contract matrix + inventories**:
  `docs/engine-surface-contract-matrix.md` (all verb-by-verb contracts,
  divergences marked PRESERVE), `scripts/parity-inventory.py` +
  `docs/engine-parity-inventory.*`, literal ratchet
  (`scripts/engine-literals-ratchet.py`, no-growth gate + pytest).
- **Typed contracts**: `backend/models/engine_contracts.py`
  (`tests/test_engine_contracts.py`), openapi export/check.
- **Frontend registry** (`frontend/src/engines/registry.js`): nine engine
  descriptors owning identity (labels, icons, repoSource/untagged fallback,
  statusRefs, managerKeys), API routing (`genericApi` +
  `venvApiPath`/`legacyVenvApiPath`; `NATIVE_GENERIC_VERBS` +
  `nativeGenericApiPath` + cancel-endpoint maps), and the families
  (sglangRuntimeUi WITHOUT vllm vs panelFamily WITH vllm — real
  divergences kept as data). Absorbed: ENGINE_LABELS,
  fallbackEngineOptions, MANAGER_CANCEL_ENDPOINTS, SGLANG_ENGINES,
  CMAKE_ENGINES, inferredEnginesByFormat, the repository_source second
  identity system, six parallel per-engine records.
- **Store** (`frontend/src/stores/engines.js`): fan-out rules declarative
  (`STATUS_REFRESH_INDEPENDENT/_EXCLUSIVE/_ACTIVATE_ONLY`), version-row
  partition driven by registry `repoSource`, all venv engines AND all
  native lifecycle verbs (reads, writes, task-cancel endpoints) now routed
  through the generic router; legacy prefixes kept live as rollback.
- **EnginesView slices 1–2c**: `openEngineModal` dispatch table, merged
  task-completion venv branch, merged venv dialog bodies
  (`VENV_DIALOG_CONFIG`) and ONE merged dialog header block for all five
  venv-family dialogs, then the native llama-family trio (llama.cpp /
  ik_llama / Unsloth — headers AND bodies) merged into one block pair via
  `NATIVE_DIALOG_CONFIG`, real a11y/copy divergences preserved as data
  (`buttonA11y`, unsloth's missing settings action, raw note paragraph,
  banner version-prefix asymmetry).
- **Cleanup/audit pass**: dead store wrappers pruned, exports audited
  (84â†’79 store exports), registry header documents all three routing
  mechanisms, allow-lists re-verified.

- **Review-response tranche (2026-11-02)**: generic native `install` /
  `install-source` now derive the build target from the URL PATH (conflicting
  `engine`/`repository_source` bodies rejected with a pinned 400; legacy
  routes keep their compatibility quirk byte-equal), and `execute_plan`
  guarantees its teardown pass for ANY handler exception while re-raising
  the original exception verbatim (shadow-suite unwrapped-propagation
  contract preserved).

- **YAML campaign S1 (2026-11-02)**: definition digests
  (`EngineDefinition.definition_digest`, full-recipe canonical sha256),
  generated/authoritative tier classification (policy still unchanged),
  and the additive `install_snapshots.yaml` ledger wired into the generic
  install dispatchers (pinned inert to responses). See
  `yaml-engine-onboarding-campaign.md` for the campaign; S5 pending.

- **YAML campaign S2 (2026-11-02)**: startup acceptance inversion —
  `register_engine_spec` loads authoritative YAML definitions into
  `ENGINE_REGISTRY` (derived view, decided Q2); derived id-sets are
  shared mutable objects refreshed in place; generated-tier refusal
  bytes untouched (pinned); exporter `--check` learns the
  authoritative-file tier rule. RED-first legacy bytes recorded in
  `test_engine_definitions_authoritative.py`.

- **YAML campaign S3 (2026-11-02)**: `lifecycle` definition section +
  `GenericVenvInstaller` (pip-wheel persona, definition-executing) +
  data-driven LIFECYCLE registration at startup + settings-defaults
  hook; tier anchor now `GENERATED_ENGINE_IDS`. Pins +8.

- **YAML campaign S4 (2026-11-02)**: `yamlcraft.yaml` fixture ships —
  live boot registers it (G1/G2 proven on a real server; status 200
  from YAML alone), G3 offline install + G4 ledger-digest equality +
  G5 zero-hardcode architecture guard in
  `test_engine_definitions_fixture.py`. Artifact-format leak found &
  fixed; lifespan-order pins made order-proof.

- **YAML campaign S5 (2026-11-03)**: descriptor-driven frontend lookup —
  payload routing facts (`generic_lifecycle`/`generic_verbs`, one live
  occurrence = nine byte-preserved), additive client overlay (poll,
  rows, ALL venv verbs, labels, task-cancel) proven by
  `engines.server.test.js`; static nine-entry tables untouched by
  construction. (Data/action layer; an interim "campaign complete"
  claim in this bullet was corrected by review — see S6.)

- **YAML campaign S6 (2026-11-03, review-driven)**: dynamic Engines-page
  card + full install/manage journey for YAML-only engines
  (`GenericEnginePanel.vue` + 11 vitest tests; offline full-HTTP journey
  through the real generic router in `test_engine_definitions_journey.py`).
  Campaign scope corrected per review: control plane is complete,
  **onboarding is NOT** — launch/discovery, retries-from-snapshot,
  specialized installers, audio migration, end-to-end inference remain.
- **S6.1 contract fixes (2026-11-03, review)**: panel reads the real
  status bytes (`operation` = NAME string, separate `progress_task_id`
  for cancel — the first cut AND its test had invented the same nested
  shape, so green tests hid the bug); all controls gated on advertised
  `generic_verbs`; journey `_ensure_venv` stub made sync (journey now
  passes `-W error::RuntimeWarning`); mid-flight string/progress_task_id
  contract asserted when the stubbed install is caught running. New
  rule: contract tests quote live/journey-captured bytes, never
  hand-written fixtures.
- **YAML LAUNCH campaign R1/R2 (2026-11-03)**: second campaign
  `docs/yaml-engine-launch-campaign.md` (signed Q1–Q4). Profile-driven
  launch interpreter `backend/proxy/profile_launch.py` — YAML-only
  engines now compile real `LaunchSpec`s from `runtime_profiles`
  (post-chain dispatch; nine unreachable BY ORDER + GENERATED gate,
  pinned). `test:backend:engine` slice and the gate-matrix fast lane
  gained `test_profile_launch.py`. (At R1 close no shipped fixture
  could launch; superseded by R3 below.)
- **YAML LAUNCH campaign R3 (2026-11-03)**: the first process Studio
  ever launched from YAML alone. `launchcraft.yaml` (11th engine,
  authoritative, fixture_only) declares its full launch contract in
  `runtime_profiles`; `test_launchcraft_journey.py` proves REAL pip
  install (hand-built local wheel, `--no-index`, zero network) â†’ REAL
  venv â†’ activation row â†’ R1 interpreter â†’ REAL `ProcessRegistry`
  spawn + production `/health` poll loop â†’ argv proven from INSIDE the
  live process (`GET /__state`) â†’ clean SIGTERM drain, zero orphans.
  Membership pins for the 11th engine updated deliberately (loader ids,
  tier counts, digest set, lifecycle upper-bound, schema dir count,
  generic-test fake loader now generated-tier-only). The HTTP surface +
  chat round-trip followed in R4 below.
- **YAML LAUNCH campaign R4 (2026-11-03)**: the proxy path, not just the
  spawn. `test_launchcraft_journey.py` now drives
  `POST /api/models/{id}/start` through the real route + `ProxyClient`
  into the Studio dataplane's lazy acquire (the passthrough body is the
  engine's own `/v1/models`), streams one `/v1/chat/completions`
  through the dataplane and asserts the engine's real SSE bytes +
  `x-request-id` + its engine-side hit counter, then `POST
  /api/models/{id}/stop` drains the process group with
  `model_start`/`model_stop` journal receipts `succeeded` (successes
  are pruned from `operations.yaml` by design — the test records
  submitted rows and asserts no open admission fence survives).
  Compilation uses production `compile_model_runtime`;
  `VALID_ENGINE_IDS` comes from the real startup gate. Owner gate Q4
  (chat round-trip through the proxy) is now MET for the fixture
  persona. No production code changed in R4 (tests + fixture only).
- **YAML LAUNCH campaign G5 — discovery wired (2026-11-03)**: YAML-only
  engines now get a real param catalog from their own definition.
  `scanner.scan_engine_version` asks
  `profile_launch.profile_scan_contract`; definition-backed engines scan
  the runtime-profile executable with the YAML-declared
  `discovery.probe_argv_prefix` and parse via `DEFINITION_HELP_PARSERS`
  (keys = the five existing scanner kinds, no vocabulary growth).
  `discovery.parser` is therefore WIRED, not decorative. Proof:
  `scan-engine-params` route â†’ `ok/param_count`, param-registry serves
  the sections, runnability flips by `active_path_fields` only;
  generated ids pinned to their legacy adapters (parametrized); typed
  refusals for missing discovery / no parse-only adapter. New
  `test_engine_definition_discovery.py` (13 tests) + journey extension.
  **R5 (policy data) deliberately DEFERRED** — schema/exporter work for
  a non-gate optimization; rationale in the campaign doc. Ratchet green.
- **Review follow-up R6 — compiler edge cases + second persona
  (2026-11-03)**: three reviewed defects fixed, each pinned.
  (1) `env.required` is validated against the environment the process
  actually receives — `os.environ + spec.env.set âˆ’ spec.env.unset` — so
  a required variable present in the process env but unset by config is
  a typed error instead of passing validation and being absent at
  launch; engine defaults satisfy it.
  (2) Studio-owned custom-arg rejection is profile-driven
  (`profile_owned_flags`: model-binding flag, port-binding flag, and
  every flag-shaped `argv_prefix` token), so custom `--model`/`--port`
  cannot override YAML-selected values; non-owned args still append;
  legacy compilers keep their fixed set byte-identical.
  (3) The GGUF branch asks the profile compiler first — a YAML engine
  declaring `gguf` artifacts is profile-launched while the nine
  generated GGUF ids keep the legacy compiler (GENERATED gate, pinned
  both ways); `_use_model_name` follows the definition for
  definition-backed GGUF engines.
  **Second persona `launchforge`** (12th engine, authoritative) covers
  the case the `fixture_only` fixture could not: `artifact_formats:
  [gguf]` (joins the derived GGUF set — the trap), `venv_python` with a
  `-m module` prefix, POSITIONAL model binding, `cwd: venv_root`,
  required env satisfied by an engine default. Real journey:
  install â†’ activate â†’ production `compile_model_runtime` â†’ start route
  â†’ OS argv proven via `/proc/self/cmdline` â†’ health â†’ streamed chat
  through the proxy â†’ drain. Engine-count pins grew 11 â†’ 12
  deliberately; `test_engine_model_contracts` now asserts the legacy
  GGUF ids in order plus only shipped-definition extras
  (registration-order-proof).
- **Review follow-up R7 — revision digests + snapshot-backed retries
  (2026-11-03)**: the plan's "Immutable definitions" provisions (plan
  line 55). `CompiledModel.profile_digest` carries the authoritative
  definition's canonical digest ("" for the nine);
  `launch_revision()` adds it to the hashed body only when non-empty, so
  generated-engine revisions/manifest bytes stay byte-identical while a
  YAML edit now changes the revision even when the compiled argv/env are
  unchanged; manifests record it. Install retries now READ the existing
  install-snapshot ledger: `retry_existing_install` refuses (400) when
  the definition digest moved since the install and prefers the
  snapshot's frozen resolved inputs over live re-resolution — "changing
  YAML must not silently reinterpret their install layout". Also fixed
  test isolation: the three journey tests snapshot/restore the global
  LIFECYCLE registry (registration skips existing ids, so leftovers made
  later registrations silently incomplete). Battery 2232; smoke #50.
- **Review follow-up R8 — installation-specific snapshots + digest sweep
  (2026-11-03)**: ledger rows now carry `version` + `phase`
  (`scheduled` dispatch rows / `published` installer rows); retries select
  by version (published row wins, scheduled rows match on their recorded
  requested version), so retrying A after B never borrows B's digest or
  resolution. Dispatch rows record the RESOLVED effective settings
  (version from saved build settings + explicit request fields), and the
  published row freezes the detected package version. The
  `model_runtime_apply` manifest writer now passes `profile_digest`, and
  a sweep test asserts every non-test `manifest_document(` call site
  passes it.
- **Phase-3 LMDeploy parity gate (2026-11-03)**: `LMDeployInstaller`
  already executes the definition's plans for all four flows; the missing
  piece was the characterization basis (2 tests). New
  `backend/tests/test_lmdeploy_definition_parity.py` (8 pins) freezes:
  executed plan == `materialize_plan(definition, flow)` step-for-step for
  release/source/sync/remove; the only engine-specific code is the pinned
  handler map per flow; executor context inputs supplied. Zero engine-id
  literals (id read from the installer); wired into the engine slice
  (318 tests, ~33s). Next gated step = swap the bespoke host class for
  the generic installer, which needs a `lifecycle` block on the generated
  definition (exporter + schema vocabulary) first.
- **Phase-3 LMDeploy EXTRACTION (2026-11-03, DONE)**: the flip. The
  `EngineSpec` gained an installer-only `lifecycle` mapping (excluded from
  `to_api_dict`; descriptor/YAML equivalence pins unchanged),
  round-tripped through `to_spec()`, emitted by the exporter into
  `definitions/lmdeploy.yaml`, and verified by the exporter itself.
  `LMDeployInstaller` is now a 72-line `GenericVenvInstaller` subclass
  (from 666) keeping only the `LMDEPLOY_BIN` binary-resolution hook plus
  the manager singleton/alias; all four flows, progress, state and
  bookkeeping are inherited. Parity pins updated: flows must be inherited
  (re-implementation fails the structural pin), plan/handler/context pins
  still hold. Disclosed behavior deltas: retries now use the shared
  snapshot-backed retry, and pre-change installs (old definition digest)
  refuse a retry with "definition changed â€¦ reinstall". Verified on
  lanes only: 11 lmdeploy tests, engine slice 318, defcheck 12 engines,
  ratchet no growth, ruff 62, openapi 0.
- **Upstream vLLM gate (2026-11-03)**: definitions/vllm.yaml now carries
  the exported lifecycle facts, and test_vllm_definition_parity.py (5
  pins) freezes the CURRENT flows: definition plan step-for-step,
  per-flow handler map (publish.finalize, env.compose, vcs.*,
  publish.forget), and executor context keys, plus a structural pin that
  the flows are still local (inverted when extraction lands).
- **Phase-3 typed installer policy (2026-11-03)**: the shared host now
  reads `lifecycle.python_min`/`python_max` (interpreter guard, legacy
  message shape), `build_env: cuda` (`env.compose` supplies the CUDA
  build env + cuda/python record metadata), `publish.finalize`
  (validation, commit-tagged naming, reuse, activation, published
  snapshot), and `forget_errors: raise` (strict version delete). Remove
  supplies both context vocabularies and handler maps are plan-filtered.
  Schema gained the `build_env`/`forget_errors` vocabularies; vllm.yaml
  exports the four policy values. Pinned by
  `test_generic_installer_policy.py` (8, synthetic persona); the generic
  host stays green with the LMDeploy pins + launchcraft journey. Engine
  slice **340 / ~36s**. The vLLM seal question was resolved and the flip
  landed (below).
- **Upstream vLLM EXTRACTION (2026-11-03, DONE)**: the seal handler now
  matches the executor's default primitive exactly (destination/
  workspace_from honoured, `checkout` rebound to the sealed path,
  `checkout_sealed` kept as a label), so vLLM runs on the shared host.
  `VllmInstaller` went 677 → ~105 lines: a `GenericVenvInstaller`
  subclass; the definition supplies names/repo, `python_min`/`python_max`,
  `build_env: cuda`, `forget_errors: raise`, and the recipes use the
  shared `publish.finalize`/`env.compose`. Only the status payload stays
  engine-specific. Parity pins rewritten (inherited flows + seal
  equivalence via `run_shared_primitive`). Engine slice **332 / ~35s**;
- **SGLang gate (2026-11-03)**: definitions/sglang.yaml now carries the
  upstream lifecycle facts (binary `python`, distribution/root/log names,
  repo, pypi package, version key), and
  `test_sglang_definition_parity.py` (6 pins) freezes the current
  contract: four flows execute the definition plan step-for-step with
  pinned handler maps/context; SOURCE and SYNC override the specialised
  `vcs.apply_patch`/`vcs.prepare_build` primitives (Phase-5 recipes the
  shared host does not implement); status carries `v100`; flows remain
  local. Flip deliberately NOT done — start messages differ from the
  shared host and status needs a declaration/subclass decision, while
  `sglang_v100` stays specialised regardless. Engine slice **340 / ~36s**.
- **Regression-harness repair (2026-11-03, DONE)**: the LMDeploy/vLLM
  regression selection went from 26 failed / 10 passed to **113 passed**
  (integration + shadow suites + version lifecycle + removal
  characterization). Real regressions found and fixed in
  `GenericVenvInstaller`: `env.compose` now produces `build_env`/
  `runtime_meta` (recipes' declared names — CUDA env was being lost),
  sync accepts `install_dir`-only rows and merges row + runtime into
  failures, remove targets the ACTIVE row's `install_dir` and releases
  the hold there, source checkouts emit the legacy 3/8 progress stages,
  and sync finalize merges the row. Harness rot fixed by repointing
  patch targets at the shared module (`unique_version_name`,
  `mark_runtime_changed`, `activate_engine_version`) and faking the
  reuse-mode `git clone` subprocess. Disclosed normalization: the shared
  host always refreshes install state on failure (legacy vLLM did not —
  idempotent bookkeeping). The previously-missed `test_lmdeploy_legacy_plan_shadow.py` (14) needed the same repair; the FULL serial suite is now green at **2265 passed, 9 skipped**. Parity inventory REGENERATED (was stale; regeneration is now idempotent and the ratchet reads the refreshed baseline — outside-engine counts: lmdeploy 119/29 files, vllm 126/26, sglang 98/26, sglang_v100 87/20, llama_cpp 275/46).
- **Upstream SGLang EXTRACTION (2026-11-03, DONE)**: `SglangLogMixin`
  now owns the log-streaming/progress machinery shared by both variants;
  `SglangUpstreamInstaller(SglangLogMixin, GenericVenvInstaller)` is the
  registered upstream installer (flows inherited; definition supplies
  names/repo/package). Only the byte contract stays local:
  `_source_started_message()` (new generic hook) and the status payload
  (same keys + `v100: False`). `SglangInstaller` now serves ONLY the V100
  fork, which keeps the specialised `vcs.apply_patch`/`vcs.prepare_build`
  recipes (Phase 5). Pins: upstream flows inherited + plan/handler/context
  step-for-step; V100 boundary asserted. SGLang suites 56 + parity 8;
  engine regression selection 93; FULL serial suite **2265 passed, 9 skipped** (parallel lane: same 2265 in 88s); engine slice **340 / ~36s**; defcheck
  12; ratchet no growth; ruff 65; openapi 0.
  defcheck 12; ratchet no growth; ruff 66; openapi 0.
- **Phase 6: LMDeploy launch from the definition (2026-11-03, DONE)**:
  `EngineSpec.runtime_profile` (API-invisible, like `lifecycle`) exports
  the byte-pinned reference profile into the generated YAML; `_compile_launch`
  consults the profile interpreter FIRST, and the GENERATED gate is now an
  explicit opt-in (only ids whose exported definition declares a profile).
  Studio's port/host reservation is unioned with the profile-owned flags.
  Goldens: profile spec == legacy `_compile_lmdeploy` field-for-field for
  defaults and structured params, dispatch, port rejection, CPU contract.
  An `env_override` (LMDEPLOY_BIN) idea was REVERTED after checking the
  legacy resolver: it never honoured that variable, so declaring it would
  change behavior. Remaining Phase-6 engines flip the same way, one gated
  item each.
- **Phase 6: upstream vLLM launch from the definition (2026-11-03, DONE)**:
  new `env.cuda` profile key resolves Studio's managed CUDA env for the
  active install and composes it like the legacy compilers; `vllm.yaml`
  declares venv-python + `-m vllm.entrypoints.openai.api_server` +
  --model/--port + venv-root cwd + cuda env, exported from the spec.
  Goldens (6) compare the profile spec to `_compile_module(... engine="vllm")`
  field-for-field (defaults, structured params, dispatch, port rejection,
  missing env/hf-id). `test_vllm_integration.py` was repaired to drive the
  active install row + `get_cuda_installer` instead of legacy resolver
  symbols. Engine slice **351**; parallel full **2288 passed, 9 skipped**;
- **Phase-6 gaps closed (2026-11-03, DONE)**: `definition_digest_for()` now
  returns the digest for every opted-in profile engine (and still None for
  the byte-frozen eight), so LMDeploy's/vLLM's profile digest reaches launch
  revisions and manifests; `_resolve_executable` regained the legacy edge
  cases (relative venv -> /app normalization, venv dir must exist, target
  must be executable). New parity tests: relative-venv equality,
  non-executable failures, digest split. LMDeploy parity 7, profile_launch
  26, engine slice **354**, parallel full **2288 passed, 9 skipped**;
  defcheck 12; ratchet no growth; ruff 67; openapi 0.
- **Phase 6: 1Cat-vLLM launch from the definition (2026-11-03, DONE)**:
  plain module profile (venv python, vLLM api_server module, --model/--port,
  venv-root cwd, no cuda env), exported from the spec; 5 goldens compare it
  to `_compile_module(... engine="1cat_vllm")` field-for-field. The vLLM
  parity helper was narrowed to the cuda-env engine (the module string is
  shared). SGLang launch stays legacy pending a `runtime_switch` design
  decision (its diffusion arm reads the scanned catalog). Engine slice
  **359**; parallel full **2288 passed, 9 skipped**; ruff 68.
- **Phase 6: SGLang launch + `runtime_switch` (2026-11-03, DONE)**: the
  R20 decision is data now - `runtime_switch` (config_key/default/variants
  with `argv_prefix` + `scanned_catalog`) lets the interpreter select
  llm/diffusion, swap the prefix, re-index params from the diffusion catalog
  (llm keys filtered) and raise the legacy typed "unavailable" error
  byte-for-byte. `sglang.yaml` carries both arms. Goldens 4 (llm, diffusion,
  error message equality, dispatch). Engine slice **363**; parallel full
  **2288 passed, 9 skipped**; ruff 69.
- **Phase 6: SGLang V100 launch from the definition (2026-11-03, DONE)**:
  the last Python-venv runtime adapter. `env.cuda` now accepts a version
  string (the fork's 12.8 fallback) and `env.required` accepts
  `{name, error}` so the managed-CUDA preflight keeps its exact legacy
  message. Profile carries venv python + `-m sglang.launch_server` +
  --model-path/--port + venv-root cwd + fixed env defaults + required
  CUDA_HOME + the llm/diffusion switch. Goldens 4 (default equality, 12.8
  fallback, byte-equal CUDA error, dispatch). Harness repairs: sglang
  preview tests and the launch-manifest helper now drive the active install
  row + fake `get_cuda_installer`. Engine slice **362**; parallel full
- **Phase-6 boundary: native GGUF launch STAYS legacy (2026-11-03,
  DECIDED)**: llama.cpp/ik/unsloth compose a model source (file vs
  --hf-repo), mmproj + draft companions, alias and library path - family
  behaviour, not per-engine YAML. They ship no profile; the decision is
  pinned by `test_native_gguf_engines_stay_on_the_legacy_compiler`. All
  Python-venv launch adapters are now definition-driven. Engine slice
  **363**; full **1958 passed, 8 skipped**; ruff 68.
- **Review P1/P2 fixed (2026-11-03)**: (P1) `get_sglang_manager()` now returns
  the SAME upstream singleton the adapter registry uses, so routes
  (status/cancel/install), adapters and retry/sync share one manager -
  pinned by an identity test. (P2) `execute_plan` now catches
  `asyncio.CancelledError`, runs the teardown pass with SHIELDED cleanup
  invocations, and re-raises the cancellation unchanged - pinned by a
  live-task cancellation test. Engine slice **365**; full **1960 passed,
  8 skipped**; static gates green.
- **Phase-5 step 1: V100 patch scripts extracted (2026-11-03, DONE)**:
  `v100_support.V100PatchMixin` owns the Marlin builder, Python-metadata and
  CUDA-IPC patch implementations VERBATIM (embedded shell scripts stay
  byte-identical); `SglangInstaller` inherits it, so the class no longer
  duplicates them. Pinned by a mixin/absence test. Sequence continues with
  env composition + prepare-build extraction, then the `SglangV100Installer`
  subclass + registration flip. Engine slice **366**; isolation checks
  green; parallel suite 1960/8 with one rotating pre-existing flake per run.
- **Phase-5 step 2: V100 build environment extracted (2026-11-03, DONE)**:
  `v100_support.V100BuildMixin` owns compiler selection, memory-aware job
  sizing, the Studio/Python/CUDA bindings and the ccache/HOME/cargo/
  dependency-preservation layer; `V100_CUDA_VERSION` moved and is
  re-exported. Static-method call convention and the async prepare hook were
  preserved deliberately; pinned by a mixin/absence test. Sglang suites 60;
  engine slice **366**; parallel full **1962 passed, 8 skipped**.
- **Phase-5 step 3: SglangV100Installer registered (2026-11-03, DONE)**:
  `GenericVenvInstaller` gained `_extra_plan_handlers()` (merged into all
  plan maps); `SglangV100Installer` supplies ONLY `vcs.apply_patch` /
  `vcs.prepare_build`, the env hooks, the 3.12 guard, source-start wording
  and the v100 status payload - all flows inherited. `get_sglang_manager`
  routes both variants to definition-driven singletons. Pins: inherited
  flows, handler set, recipe coverage, and two flow-execution pins.
  Sglang suites 70; engine slice **370**; parallel full **1965 passed,
  8 skipped**.
- **Phase-5 step 4 (legacy class deletion) DEFERRED (2026-11-03)**: the
  deletion was attempted with a `SglangManager` facade, measured (29
  failures confined to legacy seams: `_clone`/`_sync_checkout` patches,
  `unique_version_name` patched on the engine module, three parity pins)
  and REVERTED to keep the tree green. Port plan: release shadow patch
  targets -> generic module; source/sync fakes -> subprocess +
  `_sync_git_checkout`; integration constructors -> facade; parity pins
  rewritten for both variants; then delete the class and facade and verify
  the ratchet shrinks.
- **Phase-5 step 4a: release shadow ported; two real gaps closed
  (2026-11-03)**: porting the release tests to the new classes exposed (a)
  SGLang's definition-driven release had ONE detection where legacy had two
  - fixed by adding `pip.capture_metadata` to the exported release recipe -
  and (b) the fork's release->source routing lived only in the legacy class
  - `SglangV100Installer.install_release` now delegates explicitly, with the
  V100 installer lifecycle data added to the definition. Release shadow 7

- **Phase-5 step 4b attempted and REVERTED (2026-11-03)**: mechanically porting the source/sync shadow to the new classes left 13/14 tests failing because their assertions encode the LEGACY flow's call order/metadata/error strings. Reverted; those tests must be re-derived from the recipe plan rather than re-pointed. Tree verified green (sglang 64, engine slice 370).
- **Phase-5 step 4b landed as a CONTRACT harness (2026-11-03)**:
  `test_sglang_source_contract.py` (7 tests) drives the definition-driven
  managers and pins the observable contract - pip argv/cwd, checkout guard
full **1972 passed, 8 skipped**; ruff 69.
- **V100 cutover P1s FIXED (2026-11-03)**: `_v100_apply_checkout_patches`,
  `_write_v100_prefix_installer` and `_preserve_v100_dependency` moved into
  `V100PatchMixin`, so the registered `SglangV100Installer` has the patch
  and recovery steps (both were AttributeErrors). Contract tests now run the
  REAL helpers on temp checkouts (builder rewrite + dependency
  preservation). Contract harness 9; SGLang 73; slice **379**; full 1973/8
  with the known journey flake (passes isolated).
- **LEGACY SglangInstaller DELETED (2026-11-03)**: the class is gone;
  `SglangManager` is a compatibility facade over `SglangUpstreamInstaller`
  / `SglangV100Installer`. Specialised code lives in `V100PatchMixin` /
  `V100BuildMixin`; the legacy transcript file and four legacy-only
  integration tests were retired in favour of the contract harness. SGLang
  55; slice **379**; full **1956 passed, 8 skipped**.
- **Post-deletion cleanup (2026-11-03)**: pruned 8 dead imports, kept the
  `V100_CUDA_VERSION` re-export explicitly, and repointed two integration
  patch targets from the deleted installer module to `v100_support`.
  LMDeploy/vLLM confirmed as already-thin (no legacy duplication left);
  remaining legacy code is the specialised 1Cat/native/audio installers.
- **1Cat extraction scoped (2026-11-03)**: `backend/engines/vllm/onecat.py`
  (1059 lines) is already plan-driven for its four flows; the specialised
  candidates are the Rust env helpers, the wheel forge and release-wheel
  selection, plus status/binary resolution. Planned as a `OneCatRustMixin`
  extraction in the same five-step pattern used for V100.
- **1Cat Rust helpers extracted (2026-11-03)**: nine Rust-toolchain helpers
  moved verbatim into an in-module `OneCatRustMixin` (deliberately same
  module - they reference module constants; a separate module broke five
  source-build tests). 1Cat suites 34; slice **379**; full **1956 passed,
  8 skipped**.
- **1Cat wheel-forge/release helpers extracted (2026-11-03)**: `_ensure_rust`,
  `_build_source_wheels`, `_fetch_release`, `_select_release_wheels` joined
  the same in-module mixin verbatim. Left class-owned: four plan flows,
  retry, status, binary resolution, handler wiring. 1Cat suites 34; slice
  **379**; full **1956 passed, 8 skipped**.
- **1Cat step 3 mapped (2026-11-03)**: the four recipes already exist; the
  remaining work is eight `_extra_plan_handlers` entries
  (`pip.install_requirements`, `artifact.build_wheels`, `env.compose_ccache`,
  `toolchain.ensure`, `prereqs.resolve_release_assets`, `fs.prune_files`,
  `pip.uninstall`, `fs.ensure_dir`) mapping to the mixin helpers. No code
  changed; exact task list recorded in R39.
- **Recipe wiring harness (2026-11-03)**: `test_engine_recipe_wiring.py`
  validates every recipe's `requires`/`*_from` references against producers
  and (for the migrated family) the known primitive set - 44 tests, landed
  before the 1Cat recipe rewrite so that rewrite is checkable. Engine slice
  **423**; full **2000 passed, 8 skipped**.
- **Review fixes + 1Cat normalization decision (2026-11-03)**: wiring
  harness now fails loudly, checks inputs before outputs and no longer keeps
  a primitive allowlist; structural/source-layout parity pins removed. The
  1Cat `env.compose_toolchain` rename was rejected by the closed primitive
  vocabulary and reverted; the correct normalization is to FOLD ccache into
  the generic `env.compose` (one step, `build_env`, no new vocabulary).
- **1Cat env contract NORMALIZED (2026-11-03)**: `compose_env` now produces
  the shared `build_env`; `env.compose_ccache` layers ccache on top and
  declares no output (duplicate produces are rejected). Scoped strictly to
  the two 1Cat compose steps after a global replace over-reached. 1Cat+wiring
  73; slice **416**; full **1993 passed, 8 skipped**.
- **OneCatInstaller registered (2026-11-03)**: definition-driven class with
  inherited flows, `_compose_build_env` and seven specialised handlers
  (toolchain, wheels, release assets, requirements, uninstall, ensure_dir,
  prune). Adapter/route surface now uses it; legacy class remains only for
  its transcript tests. 1Cat+wiring 78; slice **416**; full **1993/8**.
- **1Cat cutover REVERTED (2026-11-03)**: the definition-driven installer ran
  its recipes, but the route/task harness leaked supervisor state across
  tests (12-15 failures; each passed in isolation). Reverted to the R37
  state (Rust mixin only; legacy manager registered). Kept: generic
  `_source_dist_dir` hook and the sglang `input.dist_dir` context pin.
  Recipe gaps found and to re-apply: source `capture_commit` + finalize
  requires it. Slice **416**; full **1974 passed, 8 skipped**.
- **Task-id harness isolation hardened (2026-11-03)**: the autouse fixture
  now quiesces any manager instance it finds (operation/current task/
  progress id) before and after each test - that leaked state was what made
  the 1Cat cutover fail across tests. File passes 28/28; slice **416**;
  full **1974/8**.
- **1Cat attempt #2 (2026-11-03)**: with R47's isolation, task-id tests
  (28) passed on the new class; deleting the legacy class then surfaced a
  SUPERVISOR resource-key conflict (direct-manager tests leave supervisor
  operations registered; quiesce clears instance state only) and the
  re-application script duplicated the class. Reverted to the clean file.
  Next: quiesce must also `finish_operation` leftovers; re-apply from a
  clean file with an idempotent script.
- **1Cat attempt #3 (2026-11-03)**: quiesce now calls `finish_operation`, the
  subclass re-applied idempotently, and task-id tests passed 28/28 with it
  registered. Deleting the legacy class re-triggered the 15-failure cascade:
  the supervisor resource admission apparently survives the fixture cleanup
  for the direct-manager cases. Reverted again; next needs instrumentation of
  `_admit_build_or_install` to find the lingering admission.
- **Supervisor admission leak CLOSED (2026-11-03)**: direct-manager tests
  build non-singleton managers, so instance quiescing could not see their
  task ids while their supervisor rows/claims persisted and blocked later
  tests. A new autouse fixture force-releases every `_resources` owner via
  `finish_operation(..., "failed", "test cleanup")`. task-id file 28/28;
  slice **416**; full **1974/8**.
- **1Cat CUTOVER COMPLETE - legacy class DELETED (2026-11-03)**:
  `OneCatInstaller` serves routes/adapters; wheel/release helpers live in the
  mixin; seven specialised handlers; `MANAGER_NAME` re-set AFTER
  `super().__init__` (the base overwrites it with the engine id, which broke
  task-id shape); 1Cat lifecycle added so source installs resolve the repo.
  `OneCatVllmInstaller`/`OneCatVllmManager` are aliases. Task-id 28/28; 1Cat
  suites 43; slice **416**; full **1974/8**.
- **Six review findings fixed (2026-11-03)**: (1) 1Cat installs only the
  requirement files its checkout ships; (2) detection includes the older
  `vllm` distribution; (3) release rows + snapshots keep `release_tag` and
  retries prefer it; (4) rust bootstrap env updates reach wheel builds;
  (5) `/api/llama-versions` lists definition-registered engines; (6) removal
  copy describes active-version removal. Slice **418**; backend **1975/8**;
  frontend **611**.
- **1Cat release install_type (2026-11-03)**: _release_meta now emits
  install_type: release with the tag, in the row and the snapshot. Backend
  **1976/8**.
- **Phase 7 scoped (2026-11-03)**: audio.cpp (manager.py 921 lines; 4,851
  across the module) is NOT a python-venv engine - the venv host does not
  apply. Its release/prebuilt path is already described by artifact/archive
  recipes; source builds are class-owned cmake. Phase 7 needs a
  GenericNativeInstaller host (shared with the native llama engines) as
  its own tranche.
- **Native host feasibility audit (2026-11-03)**: 7 of audio's 11 recipe
  primitives are already generic (archive/artifact.locate/fs); only
  rtifact.materialize_cli, rtifact.remove_bundle, publish.install,
  publish.forget need handlers. The native host is therefore a small,
  bounded tranche and would also cover the native llama engines.
- **Phase 7 first extraction (2026-11-03)**: native binary discovery
  (_binary_candidates/_find_binary) moved to
  ackend/engines/native_support.py with static delegates; audio suites
  42; slice **418**; backend **1976/8**.
- **native_support pinned (2026-11-03)**: direct unit tests for the shared
  binary discovery (layout order, bin preference, exe suffix, typed error).
  Slice **421**; backend **1979/8**; ruff 71.
- **Phase 7 host core (2026-11-03)**: `backend/engines/python/native.py` ships
  `GenericNativeInstaller` — definition-driven flows on
  `install_dir`/`build_dir` (no venv/pip), the four R55 handlers
  (`publish.install`/`publish.forget`, `artifact.materialize_cli`/
  `artifact.remove_bundle`) with an install-root removal guard, a synchronous
  core returning `{message, task_id}` (no background tasks yet), and
  `status()`. Pinned by `backend/tests/test_generic_native_installer.py`
  (15, synthetic `nativecraft`, both executor module copies patched).
  Review fix: version naming normalized AFTER uniqueness, so a 64-char
  version lost its collision suffix and reused the existing install dir —
  naming now normalizes first and reserves room for the suffix
  (`test_long_version_names_keep_their_uniqueness_suffix`). Audio is
  deliberately NOT wired yet; the audio manager's missing `native_support`
  import (R56 leftover) was repaired, which shifted literal line numbers and
  required regenerating `docs/engine-parity-inventory.{json,md}` (only that
  file's line arrays changed; markdown identical). Engine slice **436**;
  parallel full **1994 passed, 8 skipped**; ratchet no growth; parity current
  (exit 0); defcheck 12; ruff 73.
- **Phase 7: audio.cpp prebuilt on the native host (2026-11-03)**: first
  engine wired. `AudioCppInstaller` supplies only the audio deltas (streaming
  download, companion CLI materialisation, plan-derived row provenance);
  `_prebuilt_task` awaits `execute_operation` and the route-local
  `_PrebuiltPlanHost` / `_execute_prebuilt_plan` are deleted. The host gained
  `_extra_plan_handlers` (extras win), `_row_extras` and `execute_operation`
  (externally owned task; starts/finishes no operation), and the definition
  now declares the row's active-path fields via `<field>_from`. The prebuilt
  contract file keeps every progress/row/error pin and adds the published
  snapshot + a structural "own plan on the host" pin. Engine slice **437**;
  parallel full **1996 passed, 8 skipped**; parity inventory regenerated
  (current, exit 0); ratchet no growth; defcheck 12; ruff 75.
- **Phase 7: audio.cpp removal on the native host (2026-11-03)**: second
  audio flow wired. `get_native_installer(engine)` (data-driven adapter) lets
  the shared removal route run the engine's OWN `remove_version` recipe on
  `AudioCppInstaller`, so audio's bundle step (`artifact.remove_bundle`,
  moved verbatim from the manager) and `publish.forget` are host-owned;
  `AudioCppManager.delete_version_files` and the route's `remove_bundle`
  closure are DELETED. `execute_operation(store=...)` binds the caller's
  store, `publish.forget` honours `input.registered`, and the 400 mapping for
  bundle refusals is preserved (plus the pinned tree-first ordering). Removal
  pins re-derived (outside-root refusal with the real host) and a structural
  pin added. Engine slice **438**; parallel full **1997 passed, 8 skipped**;
  parity inventory regenerated (current, exit 0); ratchet no growth; defcheck
  12; ruff 78.
- **Phase 7: one streaming build runner + cmake/smoke primitives
  (2026-11-03)**: `backend/engines/native_build.py` holds the plan's single
  process runner (`run_streaming_command`, extracted verbatim from audio's
  `_run_streaming`; audio now delegates to it in production). The native host
  exposes `cmake.configure` / `cmake.build` / `smoke.test` on top of it
  (argv/cwd/env from the recipe context, RuntimeError -> typed primitive
  failure), so the source recipe can be written next. Pinned by
  `test_generic_native_build.py` (7, real subprocesses). Engine slice **445**;
  parallel full **2004 passed, 8 skipped**; parity inventory regenerated
  (current); ratchet no growth; defcheck 12; ruff 80.
- **Phase 7: audio.cpp source build on the native host (2026-11-03)**: third and
  last install flow wired. The exported `source` recipe owns the step order
  (workspace checkout → configure → build → locate → workspace publication →
  `--help` gates → commit capture → publish); `AudioCppInstaller` supplies the
  engine deltas (managed `BuildWorkspace` or plain clone, the manager's
  extracted `_prepare_configure`/`_build_argv`, layout publication, the
  binary gate, source-row extras); `_build_task` runs it via
  `execute_operation` and keeps task/notification/failure/activation
  semantics. `AudioCppManager.build_source` is DELETED (sync keeps
  `_compile_tree` until its own slice); the v2 membership pin now reads
  `{prebuilt, remove_version, source}`. A structural source pin joins the
  prebuilt contract file; the two tests that faked `manager.build_source` now
  drive the real host with step stubs. Broad subset **218**; audio contract
  **11**; engine slice **446**; parallel full **2006 passed, 8 skipped**;
  parity inventory regenerated (current); ratchet no growth; defcheck 12;
  ruff 82.
- **Phase 7: audio.cpp sync on the native host — audio COMPLETE
  (2026-11-03)**: fourth flow wired. A `sync` recipe (host.inspect → vcs.sync →
  configure → build → locate → seal → smoke gates → commit capture →
  `publish.record_update`) runs on `AudioCppInstaller`; the new
  `publish.record_update` handler merges a rebuilt row in place
  (`installed_at` preserved, `updated_at` moved, activation left to the route),
  and the shared `emit_build_progress` keeps audio's exact `build_progress` SSE
  payload. `AudioCppManager`'s build/sync machinery (`sync_source`,
  `_compile_tree`, `_capture`, `_run_streaming`, `_emit`,
  `_raise_if_cancelled`, `_terminate_active_process`, `_sync_git_checkout`) is
  DELETED; the manager keeps build configuration and sync-target helpers, plus
  a module-level `sync_git_checkout`. Membership pin:
  `{prebuilt, remove_version, source, sync}`. Audio now has release, source,
  sync and removal all on the host. Subset **132**; engine slice **447**;
  parallel full **2007 passed, 8 skipped**; parity inventory regenerated
  (current); ratchet no growth; defcheck 12; ruff 83.
- **Native-family publication CLOSED; installer-host Phase 7 complete
  (2026-11-03)**: the recipe's `publish.finalize` now owns llama/ik row
  publication (`run_source_build(..., publish_row="install"|"update",
  source_ref_type, repository_source, store)`), writing the install row
  (labels, commit from the published checkout, install provenance) or merging a
  sync row in place (`installed_at` preserved, `updated_at` moved). The route's
  two row writes and the dead `_llama_checkout_commit` helper are deleted; the
  characterization suite gained a `RecordingStore` + six publication pins
  (fork/upstream labels, install row, sync update, missing-row error,
  publication-off default) against the REAL handler. NOTE: this completes the
  campaign's installer-host tranche; the refactoring plan's Phase 7 ("migrate
  audio services/contracts") is scoped in campaign R67 and NOT started.
  Also FIXED a pre-existing cross-test isolation defect: store-IO failure
  history is process-global, so a fault-injection test's durability failure
  leaked into `test_launchcraft_journey`'s clean-status assertion under xdist
  (three consecutive parallel runs failed on it; isolation passed). A new
  autouse `conftest` fixture clears `store_io` persistence events around every
  test — the campaign's long-standing "rotating flake" is retired and the
  parallel lane is green twice in a row, journey included.
  Engine slice **447**; full parallel **2011 passed, 8 skipped** (twice);
  ratchet no growth; parity inventory current; defcheck 12; ruff 85.
- **R5 landed (2026-11-03): launch policy is engine data**. `EngineSpec.
  launch_policy` carries CPU support, GPU-only env invariants and the
  runtime-context key (moved verbatim from the proxy sets); schema-v2
  `launch_policy` validates + round-trips it and the exporter writes it into
  the generated definitions; `launch_spec`/`commands` read it from the spec and
  `_CPU_REJECTED_ENGINES`/`_GPU_ONLY_INVARIANTS`/`_RUNTIME_CONTEXT_KEYS` are
  deleted. Pinned by `test_engine_launch_policy.py` (9). Definitions + launch
  parity **136**; engine slice **456**; parallel full **2020 passed, 8
  skipped**; parity inventory regenerated (current); ratchet no growth;
  defcheck 12; ruff 88.
  The deferred policy-data item is closed; native GGUF launch remains legacy by
  decision (R23) and native snapshot retries are outside this campaign (R57).
- **Plan Phase 7 (audio services/contracts) scoped, NOT started
  (2026-11-03)**: review feedback corrected the record — the installer-host
  tranche is complete, but the plan's Phase 7 (discovery, models, voices,
  routing/transforms, model profiles, activation capability review, task/request
  surfaces) remains untouched. Campaign R67 inventories it (~4.2k engine-service
  lines, of which only `build_options.py`/`tracking.py` are static catalogs;
  ~2.5k lines in `backend/audio/` plus `model_catalog/audio_cpp_provider.py`
  and `services/audio_model_installer.py`) and lays out four verified slices:
  `build_options` → definition data; a declared probe surface; typed extension
  routes that preserve the scanned contract/voices payloads byte-for-byte; and
  the model-management surfaces. Native GGUF launch (R23) and native
  snapshot-backed retries (R57) remain explicit exceptions.
- **Plan Phase 7, S1 landed (2026-11-03)**: audio's build-option catalog is
  engine data — `EngineSpec.build_options` carries the 23-option table/6
  categories/build types/default/primary keys (moved verbatim), schema-v2
  `build_options` validates + round-trips it, the exporter writes and verifies
  it, and `audio_cpp/build_options.py` is behavior-only (derives dataclasses
  from the spec; the backend toggle is discovered via `special: backend` instead
  of a hardcoded `cuda` key). Payload/coercion/cmake argv are byte-identical.
  4 new pins (definition carries the catalog + payload exclusion; module renders
  an arbitrary spec; validated round-trip; seven malformed-catalog rejections).
  Audio+definitions subset **138**; engine slice **456**; parallel full **2024
  passed, 8 skipped**; ratchet/parity/defcheck/ruff green.
- **Plan Phase 7, S2/S3/S4 landed — phase complete (2026-11-03)**: the rest of
  audio's static surface is engine data now:
  - S2 `protocol` (inspect-probe flags + artifact path contract: builtin family,
    root GGUF name, suffix, id regex) — `inspect.py`/`artifact.py` read it;
  - S3 `extensions` — the engine-owned endpoints beyond the lifecycle verbs are
    DECLARED (`/contract`, `/status`, `/build-options`, `/build-settings`,
    `/community-voices*`) and a gate test proves the router actually serves
    every declared path+method;
  - S4 `request_defaults` — task/transcription/speech key vocabularies are data,
    consumed by the normalizers through `audio_cpp/request_vocab.py`.
  All preserved surfaces stayed byte-identical (audio request/voice/model-config
  suites untouched). Runtime-derived services (contract composition, voices,
  discovery, activation) and the model-management algorithms stay CODE by
  design — they compute from the installed binary/model_specs, so there is no
  static surface left to declare there; phase 8 owns the remaining consumers.
  10 new pins (`test_audio_protocol_contract.py`): spec+definition data,
  payload exclusion, arbitrary-spec derivation for probe/artifact/vocab, the
  served-extension gate, round-trip, nine rejection cases. Audio+definitions+
  request subset **185**; engine slice **466**; parallel full **2034 passed,
  8 skipped**; ratchet no growth; parity current; defcheck 12; ruff 97.
  Phase 7 exceptions unchanged: native GGUF launch (R23), native snapshot
  retries (R57).
- **Review corrections — "Phase 7 complete" WITHDRAWN (2026-11-03)**: campaign
  R70 fixes two real defects and corrects the record.
  1. `transcription_defaults.py` now honours the declared `bool_keys` (it had
     hardcoded `stream`, so a declared `return_text` was dropped and an
     undeclared `stream` accepted); pinned with a fake vocabulary.
  2. `extensions` is now load-bearing: `definitions/extensions.py::
     assert_extensions_served` runs at the audio router's import (fail-fast on
     drift) and is the single implementation the gate test uses. Documented
     limit: it ENFORCES the declaration, it does not replace the handlers.
  3. Not complete, still bespoke (named, not implied): audio discovery/service
     selection (`audio_cpp/discovery.py`, `model_catalog/service.py` audio
     provider selection, `backend/audio/model_config.py`,
     `services/audio_model_installer.py`); `${studio_audio_config}`
     launch-manifest materialization (`audio_cpp/runtime.py` +
     `proxy/launch_spec.py`, legacy launch per R23); runtime contract/voices/
     activation payload discovery. These are the next slices (phase 7 remainder
     / phase 8).
  Verified after the fixes: pins+request suites **59**; engine slice **467**;
  parallel full **2035 passed, 8 skipped**; ratchet/parity/defcheck/ruff green
  (98 curated files).
- **R71: two named remainders were data — the rest is recorded, not claimed
  (2026-11-03)**: `EngineSpec.catalog` now declares each engine's catalog
  providers and `model_catalog/service.py::_provider_ids` reads it (no more
  GGUF/HF id-set branching); `protocol.launch` declares
  `${studio_audio_config}` (flag + macro) and `audio_cpp/runtime.py` builds the
  manifest from it. 3 new pins. PLAN STATUS (see campaign R71): phases 1-5 done;
  phase 6 done for the five pip engines with native GGUF legacy BY DECISION;
  phase 7 installer host + static surfaces done, runtime payload services and
  model-management algorithms remain code; **phase 8 NOT done** (frontend keyed
  state/descriptors/forms, model compatibility, source identity, validation,
  recovery, orphan roots, diagnostics, startup wiring — only catalog selection
  moved); **phase 9 PARTIAL** (onboarding proven, literal ratchet is the check,
  legacy per-engine implementations NOT removed). The plan is NOT finished.
  Engine slice **470**; parallel full **2038 passed, 8 skipped**; ratchet/parity/
  defcheck/ruff green (100 curated files).
- **R72: phase-8 source identity done (2026-11-03)**: `CANONICAL_REPOSITORY_URLS`
  is deleted from `backend/repo_identity.py`; `EngineSpec.upstream` +
  schema-v2 `upstream` declare each engine's canonical repository URL, exported
  into the definitions and read by `canonical_repository_url`, so fork labels
  work for YAML-only engines too. 6 new pins; the two tests that imported the
  map now read the declaration. Engine slice **476**; parallel full **2044
  passed, 8 skipped**; ratchet no growth (nine literals retired); parity current;
  defcheck 12; ruff 102. Phase 8's remaining items (frontend surfaces, model
  compatibility, validation, recovery, orphan roots, diagnostics, startup
  wiring) and phase 9's legacy removal stay OPEN.
- **R73: phase-8 settings/validation done (2026-11-03)**: venv install-settings
  defaults (`source_repo`/`source_branch`/`pip_version`|`release_version` for
  lmdeploy/1cat_vllm/sglang/sglang_v100/vllm) moved from
  `backend/venv_install_settings.py`'s `_ENGINE_DEFAULTS` literal map onto
  `EngineSpec.settings_defaults`, exported into the definitions and read from the
  spec (`_ENGINE_DEFAULTS` remains only the authoritative-registration store, so
  startup wiring and the six patching tests keep working). 5 new pins; the
  `to_spec()` passthrough was the one gap the first run exposed. Engine slice
  **481**; parallel full **2049 passed, 8 skipped**; ratchet no growth (five
  literals retired); parity current; defcheck 12; ruff 103. Next queue item:
  action-recovery keys + orphan-root discovery from declarations.
- **R74: phase-8 recovery/orphan roots done (2026-11-03)**:
  `engine_ids_with_install_kind(kind)` reads `EngineSpec.install_kind`;
  `lifecycle.py`'s `_PYTHON_VENV_ENGINES` (venv retry rule + orphan discovery)
  and `routes/llama_versions.py`'s `_CMAKE_BUILD_ENGINES` (cmake-editability)
  are now data-derived helpers. Generated families resolve exactly as before; a
  YAML engine declaring the kind gets the behavior (fake-spec pins prove it).
  `action_recovery.py` needed no change — its keys were already
  engine-id-agnostic; the pins re-confirm that. 4 new pins; the first lane run
  caught a cross-test order-sensitivity (authoritative registrations widen the
  derived sets) and the pin now filters to GENERATED_ENGINE_IDS. Engine slice
  **485**; parallel full **2053 passed, 8 skipped**; ratchet no growth (two id
  sets retired); parity current; defcheck 12; ruff 104. Next queue item:
  diagnostics/startup wiring leftovers.
- **R75: phase-8 diagnostics/startup wiring done (2026-11-03)**: diagnostics have
  no engine-id branches (audited: `diagnostics.py` + the `/api/diagnostics/bundle`
  path are allowlisted exports), so nothing was migrated there. The real
  leftover was startup wiring: `main.py::ensure_data_directories`' hardcoded
  engine folder list is now `EngineSpec.data_dirs` + schema-v2 `data_dirs`
  (relative-path validated), exported into the definitions, created from the
  union of declarations, and created again idempotently after YAML lifecycle
  registration so onboarded engines get their folders. 6 new pins. Engine slice
  **491**; parallel full **2059 passed, 8 skipped**; ratchet no growth (nine
  startup entries retired); parity current; defcheck 12; ruff 105. Next queue
  item: model compatibility from descriptors/spec.
- **R76: phase-8 model compatibility done (2026-11-03)**:
  `EngineSpec.model_compat` + schema-v2 `model_compat` declare
  `default_for_formats` (lmdeploy -> safetensors) and `prepared_bundle_formats`
  (audio -> mixed/original); exported into the definitions and verified.
  `models/config.default_engine_for_format` and
  `models/schema._default_package_kind` read the declarations instead of
  branching on `safetensors`/`engine == "audio_cpp"` (the family rule and the
  generic gguf/safetensors rules are unchanged). 8 new pins, including fake
  engines claiming a format and declaring bundle formats. Engine slice **499**;
  parallel full **2067 passed, 8 skipped**; ratchet no growth; parity current;
  defcheck 12; ruff 106. Next queue item: frontend descriptor-driven views and
  keyed stores.
- **R77: phase-8 frontend vertical started (2026-11-03)**: the engine-readiness
  filter no longer special-cases `audio_cpp`. `audio_cpp` declares
  `launch_policy.cpu_supported: true`, `to_api_dict` exposes `cpu_supported`
  (default false; the policy block stays out), and
  `frontend/src/composables/engineReadiness.js` matches audio by declared tasks,
  excludes text by the task set, and gates CPU by `cpu_supported`. Every other
  engine's filter behavior is byte-identical (gguf engines pass via formats; YAML
  engines without declarations stay excluded). The `sm70` id branch remains and
  is recorded as the rest of this vertical (the payload has no GPU-arch fact
  yet). Frontend gate: `npx vitest run` on readiness + EnginesView.tasks +
  useAudioContract -> **26 passed**; backend pins 4; engine slice **503**;
  parallel full **2071 passed, 8 skipped**; ratchet/parity/defcheck/ruff green
  (107 curated). Container note: clear `frontend/node_modules/.vite-temp` once if
  an earlier root run left it root-owned.
- **R78: phase-8 readiness vertical completed (2026-11-03)**: `launch_policy`
  gained `gpu_archs` (sglang_v100 + 1cat_vllm declare `["sm70"]`), the descriptor
  payload exposes `gpu_archs` (sorted; empty when undeclared), and
  `engineMatchesFilters` matches the sm70 filter from the declaration — no id
  comparisons remain in the function. The worst-case
  `reference/sglang_v100.yaml` now declares its `launch_policy` so the reference
  tier stays registry-equivalent. Frontend gate: readiness + EnginesView.tasks
  -> **23 passed (2 files)**; backend launch/reference subset **48**; engine
  slice **504**; parallel full **2072 passed, 8 skipped**; ratchet/parity/
  defcheck/ruff green (107 curated). Remaining frontend: keyed stores for the
  nine local engines, descriptor-driven build-settings panels, audio workspace
  branches.
- **R79: phase-8 task-cancel vertical done (2026-11-03)**: `useTaskCancelEndpoint`
  no longer carries an audio literal — the audio descriptor declares `cancelPath`
  and the registry derives `DECLARED_CANCEL_ENDPOINTS_BY_ENGINE` from descriptors
  that have one; the backend's declared extension surface now includes
  `POST /cancel` (enforced at router import). Frontend gate: cancel + readiness +
  store -> **44 passed (3 files)**; backend protocol pins 14; engine slice
  **504**; parallel full **2072 passed, 8 skipped**; ratchet/parity/defcheck/ruff
  green (107 curated). Remaining frontend: keyed stores for the nine local
  engines, descriptor-driven build-settings panels, audio workspace branches.
- **R80: phase-8 build-settings fallback generated (2026-11-03)**:
  `scripts/export-build-defaults.py` writes
  `frontend/src/engines/build-defaults.generated.json` from
  `llama_cpp.build_options`; `frontend/src/engines/buildDefaults.js` exposes it
  and `EnginesView.vue` no longer carries the 28-key "kept in sync" literal or
  the hardcoded build-type list (order kept default-first). Gates:
  `npm run build-defaults:check` in the lint chain + backend pins comparing the
  JSON to `default_build_settings()` and the script render. Frontend 8 (2
  files); pins 3; engine slice **507**; parallel full **2075 passed, 8 skipped**;
  ratchet/parity/defcheck/ruff green (109 curated). Remaining frontend: keyed
  stores for the nine local engines, audio workspace branches, descriptor-driven
  catalog rendering.
- **R81: phase-8 keyed-status vertical done (2026-11-03)**: the store's six
  named status refs (selected by a `statusRef` descriptor field, with no
  consumer outside the store) are replaced by one `engineStatusById` map keyed
  by engine id; `venvFetchStatus` and `fetchAudioCppStatus` write it; the
  `statusRef` fields are gone from the descriptors and `nullSafeStatus` is the
  only status divergence (behavior pinned). The venv/store suites now assert the
  keyed map. One intentional inventory regeneration (the map key adds one data
  literal; the new pin lives under `frontend/src/stores/tests/` so fixture ids
  aren't counted) — ratchet reports no growth afterwards. Frontend **90 passed
  (5 files)**, prettier clean; engine slice **507**; parallel full **2075
  passed, 8 skipped**; parity current; ratchet/defcheck/ruff green (109).
  Remaining frontend: the versions refs (same pattern, larger surface), audio
  workspace branches, descriptor-driven catalog rendering.
- **R82: phase-8 keyed reads done + R81 fallout fixed (2026-11-03)**: the store
  exposes `versionsForEngine(id)` / `statusForEngine(id)`; `EnginesView`'s
  nine-entry `engineVersionLists` map resolves through the descriptor's
  `versionsKey` instead, and `SglangEnginePanel`'s per-engine ternaries are gone.
  **The R81 removal had broken that panel** (it still read the deleted
  `store.vllmStatus`/`sglangStatus` refs) — found by running the WHOLE frontend
  suite for the first time this session; fixed to `statusForEngine`, panel test
  stub updated. New pin `stores/tests/engineKeyedReads.test.js`; server-backed
  store tests now assert `statusRef` is gone. Gates: **full frontend suite 621
  passed (67 files)**; engine slice **507**; parallel full **2075 passed, 8
  skipped**; parity/ratchet/defcheck/build-defaults/ruff green (109 curated).
  Remaining frontend: the versions refs themselves (56 consumer sites), audio
  workspace branches, descriptor-driven catalog rendering.
- **R83: phase-9 legacy removal started (2026-11-03)**: an AST dead-symbol scan
  found four pre-host leftovers in `audio_cpp/manager.py` with zero production
  and zero in-module references — `_safe_slug`, `_valid_repository_url`,
  `AudioCppManager._binary_candidates`, `AudioCppManager._find_binary` — deleted
  along with the imports only they used (`native_binary_candidates`/
  `native_find_binary`, `re`). `installer.py`/`prebuilt.py` were audited the same
  way and have no dead symbols. Pin:
  `test_audio_cpp_manager.py::test_pre_host_dead_helpers_stay_removed`. Manager
  suite 13; engine slice **507**; parallel full **2076 passed, 8 skipped**;
  parity/ratchet/defcheck/ruff green (109). The frontend versions-refs
  conversion was deferred (56 consumer sites, tests assign the refs) and that
  deferral is recorded. Remaining legacy removal: the venv route modules are live
  compatibility delegates until the consumer migration.
- **R84: frontend versions refs descriptor-mapped (2026-11-03)**: the store's
  nine-entry engine-id → ref table is deleted; `VERSION_REFS_BY_KEY` is keyed by
  the descriptor's `versionsKey`, `versionRefFor(engineId)` is the only
  resolution, `fetchLlamaVersions` iterates descriptors, and `versionsForEngine`
  reads through the same mapping. The walk exposed that only two descriptors had
  declared `versionsKey` — all nine do now (the partitioning test caught the
  missing ones immediately). Pin walks the real descriptors and asserts the
  mapping agrees with `versionsForEngine`. Gates: **full frontend 622 passed (67
  files)**; engine slice **507**; parallel full **2076 passed, 8 skipped**;
  parity/ratchet (store id table retired)/defcheck/build-defaults/ruff green
  (109 curated). Remaining: converting the named refs themselves, the audio
  workspace branches, descriptor-driven catalog rendering.
- **R85: architecture check tightened with frozen aggregate caps (2026-11-04)**:
  the literal ratchet compared only against the regenerable inventory, so
  "intended" growth was one `parity-inventory.py` run away from passing.
  `scripts/engine-literals-ratchet.py` now also enforces a frozen `AREA_CAPS`
  budget for 17 areas (1375 literals total; e.g. `backend/proxy/` 76,
  `frontend/src/stores/` 323, `scripts/` 108): regenerating the inventory cannot
  raise a cap, literals in a NEW area fail outright, shrinking stays free. 3 new
  pins (live tree within caps, grouping rule, growth-fails-even-after-regen).
  Gates: ratchet exit 0; engine slice **510**; parallel full **2079 passed, 8
  skipped**; parity/defcheck/build-defaults/ruff green (109 curated). Remaining
  plan work unchanged: frontend refs conversion / audio workspace branches /
  descriptor catalog rendering, and the legacy route removal that depends on
  them.
- **R86: goal-wrap final state (2026-11-04)**: final sweep on the uncommitted
  tree — parity current, ratchet exit 0 (with AREA_CAPS), defcheck 12,
  build-defaults current, ruff curated 109 clean, engine slice **510**, parallel
  full **2079 passed, 8 skipped**, full frontend **622 passed (67 files)**.
  Deliberately remaining, each with its next step: (1) convert the nine frontend
  versions refs (mapping done in R84; ~56 consumers, tests assign them);
  (2) audio workspace branches (behavior-sensitive package detection);
  (3) descriptor-driven catalog rendering (panel still uses the API payload);
  (4) legacy venv route-module removal, blocked on (1)-(3); (5) native GGUF
  launch (R23 decision) and native snapshot retries (R57) remain exceptions.
  The plan is NOT complete; the goal stays active rather than claiming otherwise.
- **R87: post-goal review defects fixed (2026-11-04)**: (1) YAML install defaults
  were overwritten at startup — `register_definition_lifecycles` derived values
  from `lifecycle` and clobbered `settings_defaults` (declared branch `stable` /
  version `1.2.3` became `main` / empty). The derive+merge is now
  `definition_install_defaults()`: lifecycle is the fallback, declared keys win.
  (2) Declared prepared-bundle formats were ignored outside `mixed`/`original` —
  `_default_package_kind` now lets the DECLARATION decide for any listed format
  (so a declared `safetensors` bundle stays `prepared_bundle` and out of generic
  format inference); the legacy family rule keeps its historical formats. 5 new
  pins (`test_engine_yaml_launch_defaults.py`). Engine slice **515**; parallel
  full **2084 passed, 8 skipped**; parity/ratchet/defcheck/build-defaults/ruff
  green (110 curated).
- **R88: post-goal review defects fixed (2026-11-04)**: (3) `EnginesView.vue` still
  read the six named status refs removed in R81 (~20 sites, `undefined` at
  runtime — status details gone, audio tracking fell back to `main`); the view now
  uses three computeds over `statusForEngine('…')`, and the view-test stubs
  (which masked it) model the keyed map instead. (4) `launch_spec._compile_audio`
  only resolved `${studio_audio_config}`; it now derives resolvable tokens from
  the manifest's own `macros` map so ANY declared macro (e.g. `audio_cfg`) binds
  `server.json`. Pin added in `test_launch_manifests.py`. The tightened ratchet
  caught the fix's legitimate +7 in `frontend/src/views/` and the reviewed cap
  bump (369 -> 376) is recorded next to it. Manifest/runtime **61**; engine slice
  **515**; parallel full **2085 passed, 8 skipped**; full frontend **622 passed
  (67 files)**; statics green (110 curated).
- **R89: frontend versions refs converted (2026-11-04)**: the store's source is
  now `engineVersionsById` (engine-id keyed); the nine historic names are
  WRITABLE COMPUTEDS generated from the descriptors' `versionsKey` mapping, so
  every consumer — including tests that assign `store.audioCppVersions = [...]` —
  keeps working, while `fetchLlamaVersions` writes the keyed map and
  `versionsForEngine` reads it. No engine-id → ref table, no per-engine version
  refs. Pins check assignment→map and map→alias both ways plus the fetched
  partition. Full frontend **623 passed (67 files)**; engine slice **515**;
  parallel full **2085 passed, 8 skipped**; statics green (110 curated). This
  was the last item blocking legacy venv route-module removal; remaining
  frontend work: the audio workspace branches and descriptor-driven catalog
  rendering.
- **R90: one audio-package predicate (2026-11-04)**: the same engine-or-format
  check existed five times in four model surfaces; it now lives once in
  `frontend/src/utils/audioModel.js::isAudioPackage`, used by ModelRow,
  ModelLibrary, ModelWarmupButton and AudioWorkspace (their engine-id predicate
  literals are gone; AudioWorkspace keeps its genuine `?engine=audio_cpp` query).
  New matrix pin under `utils/tests/` (the first placement in `utils/` inflated
  the literal inventory and the ratchet caught it — the gate working as intended);
  the consolidated constant is one reviewed literal bump (utils cap 6 -> 7).
  Full frontend **625 passed (68 files)**; engine slice **515**; parallel full
  **2085 passed, 8 skipped**; statics green (110 curated). Remaining frontend:
  descriptor-driven catalog rendering and the deeper ModelConfig audio panels.
- **R91: ModelConfig families from descriptors (2026-11-04)**: the modal icon
  family now uses `SG_PANEL_ENGINE_IDS` (three engines; `SGLANG_RUNTIME_UI_IDS`
  is only two, so the family semantics were checked first) and the
  safetensors->other-engine normalization uses `VENV_ENGINE_IDS` instead of a
  five-id literal. Behavior identical (audio.cpp stays excluded: cmake, not
  venv). Full frontend **625 passed (68 files)**; engine slice **515**; parallel
  full **2085 passed, 8 skipped**; statics green (110 curated). Remaining
  frontend: descriptor-driven catalog rendering and the audio `ModelConfig`
  panels beyond the shared predicate.
- **R92: build-settings defaults declared (2026-11-04)**: new schema-v2
  `build_defaults` + `EngineSpec.build_defaults`; `ik_llama` declares
  `build_examples: true` (exported into its definition) and
  `_apply_engine_specific_build_defaults` layers the DECLARED defaults instead of
  branching on the engine id — same precedence as before (the ik requirement still
  wins over a user `false`). 5 new pins (declaration + exported definition + route
  layering + fake-spec proof + round-trip/rejections). Definition/build-config
  subset **71**; engine slice **520**; parallel full **2090 passed, 8 skipped**;
  statics green (111 curated).
- **R93: source-build repository identity declared (2026-11-04)**:
  `_resolve_engine_build_target` now derives the native source-build family from
  the spec (`install_kind: cmake` + `runtime_kind: llama_server`) and returns
  `spec.repository_source` + the declared `upstream.repository_url` instead of an
  id branch and `llama_manager.REPOSITORY_SOURCES`. Rejections keep their exact
  400s; a YAML-native engine with a declared upstream resolves with no code edit.
  Pin in the characterization suite (llama/ik + rejections + fake spec). Engine
  slice **520**; parallel full **2091 passed, 8 skipped**; statics green (111
  curated).
- **R94: llama-family build catalog is engine data (2026-11-04)**: the 96-option
  table moved verbatim to `backend/engines/llama_build_catalog_data.py` (pure data
  at the engines root — placing it inside the `llama_cpp/` package caused a
  registry/package import cycle the exporter caught), is carried on both
  llama-family specs with per-entry `engines` metadata (schema-validated), and is
  exported into both definitions. `llama_cpp/build_options.py` derives everything
  from the spec block: union defaults for stored configs, per-engine filtering for
  the UI catalog (payload byte-identical), build types/primary keys declared.
  All 43 pre-existing build-option tests pass untouched and the generated
  frontend build-defaults JSON is unchanged. 5 new pins. Catalog/definition
  subset **70**; engine slice **525**; parallel full **2096 passed, 8 skipped**;
  statics green (112 curated).
- **R95: catalog fetch dispatch declared (2026-11-04)**: the three
  catalog-bearing frontend descriptors declare `buildOptionsPath`; the registry
  exposes `buildOptionsPath()` and the store `fetchBuildOptionsForEngine()`
  fetches from the declared path (both view loaders use it; the audio compat
  wrapper delegates). Behavior identical (same URLs + `params.engine`). Pins in
  `stores/tests/buildOptionsDispatch.test.js`; generated-defaults test moved under
  `engines/tests/` and two comments de-literalised so the reviewed cap moved only
  42 -> 44 for the two newly declared endpoints. Full frontend **628 passed (69
  files)**; engine slice **525**; parallel full **2096 passed, 8 skipped**;
  statics green (112 curated).
- **R96: sync/retry source-repo fallbacks declared (2026-11-04)**: the two route
  sites that still resolved a display name through the manager's
  `REPOSITORY_SOURCES` map now use `_spec_repository_source()` and
  `_declared_source_repo()` (declared `upstream.repository_url`; empty on a name
  mismatch). The 400s are unchanged. Pin covers llama/ik values, unknown engine,
  and the cross-engine mismatch. `REPOSITORY_SOURCES` stays the source-build host
  contract (`source_build.py`); the routes no longer consult it. Affected suites
  **44**; engine slice **525**; parallel full **2097 passed, 8 skipped**; statics
  green (112 curated).
- **R97: dead frontend exports retired (2026-11-04)**: a surface audit found 20
  internals exported from `parseCliCommand.js` (regexes, coercers, index builders,
  skip helpers) that nothing imported; they are module-local now (surface 11
  names). `engineDescriptor`, `workspaceTitle` and the audio-package marker became
  module-local; the dead `formatFileSize` alias is removed. New pin
  `utils/tests/parseCliCommandSurface.test.js` asserts the exact export list, so
  adding surface is a reviewed diff. Full frontend **629 passed (70 files)**;
  engine slice **525**; parallel full **2097 passed, 8 skipped**; statics green
  (112 curated).
- **R98: closing sweep + honest status (2026-11-04)**: final gate sweep over the
  uncommitted tree (187 entries, nothing committed) — static gates green: parity
  current, ratchet at frozen caps, defcheck 12, build defaults current, curated
  ruff 112, engine slice **525**, parallel full **2097/8 skipped**. **The frontend
  claim in this entry was WRONG and is corrected in R99** (real result: 629 passed
  + 14 unhandled errors, exit 1). Reached: launch/lifecycle/runtime/profile/
  protocol/extensions/request-defaults/catalog/upstream/settings/build-options/
  build-defaults/data-dirs/model-compat declarations are schema-validated,
  exported, equivalence-checked and read by backend + frontend. NOT done, with
  next steps: (1) audio `ModelConfig` profile panels (needs a panel abstraction);
  (2) removal of the legacy venv route modules — live compatibility delegates for
  external callers, so it is an API-surface cutover decision, not an autonomous
  delete; (3) native GGUF launch (R23) and native snapshot retries (R57) remain
  decided exceptions; (4) tree-wide `ruff format --check` drift outside the curated
  list is pre-existing and not gated. The plan is NOT complete.
- **R99: frontend false-green fixed + pinned (2026-11-04)**: the two EnginesView
  test stubs never gained `versionsForEngine` (added R89), so SGLang panel renders
  threw inside Vue and surfaced as 14 unhandled rejections while the file still
  passed. Both stubs now spread `frontend/src/test-fixtures/engineStoreAccessors.js`
  (the sync accessor contract), and
  `views/tests/engineStoreAccessorContract.test.js` fails if a `.vue` source calls
  a `store.<x>ForEngine` the fixture lacks. Frontend gate now recorded with exit
  code + unhandled-error check: **exit 0, 630 passed (71 files), 0 unhandled**;
  engine slice **525**; parallel full **2097/8 skipped**; statics green (112
  curated).
- **R100: the frontend gate reports its own verdict (2026-11-04)**:
  `frontend/src/utils/frontendGate.js::summarizeFrontendRun` strips ANSI and
  requires exit 0 + no unhandled text + no error section + at least one passed
  test; `scripts/frontend-gate.mjs` (`npm run test:frontend:gate`) runs the suite
  through it. Pins reproduce the R98 false-green output and assert rejection.
  Verified live: `frontend gate: CLEAN (exit 0, 636 tests passed, no unhandled
  errors)`. Statics green (112 curated); engine slice **525**; parallel full
  **2097/8 skipped**.
- **R101: audio config panel declared (2026-11-04)**: the audio descriptor declares
  `configPanel: 'audio'`; the registry exposes `isAudioEngineId()` and every
  `engine === 'audio_cpp'` dispatch in `ModelConfig` (panel, labels, drafts,
  preserved-keys rule, watcher guards) now uses it. API arguments
  (`fetchParamRegistry('audio_cpp')`) and the curated package-kind list stay
  literals by design. New pin `engines/tests/audioPanelDeclaration.test.js`.
  Frontend gate **CLEAN (639 passed, 73 files, 0 unhandled)**; statics green (112
  curated); engine slice **525**; parallel full **2097/8 skipped**.
- **R102: legacy-route cutover inventory (2026-11-04)**: `scripts/legacy-route-inventory.py`
  generates `docs/legacy-route-inventory.md` (module, engine(s), legacy prefix,
  endpoint count, frontend segment + `genericApi`, covering tests) for both router
  shapes. Today: **6 mounts / 5 modules** (lmdeploy, 1cat-vllm, sglang,
  sglang-v100, vllm via the factory; unsloth llama hand-written). `--check` is in
  the lint chain and the engine lane via
  `test_legacy_route_inventory_current.py`, so the cutover decision always has a
  current blast radius. Statics green (113 curated); engine slice **527**;
  parallel full **2099/8 skipped**; frontend gate **CLEAN (639/73/0)**.
- **R103: cutover claim proven on both sides (2026-11-04)**: the R102 extractor
  missed the quoted `'1cat_vllm'` key and let a neighbouring block swallow one
  engine, so two rows falsely read "no generic surface". Replaced with a brace
  scanner + `NATIVE_GENERIC_VERBS` reader; the column is now `venv`/`native`/`no`.
  Correct result: **(4 venv + 1 native)** — all five venv engines route through
  `/api/engines/{id}/{verb}`, unsloth through declared native verbs, and NO legacy
  mount is the only surface for its engine. New pins on both sides: backend
  inventory test fails on any `no` row; frontend
  `engines/tests/genericRouteCoverage.test.js` asserts every venv descriptor is
  `genericApi` and builds generic paths (incl. 1cat-vllm). Engine slice **528**;
  parallel full **2100/8 skipped**; frontend gate **CLEAN (642/74/0)**.
- **R104: literal caps tightened (2026-11-04)**: the frozen area caps had slack,
  so retired literals left room to regrow. Lowered to the measured totals:
  `backend/routes/` 169 -> **162**, `frontend/src/components/` 86 -> **83**,
  `frontend/src/views/` 376 -> **351** (aggregate 1385 -> **1350**). New pin
  `test_engine_literal_caps_tight.py` requires EVERY cap to equal its measured
  total, so a retired literal must be paired with a tightened cap. Engine slice
  **530**; parallel full **2102/8 skipped**; frontend gate **CLEAN (642/74/0)**;
  statics green (114 curated).
- **R105: native source family declared (2026-11-04)**: `native_source_engine_ids()`
  derives the family from the specs (`cmake` + `llama_server`); the route's six
  id-tuple guards and its lifecycle-registration loop use
  `_is_native_source_engine()`. The tight-cap pin then forced `backend/routes/`
  162 -> **152** in the same slice (aggregate 1340). 3 new pins in
  `test_native_source_family_declaration.py`. Engine slice **533**; parallel full
  **2105/8 skipped**; frontend gate **CLEAN (642/74/0)**; statics green (115
  curated).
- **R106: remaining llama families declared (2026-11-04)**:
  `llama_server_engine_ids()` (llama.cpp, ik_llama, unsloth; audio excluded) drives
  the scanner branch; `lifecycle.py` derives its shared-root family from
  `native_source_engine_ids()` and its cmake guard from
  `engine_ids_with_install_kind("cmake")`. `llama_cpp/resolve.py` keeps its ordered
  tuple deliberately — it is a PRIORITY (llama.cpp first), not membership. Pins use
  ⊆/exclusion (an exact-tuple pin failed against hermetic fixture engines, so the
  assertion was corrected, not the code). Engine slice **536**; parallel full
  **2108/8 skipped**; frontend gate **CLEAN (642/74/0)**; statics green (115
  curated).
- **R107: round-budget close-out (2026-11-04)**: final sweep all green — parity
  current, ratchet exit 0 with ZERO cap slack (aggregate **1340**), definitions 12,
  build defaults current, legacy-route inventory current, curated ruff **115**,
  engine slice **536**, parallel full **2108/8 skipped**, frontend gate **CLEAN
  (exit 0, 642 passed / 74 files, 0 unhandled)**. Tree: **196 uncommitted entries**
  (100 modified, 96 untracked). REMAINING, precisely: (1) legacy compatibility
  router cutover — evidence in `docs/legacy-route-inventory.md`, no frontend
  dependency (all `venv`, one `native`), so it is an owner's API decision; (2) R23
  native GGUF launch and R57 native snapshot retries remain decided exceptions;
  (3) one deliberate API-argument literal in `AudioWorkspace`; (4)
  `llama_cpp/resolve.py` keeps its ordered tuple (binary priority). The plan is NOT
  complete; the 32-round budget is spent and this is reported, not claimed.
- **1Cat env.compose blocker RESOLVED (2026-11-03)**: option (A) was taken in
  the 1Cat recipe tranche — its recipes use the generic env vocabulary
  (`env.compose` → `build_env`/`runtime_meta` plus `env.compose_ccache`), so the
  earlier per-engine-precedence question is closed; no decision is outstanding.
## 2. File ledger (uncommitted)

### New (untracked) — engine refactor
- `backend/engines/definitions/` — declarative definition loader/schema/
  executor/profiles/startup-check.
- `backend/routes/engine_lifecycle.py` — generic router.
- `backend/models/engine_contracts.py` — typed contract models.
- `backend/tests/` — 18 new test files (shadow suite, definitions
  equivalence, contracts, generic-router 49+ pins, native-engine 34 pins,
  source-build/removal characterization, literals ratchet).
- `docs/engine-surface-contract-matrix.md`,
  `docs/engine-parity-inventory.{json,md}`, this handoff.
- `scripts/parity-inventory.py`, `scripts/engine-literals-ratchet.py`,
  `scripts/export-engine-definitions.py`.
- `frontend/src/engines/` — the registry.
- `frontend/src/stores/engines.venv.test.js` — registry/fan-out/parity pins.

### Modified (tracked) — engine refactor
- Backend: `backend/main.py` (router include), `routes/engine_versions.py`
  (LIFECYCLE registry + verbs), `routes/llama_versions.py` +
  `routes/unsloth_llama_versions.py` (native verb registrations, shared
  check-updates surface), `routes/audio_cpp_versions.py`, engine
  installers/source_build/prebuilt (contract-matrix-era touch-ups),
  3 integration tests.
- Frontend: `stores/engines.js` (+test), `test-fixtures/taskFixtures.js`,
  `composables/useTaskCancelEndpoint.js`, `composables/engineVersionDelete.js`,
  `views/EnginesView.vue` (+`EnginesView.tasks.test.js`),
  `views/ModelConfig.vue`, `components/system/VersionTable.vue`,
  `components/system/EngineBuildSettingsHint.vue`,
  `frontend/src/api/openapi.json` (exported).
- Tooling: `package.json` (prettier curated list — 14 entries;
  ruff curated list — 55 files), `scripts/git-hooks/pre-commit`.

### Modified/untracked — dev-environment (predates refactor tranches)
- `.devcontainer/`, `docker-compose.dev.yml`, `scripts/dev-init.sh`,
  `.dockerignore`, `README.md`, `docs/README.md`,
  `frontend/package-lock.json` — container-based dev setup; commit or
  discard independently of the refactor.

### Housekeeping done in the audit pass
- Deleted stray 0-byte `=1695` artifact (accidental shell redirect, created
  2026-11-02 during session work).

## 3. Gate matrix (commands + expected)

Frontend gates run on the HOST from the repo root; backend gates run in
the dev container (`docker exec --user appuser llama-cpp-studio-dev
bash -lc 'cd /app && â€¦'`).

| Gate | Command | Expected |
| --- | --- | --- |
| Vitest | `cd frontend; npx vitest run --config vite.config.mjs` | 64 files / **612** passed |
| ESLint | `cd frontend; npx eslint --max-warnings=0 src` | exit 0 |
| Prettier (curated) | root: `npm run format:check` | "All matched files use Prettier code style!" |
| Ruff (curated 61) | container: `RUFF_CACHE_DIR=/tmp/.ruff-cache npm run format:python:check` | `61 files already formatted` |
| Parity inventory | container: `python scripts/parity-inventory.py` | writes json+md |
| Literal ratchet | container: `python scripts/engine-literals-ratchet.py` | "no growth beyond the committed inventory" |
| OpenAPI | container: `npm run openapi:check` | exit 0 |
| Pytest | parallel lane: `npm run test:backend:parallel` | **1965 passed, 8 skipped** |
| Boot smoke | uvicorn on a fresh port; `grep -c validated` on log = 1; `grep -c onboarded` = 1 (YAML fixture era, post-S2) | ready=200, `/api/engines/yamlcraft/status` 200, `pgrep -c uvicorn` = 0 after `pkill -f backend.main:app` — latest #51: both YAML personas status 200 (12 engines) |

### Fast lanes (inner loop; full battery stays the tranche gate)

| Lane | Command (container) | Wall time | Notes |
| --- | --- | --- | --- |
| Engine slice | `npm run test:backend:engine` (definitions/*, routers, native lifecycle, literal ratchet, removal characterization, model contracts, launch interpreter, lmdeploy+vllm installer parity) | ~35s / 331 tests | The whole YAML-campaign surface + routers + contracts + launch interpreter |
| Focused files | `python -m pytest -q -p no:cacheprovider backend/tests/<file>.py â€¦` | seconds–20s | Default while editing one suite |
| Parallel full | `PYTHONPATH=/home/appuser/.pytest-plugins python -m pytest -q -p no:cacheprovider -n 8` (or `npm run test:backend:parallel`) | ~90s | pytest-xdist, verified green + deterministic (2 runs); plugin lives outside the venv (site-packages is read-only there) — after a container rebuild re-install: `python -m pip install --quiet --target /home/appuser/.pytest-plugins pytest-xdist && cd /home/appuser/.pytest-plugins && rm -rf _pytest pytest py.py pluggy* iniconfig* packaging*` (strip the bundled copies so the venv's own pytest is used) |
| Failed-only | seed once with `-o cache_dir=/tmp/.pytest-cache`, then add `--lf` (repeats failures) or `--ff` (failures first) | varies | Cache lives in /tmp so the container's read-only app dir stays clean |
| Frontend watch | `cd frontend; npx vitest` | incremental | Existing `npm run test:frontend:watch` |

**Policy (owner directive, 2026-11-03): the serial FULL suite is a scarce
resource.** The inner loop and normal verification use the lanes above —
targeted files while editing, the engine slice for the engine surface, and
the PARALLEL full suite (~90s) whenever cross-suite impact is plausible.
Run the serial full suite only (a) when a change is genuinely suite-wide
and no targeted lane covers it, or (b) when the owner asks for a green
stamp. Do not re-run it per step or repeatedly within one tranche; prefer
`--lf`/targeted reruns after a fix, and treat the parallel lane as the
default "everything still passes" check. The serial run remains the
recorded scoreboard number when it is taken.

## 4. What protects behavior (pin inventory)

- `tests/test_generic_engine_router.py` — legacy-vs-generic byte equality
  for the five venv engines across all verbs incl. error statuses and
  family quirks (vllm 502 vs 1cat 500 etc.).
- `tests/test_native_engine_lifecycle.py` — 44 pins: native verb equality
  (incl. response_model replication, required-body 422 replay, error
  paths), the 405 capability contract, and path-derived identity
  (legacy-vs-generic paired bytes for the llama.cpp-default quirk).
- Shadow tests per family — engine-definition/legacy-plan equivalence.
- `engines.venv.test.js` — registry shape, fan-out sequences (14
  activate/delete refresh sequences), CALLS URL matrix, cancel-endpoint
  maps, null-handling split.
- `EnginesView.tasks.test.js` — dialog dispatch/body/header render pins
  (titles, icons, aria presence/absence divergences, toast bytes).
- Characterization-first protocol: pins ran green against OLD code before
  each merge/flip, then UNCHANGED (or updated deliberately, logged in
  matrix Â§8).

## 5. Deliberate divergences preserved (do not "unify" without a decision)

- Native dialog-header icons `pi-sparkles`/`pi-bolt` differ from picker
  icons (vllm header â‰  its picker `pi-server`); trio Reload/Rescan
  buttons carry NO aria/tooltip (`buttonA11y: false`), venv pair's do.
- Settings aria-label: `Install settings` (trio) vs `Build settings`
  (lmdeploy/1cat).
- `build_source` quirk: absent `repository_source` builds llama.cpp even
  when invoked via the ik engine path — carried through unchanged on both
  surfaces.
- Generic required-body 422: legacy distinguishes absent vs null body;
  the replayed guard reports both as `missing` (frontend always sends a
  body). Documented edge.
- 1Cat versions refresh vLLM statuses too (fan-out data); audio refreshes
  on activate only; sglang-family status coerces `data || {}`,
  lmdeploy/1cat assign raw.
- Hint component `:key` prefixes differ per dialog family
  (remount-equivalent, disclosed).

## 6. Rollback notes

- Every legacy per-engine route is LIVE and byte-equal to generic; the
  frontend flip is URL-only. To roll back one engine: set its descriptor
  `genericApi: false` (venv) or remove its `NATIVE_GENERIC_VERBS`
  entries + revert the store call (native) — pins in
  `engines.venv.test.js`/`taskFixtures.js` mark every such decision.
- The pre-commit hook + ratchet gate prevent literal regrowth; inventory
  regeneration is deterministic.

## 7. Remaining work vs the original plan (honest scope)

> Campaign plan drafted 2026-11-02: `docs/yaml-engine-onboarding-campaign.md`
> (four blockers B1–B4, gates G1–G5 for the `yamlcraft` fixture engine,
> slices S1–S5, open decisions Q1–Q4). Items below are the scope; the
> campaign doc is the execution plan.

The consolidation shipped so far is real, but the plan's central promise —
onboarding an engine through YAML alone — is NOT yet delivered. What
remains, in the order recommended by the 2026-11-02 external review:

1. **Authoritative YAML definitions.** Startup still rejects YAML engines
   absent from the Python registry; definitions mirror the registry rather
   than own registration. Needs install-definition digests and recorded
   recipe versions, plus a startup acceptance path for definitions-only
   engines.
2. **Path-derived engine identity on the generic API + executor cleanup
   guarantee** (fixed 2026-11-02 tranche, see Â§1 bullet) — the review's
   correctness findings #1/#3.
3. **Frontend de-hardcoding.** Engine identity and the native verb
   capability map are still registry code, not definition-derived.
4. **Runtime/discovery migration.** Profile compilation remains
   reference/shadow work; nothing consumes definitions at runtime yet.
5. **Specialized installation.** Engine-specific installer callbacks
   still own the behavior; specialized primitive extraction (the plan's
   cmake/CUDA/patch steps as definitions) has not started.
6. **audio.cpp** — services, dialog body, and generic-router membership
   (matrix Â§7 defers deliberately; flag-gated/drift-aware surface).
7. **Proof target: one unfamiliar engine end to end through YAML only.**
   This is the test of whether the architecture pays off; more migration
   should not accumulate before it passes.
8. Known cosmetic: pydantic UserWarning (`Field name "schema" in
   "AudioContractResponse"`); llama/audio Dict-typed response surfaces;
   teardown shipped-inert.

Verification boundaries stated by the reviewer and carried here: hardware
build and inference parity are NOT verified by this suite; the
ccache-dependent backend test fails on Windows hosts by environment, not
regression.

## 8. Suggested commit slicing (for review)

1. dev-environment (item-2 list in Â§2),
2. backend definitions/shadows/contracts + inventories + ratchet + hook,
3. backend dual-surface router + venv LIFECYCLE convergence,
4. backend native adapters (+ openapi export),
5. frontend registry + store/composable/component consumers,
6. frontend generic-API flips (reads â†’ writes â†’ cancel endpoints),
7. EnginesView slices 1–2b (+ tests),
8. docs (matrix, inventory, handoff).
