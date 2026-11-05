# YAML engine LAUNCH campaign (R-slices) — draft for sign-off

> **Historical evidence — not the release entry point.** Start at the
> [delivery map](delivery-map.md); the [change inventory](change-inventory.md)
> maps this work to the review series.

Status: **SIGNED** (2026-11-03, owner via decision prompt — all four
recommended). Executed R1–R4, gate G5, review follow-ups R6–R8; optional
R5 deliberately deferred with rationale (§3). TWO YAML-only personas now
install, activate, scan, LAUNCH as real processes from YAML alone, serve
streamed chat completions through the Studio proxy, and drain cleanly —
one console-script/flag-binding/`fixture_only`, one
python-module/positional-binding/gguf-declaring. Launch revisions carry
the definition digest, and install retries are backed by
installation-specific install snapshots.
Successor to `docs/yaml-engine-onboarding-campaign.md` (control plane,
S1–S6.1 shipped). The review verdict driving this one: "runtime
launch/discovery … actual inference proof" remain the end-to-end gap.
Never edit `docs/config-driven-engine-refactoring-plan.md`; this doc
implements its launch provisions (retain `LaunchSpec`/`EnvSpec`/
manifests/revisions per plan lines 35–37; reuse scanner-kind dispatch and
params-catalog versioning, not parallel machinery).

## 1. Ground truth (dual survey, 2026-11-03, file:line verified)

- **One server runtime for all nine engines**: the Studio proxy.
  Chain: model start (POST `/api/models/{id}/start` → `routes/models.py:2188+`)
  or lazy dataplane acquire → `StudioProxyManager._launch` →
  `compile_model_runtime` → **`_compile_launch` if-chain
  (`proxy/launch_spec.py:301-344`)** → structured `LaunchSpec` →
  manifest stage → `ProcessRegistry._start`
  (`proxy/studio/registry.py:416-477`, `subprocess.Popen`,
  `start_new_session=True`, ephemeral loopback port) → uniform
  `GET /health` readiness (`registry.py:479-517`;
  `ENGINES_WITHOUT_HEALTH_ROUTE` deliberately EMPTY) → `killpg`
  drain/terminate. Supervision, receipts, journals, idle-reap: all
  engine-agnostic already.
- **Where a YAML-only engine dies**: `launch_spec.py:341`
  `RuntimeError: No runtime adapter registered for engine` — the only
  wall. Everything upstream (VALID_ENGINE_IDS membership — shared
  mutable, YAML ids pass), and everything downstream (spawn/health/
  supervise/drain) already works for unknown ids. TRAP: a YAML engine
  declaring `gguf` artifact_formats would silently fall into
  `_compile_gguf` via the derived GGUF set — fixtures must keep a
  non-gguf format (our `fixture_only` already does).
- **The designed target already exists, dormant**: schema-v2
  `runtime_profiles` (executable/argv_prefix/model_binding/
  port_binding/env/cwd) + `definitions/profiles.py`
  `compile_profile_argv/env`, test-guarded UNREFERENCED at runtime with
  argv **byte-equal to `_compile_module`/`_compile_lmdeploy`**
  (`test_engine_definitions_v2.py`), and reference YAML encoding the
  byte-exact lmdeploy + sglang_v100 (incl. CUDA env) contracts. This is
  the plan's Phase-6 launch compiler seam — activate, don't invent.
- **scanner_kind is REAL** (`SCANNER_KIND_DISPATCH`,
  `scan/scanner.py:1339-1385`, kind-keyed already); **runtime_kind is
  cosmetic** (zero production readers). Param→argv emission is already
  data-driven (`_emit_param_tokens` value-kind vocabulary); only the
  argv SKELETON is hardcoded.
- **Start/stop is MODEL-scoped** (`/api/models/{id}/start|stop`,
  preload/unload/reload) — NO engine-scoped server verbs exist for any
  engine; "launch a YAML engine" == run a MODEL on it through the proxy.
- Literal budget (ratchet-scoped): launch_spec.py 44, commands.py 34,
  activation.py 12, plus GPU-policy sets (`_CPU_REJECTED_ENGINES`,
  `_GPU_ONLY_INVARIANTS`), `_RUNTIME_CONTEXT_KEYS`, CAPCOMPAT. Platform:
  killpg/proc = Linux; all launch tests run in the container.
- Already engine-free (verified): launcher.py, manifests.py, control.py,
  runtime_observation.py, studio/manager.py, model_runtime_apply.py,
  routes/runtime.py, dataplane.py.

## 2. Gates

- **G1 Interpreter purity**: a YAML-only engine launches with ZERO
  engine-id literals in any new module; the definitions-import cutover
  allow-list grows exactly once, deliberately.
- **G2 Byte-frozen nine**: nine launch paths never traverse the new
  interpreter (dispatch ORDER guarantees it — new branch fires only for
  definition-bearing ids lacking a legacy adapter); preview cmd,
  manifest bytes, revision shas, generic-router payloads unchanged
  (existing goldens + full battery).
- **G3 Real process from YAML**: offline journey — install → activate →
  model start spawns a real process via a LaunchSpec whose argv/env are
  exactly `compile_profile_argv/env` output for the engine's YAML
  profile; `/health` reaches 2xx; `/running` + `/api/status` show it.
- **G4 Proxy inference**: an OpenAI-style chat round-trip streams
  THROUGH the Studio proxy to the YAML-launched server; stop drains,
  `killpg` leaves zero orphans; journal = succeeded.
- **G5 Discovery**: scanner dispatch produces a param catalog for the
  YAML id (existing scanner_kind reused); `runnable` flips by
  `active_path_fields` only; param registry serves it to ModelConfig.
- **G6 Budget**: literal ratchet green throughout; `launch_spec.py`'s
  count never grows (new branch counts 0 ids — keyed on definition
  presence + `executable.kind` vocabulary only).

## 3. Slices (each characterization-first, RED pins recorded)

**R1/R2 — DONE 2026-11-03** (shipped as ONE change: the interpreter is
not meaningfully testable without its executable/env resolvers).
`backend/proxy/profile_launch.py` (zero engine-id literals —
parity inventory regenerated BYTE-IDENTICAL) + post-chain dispatch
branch in `_compile_launch` + definition-aware `_use_model_name`
(hf-source profiles mirror the venv family's proxy model-name contract;
GGUF-family ids never consult definitions — the gguf-declaring-YAML
trap stays owned by `_compile_gguf`). Cutover allow-list grew exactly
once (deliberate, commented). 14 tests in `test_profile_launch.py`
including the structural gate: with a fake registry answering EVERY id
and a booby-trapped interpreter, all nine GENERATED ids still dispatch
to their legacy adapters (gate G2 = order + gate, not hope); legacy
"No runtime adapter" bytes pinned; typed-port-placeholder, positional/
flag bindings, PATH composition, cwd=venv_root from the executable
layout, env.required preflight = typed LaunchCompileError(field=NAME)
satisfiable by process env OR user `runtime_env` config. Battery:
2206 passed / 9 skipped, ratchet + inventory unchanged, smoke #45.

**R3 — DONE 2026-11-03** (the first draft had one self-inflicted YAML
newline bug — my own edit glued a comment to `schema_version:` and the
battery caught it instantly; disclosed): the launchable fixture + REAL
journey.
- `backend/engines/definitions/launchcraft.yaml` — shipped,
  authoritative, `fixture_only` format (never GGUF-routable), with the
  launch contract declared ENTIRELY in YAML: `runtime_profiles.default`
  (venv_console_script `launchcraft-server`, `--model`/huggingface_id,
  `--port`, prepend-bin PATH). `yamlcraft.yaml` untouched
  byte-for-byte (Q1 honored).
- `backend/tests/fixtures/launchcraft_server/` — stdlib-only server
  (health / models / chat / `__state`) + pyproject for readers; the
  journey hand-builds a PEP 427 wheel (zip + RECORD, no build backend)
  so REAL pip installs it with `--no-index --find-links`. Zero network,
  zero stubbed installer internals.
- `backend/tests/test_launchcraft_journey.py` — REAL generic router →
  REAL recipes/executor → REAL `python -m venv` → REAL pip → status
  installed → active row → interpreter compiles the profile → REAL
  `ProcessRegistry.acquire` spawns the console script on an ephemeral
  loopback port → REAL `/health` readiness (the production poll loop,
  not a stub) → **argv proven from inside the running process** via
  `GET /__state` (`["--model","fixture/model","--port","<actual
  port>"]`) → `registry.stop` drains, SIGTERMs the group,
  `os.kill(pid,0)` raises `ProcessLookupError` — zero orphans. First
  process Studio ever launched from YAML alone (gate G3).
- Membership pins deliberately updated for the 11th engine (loader ids,
  tier counts, digest set, lifecycle list, v2 fixture branch) — all
  set-membership assertions, no byte pin weakened. G5 caught this
  slice's own YAML quoting the banned literal — the guard working as
  designed; comment rephrased. Parity inventory regenerated
  byte-identical (launchcraft adds zero tracked literals).
- **R4 — DONE 2026-11-03** (the launch path is now the PROXY path): the
  same journey drives the real model-power HTTP surface end to end.
  `POST /api/models/{id}/start` → real `ProxyClient` →
  `GET /upstream/{proxy_name}/v1/models` → Studio dataplane lazily
  `registry.acquire` → real spawn; the passthrough response is the
  engine's OWN `/v1/models` bytes, and the route writes the durable
  `model_start` journal row (terminal `succeeded`; retention prunes
  finished successes BY DESIGN, so the test records the submitted rows
  through the store's `upsert_operation` and asserts no open admission
  fence survives). One streamed `POST /v1/chat/completions` through the
  SAME dataplane app returns the engine's real SSE (`delta` content
  chunks, `finish_reason: stop`, `[DONE]`), carries the proxy's
  `x-request-id`, and the engine's own hit counter proves the completion
  reached the process — no proxy cache. `POST /api/models/{id}/stop` →
  dataplane unload → SIGTERM drain: `os.kill(pid, 0)` raises
  `ProcessLookupError`, `model_stop` receipt `succeeded`. Compilation
  goes through the PRODUCTION `compile_model_runtime` (not the profile
  helper), and `VALID_ENGINE_IDS` is published by the real startup gate
  (`verify_engine_definitions`), with the derived sets restored after.
  Disclosed harness boundaries: the manager is built with
  `object.__new__` (no uvicorn/listening socket, no manifest staging);
  the router is mounted without the app's other middleware; the store
  thread is drained explicitly, exactly as the app's request gate does.
  No production code changed in this slice (tests + fixture only).

**G5 — DONE 2026-11-03** (discovery is YAML-declared; `discovery.parser`
is WIRED, not decorative):
- `launchcraft.yaml` declares
  `discovery: {parser: llama_help, probe_argv_prefix: [--scan-help]}`.
  The probe is deliberately NOT `--help`: the fixture prints parseable
  llama-style help only for the declared argv, so a successful catalog
  proves the declared prefix actually ran.
- `scanner.scan_engine_version` now asks
  `profile_launch.profile_scan_contract(engine)`. For a definition-backed
  engine (shipped definition, non-generated) it delegates to
  `scan_definition_engine_version`: the executable comes from the runtime
  profile (the SAME resolver the launch uses), the probe argv and parser
  come from YAML, and parsing reuses the existing parsers through a
  parse-only map whose keys are exactly the five existing scanner kinds —
  no vocabulary growth (Q3 honored).
- Route-level proof in the journey: `POST
  /api/llama-versions/scan-engine-params {engine: launchcraft}` →
  `ok: true`, `param_count >= 3`; `GET
  /api/models/param-registry?engine=launchcraft` serves those sections;
  `active_engine_row_is_runnable` flips by `venv_path`
  (`active_path_fields`) only — no engine-id code path.
- Generated engines never enter the definition path: a parametrized test
  over every GENERATED id (with a definition answering ALL ids) proves
  the legacy adapters still own them.
- Typed refusals instead of guesses: a definition without `discovery`
  yields "must declare discovery.parser and
  discovery.probe_argv_prefix"; a parser kind with no parse-only adapter
  yields "no parse-only adapter".
- New `backend/tests/test_engine_definition_discovery.py` (13 tests,
  synthetic persona, zero engine-id literals) + the journey extension.
  No production behaviour changed for the nine (their adapters are
  untouched; only the definition-backed branch was added).

**R5 — DEFERRED deliberately 2026-11-03** (documented, not a gate).
Moving `_CPU_REJECTED_ENGINES` / `_GPU_ONLY_INVARIANTS` /
`_RUNTIME_CONTEXT_KEYS` into spec/profile-backed reads requires a
schema-v2 policy block plus an exporter change and a regenerated
generated tier: an unexported spec field cannot survive
`definition.to_spec()`, and the equivalence tests compare those
round-trips. Doing that unsupervised at the end of a pass would put the
byte-frozen nine's schema equivalence at risk for a non-gate
optimization. Ratchet is GREEN (G6 met, no growth anywhere, and
`launch_spec.py`'s budget unchanged). A follow-on signed slice can land
it with per-key goldens and a demonstrated ratchet shrink.

**R6 — review follow-up: compiler edge cases + a second persona (2026-11-03,
DONE)**. Three reviewed defects, each fixed with a pin, plus the
materially-different configuration the review asked for.
1. *Required env validated the wrong environment.* `env.required` was
   checked against `os.environ + user set` and then the FINAL env was
   built separately, so a variable could pass validation and be absent
   at launch (a config `runtime_env_unset` did exactly that). Validation
   now runs against the environment `registry._start` actually constructs
   — `os.environ + spec.env.set − spec.env.unset` — and engine defaults
   count as satisfying. Pinned by
   `test_required_env_is_validated_against_the_final_launch_environment`.
2. *Owned flags were a fixed port/host list.* Custom args were checked
   against Studio's port flags only, so a custom `--model` could override
   the YAML-selected model. `profile_owned_flags(definition)` now derives
   the set from the profile: model-binding flag (flag mode), port-binding
   flag, and every flag-shaped `argv_prefix` token. Non-owned custom args
   still append (`--log-level` pin). Legacy compilers keep their fixed
   set untouched (byte-frozen).
3. *YAML engines declaring gguf artifacts bypassed the compiler.* The
   GGUF dispatch ran first and sent any `GGUF_ENGINE_IDS` member into the
   legacy implementation. The GGUF branch now asks the profile compiler
   first; it returns `None` for the nine generated GGUF ids (GENERATED
   gate), so their legacy bytes are unchanged, while a definition-backed
   engine is profile-launched. `_use_model_name` follows the definition
   for those engines too. Pinned by
   `test_gguf_declaring_definition_engine_uses_the_profile_compiler` and
   `test_generated_gguf_ids_keep_the_legacy_gguf_compiler`.

**Second persona — `launchforge`** (12th engine, authoritative) proves the
configuration the review asked for: `artifact_formats: [gguf]` (so it
JOINS the derived GGUF set — the exact trap), `venv_python` executable
with a `-m launchcraft_fixture.server` prefix, POSITIONAL model binding,
`cwd: venv_root`, and a required env variable satisfied by an engine
default. The journey (`test_launchforge_gguf_module_profile_launches_end_to_end`)
installs the same hermetic wheel, activates, compiles through production
`compile_model_runtime` (would previously have died in the legacy GGUF
compiler), starts via the real route, proves the OS argv from inside the
process via `/proc/self/cmdline` (`["-m", "launchcraft_fixture.server",
"fixture/model", "--port", "<real port>"]`), checks `cwd == venv_root`,
asserts the required env variable arrived, streams a chat completion
through the proxy, and drains the process group. The engine-count pins
grew 11 → 12 deliberately (loader ids, digests, tier counts, lifecycle
list, v2 fixture branch, schema count, LIFECYCLE bound).

**R7 — review follow-up 2: revision digests + snapshot-backed retries
(2026-11-03, DONE)**. The plan's "Immutable definitions" provisions
(plan line 55), implemented against existing machinery.
1. *Profile digests in launch revisions.* `CompiledModel.profile_digest`
   carries the canonical `definition_digest` of the authoritative
   definition (empty for the nine generated engines).
   `launch_revision(spec, profile_digest=…)` adds the digest to the hashed
   body ONLY when non-empty, so the nine's revisions and manifest bytes
   stay byte-identical. Manifest documents record `profile_digest` for
   definition-backed launches (`manager._launch` passes it through), so a
   YAML edit is visible to published pointers/staleness checks even when
   the compiled argv/env happen to be unchanged. Pinned by
   `test_launch_revision_includes_the_definition_digest` (a label-only
   definition change alters the revision while the launch document is
   identical), `test_manifest_document_records_the_profile_digest`, and
   `test_generated_engines_keep_their_historical_revision`.
2. *Snapshot-backed retries.* The install-snapshot ledger
   (`install_snapshots.yaml`: definition digest + recipe version +
   resolved inputs, recorded by the generic router on every scheduled
   install) is now READ by the retry path instead of adding a second
   mechanism: `GenericVenvInstaller.retry_existing_install` refuses with a
   typed `ValueError` (→ HTTP 400) when the engine's definition digest no
   longer matches the recorded snapshot, and prefers the snapshot's frozen
   resolved inputs (version/branch) over re-resolving against current
   settings. This is the plan's "changing YAML must not silently
   reinterpret their install layout". Pinned by
   `test_retry_is_backed_by_the_install_snapshot_ledger` (frozen version
   wins; changed digest refuses) and the journey now asserts the ledger
   row for the real install. A definition change therefore requires an
   explicit reinstall, never a silent retry.
3. Review follow-up on test isolation (found by running the journeys in an
   adversarial order): the three journey tests now snapshot/restore the
   process-global `LIFECYCLE` registry instead of popping only their own
   engine — registration skips existing ids, so a leftover entry made a
   later registration silently incomplete and broke install-settings
   defaults. Order-independent now (verified by running the journeys
   first, then the yamlcraft journey).

**R8 — review follow-up 3: installation-specific snapshots + complete digest
propagation (2026-11-03, DONE)**. Three defects in the R7 work, each fixed
with a pin.
1. *Retries selected the wrong snapshot.* Rows now carry `version` (stable
   installation association) and `phase` (`scheduled` | `published`). The
   installer records a `published` row at publish time — final version
   name, frozen `package_version`, venv/install dir — and
   `retry_existing_install` selects rows for THAT version (published
   first, then scheduled rows whose recorded requested version matches).
   Retrying A after B is installed can no longer borrow B's digest or
   resolution. Pinned by
   `test_retry_selects_the_snapshot_for_its_own_version` (A then B, plus a
   changed digest on A leaving B's retry working) and
   `test_retry_matches_a_scheduled_snapshot_by_requested_version`.
2. *Recorded inputs were not resolved.* The router now records the
   effective settings the dispatch actually used (version resolved from
   saved build settings, repo/branch after defaults) instead of the raw
   request, and the published row adds the detected package version — so a
   retry is reproducible. Proof: the journey now installs with an EMPTY
   payload and asserts the ledger captured `0.1.0` resolved from saved
   settings, plus the published `package_version`.
3. *One manifest writer omitted the digest.*
   `backend/services/model_runtime_apply.py` now passes
   `profile_digest`. A sweep test asserts EVERY non-test
   `manifest_document(` call site in `backend/` passes it, so a future
   writer cannot silently drop the digest.

**R9 — Phase-3 LMDeploy parity gate (2026-11-03, DONE)**. The plan's
Phase 3 says "Migrate LMDeploy first … preserving
release/source/retry/sync/activation behavior at each gate". Recon
finding: `LMDeployInstaller` ALREADY materializes and executes the
engine definition's plans for all four flows (release/source/sync/remove)
via `execute_plan` — the remaining migration work is host-class
extraction, and the gate it needs was missing (LMDeploy had two tests).
`backend/tests/test_lmdeploy_definition_parity.py` now freezes the
contract: for every flow, (a) the executed plan is
`materialize_plan(definition, flow)` step-for-step (no hand-built
plans), (b) the ONLY engine-specific code is the pinned handler map
(`release`→publish.install; `source`→workspace_checkout/seal_checkout/
publish.install; `sync`→vcs.sync/record_update;
`remove`→publish.forget), and (c) the executor context carries the same
`input.*` values. Eight pins, zero engine-id literals (the id is read
from the installer), added to the engine slice lane (now 318 tests,
~33s). The class flip to the generic installer is the NEXT gated step: it
requires a `lifecycle` block on the generated definition (exporter +
schema vocabulary) and is blocked until it is in place.

**R10 — Phase-3 LMDeploy EXTRACTION (2026-11-03, DONE)**. The flip above
is complete:
- `EngineSpec` gained an installer-only `lifecycle` mapping (excluded from
  `to_api_dict`, so the descriptor payload and every YAML/spec equivalence
  pin are unchanged), round-tripped through `definition.to_spec()` and
  emitted by the exporter for LMDeploy (`definitions/lmdeploy.yaml` now
  carries binary/root/log/state names, distribution names, repo defaults,
  `version_setting_key`, `update_source`, `status_fallback`). The
  exporter's own equivalence check now also compares lifecycle.
- `LMDeployInstaller` was reduced from 666 lines of duplicated venv/pip/
  plan plumbing to 72 lines: a `GenericVenvInstaller` subclass keeping
  only the genuinely engine-specific hook (`LMDEPLOY_BIN` override) plus
  the manager singleton/alias. The four flows, progress, state and
  bookkeeping are inherited.
- `test_lmdeploy_definition_parity.py` was updated for the landed state:
  the four flows must be **inherited** (re-implementing any flow in the
  module fails the structural pin); the plan/handler/context pins still
  hold through the inherited methods.
- Deliberate behavior changes, disclosed: retries now use the shared
  snapshot-backed retry (definition-digest guard + frozen resolved
  version) instead of blind re-install; because the definition digest now
  includes lifecycle data, retrying an installation recorded before this
  change refuses with "definition changed … reinstall". Install layout,
  argv, response messages and version-row shape are unchanged.
- Verification (lanes only, no full suite): lmdeploy parity + legacy
  installer tests 11 passed; engine slice 318 passed (~33s); defcheck 12
  engines; ratchet no growth; curated ruff 62; openapi 0.

**R11 — upstream vLLM: lifecycle data + parity gate (2026-11-03, DONE for
this slice; extraction NOT done)**. Plan Phase 3's next engine.
- `definitions/vllm.yaml` now carries the lifecycle facts exported from
  the registry (binary/distribution/root/log names, repo defaults,
  `version_setting_key`, `update_source`, `status_fallback`) — inert data
  until the host class is swapped.
- `backend/tests/test_vllm_definition_parity.py` (5 pins) freezes the
  CURRENT contract: every flow executes the definition's plan step-for-
  step; the per-flow handler map is pinned (`release`→publish.finalize;
  `source`→env.compose/workspace_checkout/publish.finalize;
  `sync`→env.compose/vcs.sync/publish.finalize;
  `remove`→publish.forget) and so are the executor context keys; a
  structural pin records that the flows are STILL local and will be
  inverted when extraction lands.
- Extraction is deliberately NOT done: vLLM's flows are not the generic
  ones. Its definition uses a different primitive vocabulary and its
  handlers carry engine semantics the shared host cannot express yet:
  `publish.finalize` (import validation before publish, reuse-existing
  version naming, progress stage 98), `env.compose` (CUDA build
  environment + runtime metadata), `remove` context inputs
  (`install_dir`/`installs_root`) with a NON-swallowing version delete,
  a Python-version guard, and a custom `status()`. Migrating these needs
  a typed finalize contract + declared build-env/python policy in the
  definition vocabulary — the next slice, not an improvisation.

**R12 — Phase-3 typed installer policy in the shared host (2026-11-03,
DONE)**. The capabilities the vLLM extraction needs now live in
`GenericVenvInstaller`, declared by the definition:
- `lifecycle.python_min`/`python_max` → interpreter guard in
  `_ensure_venv()` with the legacy message shape
  (`"<label> requires Python 3.10–3.13; Studio is running Python X.Y"`).
- `lifecycle.build_env: cuda` → `env.compose` primitive supplies the CUDA
  build environment and the `cuda_version`/`cuda_path`/`python_version`
  install-record metadata.
- `publish.finalize` primitive → validated publication: the installed
  package must import, source builds compose a commit-tagged version
  name, reuse-existing is honored, the version row is written + activated,
  and the install-snapshot ledger gets the published row.
- `lifecycle.forget_errors: raise` → strict version-delete policy (the
  historical swallow-everything default is unchanged).
- `remove` context now supplies BOTH vocabularies (`install_dir`/
  `installs_root` and `venv_path`/`venv_python_exists`), so wheel-family
  remove recipes run on the shared host too.
- Handler maps are plan-filtered (`_plan_handlers`): only primitives the
  recipe actually uses are overridden.
Schema: `BUILD_ENVS`/`FORGET_ERROR_POLICIES` vocabularies + validation;
`definitions/vllm.yaml` exports `python_min: '3.10'`, `python_max:
'3.13'`, `build_env: cuda`, `forget_errors: raise`.
Verification (lanes): 8 new policy pins (`test_generic_installer_policy.py`,
synthetic persona); generic host still green with the LMDeploy parity pins,
launchcraft journey and generic suite (21 passed); engine slice **331
passed / 35s**; defcheck 12 engines; ratchet no growth; ruff 64; openapi 0.
**vLLM flip NOT done yet** — one semantic question remains: the generic
source flow overrides `vcs.seal_checkout` (its recipe uses it) while
vLLM's own map does not, so sealing must be proven equivalent (or
declared) before the class swap; that plus the four-flow shadow comparison
is the next step.

**R13 — upstream vLLM EXTRACTION (2026-11-03, DONE)**. The swap landed
after resolving the one open semantic question:
- The shared host's `vcs.seal_checkout` handler now matches the executor's
  DEFAULT primitive exactly — it honours the step's `destination` (and
  `workspace_from`), rebinds `checkout` to the sealed path, and keeps the
  `checkout_sealed` label for recipes that declare it; the checkout step
  publishes the workspace into the plan context. That is the behaviour
  vLLM's recipe relies on (`capture_commit` requires `checkout` AFTER
  sealing).
- `VllmInstaller` dropped from 677 lines to ~105: a `GenericVenvInstaller`
  subclass whose definition supplies the lifecycle (names/repo,
  `python_min`/`python_max`, `build_env: cuda`, `forget_errors: raise`) and
  whose recipes' `publish.finalize`/`env.compose` are implemented by the
  shared host. Only the status payload (venv python as `binary_path` +
  CUDA record fields) remains engine-specific.
- `test_vllm_definition_parity.py` rewritten for the migrated state: the
  four flows must be INHERITED (re-implementation fails the structural
  pin), plan/handler/context pins hold through the inherited methods, and
  a dedicated **seal-equivalence pin** compares the shared handler's
  output with `run_shared_primitive` (same destination, same rebound
  `checkout`).
- Verification (lanes only): vLLM + LMDeploy parity pins 15 passed;
  engine slice **332 passed / ~35s**; defcheck 12 engines; ratchet no
  growth; ruff 64; openapi 0.

**R14 — SGLang: lifecycle data + gate; flip NOT done (2026-11-03)**.
Plan Phase 3's last upstream venv engine; Phase 5 keeps the V100 fork
specialised.
- `definitions/sglang.yaml` now carries the upstream variant's lifecycle
  facts (binary `python`, distribution `sglang`, root/log names, repo,
  pypi package, `version_setting_key`) — inert until the class moves.
- `test_sglang_definition_parity.py` (6 pins) freezes the current
  contract: the four flows execute the definition plan step-for-step with
  pinned handler maps and context; the SOURCE/SYNC maps include
  `vcs.apply_patch` and `vcs.prepare_build`, which the shared host does
  NOT implement (Phase-5 specialised recipes); status carries the `v100`
  flag; the flows are still local (inverse pin for the future flip).
- Why the flip is not done (evidence, not preference): upstream SGLang's
  source recipe overrides the two specialised primitives above; its
  start messages differ from the shared host's ("<label> source
  installation started" vs "<label> install from source started"); and
  both variants' status payload includes `v100`. Moving upstream SGLang
  needs either definition-declared message/status extensions or a thin
  upstream subclass with byte-pinned payloads — while `sglang_v100`
  keeps the specialised recipes regardless. That decision is the next
  gated step; improvising it would change user-visible strings.
- Verification (lanes only): SGLang gate 6 passed; engine slice **338
  passed / ~37s**; defcheck 12 engines (exporter re-validated); ratchet
  no growth; ruff 65; openapi 0.

**R15 — regression-harness repair after the LMDeploy/vLLM cutovers
(2026-11-03, DONE)**. The existing engine regression selection was red
(26 failed / 10 passed). Triage separated harness rot from REAL behavior
regressions and fixed both in the shared host, not by weakening the
assertions:
- *Real regressions found by the old tests* (and fixed in
  `GenericVenvInstaller`): `env.compose` produced `env`/`runtime` while
  recipes declare `build_env`/`runtime_meta`, so vLLM source installs
  lost the CUDA build env; `sync_source_version` required an explicit
  `venv_path` (legacy rows carry only `install_dir`) and its failure path
  dropped the row + runtime metadata; `remove` targeted the process base
  dir instead of the ACTIVE row's `install_dir`, and released the launch
  hold only for a venv path (legacy source rows hold it on the install
  dir); source checkouts no longer emitted the legacy progress stages
  (3 "Cloning source" / 8 "Source checkout ready"); the sync
  `publish.finalize` now merges the version row so provenance
  (`source_repo`, commit) reaches the row.
- *Harness rot*: the vLLM shadow tests patched symbols that moved into
  `backend.engines.python.generic` (`unique_version_name`,
  `mark_runtime_changed`, `activate_engine_version`) and mocked the
  removed `_cuda_build_env`/`_clone`; they now patch the shared module,
  drive the reuse-mode checkout through an asyncio subprocess fake, and
  mock the compose/runtime hooks. Assertions (pip argv/order, cwd/env,
  version naming/reuse, rows, error strings, remove policies) were kept.
- Disclosed normalization: the shared host always refreshes the install
  state after a failure; legacy vLLM deliberately did not (idempotent
  bookkeeping only, no user-visible difference). If byte-identical state
  files matter, declare it as another lifecycle policy.
- Verification: the engine regression selection is green (**113 passed**:
  lmdeploy/vllm/sglang integration + shadow suites + version lifecycle +
  removal characterization; the previously-missed
  	est_lmdeploy_legacy_plan_shadow.py needed the same patch-target repair
  and now passes), the FULL serial suite is green (**2263 passed, 9
  skipped**), engine slice **338 passed / ~36s**,
  defcheck 12, ratchet no growth, ruff 65, openapi 0. The parity
  inventory was stale and has been REGENERATED (`docs/engine-parity-inventory.{json,md}`).

**R16 - upstream SGLang EXTRACTION (2026-11-03, DONE)**. The swap the R14
gate was preparing:
- The SGLang log/progress machinery (log streaming with heartbeat,
  compiler-noise filtering, failure-context replay, progress-stage mapping,
  pip runner) moved into `SglangLogMixin`, shared by BOTH variants.
- `SglangUpstreamInstaller(SglangLogMixin, GenericVenvInstaller)` is now
  the registered upstream installer: flows, retries, progress bookkeeping
  and pending versions are inherited; the definition supplies names, repo
  and package. Only byte-contract deltas stay on the subclass:
  `_source_started_message()` ("<label> source installation started", a new
  generic hook) and the full `status()` payload (identical keys plus
  `v100: False`).
- `SglangInstaller` (legacy class) now serves ONLY the V100 fork, whose
  source/sync recipes override the specialised `vcs.apply_patch` /
  `vcs.prepare_build` primitives (Phase 5).
- `test_sglang_definition_parity.py` was rewritten for the migrated state:
  the four upstream flows must be INHERITED and execute the definition
  plan step-for-step with plan-filtered handlers/context; the byte
  contract pin checks the source-start wording; the V100 boundary pin
  asserts the fork still owns its flows and specialised primitives.
- Verification (lanes): SGLang suites 56 passed + parity 8 passed;
  engine regression selection 93 passed; engine slice **340 passed /
  ~36s**; defcheck 12 engines; ratchet no growth; ruff 65; openapi 0.

**R17 - Phase 6: LMDeploy launch compiles from its definition (2026-11-03,
DONE)**. The first runtime-profile cutover:
- `EngineSpec.runtime_profile` (API-invisible, like `lifecycle`) carries
  LMDeploy's profile verbatim from the reference YAML; the exporter writes
  it as `runtime_profiles.default` and verifies definition<->registry
  equality.
- `_compile_launch` consults the profile interpreter FIRST. The
  GENERATED-engine gate became an explicit OPT-IN: a generated id qualifies
  only when its exported definition declares a runtime profile, so the other
  eight keep their byte-frozen compilers until their own gated item.
- `profile_owned_flags` is unioned with Studio's universal port/host
  reservation, so custom args can re-declare neither profile-owned
  bindings/prefix flags nor the runtime's port surface.
- Goldens (`test_lmdeploy_profile_launch_parity.py`, 5): profile spec ==
  legacy `_compile_lmdeploy` field-for-field for defaults and structured
  parameter tokens; dispatch prefers the profile; port flag rejected on both
  paths; CPU contract identical.
- Finding reverted: an `executable.env_override` (LMDEPLOY_BIN) vocabulary
  was added and then REMOVED — the legacy LMDeploy resolver never honoured
  that variable, so declaring it would have changed behavior rather than
  preserved it. The profile declares only what the legacy path did.
- Verification (lanes): parity 5 + profile_launch 25; definition/launch
  suites 123; engine slice **340 passed / ~37s**; defcheck 12; ratchet no
  growth; ruff 66; openapi 0.

**R18 - Phase 6: upstream vLLM launch compiles from its definition
(2026-11-03, DONE)**. Second runtime-profile cutover, now with the dynamic
env vocabulary:
- New profile key `env.cuda`: the Studio-managed CUDA env is resolved for
  the ACTIVE install and composed exactly like the legacy compilers
  (LD_LIBRARY_PATH/PATH split into compositional keys, the rest as engine
  defaults) - the mirror of the installer's `build_env: cuda`.
- `vllm.yaml` now declares `runtime_profiles.default` (venv python,
  `-m vllm.entrypoints.openai.api_server`, --model/--port, `cwd: venv_root`,
  `cuda: true`), exported from the spec and verified against it.
- `test_vllm_profile_launch_parity.py` (6): profile spec == legacy
  `_compile_module(... engine="vllm")` field-for-field for defaults and
  structured params, dispatch preference, port rejection, missing
  environment and missing huggingface_id error paths.
- Harness repair: `test_vllm_integration.py` patched the legacy resolver
  symbols (`_resolve_sglang_bin` / `_resolve_sglang_cuda_env`) that the
  profile path no longer consults; it now drives the active install row +
  `get_cuda_installer`, keeping every assertion (cmd/env/ok).
- Verification: vLLM parity 6; definition/launch suites 128; engine slice
  **351 passed / ~40s**; FULL parallel suite **2276 passed, 9 skipped**;
  defcheck 12; ratchet no growth; ruff 67; openapi 0.
**R19 - Phase-6 gaps closed: definition digest + executable parity
(2026-11-03, DONE)**. Two findings from review, fixed before the next
runtime migrates:
- `definition_digest_for()` now returns the shipped definition digest for
  every Phase-6 OPT-IN engine (a generated id whose definition declares a
  runtime profile) and still returns None for the byte-frozen eight. Launch
  revisions/manifests therefore carry LMDeploy's and vLLM's profile digest;
  the historical-revision pin was updated to assert exactly that split.
- `_resolve_executable` regained the legacy resolver's edge cases: relative
  venv paths normalize against `/app` before probing, the venv directory
  must exist, and the target must be a FILE that is executable
  (`os.access(..., X_OK)`), not merely present.
- New parity tests: relative-venv normalization (equal specs + executable),
  non-executable binary failures on both paths, and a digest test proving
  opted-in engines return their digest while the eight return None.
- Verification: LMDeploy parity 7, profile_launch 26, engine slice
  **354 passed / ~39s**, FULL parallel suite **2279 passed, 9 skipped**;
  defcheck 12; ratchet no growth; ruff 67; openapi 0.
**R20 - Phase 6: 1Cat-vLLM launch compiles from its definition
(2026-11-03, DONE)**. Third runtime-profile cutover:
- `1cat_vllm.yaml` now declares the plain module profile (venv python,
  `-m vllm.entrypoints.openai.api_server`, --model/--port, `cwd: venv_root`,
  NO managed-CUDA env - the legacy path injects none), exported from the
  spec and verified against it.
- `test_onecat_profile_launch_parity.py` (5): profile spec == legacy
  `_compile_module(... engine="1cat_vllm")` field-for-field for defaults and
  structured params, dispatch preference, port rejection, missing hf-id.
- The vLLM parity helper was narrowed to the cuda-env engine (both engines
  share the module string) so each golden file selects its engine by
  DECLARED property, not by id.
- SGLang's launch is DELIBERATELY still legacy: it adds a runtime switch
  (`sglang_runtime` llm/diffusion) whose diffusion arm reads the scanned
  parameter catalog and can raise a typed "diffusion unavailable" error,
  plus `sglang_server_argv`. Expressing that needs either a profile
  `runtime_switch` block or leaving the switch on the engine. Decision
  recorded for the next item; not smuggled in as engine-id logic.
- Verification: 1Cat parity 5; engine slice **359 passed / ~41s**; FULL
  parallel suite **2284 passed, 9 skipped**; defcheck 12; ratchet no growth;
  ruff 68; openapi 0.
**R21 - Phase 6: SGLang launch + declared runtime switch (2026-11-03,
DONE)**. Fourth runtime-profile cutover; the design decision recorded in
R20 is implemented as DATA:
- New profile block `runtime_switch` (config_key / default / variants;
  each variant may declare `argv_prefix` and `scanned_catalog`). The
  interpreter selects the variant from config, swaps the prefix and - when
  a variant names a scanned catalog - re-indexes parameters from
  `runtime_profiles.<catalog>`, keeps the default runtime's keys out of
  argv, and raises the legacy typed error when that catalog has a
  `scan_error` or no sections ("<label> <catalog> is unavailable: <reason>.
  Rescan engine parameters after installing the fork's diffusion
  dependencies.").
- `sglang.yaml` declares the llm arm (`-m sglang.launch_server`,
  `--model-path`, venv-root cwd, managed-CUDA env) and the diffusion arm
  (`-c "from sglang.cli.main import main; main()" serve --model-type
  diffusion`, catalog `diffusion`).
- Goldens (4): llm spec == legacy field-for-field; diffusion spec == legacy
  (prefix swap + filtered params: llm-only keys dropped, diffusion keys
  kept); the diffusion-unavailable error message is BYTE-EQUAL to legacy;
  dispatch prefers the profile.
- Owned-flag note: variant prefix flags (e.g. `--model-type`) join the
  profile-owned set, so a custom arg cannot fork the runtime identity -
  the same stricter policy the reviewer requested for profile bindings.
- Verification: SGLang launch parity 4; engine slice **363 passed / ~43s**;
  FULL parallel suite **2288 passed, 9 skipped**; defcheck 12; ratchet no
  growth; ruff 69; openapi 0.
**R22 - Phase 6: SGLang V100 launch compiles from its definition
(2026-11-03, DONE)**. Fifth and LAST Python-venv runtime adapter:
- Two env vocabulary extensions, both declared by the profile:
  `env.cuda` may be a VERSION STRING (the fork's 12.8 fallback when the
  active row has no cuda_version), and `env.required` entries may be
  `{name, error}` so the managed-CUDA preflight keeps its EXACT legacy
  message ("SGLang V100 requires Studio-managed CUDA 12.8", field
  CUDA_HOME).
- `sglang_v100.yaml` declares the fork profile: venv python,
  `-m sglang.launch_server`, --model-path/--port, venv-root cwd, CUDA env
  with 12.8 fallback, the three fixed defaults, required CUDA_HOME, and the
  same llm/diffusion runtime switch.
- Goldens (4): default spec == legacy field-for-field (incl. fixed env
  defaults), 12.8 fallback asserted on the resolver call, missing-CUDA_HOME
  error BYTE-EQUAL with the same field, dispatch preference.
- Harness repairs while landing it: the preview tests
  (`test_sglang_integration`) and the launch-manifest helper patched the
  legacy resolver symbols / real CUDA installer; both now drive the active
  install row + a fake `get_cuda_installer`, keeping every assertion.
- Verification: V100 launch parity 4; engine slice **362 passed / ~39s**;
  FULL parallel suite **1957 passed, 8 skipped** (the other agent's test
  pruning is reflected in this baseline); defcheck 12; ratchet no growth;
  ruff 69; openapi 0.
**R23 - Phase 6 boundary: native GGUF launch STAYS legacy (2026-11-03,
DECIDED)**. Investigated llama.cpp / ik_llama / unsloth before writing any
vocabulary. Their launch composition is FAMILY behaviour, not per-engine
YAML: it resolves a model source (local file vs `--hf-repo`), mmproj and
draft companions (with `--spec-type`), the model alias, and the shared
library path from the binary directory, plus relative-binary `/app`
normalisation and a resolved invocation path.
- Decision: these engines ship NO runtime profile and keep `_compile_gguf`
  byte-frozen. The alternative was a speculative `model_bindings` /
  `companions` / `alias_binding` vocabulary with exactly one consumer -
  the kind of premature abstraction this campaign exists to remove.
- The decision is now a TEST, not a comment:
  `test_native_gguf_engines_stay_on_the_legacy_compiler` asserts the native
  GGUF ids are profile-less and fail the interpreter gate, so adding a
  profile to them later forces a deliberate golden comparison.
- Phase-6 status: ALL Python-venv launch adapters (LMDeploy, vLLM,
  1Cat-vLLM, SGLang, SGLang V100) compile from their definitions; native
  GGUF and audio.cpp remain legacy by this decision.
- Verification: engine slice **363 passed / ~40s**; FULL parallel suite
  **1958 passed, 8 skipped**; defcheck 12; ratchet no growth; ruff 68.
**R24 - Review findings P1/P2: one SGLang manager + cancellation cleanup
(2026-11-03, DONE)**. Both were real; both are fixed and pinned.
- [P1] `get_sglang_manager()` now routes upstream SGLang to the SAME
  singleton the adapter registry returns (`get_sglang_upstream_manager`),
  instead of constructing a legacy `SglangInstaller` from a second instance
  dict. Routes (status/cancel/install), adapters and retry/sync therefore
  observe one manager state. Pinned by
  `test_upstream_sglang_has_one_manager_everywhere` (manager identity +
  adapter identity). The stale registry comment was corrected too.
- [P2] `execute_plan` caught `Exception` only, so real asyncio task
  cancellation (a BaseException) skipped the recipe's cleanup pass. It now
  catches `asyncio.CancelledError`, runs the teardown pass with each cleanup
  invocation SHIELDED (a second cancellation cannot strand the remaining
  steps), records interrupted cleanups as failed, and re-raises the original
  CancelledError unchanged. Predicate-based cancellation behavior is
  untouched. Pinned by
  `test_task_cancellation_still_runs_the_cleanup_pass` (cancels a live task
  mid-body and asserts the cleanup step ran).
- Verification: engine slice **365 passed / ~39s**; FULL parallel suite
  **1960 passed, 8 skipped**; defcheck 12; ratchet no growth; ruff 68;
  openapi 0.
**R25 - Phase 5, step 1: V100 patch scripts extracted to one home
(2026-11-03, DONE)**. Scoped the V100 extraction as a SEQUENCE (extraction
alone cannot delete the class - it still owns env composition, publication,
recovery and lifecycle flows); step 1 is the patch scripts:
- `backend/engines/sglang/v100_support.py` now carries `V100PatchMixin`
  with the three patch implementations (Marlin builder, Python metadata,
  CUDA-IPC tuple access), copied VERBATIM (embedded shell scripts must stay
  byte-identical - a naive dedent extraction silently changed their
  indentation and was caught by the Marlin integration test).
- `SglangInstaller` inherits the mixin; the methods no longer exist in the
  class, so the definition-driven host can supply the SAME implementations
  as `vcs.apply_patch` handlers without a second copy.
- Pinned by `test_v100_patch_scripts_live_in_the_shared_mixin` (methods in
  the mixin, absent from the class) so the extraction cannot silently
  regress.
- Next steps in the sequence (planned, not started): extract
  `_v100_build_environment` / `_v100_prepare_build_env` the same way, add
  the `SglangV100Installer` subclass supplying `env.compose` /
  `vcs.apply_patch` / `vcs.prepare_build` + `status()`, flip the adapter
  and `get_sglang_manager`, then delete the legacy class.
- Verification: sglang integration + both shadow suites **50 passed**;
  engine slice **366 passed / ~36s**; defcheck 12; ratchet no growth
  (temp-script litter removed); ruff 68. Full parallel suite **1960 passed,
  8 skipped** with one different pre-existing flake per run
  (launchcraft journey, then proxy accounting - both pass in isolation).
**R26 - Phase 5, step 2: V100 build environment extracted (2026-11-03,
DONE)**. Second slice of the V100 sequence:
- `v100_support.V100BuildMixin` now owns the fork's build environment:
  compiler selection (`_compiler_major`, `_select_v100_compiler`),
  memory-aware job sizing (`_read_meminfo_available_kib`,
  `_read_cgroup_available_kib`, `_v100_safe_jobs`), the Studio/Python/CUDA
  bindings (`_v100_build_environment`) and the ccache/HOME/cargo/dependency
  preservation layer (`_v100_prepare_build_env`). `V100_CUDA_VERSION` moved
  with them and is re-exported from the installer.
- `SglangInstaller` inherits the mixin; the methods are absent from the
  class, so the future `SglangV100Installer` supplies the SAME
  implementations as `env.compose` / `vcs.prepare_build` handlers.
- Care points hit and handled: the job-sizing helper is a STATIC method
  called as `SglangManager._v100_safe_jobs()`, so it keeps a late import of
  the manager class (tests patch the read helpers on that class); the
  prepare-build hook is `async` (the boundary regex must be async-aware).
- Pinned by `test_v100_build_environment_lives_in_the_shared_mixin`
  (methods in the mixin, absent from the class, constant re-exported).
- Verification: sglang suites **60 passed**; engine slice **366 passed /
  ~35s**; FULL parallel suite **1962 passed, 8 skipped**; defcheck 12;
  ratchet no growth (temp litter removed); ruff 68.
**R27 - Phase 5, step 3: SglangV100Installer registered (2026-11-03,
DONE)**. The fork now runs on the definition-driven host:
- `GenericVenvInstaller` gained one hook, `_extra_plan_handlers()`, merged
  into all four plan handler maps (overrides still win), so a specialised
  fork can supply primitives its recipes declare without re-implementing
  flow control.
- `SglangV100Installer(SglangLogMixin, V100PatchMixin, V100BuildMixin,
  GenericVenvInstaller)` supplies exactly two primitives -
  `vcs.apply_patch` -> `_v100_apply_checkout_patches` and
  `vcs.prepare_build` -> `_v100_prepare_build_env` - plus the env hooks
  (`_compose_build_env` / `_runtime_metadata`), the 3.12 interpreter guard,
  the source-start wording and the full status payload (`v100: True`).
- `get_sglang_manager()` routes BOTH variants to their definition-driven
  singletons; `get_sglang_v100_manager` is exported alongside the upstream
  one. The legacy `SglangInstaller` is now used only by its own shadow
  tests (step 5 deletes it).
- A test-authoring bug was caught and fixed immediately: the new generic
  hook initially landed between `@staticmethod` and `_plan_primitives`,
  making `_extra_plan_handlers` a staticmethod (policy tests failed with
  "missing self"); the decorator was restored.
- Pins: inherited flows for all nine flow methods, `_extra_plan_handlers`
  == the two specialised primitives, both recipe primitives covered, status
  `v100` true, and two flow-execution pins asserting the V100 source/sync
  plans execute with exactly the expected handler sets.
- Verification: sglang suites **70 passed**; engine slice **370 passed /
  ~39s**; FULL parallel suite **1965 passed, 8 skipped**; defcheck 12;
  ratchet no growth; ruff 68; openapi 0.
**R28 - Phase 5, step 4 attempted: legacy class deletion DEFERRED
(2026-11-03)**. Deleting `SglangInstaller` now would require porting the
legacy seams the shadow tests drive, so the deletion was attempted,
MEASURED and reverted rather than left half-done:
- Attempt: replaced the class with a `SglangManager` facade dispatching to
  the two definition-driven classes (same construction surface), deleted
  the 946-line legacy class, ran the sglang suites.
- Result: 29 failures concentrated exactly where the legacy seams live -
  the source/sync shadow tests patch `_clone` / `_sync_checkout` (the
  shared host clones via subprocess and syncs via `_sync_git_checkout`),
  the release shadow tests patch `unique_version_name` on the engine module
  (moved to the generic host), and three parity pins still describe the
  legacy class. The tree was restored byte-for-byte; sglang suites green
  again (64 passed) and the engine slice is unchanged.
- Port plan for the deletion tranche (in order): (1) release shadow tests:
  point patch targets at `backend.engines.python.generic` and construct via
  the facade; (2) source/sync shadow tests: replace `_clone`/
  `_sync_checkout` fakes with `asyncio.create_subprocess_exec` and
  `_sync_git_checkout`; (3) integration tests: same constructor facade and
  seam swaps; (4) rewrite the parity pins to assert BOTH variants are
  inherited and only the specialised handlers remain; (5) delete the class
  and the facade, then verify the ratchet SHRINKS.
- No behavior change landed; this is a scoping record, not a failure of the
  extraction itself.
**R29 - Phase 5, step 4a: release shadow ported; real gaps closed
(2026-11-03, DONE)**. Porting the release shadow tests to the new classes
surfaced two genuine issues that the legacy-only tests could not see:
- SGLang's definition-driven release recipe had ONE detection, while legacy
  detected twice (runner detection fed the row's base version, then finalize
  detected again). Fixed in the EXPORTER recipe: a `pip.capture_metadata`
  step now sits between `pip.install_package` and `publish.finalize`,
  restoring the two-detection order and the base-version semantics.
- The fork's "release installs build from source" routing only existed in
  the legacy class. `SglangV100Installer.install_release` now performs that
  delegation explicitly (repo/branch from the definition lifecycle), and
  the V100 definition gained its installer lifecycle (repo/root/log/version
  keys) so the delegation has a data source.
- `test_sglang_release_plan_shadow.py` now constructs `SglangUpstreamInstaller`
  / `SglangV100Installer`, patches `unique_version_name` and the activation
  hook on `backend.engines.python.generic`, and documents the one
  normalization: the shared host refreshes install state after a failure
  (the pin asserts the real contract - no publication - instead of "no
  detection").
- The V100 parity pin was updated to record the ONE intentional flow
  override (release -> source) alongside the inherited rest.
- Verification: release shadow **7 passed**; engine slice **370 passed /
  ~39s**; FULL parallel suite **1965 passed, 8 skipped**; defcheck 12;
  ratchet no growth; ruff 68.
**R30 - Phase 5, step 4b attempted: source/sync shadow port REVERTED
(2026-11-03)**. The mechanical part is easy; the assertions are not:
- Ported `test_sglang_legacy_plan_shadow.py` to the new classes: constructor
  switch, `_sync_checkout` -> `_sync_git_checkout`, `_clone` fakes ->
  `asyncio.create_subprocess_exec`. Result: 13 of 14 tests failed, because
  those assertions describe the LEGACY flow's exact call sequence,
  metadata shape and error strings, not merely the seams they patch.
- Reverted to the green state. The lesson for the deletion tranche: these
  tests must be re-derived against the recipe plan (expected step outcomes,
  rows, error strings) rather than mechanically re-pointed; the release
  shadow port (R29) worked because only patch targets/constructors moved.
- Verification after revert: sglang suites **64 passed**; engine slice
  **370 passed / ~36s**; ratchet no growth.
**R31 - Phase 5, step 4b: source/sync CONTRACT harness (2026-11-03, DONE)**.
Instead of re-pointing the legacy transcript assertions (R30 showed they
encode the old flow), the contract is now pinned by
`test_sglang_source_contract.py` (7 tests) driving the definition-driven
managers directly:
- reuse source: bootstrap argv + editable of the checkout's `python`
  subpackage with cwd at the checkout, then the published row
  (install_type/source_repo/source_branch);
- guard: missing `python/` fails at `assert_checkout` with the legacy
  string BEFORE any editable install;
- recipe error strings: `pip upgrade failed with status 7`;
- workspace mode: acquire + sync_git + seal + captured commit in the row;
- sync: republish keeps the version, row provenance merged; a non-primitive
  sync failure propagates after cleanup (legacy parity) with the legacy
  message;
- V100: the fork builder runs as `bash <rewritten script>` with the Studio
  env and the published row carries `cuda_version` + commit.
- The legacy transcript file remains until the class is deleted (it still
  documents the old flow and keeps the class honest while it exists).
- Verification: harness **7 passed**; engine slice **377 passed / ~37s**;
  FULL parallel suite **1972 passed, 8 skipped**; ratchet no growth (69
  curated files, contract file added; a stray `_tmp` script was removed).
**R32 - Phase 5, step 4c attempted: integration constructor swap
(2026-11-03)**. Switched all 11 `SglangManager(...)` constructions in
`test_sglang_integration.py` to the definition-driven classes via a local
factory. Result: 25/29 passed; the four failures are legacy-only surfaces
(`_finalize_install` and the `_clone`-driven V100 source flows), the same
class of remaining work as R30. Reverted to keep the tree green; the attempt
also proved the other 25 integration tests already run unchanged on the new
classes.
- Note for the deletion tranche: `_finalize_install` is a legacy wrapper not
  used by the shared host, so ITS test should be re-derived against
  `publish.finalize`, and the three V100 source-flow tests against the
  recipe contract (R31 harness already covers the equivalent behavior).
- Verification after revert: sglang suites **71 passed**; engine slice
  **377 passed / ~35s**; ratchet no growth (a stray `_tmp` script was
  removed); ruff 69.
**R33 - P1/P1 FIXED: the V100 cutover was genuinely broken; real-helper
coverage added (2026-11-03)**. Review findings confirmed by probe:
- [P1] `_v100_apply_checkout_patches`, `_write_v100_prefix_installer` and
  [P1] `_preserve_v100_dependency` were left on the legacy class, so the
  registered `SglangV100Installer` would fail at the patch step and during
  Marlin/TurboMind recovery (AttributeError). All three now live in
  `V100PatchMixin` (verbatim), so BOTH classes share one implementation and
  the definition-driven installer has them. `tempfile`/`time` imports were
  added to the support module for the moved recovery helper.
- Coverage gap closed: two contract tests now exercise the REAL helpers on
  temporary checkouts - the builder-script rewrite (Studio bootstrap +
  TurboMind sparse list + Python dependency overrides) and dependency
  recovery (stale Marlin checkout preserved and rebuilt, controlled
  HOME/cargo env), instead of stubbing them.
- Verification: contract harness **9 passed**; SGLang suites **73 passed**;
  engine slice **379 passed / ~40s**; FULL parallel suite **1973 passed, 8
  skipped** with one known rotating flake (`test_launchcraft_journey`, passes
  in isolation); ratchet no growth.
**R34 - Phase 5, step 5: LEGACY SglangInstaller DELETED (2026-11-03,
DONE)**. First legacy engine implementation removed:
- `SglangInstaller` (the ~950-line legacy class) is gone; `SglangManager`
  is now a thin compatibility FACADE that constructs the definition-driven
  `SglangUpstreamInstaller` / `SglangV100Installer` (same construction
  surface for routes/tests). Both singletons, the adapter and the facade
  therefore point at one implementation each.
- Specialised implementations stay in the shared mixins
  (`V100PatchMixin`: builder rewrite, patch scripts, dependency recovery;
  `V100BuildMixin`: toolchain, job sizing, Studio env). `_v100_safe_jobs`
  now consults the mixin instead of the deleted class.
- Tests retired/rewritten by contract: the legacy transcript file and the
  four integration tests that asserted legacy-only surfaces are replaced by
  the R31 contract harness (reuse/workspace/builder/guard/sync/remove) and
  the real-helper tests from R33; the parity pins now assert that the mixins
  own the specialised implementations and the v100 status flag lives on the
  definition-driven classes.
- Verification: SGLang suites **55 passed** (integration 23, parity,
  release shadow 7, contract harness 9); engine slice **379 passed / ~39s**;
  FULL parallel suite **1956 passed, 8 skipped** (net -17 legacy-transcript
  tests retired); defcheck 12; ratchet no growth; ruff 68.
**R35 - Post-deletion cleanup: dead imports pruned, seams rehomed
(2026-11-03, DONE)**. The class deletion left dead imports and stale
monkeypatch targets:
- `ruff check --select F401` found and removed 8 unused imports across
  `sglang/installer.py`, `v100_support.py` and `sglang/__init__.py`; the
  `V100_CUDA_VERSION` re-export is kept explicitly (`# noqa: F401`) because
  tests import it from the installer module.
- Two integration tests patched `sglang.installer.shutil` /
  `.subprocess`; the helpers now live in `v100_support`, so the patch
  targets moved there too (behaviour unchanged).
- Also confirmed LMDeploy and vLLM carry NO legacy duplication: LMDeploy's
  installer is a thin subclass whose only behaviour is the `LMDEPLOY_BIN`
  install-time override; vLLM's keeps only its status payload. Their
  "deletion" is already done by construction - the remaining legacy code is
  the specialised 1Cat/native/audio installers (Phase 5/7).
- Verification: SGLang **55 passed**; engine slice **379 passed / ~35s**;
  FULL parallel suite **1956 passed, 8 skipped**; defcheck 12; ratchet no
  growth; ruff 68.
**R36 - Phase 5: 1Cat-vLLM extraction scoped (2026-11-03)**. Recon of the
LAST legacy Python-venv installer (`backend/engines/vllm/onecat.py`, 1059
lines, class `OneCatVllmInstaller(PythonVenvInstaller)`):
- Already plan-driven: `install_release`/`install_from_source`/`sync`/
  `remove` delegate to `_execute_{release,source,sync,remove}_plan` (the
  same recipe shape as the migrated engines); `retry_existing_install` is
  class-owned.
- Genuinely specialised (candidates for a `OneCatRustMixin`/support module,
  verbatim like V100): Rust discovery/retargeting (`_managed_rust_root`,
  `_rust_exe`, `_bin_dir_has_rust`, `_dir_writable`,
  `_retarget_unwritable_rust_homes`, `_apply_rust_bin_dir`,
  `_rust_bin_candidates`, `_discover_rust_bin_dir`, `_build_env`), the
  wheel forge (`_ensure_rust`, `_build_source_wheels`) and release
  wheel selection (`_fetch_release`, `_select_release_wheels`), plus
  `status()` and `_resolve_binary_path`.
- Extraction order (mirrors V100's proven sequence): (1) pure Rust-env
  helpers -> mixin; (2) wheel-forge/release-selection helpers -> mixin +
  recipe handlers (`vcs.prepare_build` analogue); (3) definition-driven
  subclass + `_extra_plan_handlers`; (4) adapter flip; (5) delete the class
  and re-derive its transcript tests as contract tests.
- No code changed in this step; the map above is the work plan.
**R37 - Phase 5: 1Cat Rust helpers extracted (2026-11-03, DONE)**. R36
step 1:
- The nine Rust-toolchain helpers (`_managed_rust_root`, `_rust_exe`,
  `_bin_dir_has_rust`, `_dir_writable`, `_retarget_unwritable_rust_homes`,
  `_apply_rust_bin_dir`, `_rust_bin_candidates`, `_discover_rust_bin_dir`,
  `_build_env`) moved VERBATIM into an in-module `OneCatRustMixin`;
  `OneCatVllmInstaller` inherits it.
- Deliberate choice: the mixin lives in `onecat.py`, NOT a separate module -
  the helpers reference module-level constants/state, and a separate file
  made five source-build tests fail with unresolved names. Same-module keeps
  the extraction risk-free until those constants are data-driven.
- Verification: 1Cat suites (source build, wheels, legacy shadow, launch
  parity) **34 passed**; engine slice **379 passed / ~41s**; FULL parallel
  suite **1956 passed, 8 skipped**; defcheck 12; ratchet no growth; ruff 68.
**R38 - Phase 5: 1Cat wheel-forge/release helpers extracted
(2026-11-03, DONE)**. R37 step 2 (all four moved VERBATIM into the SAME
in-module `OneCatRustMixin`, deliberately not a second base):
- `_ensure_rust` (managed toolchain bootstrap), `_build_source_wheels`
  (the wheel forge), `_fetch_release` and `_select_release_wheels` (release
  wheel selection). They use module constants/state like the Rust helpers,
  so same-module extraction remains the safe choice.
- What is left class-owned in `OneCatVllmInstaller`: the four flows
  (`install_release`/`install_from_source`/`sync`/`remove` -> generic-style
  `_execute_*_plan`), `retry_existing_install`, `status()`,
  `_resolve_binary_path`, and the plan handler wiring. Next: definition-
  driven subclass + `_extra_plan_handlers` for the wheel-forge/release
  primitives, adapter flip, then delete the class and convert its transcript
  tests to contract tests.
- Verification: 1Cat suites (source build, wheels, legacy shadow, launch
  parity) **34 passed**; engine slice **379 passed / ~39s**; FULL parallel
  suite **1956 passed, 8 skipped**; defcheck 12; ratchet no growth; ruff 68.
**R39 - Phase 5: 1Cat step 3 mapped (2026-11-03)**. The definition already
carries all four recipes; the remaining work is a HANDLER MAP for the
primitives the generic host does not implement. Exact inventory from
`definitions/1cat_vllm.yaml`:
- already generic: `env.compose`, `vcs.workspace_checkout`, `vcs.sync`,
  `vcs.seal_checkout`, `publish.finalize`/`install`/`record_update`/
  `forget`, `pip.install_package`, `venv.create`, `fs.remove_tree`.
- specialised (needs `_extra_plan_handlers`): `pip.install_requirements`,
  `artifact.build_wheels`, `env.compose_ccache`, `toolchain.ensure`,
  `prereqs.resolve_release_assets`, `fs.prune_files`, `pip.uninstall`,
  `fs.ensure_dir`.
- Implementation map (each handler receives step/requirements/context and
  must return the step's `produces`): `toolchain.ensure` -> `_ensure_rust`;
  `env.compose_ccache` -> `_build_env` (+ ccache env); `env.compose` -> rust
  build env; `artifact.build_wheels` -> `_build_source_wheels`;
  `pip.install_requirements` -> a requirements-install loop over the
  checkout; `prereqs.resolve_release_assets` -> `_fetch_release` +
  `_select_release_wheels`; `fs.prune_files` / `pip.uninstall` /
  `fs.ensure_dir` -> file/pip helpers.
- Next: write these eight handlers on a `OneCatInstaller(OneCatRustMixin,
  GenericVenvInstaller)` subclass (mirroring the V100 registration), pin the
  recipes' plans + handler sets, then flip the adapter and delete the class.
- No code changed this step; the inventory above is the exact task list.
**R40 - Phase 5: 1Cat handler map hits a CONTRACT conflict (blocked,
2026-11-03)**. Inspecting the legacy `_execute_source_plan` handlers before
writing the new subclass surfaced a real mismatch, not just missing code:
- `env.compose` in the 1Cat recipes produces **`toolchain_env`**, and a
  SEPARATE `env.compose_ccache` produces the ccache env. The generic host's
  `env.compose` contract produces `build_env` + `runtime_meta`, and its
  handlers are applied AFTER `_extra_plan_handlers` (core overrides win),
  so a 1Cat subclass cannot substitute its env behaviour through the
  current hook without changing the generic contract or the recipes.
- The legacy handlers are closures over the flow (`clone_dir`,
  `workspace_holder`, runs with append=False, stage strings) and several
  primitives are 1Cat-shaped (`artifact.build_wheels`, `pip.uninstall`,
  `fs.prune_files`, `prereqs.resolve_release_assets`).
- Two honest options, needing a decision before code:
  (A) Rewrite the four 1Cat recipes onto the generic vocabularies
      (`build_env`/`runtime_meta`, `vcs.workspace_checkout`, wheel-forge as
      `exec.run_logged`/`artifact.build_wheels` with declared produces), then
      a thin subclass needs only `_build_env` + wheel helpers;
  (B) extend the host with per-engine handler PRECEDENCE (extra handlers win
      over core ones) and keep 1Cat's env vocabulary.
- Recommendation: (A). It keeps one env contract across engines and avoids a
  precedence rule that could silently shadow core primitives; (B) is a
  smaller diff but weakens the host's invariants.
- No code changed; 1Cat remains on its legacy class until this is decided.
- Verification: tree green at R38 state (1Cat 34; slice 379; full 1956/8).
**R41 - Recipe wiring invariant harness (2026-11-03, DONE)**. Before
rewriting 1Cat's recipes (option A), added the harness that makes such a
rewrite safe:
- `test_engine_recipe_wiring.py` loads EVERY shipped definition and every
  recipe and asserts: each `requires` name was produced earlier or is an
  `input.*`/context key; every `with` reference ending `_from` names a
  producer (or an input/settings reference); and (for the migrated
  python-venv family) every primitive is one the shared host/executor
  knows.
- The harness immediately found two things: `workspace_from: workspace` is
  delivered by the checkout HANDLER (undeclared extra output) rather than a
  declared produce - allowed explicitly; and native/audio engines use
  engine-local primitives (`cmake.configure`, `cmake.build`,
  `artifact.remove_bundle`), so the vocabulary check is scoped to the
  shared-host family and documented as such.
- Note: it does NOT catch the 1Cat env-contract conflict by itself (the
  recipe is internally consistent); that remains a recipe-vocabulary
  rewrite decision (R40 option A).
- Verification: harness **44 passed**; engine slice **423 passed / ~42s**;
  FULL parallel suite **2000 passed, 8 skipped**; defcheck 12; ratchet no
  growth; ruff 69.
**R42 - Review fixes + 1Cat normalization attempt (2026-11-03)**.
- Wiring harness corrected per review: materialization failures now RAISE
  (no silent skip); each step's `requires` are checked BEFORE its outputs
  are recorded (a step can no longer satisfy itself); the duplicate
  primitive allowlist test was DELETED; only real recipes are iterated
  (`definition.recipes`), with loud failures.
- Structural/source-layout pins removed from the SGLang parity file (flow
  inheritance, mixin layout, status source scans); behavioural flow pins
  remain. One shared constant was restored inside the V100 flow pin.
- 1Cat normalization ATTEMPT: renaming `env.compose` ->
  `env.compose_toolchain` was implemented, and REJECTED by the schema's
  known-primitive validation - i.e. the vocabulary is closed by design and
  a new primitive name is not the answer. Reverted (export + handler maps),
  tree green.
- Correct normalization (next, now unambiguous): FOLD ccache into the
  generic `env.compose` - one step producing `build_env` (+ runtime_meta),
  delete the `env.compose_ccache` step, layer `ccache_environment(...)`
  inside the 1Cat build-env hook, and keep downstream `env_from: build_env`
  references. No new vocabulary.
- Verification: 1Cat+wiring **73 passed**; engine slice **416 passed /
  ~40s**; FULL parallel suite **1993 passed, 8 skipped**; ratchet no growth;
  ruff 69.
**R43 - Phase 5: 1Cat env contract NORMALIZED (2026-11-03, DONE)**. R42's
fold executed, after the schema rejected both a new primitive name and a
duplicate declared output:
- 1Cat's `compose_env` step now produces the shared vocabulary name
  **`build_env`** (its handler is the rust build env); `layer_ccache`
  (generic `env.compose_ccache`) layers ccache ON TOP and declares no output
  (a duplicate `produces` is rejected) - downstream steps keep their
  `env_from: build_env` references unchanged.
- Legacy `compose_env` handlers return `build_env`; no new primitive was
  introduced; the closed primitive vocabulary stayed closed.
- Pitfall fixed mid-flight: a global textual replace of the produces list hit
  EIGHT steps across other engines and the export rejected it (vllm's
  finalize requires runtime_meta); the change is now scoped to the two 1Cat
  compose steps only, and the exporter was re-formatted.
- Verification: 1Cat+wiring **73 passed**; engine slice **416 passed /
  ~40s**; defcheck 12; ratchet no growth (a stray temp script removed);
  ruff 69.
**R44 - Phase 5: 1Cat handler contracts enumerated from the live recipes
(2026-11-03)**. With the env contract normalized, the exact handler surface
for the definition-driven subclass is now machine-derived from
`materialize_plan(1cat_vllm, recipe)`:
- release: `prereqs.resolve_release_assets` (input.requested_version ->
  release_tag + wheel_urls), `venv.create`, two `pip.install_package`
  (bootstrap + wheels), `publish.finalize` (requires release_tag).
- source: create_venv, env.compose (build_env), fs.ensure_dir
  (input.dist_dir), vcs.workspace_checkout, env.compose_ccache
  (build_env+checkout), toolchain.ensure (build_env), bootstrap pip,
  pip.install_requirements (requirement_files/extra_index_url), build
  packages, artifact.build_wheels (checkout+build_env+dist_dir -> wheels,
  empty_error), install wheels, vcs.seal_checkout, publish.finalize.
- sync: same env/ccache/toolchain chain + vcs.sync, fs.prune_files
  (dist_dir, suffix), wheels build/install, publish.record_update.
- remove: pip.uninstall (input.venv_path, packages), fs.remove_tree
  (path_from + retire_references_from), publish.forget.
- Handler mapping (all implemented by mixin/legacy helpers):
  toolchain.ensure -> `_ensure_rust`; artifact.build_wheels ->
  `_build_source_wheels`; prereqs.resolve_release_assets ->
  `_fetch_release` + `_select_release_wheels`; pip.install_requirements /
  pip.uninstall / fs.prune_files / fs.ensure_dir -> small wrappers over
  `_run_pip`/`_run_logged`/fs helpers.
- Next implementation step: subclass + these seven handlers + plan/handler
  pins, then adapter flip. No code changed this step (analysis only).
**R45 - Phase 5: OneCatInstaller registered (2026-11-03, DONE)**. The
definition-driven 1Cat class now serves the adapter/route surface:
- `OneCatInstaller(OneCatRustMixin, GenericVenvInstaller)`: inherited flows;
  `_compose_build_env` -> `_build_env()` (rust + ccache chain unchanged);
  1Cat's seven specialised primitives supplied via `_extra_plan_handlers`
  (`toolchain.ensure` -> `_ensure_rust`; `artifact.build_wheels` ->
  `_build_source_wheels`; `prereqs.resolve_release_assets` ->
  `_fetch_release` + `_select_release_wheels`; plus thin wrappers for
  `pip.install_requirements`, `pip.uninstall`, `fs.ensure_dir`,
  `fs.prune_files`).
- `get_onecat_vllm_manager()` now returns the new singleton; the legacy
  class remains ONLY for its transcript tests until they are ported.
- Verification: 1Cat+wiring **78 passed**; engine slice **416 passed /
  ~41s**; FULL parallel suite **1993 passed, 8 skipped**; defcheck 12;
  ratchet no growth (stray temp removed); ruff 69.
**R46 - Phase 5: 1Cat cutover ATTEMPTED and REVERTED (2026-11-03)**. The
definition-driven `OneCatInstaller` was implemented (inherited flows,
seven specialised handlers, `MANAGER_NAME = "onecat_vllm"`, dist-dir hook,
source-recipe `capture_commit`) and could run its recipes, but the cutover
FAILED INTEGRATION:
- 12-15 cross-test failures appeared in `test_task_id_integration.py`: the
  route/task harness starts real operations on the manager fixture and the
  new class left supervisor state behind for later tests (cuda/llama cases
  then failed); individual cases passed in isolation.
- Earlier in the same attempt, the contract harness exposed TWO REAL recipe
  gaps (now recorded): 1Cat's source recipe had no `capture_commit` step
  (rows lost the built commit) and `finalize` did not require it.
- Decision: revert to the last green 1Cat state (R37: Rust helpers in an
  in-module mixin; wheel/release helpers back on the class; legacy manager
  registered). The generic `_source_dist_dir` hook and the sglang context
  pin (`input.dist_dir`) KEPT - they are harmless and correct.
- Re-application plan (fresh context): (1) re-extract the wheel/release
  helpers into the mixin; (2) subclass + handlers (recipe changes now known:
  `capture_commit` after seal, finalize requires it); (3) fix the task-id
  harness fixture isolation (per-test supervisor/journal reset) BEFORE the
  adapter flip; (4) contract tests; (5) delete the class.
- Verification after revert: engine slice **416 passed / ~39s**; FULL
  parallel suite **1974 passed, 8 skipped**; defcheck 12; ratchet no growth;
  ruff 68.
**R47 - Task-id harness isolation hardened (2026-11-03, DONE)**. The R46
revert identified the real blocker: `test_task_id_integration.py` cleared
only the module-level SINGLETON globals between tests, so a manager
constructed later by a route (the migrated classes) kept `_operation` /
`_current_task` / `_progress_task_id` state, and the next test's
`_begin_operation` raised - every later failure cascaded from that.
- The autouse fixture now QUIESCES any instance it finds (operation, current
  task, progress task id, started-at, cancelling) both before and after each
  test, and resets the singletons as before. This is the isolation the 1Cat
  cutover needed and is independent of which manager class is registered.
- Verification: `test_task_id_integration.py` **28 passed** (full file);
  engine slice **416 passed / ~40s**; FULL parallel suite **1974 passed, 8
  skipped**; defcheck 12; ratchet no growth; ruff 68.
**R48 - Phase 5: 1Cat re-application attempt #2 (2026-11-03)**. With the
task-id harness isolation in place (R47), the 1Cat cutover was re-applied:
- wheel/release helpers re-homed into the mixin; subclass + handlers +
  `MANAGER_NAME` re-added. `test_task_id_integration.py` (28) PASSED with
  the new class registered - the R47 isolation fix works.
- Deleting the legacy class then exposed a RESOURCE-KEY conflict: the
  supervisor still admits by `engine:<id>:<base_dir>` and the
  direct-manager parametrized tests leave supervisor operations registered
  (the quiesce clears INSTANCE state, not supervisor entries), so later
  cases fail with ResourceBusyError. The re-application scripts also
  duplicated the class definition, producing two `OneCatInstaller` classes.
- Reverted to the clean pre-cutover file to keep the tree green. The
  harness isolation stays; the 1Cat code changes do not.
- Next, precise: (1) make `_quiesce` ALSO call
  `get_supervisor().finish_operation(task_id, ...)` for a leftover
  `_progress_task_id`; (2) re-apply the subclass from a CLEAN file with an
  idempotent script (assert no existing `OneCatInstaller`); (3) verify the
  task-id file, then delete the legacy class.
- Verification after revert: engine slice **416 passed / ~39s**; FULL
  parallel suite **1974 passed, 8 skipped**; defcheck 12; ratchet no growth;
  ruff 68.
**R49 - Phase 5: 1Cat attempt #3 - task-id isolation fixed, class deletion
still blocked (2026-11-03)**. With the supervisor-finish added to the
quiesce helper, the definition-driven installer was re-applied (idempotent
guard), `test_task_id_integration.py` passed 28/28 and all 1Cat suites were
green (43). Deleting the legacy class then re-triggered the SAME cascade:
- 15 failures in the task-id file, including CUDA cases that pass alone.
  The supervisor resource-key (`engine:1cat_vllm:<base_dir>`) stays admitted
  because the direct-manager parametrized tests build managers whose
  `_progress_task_id` is registered in the supervisor and never finished;
  `finish_operation` in `_quiesce` is not closing the admission in the path
  those tests take (likely the cancelled/failed state path or a different
  registry entry).
- Reverted to the green pre-cutover file again. The quiesce change (which is
  still an improvement) remains.
- Precise remaining diagnosis: instrument `supervisor._admit_build_or_install`
  during the 4 parametrized manager cases to see which operation id stays
  admitted, and close it in the fixture via the same API the supervisor uses
  internally (or expose a `release_resource(key)` for tests).
- Verification after revert: 1Cat+task-id **43 passed**; engine slice **416
  passed / ~39s**; FULL parallel suite **1974 passed, 8 skipped**; defcheck
  12; ratchet no growth; ruff 68.
**R50 - Task-id harness: supervisor admission leak CLOSED (2026-11-03,
DONE)**. The R49 diagnosis was right but the mechanism was different from
instance quiescing: the parametrized DIRECT-MANAGER cases construct managers
that are NOT the module singleton, so instance-level cleanup cannot see
their task ids - while their supervisor rows/claims persisted and blocked
later tests (ResourceBusyError cascade, including unrelated CUDA cases).
- Fix: a second autouse fixture snapshots `supervisor._resources` owners
  after each test and calls `finish_operation(owner, "failed", "test
  cleanup")`, force-releasing every admission. Test-only; no production
  behaviour touched.
- Verification: `test_task_id_integration.py` **28 passed** in one run;
  engine slice **416 passed / ~38s**; FULL parallel suite **1974 passed, 8
  skipped**; defcheck 12; ratchet no growth; ruff 68.
- With this in place the 1Cat subclass re-application can proceed; the
  subclass script (idempotent guard, handlers, MANAGER_NAME) is the R48/R49
  recipe and the legacy class deletion is the final step.
**R51 - Phase 5: 1Cat CUTOVER COMPLETE - legacy class DELETED
(2026-11-03, DONE)**. Final sequence:
- Subclass re-applied from the clean file (idempotent guard), wheel/release
  helpers in the mixin, seven specialised handlers, `_source_dist_dir`,
  `MANAGER_NAME`.
- CRITICAL detail found: `PythonVenvInstaller.__init__` sets
  `self.MANAGER_NAME = engine_id`, overwriting a class attribute - the new
  class now sets `self.MANAGER_NAME = "onecat_vllm"` AFTER super().__init__,
  so task ids keep the legacy token (`install_onecat_vllm_...`).
- The fork's definition gained its installer `lifecycle` (repo
  `https://github.com/1CatAI/1Cat-vLLM.git`, root/log/dist/version keys), so
  `install_from_source` resolves the default repo.
- The legacy `OneCatVllmInstaller` class is DELETED; `OneCatVllmInstaller` /
  `OneCatVllmManager` are compatibility aliases of `OneCatInstaller` and the
  manager getter returns the definition-driven singleton.
- Verification: task-id + 1Cat suites **43 passed** (task-id file 28/28);
  engine slice **416 passed / ~36s**; FULL parallel suite **1974 passed, 8
  skipped**; defcheck 12; ratchet no growth; ruff 68.
**R52 - Review fixes: six findings closed (2026-11-03, DONE)**.
- [P1] 1Cat source requirements: only files that EXIST in the checkout are
  installed (candidate lists span fork layouts); the handler reports the
  count. [P1] Detection now also recognises the `vllm` distribution used by
  older two-wheel releases (lifecycle `distribution_names`).
- [P2] Release publication preserves the selected GitHub tag: generic
  `_release_meta(requirements)` hook (1Cat returns `release_tag`), merged
  into the published row and into the frozen snapshot inputs; retry prefers
  the frozen tag over the package version. [P2] `_ensure_rust` env updates
  (PATH/CARGO_HOME/RUSTUP_HOME) now flow to later wheel builds by returning
  the mutated `build_env`.
- [P2] `/api/llama-versions` now also enumerates DEFINITION-registered
  engines (not just the hardcoded nine), so a YAML-engine install shows its
  version card; new isolated test. [P2] GenericEnginePanel removal copy now
  states it removes the ACTIVE version + its row (and keeps the others).
- Verification: engine slice **418 passed / ~38s**; FULL backend suite
  **1975 passed, 8 skipped**; frontend **611 passed**; defcheck 12; ratchet
  no growth; ruff 70.
**R53 - GitHub-wheel installs publish `install_type: release` (2026-11-03,
DONE)**. Final review mismatch: the 1Cat release path kept the generic
`install_type: pip` even though the row carries `release_tag`. The
`_release_meta` hook now returns `install_type: "release"` together with the
tag, so the published row AND the frozen snapshot inputs record the release
provenance; retry still routes through `install_release` (kind is not
source). Test asserts both values.
- Verification: release/1Cat/task-id suites **39 passed**; engine slice
  **418 passed / ~38s**; FULL backend suite **1976 passed, 8 skipped**;
  defcheck 12; ratchet no growth; ruff 70. (Frontend untouched this step.)
**R54 - Phase 7: audio.cpp scoped - it needs a NATIVE host, not the venv
host (2026-11-03)**. Recon before writing code:
- `backend/engines/audio_cpp/` is 4,851 lines across 16 modules; the
  installer-ish core is `manager.py` (921 lines: `build_source`,
  `sync_source`, `delete_version_files`, `_cmake_args`, `_compile_tree`,
  streaming runner, binary discovery, local-checkout/sync-dir rules).
- `audio_cpp.yaml` already carries artifact/archive recipes
  (`archive.download`, `archive.extract_tar`,
  `archive.verify_checksums_file`, `artifact.locate`,
  `artifact.materialize_cli`, `artifact.remove_bundle`,
  `publish.install`/`forget`, `fs.*`) - i.e. the RELEASE/prebuilt path is
  declarable, but the SOURCE build is class-owned cmake logic.
- KEY finding: audio is NOT a python-venv engine; the shared
  `GenericVenvInstaller` host does not apply. Phase 7 therefore needs either
  (A) a small `GenericNativeInstaller` host (install dir + recipe executor +
  publish/forget) that audio and the native llama engines could share, or
  (B) keep audio's manager and only move its release path onto recipes.
- Recommendation: (A), but as its own tranche - it is a new host, not an
  extraction; the V100/1Cat mixin pattern will NOT delete audio's manager.
- No code changed this step.
**R55 - Phase 7 feasibility audit: the native host is SMALL (2026-11-03)**.
Programmatic audit of audio's recipe primitives against the executor's
generic implementations:
- generic already: `archive.download`, `archive.extract_tar`,
  `archive.verify_checksums_file`, `artifact.locate`, `fs.remove_tree`,
  `fs.ensure_dir`, `fs.remove_file` (7 of 11).
- need handlers (4): `artifact.materialize_cli` (audio's symlink/CLI
  materialisation incl. `$studio_audio_config`), `artifact.remove_bundle`,
  `publish.install`, `publish.forget` (install-dir rows instead of venv
  rows).
- Therefore option (A) from R54 is tractable: a `GenericNativeInstaller`
  needs only those four handlers plus install-dir/version-row bookkeeping
  and the four flows; the archive/artifact/fs vocabulary is already generic.
  It can then host audio.cpp AND the native llama engines, replacing three
  legacy managers rather than one.
- Next implementation slice (fresh context): (1) `GenericNativeInstaller`
  with the four flows mirroring the venv host but on `install_dir` +
  `build_dir`; (2) the four handlers; (3) move audio's release path onto
  recipes, then its source build (`cmake.*` primitives already exist);
  (4) native llama engines afterwards.
- No code changed this step (audit only).
**R56 - Phase 7, first extraction: native binary resolution shared
(2026-11-03, DONE)**. `AudioCppManager`'s binary discovery
(`_binary_candidates` / `_find_binary`) moved into the new
`backend/engines/native_support.py` as module functions; the manager keeps
static delegates so existing call sites and tests are unchanged. First code
step toward the native host, and reusable by the native llama engines.
- Verification: audio suites **42 passed**; engine slice **418 passed /
  ~38s**; FULL backend suite **1976 passed, 8 skipped**; defcheck 12;
  ratchet no growth; ruff 70.
**R57 - Phase 7: native_support pinned (2026-11-03, DONE)**. The shared
native helper module now has direct unit pins instead of only indirect audio
coverage: candidate ordering across common cmake layouts, `bin/` preference,
absolute-path return, executable suffix per platform, and the typed
"binary not found" error including the searched root.
- Verification: native+audio tests **14 passed**; engine slice **421 passed
  / ~39s**; FULL backend suite **1979 passed, 8 skipped**; defcheck 12;
  ratchet no growth; ruff 71.
- Next: the `GenericNativeInstaller` host core (four flows on
  install_dir/build_dir) + the four handlers from R55, pinned with a
  synthetic definition before audio is wired.
- STILL OPEN beyond this campaign: snapshot-backed retries, the
  specialized-installer extraction, audio migration, and the remaining
  eight-engine launch cutover (LMDeploy now compiles from its definition;
  the others keep their legacy adapters until their gated item).
**R58 - Phase 7: GenericNativeInstaller host core (2026-11-03, DONE)**. The
native host exists and stands on its own tests; audio is deliberately NOT
wired to it yet.
- `backend/engines/python/native.py`: `GenericNativeInstaller(definition,
  install_root=…, log_path=…)`. Same flow shape as the venv host (`_recipe` ->
  `materialize_plan` -> `execute_plan` with `make_engine_runner(self,
  plan-filtered handlers)`), but the bound paths are `install_dir` +
  `build_dir` — no venv, no pip, no venv row activation. Surface:
  `install_release` / `install_from_source` / `sync_source_version` (+ the
  `sync_source` protocol alias) / `remove` / `status` / `read_log_tail`.
  Synchronous core: each flow awaits `execute_plan`, finishes its operation
  and returns the routes' started shape `{"message", "task_id"}`; no
  background tasks are spawned in this slice.
- The four R55 handlers with install-dir semantics: `publish.install` writes
  the row (version/type/install_type/source fields/install_dir/installed_at/
  repository_source plus recipe-declared `<field>_from: <produced>` path
  fields), records the version-associated published snapshot, then activates
  when `input.auto_activate`; `publish.forget` deletes the row under
  `lifecycle.forget_errors` (swallow default, raise strict);
  `artifact.materialize_cli` creates an ABSOLUTE entry (symlink, executable
  copy fallback) and is idempotent (an existing entry is current; a symlink
  pointing elsewhere is replaced); `artifact.remove_bundle` removes one tree
  with a realpath containment guard — a path outside the install root (or the
  root itself) is skipped, never deleted. `archive.download` is served by the
  host's HTTP(S)-only `_download_file`.
- Flow policy: install dirs must live INSIDE `install_root` (refused
  otherwise); sync republishes the recorded version and re-activates only the
  version that was already active (legacy audio semantics); remove targets the
  ACTIVE row's `install_dir` and clears the binding afterwards.
- Review fix (P1, disclosed): version naming normalized AFTER the uniqueness
  check, so a 64-char version lost its `-<stamp>` collision suffix and the
  second install resolved to the FIRST installation's directory. Naming now
  normalizes first and appends the suffix to the normalized stem with room
  reserved for it (`_unique_version_name`); `_safe_slug` is idempotent and
  never leaves a trailing separator, and `_claim_install_dir` derives the
  install dir from the already-normalized name (never re-slugs it). Pinned by
  `test_long_version_names_keep_their_uniqueness_suffix` (three consecutive
  installs of the same 64-char version: distinct names, distinct dirs, every
  name ≤ 64 chars and the suffix intact). RED evidence on the old expression:
  `_safe_slug(unique_version_name(store, id, "v"*64)) == "v"*64`.
- `backend/tests/test_generic_native_installer.py` (15 tests, synthetic
  `nativecraft`, zero engine-id literals): real tar.gz release journey
  (download hook -> extract -> verify -> locate -> materialize CLI -> publish)
  asserting row fields, the published snapshot and the activation hook;
  materialize idempotency; remove + the outside-root guard; a handler failure
  leaving no row while surfacing the error; source/sync provenance including
  inactive-sync no-activation; the long-version collision above; and a
  plan/handler/context pin for all four flows patching BOTH
  `definitions.executor` and this module's bound copies.
- Repair disclosed: `backend/engines/audio_cpp/manager.py` was missing its
  `native_support` import (R56 left the static delegates calling undefined
  names); `ruff check` caught it, one import added, audio suites green. That
  line shift also made `docs/engine-parity-inventory.{json,md}` stale (it
  records literal line numbers inside the engine home); the inventory was
  REGENERATED — the only diff is `audio_cpp/manager.py` 135/138/631 ->
  136/139/632, markdown byte-identical, `--check` exit 0.
- Verification (lanes only): new file **15 passed**; targeted set (native
  host + installer policy + native_support + recipe wiring) **70 passed**;
  audio core suites **55 passed**; engine slice **436 passed / ~37s**; FULL
  parallel **1994 passed, 8 skipped**; ratchet no growth (exit 0); parity
  inventory current (exit 0); defcheck 12 engines; ruff curated 73 files.
- Next: move audio.cpp's release path onto the `prebuilt` recipe with this
  host, then its source build (`cmake.*`), then the native llama engines.
**R59 - Phase 7: audio.cpp's PREBUILT (release) path runs on the native host
(2026-11-03, DONE)**. First engine wired to `GenericNativeInstaller`; the
route keeps its task plumbing and the legacy route-local plan host is deleted.
- Host seams added (no vocabulary growth): `_extra_plan_handlers()` — engine
  primitives REPLACE core handlers (extras win, deliberately: the override is
  the engine's declared difference) — and `_row_extras(context)` for
  plan-derived row provenance a `with` scalar cannot express, plus
  `execute_operation(recipe, task_id=…, install_dir=…, context=…)` for routes
  that still own admission/task/terminal bookkeeping: it binds the route's
  task id for progress and the install dir, restores both, and starts/finishes
  NO operation.
- `backend/engines/audio_cpp/installer.py`: `AudioCppInstaller(
  GenericNativeInstaller)` supplies ONLY the three audio deltas, delegating to
  the existing helpers (no second implementation): `_download_file` ->
  `prebuilt.stream_release_download` (HTTP(S) guard), `artifact.
  materialize_cli` -> `prebuilt.materialize_cli` (CPU companion archive,
  progress 55), `_row_extras` -> `cuda_version`/`build_config`/
  `source_branch`/`is_fork` from the resolved `PrebuiltPlan`.
- The definition owns the row's active-path fields: `audio_cpp.yaml`'s
  `publish_row` now declares `server_binary_path_from: server` /
  `cli_binary_path_from: cli_final` (exported by
  `scripts/export-engine-definitions.py`, so the recipe stays generated).
- Route cutover: `_prebuilt_task` awaits `installer.execute_operation(
  "prebuilt", …)` and keeps every surrounding contract (admission/fence,
  `pm.complete_task`/`fail_task`, success/failure notifications, cancelled-row
  marking, archive cleanup). `_PrebuiltPlanHost` and `_execute_prebuilt_plan`
  (the route-local host plus its publish/materialize closures) are DELETED.
  Version naming, task ids and response bytes are unchanged (the route still
  claims version names).
- Contract pins preserved and extended
  (`test_audio_cpp_legacy_plan_shadow.py`): same progress sequence (15
  download, 55 companion), same archive-deletion point, same error/cancel
  strings, same row fields, PLUS the published snapshot row and a structural
  pin that the route executes audio's OWN plan on `AudioCppInstaller` with the
  plan-filtered handler set and the exact context vocabulary. Harness port:
  readiness/activation/store are patched on `backend.engines.python.native`
  and shared-primitive progress on
  `backend.operations.cancellable.get_progress_manager`; the host unit pin
  `test_execute_operation_binds_the_route_task_and_install_dir` covers the
  binding/restore.
- Verification (lanes only): audio + host + wiring subset **116 passed**;
  engine slice **437 passed / ~37s**; FULL parallel **1996 passed, 8 skipped**;
  ratchet no growth (exit 0); parity inventory REGENERATED and current
  (exit 0) — audio-route literal count 52 -> 48 from the deleted closures,
  plus the new installer file and the exporter's line shift; defcheck 12
  engines; ruff curated 75 files.
- Next: audio's remove path (`remove_version` still runs in
  `routes/llama_versions.py` with its own handler pack — the host's
  `publish.forget`/`artifact.remove_bundle` are ready for it), then the source
  build (`cmake.*`), then the native llama engines.
**R60 - Phase 7: audio.cpp's version REMOVAL runs on the native host
(2026-11-03, DONE)**. Second audio flow on `GenericNativeInstaller`, and the
manager's removal helper is deleted.
- Host: `execute_operation(..., store=...)` binds the caller's store for
  bookkeeping (restored afterwards), so a route that owns its store keeps test
  and production parity; `publish.forget` now honours `input.registered`
  (removal flows set it) and every bookkeeping call goes through `_get_store()`.
- Adapters: `get_native_installer(engine)` is the data-driven seam for engines
  migrated to the host (audio.cpp registered, constructed with the manager's
  `builds_dir`). The shared route gained NO engine-ID branch.
- `AudioCppInstaller` now also owns `artifact.remove_bundle`: the manager's
  version-bundle resolution moved VERBATIM into the host subclass
  (`source_path` -> `install_dir` -> `<builds>/<version>`; refuse outside the
  builds root; walk up to the version directory, never the root; remove with
  the launch hold released; the `ValueError` refusals preserved). It skips when
  the recipe's own tree step already removed the same target.
- `AudioCppManager.delete_version_files` DELETED (its only caller, the route's
  `remove_bundle` closure, is deleted with it). `Path` import pruned.
- Route: `_remove_engine_version` builds the plan context once (now carrying
  `input.version_entry`), runs the recipe on the engine's native host when one
  is registered (`task_id=""` — no operation is started), and keeps its legacy
  handler pack for engines without a host. Error mapping preserved:
  `VersionDeletionRejected`/`FilesystemRefusal` first, then `ValueError` -> 400
  (the audio refusal contract, now enforced for the host path), else 500.
- Pins re-derived/added: audio removal deletes the tree + row through the
  host; an outside-builds-root row still maps to 400 with the pinned ordering
  (tree step first); a structural pin proves the route executes audio's OWN
  `remove_version` plan on `AudioCppInstaller` with handlers
  `{artifact.remove_bundle, publish.forget}` and the pinned context.
- Verification (lanes only): removal+audio+host+router+wiring subset
  **176 passed**; engine slice **438 passed / ~37s**; FULL parallel
  **1997 passed, 8 skipped**; ratchet no growth (exit 0); parity inventory
  REGENERATED and current (exit 0) — route/manager/adapters line shifts plus
  the new engines-home literal site; defcheck 12 engines; ruff curated 78.
- Next: audio's SOURCE build — the last class-owned flow; needs the `cmake.*`
  primitives and workspace checkout on the host — then the native llama
  engines.
**R61 - Phase 7, source-build step 1: ONE streaming runner + cmake/smoke
primitives (2026-11-03, DONE)**. The plan's "one process runner for logs,
cancellation, process-group termination and progress events" now exists;
audio's source build is production-verified on it, and the host exposes the
build vocabulary the source recipe needs.
- `backend/engines/native_build.py`: `run_streaming_command(argv, …)` extracted
  VERBATIM from `AudioCppManager._run_streaming` (same 0.25 s readline /
  0.4 s flush / 12-line batch thresholds, `BuildProgressTracker` +
  `cmake_stage_window` math, the `"<stage> failed with exit code N:\n<last 40
  lines>"` error, cancellation and terminate-before-raise semantics). It is
  callback-shaped on purpose: callers own their progress transport (audio's SSE
  `build_progress`, the host's task updates), their cancellation source
  (task-cancel registry / bound operation task) and their active-process
  bookkeeping. No engine, stage vocabulary or progress manager inside.
- `AudioCppManager._run_streaming` is now a thin adapter over the shared runner
  (same signature; stage messages via `_STAGE_MESSAGES`), so the production
  source-build path exercises the shared implementation.
- Native host: `_run_build_command` (streams through the shared runner honoring
  the bound task id's cancel registry and terminating on cancel),
  `_emit_build_progress`, `_terminate_active_process`, `_capture_command`, plus
  the plan-filtered handlers `cmake.configure` / `cmake.build` (`argv` /
  `argv_from`, `cwd_from`, `env_from`, `progress`, `progress_message`;
  RuntimeError -> PrimitiveExecutionError) and `smoke.test` (`argv` from
  context, `require_output` default true, configurable error).
- No vocabulary growth (all three names were already in
  `INSTALL_PRIMITIVES`); recipes declare argv, so engine build-option mapping
  stays data/engine-side.
- Pins (`backend/tests/test_generic_native_build.py`, 7) use REAL subprocesses
  (the interpreter as the fake toolchain): configure/build/smoke run end to end
  through `execute_operation` with the declared argv/cwd/env (a marker proves
  both) and publish the row; a failing build reports the pinned runner error and
  publishes nothing; empty smoke output fails with the declared message; a cmake
  step without argv is a typed failure; the runner streams/returns lines and
  emits the completion stage; a non-zero exit carries the output tail; a
  pre-requested cancellation refuses before spawning.
- Verification (lanes only): build+host+audio+removal subset **90 passed**;
  engine slice **445 passed / ~39s**; FULL parallel **2004 passed, 8 skipped**;
  ratchet no growth (exit 0); parity inventory REGENERATED and current (exit 0)
  — the audio manager's streaming block line shift; defcheck 12 engines; ruff
  curated 80.
- Next: audio's `source` recipe + `_build_task` cutover (workspace-checkout
  primitive + source-row publication), then the native llama engines.
**R62 - Phase 7: audio.cpp's SOURCE build runs on the native host
(2026-11-03, DONE)**. Third and last install flow on `GenericNativeInstaller`;
the manager's source-build flow is deleted.
- Definition: `audio_cpp.yaml` gains a `source` recipe (exported) that owns the
  STEP ORDER — workspace checkout → configure → build → locate both binaries →
  workspace-layout publication → `--help` gate ×2 → commit capture →
  `publish.install`.
- Host: `execute_operation(create_install_dir=…)` (a source build creates its
  version dir itself so the workspace publisher never replaces one),
  `_raise_if_cancelled()` (bound task id) and `_git_head` (the
  `vcs.capture_commit` hook).
- `AudioCppInstaller` source deltas (definition-driven, engine-owned): managed
  `BuildWorkspace` or plain recursive clone/checkout (`_checkout_source`,
  workspace held on the installer and released by the route's `finally`);
  configure/build via the manager's helpers (`_prepare_configure` /
  `_build_argv`, EXTRACTED from `_compile_tree` so sync keeps one
  implementation); `vcs.seal_checkout` publishes the workspace layout into the
  version dir and rebinds checkout/build/binaries; `smoke.test` runs the
  built `--help` with the legacy empty-output error; `_row_extras` now emits
  source provenance (`type`/`is_fork` from the route-selected labels,
  `source_branch`, `source_path`, `source_commit`, `build_config` with
  `prebuilt: False`/`release_tag` for release-tag builds, and the
  manager-path fields).
- Route: `_build_task` runs the recipe on the host (`execute_operation("source",
  …)`) and keeps every surrounding contract — admission/task, success
  notification, cancelled/failed rows with the legacy extras, `mark_runtime_changed`
  when not activating, activation through `publish.install`
  (`covered_by=task_id` so no separate activate operation is taken).
- `AudioCppManager.build_source` DELETED. The manager keeps the helpies the
  SYNC flow still uses (`sync_source` + `_compile_tree`); sync is the last
  class-owned flow and gets its own slice.
- Membership pin updated deliberately: audio_cpp's v2 recipe set is now
  `{prebuilt, remove_version, source}`.
- Pins: the prebuilt contract file gains a structural source pin (definition
  plan step-for-step, host instance, plan-filtered handler surface
  `{cmake.build, cmake.configure, publish.install, smoke.test,
  vcs.seal_checkout, vcs.workspace_checkout}` and the 12-key context); the two
  tests that used to fake `manager.build_source` now drive the REAL host with
  fast step stubs, so row/activation/fork-label assertions keep their meaning.
- Verification (lanes only): broad targeted subset (audio suites + recipe
  wiring + definitions equivalence/schema + task ids + version lifecycle +
  llama source build) **218 passed**; audio plan contract **11 passed**; engine
  slice **446 passed / ~37s**; FULL parallel **2006 passed, 8 skipped**; ratchet
  no growth (exit 0); parity inventory REGENERATED and current (exit 0) —
  exporter/route/manager line shifts and the deleted manager flow; defcheck 12
  engines; ruff curated 82.
- Remaining: audio's SYNC flow (`AudioCppManager.sync_source`), then the native
  llama family.
**R63 - Phase 7: audio.cpp's SYNC flow runs on the native host; audio complete
(2026-11-03, DONE)**. Fourth and last audio flow; the manager's build/sync
machinery is gone and the shared runner's SSE payload is preserved.
- Definition: `audio_cpp.yaml` gains a `sync` recipe (exported) for an EXISTING
  source install: `host.inspect` (resolve + validate the row's checkout/build
  dir) → `vcs.sync` → configure → build → locate ×2 → `vcs.seal_checkout`
  (passthrough, no workspace) → `--help` gates ×2 → commit capture →
  `publish.record_update`.
- Host: new `publish.record_update` primitive handler — merges a rebuilt row
  in place (`installed_at` PRESERVED, `updated_at` moves, `source_*` and the
  recipe's `<field>_from` plus engine row extras) and leaves activation to the
  caller. `AudioCppInstaller` gained `host.inspect` (recorded-layout resolution
  with the legacy refusals as typed primitive errors) and `vcs.sync` (the
  extracted `sync_git_checkout` helper driven through the shared runner);
  configure/build/validate reuse the source handlers.
- The streaming progress payload is now shared: `native_build.emit_build_progress`
  carries audio's exact `build_progress` SSE body (task update + delta), and the
  host emits through it, so the source/sync build UI sees the same events it
  did from the manager.
- Route: `_sync_task` runs `execute_operation("sync", …)` and keeps the legacy
  policy exactly — row re-read, `mark_runtime_changed`, scan only when the
  version is NOT active, activation when it was (or nothing else is), the
  "Synced audio.cpp …"/notification strings, cancelled/failed task handling.
- Deleted from `AudioCppManager`: `sync_source`, the `_compile_tree`/
  `_capture`/`_run_streaming`/`_emit`/`_raise_if_cancelled`/
  `_terminate_active_process`/`_sync_git_checkout` machinery and their unused
  imports. What remains is build configuration (`_prepare_configure`/
  `_build_argv`/`_cmake_args`), the sync-target resolution helpers and the
  module-level `sync_git_checkout` used by the host.
- Membership pin updated deliberately: audio_cpp's v2 recipe set is now
  `{prebuilt, remove_version, source, sync}`.
- Pins: the audio contract file gains a structural sync pin (definition plan,
  plan-filtered handler surface `{cmake.build, cmake.configure, host.inspect,
  publish.record_update, smoke.test, vcs.seal_checkout, vcs.sync}` and the
  context); the two sync-task tests now drive the REAL host with step stubs
  (row/scan/activation assertions unchanged); the manager's sync tests moved to
  the shared helper / host refuse-before-spawn seam.
- Verification (lanes only): audio+manager+build+contract+removal subset
  **132 passed**; engine slice **447 passed / ~37s**; FULL parallel **2007
  passed, 8 skipped**; ratchet no growth (exit 0); parity inventory regenerated
  and current (exit 0); defcheck 12 engines; ruff curated 83.
- audio.cpp is now COMPLETE on the native host (release/prebuilt, source build,
  sync, removal).
**R64 - Native-family audit: the llama engines are already recipe-executed
(2026-11-03, VERIFIED — one residual boundary recorded)**. Before scheduling a
"host migration" tranche, the native family was audited against the campaign's
own completion test ("the definition decides the behavior"). Findings:
- `llama_cpp`/`ik_llama`: `llama_cpp/source_build.py::run_source_build` loads
  the shipped definition, materializes the `source` recipe and executes it
  through the shared executor with a declared handler pack
  (`vcs.workspace_checkout`, `vcs.apply_patch`, `cmake.configure`,
  `cmake.build`, `artifact.locate`, `publish.finalize`). The recipe decides the
  STEPS; the handlers are the engines' declared primitives — the same pattern
  audio.cpp now uses (`_extra_plan_handlers`). `publish.finalize` materializes
  and validates the artifact (workspace layout/copy, `--version` smoke,
  progress 92-100).
- `unsloth_llama`: its shipped `release` recipe is likewise materialized and
  executed, and its `publish.finalize` handler writes the version row itself
  (release tag, descriptor/build_config, cuda_version) — publication is
  definition-driven there too.
- Removal is one shared, definition-driven flow for every engine (R60).
- RESIDUAL (deliberately recorded, not silently refactored): for llama/ik the
  version ROW is still written by `routes/llama_versions.py::build_source_task`
  after `run_source_build` returns (labels, `binary_path`, source commit/ref,
  `install_dir`, `installed_at`), even though the recipe declares
  `publish.finalize`. Moving that write into `_step_publish` is mechanical but
  requires threading `source_ref_type` into the build state and re-deriving the
  two llama fork-label tests (which fake `llama_manager.build_source` and would
  no longer publish) against the real `publish.finalize` handler. Doing it
  blind at the end of this pass would have risked the 1.7k-line source-build
  flow for a boundary with no user-visible difference today; it is the single
  remaining Phase-7 boundary, with the plan above.
- No code changed in this entry; verification stands at engine slice **447**,
  parallel full **2007 passed, 8 skipped**, ratchet/parity/defcheck/ruff green.
**R65 - Phase 7 COMPLETE: llama/ik publication moves into the recipe; the
journey flake is FIXED (2026-11-03, DONE)**. The R64 boundary was closed and one
cross-test isolation defect was repaired.
- Publication contract: `run_source_build(..., publish_row=False | "install" |
  "update", source_ref_type=…, repository_source=…, store=…)` threads the
  contract into `_BuildState`; the recipe's `publish.finalize` step now calls
  `_publish_source_row`:
  - `"install"`: upserts the row the route used to write (labels from the
    repository identity incl. fork/patched, the resolved commit from the
    PUBLISHED checkout via `git rev-parse`, source ref/type/branch,
    `install_dir`, `installed_at`, `build_config` + `repository_source`);
  - `"update"`: merges a rebuilt sync row IN PLACE (`installed_at` preserved,
    `updated_at` moved, branch provenance) and raises the historical
    `Version '…' disappeared during sync`;
  - `False`: no publication (characterization-harness default — no store
    leakage).
- `llama_manager.build_source` passes the contract through;
  `routes/llama_versions.py`'s `build_source_task` and
  `sync_source_build_task` supply it and no longer write rows; the dead
  `_llama_checkout_commit` helper and the `mark_engine_version_ready` import are
  deleted.
- Tests: the 470-line characterization suite gained a `RecordingStore` and SIX
  publication pins (install row, fork labels, upstream stays source,
  publication off unless asked, sync update preserving install provenance,
  missing-row error) — driving the REAL handler. The two route-level
  fake-`build_source` fork tests were retired; their coverage now lives at this
  seam (no behavior left unpinned).
- Flake FIXED (disclosed, pre-existing): store-IO failure history is
  process-global, so a fault-injection test's durability failure leaked into
  `test_launchcraft_journey`'s clean-status assertion under xdist (three
  consecutive parallel runs failed on it while isolation passed). A new autouse
  `conftest` fixture clears `store_io` persistence events around every test —
  the long-standing "rotating flake" is retired: the parallel lane is green
  twice in a row including the journey.
- Verification (lanes only): llama/fork/lifecycle/task-id subset **68 passed**;
  characterization **19 passed**; engine slice **447 passed / ~39s**; FULL
  parallel **2011 passed, 8 skipped** (twice); ratchet no growth (exit 0);
  parity inventory regenerated and current (exit 0); defcheck 12 engines; ruff
  curated 85 files.
- Phase 7 status: COMPLETE. Every engine's install/release/source/sync/remove
  flow executes its shipped definition recipe; engine-specific code lives in
  declared primitive handlers; audio.cpp is fully host-driven; llama/ik/unsloth
  publish from their recipes. The campaign's remaining items are the OTHER
  phases' gated decisions (the eight-engine launch cutover per R23/R18-R22 and
  the deferred R5 policy data), not Phase-7 slices.
**R66 - R5 LANDED: launch policy is engine data (2026-11-03, DONE)**. The
deliberately deferred policy-data item is closed; the proxy no longer knows
engine ids for launch constraints.
- `EngineSpec.launch_policy` (registry) carries the three historical facts,
  moved verbatim: `cpu_supported` (the five pip-wheel engines reject CPU mode),
  `gpu_only_invariants` (sglang_v100's CUDA_HOME / TORCH_CUDA_ARCH_LIST /
  FLASHINFER_DISABLE_VERSION_CHECK) and `runtime_context_key` (ctx_size /
  session_len / max_model_len / context_length). Excluded from the descriptor
  API like lifecycle/runtime_profile.
- Schema v2 `launch_policy` block: mapping with only those three keys, validated
  (bool / env-name list / string), normalized (tuple invariants) and
  round-tripped by `to_spec()`; schema-v1 documents reject it. The exporter
  writes the block into the generated definitions and verifies
  definition-vs-registry equality — so a definition-only engine declares its
  constraints in YAML.
- Proxy reads: `launch_spec._cpu_supported` / `_gpu_only_invariants` and
  `commands._runtime_context_key` resolve the policy from the engine spec.
  `_CPU_REJECTED_ENGINES`, `_GPU_ONLY_INVARIANTS` and `_RUNTIME_CONTEXT_KEYS`
  are DELETED (no engine-id set outside the engine home).
- Pins: `backend/tests/test_engine_launch_policy.py` (9) freezes the historical
  per-engine values, the read helpers (including unknown/explicit-none
  engines), descriptor-payload exclusion, the definition round-trip and the
  validation errors (unknown key, wrong types, v1 rejection).
- Verification (lanes only): definitions + all launch-parity suites
  **136 passed**; launch-policy pins **9 passed**; engine slice **456 passed /
  ~37s**; FULL parallel **2020 passed, 8 skipped**; ratchet no growth (proxy
  literals retired, inventory regenerated); parity current (exit 0); defcheck
  12 engines; ruff curated 88 files.
- Campaign status: R5 was the last signed-open item. The remaining launch
  boundary is DECIDED, not pending: native GGUF engines keep the legacy
  `_compile_gguf` per R23, and native snapshot-backed retries remain outside
  this campaign per R57.
**R67 - Scope correction + PLAN-PHASE-7 SCOPING: audio services/contracts
(2026-11-03, AUDIT ONLY)**. Review feedback is right: "Phase 7 complete" in
R63/R65 describes THIS campaign's Phase-7 tranche (the native installer host and
audio's install/source/sync/removal flows), not the refactoring plan's Phase 7
("Migrate audio services/contracts", plan line 136). That plan phase is
untouched and is scoped here; no code changed in this entry.
- What is complete: audio installs/releases/builds/syncs/removes through
  definition recipes on `GenericNativeInstaller`; llama/ik publication is a
  recipe step; unsloth published in-handler already; one shared removal flow;
  launch constraints are engine policy data (R66).
- What plan-Phase 7 covers (plan line 136): discovery, models, voices,
  routing/transforms, model profiles, activation capability review, and all
  task/request surfaces. Inventory (audited this pass):
  - engine services, ~4.2k lines in `backend/engines/audio_cpp/`:
    `contracts.py` 412, `spec_fields.py` 516, `voices.py` 336, `activation.py`
    299, `runtime.py` 313, `model_managers.py` 198, `discovery.py` 149,
    `builtin.py` 136, `artifact.py` 149, `inspect.py` 48, `prebuilt.py` 398
    (runtime-derived or engine algorithms — NOT static data);
    `build_options.py` 235 and `tracking.py` 168 are the only STATIC catalogs;
  - model-management surface: `backend/audio/` ~2.5k lines (10 modules:
    model_config 556, voice_presets 510, task_profiles 375, format_convert 340,
    community_voices 288, request_policy 192, request_defaults(+validation) 191,
    transcription_defaults 51), `model_catalog/audio_cpp_provider.py` 531,
    `services/audio_model_installer.py` 1440, plus audio branches in
    `routes/models.py` and `model_catalog/service.py`;
  - literal map (parity inventory): 62 audio_cpp sites inside the engine home,
    47 in its own route (legitimate), and the rest spread across frontend views
    and shared backend consumers — the plan's "remaining consumers" (Phase 8),
    not installer work.
- Slice plan (each its own verified slice; no `audio: true` flag anywhere):
  - S1 STATIC DATA: `build_options` becomes a schema-v2 definition block
    (`build_options`: key/flag/type/default/label/help/group, exported from an
    audio-owned data module), `GET /api/audio-cpp/build-options` and the cmake
    argv read it. Gate: payload byte-identical, build argv goldens, ratchet
    shrink.
  - S2 DECLARED PROBE SURFACE: `inspect.argv`/artifact path contract
    (`inspect.py`, `artifact.py`) declared by the definition instead of
    hardcoded audio constants; gate: inspect argv/env goldens + package
    discovery parity.
  - S3 TYPED EXTENSIONS: contract/voices/discovery/activation surfaces become
    definition-declared typed extension routes (plan line 122's model), with the
    existing `/contract`, `/community-voices*`, `/status` payloads pinned
    byte-for-byte. These stay RUNTIME-derived (scanned from the installed
    binary) — the definition declares the surface, never the discovered data.
  - S4 MODEL-MANAGEMENT SURFACES: catalog provider / audio model install /
    task+request defaults declare their engine surface in the definition
    (`model_sources`/`discovery` blocks) while preserving every audio contract
    test and the audio model config UI.
- Explicit exceptions carried, not silently dropped: native GGUF launch stays
  legacy (R23); native snapshot-backed retries stay out of scope (R57).
- Why not one shot: the surface is ~8k lines across 25+ modules, mostly
  runtime-derived; blind data-ification would regress discovered contracts. The
  ordered plan keeps every slice behavior-pinned before the next.
**R68 - Plan-Phase-7 S1: audio build options are engine DATA (2026-11-03,
DONE)**. First slice of the audio services/contracts plan phase.
- `EngineSpec.build_options` (audio.cpp) now carries the catalog moved
  VERBATIM from the code module: 23 options (key/field/kind/default/label/desc/
  category/cmake/requires/enum_values/special), 6 UI categories (one
  conditional on `cuda`), the four build types plus `build_type_default`
  (RelWithDebInfo), and `primary_keys`. Excluded from the descriptor API like
  lifecycle/launch_policy/runtime_profile.
- Schema v2 `build_options` block: validated structurally (unique keys/fields,
  category references, conditional `requires` incl. the `model_set_custom` /
  `cuda_or_hip` pseudos, enum defaults inside enum_values, per-kind default
  types) and round-tripped by `to_spec()`; the exporter writes it into the
  generated definition and verifies definition-vs-registry equality.
- `backend/engines/audio_cpp/build_options.py` (279 → 217 lines) is now
  BEHAVIOR ONLY: it derives its option dataclasses/categories from the spec on
  demand. The derived backend discovers the `special: backend` option instead of
  hardcoding the `cuda` key, so a definition-only catalog works unchanged.
  `GET /api/audio-cpp/build-options`, coercion, defaults and cmake argv are
  byte-identical (the pre-existing API/catalog/manager tests pass untouched).
- Pins (4 new in `test_audio_build_options.py`): the definition carries exactly
  the spec catalog and the payload excludes it; the module renders an ARBITRARY
  spec (fake 2-option catalog) proving no private table remains; a validated
  definition round-trip; and seven malformed-catalog rejections.
- Verification (lanes only): audio+definitions subset **138 passed**; engine
  slice **456 passed / ~39s**; FULL parallel **2024 passed, 8 skipped**; ratchet
  no growth; parity inventory current (exit 0); defcheck 12 engines; ruff
  curated 90 files.
- Next in this plan phase: S2 declared probe surface (`inspect.py` argv/env +
  `artifact.py` path contract), S3 typed extension routes (contract/voices/
  discovery/activation payloads pinned), S4 model-management surfaces.
**R69 - Plan-Phase-7 S2/S3/S4: the audio surface is DECLARED (2026-11-03,
DONE)**. The remaining audio services/contracts slices landed as engine-data
declarations with runtime reads and gates; every preserved surface stayed
byte-identical.
- S2 protocol contract: `EngineSpec.protocol` + schema-v2 `protocol` block
  (`probe`: model/family/spec-override/load-option/inspect/json flags;
  `artifact`: builtin family, root GGUF name, GGUF suffix, builtin-id regex).
  `inspect.py` builds argv from the declared flags and `artifact.py` resolves
  paths from the declared shape (`BUILTIN_AUDIO_FAMILY` is the declared value).
  Validation rejects non-flag probes, bad suffixes, invalid regexes, unknown
  keys.
- S3 declared extensions: `EngineSpec.extensions` + schema-v2 `extensions`
  block declaring the engine-owned endpoints beyond the lifecycle verbs
  (`/contract`, `/status`, `/build-options`, `/build-settings`,
  `/community-voices`, `/community-voices/install`, methods included). The gate
  test asserts every declared path/method pair is actually served by the audio
  router — the surface is explicit instead of implicit, with no route changes.
- S4 request-default vocabularies: `EngineSpec.request_defaults` + schema-v2
  `request_defaults` block (per-surface `keys`/`bool_keys`/`int_keys`/
  `float_keys`) consumed by the task, transcription and speech normalizers via a
  new engine-home accessor (`audio_cpp/request_vocab.py`). Those modules keep
  only the coercion behavior; the key vocabularies are data. Coercion parity is
  pinned by the pre-existing request-default/voice-preset/model-config suites.
- What deliberately stays code (not config): the model-management ALGORITHMS —
  `model_catalog/audio_cpp_provider.py`, `services/audio_model_installer.py`,
  `backend/audio/model_config.py` and the runtime-derived contract/voices/
  discovery/activation services. They compute from the installed binary and
  model_specs; there is no static surface to declare beyond what is now
  declared, and the plan's phase 8 covers the remaining consumers (catalog/
  model compatibility, frontend keyed state).
- Pins: `test_audio_protocol_contract.py` (10) — protocol/extension/
  request-default data in spec+definition, descriptor-payload exclusion, the
  module-renders-an-arbitrary-spec proofs (probe flags, artifact shape,
  request vocab), the served-extension gate, round-trip, and nine rejection
  cases (unknown keys/sections, non-flag probe, bad suffix, bad regex, bad
  path, lower-case method, duplicate path, duplicate/typed key collisions).
- Verification (lanes only): audio+definitions+request subset **185 passed**;
  surface pins **10 passed**; engine slice **466 passed / ~40s**; FULL parallel
  **2034 passed, 8 skipped**; ratchet no growth; parity inventory current
  (exit 0); defcheck 12 engines; ruff curated 97 files.
- Phase status (SUPERSEDED by R70 — the "complete" claim below is withdrawn):
  the plan's Phase 7 (audio services/contracts) is now COMPLETE in
  the only sense this codebase can express: every static surface (build options,
  probe flags, artifact contract, extension routes, request vocabularies) is
  engine data, exported and verified, and every runtime-derived surface
  (contract, voices, discovery, activation, model management) is preserved,
  declared where declarable, and covered by its existing suites. Explicit
  exceptions remain: native GGUF launch (R23), native snapshot retries (R57).
**R70 - Review corrections: the R69 "complete" claim is WITHDRAWN
(2026-11-03, DONE)**. Three findings were valid; two are fixed in code, the
third is a documentation correction plus an explicit remaining-work list.
- FIX 1 (vocabulary partly ignored): `transcription_defaults.py` hardcoded the
  boolean `stream` while reading only the numeric vocabularies — a declared
  boolean (`return_text`) was dropped and an undeclared one accepted. It now
  honours the declared `bool_keys` generically (coerced whenever present, in
  declaration order); a pin drives a fake vocabulary where `return_text` is
  accepted and `stream` is dropped, and the real data pin asserts
  `transcription.bool_keys == ["stream"]`.
- FIX 2 (extension declarations had no consumer): `backend/engines/definitions/
  extensions.py` now owns the single implementation
  (`assert_extensions_served`) and the **audio router calls it at import time**,
  so a drifted declaration refuses import/startup instead of being documentation
  nobody reads; the gate test uses the same helper and pins two drift cases
  (missing route, missing method). Honest limit, now documented in code and
  here: this ENFORCES the declaration — it does not replace the bespoke
  handlers; the declaration is the contract for a future generic mount, and
  no claim of handler replacement is made.
- FIX 3 (completion overstated): R69's "Phase 7 COMPLETE" is WITHDRAWN. What is
  actually complete is the installer host (R63/R65) plus the STATIC surface
  declarations (R68/R69, corrected by this entry). Still bespoke, by file:
  - audio discovery / service selection: `helpers` in
    `audio_cpp/discovery.py`, the audio provider selection in
    `model_catalog/service.py`, `backend/audio/model_config.py`, and the
    model-management algorithms in `services/audio_model_installer.py`;
  - `${studio_audio_config}` materialization: `audio_cpp/runtime.py` builds the
    launch manifest (macro value = sidecar path) and
    `proxy/launch_spec.py` resolves it — native GGUF/audio launch stays legacy
    per R23, so this is runtime-adapter code, not a declaration;
  - runtime-derived contract/voices/activation services (their endpoints are
    now declared, their payloads are discovered from the installed binary).
  These are the real next slices (plan phase 7 remainder / phase 8), and the
  campaign records them instead of claiming completion.
- Verification (lanes only): corrected surface pins + request/voice suites
  **59 passed**; engine slice **467 passed / ~39s**; FULL parallel **2035
  passed, 8 skipped**; ratchet no growth; parity inventory current (exit 0);
  defcheck 12 engines; ruff curated 98 files.
**R71 - Named remainders closed where they were really data; PLAN STATUS
(2026-11-03, DONE for this slice)**. Two R70 remainders turned out to be data
after all and are now declared + consumed; the rest are recorded honestly.
- Catalog surface: `EngineSpec.catalog` + schema-v2 `catalog` block declares the
  catalog providers per engine (audio -> `[audio_cpp]`, the eight others ->
  `[huggingface]`, matching the previous id-branching exactly).
  `model_catalog/service.py::_provider_ids` now reads the declaration instead of
  `GGUF_ENGINE_IDS`/`HF_SNAPSHOT_ENGINE_IDS`/engine equality; explicit provider
  filters still win and unknown engines still get both catalogs.
- Launch macro: `protocol.launch` (`config_flag`, `config_macro`) declares
  audio's `${studio_audio_config}` manifest macro; `audio_cpp/runtime.py` builds
  the argv/macros from it (`launch_macro()`), so the macro name and flag are no
  longer literals in the adapter.
- Pins added (3): the declared launch macro + a fake-spec derivation, the
  per-engine catalog providers (with descriptor-payload exclusion), and the
  catalog service reading a fake declaration. The extension enforcement from R70
  stays: the audio router imports `assert_extensions_served` and refuses drift.
- PLAN STATUS (docs/config-driven-engine-refactoring-plan.md), stated plainly:
  - phases 1-2 (inventory, schema/shadow) DONE;
  - phase 3 (executor/lifecycle) DONE for all Python engines;
  - phase 4 (native builds/release) DONE (llama/ik source, unsloth release,
    audio prebuilt/source on the native host);
  - phase 5 (specialized recipes) DONE (1Cat, SGLang V100);
  - phase 6 (launch/discovery profiles) DONE for the five pip engines; native
    GGUF launch remains legacy BY DECISION (R23) — an explicit exception, not an
    oversight;
  - phase 7 (audio services/contracts) installer host DONE + static surfaces
    declared; the RUNTIME payload services (contract composition, voices,
    activation deltas) and the model-management algorithms remain code; their
    endpoints are declared and enforced;
  - phase 8 (API/frontend + remaining consumers) NOT DONE: only the catalog
    provider selection moved. Still open: frontend keyed state/descriptor-driven
    views and forms, model compatibility, source identity, validation, recovery,
    orphan roots, diagnostics, startup wiring;
  - phase 9 (config-only onboarding + legacy removal) PARTIAL: the fixture
    engines and the definition journey prove onboarding; the literal ratchet is
    the architecture check; the LEGACY IMPLEMENTATIONS (per-engine managers/
    routes) are NOT removed yet.
  The plan is therefore NOT finished, and this campaign does not claim that.
- Verification (lanes only): pins **14 passed**; engine slice **470 passed /
  ~39s**; FULL parallel **2038 passed, 8 skipped**; ratchet no growth; parity
  current (exit 0); defcheck 12 engines; ruff curated 100 files.
**R72 - Phase-8 source identity: upstream URL is engine data (2026-11-03,
DONE)**. `backend/repo_identity.py`'s `CANONICAL_REPOSITORY_URLS` (nine
engine-id entries outside the engine home) is DELETED; each engine's spec
declares `upstream.repository_url`, exported into the generated definition, and
fork detection reads it from the spec.
- `EngineSpec.upstream` + schema-v2 `upstream` block (`repository_url`: remote
  URL, validated; unknown keys/non-URLs rejected); excluded from the descriptor
  API. `canonical_repository_url(engine_id)` is the single reader;
  `is_github_fork_for_engine` / `source_build_type_labels_for_engine` use it, so
  a YAML-only engine can declare its upstream and get correct fork labels.
  Engines without a declaration are simply "not a fork" (previous map-miss
  behavior).
- Pins (6 new, `test_engine_upstream_declaration.py`): the nine declared URLs
  (with descriptor exclusion), the exported definitions, fork detection through
  a FAKE spec (fork/not-fork/labels), the no-upstream engine, the definition
  round-trip, and three rejections (non-URL, unknown key, schema v1).
  `test_source_fork_labeling.py` / `test_unsloth_llama_integration.py` now read
  the declared value instead of the deleted map.
- Verification (lanes only): fork/definition subset **173 + 6 passed**; engine
  slice **476 passed / ~40s**; FULL parallel **2044 passed, 8 skipped**; ratchet
  no growth (nine literals retired); parity inventory current (exit 0); defcheck
  12 engines; ruff curated 102 files.
- Plan status after this step: phase 8's "source identity" item is DONE; the
  rest of phase 8 (frontend keyed state/descriptors/forms, model compatibility,
  validation, recovery, orphan roots, diagnostics, startup wiring) and phase 9's
  legacy-implementation removal remain OPEN, as recorded in R71.
**R73 - Phase-8 settings/validation: venv install defaults are engine data
(2026-11-03, DONE)**. `backend/venv_install_settings.py`'s five-entry
`_ENGINE_DEFAULTS` map (lmdeploy/1cat_vllm/sglang/sglang_v100/vllm) is GONE.
- `EngineSpec.settings_defaults` + the existing schema-v2 `settings_defaults`
  block now carry the frozen `source_repo`/`source_branch`/`pip_version` (or
  `release_version`) defaults; the exporter writes them into the generated
  definitions and verifies definition-vs-registry equality. Excluded from the
  descriptor payload; `to_spec()` passes them through (that round-trip was the
  one thing the first run missed).
- `venv_install_settings.default_install_settings` reads the spec (lazy import,
  so tests can drive a fake spec) and falls back to authoritative registrations;
  `assert_venv_engine` accepts a declared spec; `coerce_install_settings` and the
  settings HTTP surface are unchanged. `_ENGINE_DEFAULTS` survives as the
  authoritative-registration store with the same name, so startup wiring and the
  six tests that patch it keep working (they now isolate registrations, not
  generated data).
- Pins (5 new, `test_engine_settings_defaults_declaration.py`): the historical
  per-engine defaults + descriptor exclusion, the exported definitions, a
  fake-spec read proving spec-driven behavior, the unknown-engine ValueError,
  and authoritative registration still winning (cleaned up in a finally).
- Verification (lanes only): affected suites **84 + 56 passed**; engine slice
  **481 passed / ~40s**; FULL parallel **2049 passed, 8 skipped**; ratchet no
  growth (five literals retired); parity inventory current (exit 0); defcheck 12
  engines; ruff curated 103 files.
- Plan status: phase 8's "settings/validation" item is DONE. Next queue item:
  action-recovery keys + orphan-root discovery from declarations.
**R74 - Phase-8 recovery/orphan roots: install-kind capabilities are declared
(2026-11-03, DONE)**. Two engine-id tuples in shared code became data-derived.
- `engine_ids_with_install_kind(kind)` on the registry reads
  `EngineSpec.install_kind`; `backend/engines/lifecycle.py`'s
  `_PYTHON_VENV_ENGINES` (venv retry rule + orphan-discovery loop) is replaced by
  `python_venv_engine_ids()`, and `routes/llama_versions.py`'s
  `_CMAKE_BUILD_ENGINES` (cmake-editability + build-config endpoint guard) by
  `_cmake_build_engines()`. The generated families resolve to exactly the
  historical sets; a YAML engine declaring `python_venv`/`cmake` now gets the
  same behavior with no code edit (proved by fake-spec pins).
- `action_recovery.py` needed no change: its keys (`engine:<id>`,
  `cuda:toolkit`, `proxy:runtime`) were already engine-id-agnostic strings, which
  the pins re-confirm rather than pretend to migrate.
- Pins (4 new, `test_engine_install_kind_capabilities.py`): the historical
  families (filtered to GENERATED_ENGINE_IDS, since authoritative engines
  legitimately add themselves at startup — the first lane run caught that
  order-sensitivity and the pin now tolerates it), a declared venv engine
  becoming retryable, a declared cmake engine becoming cmake-editable, and a
  source-level check that the old tuples do not return.
- Verification (lanes only): lifecycle/cmake subset **108 passed**; pins **4
  passed**; engine slice **485 passed / ~41s**; FULL parallel **2053 passed, 8
  skipped**; ratchet no growth (two id sets retired); parity inventory current
  (exit 0); defcheck 12 engines; ruff curated 104 files.
- Plan status: phase 8's "recovery/orphan roots" item is DONE (recovery keys
  were already declarations; orphan discovery now follows install_kind). Next
  queue item: diagnostics/startup wiring leftovers.
**R75 - Phase-8 diagnostics/startup wiring: data directories are declared
(2026-11-03, DONE)**. Audited diagnostics first: `backend/diagnostics.py` and
`routes/status.py`'s bundle have NO engine-id branches (allowlisted
proxy/runtime fields + persistence/settings export), so nothing to migrate there
— recorded rather than invented.
- The real leftover was startup wiring: `backend/main.py::
  ensure_data_directories` hardcoded every engine folder (`llama-cpp`,
  `audio-cpp`, `models/audio-cpp`, `1cat-vllm`, `sglang-v100`, ...). The folders
  are now `EngineSpec.data_dirs` + schema-v2 `data_dirs` (relative paths only:
  absolute paths, `..`, and duplicates are rejected), exported into the
  generated definitions and verified. Startup creates the shared folders
  (`config`, `logs`, `temp`) plus the union of declared engine dirs, and the
  pass runs again (idempotently) after YAML lifecycle registration so an
  onboarded engine's folders exist too.
- Pins (6 new, `test_engine_data_dirs_declaration.py`): the per-engine
  directories + descriptor exclusion, the exported definitions, startup creation
  under a temp data root, a FAKE declared engine's dirs appearing (proves the
  list is spec-driven), the definition round-trip, and four rejections
  (absolute path, `..`, duplicate, schema v1).
- Verification (lanes only): pins **6 passed**; engine slice **491 passed /
  ~42s**; FULL parallel **2059 passed, 8 skipped**; ratchet no growth (nine
  startup entries retired); parity inventory current (exit 0); defcheck 12
  engines; ruff curated 105 files.
- Plan status: phase 8's "diagnostics/startup wiring" item is DONE (diagnostics
  needed no change; startup directory wiring is declared). Next queue item:
  model compatibility from descriptors/spec.
**R76 - Phase-8 model compatibility: declared, not branched (2026-11-03,
DONE)**. Two engine-id branches in the model layer became engine data.
- `EngineSpec.model_compat` + schema-v2 `model_compat` with two declared
  capabilities: `default_for_formats` (which engine is the default for an
  artifact format) and `prepared_bundle_formats` (which formats are stored as a
  prepared bundle). Exported into the generated definitions and verified;
  excluded from the descriptor payload.
- `models/config.default_engine_for_format` now scans the declarations
  (lmdeploy claims `safetensors`) and falls back to `DEFAULT_ENGINE`; the
  `safetensors -> lmdeploy` literal branch is gone.
- `models/schema._default_package_kind` reads `prepared_bundle_formats`
  (audio claims `mixed`/`original`) instead of branching on
  `engine == "audio_cpp"`; the "record has a family" rule is preserved verbatim,
  as are the generic gguf/safetensors rules.
- Pins (8 new, `test_engine_model_compat_declaration.py`): the declared values +
  descriptor exclusion, the exported definitions, the format defaults
  (case-insensitive; gguf/None/empty -> DEFAULT_ENGINE), a FAKE engine claiming a
  format, the package-kind matrix (audio mixed/original, audio gguf, family
  record, llama_cpp mixed -> unknown), a FAKE engine declaring bundle formats,
  the definition round-trip, and four rejections (unknown key, empty list,
  duplicate, schema v1).
- Verification (lanes only): model/definition subset **130 passed**; pins **8
  passed**; engine slice **499 passed / ~43s**; FULL parallel **2067 passed, 8
  skipped**; ratchet no growth (id branches retired); parity inventory current
  (exit 0); defcheck 12 engines; ruff curated 106 files.
- Plan status: phase 8's "model compatibility" item is DONE. Next queue item:
  frontend descriptor-driven views and keyed stores.
**R77 - Phase-8 frontend, first vertical: readiness filters are
descriptor-driven (2026-11-03, DONE)**. `frontend/src/composables/
engineReadiness.js::engineMatchesFilters` special-cased the `audio_cpp` engine id
in three places (audio-task match, text exclusion, CPU hardware filter).
- Backend: `audio_cpp` now DECLARES `launch_policy.cpu_supported: true`, and
  `EngineSpec.to_api_dict` exposes `cpu_supported` as an explicit descriptor
  field (the policy block itself still stays out of the payload). The default is
  `false`, which keeps every other engine's filter behavior byte-identical: gguf
  engines pass the CPU filter via `artifact_formats`, the five pip engines stay
  excluded, and a YAML engine that declares neither gguf nor cpu support stays
  excluded exactly as before.
- Frontend: the audio filter matches by declared `tasks ∩ AUDIO_TASKS`, the text
  filter relies on the task set (audio declares no text-generation task), and
  the CPU filter uses `descriptor.cpu_supported`. No engine-id special case
  remains in those three branches. The `sm70` branch still matches
  `sglang_v100`/`1cat_vllm` ids because the payload carries no GPU-architecture
  fact yet — recorded as the remaining piece of this vertical, not silently
  dropped.
- The R66 pin that asserted "audio has no launch policy" is updated to the new
  declared intent (documented data change, not a regression).
- Pins: frontend `engineReadiness.test.js` gained a descriptor-driven matrix
  (audio via tasks, text exclusion, cpu via cpu_supported, YAML engine without
  declarations still excluded, custom audio-family engine matched by tasks);
  backend `test_engine_descriptor_cpu_supported.py` (4) pins the payload field
  per engine, the policy block's continued exclusion, and the audio
  definition round-trip.
- Verification (lanes only): frontend `npx vitest run` on the readiness +
  EnginesView.tasks + useAudioContract files -> **26 passed (3 files)**; backend
  descriptor/definition subset **112 passed**; engine slice **503 passed /
  ~45s**; FULL parallel **2071 passed, 8 skipped**; ratchet no growth; parity
  inventory current (exit 0); defcheck 12 engines; ruff curated 107 files.
  Container note: vitest needed `frontend/node_modules/.vite-temp` removed once
  (root-owned by an earlier root run) before appuser could rebuild it.
- Plan status: phase 8's frontend item is STARTED — readiness filters done;
  remaining frontend work: keyed stores for the nine local engines,
  descriptor-driven build-settings panels, and the audio workspace branches.
**R78 - Phase-8 frontend: readiness vertical COMPLETED (2026-11-03)**. The last
engine-id branch in the readiness filters (the `sm70` hardware filter) is gone.
- `launch_policy` gained a declared `gpu_archs` list (token-validated,
  tuple-normalized); `sglang_v100` and `1cat_vllm` declare `["sm70"]`, and
  `to_api_dict` exposes `gpu_archs` alongside `cpu_supported` (sorted, empty for
  engines with no declaration). `engineMatchesFilters` now reads
  `descriptor.gpu_archs` — no id comparisons remain in the function.
- The reference definitions needed updating to stay registry-equivalent: the
  worst-case `reference/sglang_v100.yaml` now declares its `launch_policy`
  (`gpu_archs: [sm70]` + the three GPU-only env invariants), which also makes the
  hand-authored YAML express the capability the frontend consumes.
- Behavior parity: the payload for the nine is unchanged except the two new
  derived fields (cpu_supported/gpu_archs); the sm70 filter still matches exactly
  sglang_v100/1cat_vllm on the real payload, and an engine with no declaration
  matches nothing.
- Pins: frontend `engineReadiness.test.js` gained an sm70 matrix (declared
  engines match, undeclared don't, no id fallback); backend
  `test_engine_descriptor_cpu_supported.py` pins the payload `gpu_archs` per
  engine + the reference definition; `test_engine_launch_policy` expectations
  and the reference YAML were updated for the new declared data.
- Verification (lanes only): frontend `npx vitest run` (readiness +
  EnginesView.tasks) -> **23 passed (2 files)**; backend launch/reference subset
  **48 passed**; engine slice **504 passed / ~45s**; FULL parallel **2072 passed,
  8 skipped**; ratchet no growth; parity inventory current (exit 0); defcheck 12
  engines; ruff curated 107 files.
- Plan status: the readiness-filters vertical of phase 8's frontend item is
  COMPLETE (no engine-id branches left in `engineMatchesFilters`). Remaining
  frontend: keyed stores for the nine local engines, descriptor-driven
  build-settings panels, audio workspace branches.
**R79 - Phase-8 frontend, second vertical: task-cancel routing is
descriptor-driven (2026-11-03, DONE)**. `useTaskCancelEndpoint.js` carried a
literal `audio_cpp: '/api/audio-cpp/cancel'` map entry next to the already
derived native/venv maps.
- Frontend: the audio descriptor now declares `cancelPath`, and the registry
  derives `DECLARED_CANCEL_ENDPOINTS_BY_ENGINE` from any descriptor that has one;
  the composable spreads that map. Adding such an engine is now ONE descriptor
  field, not a composable edit.
- Backend: audio's declared extension surface now includes `cancel`
  (`POST /cancel`) — the router import gate (`assert_extensions_served`) proves
  the declaration matches the served route, and the generated definition carries
  it.
- Pins: frontend `useTaskCancelEndpoint.test.js` asserts the derived map equals
  `{audio_cpp: '/api/audio-cpp/cancel'}`, that an audio build task routes there,
  and that unknown engines/managers fall back to the type-level legacy endpoint;
  backend `test_audio_protocol_contract` EXPECTED_EXTENSIONS gained `cancel`.
- Verification (lanes only): frontend `npx vitest run` (cancel + readiness +
  store) -> **44 passed (3 files)**; backend protocol pins **14 passed**; engine
  slice **504 passed / ~44s**; FULL parallel **2072 passed, 8 skipped**; ratchet
  no growth; parity inventory current (exit 0); defcheck 12 engines; ruff
  curated 107 files.
- Plan status: the task-cancel vertical of phase 8's frontend item is DONE.
  Remaining frontend: keyed stores for the nine local engines,
  descriptor-driven build-settings panels, audio workspace branches.
**R80 - Phase-8 frontend, third vertical: the build-settings fallback is
generated, not hand-copied (2026-11-03, DONE)**. `EnginesView.vue` carried a
28-key `FALLBACK_BUILD_DEFAULTS` literal explicitly "kept in sync with backend
defaults" plus a hardcoded build-type option list — a drift risk with no gate.
- New `scripts/export-build-defaults.py` writes
  `frontend/src/engines/build-defaults.generated.json` from the authoritative
  backend catalog (`llama_cpp.build_options`); the Vue panel imports it through
  `frontend/src/engines/buildDefaults.js` and the 28-key literal is deleted.
  Build-type order is deliberately default-first (`Release, Debug,
  RelWithDebInfo, MinSizeRel`), preserving the panel's historic UI order even
  though the backend catalog lists Debug first.
- Drift gates: `npm run build-defaults:check` (the script's `--check`) is part
  of the `lint` chain, and a backend pin asserts the checked-in JSON equals
  `default_build_settings()` and the script's own render — so a backend default
  change fails a lane until the JSON is regenerated.
- Pins: backend `test_frontend_build_defaults_current.py` (3: values, default-
  first order, script render) and frontend `buildDefaults.test.js` (values,
  frozen, order).
- Verification (lanes only): frontend `npx vitest run` (buildDefaults +
  EnginesView.buildConfig) -> **8 passed (2 files)**; backend pins **3 passed**;
  engine slice **507 passed / ~45s**; FULL parallel **2075 passed, 8 skipped**;
  ratchet no growth; parity inventory current (exit 0); defcheck 12 engines;
  build-defaults check current; ruff curated 109 files.
- Plan status: the build-settings fallback vertical is DONE (the panel still
  renders its catalog from `GET /build-options`; the fallback is generated).
  Remaining frontend: keyed stores for the nine local engines, the audio
  workspace branches, and rendering the catalog schema from the descriptor
  rather than the API response.
**R81 - Phase-8 frontend: per-engine status is ONE keyed map (2026-11-03,
DONE)**. The store kept six named status refs (`lmdeployStatus`,
`onecatVllmStatus`, `sglangStatus`, `sglangV100Status`, `vllmStatus`,
`audioCppStatus`), selected by a `statusRef` descriptor field, with **no
consumer outside the store** (views and tests only touched them through the
dynamic field).
- `engineStatusById` (one ref keyed by engine id) replaces them: the generic
  `venvFetchStatus` writes `engineStatusById[engine]`, `fetchAudioCppStatus`
  writes the `audio_cpp` entry, and the descriptors drop their `statusRef`
  field (the header doc now names `nullSafeStatus` as the only status
  divergence). `nullSafeStatus` behavior is unchanged and pinned: the SGLang
  family coerces a nullish payload to `{}`, lmdeploy/1Cat keep the raw payload.
- One inventory regeneration was required and is intentional: the store's map
  key adds one `audio_cpp` occurrence (a data key, not dispatch), and the new
  pin lives under `frontend/src/stores/tests/` so its fixture ids are not
  counted. The ratchet reports no growth after regeneration.
- Pins: new `frontend/src/stores/tests/engineStatusMap.test.js` (per-engine map
  writes for venv + audio, the nullSafeStatus split, no `statusRef` on any
  descriptor); the venv/store suites were updated from the named refs to the
  keyed map (that update IS the behavior contract).
- Verification (lanes only): frontend `npx vitest run` on the status map +
  venv/store/EnginesView suites -> **90 passed (5 files)**, prettier clean on
  the touched files; engine slice **507 passed / ~46s**; FULL parallel **2075
  passed, 8 skipped**; parity inventory regenerated and current; ratchet no
  growth; defcheck 12 engines; ruff curated 109 files.
- Plan status: the keyed-status vertical of phase 8's frontend item is DONE
  (one keyed map, no per-engine named status refs). Remaining frontend: the
  nine local engine VERSIONS refs (same pattern, larger surface), the audio
  workspace branches, and descriptor-driven catalog rendering.
**R82 - Phase-8 frontend: generic keyed reads + R81 fallout fixed (2026-11-03,
DONE)**. Two things landed: the generic read surface, and a real bug the
previous round left behind.
- The store gains `versionsForEngine(engineId)` / `statusForEngine(engineId)`.
  `EnginesView`'s nine-entry `engineVersionLists` map is GONE — it resolves the
  list through the descriptor's declared `versionsKey`, which is exactly the
  kind of lookup the map duplicated. `SglangEnginePanel`'s per-engine
  `isV100/isVllm` ternaries for versions AND status are gone too.
- FALLOUT FIX (found by running the whole frontend suite for the first time
  this session): R81 removed the named status refs, but `SglangEnginePanel`
  still read `store.sglangV100Status` / `store.vllmStatus` / `store.sglangStatus`
  — undefined at runtime. It now reads `statusForEngine(props.engineId)`; the
  panel's test stub was updated to the accessor surface.
- Pins: new `frontend/src/stores/tests/engineKeyedReads.test.js` (versions by id
  incl. unknown -> [], status by id with the `{}` fallback); the server-backed
  store tests assert `statusRef` is gone and use `statusForEngine`; the venv
  suite already covers the keyed status map (R81).
- Verification (lanes only): **FULL frontend suite 621 passed (67 files)**;
  engine slice **507 passed / ~45s**; FULL parallel **2075 passed, 8 skipped**;
  parity inventory current (exit 0); ratchet no growth; defcheck 12 engines;
  build-defaults current; ruff curated 109 files.
- Plan status: the keyed-reads vertical of phase 8's frontend item is DONE. The
  nine engines' versions refs themselves remain (56 consumer sites still read
  them; converting the refs, not the lookups, is the remaining work), plus the
  audio workspace branches and descriptor-driven catalog rendering.
**R83 - Phase-9 first legacy removal: dead pre-host audio helpers deleted
(2026-11-03, DONE)**. Queue item (6). The versions-refs conversion was
deliberately deferred (56 consumer sites, several tests assign the refs) so the
ready legacy-removal slice could land first; that deferral is recorded, not
silent.
- AUDIT METHOD: an AST scan reported, per module-level symbol, whether it is
  referenced elsewhere in its own module and in production/test code. In
  `audio_cpp/manager.py` four helpers had ZERO production and ZERO in-module
  references: `_safe_slug`, `_valid_repository_url`,
  `AudioCppManager._binary_candidates`, `AudioCppManager._find_binary` — the
  pre-host leftovers (the host materializes binaries itself). `installer.py` and
  `prebuilt.py` were audited the same way and have no dead symbols.
- Deleted the four helpers plus the imports only they needed
  (`native_support.native_binary_candidates/native_find_binary`, `re`). The
  module keeps exactly what live code calls: `AudioCppBuildConfig`, the cmake
  arg composition used by `_prepare_configure`, the sync-target resolution the
  installer's handlers use, and `sync_git_checkout`/`get_audio_cpp_manager`.
- Pin: `test_audio_cpp_manager.py::test_pre_host_dead_helpers_stay_removed`
  asserts the four names stay out of the module.
- Verification (lanes only): manager suite **13 passed**; engine slice **507
  passed / ~45s**; FULL parallel **2076 passed, 8 skipped**; parity inventory
  current (exit 0); ratchet no growth; defcheck 12 engines; ruff curated 109
  files.
- Plan status: phase 9's dead-code removal is STARTED and this slice is DONE.
  Remaining legacy removal: the per-engine route modules for the venv engines
  are LIVE compatibility delegates (they cannot be deleted until the frontend/API
  consumer migration lands), and the nine frontend versions refs remain.
**R84 - Phase-8 follow-up: versions refs are descriptor-mapped (2026-11-03,
DONE)**. The deferred half of R82.
- The store's nine-entry engine-id → ref table (`VERSION_ROWS_BY_ENGINE`) is
  GONE. `VERSION_REFS_BY_KEY` is keyed by the descriptor's `versionsKey` and
  `versionRefFor(engineId)` is the only id → ref resolution;
  `fetchLlamaVersions` now iterates the descriptors (partition rule unchanged)
  and `versionsForEngine` reads through the same mapping.
- The descriptor walk exposed a real gap: only lmdeploy and 1cat_vllm declared
  `versionsKey` — the other seven local descriptors did not, which the
  partitioning test caught immediately (llama rows stopped landing). All nine
  now declare it, so "which named ref holds this engine's rows" is descriptor
  data rather than a store table.
- Pin: `engineKeyedReads.test.js` walks the real descriptors and asserts every
  local engine declares `versionsKey`, that the named ref exists, and that
  `versionsForEngine(id)` agrees with it after an assignment (no id table can
  sneak back without failing this).
- Verification (lanes only): **FULL frontend suite 622 passed (67 files)**;
  engine slice **507 passed / ~45s**; FULL parallel **2076 passed, 8 skipped**;
  parity inventory regenerated and current (exit 0); ratchet no growth (the
  store's engine-id table retired); defcheck 12 engines; build-defaults current;
  ruff curated 109 files.
- Plan status: the keyed-versions vertical of phase 8's frontend item is DONE at
  the MAPPING level (named refs remain as compatibility aliases for the ~56
  consumer sites). Remaining frontend: converting the refs themselves, the audio
  workspace branches, descriptor-driven catalog rendering.
**R85 - Phase-9 architecture check tightened: frozen aggregate caps
(2026-11-04, DONE)**. The literal ratchet compared the live tree to
`docs/engine-parity-inventory.json`, which is REGENERATED whenever growth is
"intended" — so per-file growth was one command away from passing silently. The
plan asks for an architecture check against engine-ID dispatch outside
definitions; this closes that hole.
- `scripts/engine-literals-ratchet.py` now ALSO enforces `AREA_CAPS`: a frozen
  aggregate literal budget for 17 areas (e.g. `backend/proxy/` 76,
  `backend/routes/` 169, `frontend/src/stores/` 323, `scripts/` 108; 1375
  literals total). Regenerating the inventory cannot raise a cap — growing an
  area requires editing the table (an explicit reviewed diff), and literals in a
  NEW area fail outright. Shrinking is always allowed.
- `area_for` groups paths at directory level for both languages (backend module
  dirs, `frontend/src/<dir>/`, `frontend/<dir>/`, `scripts/`), and
  `compare_areas` reports growth + unknown areas.
- Pins (3 new in `test_engine_literals_ratchet.py`): the live tree is within
  every cap, the grouping rule (including top-level backend files), and a
  synthetic case where growth fails EVEN IF the inventory would be regenerated.
- Verification (lanes only): ratchet **exit 0** with the new aggregate check;
  engine slice **510 passed / ~50s**; FULL parallel **2079 passed, 8 skipped**;
  parity inventory current (exit 0); defcheck 12 engines; build-defaults current;
  ruff curated 109 files.
- Plan status: phase 9's architecture-check tightening is DONE. Remaining plan
  work (unchanged): the frontend refs conversion / audio workspace branches /
  descriptor-driven catalog rendering, and legacy route-module removal that
  depends on them.
**R86 - Goal-round wrap-up: final verification + honest remaining work
(2026-11-04)**. Ten autonomous rounds advanced the plan; this entry records the
FINAL verified state and what is deliberately NOT done. The plan is NOT complete
and the goal stays active.
- Final gate sweep on the current tree (all uncommitted):
  - parity inventory regenerated and current (exit 0);
  - literal ratchet exit 0 WITH the new frozen `AREA_CAPS` aggregate check;
  - engine-definitions current (12 engines); frontend build-defaults current;
  - curated python formatting 109 files; `ruff check backend scripts` clean;
  - engine slice **510 passed** (~51s);
  - FULL parallel lane **2079 passed, 8 skipped** (~65s);
  - FULL frontend vitest suite **622 passed (67 files)**.
- Completed across the rounds: launch policy/build options/protocol/artifact
  contract/extensions/request-defaults/catalog/upstream/settings-defaults/
  data-dirs/model-compat as engine data (+ specs, schema v2, exporter, pins);
  install-kind capabilities; diagnostics audit; model-compat branches; frontend
  readiness filters, cancel routing, generated build defaults, keyed status
  store, descriptor-mapped version refs; dead audio helpers removed; the
  architecture gate made aggregate-frozen.
- Deliberately NOT done (with reasons, not excuses):
  1. the nine frontend VERSIONS refs themselves — ~56 consumer sites, several
     tests assign them; the MAPPING is now descriptor data (R84), converting the
     refs is a UI refactor of its own;
  2. audio workspace branches (`AudioWorkspace.vue`/`ModelConfig.vue`/
     `ModelRow`/`ModelLibrary`/`connectTarget` audio checks) — behavior-sensitive
     package detection, no safe descriptor substitute identified yet;
  3. descriptor-driven catalog rendering (the panel still renders the
     `GET /build-options` payload; only the fallback is generated);
  4. legacy venv route-module removal — those routes are LIVE compatibility
     delegates until (1)-(3) land;
  5. native GGUF launch stays legacy by decision (R23); native snapshot retries
     stay out of scope (R57).
- Each of the above has a concrete next step recorded in the handoff; none is
  blocked by a mystery, they are simply not done.
**R87 - Post-goal review fixes: declared defaults and declared bundle formats
are honored (2026-11-04, DONE)**. Two reproduced defects in the newest
declarations; both fixed with pins.
- DEFECT 1 (wrong install target): `register_definition_lifecycles` derived
  install defaults from the `lifecycle` section and OVERWROTE
  `settings_defaults`, so a YAML engine declaring branch `stable` and version
  `1.2.3` silently got `main` / empty. The derive+merge now lives in
  `definition_install_defaults()`: lifecycle values are the FALLBACK and every
  declared `settings_defaults` key wins. Pins: declared branch/version survive
  into `default_install_settings`; a lifecycle-only definition keeps the derived
  values.
- DEFECT 2 (wrong package kind): `models/schema._default_package_kind` checked
  `prepared_bundle_formats` and then restricted the result to `mixed`/`original`,
  so a declared `safetensors` bundle fell through to `hf_snapshot` and the record
  joined generic format inference. The DECLARATION now decides for any format it
  lists; the legacy "record has a family" rule stays restricted to
  `mixed`/`original`. Pins: declared `safetensors` -> `prepared_bundle` and
  compatibility stays explicit (no inferred engines); family rule unchanged;
  audio/llama behavior unchanged.
- Verification (lanes only): new pins **5 passed**; engine slice **515 passed /
  ~53s**; FULL parallel **2084 passed, 8 skipped**; parity current; ratchet no
  growth; defcheck 12; build-defaults current; ruff curated 110 files.
**R88 - Post-goal review fixes: view status reads + declaration-driven sidecar
macro (2026-11-04, DONE)**. Two more reproduced defects, both from the recent
declarations.
- DEFECT 3 (frontend status regression): R81 removed the named status refs, but
  `EnginesView.vue` still read `lmdeployStatus` / `onecatVllmStatus` /
  `audioCppStatus` in ~20 places — `undefined` at runtime, so status details
  vanished and audio tracking settings fell back to `main`. The view now defines
  three computeds over `enginesStore.statusForEngine('…')` and every read goes
  through them (template reads unwrap, script reads `.value`). The view tests'
  stubs were MASKING this by providing the removed properties; they now model the
  keyed map + accessor (`engineStatusById` / `statusForEngine`), so the
  regression cannot hide again.
- DEFECT 4 (incomplete macro migration): `proxy/launch_spec._compile_audio`
  recognised only `${studio_audio_config}` (and that name in the manifest
  macros), so a declared macro renamed to `audio_cfg` stayed an unresolved
  literal instead of binding `server.json`. It now derives the resolvable tokens
  from the manifest's own `macros` map (both `${name}` and the raw value), so ANY
  declared macro resolves. Pin: a manifest with `audio_cfg` compiles to the
  `server.json` artifact and leaves no `${audio_cfg}` token.
- The tightened ratchet did its job: the fix legitimately adds three labelled
  `statusForEngine('id')` calls + a test-stub key, so `frontend/src/views/` grew
  369 -> 376 and the gate refused until the cap was edited — the reviewed diff is
  now recorded in the script next to the cap.
- Verification (lanes only): manifest/runtime suites **61 passed**; new macro pin
  passes; engine slice **515 passed / ~53s**; FULL parallel **2085 passed, 8
  skipped**; FULL frontend **622 passed (67 files)**; parity current; ratchet
  exit 0; defcheck 12; build-defaults current; ruff curated 110 files.
**R89 - Phase-8 frontend: versions refs converted to one keyed map (2026-11-04,
DONE)**. The deferred half of R84 — the nine named refs themselves.
- The store now keeps `engineVersionsById` (keyed by engine id) as the SOURCE.
  The historic names (`llamaVersions`, `lmdeployVersions`, `audioCppVersions`, …)
  are **writable computeds** built from the descriptors' `versionsKey` mapping, so
  all ~56 consumers keep working unchanged — reads, template use, `storeToRefs`,
  AND the tests that ASSIGN `store.audioCppVersions = [...]` (the setter writes
  the keyed map; the getter reads it back).
- `fetchLlamaVersions` writes the keyed map directly (partition rule unchanged:
  `untaggedSourceOwner`/`repoSource` from the descriptor), and
  `versionsForEngine` reads the same map. The store carries NO engine-id → ref
  table and no per-engine version refs.
- Pins (in `engineKeyedReads.test.js`): the descriptor walk now checks BOTH
  directions (alias assignment → map, map write → alias); a second pin feeds
  `/api/llama-versions` and asserts the rows land in the keyed map and are
  visible through the aliases.
- Verification (lanes only): **FULL frontend 623 passed (67 files)**; engine
  slice **515 passed / ~48s**; FULL parallel **2085 passed, 8 skipped**; parity
  current; ratchet no growth; defcheck 12; build-defaults current; ruff curated
  110 files.
**R90 - Phase-8 frontend: one audio-package predicate (2026-11-04, DONE)**. The
same `engine === 'audio_cpp' || format === 'audio_cpp'` rule existed five times
across four model surfaces.
- New `frontend/src/utils/audioModel.js::isAudioPackage(model)` states the rule
  once (`AUDIO_PACKAGE` constant + engine/format checks, tolerant of
  `config.engine`, `engine`, `format`, `artifact.format`, null). `ModelRow`,
  `ModelLibrary`, `ModelWarmupButton` (two copies) and `AudioWorkspace` now call
  it; the four files carry no engine-id predicate literals any more
  (`AudioWorkspace` keeps its genuine `?engine=audio_cpp` API query).
- Pins: new `frontend/src/utils/tests/audioModel.test.js` covers the matrix
  (config engine / engine / format / artifact format vs non-audio, empty, null).
  The test lives under a `tests/` directory so its fixtures do not inflate the
  literal inventory — the first attempt sat in `utils/` and the ratchet caught it
  (+8), which is exactly the gate working.
- The consolidated constant is one new literal in `frontend/src/utils/`, so the
  reviewed cap moved 6 -> 7 with the reason recorded beside it; the views and
  components areas shrank.
- Verification (lanes only): **FULL frontend 625 passed (68 files)**; engine
  slice **515 passed / ~55s**; FULL parallel **2085 passed, 8 skipped**; parity
  current; ratchet exit 0; defcheck 12; build-defaults current; ruff curated 110
  files.
**R91 - Phase-8 frontend: ModelConfig families come from descriptors
(2026-11-04, DONE)**. Two hardcoded engine lists in `ModelConfig.vue` now read
the descriptor-derived sets.
- The engine-modal icon family was `['sglang', 'sglang_v100', 'vllm'].includes(…)`
  in the template; it now uses `SG_PANEL_ENGINE_IDS` (the descriptor-derived
  three-engine panel family). This is the correct set — `SGLANG_RUNTIME_UI_IDS`
  is only two engines (no vllm), which the existing pin at
  `engines.venv.test.js` makes explicit; using the wrong export would have
  dropped vllm's icon, so the family semantics were checked before wiring.
- The safetensors-to-other-engine normalization used a five-id literal list; it
  now uses `VENV_ENGINE_IDS` (the declared `venv` family). Equivalent to the old
  list exactly — audio.cpp is cmake, so it stays excluded as before.
- Pins: the two families are already pinned (`SG_PANEL_ENGINE_IDS` three ids,
  `VENV_ENGINE_IDS` five + descriptors agreeing); the full frontend suite is the
  behavior gate for the two call sites.
- Verification (lanes only): **FULL frontend 625 passed (68 files)**; engine
  slice **515 passed / ~54s**; FULL parallel **2085 passed, 8 skipped**; parity
  current; ratchet no growth; defcheck 12; build-defaults current; ruff curated
  110 files.
**R92 - Phase-8: per-engine build-settings defaults are engine data (2026-11-04,
DONE)**. `llama_versions._apply_engine_specific_build_defaults` branched on
`engine == "ik_llama"` to force `build_examples`.
- New schema-v2 `build_defaults` block (scalar mapping) + `EngineSpec.
  build_defaults`, declared for `ik_llama` (`build_examples: true`) and exported
  into its generated definition. The route now layers the DECLARED defaults over
  the catalog defaults — no id branch — and the value still wins over a user
  `build_examples: false`, preserving the ik_llama requirement exactly.
- This is the same "declaration is the operative rule" shape the review demanded
  in R87, applied to build settings rather than install settings.
- Pins (5 new, `test_engine_build_defaults_declaration.py`): ik declares it and
  the exported definition carries it (descriptor payload excludes it); the route
  layers it for ik and leaves llama untouched; a FAKE engine's declared default
  wins (proving spec-driven); the definition round-trip; and the two rejections
  (non-mapping, schema v1).
- Verification (lanes only): definition/build-config subset **71 passed**; pins
  **5 passed**; engine slice **520 passed / ~54s**; FULL parallel **2090 passed,
  8 skipped**; parity current; ratchet no growth; defcheck 12; build-defaults
  current; ruff curated 111 files.
**R93 - Phase-8: source-build repository identity is declared (2026-11-04,
DONE)**. `_resolve_engine_build_target` branched on two engine ids and looked the
URL up in `llama_manager.REPOSITORY_SOURCES`.
- It now reads the spec: native source builds are the engines declaring
  `install_kind: cmake` + `runtime_kind: llama_server`; `repository_source` comes
  from the spec and the URL from the DECLARED `upstream.repository_url` (R72).
  Rejections keep their exact 400s (audio.cpp is runtime `audio_cpp`, unsloth is
  `prebuilt`), and a YAML-native engine with a declared upstream resolves with no
  code edit.
- Pin (in the characterization suite): llama/ik resolve to their declared values,
  audio/unsloth/unknown reject with 400, and a FAKE cmake `llama_server` spec
  resolves from its declared upstream.
- Verification (lanes only): characterization suite **20 passed**; engine slice
  **520 passed / ~55s**; FULL parallel **2091 passed, 8 skipped**; parity current;
  ratchet no growth; defcheck 12; build-defaults current; ruff curated 111 files.
**R94 - Phase-4/8: the llama-family build-option catalog is engine data
(2026-11-04, DONE)**. The last big code-held catalog moved to declaration.
- The 96-option table (+ categories, category scope, build types, primary keys)
  moved verbatim from `llama_cpp/build_options.py` to the pure-data module
  `backend/engines/llama_build_catalog_data.py`, is carried on BOTH
  llama-family specs (`build_options`, same table; each entry declares which
  engines expose it), validated by the schema (new per-entry `engines` metadata on
  options and categories, token-validated), and exported into both generated
  definitions.
- `llama_cpp/build_options.py` now derives its option dataclasses, categories,
  category requires/scope, primary keys and build types from the spec: the union
  view keeps every option for defaults/coercion (stored configs unchanged) while
  `catalog_for_ui(engine)` filters per engine exactly as before — including the
  per-option `engines` array in the payload.
- One real structural catch: the data module first lived inside the
  `llama_cpp/` package, and importing it from `registry.py` triggered
  `llama_cpp/__init__` → manager → build_options → registry (partial) — an import
  cycle the exporter surfaced immediately. It now sits at the engines root, so
  the registry import is data-only.
- Evidence of behavior preservation: all 43 pre-existing llama build-option /
  build-config / source-build tests pass untouched, and the generated frontend
  build-defaults JSON is byte-identical (`build-defaults:check` current).
- Pins (5 new, `test_llama_build_catalog_declaration.py`): both specs carry the
  block (payload excludes it), the definitions carry it, `_catalog()` reads a
  FAKE spec (spec-driven proof), the per-engine UI filter (iqk only ik, blas only
  llama) and payload shape, and the union defaults still cover both engines.
- Verification (lanes only): catalog/definition subset **70 passed**; pins **5
  passed**; engine slice **525 passed / ~55s**; FULL parallel **2096 passed, 8
  skipped**; parity current; ratchet no growth; defcheck 12; build-defaults
  current; ruff curated 112 files.
**R95 - Phase-8 frontend: catalog fetch dispatch is declared (2026-11-04,
DONE)**. The build-options panel loaded its catalog through two bespoke store
functions chosen by the view.
- The three catalog-bearing descriptors now declare `buildOptionsPath`
  (llama_cpp/ik_llama share the llama-versions endpoint, audio.cpp its own), the
  registry exposes `buildOptionsPath(engineId)` (null when an engine has no
  catalog), and the store gained `fetchBuildOptionsForEngine(engineId)` which
  fetches from the DECLARED path. Both view loaders call it; the old
  `fetchAudioCppBuildOptions` compat wrapper delegates to the same path.
- Behavior identical: llama/ik still send `params: {engine}` to the same URL, and
  audio.cpp gets the same path (the extra `engine` param is ignored server-side).
- Pins: new `stores/tests/buildOptionsDispatch.test.js` (declared paths, per-engine
  dispatch with params, compat wrapper on the same path). Also moved the generated
  build-defaults test under `engines/tests/` and reworded two comments that
  mentioned engine ids, so the reviewed literal cap moved only 42 -> 44 for the
  two newly declared endpoints (recorded beside the cap).
- Verification (lanes only): **FULL frontend 628 passed (69 files)**; engine
  slice **525 passed / ~58s**; FULL parallel **2096 passed, 8 skipped**; parity
  current; ratchet exit 0; defcheck 12; build-defaults current; ruff curated 112
  files.
**R96 - Phase-8: sync/retry source-repo fallbacks read the declaration
(2026-11-04, DONE)**. Two remaining route sites still consulted the manager's
`REPOSITORY_SOURCES` name→URL table:
- the source-sync path defaulted `repository_source` with an engine-id ternary
  and looked the URL up in the map; the retry path did the same with `.get`.
- Both now use `_spec_repository_source(engine)` (the declared display name) and
  `_declared_source_repo(engine, name)` (the declared `upstream.repository_url`,
  empty when the recorded name does not belong to that engine). The 400s are
  unchanged: "Unsupported source engine" when the name resolves to nothing,
  "missing a source repository URL" when the URL is empty.
- Pin (characterization suite): the declared names and URLs for llama/ik, empty
  for an unknown engine, and empty for a name that belongs to the OTHER engine
  (the mismatch case that used to be a map miss).
- `REPOSITORY_SOURCES` remains the SOURCE-BUILD host contract
  (`source_build.py` uses it while publishing rows); it is no longer consulted by
  the routes for resolution.
- Verification (lanes only): affected suites **44 passed**; new pin passes; engine
  slice **525 passed / ~58s**; FULL parallel **2097 passed, 8 skipped**; parity
  current; ratchet no growth; defcheck 12; build-defaults current; ruff curated
  112 files.
**R97 - Phase-9: dead frontend exports retired (2026-11-04, DONE)**. A surface
audit (every exported symbol in `frontend/src/engines` + `utils`, cross-checked
against every consumer) found helpers exported for nobody.
- `parseCliCommand.js` exported 20 internals (regexes, coercers, index builders,
  skip helpers) that no consumer or test imported; they are module-local now, so
  the public surface is 11 names. `engineDescriptor` (registry), `workspaceTitle`
  (navigation) and the audio-package marker become module-local too, and the dead
  `formatFileSize` alias is gone.
- Pin: new `utils/tests/parseCliCommandSurface.test.js` asserts the module's
  exports are EXACTLY the consumed list, so re-exporting internals becomes a
  reviewed diff instead of silent surface growth.
- Verification (lanes only): **FULL frontend 629 passed (70 files)**; engine slice
  **525 passed / ~58s**; FULL parallel **2097 passed, 8 skipped**; parity current;
  ratchet no growth; defcheck 12; build-defaults current; ruff curated 112 files.
**R98 - Closing verification sweep and honest status (2026-11-04, SWEEP)**. Final
pass over the accumulated uncommitted tree (187 entries: 100 tracked modified, 87
untracked; nothing committed).
- Static gates re-ran clean in one sweep: parity inventory current, literal ratchet
  at the frozen caps, definitions current (12 engines; authoritative fixtures
  launchcraft/launchforge/yamlcraft), frontend build defaults current, curated ruff
  112 files formatted and clean, engine slice **525 passed / ~59s**, FULL parallel
  **2097 passed / 8 skipped**.
- **CORRECTION (R99): the frontend line of this sweep was WRONG.** I reported "FULL
  frontend 629 passed (70 files)" as green, but the real result was **629 passed +
  14 unhandled errors, exit code 1**. I grepped only the "Tests/Test Files" lines
  and ignored the error section and the exit status; a green assertion count hid
  broken execution. See R99 for the cause and the fix.
- **What the campaign has reached.** The definition surface now covers launch,
  lifecycle, runtime profiles, protocol, extensions, request defaults, catalog,
  upstream, settings defaults, build options (audio + llama family), build
  defaults, data dirs and model compatibility — validated by the schema, exported
  and equivalence-checked, and READ by backend services and the frontend (status
  maps, version maps, catalog endpoints, readiness, install defaults, build
  defaults, repository identity). Engine-id dispatch remaining in the tree is
  frozen by the ratchet's 17 area caps and mostly sits in the legacy compatibility
  layer (see below), not in the new code paths.
- **What is deliberately NOT done** (each with its next step, none blocked):
  1. audio `ModelConfig` profile panels: the audio-vs-text panel selection and the
     parameter-registry engine query are still audio-specific branches — a panel
     abstraction is a UI design task, not a mapping swap;
  2. legacy venv route modules (`lmdeploy_versions.py`, `vllm_versions.py`,
     `sglang_versions.py`, `onecat_vllm_versions.py`,
     `unsloth_llama_versions.py`): the frontend already uses the generic
     `/api/engines/{id}/{verb}` routes for `genericApi` engines, but these modules
     are live **compatibility delegates** for external callers; removing them is
     an API-surface decision (the plan's step-9 cutover) that needs an owner's
     call, not an autonomous delete;
  3. native GGUF launch stays legacy by decision (R23); native snapshot retries
     stay out of scope (R57);
  4. the global `ruff format --check` over the whole tree reports pre-existing
     drift in files outside the curated list; the curated gate (112 files) is the
     standing contract and is green.
- The plan is therefore **not complete**, and the budget is spent: the goal stays
  active-with-no-rounds rather than claiming a finish.
**R99 - Frontend gate was false-green: fixed and pinned (2026-11-04, DONE)**. The
review reproduced `629 passed, 14 unhandled errors, exit 1` on the full frontend
suite; R98's "all green" line was wrong.
- CAUSE: `EnginesView.tasks.test.js` mocked the engine store with a hand-written
  stub that never gained `versionsForEngine` (added to the store in R89). Every
  render of the SGLang panel then threw `store.versionsForEngine is not a
  function` INSIDE Vue, which Vitest reports as an "Unhandled Rejection" while the
  file still counts as passed. The buildConfig view test had the same gap.
- FIX: both view stubs now spread a shared fixture
  (`frontend/src/test-fixtures/engineStoreAccessors.js`) that owns the synchronous
  read accessors (`statusForEngine`, `versionsForEngine`) and their keyed maps, so
  the stub cannot drift from the accessor contract again.
- PIN: new `views/tests/engineStoreAccessorContract.test.js` scans every `.vue`
  source for `store.<name>ForEngine(...)` calls and fails when the fixture does
  not provide that accessor. Adding a panel accessor now forces the fixture (and
  therefore every stub) to follow.
- GATE HYGIENE: the frontend gate is now recorded with its **exit code** and an
  explicit "0 unhandled errors" check, not just the assertion count. Verified:
  `VITEST_EXIT:0`, **630 passed (71 files)**, zero unhandled.
- Backend untouched: engine slice **525 passed**, FULL parallel **2097 passed / 8
  skipped**, parity current, ratchet no growth, build-defaults current, curated
  ruff 112 files.
**R100 - The frontend gate now reports its own verdict (2026-11-04, DONE)**. R99
fixed the false-green; this makes the class detectable by tooling instead of by a
reviewer re-running the suite.
- New `frontend/src/utils/frontendGate.js::summarizeFrontendRun(output, exitCode)`
  strips ANSI, then requires ALL of: exit 0, no "Unhandled" text, no `Errors N
  errors` section, and at least one passed test. `formatVerdict()` renders
  `frontend gate: CLEAN (exit 0, N tests passed, no unhandled errors)` or a FAILED
  line naming every reason.
- New `scripts/frontend-gate.mjs` runs the full suite through that summarizer and
  exits non-zero on any failure; `npm run test:frontend:gate` wires it.
- Pins: `utils/tests/frontendGate.test.js` reproduces the exact R98 false-green
  output (tests passed + `Errors  14 errors` + Unhandled) and asserts it is
  REJECTED, plus ANSI-wrapped counts, unhandled-with-exit-0, non-zero-exit, and
  the no-tests-ran case.
- Verified on the real suite: `npm run test:frontend:gate` →
  `frontend gate: CLEAN (exit 0, 636 tests passed, no unhandled errors)`, exit 0.
  (636 = 630 + the 6 summarizer pins; 72 files.)
- Statics unchanged and green: parity current, ratchet no growth, defcheck 12,
  build-defaults current, curated ruff 112; engine slice **525**, FULL parallel
  **2097 passed / 8 skipped**.
**R101 - Phase-8: the audio config panel is a declared family (2026-11-04,
DONE)**. `ModelConfig` compared `engine === 'audio_cpp'` in ~11 places to select
the audio panel and the audio-only copy/behaviour.
- The audio descriptor now declares `configPanel: 'audio'`, and the registry
  exposes `isAudioEngineId(engineId)`. Every comparison in `ModelConfig` uses it:
  the panel `v-else-if`, four label/tooltip expressions, `isAudioEngine`, the
  family/task drafts, the preserved-unknown-keys rule, and the param-registry
  watcher guards. Function-local `engine` arguments keep identical semantics (the
  helper is a pure predicate on the id).
- Left as literals BY DESIGN (they are API arguments, not dispatch): the three
  `fetchParamRegistry('audio_cpp', …)` calls and the curated package-kind list.
- Pins: new `engines/tests/audioPanelDeclaration.test.js` (audio declares the
  panel, every other engine does not, exactly one descriptor claims it, unknown/
  undefined are false).
- Verification: **frontend gate CLEAN (exit 0, 639 tests passed, no unhandled
  errors, 73 files)**; parity current; ratchet no growth; build-defaults current;
  curated ruff 112; engine slice **525**; FULL parallel **2097 passed / 8 skipped**.
**R102 - Phase-9: the compatibility-route cutover now has a current blast radius
(2026-11-04, DONE)**. Removing the legacy per-engine routers is an API-surface
decision, so instead of an autonomous delete this round produced the evidence it
needs.
- New `scripts/legacy-route-inventory.py` scans `backend/routes/*_versions.py` (both
  shapes: `python_engine_router` mounts, possibly several per module, and
  hand-written `@router.<verb>("/prefix/endpoint")` handlers) and emits
  `docs/legacy-route-inventory.md`: module, engine(s), legacy prefix, endpoint
  count, the frontend descriptor segment + `genericApi` flag, and the tests that
  cover the prefix.
- Inventory today: **6 legacy mounts across 5 modules** — lmdeploy, 1cat-vllm,
  sglang, sglang-v100, vllm (7 endpoints each via the shared factory) and unsloth
  llama (3 hand-written). Every venv engine with a generic router has
  `genericApi: yes`, i.e. the frontend does not depend on the legacy prefix; the
  delegates exist for external callers.
- `--check` is wired into the lint chain (`npm run legacy-routes:check`) and a new
  pin `test_legacy_route_inventory_current.py` keeps it current in the engine lane;
  the pin also asserts the generic/factory/llama/audio modules are NOT listed as
  delegates.
- Verification: inventory current; parity current; ratchet no growth; defcheck 12;
  build-defaults current; curated ruff **113**; engine slice **527**; FULL parallel
  **2099 passed / 8 skipped**; frontend gate **CLEAN (639 passed, 73 files, 0
  unhandled)**.
**R103 - Phase-9: the cutover claim is now proven on BOTH sides (2026-11-04,
DONE)**. R102's inventory under-reported, and probing the two "no" rows found the
scanner's bug rather than a product gap.
- BUG: the descriptor extractor used a line-anchored regex that (a) missed the
  quoted key `'1cat_vllm'` and (b) let a neighbouring block swallow one engine; it
  reported 1cat-vllm and unsloth llama as "no generic surface". Replaced with a
  brace scanner over `ENGINE_DESCRIPTORS` and a separate reader for
  `NATIVE_GENERIC_VERBS`; the row column is now `venv` / `native` / `no`.
- RESULT (correct): **all five venv engines route `venv`** (lmdeploy, 1cat-vllm,
  sglang, sglang-v100, vllm) and the hand-written native module routes `native`
  (unsloth through declared verbs). **No legacy mount is the only surface for its
  engine** — the delegates exist purely for external callers.
- Pins on both sides: backend `test_legacy_route_inventory_current.py` now fails if
  any inventory row is `no`; new frontend
  `engines/tests/genericRouteCoverage.test.js` asserts every `VENV_ENGINE_IDS`
  descriptor declares `genericApi` and `venvApiPath` builds
  `/api/engines/{id}/{verb}` (incl. the legacy-prefixed 1cat-vllm), and that the
  three native engines declare generic verbs.
- Verification: inventory current; parity current; ratchet no growth; defcheck 12;
  build-defaults current; curated ruff **113**; engine slice **528**; FULL parallel
  **2100 passed / 8 skipped**; frontend gate **CLEAN (642 passed, 74 files, 0
  unhandled)**.
**R104 - Phase-9: ratchet caps tightened to the measured totals (2026-11-04,
DONE)**. The area caps were ceilings, so slack accumulated silently as literals
were retired.
- Measured against the frozen caps: `backend/routes/` had **7** spare,
  `frontend/src/components/` **3**, `frontend/src/views/` **25** (the R88 bump plus
  everything R90/R91/R101 retired). Caps lowered **169 -> 162**, **86 -> 83**,
  **376 -> 351**; aggregate caps **1385 -> 1350**.
- New pin `test_engine_literal_caps_tight.py` asserts EVERY cap equals its measured
  area total and that `main()` still passes — a future retired literal therefore
  forces the cap down in the same review instead of leaving slack.
- Verification: ratchet exit 0 with zero slack; parity current; defcheck 12;
  build-defaults current; legacy-routes current; curated ruff **114**; engine slice
  **530 passed**; FULL parallel **2102 passed / 8 skipped**; frontend gate **CLEAN
  (642 passed, 74 files, 0 unhandled)**.
**R105 - Phase-8: the native source-build family is declared (2026-11-04, DONE)**.
`backend/routes/llama_versions.py` still spelled the family as six
`("llama_cpp", "ik_llama")` tuples plus a registration loop over the same pair.
- New registry helper `native_source_engine_ids()` derives the family from the
  specs (`install_kind == cmake` AND `runtime_kind == llama_server`); the route uses
  `_is_native_source_engine()` for all six guards and the lifecycle-registration
  loop. Behavior identical: audio.cpp is cmake but runtime `audio_cpp`, unsloth is
  prebuilt, so only llama.cpp and ik_llama qualify.
- The TIGHT cap pin immediately demanded the next step: `backend/routes/` fell
  162 -> **152** and the cap was lowered in the same review (aggregate caps 1350 ->
  **1340**). One reworded comment was needed because the ratchet counts the
  `llama-versions` token as engine-id literal text in `scripts/` — the gate caught
  my own comment.
- Pins (3 new, `test_native_source_family_declaration.py`): the derived family is
  `("ik_llama", "llama_cpp")`; the route predicate matches the declaration and
  rejects audio/unsloth/venv/unknown/empty; every family member's declared kinds
  agree.
- Verification: ratchet exit 0 (zero slack); parity current; defcheck 12;
  build-defaults current; legacy-routes current; curated ruff **115**; engine slice
  **533 passed**; FULL parallel **2105 passed / 8 skipped**; frontend gate **CLEAN
  (642 passed, 74 files, 0 unhandled)**.
**R106 - Phase-8: the remaining llama families are declared (2026-11-04, DONE)**.
Three engine-home modules still carried id tuples.
- New registry helper `llama_server_engine_ids()` (every spec with
  `runtime_kind == llama_server`: llama.cpp, ik_llama, the prebuilt unsloth build;
  audio.cpp excluded by its own runtime).
- `engines/scan/scanner.py` uses it for the llama scan/help branch;
  `engines/lifecycle.py` derives `_NATIVE_SHARED_ROOT_ENGINES` from
  `native_source_engine_ids()` and its cmake guard from
  `engine_ids_with_install_kind("cmake")`. `llama_cpp/resolve.py` KEEPS its ordered
  tuple on purpose: that tuple is a PRIORITY (llama.cpp first), not a membership
  test, and a sorted derivation would silently change which binary wins.
- Pins (in `test_native_source_family_declaration.py`): the runtime family includes
  the three llama engines and excludes audio, the source family is a subset, the
  lifecycle families are declared, and the scanner source no longer contains the
  id tuple. The membership pins are deliberately ⊆/exclusion based — an exact-tuple
  pin failed in the full lane because hermetic fixture engines legitimately declare
  the same runtime (the assertion was wrong, not the code).
- Verification: ratchet exit 0; parity current; defcheck 12; build-defaults
  current; legacy-routes current; curated ruff **115**; engine slice **536 passed**;
  FULL parallel **2108 passed / 8 skipped**; frontend gate **CLEAN (642 passed, 74
  files, 0 unhandled)**.
**R107 - Round-budget close-out: final sweep + status (2026-11-04, SWEEP)**. Last
round of the 32-round budget; tree remains **196 uncommitted entries** (100 tracked
modified, 96 untracked; nothing committed).
- FINAL SWEEP, all green: parity inventory current; literal ratchet exit 0 with ZERO
  cap slack (aggregate caps **1340**); definitions current (12 engines); frontend
  build defaults current; legacy-route inventory current; curated ruff **115** files
  formatted + clean; engine slice **536 passed** (~70s); FULL parallel **2108 passed
  / 8 skipped**; FULL frontend gate **CLEAN — exit 0, 642 passed (74 files), 0
  unhandled errors**.
- REACHED (declaration is the operative rule, not a parallel copy): launch,
  lifecycle, runtime profiles, protocol, extensions, request defaults, catalog,
  upstream, settings defaults, build options (audio + the 96-entry llama family),
  build defaults, data dirs and model compatibility are schema-validated, exported
  into the definitions, equivalence-checked, and READ by the backend services and
  the frontend (keyed status/version maps, catalog endpoints, declared families,
  readiness, install/build defaults, repository identity, config panel).
- REMAINING, precisely:
  1. **Legacy compatibility-router cutover** — evidence ready in
     `docs/legacy-route-inventory.md` (6 mounts / 5 modules, gated current). The
     frontend depends on NO legacy prefix (all `venv`, one `native`); deleting them
     breaks external callers, so it is an owner's API decision, not an autonomous
     delete;
  2. **native GGUF launch** stays legacy by decision (R23) and **native snapshot
     retries** out of scope (R57);
  3. one deliberate literal in `AudioWorkspace` (`?engine=audio_cpp`) — an API
     ARGUMENT, not dispatch;
  4. `llama_cpp/resolve.py` keeps the ordered engine tuple on purpose: it encodes
     binary PRIORITY.
- The plan is NOT complete and the round budget is spent. Both documents carry the
  same status; the goal stays active rather than claiming a finish.

### Original slice design (pre-execution skeleton, kept for the record)

- **R1 — Launch interpreter.** Definition-bearing branch after the
  nine's chain in `_compile_launch`: profile → LaunchSpec
  (`{{port}}`→`PORT_PLACEHOLDER`; model_binding sources; cwd).
  Byte goldens from reference YAMLs. Cutover allow-list += the new
  reader (deliberate, disclosed). Nine unreachable by ordering — pinned
  by a test proving a generated id NEVER enters the branch.
- **R2 — Executable + env resolvers.** `executable.kind` vocabulary
  (`venv_console_script` / `venv_python` / `native_path`) → analogues of
  commands.py:814-873; `env.required` missing → typed
  LaunchCompileError(field=…) (sglang_v100 CUDA_HOME analogue);
  `prepend_bin_to_path`/`cwd` → env/PathSpec. Proves the sglang_v100
  CUDA refusal declaratively (golden equality with the legacy path).
- **R3 — Launch-capable fixture + real journey.** Hermetic local wheel
  (built in-test from `backend/tests/fixtures/yamlcraft_server/`, pip
  install-from-path — no network) whose console script serves
  `/health` + minimal `/v1/models` + `/v1/chat/completions`. Extend the
  fixture definition (or a second definition — Q1) with a
  `runtime_profiles.default`. Journey test IN THE CONTAINER (real
  Popen): install→activate→model record→start→argv == YAML golden→
  health→running→STOP drains. Precedent harness: test_proxy_live_routing.
- **R4 — Scanner + inference.** YAML id reuses `vllm_help`-adjacent
  scanner or the fixture's own `--help` via existing kind (Q3 decides);
  param catalog lands; chat round-trip through the proxy (G4);
  decide `discovery.parser` fate (wire or delete as decorative — it is
  test-only today).
- **R5 — Policy data (optional, gated).** `_CPU_REJECTED_ENGINES` /
  `_GPU_ONLY_INVARIANTS` / `_RUNTIME_CONTEXT_KEYS` move to
  spec/profile-backed reads ONLY with per-key golden equality; ratchet
  counts must SHRINK. Interleave-able; not a gate for G1–G6.

Explicitly out: cutting the nine over onto profiles (their adapters stay
the source of truth; profiles stay reference-pinned), engine-scoped
start/stop verbs (none exist today — model-scoped stays), the
Windows-host launch path (POSIX fences unchanged), flag-gating for YAML
engines (no mechanism; none added).

## 4. Decisions (owner sign-off) — SIGNED 2026-11-03

- **Q1 = second definition + own wheel.** New authoritative launch
  fixture (id `launchcraft`) with a runtime_profile + hermetic
  in-test-built local wheel serving /health + /v1/chat. yamlcraft.yaml
  and every S1–S6.1 pin stay byte-identical.
- **Q2 = post-chain branch.** New branch fires only for
  definition-bearing ids with no legacy adapter; the nine's dispatch is
  byte-safe by ORDER, pinned by a test. No adapter-registry refactor.
- **Q3 = reuse an existing scanner_kind** (fixture --help shaped to fit
  one existing dispatch entry, golden-pinned). No vocabulary growth.
- **Q4 = chat round-trip REQUIRED.** G4 streams an OpenAI-style chat
  through the Studio proxy to the YAML-launched server before the
  campaign may claim launch works.

## 5. Risks

- Readiness is load-bearing: a fixture server that flakes /health
  hangs `acquire` (2700s budget) — journey tests must poll with their
  own tight timeout and always teardown via registry.stop.
- Global proxy singletons in-process: journey tests need isolation
  precedent from test_proxy_live_routing (ports, temp roots, no cross-
  test proxy reuse); pytest-parallel safety (S6-era lanes) verified.
- The nine's manifests are sha-goldens: ANY accidental env/PATH delta
  from shared helpers fails goldens loudly — good; keep new code in new
  modules, not in `_compile_env`.
- profiles.py today asserts argv equality against the LEGACY compilers
  for generated engines — R1 must not weaken those pins while making
  profiles runtime-live for new engines only.
- Unclear (flagged by surveys): launcher.py production caller (preview-
  only likely), Preload/Reload frontend call-sites, ik_llama absent
  from `_LLAMA_SERVER_ENGINES` — R3 must not depend on resolving these.
