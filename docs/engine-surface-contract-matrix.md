# Engine Surface Contract Matrix

> **Historical evidence — not the release entry point.** Start at the
> [delivery map](delivery-map.md); the [change inventory](change-inventory.md)
> maps this work to the review series.

Status: **survey + decisions** (Phase 3 gate). This document unblocks the
generic `/api/engines/{engine_id}` router by satisfying the plan's
precondition: *"Response contracts … must be unified or explicitly typed per
surface before delegation"*
(`docs/config-driven-engine-refactoring-plan.md:122`).

The typing half of the gate ships in
[`backend/models/engine_contracts.py`](../backend/models/engine_contracts.py),
structurally pinned by
[`backend/tests/test_engine_contracts.py`](../backend/tests/test_engine_contracts.py).
No runtime behavior changed: **nothing is wired yet.** This matrix is the
decision record a delegation tranche must honor.

Scope: the **engine-lifecycle HTTP surfaces** only — the endpoints a generic
router would subsume plus the extensions it would declare. Data-plane audio
(`/v1/audio*`, `/v1/tasks|batches`) and model/runtime/config routers are out
of scope.

Citations are `file:line` from the pre-delegation tree. Line numbers drift;
the *key sets* and *status codes* are the load-bearing claims, pinned by
`test_engine_contracts.py` where they are typed.

---

## 1. Surface inventory (who serves what today)

| Family | Router file | Mount | Engine id in path? |
|---|---|---|---|
| venv ×5 (lmdeploy, vllm, sglang, sglang_v100, 1cat_vllm) | `routes/engine_versions.py::python_engine_router` | `/api` + per-engine static prefix (`/lmdeploy`, `/vllm`, `/sglang`, `/sglang-v100`, `/1cat-vllm`) | **No** — baked into prefix; hyphen↔underscore mismatch (`/sglang-v100`↔`sglang_v100`) |
| llama family (llama_cpp, ik_llama, unsloth_llama) | `routes/llama_versions.py` | `/api/llama-versions` | **No** — `engine` is a query/body field or inferred from `version_id` prefix |
| unsloth (extra) | `routes/unsloth_llama_versions.py` | `/api` + `/unsloth-llama` | **No** |
| audio.cpp | `routes/audio_cpp_versions.py` | `/api/audio-cpp` | **No** — single-engine, flag-gated (404 when `AUDIO_CPP_ENABLED` off) |
| cross-family DELETE | `routes/llama_versions.py::DELETE /{version_id}` | `/api/llama-versions/{version_id}` | **Yes** — `version_id` = `engine:version`, dispatches through the now-plan-driven `_remove_engine_version` |

**None of these declare `response_model`** except llama's `check-updates`
(`response_model=UpdateCheckResponse`, `llama_versions.py:553`). Every other
route is annotated `-> Dict`, so OpenAPI emits an untyped
`object / additionalProperties` — the frontend consumes shapes only humans
enforce.

---

## 2. The check-updates surface — the plan's headline divergence

Five genuinely different contracts, all keyed "check-updates":

| Producer | Path | Key set |
|---|---|---|
| llama.cpp (`upstream_versions.check_engine_updates`, typed) | `/llama-versions/check-updates?source=` | `{latest_release: {tag_name,published_at,html_url}, latest_commit: {sha,commit_date,message}}` |
| factory pypi (lmdeploy, vllm, sglang) | `/{p}/check-updates` | `{latest_version, current_version, update_available, releases: [..], url}` |
| factory github_commit (sglang_v100) | `/sglang-v100/check-updates` | `{latest_version, latest_commit, current_version, update_available, url}` — **no `releases`** |
| factory github_releases (1cat_vllm) | `/1cat-vllm/check-updates` | **`{latest_version, releases}`** — no `current_version`, no `update_available`, no `url` |
| unsloth | `/unsloth-llama/check-updates` | `{latest_version, current_version, update_available, url, release_url}` — `url` == `release_url` always |
| audio.cpp | `/audio-cpp/check-updates?ref=` | drift-aware: `{current_version, latest_version, update_available, latest_commit: {…}, latest_release: {…}|None, update_channel, tracking_ref, repository_url, contract_fingerprint}` |

Note the sub-divergence *inside* the "factory releases dictionary" the plan
names: it is **three** shapes, chosen by `UpdateSource.kind`, so it was never
a single thing to unify with. Failure mapping also diverges — the factory
returns the error **as the response's own status** (500 for lmdeploy/1cat,
502 for vllm/sglang/sglang_v100), audio maps GitHub 403→429 / 404→404 /
else→502, llama maps `UpstreamRequestError` 403→429 / 404→404 / else→500.

**Decision — explicitly typed per producer, one tagged union.** There is no
harmless common denominator (`github_releases` omits `update_available`
outright; llama's payload has no scalar `latest_version` at all). Forcing one
shape would *invent* fields, violating the byte-preservation rule. Ship the
per-producer models already in `engine_contracts.py`
(`PypiCheckUpdates`, `CommitCheckUpdates`, `ReleasesCheckUpdates`,
`UnslothCheckUpdates`, `AudioCheckUpdates`, re-exported `UpdateCheckResponse`).
A generic router exposes `GET /api/engines/{id}/check-updates` whose
`response_model` is a **tagged union** of these selected by the definition's
declared update source — OpenAPI `oneOf`, each variant already typed.
**Unify later only if a client is found to actually need it.**

---

## 3. The status surface

Two distinct contracts plus a three-way split inside the venv family:

- **venv `status`** (`VenvEngineStatus`) — but three producers emit different
  key sets: `_stateful_status` (lmdeploy, 1cat: **no `engine` key**, has
  `removed_at`), hand-rolled vLLM (`engine`, `source_commit`, `cuda_*`,
  `python_version`, **no `removed_at`**, `binary_path` = the venv
  `python.exe`, not an engine binary), SGLang's copy of that **plus `"v100":
  bool`**. `status` **never errors** — a swallowed exception returns a 200
  fallback whose extra keys are exactly the engine's `status_fallback`
  overlay (lmdeploy/1cat add `installed_at`/`removed_at`/`log_path`; 1cat also
  `release_tag`; others none).
- **audio `status`** (`AudioEngineStatus`) — the drift-aware dict the plan
  names: `capability_delta` (a 6-key empty fallback vs a 9-key populated
  producer), `drift_warnings: list[str]`, `contract_fingerprint` (sha256
  hex), `affected_models`. Structurally different family; see §5.

**Decision — `VenvEngineStatus` typed as a shared core (`extra="allow"`) with
per-field `Optional`; audio `status` stays a declared extension sub-route with
`AudioEngineStatus`.** The venv family genuinely shares a status *concept*
(installed/version/binary/venv/operation/…), so one lenient core is honest and
matches the 200-always contract. The family-specific extra keys (`v100`,
`cuda_*`, `release_tag`, `removed_at`) ride as extras — exactly the `_OpenModel`
philosophy. `status` is the **best unification candidate** in the whole survey;
a generic `GET /api/engines/{id}/status` can use `VenvEngineStatus` directly
once the definition supplies the installer's status dict.

---

## 4. Shared lifecycle surfaces (the generic router's real payload)

These are already near-uniform and are what a generic router should own:

| Surface | venv family | llama family | audio |
|---|---|---|---|
| **install accept** | `{message, task_id}` (`_started_response`); sglang_v100 → `{message, task_id, repo, branch}` | `{message, task_id, commit_sha, …}` (`_schedule_source_build`) | `{message, task_id, status:"started", version_name, source_ref, source_ref_type, …}` |
| **remove accept** | `{message, task_id}` | handled by `DELETE /{version_id}` (this tranche's flow) | `DELETE` via same shared route |
| **cancel** | `{ok, message[, terminated, task_id]}` (`CancellableOperationManager`, always 200) | `{ok/message}` (`BuildTaskManager`) | `{ok, terminated?, message, task_id?}` (`BuildTaskManager`, **only `type=="build"`** cancelable) |
| **activate** | (venv uses `set_active`, not this surface) | `{message}` sync (`activation.py:139`) | `{message, engine, version, operation_id, background:true}` async (`activation.py:280-286`) |
| **build-settings GET/PUT** | engine-specific keys (`pip_version`/`release_version`/none for v100) | llama cmake keys + `tracking_ref`/`repository_url` | cmake keys + `backend` + tracking |

**Decisions:**
- **install/remove accept → unify on `InstallAccepted`** (`{message, task_id,
  status}` core, extras allowed). All families already emit `message`+`task_id`;
  `status:"started"` is audio-only but harmless to default. Extras
  (`repo`/`branch`/`version_name`/`source_ref`) ride. This is a genuine
  unification, not an invention.
- **cancel → `CancelAccepted`** (`{ok?, terminated?, cancelled?, message?,
  task_id?}`, all-optional + extras): the two managers (`CancellableOperationManager`,
  `BuildTaskManager`) already share `ok`/`message`/`terminated`/`task_id`; the
  *only* honest typing is all-optional because `ok:false` bodies omit
  `terminated`/`task_id`. Do not unify the "which task types are cancelable"
  rule — that is a behavioral divergence, not a contract one (flagged in §6).
- **build-settings → per-family typed, NOT unified.** The key sets are
  engine semantics (version-source key differs; v100 has none). A generic
  route can pass a `Dict` through, but the *typed* contract stays per-engine.
- **activate → leave out of v1 of a generic router.** venv engines don't
  expose it; llama sync-returns `{message}` while audio async-returns
  `{operation_id, background}`. Unifying activation is a behavior change
  (sync vs background), explicitly out of scope for a contract tranche.

---

## 5. Declared extension surfaces (typed sub-routes, never folded in)

The plan carves these out; each becomes a definition-declared extension with
its own typed model. They are **not** candidates for a generic `…/{engine_id}/…`
lifecycle route:

| Extension | Model | Notes |
|---|---|---|
| audio `/contract` | `AudioContractResponse` | composed descriptor (schema/operations/task_routing/family_routing/drift_warnings/descriptor_fingerprint) + envelope; 409 string when no composed contract |
| audio `/status` | `AudioEngineStatus` | §3 |
| audio `/community-voices{,/install}`, DELETE | `CommunityVoicesStatus` | `{source, installed, voice_dir, items[]}`; 502 install / 500 remove on `CommunityVoiceError` |
| llama `/build-capabilities` | (extension, Dict today) | CUDA/OpenBLAS probe with safe-default fallback |
| llama `/cuda-status`, `/cuda-install`, `/cuda-logs`, `/cuda-uninstall` | (extension, Dict today) | host CUDA toolkit management, not engine-version lifecycle |
| llama `/build-options` | (extension, Dict today) | cmake catalog for the settings dialog |
| venv `verify`? | — | venv has **no** verify; llama has `GET /verify/{version}` → `{version, verification, commands, all_available}`. Keep llama-only. |

`engine_contracts.py` ships typed models for the three **audio** extensions
(the ones the plan explicitly names); the llama build/cuda extensions are
documented here but left Dict-typed — they are host-management surfaces with no
cross-family sharing, so typing them adds ceremony without a delegation win.
A future tranche may type them if a stable client contract is desired.

---

## 6. Cross-cutting divergences a delegation tranche MUST preserve (not silently "unify")

These are behavioral, discovered by the survey, and are the reason a generic
router cannot be a blind pass-through:

1. **409 detail is a dict in some pairs, a string in its siblings.**
   `ActionAdmissionError` → 409 **dict** `{admit, code, retry, operation_id,
   state_token, message}`; a busy `RuntimeError`/`ResourceBusyError` → 409
   **string** — in the *same* install endpoint. `FilesystemRefusal` → 409
   **dict** `{code, message}`. The global handlers (`main.py:322-339`) are
   shadowed by these local `except`s. Unifying them would change client-visible
   error bodies → **must be preserved as-is**, typed as "detail is dict-or-string."
2. **check-updates failure status is per-engine** (500 vs 502). Preserved.
3. **cancel semantics differ by manager** (build vs cancellable; only
   `type=="build"` cancelable via `BuildTaskManager`). A generic router must
   dispatch cancel to the right manager, not assume one.
4. **`sglang_v100/install` ignores `version`/`force_reinstall`** and schedules a
   source build (`prefer_source_install`). A generic install route must read a
   definition flag for this, not assume release install.
5. **`binary_path` means different things** across venv engines (console script
   vs venv `python.exe`). Do not reinterpret it in a shared contract.
6. **audio is flag-gated** — every audio route 404s when the feature flag is
   off. A generic router that routes to audio must reproduce the gate.
7. **engine id never enters the venv path as a parameter** (static prefixes,
   hyphen/underscore mismatch). Introducing `/api/engines/{engine_id}/…` is a
   NEW path surface; the old prefixes must remain (aliases) or be migrated in
   lockstep with the frontend's `openapi.json`.

---

## 7. Delegation seam (SHIPPED — generic router v1)

Shipped as `routes/engine_lifecycle.py`, mounted at `/api/engines` beside the
existing list route. What v1 does and how it honors this seam:

1. ONE generic router over the uniform surfaces — `status`, `check-updates`,
   `build-settings` GET/PUT, `install`, `install-source`, `remove`, `cancel` —
   built on the same `lifecycle_*` callables the legacy factory routes call
   (`routes/engine_versions.py` now registers an `EngineLifecycle` record per
   engine at construction; the generic router resolves `engine_id` against
   that registry). Bodies and error contracts are identical BY CONSTRUCTION,
   pinned by `tests/test_generic_engine_router.py` (same-payload equality of
   both surfaces for every verb, including 400/500/502 error paths).
2. Per-family semantics ride the record — `prefer_source_install`,
   `version_setting_key`, `status_fallback` overlays, update-source kind — so
   the generic surface reproduces every divergence in §6 without a single
   engine-id literal (asserted by the suite's source-scan test).
3. Typed contracts attached via `responses=` (OpenAPI documentation) NOT
   `response_model=` (which would re-serialize and inject null keys,
   violating byte-equality). `VenvEngineStatus`, `InstallAccepted`, and
   `CancelAccepted` appear in the generated schema.
4. Legacy prefixes are UNTOUCHED (still the primary frontend surface); the
   openapi delta is additions-only. Alias/migration is a separate decision.

**Still deferred to later tranches** (not v1):
- Dispatch membership is currently "engine has a lifecycle record" (the five
  venv engines). Native/audio join only when their definitions declare the
  generic lifecycle AND a family installer adapter exists (matrix §4/§6.3,6).
- `activate` (sync-vs-background divergence, §4) and `verify` (llama-only).
- The llama/audio **extension** surfaces (audio `/contract`, `/community-voices`,
  drift `/status`; llama `/build-capabilities`, `/cuda-*`, `/build-options`)
  stay on their current routers as declared extensions.
- The frontend still calls legacy prefixes; migrating it (and possibly
  retiring the prefixes) is the frontend tranche.

---

## 8. Frontend consumption (SHIPPED — registry v1→v3, generic surface v2)

The literal survey (union grep, ~1,150 matching lines / 45 files) put the
densest per-engine duplication in `stores/engines.js` (~45 hand-written URL
literals: near-identical 6-function blocks for lmdeploy vs 1Cat, a third
slug-mapped family for sglang/sglang_v100/vllm where **vLLM status is called
as `fetchSglangStatus('vllm')`**) — the frontend twin of the backend factory
this document started from.

**v1 shipped** (`frontend/src/engines/registry.js`):

- `VENV_DESCRIPTORS` — the five python engines with `apiSegment` (hyphen
  spellings), `statusRef`, and `nullSafeStatus` (the real SGLang-family
  `data || {}` divergence from §6, carried as data so callers cannot
  "accidentally unify" it).
- `stores/engines.js` now runs ONE implementation per verb
  (`venvFetchStatus`, `venvInstall`, …); all historical export names/signatures
  remain as thin wrappers, so zero component call-sites changed.
  `engines.venv.test.js` is a 29-pin characterization matrix (exact axios
  method+URL+payload+state-ref per engine × verb, including remove's
  status→versions refresh ORDER and the null-handling split); it was proven
  green against the hand-rolled store BEFORE the refactor and passes
  UNCHANGED after it.
- **v2 (generic surface)**: all five descriptors carry `genericApi: true` —
  a per-engine allow-list — and the store's eight verbs now call the
  generic `/api/engines/{id}/*` router. `legacyVenvApiPath` keeps the
  legacy segments as a rollback path (pinned by literal). The task-cancel
  map (`useTaskCancelEndpoint`) derives its venv entries from descriptor
  `managerKey`s (`onecat_vllm` manager → `/api/engines/1cat_vllm/cancel`);
  non-venv managers (cuda, audio_cpp, unsloth_llama) keep their legacy
  endpoints. The URL move was made deliberately across
  `engines.test.js`/`taskFixtures.js` — response bytes are unchanged
  because the backend pins generic == legacy per verb (v1 §7), and this is
  the FIRST PRODUCTION TRAFFIC over the generic router.
- **v3 (identity convergence)**: `ENGINE_DESCRIPTORS` now covers all NINE
  engines and absorbs the six parallel records as descriptor fields /
  derived views: `label`+`icon` (was `ENGINE_LABELS`,
  `fallbackEngineOptions` — picker order pinned), `buildHint` (was
  `HINT_COPY`, unsloth's no-hint gap kept as data), `cmakeCapable` (was
  `CMAKE_ENGINES`), `sglangRuntimeUi` vs `panelFamily` (the TWO distinct
  sglang families — with/without vllm — pinned apart, never merged), and
  `fallbackFormats` (was `inferredEnginesByFormat`). `managerKey` gained a
  reverse `ENGINE_ID_BY_MANAGER` + manager-aware `engineLabel()`; the first
  EnginesView call sites (panel-family branch, four toast labels) derive
  from it byte-identically. `cancelSglang`/`venvCancel` (zero callers)
  retired.
- **v3 close-out (store branching)**: the duplicated
  `activateVersion`/`deleteVersion` versionId substring chains are ONE
  declarative fan-out (`STATUS_REFRESH_INDEPENDENT` / `_EXCLUSIVE` /
  `_ACTIVATE_ONLY`) — order, first-match `break`, audio-activate-only, and
  the 1Cat-implies-vLLM refresh quirk encoded as data and pinned by 14
  sequences written GREEN against the old chains first.
  `fetchLlamaVersions`'s `repository_source` partition (the second identity
  system) reads descriptor `repoSource` fields (llama.cpp keeps untagged
  rows via `untaggedSourceOwner`). `stores/engines.js` is now free of
  per-engine if-chains.
- **EnginesView slice 1 (dispatch convergence)**: `openEngineModal`'s
  7-branch if-chain is a declarative dispatch table (panel family
  generated from `SG_PANEL_ENGINE_IDS`; the llama/ik branches' missing
  version-list refresh preserved as a real divergence). The task-completion
  handler's two venv branches merged into one `engineIdForManager()` branch
  with the toast-detail/life/status-fetch divergences kept byte-identical
  as data, and the unsloth toast label now derives from the registry.
  Pinned by a dispatch + cross-absence matrix and byte-exact toast
  sequences written GREEN against the old chains first
  (`EnginesView.tasks.test.js`). `engineIdForManager`/`engineLabel` are
  `__proto__`-safe (`Object.hasOwn`). Per-family DIALOG BLOCKS (~390
  template lines) remain — slice 2.
- **EnginesView slice 2 (venv dialog pair merged)**: the near-duplicate
  LMDeploy and 1Cat-vLLM dialog header+body blocks (≈250 template lines)
  are ONE parameterized template pair driven by a `VENV_DIALOG_CONFIG`
  record (state refs, update link/label/tooltip, install-panel copy, V100
  note, handler closures) plus registry fields `versionsKey` and
  `dialogIcon` (1Cat's header icon pi-bolt intentionally differs from its
  picker icon — carried as data). Render-level pins (banner href/label
  bytes, install-button labels, note cross-absence, check flow) were
  written GREEN against the duplicate blocks and pass UNCHANGED after the
  merge. Remaining dialog blocks: llama/ik/unsloth, sglang panel trio,
  audio.
- **EnginesView slice 2b (panel-trio headers merged)**: the
  sglang/sglang_v100/vllm dialog HEADERS joined the merged header block
  via `VENV_DIALOG_CONFIG` entries marked `panel: true` (title/icon from
  registry `dialogIcon`; settings aria-label is `Install settings` vs the
  venv pair's `Build settings`; Reload/Rescan buttons carry **no**
  aria/tooltip — `buttonA11y: false` keeps that divergence byte-identical;
  actions drive the `SglangEnginePanel` refs). Bodies untouched (already a
  shared component). Header render pins (title bytes, icons — incl. vllm's
  `pi-sparkles` ≠ picker `pi-server` — settings label variants, aria
  presence/absence) written GREEN against the five old header blocks
  first, UNCHANGED after the merge. Dialog bodies remaining: llama/ik/
  unsloth and audio (different shapes); vllm dialog coverage in the test
  harness was previously impossible (mock store lacked `vllmVersions`/
  `vllmStatus` — fixed, latent gap disclosed).
- **Backend native adapters (road 2)**: the three llama-family natives
  joined `LIFECYCLE` with per-verb `verbs` override maps — no installer
  machinery invented. llama.cpp/ik_llama implement check-updates (legacy
  `response_model` filtering byte-replicated on the generic surface),
  build-settings GET/PUT, install (= legacy `/update` build-latest flow),
  install-source (= shared `build-source`, `repository_source` default
  quirk carried through unchanged) and cancel (= `build-cancel`); unsloth
  implements check-updates/install/cancel only. Overrides delegate to the
  legacy route FUNCTIONS themselves, so both surfaces stay identical by
  construction; the shared `check_updates_surface` helper keeps the legacy
  route on the same code path. Verbs the natives genuinely lack answer
  **405** with `"{verb} is not available for engine '{id}'"` — pinned,
  never faked (no per-engine status: versions ride the shared version
  list; deletion is version-scoped). Required-body 422s are replicated via
  `_require_json_body` because the generic route envelope is body-optional
  (documented edge: legacy distinguishes absent vs null body, the generic
  guard reports both as `missing`). 34 new pins in
  `tests/test_native_engine_lifecycle.py`; the v1 equality suite renamed
  its coverage test deliberately (registry = venv 5 + natives 3; audio
  still 404). Frontend NOT flipped: the store still calls legacy native
  URLs; a `genericApi` decision for natives is a separate tranche.
- **Native read-verb flip (frontend)**: `checkLlamaCppUpdates`,
  `checkIkLlamaUpdates` (the `?source=` query dies — engine rides the
  path), `checkUnslothLlamaUpdates`, and the llama-family
  `fetchBuildSettings`/`saveBuildSettings` now hit
  `/api/engines/{id}/…` via registry `NATIVE_GENERIC_VERBS` +
  `nativeGenericApiPath` (client-side capability mirror; builds a URL only
  for verbs the backend actually serves, else throws at dev time).
  `fetchBuildSettings` keeps the legacy `?engine=` route as fallback for
  non-native ids (pinned), so the store is a strict superset of old
  reachability. WRITE verbs (`update`, `build-source`, `build-cancel`,
  unsloth install/cancel) deliberately stay legacy — flipping them is a
  separate verb-by-verb decision. Baseline evidence: the flip pins were
  first run RED against the old code (failure output recorded the exact
  legacy URLs) before the store change made them green; write-verb and
  fallback pins ran green BEFORE and UNCHANGED after. 5 new store tests.
- **Native write-verb flip (frontend)**: `updateEngine` (→ `install`,
  engine id added to the path while the body keeps its `engine` field —
  `update_engine` still reads it there, so identical bytes reach the same
  function), `buildSource` (→ `install-source`, engine id derived from
  the payload's `repository_source` with the llama.cpp default quirk
  preserved: absent source builds llama.cpp on BOTH surfaces),
  `cancelSourceBuild` (→ `llama_cpp/cancel` — llama and ik share one
  BuildTaskManager and one cancel delegate, so the engine id is routing,
  not behavior) and unsloth install/cancel all ride the generic router
  now. Baseline pins (exact legacy URLs, payloads, and the post-update
  version refresh) ran GREEN against old code, then were updated
  DELIBERATELY to the generic URLs (payload bytes unchanged); the
  engines.test.js `cancelSourceBuild` pin updated the same way. Legacy
  routes remain live as rollback. STILL legacy on purpose:
  task-notification cancel routing (`useTaskCancelEndpoint` manager/type
  maps — pinned by the taskFixtures suites) and the version-scoped
  sync/retry/activate/delete + verify/capabilities family; audio stays
  on its own router.
- **Native task-cancel endpoints (frontend)**: `useTaskCancelEndpoint`
  now resolves llama_cpp/ik_llama build tasks (metadata.engine) and the
  unsloth_llama manager through registry
  `NATIVE_CANCEL_ENDPOINTS_BY_ENGINE` → generic `/api/engines/{id}/cancel`
  (byte-equality already backend-pinned). The `type: build` legacy
  fallback stays for pre-registry task shapes WITHOUT engine metadata
  (new fixture pins that); the hard-coded `unsloth_llama` legacy entry is
  gone, replaced by the registry-derived map. Fixtures updated
  deliberately (2 llama builds → generic) and extended (ik_llama build →
  ik_llama cancel, unsloth install → generic, metadata-less build →
  legacy). cuda and audio.cpp cancel endpoints are untouched (different
  backends). Baseline: the pre-change fixture suites ran green (prior
  run), then pins moved DELIBERATELY; all fixture-consuming suites green
  with the new endpoints (591 total, +16 across those suites).
- **Cleanup/audit pass**: export-vs-usage scan over `engines.js` (84
  exports) and `registry.js` (21 exports). Pruned zero-reference store
  wrappers (`fetchCudaLogs`, `cancelAudioCppBuild`, `applyLaunchChanges`,
  `clearRuntimePlanState`, and the unused `fetchGpuInfo` export — the
  function itself stays, used by `fetchSystemStatus`); demoted
  `ENGINE_ID_BY_MANAGER` to an internal registry const (`engineIdForManager`
  is the hardened public accessor). KEPT deliberately: four test-only
  store functions (`cancelSourceBuild`, `cancelUnslothLlama`,
  `removeLmdeploy`, `removeOnecatVllm`) — public API with backend
  byte-equality pinned and store-parity pins, no production caller today;
  `legacyVenvApiPath`/`VENV_ENGINE_IDS`/`VENV_DESCRIPTORS` exports (pinned
  rollback surface); `engineDescriptor` (public accessor used by
  `engineBuildHint`, slated for dialog-body merges). Registry header
  documents the native routing mechanism. Allow-lists re-verified: all
  11 prettier and 44 ruff entries live and formatted, ratchet no-growth,
  openapi diff zero. Full suite green after pruning (591) — nothing
  referenced the removed code.
- **EnginesView slice 2c — native llama-family dialogs merged**: the
  llama.cpp/ik_llama/Unsloth dialog headers AND bodies (six raw per-engine
  template blocks) became ONE header + ONE body block driven by a new
  `NATIVE_DIALOG_CONFIG` + `nativeDialog` computed (same shape as
  `VENV_DIALOG_CONFIG`). Baseline first: two new pin tests went GREEN
  against the raw blocks (engine-mark leading, unsloth's MISSING settings
  action, full aria on all three, per-engine CTA hints, install-button
  sets, banner latest-version RAW — no `v` prefix, unlike the venv pair —
  ik's `View commit` link label, per-family update tooltips, the raw
  `<p class="build-note--info">` note on unsloth, VersionTable
  `can-verify` (llama/ik) vs absent (unsloth), distinct empty-messages),
  then passed UNCHANGED after the merge. Deliberate micro-decisions, all
  render-identical: unsloth's `onOpenSettings` is a no-op closure (its
  settings slot never renders); llama's `update-tooltip` moves from
  prop-default to the identical explicit string; VersionTable's `@verify`
  listener is bound for all three (unreachable when `canVerify` is false);
  the merged `:can-verify` uses real Boolean prop semantics so `false`
  omits the attr exactly like the omitted original. Test-side: CTA stub
  gained `data-hint` pass-through; the VersionTable stub declares
  `canVerify`/`emptyMessage` props and the pins assert on PROPS (stronger
  than attr sniffing). Pins 591 → **593**, green before and after.
- **Review-response tranche (path identity + executor cleanup)**: closed
  external-review findings #1/#3; the handoff's remaining-work scope was
  corrected (finding #4) in the same pass. (a) GENERIC PATH IDENTITY:
  native `install`/`install_source` verbs no longer trust the body —
  `_native_install_body`/`_native_source_body` derive the target from the
  URL PATH, normalize absent/agreeing `engine` and `repository_source`
  fields to the path engine, and reject conflicts with a byte-pinned 400
  (`Engine identity conflict: path engine '…' does not accept body …`).
  This is a DELIBERATE divergence of the generic surface from the legacy
  llama.cpp-default quirk — the legacy routes keep their behavior
  byte-equal (paired pins record both sides). RED-first: 8 new pins
  recorded the old wrong-engine/propagation bytes before the fix. Native
  lifecycle pins 34 → 44. (b) EXECUTOR CLEANUP GUARANTEE: `execute_plan`
  previously let non-`PrimitiveExecutionError` handler failures bypass the
  teardown pass entirely. v1 fix (convert to step failure) was REJECTED
  by the suite: 20 shadow/characterization tests contract that foreign
  exceptions propagate UNWRAPPED (legacy callers own wrapping). The
  shipped fix preserves both: foreign errors run the teardown pass
  (teardown-internal foreign errors no longer strand later cleanups),
  then the FIRST foreign error is re-raised verbatim — cleanup
  guaranteed, propagation byte-identical. StepOutcomes for foreign
  failures carry `TypeError: …`-style text.
- **YAML campaign S1 — definition tiers + digest groundwork**
  (behavior-neutral by design; the unchanged existing gates ARE the
  baseline): `EngineDefinition.definition_digest` = canonical sha256 over
  the FULL validated definition (recipes included — install snapshots must
  freeze the effective recipe; host-dependent `source_path` excluded).
  `definition_tier()` classifies generated vs authoritative as pure data
  — acceptance policy UNCHANGED (authoritative engines still refuse
  startup; the refusal + exporter byte pins pass unchanged). New
  additive ledger `install_snapshots.yaml` (`DataStore.
  record_install_snapshot`/`list_install_snapshots`): generic install /
  install-source dispatchers record definition digest + recipe version +
  sanitized resolved inputs (confirmation tokens stripped) AFTER the
  response is produced; bookkeeping is best-effort and pinned inert
  (a ledger explosion cannot change the install response). The cutover
  allow-list gained `backend/routes/engine_versions.py` deliberately
  (bookkeeping reader, steers nothing). Curated ruff list 44 → 46
  (new test file + `data_store.py`).
- **YAML campaign S2 — startup acceptance inversion** (the one
  deliberate behavior move; legacy bytes recorded RED-first in
  `test_engine_definitions_authoritative.py`). `register_engine_spec`
  adds AUTHORITATIVE definitions to `ENGINE_REGISTRY` at startup —
  idempotent on identical specs, refuses (never overwrites) on any
  disagreement; derived id-sets (`VALID/EMBEDDINGS/NATIVE/GGUF/
  HF_SNAPSHOT_ENGINE_IDS`) became SHARED mutable `set`s updated IN
  PLACE so `from`-import holders observe registrations. Generated-tier
  drift/missing refusal message is byte-pinned unchanged; same-id
  conflicts refuse through it. Exporter `--check`: extra YAML files are
  authoritative sources (loader-validated) instead of `unexpected:`
  drift, unless their id claims a generated engine; new test-only
  `--definitions-dir` (refused without `--check`). Startup log line
  reworded (no longer "shadow mode" — reads now exist); smoke marker
  convention is now `grep -c "Engine definitions validated"` = 1.
- **YAML campaign S3 — data-driven lifecycle + generic installer.**
  Optional strict `lifecycle` definition section (v2-only; the nine
  generated YAMLs untouched — defcheck byte-green).
  `backend/engines/python/generic.py` `GenericVenvInstaller`: one
  pip-wheel-persona installer shaped entirely by the definition
  (identity/labels/packages/venv roots from `lifecycle`, install flows
  execute the definition's own recipes through the SAME executor +
  shared primitives the cut-over engines use; persona-external
  primitives fail loudly, never silently). `build_definition_
  lifecycle`/`register_definition_lifecycles` (startup, after tier
  checks; Python registration always wins; no legacy-prefix routers —
  authoritative engines live on `/api/engines/{id}` only, deliberate
  boundary). `register_engine_install_settings` feeds build-settings
  defaults from lifecycle data (hidden seam found by the surface audit:
  `coerce_install_settings` refused unknown ids). Tier anchor moved to
  `GENERATED_ENGINE_IDS` (import-time snapshot) — S2 registration had
  made membership-based tiering stale. Pins +8 incl. offline
  end-to-end release install (stubbed venv/pip, real executor).
- **YAML campaign S4 — `yamlcraft` fixture ships.** The definitions dir
  now carries one AUTHORITATIVE file (hand-authored, header states its
  fixture nature): startup registers it live (`…(9 generated
  registry-equivalent, 1 authoritative)` + `Registered YAML-onboarded
  engines: yamlcraft`), and `GET /api/engines/yamlcraft/status → 200`
  on the boot smoke — an endpoint whose ONLY origin is YAML. Unknown
  ids still 404. G5 guard: the literal `yamlcraft` is banned from all
  non-test, non-fixture, non-doc sources (pytest-enforced scan; adding
  the engine ever required zero code edits — that's the assertion).
  Fixture honesty: `artifact_formats: [fixture_only]` keeps it out of
  safetensors/gguf compatibility inference and the model catalog
  (full-suite runs caught that leak pre-fix; five count/identity pins
  became order-proof against lifespan-time registration, incl.
  `LIFECYCLE` bounded assertion). No legacy-prefix router, no
  remove_version recipe (persona boundary), no launch path (runtime
  campaign owns it).
- **YAML campaign S5 — descriptor-driven frontend lookup (campaign
  closes).** `GET /api/engines` adds `generic_lifecycle: true` +
  `generic_verbs` for AUTHORITATIVE engines ONLY — the nine generated
  descriptors stay byte-identical on the wire (pinned; live smoke shows
  exactly one occurrence). `registry.js` gains the additive
  server-backed overlay: `applyServerDescriptors()` refreshes
  label/cancel lookups from the payload, `serverBackedDescriptor()`
  synthesizes store descriptors (`status:<id>` dynamic refs, manager ==
  engine id), and the store's generic venv helpers (ALL verbs) route
  server-backed ids through `/api/engines/{id}/{verb}` with dynamic
  `serverStatuses`/`serverVersions` maps. The static nine-entry tables
  are UNCHANGED — every pre-existing frontend pin passes untouched
  (601 total). Card-grid data-ification, ModelConfig's venv-coercion
  list, and hardware-filter special cases stay curated (presentation
  concerns; disclosed boundaries).
- **YAML campaign S6 — dynamic engine card closes the Engines-page gap
  (review-driven slice).** `GenericEnginePanel.vue` renders card +
  manage dialog (release install w/ check-updates prefill, source
  install prefilled from DECLARED defaults, live operation + Cancel,
  version list, settings save, confirmed Remove) for ANY
  `generic_lifecycle` descriptor; the payload-driven v-for section is
  appended to the curated nine-card grid, which stays untouched.
  Proven twice: 9 vitest interaction tests, and the FULL HTTP journey
  offline through the real generic router
  (`test_engine_definitions_journey.py`: settings roundtrip → recipe
  install → status poll to ready → ready version row → digest ledger
  row → 400/404 contracts). Scope correction recorded in the campaign
  doc: control plane complete ≠ onboarding complete — launch/
  discovery, snapshot-guided retries, specialized installers, audio
  migration, and end-to-end inference remain open.

**Survey facts, later-tranche status**: the only remaining per-engine
dialog BODY in `EnginesView.vue` is the audio.cpp block (a different
shape from every other family, flag-gated/drift-aware — see §7);
llama/ik/unsloth headers AND bodies merged in slice 2c (bullet above).
The venv pair
merged in slice 2 and all five venv-family headers in slice 2b (bullets
above). The 7-branch
`openEngineModal` chain and the
task-completion manager chain converged in slice 1 (bullet above). The
former survey items `ENGINE_LABELS`, `fallbackEngineOptions`,
`MANAGER_CANCEL_ENDPOINTS`, `SGLANG_ENGINES`, `CMAKE_ENGINES`,
`inferredEnginesByFormat`, the `repository_source` second identity system,
and the store's versionId substring chains all CONVERGED onto the registry
in v3 and its close-outs (bullets above). The server descriptors from
`/api/engines` already
drive ordering/labels/gating — the registry is the client half, now complete
for identity and lifecycle fan-out.

## 9. Honest boundaries

- This is a **survey + typing** tranche. **Zero runtime change, zero cutover.**
  The models are imported only by their test; no route uses them yet.
- Line numbers cite the tree as of this writing and will drift.
- The llama build/cuda extension surfaces are **documented, not typed** —
  deliberate (§5), to avoid ceremony with no sharing payoff.
- `capability_delta`, `affected_models`, and audio descriptor internals are
  Dict-typed inside the models (engine data), not exhaustively modeled.
- The `test_engine_contracts.py` pins guard the **typed fields** with
  `extra="allow"` models; they do **not** assert that producers emit *exactly*
  those keys — exhaustiveness is this document's responsibility and can drift.
- The check-updates **tagged union** is intentionally **not** shipped: the five
  producers share no discriminator, so a union would require inventing one
  (§7). Ship the variants; defer the union.
