# Engine refactor requirement matrix (rebaselined 2026-11-04, slice 6)

This replaces campaign scoreboards as the status of record. It maps every
substantive requirement in `docs/config-driven-engine-refactoring-plan.md` to
current tree evidence, a status, the concrete gap, an observable exit
criterion, and the next implementation step.

Historical documents (`yaml-engine-onboarding-campaign.md`,
`yaml-engine-launch-campaign.md`, `engine-refactor-handoff.md`,
`engine-surface-contract-matrix.md`, `engine-parity-inventory.md`,
`test-suite-audit.md`) are **evidence of work performed**, not proof that a
requirement is complete.

Status vocabulary:

- **complete** — requirement met with production evidence and tests.
- **transitional** — a working path exists, but an old duplicate or a
  non-mechanism implementation still exists for some engines.
- **remaining** — not implemented, or implemented only for a subset whose
  mechanism is not yet reusable.
- **exception** — deliberately out of the original scope; must not be counted
  as complete.

| # | Original requirement | Current evidence | Status | Gap | Exit criterion | Next step |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | One validated YAML definition per engine; load/validate once; definitions own registration rather than mirroring Python | `backend/engines/definitions/{schema,loader,startup_check}.py`; import-time bootstrap registers all nine authoritative definitions; no `EngineSpec` literal remains in `registry.py`; `scripts/export-engine-definitions.py` deleted; exporter tier removed entirely (`GENERATED_ENGINE_IDS` no longer exists; startup tiers are definition-owned vs accepted fixtures); startup gate verifies definition/registry equality | complete | — | Editing a definition changes production with no regeneration and no Python edit | Keep the startup gate and schema strictness as the contract |
| 2 | Reusable installation primitives with typed inputs/outputs and one process runner | Shared primitives plus new reusable mechanisms: `vcs.apply_transform` (versioned applicability-checked assets), `build.prepare_fork` (declared builder environment), `artifact.build_wheels`, `toolchain.ensure` (Rust), `prereqs.resolve_release_assets` (declared wheel groups); V100/1Cat recipes exercise them | complete for shipped definitions | Native host still has llama-specific handlers (see #4) | Every step a shipped definition uses is a documented primitive or a declared asset; no engine-specific callback map remains | Reuse the fork/wheel primitives for the next converted engine |
| 3 | Specialized V100 and 1Cat behaviour expressed by definitions/assets, not engine subclasses or hidden callbacks | `backend/engines/definitions/assets/sglang_v100/*.yaml`; `sglang_v100.yaml` declares transforms/build env/status/log/start/python; the per-engine venv modules/adapters are deleted; tests drive the assets and primitives | complete | — | Deleting the specialised class changes no behavior; failures name the asset/upstream text | Keep assets versioned with the fork |
| 4 | Native builds and release assets selectable from declarations | Native source builds run the declared `source` recipe through the shared `CmakeSourceInstaller(GenericNativeInstaller)` (`backend/engines/python/cmake_source.py`); repository name, source dir, binary name, examples, BLAS and stable-tag policy come from the new `source_build` definition section; the third `nativecraft` fixture proves a fork can be configured without a new installer subclass; audio release/source/sync/removal on the native host; Unsloth background install uses the generic release host, shared publication/activation/snapshot and YAML `release_assets`; capcompat is a declared `launch_policy` flag | complete | The CMake/toolchain helper surface is still injected by `LlamaManager`, but repository/artifact/build-policy differences are declaration data | A third native fork is configured by YAML plus the shared host, not a family subclass | Keep `nativecraft` and the source characterization pins as the regression gate |
| 5 | Launch profiles and discovery are definition-owned | `proxy/profile_launch.py`; all five venv profiles; SGLang llm/diffusion switch; native GGUF profiles declare `launch_family: gguf_llama_server` and compile through `proxy/gguf_launch.py`; the legacy `_compile_gguf` branch/function is deleted; all venv `discovery` blocks scan through `scan_definition_engine_version`; port/host reservation union | complete | Other engine families and audio still need their own representative-hardware passes (see the privately maintained hardware verification record) | Every shipped launch path is declaration-routed; GGUF document parity is pinned before deletion | Keep the characterization contract as the regression gate |
| 6 | Installation identity, definition digests, snapshots, retry and sync semantics | `EngineDefinition.definition_digest`; install snapshot ledger; shared `backend/engines/install_snapshots.py`; venv retries, native llama/ik source rebuilds, audio.cpp rebuilds and Unsloth release retries resolve frozen inputs and refuse a changed definition; `profile_digest` in launch revisions/manifests | complete | — | Every install records digest/resolved inputs and retries resolve through them | Keep the shared guard as the contract for new hosts |
| 7 | Audio discovery, model preparation, voices, contracts and sidecars preserved and service-selectable | static surfaces declared (`build_options`, `protocol`, `extensions`, `request_defaults`); `${studio_audio_config}` resolves to the real sidecar; audio install/mutation on the native host; `backend/engines/audio_cpp/identity.py` resolves the audio engine from declared `runtime_kind`; catalog provider, installer, model config/routes, voice presets, activation/tracking/runtime/proxy-routing/artifact/inspect/request-vocab modules and the versions route use that identity instead of literals; the extension router is mounted once per declared audio engine and a request-context middleware selects the owner (prefix from the engine id) | complete | Contract/voice payload assembly remains audio-domain code, which the plan explicitly permits | Another definition declaring `runtime_kind: audio_cpp` gets the audio service layer and extension prefix without code edits | Keep the capability test as the contract for additional audio engines |
| 8 | Generic API behaviour with compatibility delegates over one implementation | `routes/engine_lifecycle.py` generic router; `LIFECYCLE` records; native verb overrides; audio typed extensions; all five venv compatibility prefixes (LMDeploy, vLLM, SGLang, SGLang V100, 1Cat) are now `definition_compat_router` delegates with YAML-owned update source/settings/status facts; body/path identity pinned | complete for covered engines | audio.cpp keeps its declared extension router; some native verbs still delegate to route functions | Every engine's control-plane verbs resolve one implementation; compat modules contain only prefix wiring | Retire compat modules only with the API cutover |
| 9 | Frontend descriptors, keyed state, schema forms, runtime choices and capabilities without engine-ID control flow | `frontend/src/engines/registry.js`; keyed status/version maps; `GenericEnginePanel` renders a full install/manage journey for any server-described engine with no local descriptor; all nine shipped engine cards render from one descriptor-driven loop; the SGLang-family panel takes its validator/label/repo/project identity from descriptor `panel` data; dialog panel dispatch uses descriptor `panelFamily`/`configPanel`; LMDeploy/1Cat install-settings dialogs share `EngineInstallSettingsDialog`; audio.cpp and llama/ik build settings render through one `EngineBuildOptionsDialog`; catalog search/library options/defaults and ModelConfig engine marks derive from descriptors; SGLang runtime-selector options and the audio engine id are descriptor-derived | transitional | Some SGLang runtime-profile logic and audio presentation copy still branch on the engine family; frozen literal caps remain | A new engine needs zero frontend edits on the generic onboarding path | Move the SGLang runtime selector and audio-package branches behind descriptors |
| 10 | Legacy implementation removal (no duplicate business logic) | V100 `v100_support.py` and 1Cat installer class/Rust mixin deleted; upstream SGLang/vLLM installer classes deleted; scan adapters, lmdeploy resolver, launch compiler branches, exporter tier and the unused legacy `python_engine_router` factory removed; nine frontend engine cards collapse to one descriptor-driven loop; every venv compatibility prefix is thin `definition_compat_router` wiring; `test_engine_refactor_dead_branches.py` audits exporter reads, no venv engine modules/maps, thin compat modules, declared native hosts and removed scan adapters | complete | Native llama source handlers are active shared mechanisms, not duplicates; shipped dialog bodies are UI cleanup | No unreachable compiler/installer/scan branch remains; compatibility URLs are thin delegates | Keep the audit test as the regression gate |
| 11 | Configuration-only onboarding proven end to end | `yamlcraft` (control plane), `launchcraft` (real pip install → venv → status → discovery scan → process → health → streamed chat → stop → remove → row/venv cleanup), `launchforge` (GGUF-declaring, positional binding, venv module) harnesses; `test_vllm_shipped_engine_journey.py` proves a SHIPPED engine's YAML release recipe installs a local wheel, records snapshots, runs the declared discovery probe, compiles and launches the declared profile through `ProcessRegistry`, serves health + a streamed chat request, then runs the remove recipe and cleans the row/venv; `test_engine_definitions_journey.py` proves editing `lmdeploy.yaml` changes discovery and launch; new YAML engines appear through `GET /api/engines` + `GenericEnginePanel` | complete for the fixture personas and a shipped engine | Specialised GPU build/inference parity still needs representative hardware (documented limitation, not a configuration-only gap) | One YAML file adds an engine and the harness proves install→discovery→launch→request→stop/remove for its mechanism | Keep the shipped-engine journey as the regression gate for recipe/profile changes |

## Recorded departures from the original scope

These are **not** complete requirements and must not be reported as such:

1. **Native GGUF launch stays on the legacy compiler** (campaign decision R23).
   llama.cpp / ik_llama / Unsloth model-source composition (GGUF file vs HF
   repository, mmproj and draft companions, aliases, library paths) is still
   Python. No definition profile launches a native GGUF engine.
2. **Native snapshot-backed retries are resolved** for every shipped engine:
   llama.cpp/ik_llama source rebuilds, audio.cpp rebuilds and the Unsloth release
   installer now record the definition digest and resolved inputs, resolve
   retries through the frozen snapshot, and refuse a changed definition through
   `backend/engines/install_snapshots.py`.

Both are original-plan requirements (native builds, installation identity and
retry). They remain open scope decisions, not completed work.

## Corrections to the active handoff

`docs/engine-refactor-handoff.md` remains a historical tranche log. Where its
summaries conflict with this matrix, this matrix wins:

- **R98's frontend summary was wrong** (it claimed green while the process
  exited 1 with 14 unhandled errors). R99 fixed the cause and R100 made the
  gate report its own verdict; the gate now reports exit 0, 642 tests, zero
  unhandled errors.
- **"Phase 7 complete" was withdrawn** in R70's review corrections; the matrix
  records audio runtime services as remaining, not complete.
- **"Phase 6 done"** means only the venv launch compilers; native GGUF launch
  is a departure.
- **Tier language is historical**: the exporter tier has been removed from the
  registry (`GENERATED_ENGINE_IDS` no longer exists); all nine shipped engines
  are definition-owned and registered at import.
- The old "two of nine converted" status is stale: this rebaseline follows the
  completion of the V100 and 1Cat subclasses.

## Verification for this rebaseline

Run in the Docker dev container (`/app`, Python 3.12), plus host Node gates:

- Backend: `2180 passed, 8 skipped` (full suite).
- Frontend gate: `CLEAN`, exit 0, 660 tests passed, no unhandled errors.
- Static: ruff check/format (curated), engine literal ratchet, legacy-route
  inventory, engine-parity inventory, OpenAPI, build-defaults and version gates
  all current.
- New focused mechanism tests:
  `backend/tests/test_engine_definition_mechanisms.py` (transform strictness /
  applicability / idempotence, fork option validation, ordered release groups,
  definition load asset validation, shared snapshot retry guard) plus native
  host and audio.cpp retry-guard tests.
- The YAML-only `launchcraft` end-to-end journey now also proves lifecycle
  removal: after stop, the same definition-owned installer's `remove` recipe
  deletes the version row and venv.

## Final status summary

Completed requirements: 1 (definition-owned registry), 2 (reusable primitives),
3 (V100/1Cat boundary), 4 (native source/release from definitions), 5 (native
GGUF launch profiles), 6 (snapshots/retries), 7 (audio capability services),
8 (generic API plus thin compatibility URLs), 10 (dead-branch removal) and 11
(configuration-only install→launch→request→remove proof for fixtures and a
shipped engine).

Transitional mechanism (working, documented, not counted as complete):

- Requirement 9: catalog rendering still has engine-specific consumers;
  onboarding, cards, panels, install/build dialogs and audio build forms are
  descriptor-driven and a new engine needs no frontend edit.

Hardware evidence: native llama.cpp source build through `LlamaNativeInstaller`
and GGUF launch/inference through `gguf_launch` were run on a remote V100 with
CUDA 12.8; see the privately maintained hardware verification record. The other engine families
and audio still need their own representative-hardware passes.

Open verification defects: none known. The earlier transient failure of
`test_action_recovery.py::test_an_unknown_fence_does_not_authorize_a_start` was
caused by the test reading the row before the supervisor's queued terminal
write landed; the test now waits for the terminal status (15 stress runs plus
the full suite clean).

See `docs/engine-refactor-final-audit.md` for the ownership/compatibility
inventory and the privately maintained hardware verification record for the unexecuted GPU
verification matrix.
