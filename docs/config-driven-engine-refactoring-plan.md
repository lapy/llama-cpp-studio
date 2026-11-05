# Configuration-driven engine integrations

Date: 2026-11-01
Status: proposal; no runtime implementation changes

## Objective and scope

Add an engine by adding one validated YAML definition, without adding an installer class, runtime compiler branch, API router, frontend component, or engine-ID allowlist. Preserve every currently supported installation, version lifecycle, model configuration, discovery, and inference feature.

This proposal follows the current working tree, including the native runtime migration described in `studio-native-refactoring-plan.md`. It does not restore the external proxy being removed there. Preserve the native equivalents of capabilities, aliases, authentication, process supervision, and launch manifests. Some existing names/comments still refer to llama-swap; those are not a reason to retain obsolete infrastructure.

YAML can describe combinations of implemented behavior. It cannot implement an arbitrary future protocol or toolchain without executable code somewhere. The concrete commitment is: all nine existing engines, and new engines using their supported primitives, require only YAML. A genuinely new mechanism requires one reusable primitive, never a new engine class. Arbitrary Python import hooks or embedded scripts must not become the normal configuration interface.

## Current implementation inventory

The nine registered IDs are defined in `backend/engines/registry.py`, which is already a declarative, data-only source: each frozen `EngineSpec` carries `install_kind`, `runtime_kind`, `scanner_kind`, active-path fields, artifact formats, tasks, modalities and maturity surfaces, and the API payload already emits its own `schema_version`. The gap is coverage and fidelity, not the declarative shape — YAML definitions should replace the literal map while retaining this kind-dispatch pattern. Installation language, distribution mechanism, model format, and inference protocol are independent axes; a single `python` versus `cpp` discriminator is insufficient.

| Engine | Installation and version sources | Execution and integration | Features requiring explicit preservation |
| --- | --- | --- | --- |
| `llama_cpp` | Native source/CMake workflow; GitHub refs and release tracking | Active `llama-server` binary; GGUF file or HF repository; native launch manifest | Build option discovery, CUDA/toolchain checks, configurable architectures/threads, patches, source workspaces/sync, shared-library paths, vision projector, draft model/speculation, embeddings, parameter scanning |
| `ik_llama` | Related native source/CMake workflow with separate repository and settings | llama-server family with version-specific flags | Fork identity, IQK support, independently scanned parameters and build options, independent active installation |
| `unsloth_llama` | Published GitHub release archives; manifest/filename resolution; host-aware CPU/CUDA selection | Same llama-server runtime family | Checksums, safe extraction, GPU architecture coverage, CUDA compatibility, release ordering and upstream build tracking; do not invent a source installation action |
| `lmdeploy` | PyPI release or Git checkout into a versioned venv | `lmdeploy serve api_server <HF-id> --server-port <port>` | Positional model argument, CLI resolution, package metadata/version detection, source retry/sync, install defaults |
| `vllm` | PyPI or editable source checkout into a versioned venv | Venv Python, `-m vllm.entrypoints.openai.api_server`, `--model`, `--port` | Python version constraints, managed CUDA build/runtime environment, embeddings/reranking and multimodal metadata |
| `1cat_vllm` | GitHub release wheels or source-built wheels into a venv | Same vLLM module family | Current bundled wheel and older two-wheel releases, CUDA 12.8 PyTorch index, SM70, requirement-file layout fallbacks, Rust discovery/bootstrap and writable toolchain homes |
| `sglang` | PyPI or editable `python/` source package into a venv | Language module plus a separate diffusion callable entrypoint | Separate parameter catalogs and availability gates per runtime; retain inactive runtime settings when switching |
| `sglang_v100` | Specialized source recipe into a managed venv | SGLang language/diffusion profiles with SM70 environment; numeric CUDA UUID-to-index mapping is applied by `proxy/launch_spec.py`, not the installer | Managed CUDA 12.8, compiler selection, memory/cgroup-aware build concurrency, upstream installer adaptation, metadata/IPC/Marlin patches, dependency preservation on retry, dependency pins and smoke checks |
| `audio_cpp` | Native CMake build; compatible official release archives with source fallback | Server + CLI + model-manager artifacts; task/model-aware runtime and sidecars | CPU/CUDA/Vulkan and portable selection, all-host-GPU compatibility, build capabilities, catalog/loaders/inspect discovery, model preparation and builtin models, model profiles, contract provenance/drift, activation gates, voices, request defaults, multipart/streaming/audio conversion and task routes |

Cross-cutting behavior already exists and should be reused:

- `protocol.py` exposes an installer-only protocol. `adapters.py` selects only the five Python installer variants; it is not a complete engine abstraction.
- `python/installer.py` implements venv paths, state, logged subprocesses, pip, cancellation, progress, pending/ready/failed rows, and source synchronization.
- `lifecycle.py`, `activation.py`, and `build_workspace.py` handle persisted versions, retry, stale tasks, orphan discovery, activation, and source workspaces. The adjacent responsibilities live elsewhere and must be located correctly during extraction: launch-reference holds and retirement are implemented in `utils/fs_ops.py` over `proxy/manifests.py` generation bookkeeping, and cancellation lives in `backend/operations/` (`cancellable.py`, `cancel.py`) plus `task_cancel_registry.py`, not inside these engine modules.
- `scan/scanner.py`, `scan/help_parsers.py`, and `params.py` supply version-specific CLI catalogs. Scanner selection is already kind-keyed, not engine-ID-keyed: `SCANNER_KIND_DISPATCH` chooses parsers from `EngineSpec.scanner_kind`, which is the same probe-name mechanism the definitions plan proposes — extend it rather than invent it. The persisted parameter catalog is already versioned (`engine_params_catalog.yaml` at schema version 2), as are launch manifests and `launch_spec.py`; reuse this schema/version machinery instead of adding a parallel one. Audio also has model-specific fingerprints and composed integration contracts.
- `proxy/launch_spec.py` already provides structured `LaunchSpec`, `EnvSpec`, file identities, revisions, and preview compilation. Retain these contracts and the native launcher/supervisor.
- Hardcoded choices remain in `routes/llama_versions.py`, individual version routers, `venv_install_settings.py`, startup registration, model validation/catalog selection, environment/GPU policies, and source identity/path inference. Additional verified sites the extraction must consume, not merely discover: `proxy/commands.py` (`CAPCOMPAT_ENGINE_IDS`, per-engine runtime-context keys, `infer_engine_id_for_binary`), further `proxy/launch_spec.py` branches and CPU-rejection/GPU-only sets, `proxy/studio/tokenization.py` engine frozensets, `routes/models.py` audio profile/fingerprint/voice branches, `audio/model_config.py` and `audio/voice_presets.py` gating, `services/audio_model_installer.py` model records hardwired to `engine: audio_cpp`, `services/upstream_versions.py` llama/ik update dispatch, `models/config.py` default-engine-for-format mapping, `routes/status.py` runtime-type fallback, `feature_flags.AUDIO_CPP_ENABLED`, and `main.py` per-engine route includes plus hardcoded engine data-directory names. The phase-1 checklist must enumerate every quoted engine-ID literal outside `backend/engines/` — the worst offenders carry dozens each — or later parity claims are unverifiable.
- Scale baseline for the gates that follow: `backend/engines/` is roughly 21k lines across 63 modules (audio_cpp 5.7k, scan 4.0k, llama_cpp 3.6k, vllm 1.4k, sglang 1.3k, unsloth_llama 1.0k, lmdeploy 0.5k), and functionally part of the same system but located elsewhere are `backend/operations/` (~2.4k), `build_progress.py` (1.2k), `build_workspace.py` (~1.0k, rpath isolation and atomic snapshot publication are correctness-critical subsystems, not simple primitives), `cuda_installer.py` (2.4k), `utils/fs_ops.py`, and `proxy/launch_spec.py` (1.1k). Existing automated coverage is approximately 1,276 backend tests, 420 frontend unit tests and 57 browser tests.
- `frontend/src/stores/engines.js`, `views/EnginesView.vue`, and model configuration/composables still contain engine-specific state, actions, cards, and runtime choices. The magnitude is roughly 711 engine-ID literals across 23 frontend files: `EnginesView.vue` (~5,300 lines) renders nine hardcoded card blocks and per-engine modals rather than a loop; `engines.js` keeps per-engine version/status refs and ~50 engine-named actions with bespoke URLs, including substring engine-ID dispatch chains in activate/delete; `useTaskCancelEndpoint.js` maps each manager's cancel endpoint; `ModelConfig.vue` hardcodes the engine option list and per-engine branches. Partial precedents to extend exist — the generic `/api/engines` descriptor list, the typed build-options catalog rendered by `opt.type`, and audio's contract-driven `AudioParamField.vue` — but no schema-form machinery exists. Backend YAML alone will not make onboarding configuration-only.

## Target architecture

### Immutable definitions, observed capabilities, mutable installation state

Keep three distinct documents:

1. **EngineDefinition**: repository-owned `backend/engines/definitions/*.yaml`; schema version, identity, sources, recipes, artifacts, runtime profiles, discovery, compatibility, protocol selection, settings, and presentation.
2. **EngineInstallation**: existing user state in `data/config/engines.yaml`; resolved version/ref/commit, artifact paths, package versions, build settings, task state, and errors are recorded today as per-engine version rows with `build_status`. A stable installation ID and original-engine-ID field do not exist yet — they are new fields added by a state migration, not data to preserve verbatim. Preserve existing paths and rows during migration.
3. **EngineObservation**: discovered parameters, supported runtime profiles, model contracts, capability provenance and fingerprints, tied to an installation and optionally a model. Audio's `audio_cpp/integration/` composer and contract registry already implement this observed-over-declared merging with provenance and drift today; generalize that working mechanism instead of inventing a new one.

A definition declares potential support; the effective descriptor intersects it with installed artifacts, hardware, scans and model evidence. Never turn a declared `safetensors` format or task into verified support for every model. Preserve the registry's conservative audio compatibility inference.

Load and validate definitions once at startup into immutable typed objects. Reject duplicate YAML keys/IDs, unknown fields/primitives, missing references, inheritance cycles, invalid condition types and ambiguous artifact selectors. Definitions are application configuration, not mutable state. Inheritance uses explicit templates with deterministic map merging and list replacement; avoid implicit append/override behavior.

Record the resolved definition digest, recipe version, and resolved install inputs on every new installation. Persist the effective runtime profile identity/digest in launch metadata so definition changes participate in revision calculation. Reuse existing revision/file identity machinery. Existing installations without snapshots resolve through an explicit legacy compatibility mapping; changing YAML must not silently reinterpret their install layout.

### One adapter and several reusable services

`ConfiguredEngineAdapter(definition, services)` exposes a uniform application interface:

- `describe(context)` and `validate_model(model, config)`
- `plan_install(request)`, `install(plan)`, `retry(installation)`, `sync_source(installation)`
- `check_updates(installation)`, `status()`, `logs(task_id)`, `cancel(task_id)`
- `discover(installation, model=None)`
- `activate(installation, policy)`, `verify(installation)`, `remove(installation, reference_policy)`
- `compile_launch(installation, model, config) -> LaunchSpec`

Unsupported actions return a typed unsupported-action result and are omitted from the UI. They must not exist as fake success/no-op methods. Task-starting methods retain task IDs and asynchronous operation semantics.

The adapter composes services rather than subclasses:

| Service | Responsibility |
| --- | --- |
| Definition loader | Typed YAML validation, template expansion, immutable registry/API descriptors |
| Version resolver | PyPI, GitHub releases/manifests/refs, source commit identity and comparison policy |
| Recipe planner/executor | Ordered typed steps, dependency outputs, conditions, progress, cancellation, artifact verification |
| Lifecycle coordinator | Persistence, admission/recovery, locks, activation, references, retry, deletion and orphan recovery |
| Discovery service | CLI help and JSON/inspect probes, parser selection, cache fingerprints, provenance and drift |
| Launch compiler | Entrypoints, model bindings, parameters, cwd, environment, GPU policy and existing structured manifests |
| Protocol/model services | Model sources/preparation and inference contract handlers, independent of engine identity |

Avoid replacing bespoke classes with one giant adapter full of engine-ID switches. A primitive receives typed inputs and outputs and cannot dispatch on an engine ID.

### Installation recipes

Initial primitive vocabulary should cover current behavior:

- Resolve version/ref/release assets; inspect host OS/architecture/CPU features/CUDA and **all** visible GPU compute capabilities.
- Create/reuse managed workspace, clone/sync checkout, apply checked patches, preserve incompatible dependency checkout, create directories.
- Select Python/compiler/toolkit/Rust; create venv; build dependency environment; select memory/cgroup-aware concurrency.
- Install pip requirements/packages/wheels, build wheels, CMake configure/build, download/verify/extract archive.
- Resolve artifact candidates, capture package/git metadata, smoke test, publish installation outputs.

Conditions are a small typed vocabulary (`all`, `any`, equality, version range, existence, capability containment), not Python/Jinja evaluation. Expressions reference known request/context/step outputs. Commands use argv arrays, explicit cwd/environment, accepted exit codes and timeouts. There is one process runner for logs, cancellation, process-group termination and progress events.

Separate release and source recipes. Retry uses the failed installation's recorded recipe/inputs and safe reuse policy. Sync uses its recorded source/fork/ref and settings. Asset fallback must be declared: audio can build the selected release tag when no compatible archive exists; Unsloth must preserve its current supported behavior rather than inherit audio's fallback.

SGLang V100 is the extraction stress test. Convert source rewrites into versioned patch assets with applicability checks and descriptive failures; represent compiler selection, managed prefixes, dependency preservation and upstream installer invocation explicitly. Temporary legacy strategy references can support migration, but an engine-specific callback/script hidden behind a YAML key is not completion. Before deleting the old installer, every rewrite and recovery path must have a declared replacement and parity evidence.

### Runtime profiles and parameter discovery

Each definition contains named profiles. A profile declares executable resolution (native artifact, venv console script, Python module or Python callable), fixed argv, positional/flag model binding, port binding, cwd, owned flags, parameter catalog and readiness policy.

Model source strategies cover GGUF/HF fallback, HF identity, projector/draft companions and prepared/builtin audio bundles. Generic binding operators can bind optional companions without copying GGUF launch logic per engine.

Keep the existing parameter catalog as the authority for accepted options, aliases, value types, boolean/list serialization and version-specific flags. YAML supplies scan probes/parser names and explicit overrides. Preserve raw help output, failure diagnostics, rescans and cache invalidation. Profiles can probe independently; an unavailable diffusion probe does not erase language support. Runtime selection filters emitted parameters while retaining saved inactive settings.

Environment policy describes PATH/library composition, defaults, required values, allowed unsets and precedence. GPU policy declares CPU support, UUID versus numeric device identity, toolkit requirements and parallelism keys. Preserve user overrides, selected GPU order, inherited visibility semantics and conflicts detected by the current compiler. Preserve custom argument parsing and owned-flag rejection.

Health/startup/shutdown behavior belongs to the runtime/protocol profile, including actual endpoint/status expectations and timeout budgets. The existing native process owner remains responsible for readiness, crash recovery, termination, streaming and admission. Definition readiness policies must be written against the owner's actual semantics — health-endpoint polling with a bounded budget, exit observation with a recorded failure reason, idle reaping, lazy reload on the next admission rather than eager restart, and startup journal reconciliation — not an assumed restart-on-crash model.

### Protocols, models, and audio

Reuse protocol handlers such as OpenAI HTTP forwarding and audio task-contract handling. YAML selects handlers and supplies endpoint/field mappings where transformations are declarative. Streaming, multipart parsing, audio conversion, authorization and lifecycle ownership remain tested shared code.

Generalize audio's discovery pipeline into probe/normalize/compose operations while preserving observed-over-default precedence, upstream overrides, provenance, drift warnings and model-specific fingerprints. Model preparation, catalogs, builtin artifacts and voices are separate capability services selected by definitions. An engine accepting HF models must not automatically inherit audio's prepared-bundle or builtin behavior.

The entire audio task surface remains available, including ASR, TTS/cloning, alignment, separation, generation, conversion, analysis and generic pipelines, plus task/batch APIs and request-default validation. Existing audio family logic can remain domain code; it must become independent of a hardcoded engine ID where needed for another definition to reuse it. Do not attempt to encode signal processing or arbitrary protocol algorithms as YAML.

### API and frontend

Add generic routes under `/api/engines/{engine_id}` for versions, status, settings/schema, updates, install planning/execution, verify, retry, sync, activation, deletion, scans, logs and cancellation. A generated per-engine router factory already exists: `python_engine_router()` in `routes/engine_versions.py` mounts eight near-uniform endpoints for the five Python engines, while the native families are hand-written. A single generic router can subsume that venv lifecycle plus the cross-family version operations (`/api/llama-versions/*` retry/activate/delete/sync), but not everything: llama.cpp build-capabilities and CUDA status/install/logs, audio `/contract` and `/community-voices*`, and audio's drift-aware `/status` require typed extension sub-routes declared by the definition. Response contracts also diverge today — llama.cpp `UpdateCheckResponse` versus the factory's releases dictionary versus unsloth_llama's `{url, release_url}` versus audio's `contract_fingerprint`/`capability_delta`/`drift_warnings` status — and must be unified or explicitly typed per surface before delegation. Register one generic router plus declared extensions, not further generated bespoke functions per engine. Reuse existing operation admission and recovery (`operations/action_recovery.py` keys are engine-ID-agnostic strings already reused by the shared factory); a plan is reviewable data, not an extra mandatory confirmation flow.

Keep existing API routes as compatibility delegates until callers migrate. Preserve response/task/error contracts and version identity during transition. The frontend contract is the checked-in `frontend/src/api/openapi.json` with staleness gates (`scripts/export-openapi.py`, `npm run openapi:check`, and the pre-commit hook), consumed by hand-written axios calls and TypeScript types rather than generated client code; re-export it when API behavior changes. Route, task, idempotency, revision-token and 409 action-admission error semantics are documented in `docs/api-reference.md`.

Replace individual Pinia refs/actions with keyed maps and generic methods. Render cards, install methods, build settings, update banners, runtime selectors, capability warnings and version actions from descriptors and typed schemas. Supply reusable field widgets and panels by capability, not engine ID. Preserve links, keyboard/accessibility behavior, progress reconnect/cancel, hardware filters and source-fork labels. Settings schemas must cover native build options as well as Python install defaults and retain advanced fields.

## Implementation sequence and exit gates

1. **Freeze the behavior inventory.** Enumerate current routes, settings/defaults, operation responses, artifact layouts, CLI/JSON probes and frontend actions for all nine engines. Build a parity checklist with one row per feature, implementation reference and existing/new verification. Resolve unknowns through source/fixtures before claiming parity. Treat unrelated working-tree changes as the baseline, not files to overwrite.
2. **Introduce schema and definitions in shadow mode.** Export registry data and install defaults to YAML; add typed loader, templates and generic descriptors. Assert descriptor/default equivalence. No installer or launch behavior changes yet.
3. **Extract executor and lifecycle services.** Reuse existing cancellable operations, progress and storage. Add install plans and typed outputs. Migrate LMDeploy first, then upstream vLLM and SGLang, preserving release/source/retry/sync/activation behavior at each gate.
4. **Extract native builds and release assets.** Migrate llama.cpp/ik build recipes and options; extract host-aware artifact selection, checksums and archives for Unsloth and audio. Retain supported source targets, existing folders and build provenance.
5. **Handle specialized source recipes.** Migrate 1Cat wheel layouts/Rust/build requirements and SGLang V100 patches, toolchains and recovery. Do not mark these engines migrated while an engine-specific class or unrestricted hook still owns their recipe.
6. **Switch launch/discovery to profiles.** Compare old and new launch specs and scan outputs for all engines, including SGLang diffusion, GPU/environment edge cases and GGUF companions. Keep the native launcher/supervisor and manifests intact.
7. **Migrate audio services/contracts.** Preserve discovery, models, voices, routing/transforms, model profiles, activation capability review and all task/request surfaces. This is a separate substantial phase, not a generic `audio: true` field.
8. **Migrate API/frontend and remaining consumers.** Move keyed state, schema forms, catalogs/model compatibility, source identity, validation, recovery, orphan roots, diagnostics and startup wiring to definitions. Retain compatibility delegates as required.
9. **Prove configuration-only onboarding and remove legacy implementations.** Add an unrecognized fixture engine ID entirely through YAML; verify descriptor/UI presence, install/update/scan/launch/lifecycle/model compatibility without edits elsewhere. Remove old maps/classes/branches only after parity gates pass. Add an architecture check against engine-ID dispatch outside definitions and explicit legacy migrations; no import-linter configuration or custom lint rules exist today, so this is net-new tooling, with `tests/test_models_routes_architecture_contract.py` as the only (text-grep) precedent to extend.

Each engine has an independently selectable migration path during development. Select the legacy or new implementation before starting an operation; never retry a failed new install automatically through the old implementation. Failed parity blocks cutover. Retain state backups and old install layouts; rollback must not require reinstalling models or deleting user files.

## Verification and definition of done

Reuse and parameterize meaningful existing coverage, particularly `test_engine_version_lifecycle.py`, `test_launch_manifests.py`, `test_engine_model_contracts.py`, scanner/catalog tests, integration tests for each engine, audio contract/profile/proxy tests, and frontend engine/build/runtime tests. Note the gaps when budgeting parameterization: ik_llama has no dedicated integration file — its coverage is spread across roughly eighteen shared files — and LMDeploy's dedicated file (`test_lmdeploy_installer.py`) holds only two tests.

Required evidence:

- Definition schema failures, deterministic expansion, supported-action derivation, invalid references and definition snapshot compatibility.
- Recipe execution using fake toolchain/process/network/filesystem fixtures: argv/env/cwd, exit handling, retries, source sync, cancellation of descendants, persistence failures, interrupted builds and logs/progress. Precedent exists — the suite-wide network guard and data-directory isolation in `tests/conftest.py`, recorded CLI help/inspect fixtures and parameter-scan dumps under `tests/fixtures/`, and per-file `Fake*` installer/store/client classes — but every file hand-rolls its own fakes; a shared fake-toolchain/process/release-payload fixture library is net-new work.
- Recorded host/release fixtures covering CPU, CUDA, Vulkan, SM70, mixed GPU architectures, unsupported platform/asset and source fallback. Include old/current 1Cat releases and V100 patch applicability failures.
- Semantic old/new comparisons of launch specs, catalogs, settings, capability/provenance contracts, route outputs and error behavior. Normalize only incidental timestamps/task IDs, never flags or environment ordering that affects behavior.
- Lifecycle scenarios: existing installation upgrade without path moves, inactive versions, reference retirement, deletion containment, orphan recovery, runtime activation failures and capability review.
- Native inference checks for streaming text, embeddings/reranking where supported, multimodal companions, runtime switching and audio multipart/streaming/tasks. Mocked tests alone cannot establish GPU compilation or inference parity; require representative hardware runs for each specialized path before retirement.
- Frontend journeys for dynamically supplied fixture engine, install/source settings, status/progress/cancel/retry/sync/activation/delete, model config, diffusion availability, audio contract review and inaccessible actions.

Completion means all nine existing definitions operate through the same adapter, all inventoried features have passing evidence, existing user state remains usable, and a new engine built from supported primitives needs exactly a YAML addition. No new engine-ID edits are allowed in application control flow, UI, validation, catalog compatibility or routing. Shared algorithmic services remain code; engine identity and composition become configuration.

This document is based on source inspection. No application tests or hardware builds were run for this planning-only change.
