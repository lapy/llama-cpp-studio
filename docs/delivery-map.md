# Delivery map: config-driven engine definitions

**Status:** one feature branch, an eight-commit dependent review series, and one
coordinated release. The frozen tree is checkpoint `1fd002e`; the review series
is reconstructed from it commit by commit. Start with the
[change inventory](change-inventory.md) for the exact file/commit split, and
[verification evidence](review-verification-evidence.md) for commands, counts
and hardware limits.

This document replaces the campaign logs as the entry point. The campaign
documents remain as historical evidence, not as the statement of what ships.

## What users gain

- **Engines are configuration, not code.** All nine shipped engines are loaded
  from validated YAML at import, so changing a definition changes production
  without an exporter run or a Python edit. The bootstrap is
  [`registry.py`](../backend/engines/registry.py) and the loaded set is
  asserted in [`test_engine_definitions_authoritative.py`](../backend/tests/test_engine_definitions_authoritative.py).
- **A new engine can be onboarded without a frontend edit.** The server
  describes its own lifecycle, and
  [`GenericEnginePanel.vue`](../frontend/src/components/engines/GenericEnginePanel.vue)
  renders install/manage/versions for any described engine. The YAML-only
  install → discovery → launch → request → stop/remove journey is
  [`test_launchcraft_journey.py`](../backend/tests/test_launchcraft_journey.py);
  the shipped-engine equivalent is
  [`test_vllm_shipped_engine_journey.py`](../backend/tests/test_vllm_shipped_engine_journey.py).
- **Retries cannot silently reinterpret an old install.** Each successful install
  records a version-associated snapshot (definition digest plus resolved inputs)
  at publish — venv, native release, native source and Unsloth — and each
  attempt also records a pending snapshot against its pre-publication row name,
  including audio source/sync builds, so a FAILED install still refuses a retry
  after the definition changed. Installation identity is kept separate from the
  resolved package version: a timestamp-pip row name is never passed to pip as
  a version. The digest covers the full validated definition and the content of
  referenced transform assets. An unreadable ledger raises instead of silently
  disabling the guard; explicit version removal prunes that version's snapshots,
  while rollback does not. The guard is
  [`install_snapshots.py`](../backend/engines/install_snapshots.py), exercised by
  [`test_engine_definition_mechanisms.py`](../backend/tests/test_engine_definition_mechanisms.py),
  [`test_audio_cpp_versions_api.py`](../backend/tests/test_audio_cpp_versions_api.py),
  [`test_nativecraft_journey.py`](../backend/tests/test_nativecraft_journey.py)
  and the host-specific retry tests.
- **One control plane, same URLs.** The generic
  `/api/engines/{engine_id}/{verb}` router runs the same callables as the legacy
  per-engine prefixes; equality is pinned payload-for-payload in
  [`test_generic_engine_router.py`](../backend/tests/test_generic_engine_router.py).
- **Launch revisions name the definition that produced them — for profile-driven
  engines.** Profile-driven launches (all shipped engines except audio.cpp)
  include the definition/profile digest, so editing YAML does not silently alter
  a running model; applying still requires an explicit restart or next-start
  publish. See [`profile_launch.py`](../backend/proxy/profile_launch.py) and
  [`test_profile_launch.py`](../backend/tests/test_profile_launch.py).
- **No cross-engine state bleed.** Status, versions, build options and task
  cancellation are keyed per engine in the store; the keyed reads are pinned in
  [`engineKeyedReads.test.js`](../frontend/src/stores/tests/engineKeyedReads.test.js)
  and the accessor contract in
  [`engineStoreAccessorContract.test.js`](../frontend/src/views/tests/engineStoreAccessorContract.test.js).
- **Audio is a declared capability.** An engine participates by declaring
  `runtime_kind: audio_cpp`; identity resolution lives in
  [`identity.py`](../backend/engines/audio_cpp/identity.py), and the extension
  router is request-scoped, so a second audio engine does not require a second
  copy of the service layer. Launch compilation is still audio.cpp-specific (see
  remaining work).
- **The development environment is part of the branch.** A dev container and
  compose file make the Linux/Python 3.12 environment used for verification
  reproducible ([`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json),
  [`docker-compose.dev.yml`](../docker-compose.dev.yml)).

## What architecture changed

| Layer | Before | Now | Evidence |
| --- | --- | --- | --- |
| Definition | Python `EngineSpec` literals and an exporter tier | Validated YAML is authoritative; the registry is a derived view built at import | [`schema.py`](../backend/engines/definitions/schema.py), [`loader.py`](../backend/engines/definitions/loader.py), [`startup_check.py`](../backend/engines/definitions/startup_check.py) |
| Installation | Per-engine installer classes and subclasses | Shared `GenericVenvInstaller`, `GenericNativeInstaller` and `CmakeSourceInstaller` execute declared recipes and primitives | [`generic.py`](../backend/engines/python/generic.py), [`native.py`](../backend/engines/python/native.py), [`cmake_source.py`](../backend/engines/python/cmake_source.py) |
| Install identity | Nothing tied a version row to the definition that produced it | Digest + resolved inputs ledger; retry resolution refuses drift | [`install_snapshots.py`](../backend/engines/install_snapshots.py), [`data_store.py`](../backend/data_store.py) |
| Launch | Engine-id branch in the launch compiler; native GGUF frozen in one function | Declared profiles; GGUF is a `launch_family` mechanism; revision carries the profile digest | [`profile_launch.py`](../backend/proxy/profile_launch.py), [`gguf_launch.py`](../backend/proxy/gguf_launch.py), [`launch_spec.py`](../backend/proxy/launch_spec.py) |
| API | Per-engine routers with duplicated lifecycle code | One `LIFECYCLE` record per engine; generic router + thin compatibility delegates | [`engine_versions.py`](../backend/routes/engine_versions.py), [`engine_lifecycle.py`](../backend/routes/engine_lifecycle.py) |
| Frontend | Nine parallel engine cards/dialogs and engine-ID conditionals | Descriptors, keyed state, one generic panel and shared dialogs | [`registry.js`](../frontend/src/engines/registry.js), [`GenericEnginePanel.vue`](../frontend/src/components/engines/GenericEnginePanel.vue) |

## What compatibility remains

- **Existing installations are read, not migrated.** Engine IDs, version-row
  fields (`binary_path`, `venv_path`, `install_dir`, `build_config`) and install
  directories are unchanged. The new `install_snapshots.yaml` is additive and
  is not created for old installs. A hand-written pre-refactor configuration is
  driven through list/status/activation/launch/remove in
  [`test_pre_refactor_installation_smoke.py`](../backend/tests/test_pre_refactor_installation_smoke.py).
- **Legacy URLs stay.** LMDeploy, vLLM, SGLang, SGLang V100 and 1Cat prefixes
  are thin `definition_compat_router` delegates, e.g.
  [`lmdeploy_versions.py`](../backend/routes/lmdeploy_versions.py). The generic
  router is byte-equal to them by construction and by test.
- **Native GGUF argv is characterised, not rewritten.** The shared family
  compiler is pinned against the pre-refactor composer in
  [`test_gguf_launch_characterization.py`](../backend/tests/test_gguf_launch_characterization.py).
- **Audio endpoints keep their paths and payloads.** The declared extension
  surface is still mounted under `/api/audio-cpp`; the audio contract tests are
  [`test_audio_protocol_contract.py`](../backend/tests/test_audio_protocol_contract.py)
  and [`test_audio_cpp_versions_api.py`](../backend/tests/test_audio_cpp_versions_api.py).
- **Installer imports keep working.** `LMDeployInstaller`, `OneCatVllmInstaller`
  and `SglangManager` remain as constructor-only compatibility facades; the
  architecture test pins that they define no other behaviour
  ([`test_engine_refactor_dead_branches.py`](../backend/tests/test_engine_refactor_dead_branches.py)).
- **The frontend store contract is unchanged for existing consumers.** The
  accessor contract test fails if an export or keyed accessor changes shape.

## Intentional behaviour changes

- **Startup refuses an invalid or drifted definition.** Previously a broken
  Python literal could not exist; now a bad YAML file is a startup failure
  instead of a partially understood engine. See `verify_engine_definitions()`
  in [`startup_check.py`](../backend/engines/definitions/startup_check.py).
- **A retry after a definition edit is refused.** The error says to reinstall
  rather than reinterpret the recorded layout; an install with no snapshot (all
  pre-refactor installs) is still retryable.
- **Generic native install derives the target from the URL path.** Conflicting
  `engine`/`repository_source` bodies are a pinned 400 on the generic surface,
  while legacy routes keep their historical quirk.
- **Unimplemented generic verbs answer 405.** Native engines no longer fake a
  per-engine `status` or engine-wide `remove`.
- **The audio extension router is request-scoped.** It is mounted once per
  declared audio engine and selects the owner from the request path, instead of
  assuming a single hardcoded engine id.
- **Installer/route bodies are delegates.** Specialised classes were reduced to
  facades and per-engine routes to prefix wiring; responses are pinned as
  identical.
- **Profile-driven launch revisions change once on upgrade.** Manifests written
  before this release carry no profile digest, so the first apply after the
  upgrade produces a new revision for llama.cpp, ik_llama, Unsloth and the venv
  engines. The upgrade does not restart a running process; an explicit
  **Restart this model** restarts it, and **Use on next start** publishes for the
  next start. The digest covers the whole validated definition, recipes
  included, so an install-only edit also changes the launch revision — that is
  deliberate, so retry provenance and launch provenance name the same frozen
  definition. Audio.cpp has no runtime profile, but its launch revision and
  manifest now carry the definition digest as well.

## Verification performed and its limits

Verification is recorded in
[verification evidence](review-verification-evidence.md), including the exact
commands and totals. In summary:

- The full backend suite was run in the project's Linux/Python 3.12 dev
  container as a non-root user; the full frontend gate and the static gates were
  run on the assembled revision.
- The focused upgrade smoke test and the third-fork journey described above are
  new in this series and run with the same suite.
- **Hardware evidence is native-llama-only.** A real CUDA llama.cpp source build
  and GGUF launch/inference were exercised on a remote V100; those paths have
  representative-hardware evidence. The other engine families and audio do not,
  and are covered here only by fixtures/fake toolchains. This limit is stated in
  the privately maintained hardware verification record.
- **No GPU was available in the verification workspace.** CUDA compilation,
  CUDA runtime loading and real inference for venv/audio families are therefore
  unverified here. The Windows host is not a supported test environment for the
  suite (POSIX `/proc` and permission semantics differ); the container is.

## Known remaining work

- **Requirement 9 (frontend control flow).** The SGLang runtime-selector copy
  and audio-package branches still carry engine-specific UI, and frozen literal
  caps remain. Catalog search/library options/defaults, ModelConfig engine marks,
  cards, panels and install/build dialogs are descriptor-driven. This is the
  only requirement recorded as transitional.
- **Compatibility-route retirement** waits for an API cutover; the delegates are
  intentionally kept for this release.
- **Audio.cpp launch remains engine-specific.** The audio service layer, routing
  and identity are capability-based; launch compilation still dispatches on
  `audio_cpp`, although its revision/manifest now carries the definition digest.
  A second audio engine is not a proven launch path.
- **Retry of a failed latest PyPI install re-resolves latest** when neither the
  version row nor the snapshot recorded a concrete package version. Release-asset
  installs (Unsloth, 1Cat-vLLM) now write their resolved tag into the pending
  snapshot, so a failed attempt retries that tag. For PyPI installs that failed
  before publication, reinstall an explicit version when reproducibility
  matters.
- **Hardware coverage** for Unsloth release assets, the venv engines and audio
  still needs the representative-GPU matrix.
- **Fixture-only engines** (`yamlcraft`, `launchcraft`, `launchforge`,
  `nativecraft`) are test assets, not shipped engines.

## Review risk map

| Practical risk | Status | Where it is pinned |
| --- | --- | --- |
| Existing install loses paths or state | Covered | Pre-refactor smoke: rows, paths, activation, launch, removal, no invented snapshot |
| Retry/sync/cancel/activation/removal | Covered | `test_engine_definition_mechanisms.py`, `test_audio_cpp_versions_api.py`, `test_native_engine_lifecycle.py`, `test_engine_version_lifecycle.py` |
| Launch revision changes affect running models | Covered | `test_profile_launch.py`, `test_launch_manifests.py`, `test_gguf_launch_characterization.py` |
| Compatibility URLs / intentional response changes | Covered | `test_generic_engine_router.py`, `test_legacy_route_inventory_current.py`, compatibility-route diffs |
| Frontend flows use the real store contract | Covered | `engines.server.test.js`, `engines.venv.test.js`, keyed-read and accessor-contract tests |
| Third fork is different in data only | **Strengthened in this series** | `test_nativecraft_journey.py` builds, publishes and launches the fixture's own `native-server` layout and fails if the recipe output names drift |
| Definition changes are invisible to a running model | Covered | Digest in revision/manifest; apply is explicit |
