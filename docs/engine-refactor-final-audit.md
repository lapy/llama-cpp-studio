# Engine refactor final audit (2026-11-04)

This audit accompanies `docs/engine-refactor-requirement-matrix.md` and records
what is definition-owned, what compatibility wiring deliberately remains, and
what is still transitional. It is a snapshot of the committed engine tree after the definition-only installer cutover.

## Definition ownership

- All nine shipped engines are loaded from authoritative YAML at import:
  `1cat_vllm`, `audio_cpp`, `ik_llama`, `llama_cpp`, `lmdeploy`, `sglang`,
  `sglang_v100`, `unsloth_llama`, `vllm`.
- `backend/engines/registry.py` contains no `EngineSpec` literal or exporter
  data map. It bootstraps the registry from the validated definitions.
- The exporter (`scripts/export-engine-definitions.py`), the exporter tier
  (`GENERATED_ENGINE_IDS`) and all exporter registration branches are gone.
- Identity, recipes, lifecycle facts, launch profiles, discovery, policies,
  settings defaults, catalog/extension surfaces and artifact formats live in
  YAML. Python supplies mechanisms (hosts, primitives) and domain algorithms.

## Compatibility wiring that deliberately remains

- `backend/routes/lmdeploy_versions.py`, `vllm_versions.py`,
  `onecat_vllm_versions.py`, `sglang_versions.py`: thin
  `definition_compat_router` prefixes over the shared lifecycle record.
- The per-engine venv modules (`lmdeploy`, `vllm`, `sglang`) and their
  installer maps are deleted. Venv installers resolve through the definition's
  lifecycle; old test-only constructors live under `backend/tests`.
- Native host selection is declared as `native_host` (`cmake_source`,
  `audio_cpp`, or `release_assets`). There is no engine-id host map.
- Unsloth's installer class is gone; its release flow runs through the generic
  `backend/engines/python/release_host.py` configured by `release_assets`.
- `LlamaManager`'s build/CUDA helper surface moved to
  `backend/engines/python/cmake_build.py`; the llama module is now a thin
  compatibility re-export and `CmakeSourceInstaller` no longer imports it.
- `audio_cpp_versions.py`: declared audio extension surface, now mounted once
  per engine declaring `runtime_kind: audio_cpp` with request-scoped identity.
- Native source orchestration runs through the shared
  `CmakeSourceInstaller` in `backend/engines/python/cmake_source.py`;
  `source_build.py` is deleted and `llama_versions.py` calls the shared host.
  Repository/artifact/build-policy differences come from the `source_build`
  definition section. The `nativecraft` fixture is the third-fork proof that a
  new CMake source engine needs data, not a new installer subclass.
- Unsloth installs use the shared `start_release_install`/`publish.install`
  flow. The compatibility delegate `LlamaManager.build_source` remains for
  existing callers.

## Transitional requirements (not counted as complete)

- Requirement 9: the SGLang runtime selector copy and audio-package branches
  still carry engine-specific UI; catalog search/library options/defaults,
  ModelConfig engine marks, cards, panels and install/build dialogs are
  descriptor-driven.

## Hardware verification

Native llama.cpp source builds (through `LlamaNativeInstaller`) and GGUF
launch/inference (through `gguf_launch`) were exercised on a remote V100 with
CUDA 12.8; details and remaining family coverage are in
the privately maintained hardware verification record.

## Verification snapshot

- Backend: 2180 passed, 8 skipped (full suite).
- Frontend: 660 tests passed, no unhandled errors.
- Static: ruff, parity inventory, OpenAPI, build-defaults, legacy-route
  inventory, literal ratchet, version — all current.
- The architecture audit test
  (`backend/tests/test_engine_refactor_dead_branches.py`) pins: no exporter
  symbol/branch, no venv engine-module/map, thin compatibility route modules,
  and removal of the legacy scan adapters.

## Known flake (resolved)

`test_action_recovery.py::test_an_unknown_fence_does_not_authorize_a_start`
occasionally read the operation row before the supervisor's queued terminal
write landed. The test now waits for the terminal status; 15 stress runs and a
full suite pass.
