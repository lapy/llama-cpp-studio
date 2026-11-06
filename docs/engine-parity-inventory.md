# Engine behavior parity inventory

Derived by `scripts/parity-inventory.py`; regenerate and commit with any
behavior migration. Inputs to the parity checklist for
`docs/config-driven-engine-refactoring-plan.md` (Phase 1). See the JSON twin
for machine consumption.

- Engines: 9
- `/api` routes: 157
- Scanner kinds dispatched: audio_cpp, llama_help
- Shadow definitions loaded: 13

## Hardcoded engine-ID literals (outside `backend/engines/` homes)

The plan's Phase-1 checklist must consume every row: each file is a site
the definition-driven code path must reach. Counts include both spellings
where a specialized id nests in a family id (`sglang_v100` also matches
`sglang`; `onecat` ids also match `vllm`).

| Engine | Out-of-home literals | Files | Top files |
| --- | --- | --- | --- |
| `1cat_vllm` | 73 | 14 | `frontend/src/stores/engines.venv.test.js` (17), `frontend/src/test-fixtures/taskFixtures.js` (12), `frontend/src/stores/engines.js` (8), `frontend/src/views/EnginesView.vue` (8) |
| `audio_cpp` | 188 | 45 | `frontend/src/views/AudioWorkspace.test.js` (23), `frontend/src/stores/engines.venv.test.js` (13), `frontend/src/views/ModelConfig.audioProfiles.test.js` (11), `backend/routes/llama_versions.py` (10) |
| `ik_llama` | 182 | 24 | `frontend/src/views/EnginesView.vue` (41), `frontend/src/stores/engines.venv.test.js` (27), `backend/routes/llama_versions.py` (22), `frontend/src/stores/engines.js` (20) |
| `llama_cpp` | 246 | 43 | `frontend/src/stores/engines.test.js` (25), `frontend/src/views/EnginesView.vue` (24), `backend/routes/llama_versions.py` (23), `frontend/src/stores/engines.venv.test.js` (22) |
| `lmdeploy` | 92 | 20 | `frontend/src/views/EnginesView.vue` (15), `frontend/src/stores/engines.venv.test.js` (13), `frontend/src/test-fixtures/taskFixtures.js` (12), `frontend/src/views/EnginesView.tasks.test.js` (11) |
| `sglang` | 76 | 19 | `frontend/src/engines/registry.js` (20), `frontend/src/stores/engines.venv.test.js` (15), `frontend/src/stores/engines.js` (8), `frontend/src/views/EnginesView.tasks.test.js` (6) |
| `sglang_v100` | 59 | 13 | `frontend/src/stores/engines.venv.test.js` (19), `frontend/src/components/system/SglangEnginePanel.test.js` (7), `frontend/src/views/EnginesView.tasks.test.js` (5), `frontend/src/views/EnginesView.vue` (5) |
| `unsloth_llama` | 57 | 13 | `frontend/src/stores/engines.venv.test.js` (16), `frontend/src/views/EnginesView.tasks.test.js` (8), `frontend/src/engines/registry.js` (7), `frontend/src/views/EnginesView.vue` (7) |
| `vllm` | 88 | 20 | `frontend/src/stores/engines.venv.test.js` (19), `frontend/src/engines/registry.js` (12), `frontend/src/composables/engineReadiness.test.js` (9), `frontend/src/views/EnginesView.tasks.test.js` (7) |

## Per-engine operation endpoints

Current route surface per lifecycle operation; the generic adapter must
reproduce each endpoint family (or explicitly type a response-shape
difference) before its legacy router can retire.

| Engine | install | source | retry | sync | updates | activate | delete | status | logs | cancel | scan | other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1cat_vllm` | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | settings |
| `audio_cpp` | 1 | 1 | 0 | 0 | 2 | 0 | 0 | 1 | 0 | 1 | 0 | contract, settings |
| `ik_llama` | 0 | 1 | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 1 | 1 | settings |
| `llama_cpp` | 0 | 1 | 1 | 1 | 2 | 1 | 0 | 0 | 0 | 1 | 1 | settings |
| `lmdeploy` | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | settings |
| `sglang` | 2 | 2 | 0 | 0 | 2 | 0 | 2 | 2 | 0 | 2 | 0 | settings |
| `sglang_v100` | 1 | 1 | 0 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0 | settings |
| `unsloth_llama` | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| `vllm` | 2 | 2 | 0 | 0 | 2 | 0 | 2 | 2 | 0 | 2 | 0 | settings |

## Response-shape divergences (adapter unification inputs)

The plan requires these shapes unified or explicitly typed per surface
before legacy routers retire; today they differ per engine.

| Engine | update-check | status | install | contract |
| --- | --- | --- | --- | --- |
| `1cat_vllm` | `GET /api/1cat-vllm/check-updates -> object` | `GET /api/1cat-vllm/status -> object` | `POST /api/1cat-vllm/install -> object` | — |
| `audio_cpp` | `GET /api/audio-cpp/check-updates -> unknown` | `GET /api/audio-cpp/status -> unknown` | `POST /api/audio-cpp/community-voices/install -> unknown` | `GET /api/audio-cpp/contract -> unknown` |
| `ik_llama` | `GET /api/llama-versions/check-updates -> UpdateCheckResponse` | — | — | — |
| `llama_cpp` | `GET /api/llama-versions/check-updates -> UpdateCheckResponse` | — | — | — |
| `lmdeploy` | `GET /api/lmdeploy/check-updates -> object` | `GET /api/lmdeploy/status -> object` | `POST /api/lmdeploy/install -> object` | — |
| `sglang` | `GET /api/sglang/check-updates -> object` | `GET /api/sglang/status -> object` | `POST /api/sglang/install -> object` | — |
| `sglang_v100` | `GET /api/sglang-v100/check-updates -> object` | `GET /api/sglang-v100/status -> object` | `POST /api/sglang-v100/install -> object` | — |
| `unsloth_llama` | `GET /api/unsloth-llama/check-updates -> object` | — | `POST /api/unsloth-llama/install -> object` | — |
| `vllm` | `GET /api/vllm/check-updates -> object` | `GET /api/vllm/status -> object` | `POST /api/vllm/install -> object` | — |

## Install/build default surfaces

Persisted defaults each engine must keep producing after migration
(full values in the JSON twin; keys listed here).

| Engine | Default source | Keys |
| --- | --- | --- |
| `1cat_vllm` | `backend/venv_install_settings.py:default_install_settings` | `release_version`, `source_branch`, `source_repo` |
| `audio_cpp` | `backend/engines/audio_cpp/build_options.py:default_build_settings` | `build_settings`, `source_ref_default` |
| `ik_llama` | `backend/engines/llama_cpp/build_options.py:default_build_settings (shared llama-server family)` | `accelerate`, `all_warnings`, `amx_bf16`, `amx_int8`, `amx_tile`, `avx`, `avx2`, `avx512`, `avx512_bf16`, `avx512_vbmi`, `avx512_vnni`, `avx_vnni`, `backend_dir`, `backend_dl`, `blas`, `blas_vendor`, `bmi2`, `build_app`, `build_common`, `build_examples`, `build_mtmd`, `build_server`, `build_tests`, `build_tools`, `build_type`, `build_ui`, `ccache`, `cflags`, `cpu`, `cpu_all_variants`, `cpu_arm_arch`, `cpu_hbm`, `cpu_kleidiai`, `cpu_powerpc_cputype`, `cpu_repack`, `cuda`, `cuda_architectures`, `cuda_compression_mode`, `cuda_dmmv_x`, `cuda_f16`, `cuda_fa`, `cuda_fa_quants`, `cuda_force_cublas`, `cuda_force_dmmv`, `cuda_force_mmq`, `cuda_fusion`, `cuda_graphs`, `cuda_iqk_force_bf16`, `cuda_kquants_iter`, `cuda_min_batch_offload`, `cuda_mmv_y`, `cuda_nccl`, `cuda_no_peer_copy`, `cuda_no_vmm`, `cuda_peer_max_batch_size`, `custom_cmake_args`, `cxxflags`, `expert_chunking`, `f16c`, `fatal_warnings`, `flash_attention`, `fma`, `gprof`, `install_tests`, `install_tools`, `iqk_fa_all_quants`, `iqk_flash_attention`, `iqk_mul_mat`, `lasx`, `llamafile`, `llguidance`, `lsx`, `lto`, `max_contexts`, `native`, `nccl`, `openmp`, `openmp_fetch`, `openssl`, `rv_zfh`, `rv_zicbop`, `rv_zihintpause`, `rv_zvfbfwma`, `rv_zvfh`, `rvv`, `sanitize_address`, `sanitize_thread`, `sanitize_undefined`, `sched_max_copies`, `sched_no_realloc`, `sse42`, `static`, `subprocess`, `sve`, `use_prebuilt_ui`, `vxe`, `xtheadvector` |
| `llama_cpp` | `backend/engines/llama_cpp/build_options.py:default_build_settings (shared llama-server family)` | `accelerate`, `all_warnings`, `amx_bf16`, `amx_int8`, `amx_tile`, `avx`, `avx2`, `avx512`, `avx512_bf16`, `avx512_vbmi`, `avx512_vnni`, `avx_vnni`, `backend_dir`, `backend_dl`, `blas`, `blas_vendor`, `bmi2`, `build_app`, `build_common`, `build_examples`, `build_mtmd`, `build_server`, `build_tests`, `build_tools`, `build_type`, `build_ui`, `ccache`, `cflags`, `cpu`, `cpu_all_variants`, `cpu_arm_arch`, `cpu_hbm`, `cpu_kleidiai`, `cpu_powerpc_cputype`, `cpu_repack`, `cuda`, `cuda_architectures`, `cuda_compression_mode`, `cuda_dmmv_x`, `cuda_f16`, `cuda_fa`, `cuda_fa_quants`, `cuda_force_cublas`, `cuda_force_dmmv`, `cuda_force_mmq`, `cuda_fusion`, `cuda_graphs`, `cuda_iqk_force_bf16`, `cuda_kquants_iter`, `cuda_min_batch_offload`, `cuda_mmv_y`, `cuda_nccl`, `cuda_no_peer_copy`, `cuda_no_vmm`, `cuda_peer_max_batch_size`, `custom_cmake_args`, `cxxflags`, `expert_chunking`, `f16c`, `fatal_warnings`, `flash_attention`, `fma`, `gprof`, `install_tests`, `install_tools`, `iqk_fa_all_quants`, `iqk_flash_attention`, `iqk_mul_mat`, `lasx`, `llamafile`, `llguidance`, `lsx`, `lto`, `max_contexts`, `native`, `nccl`, `openmp`, `openmp_fetch`, `openssl`, `rv_zfh`, `rv_zicbop`, `rv_zihintpause`, `rv_zvfbfwma`, `rv_zvfh`, `rvv`, `sanitize_address`, `sanitize_thread`, `sanitize_undefined`, `sched_max_copies`, `sched_no_realloc`, `sse42`, `static`, `subprocess`, `sve`, `use_prebuilt_ui`, `vxe`, `xtheadvector` |
| `lmdeploy` | `backend/venv_install_settings.py:default_install_settings` | `compat_sm70`, `compat_torch_index_url`, `compat_torch_spec`, `pip_version`, `source_branch`, `source_repo` |
| `sglang` | `backend/venv_install_settings.py:default_install_settings` | `pip_version`, `source_branch`, `source_repo` |
| `sglang_v100` | `backend/venv_install_settings.py:default_install_settings` | `source_branch`, `source_repo` |
| `unsloth_llama` | `—` | prebuilt installs have no persisted settings surface |
| `vllm` | `backend/venv_install_settings.py:default_install_settings` | `pip_version`, `source_branch`, `source_repo` |

## Route inventory

| Path | Methods → success schema | Attributed engines |
| --- | --- | --- |
| `/api/1cat-vllm/build-settings` | GET → object, PUT → object | 1cat_vllm, vllm |
| `/api/1cat-vllm/cancel` | POST → object | 1cat_vllm, vllm |
| `/api/1cat-vllm/check-updates` | GET → object | 1cat_vllm, vllm |
| `/api/1cat-vllm/install` | POST → object | 1cat_vllm, vllm |
| `/api/1cat-vllm/install-source` | POST → object | 1cat_vllm, vllm |
| `/api/1cat-vllm/remove` | POST → object | 1cat_vllm, vllm |
| `/api/1cat-vllm/status` | GET → object | 1cat_vllm, vllm |
| `/api/access` | GET → unknown | — |
| `/api/audio-cpp/build-options` | GET → unknown | audio_cpp |
| `/api/audio-cpp/build-settings` | GET → unknown, PUT → unknown | audio_cpp |
| `/api/audio-cpp/build-source` | POST → unknown | audio_cpp |
| `/api/audio-cpp/cancel` | POST → unknown | audio_cpp |
| `/api/audio-cpp/check-updates` | GET → unknown | audio_cpp |
| `/api/audio-cpp/community-voices` | DELETE → unknown, GET → unknown | audio_cpp |
| `/api/audio-cpp/community-voices/install` | POST → unknown | audio_cpp |
| `/api/audio-cpp/contract` | GET → unknown | audio_cpp |
| `/api/audio-cpp/status` | GET → unknown | audio_cpp |
| `/api/audio-cpp/update` | POST → unknown | audio_cpp |
| `/api/benchmarks/run` | POST → BenchmarkResponse | — |
| `/api/benchmarks/{model_id}` | GET → BenchmarkResponse[] | — |
| `/api/config-backup` | GET → unknown | — |
| `/api/config-backup/apply` | POST → BackupApplyResponse | — |
| `/api/config-backup/preview` | POST → BackupPreviewResponse | — |
| `/api/config-backup/reconcile` | POST → BackupReconcileResponse | — |
| `/api/config-history` | GET → HistoryEntryResponse[] | — |
| `/api/config-history/{entry_id}` | GET → HistoryDiffResponse | — |
| `/api/config-history/{entry_id}/restore` | POST → HistoryRestoreResponse | — |
| `/api/diagnostics/bundle` | GET → unknown | — |
| `/api/engines` | GET → unknown | — |
| `/api/engines/{engine_id}/build-settings` | GET → object, PUT → object | — |
| `/api/engines/{engine_id}/cancel` | POST → CancelAccepted | — |
| `/api/engines/{engine_id}/check-updates` | GET → object | — |
| `/api/engines/{engine_id}/install` | POST → InstallAccepted | — |
| `/api/engines/{engine_id}/install-source` | POST → InstallAccepted | — |
| `/api/engines/{engine_id}/remove` | POST → InstallAccepted | — |
| `/api/engines/{engine_id}/status` | GET → VenvEngineStatus | — |
| `/api/events` | GET → unknown | — |
| `/api/gpu-info` | GET → unknown | — |
| `/api/gpu-list` | GET → unknown | — |
| `/api/inference/keys` | GET → unknown, POST → unknown | — |
| `/api/inference/keys/{key_id}` | DELETE → empty, PATCH → unknown | — |
| `/api/inference/keys/{key_id}/rotate` | POST → unknown | — |
| `/api/live` | GET → unknown | — |
| `/api/llama-versions` | GET → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-cancel` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-capabilities` | GET → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-options` | GET → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-settings` | GET → unknown, PUT → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-source` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/check-updates` | GET → UpdateCheckResponse | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-install` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-logs` | GET → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-status` | GET → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-uninstall` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda/cancel` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/scan-engine-params` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/update` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/verify/{version}` | GET → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/activate` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/build-config` | PUT → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/retry` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/sync` | POST → unknown | ik_llama, llama_cpp |
| `/api/llama-versions/{version_id}` | DELETE → unknown | ik_llama, llama_cpp |
| `/api/lmdeploy/build-settings` | GET → object, PUT → object | lmdeploy |
| `/api/lmdeploy/cancel` | POST → object | lmdeploy |
| `/api/lmdeploy/check-updates` | GET → object | lmdeploy |
| `/api/lmdeploy/install` | POST → object | lmdeploy |
| `/api/lmdeploy/install-source` | POST → object | lmdeploy |
| `/api/lmdeploy/remove` | POST → object | lmdeploy |
| `/api/lmdeploy/status` | GET → object | lmdeploy |
| `/api/model-catalog/import` | POST → unknown | — |
| `/api/model-catalog/install` | POST → unknown | — |
| `/api/model-catalog/search` | POST → unknown | — |
| `/api/model-catalog/tasks/{task_id}/cancel` | POST → unknown | — |
| `/api/model-config-templates` | GET → object[], POST → object | — |
| `/api/model-config-templates/{template_id}` | DELETE → object, PUT → object | — |
| `/api/models` | GET → unknown | — |
| `/api/models/delete-group` | POST → unknown | — |
| `/api/models/download` | POST → unknown | — |
| `/api/models/downloads/cancel` | POST → unknown | — |
| `/api/models/gguf/download-bundle` | POST → unknown | — |
| `/api/models/huggingface-token` | GET → unknown, POST → unknown | — |
| `/api/models/param-registry` | GET → unknown | — |
| `/api/models/quantization-sizes` | POST → unknown | — |
| `/api/models/safetensors` | DELETE → unknown, GET → unknown | — |
| `/api/models/safetensors/download-bundle` | POST → unknown | — |
| `/api/models/search` | POST → unknown | — |
| `/api/models/search/{model_id}/file-sizes` | GET → unknown | — |
| `/api/models/{model_id}` | DELETE → unknown | — |
| `/api/models/{model_id}/companions` | GET → unknown | — |
| `/api/models/{model_id}/config` | GET → unknown, PUT → unknown | — |
| `/api/models/{model_id}/config/apply-template` | POST → unknown | — |
| `/api/models/{model_id}/config/save-template` | POST → unknown | — |
| `/api/models/{model_id}/connect-test` | POST → unknown | — |
| `/api/models/{model_id}/dflash` | POST → unknown | — |
| `/api/models/{model_id}/mtp` | POST → unknown | — |
| `/api/models/{model_id}/preview-command` | POST → unknown | — |
| `/api/models/{model_id}/projector` | POST → unknown | — |
| `/api/models/{model_id}/reference-audio` | GET → unknown, POST → unknown | — |
| `/api/models/{model_id}/reference-audio/{filename}` | DELETE → unknown | — |
| `/api/models/{model_id}/refresh` | POST → unknown | — |
| `/api/models/{model_id}/runtime/apply` | POST → unknown | — |
| `/api/models/{model_id}/runtime/apply/{operation_id}` | GET → unknown | — |
| `/api/models/{model_id}/runtime/apply/{operation_id}/cancel` | POST → unknown | — |
| `/api/models/{model_id}/saved-command` | GET → unknown | — |
| `/api/models/{model_id}/start` | POST → unknown | — |
| `/api/models/{model_id}/stop` | POST → unknown | — |
| `/api/operations/reconcile` | POST → unknown | — |
| `/api/operations/recovery` | GET → unknown | — |
| `/api/proxy` | GET → unknown | — |
| `/api/proxy/lifecycle/{action}` | POST → unknown | — |
| `/api/proxy/models/{model_id}/logs` | DELETE → empty, GET → unknown | — |
| `/api/proxy/models/{model_id}/preload` | POST → unknown | — |
| `/api/proxy/models/{model_id}/reload` | POST → unknown | — |
| `/api/proxy/models/{model_id}/unload` | POST → unknown | — |
| `/api/proxy/requests` | GET → unknown | — |
| `/api/proxy/settings` | PUT → unknown | — |
| `/api/proxy/stats/reset` | POST → unknown | — |
| `/api/ready` | GET → unknown | — |
| `/api/runtime/apply-launch` | POST → unknown | — |
| `/api/runtime/pending` | GET → unknown | — |
| `/api/runtime/stale` | GET → unknown | — |
| `/api/session` | POST → unknown | — |
| `/api/settings/inference` | GET → unknown, PUT → unknown | — |
| `/api/sglang-v100/build-settings` | GET → object, PUT → object | sglang, sglang_v100 |
| `/api/sglang-v100/cancel` | POST → object | sglang, sglang_v100 |
| `/api/sglang-v100/check-updates` | GET → object | sglang, sglang_v100 |
| `/api/sglang-v100/install` | POST → object | sglang, sglang_v100 |
| `/api/sglang-v100/install-source` | POST → object | sglang, sglang_v100 |
| `/api/sglang-v100/remove` | POST → object | sglang, sglang_v100 |
| `/api/sglang-v100/status` | GET → object | sglang, sglang_v100 |
| `/api/sglang/build-settings` | GET → object, PUT → object | sglang |
| `/api/sglang/cancel` | POST → object | sglang |
| `/api/sglang/check-updates` | GET → object | sglang |
| `/api/sglang/install` | POST → object | sglang |
| `/api/sglang/install-source` | POST → object | sglang |
| `/api/sglang/remove` | POST → object | sglang |
| `/api/sglang/status` | GET → object | sglang |
| `/api/status` | GET → unknown | — |
| `/api/tasks/dismiss` | POST → unknown | — |
| `/api/unsloth-llama/cancel` | POST → object | unsloth_llama |
| `/api/unsloth-llama/check-updates` | GET → object | unsloth_llama |
| `/api/unsloth-llama/install` | POST → object | unsloth_llama |
| `/api/vllm/build-settings` | GET → object, PUT → object | vllm |
| `/api/vllm/cancel` | POST → object | vllm |
| `/api/vllm/check-updates` | GET → object | vllm |
| `/api/vllm/install` | POST → object | vllm |
| `/api/vllm/install-source` | POST → object | vllm |
| `/api/vllm/remove` | POST → object | vllm |
| `/api/vllm/status` | GET → object | vllm |
| `/v1/audio/alignments` | POST → unknown | — |
| `/v1/audio/speech` | POST → unknown | — |
| `/v1/audio/transcriptions` | POST → unknown | — |
| `/v1/audio/transcriptions/details` | POST → unknown | — |
| `/v1/audio/{rest}` | DELETE → unknown, GET → unknown, HEAD → unknown, OPTIONS → unknown, PATCH → unknown, POST → unknown, PUT → unknown | — |
| `/v1/batches/transcriptions` | POST → unknown | — |
| `/v1/tasks/run` | POST → unknown | — |
