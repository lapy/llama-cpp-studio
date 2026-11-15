# Engine behavior parity inventory

Derived by `scripts/parity-inventory.py`; regenerate and commit with any
behavior migration. Inputs to the parity checklist for
`docs/config-driven-engine-refactoring-plan.md` (Phase 1). See the JSON twin
for machine consumption.

- Engines: 11
- `/api` routes: 172
- Scanner kinds dispatched: audio_cpp, llama_help
- Shadow definitions loaded: 11

## Hardcoded engine-ID literals (outside `backend/engines/` homes)

The plan's Phase-1 checklist must consume every row: each file is a site
the definition-driven code path must reach. Counts include both spellings
where a specialized id nests in a family id (`sglang_v100` also matches
`sglang`; `onecat` ids also match `vllm`).

| Engine | Out-of-home literals | Files | Top files |
| --- | --- | --- | --- |
| `1cat_vllm` | 72 | 13 | `frontend/src/stores/engines.venv.test.js` (17), `frontend/src/test-fixtures/taskFixtures.js` (12), `frontend/src/stores/engines.js` (8), `frontend/src/views/EnginesView.vue` (8) |
| `audio_cpp` | 192 | 46 | `frontend/src/views/AudioWorkspace.test.js` (23), `frontend/src/stores/engines.venv.test.js` (13), `frontend/src/views/ModelConfig.audioProfiles.test.js` (11), `backend/routes/llama_versions.py` (10) |
| `ik_llama` | 185 | 24 | `frontend/src/views/EnginesView.vue` (41), `frontend/src/stores/engines.venv.test.js` (27), `backend/routes/llama_versions.py` (22), `frontend/src/stores/engines.js` (22) |
| `llama_cpp` | 253 | 45 | `frontend/src/stores/engines.test.js` (25), `frontend/src/views/EnginesView.vue` (24), `backend/routes/llama_versions.py` (23), `frontend/src/stores/engines.venv.test.js` (22) |
| `lmdeploy` | 92 | 20 | `frontend/src/views/EnginesView.vue` (15), `frontend/src/stores/engines.venv.test.js` (13), `frontend/src/test-fixtures/taskFixtures.js` (12), `frontend/src/views/EnginesView.tasks.test.js` (11) |
| `sglang` | 69 | 19 | `frontend/src/stores/engines.venv.test.js` (15), `frontend/src/engines/registry.js` (13), `frontend/src/stores/engines.js` (8), `frontend/src/views/EnginesView.tasks.test.js` (6) |
| `sglang_sm70` | 0 | 0 |  |
| `sglang_sxm2` | 0 | 0 |  |
| `sglang_v100` | 60 | 13 | `frontend/src/stores/engines.venv.test.js` (19), `frontend/src/components/system/SglangEnginePanel.test.js` (7), `frontend/src/engines/registry.js` (5), `frontend/src/views/EnginesView.tasks.test.js` (5) |
| `unsloth_llama` | 56 | 12 | `frontend/src/stores/engines.venv.test.js` (16), `frontend/src/views/EnginesView.tasks.test.js` (8), `frontend/src/engines/registry.js` (7), `frontend/src/views/EnginesView.vue` (7) |
| `vllm` | 89 | 20 | `frontend/src/stores/engines.venv.test.js` (19), `frontend/src/engines/registry.js` (12), `frontend/src/composables/engineReadiness.test.js` (10), `frontend/src/views/EnginesView.tasks.test.js` (7) |

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
| `sglang` | 4 | 4 | 0 | 0 | 4 | 0 | 4 | 4 | 0 | 4 | 0 | settings |
| `sglang_sm70` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `sglang_sxm2` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
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
| `sglang_sm70` | — | — | — | — |
| `sglang_sxm2` | — | — | — | — |
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
| `sglang_sm70` | `backend/venv_install_settings.py:default_install_settings` | `source_branch`, `source_repo` |
| `sglang_sxm2` | `backend/venv_install_settings.py:default_install_settings` | `source_branch`, `source_repo` |
| `sglang_v100` | `backend/venv_install_settings.py:default_install_settings` | `source_branch`, `source_repo` |
| `unsloth_llama` | `—` | prebuilt installs have no persisted settings surface |
| `vllm` | `backend/venv_install_settings.py:default_install_settings` | `pip_version`, `source_branch`, `source_repo` |

## Route inventory

| Path | Methods â†’ success schema | Attributed engines |
| --- | --- | --- |
| `/api/1cat-vllm/build-settings` | GET â†’ object, PUT â†’ object | 1cat_vllm, vllm |
| `/api/1cat-vllm/cancel` | POST â†’ object | 1cat_vllm, vllm |
| `/api/1cat-vllm/check-updates` | GET â†’ object | 1cat_vllm, vllm |
| `/api/1cat-vllm/install` | POST â†’ object | 1cat_vllm, vllm |
| `/api/1cat-vllm/install-source` | POST â†’ object | 1cat_vllm, vllm |
| `/api/1cat-vllm/remove` | POST â†’ object | 1cat_vllm, vllm |
| `/api/1cat-vllm/status` | GET â†’ object | 1cat_vllm, vllm |
| `/api/access` | GET â†’ unknown | — |
| `/api/audio-cpp/build-options` | GET â†’ unknown | audio_cpp |
| `/api/audio-cpp/build-settings` | GET â†’ unknown, PUT â†’ unknown | audio_cpp |
| `/api/audio-cpp/build-source` | POST â†’ unknown | audio_cpp |
| `/api/audio-cpp/cancel` | POST â†’ unknown | audio_cpp |
| `/api/audio-cpp/check-updates` | GET â†’ unknown | audio_cpp |
| `/api/audio-cpp/community-voices` | DELETE â†’ unknown, GET â†’ unknown | audio_cpp |
| `/api/audio-cpp/community-voices/install` | POST â†’ unknown | audio_cpp |
| `/api/audio-cpp/contract` | GET â†’ unknown | audio_cpp |
| `/api/audio-cpp/status` | GET â†’ unknown | audio_cpp |
| `/api/audio-cpp/update` | POST â†’ unknown | audio_cpp |
| `/api/benchmarks/run` | POST â†’ BenchmarkResponse | — |
| `/api/benchmarks/{model_id}` | GET â†’ BenchmarkResponse[] | — |
| `/api/config-backup` | GET â†’ unknown | — |
| `/api/config-backup/apply` | POST â†’ BackupApplyResponse | — |
| `/api/config-backup/preview` | POST â†’ BackupPreviewResponse | — |
| `/api/config-backup/reconcile` | POST â†’ BackupReconcileResponse | — |
| `/api/config-history` | GET â†’ HistoryEntryResponse[] | — |
| `/api/config-history/{entry_id}` | GET â†’ HistoryDiffResponse | — |
| `/api/config-history/{entry_id}/restore` | POST â†’ HistoryRestoreResponse | — |
| `/api/diagnostics/bundle` | GET â†’ unknown | — |
| `/api/engines` | GET â†’ unknown | — |
| `/api/engines/{engine_id}/build-settings` | GET â†’ object, PUT â†’ object | — |
| `/api/engines/{engine_id}/cancel` | POST â†’ CancelAccepted | — |
| `/api/engines/{engine_id}/check-updates` | GET â†’ object | — |
| `/api/engines/{engine_id}/install` | POST â†’ InstallAccepted | — |
| `/api/engines/{engine_id}/install-source` | POST â†’ InstallAccepted | — |
| `/api/engines/{engine_id}/remove` | POST â†’ InstallAccepted | — |
| `/api/engines/{engine_id}/status` | GET â†’ VenvEngineStatus | — |
| `/api/events` | GET â†’ unknown | — |
| `/api/gpu-info` | GET â†’ unknown | — |
| `/api/gpu-list` | GET â†’ unknown | — |
| `/api/inference/keys` | GET â†’ unknown, POST â†’ unknown | — |
| `/api/inference/keys/{key_id}` | DELETE â†’ empty, PATCH â†’ unknown | — |
| `/api/inference/keys/{key_id}/rotate` | POST â†’ unknown | — |
| `/api/live` | GET â†’ unknown | — |
| `/api/llama-versions` | GET â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-cancel` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-capabilities` | GET â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-options` | GET â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-settings` | GET â†’ unknown, PUT â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/build-source` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/check-updates` | GET â†’ UpdateCheckResponse | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-install` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-logs` | GET â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-status` | GET â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda-uninstall` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/cuda/cancel` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/scan-engine-params` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/update` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/verify/{version}` | GET â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/activate` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/build-config` | PUT â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/retry` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/versions/sync` | POST â†’ unknown | ik_llama, llama_cpp |
| `/api/llama-versions/{version_id}` | DELETE â†’ unknown | ik_llama, llama_cpp |
| `/api/lmdeploy/build-settings` | GET â†’ object, PUT â†’ object | lmdeploy |
| `/api/lmdeploy/cancel` | POST â†’ object | lmdeploy |
| `/api/lmdeploy/check-updates` | GET â†’ object | lmdeploy |
| `/api/lmdeploy/install` | POST â†’ object | lmdeploy |
| `/api/lmdeploy/install-source` | POST â†’ object | lmdeploy |
| `/api/lmdeploy/remove` | POST â†’ object | lmdeploy |
| `/api/lmdeploy/status` | GET â†’ object | lmdeploy |
| `/api/model-catalog/import` | POST â†’ unknown | — |
| `/api/model-catalog/install` | POST â†’ unknown | — |
| `/api/model-catalog/search` | POST â†’ unknown | — |
| `/api/model-catalog/tasks/{task_id}/cancel` | POST â†’ unknown | — |
| `/api/model-config-templates` | GET â†’ object[], POST â†’ object | — |
| `/api/model-config-templates/{template_id}` | DELETE â†’ object, PUT â†’ object | — |
| `/api/models` | GET â†’ unknown | — |
| `/api/models/delete-group` | POST â†’ unknown | — |
| `/api/models/download` | POST â†’ unknown | — |
| `/api/models/downloads/cancel` | POST â†’ unknown | — |
| `/api/models/gguf/download-bundle` | POST â†’ unknown | — |
| `/api/models/huggingface-token` | GET â†’ unknown, POST â†’ unknown | — |
| `/api/models/param-registry` | GET â†’ unknown | — |
| `/api/models/quantization-sizes` | POST â†’ unknown | — |
| `/api/models/safetensors` | DELETE â†’ unknown, GET â†’ unknown | — |
| `/api/models/safetensors/download-bundle` | POST â†’ unknown | — |
| `/api/models/search` | POST â†’ unknown | — |
| `/api/models/search/{model_id}/file-sizes` | GET â†’ unknown | — |
| `/api/models/{model_id}` | DELETE â†’ unknown | — |
| `/api/models/{model_id}/companions` | GET â†’ unknown | — |
| `/api/models/{model_id}/config` | GET â†’ unknown, PUT â†’ unknown | — |
| `/api/models/{model_id}/config/apply-template` | POST â†’ unknown | — |
| `/api/models/{model_id}/config/save-template` | POST â†’ unknown | — |
| `/api/models/{model_id}/connect-test` | POST â†’ unknown | — |
| `/api/models/{model_id}/dflash` | POST â†’ unknown | — |
| `/api/models/{model_id}/dspark` | POST â†’ unknown | — |
| `/api/models/{model_id}/mtp` | POST â†’ unknown | — |
| `/api/models/{model_id}/preview-command` | POST â†’ unknown | — |
| `/api/models/{model_id}/projector` | POST â†’ unknown | — |
| `/api/models/{model_id}/reference-audio` | GET â†’ unknown, POST â†’ unknown | — |
| `/api/models/{model_id}/reference-audio/{filename}` | DELETE â†’ unknown | — |
| `/api/models/{model_id}/refresh` | POST â†’ unknown | — |
| `/api/models/{model_id}/runtime/apply` | POST â†’ unknown | — |
| `/api/models/{model_id}/runtime/apply/{operation_id}` | GET â†’ unknown | — |
| `/api/models/{model_id}/runtime/apply/{operation_id}/cancel` | POST â†’ unknown | — |
| `/api/models/{model_id}/saved-command` | GET â†’ unknown | — |
| `/api/models/{model_id}/start` | POST â†’ unknown | — |
| `/api/models/{model_id}/stop` | POST â†’ unknown | — |
| `/api/operations/reconcile` | POST â†’ unknown | — |
| `/api/operations/recovery` | GET â†’ unknown | — |
| `/api/proxy` | GET â†’ unknown | — |
| `/api/proxy/lifecycle/{action}` | POST â†’ unknown | — |
| `/api/proxy/models/{model_id}/logs` | DELETE â†’ empty, GET â†’ unknown | — |
| `/api/proxy/models/{model_id}/preload` | POST â†’ unknown | — |
| `/api/proxy/models/{model_id}/reload` | POST â†’ unknown | — |
| `/api/proxy/models/{model_id}/unload` | POST â†’ unknown | — |
| `/api/proxy/requests` | GET â†’ unknown | — |
| `/api/proxy/settings` | PUT â†’ unknown | — |
| `/api/proxy/stats/reset` | POST â†’ unknown | — |
| `/api/ready` | GET â†’ unknown | — |
| `/api/runtime/apply-launch` | POST â†’ unknown | — |
| `/api/runtime/pending` | GET â†’ unknown | — |
| `/api/runtime/stale` | GET â†’ unknown | — |
| `/api/session` | POST â†’ unknown | — |
| `/api/settings/inference` | GET â†’ unknown, PUT â†’ unknown | — |
| `/api/sglang-sm70/build-settings` | GET â†’ object, PUT â†’ object | sglang |
| `/api/sglang-sm70/cancel` | POST â†’ object | sglang |
| `/api/sglang-sm70/check-updates` | GET â†’ object | sglang |
| `/api/sglang-sm70/install` | POST â†’ object | sglang |
| `/api/sglang-sm70/install-source` | POST â†’ object | sglang |
| `/api/sglang-sm70/remove` | POST â†’ object | sglang |
| `/api/sglang-sm70/status` | GET â†’ object | sglang |
| `/api/sglang-sxm2/build-settings` | GET â†’ object, PUT â†’ object | sglang |
| `/api/sglang-sxm2/cancel` | POST â†’ object | sglang |
| `/api/sglang-sxm2/check-updates` | GET â†’ object | sglang |
| `/api/sglang-sxm2/install` | POST â†’ object | sglang |
| `/api/sglang-sxm2/install-source` | POST â†’ object | sglang |
| `/api/sglang-sxm2/remove` | POST â†’ object | sglang |
| `/api/sglang-sxm2/status` | GET â†’ object | sglang |
| `/api/sglang-v100/build-settings` | GET â†’ object, PUT â†’ object | sglang, sglang_v100 |
| `/api/sglang-v100/cancel` | POST â†’ object | sglang, sglang_v100 |
| `/api/sglang-v100/check-updates` | GET â†’ object | sglang, sglang_v100 |
| `/api/sglang-v100/install` | POST â†’ object | sglang, sglang_v100 |
| `/api/sglang-v100/install-source` | POST â†’ object | sglang, sglang_v100 |
| `/api/sglang-v100/remove` | POST â†’ object | sglang, sglang_v100 |
| `/api/sglang-v100/status` | GET â†’ object | sglang, sglang_v100 |
| `/api/sglang/build-settings` | GET â†’ object, PUT â†’ object | sglang |
| `/api/sglang/cancel` | POST â†’ object | sglang |
| `/api/sglang/check-updates` | GET â†’ object | sglang |
| `/api/sglang/install` | POST â†’ object | sglang |
| `/api/sglang/install-source` | POST â†’ object | sglang |
| `/api/sglang/remove` | POST â†’ object | sglang |
| `/api/sglang/status` | GET â†’ object | sglang |
| `/api/status` | GET â†’ unknown | — |
| `/api/tasks/dismiss` | POST â†’ unknown | — |
| `/api/unsloth-llama/cancel` | POST â†’ object | unsloth_llama |
| `/api/unsloth-llama/check-updates` | GET â†’ object | unsloth_llama |
| `/api/unsloth-llama/install` | POST â†’ object | unsloth_llama |
| `/api/vllm/build-settings` | GET â†’ object, PUT â†’ object | vllm |
| `/api/vllm/cancel` | POST â†’ object | vllm |
| `/api/vllm/check-updates` | GET â†’ object | vllm |
| `/api/vllm/install` | POST â†’ object | vllm |
| `/api/vllm/install-source` | POST â†’ object | vllm |
| `/api/vllm/remove` | POST â†’ object | vllm |
| `/api/vllm/status` | GET â†’ object | vllm |
| `/v1/audio/alignments` | POST â†’ unknown | — |
| `/v1/audio/speech` | POST â†’ unknown | — |
| `/v1/audio/transcriptions` | POST â†’ unknown | — |
| `/v1/audio/transcriptions/details` | POST â†’ unknown | — |
| `/v1/audio/{rest}` | DELETE â†’ unknown, GET â†’ unknown, HEAD â†’ unknown, OPTIONS â†’ unknown, PATCH â†’ unknown, POST â†’ unknown, PUT â†’ unknown | — |
| `/v1/batches/transcriptions` | POST â†’ unknown | — |
| `/v1/tasks/run` | POST â†’ unknown | — |
