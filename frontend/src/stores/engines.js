import { computed, ref } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'
import {
  applyServerDescriptors,
  ENGINE_DESCRIPTORS,
  NATIVE_GENERIC_VERBS,
  SG_PANEL_ENGINE_IDS,
  nativeGenericApiPath,
  serverBackedApiPath,
  serverBackedDescriptor,
  serverBackedEngines,
  venvApiPath,
  venvDescriptor,
  buildOptionsPath,
} from '../engines/registry.js'

export const useEnginesStore = defineStore('engines', () => {
  const engineDescriptors = ref([])
  /** YAML campaign S5: statuses/rows for SERVER-BACKED engines (YAML-only,
   *  marked generic_lifecycle on GET /api/engines), keyed by engine id.
   *  The named refs below stay exclusive to the nine local engines. */
  const serverStatuses = ref({})
  const serverVersions = ref({})
  /** Version rows for the nine local engines, keyed by engine id (R89). The
   *  historical per-engine names below are WRITABLE ALIASES into this map, so
   *  existing consumers keep reading/assigning them while the store itself
   *  carries one keyed map driven by the descriptors' `versionsKey`. */
  const engineVersionsById = ref({})
  /** Per-engine GET status, keyed by engine id (R81: one keyed map instead of
   *  one named ref per engine; `nullSafeStatus` still decides whether a null
   *  payload becomes `{}`). */
  const engineStatusById = ref({})
  const cudaStatus = ref({})
  const gpuInfo = ref({})
  const systemStatus = ref({})
  const loading = ref(false)
  /** Full diff from GET /api/runtime/pending (expensive; load on demand in the apply dialog). */
  const runtimePlan = ref({
    applicable: false,
    pending: false,
    changes: [],
    reason: null,
  })
  /** Cheap badge state from GET /api/runtime/stale */
  const runtimePlanState = ref({
    applicable: false,
    stale: false,
  })
  let runtimePlanStateRequest = null
  let runtimePlanStateEpoch = 0
  let gpuInfoRequest = null

  // --- llama.cpp versions ---

  async function fetchEngineDescriptors() {
    const { data } = await axios.get('/api/engines')
    engineDescriptors.value = Array.isArray(data?.engines) ? data.engines : []
    // YAML campaign S5: refresh the label/cancel overlay so pure lookup
    // helpers (toasts, delete dialogs, task-cancel routing) see
    // YAML-onboarded engines without any local table entry.
    applyServerDescriptors(engineDescriptors.value)
    return engineDescriptors.value
  }

  async function verifyLlamaVersion(versionId) {
    const raw = String(versionId || '')
    const name = raw.includes(':') ? raw.split(':').slice(1).join(':') : raw
    const { data } = await axios.get(`/api/llama-versions/verify/${encodeURIComponent(name)}`)
    return data
  }

  async function fetchBuildCapabilities() {
    const { data } = await axios.get('/api/llama-versions/build-capabilities')
    return data
  }

  // Version rows carry a repository_source display name (the second engine
  // identity system); the registry owns those names and this partitions the
  // row list generically. llama.cpp also owns untagged rows.
  //
  // R84/R89: the descriptor's `versionsKey` is the only name → engine mapping;
  // the store carries one keyed map and exposes writable computeds under the
  // historic names so consumers (and tests that ASSIGN them) keep working.
  const ENGINE_ID_BY_VERSIONS_KEY = Object.fromEntries(
    Object.entries(ENGINE_DESCRIPTORS)
      .filter(([, descriptor]) => descriptor.versionsKey)
      .map(([engineId, descriptor]) => [descriptor.versionsKey, engineId]),
  )

  const versionAliases = {}
  for (const [aliasKey, engineId] of Object.entries(ENGINE_ID_BY_VERSIONS_KEY)) {
    versionAliases[aliasKey] = computed({
      get: () => engineVersionsById.value[engineId] ?? [],
      set: (value) => {
        engineVersionsById.value[engineId] = value
      },
    })
  }

  async function fetchLlamaVersions() {
    const { data } = await axios.get('/api/llama-versions')
    const all = Array.isArray(data) ? data : []
    for (const [engineId, descriptor] of Object.entries(ENGINE_DESCRIPTORS)) {
      if (!descriptor.versionsKey) continue
      engineVersionsById.value[engineId] = all.filter((v) =>
        descriptor.untaggedSourceOwner
          ? !v.repository_source || v.repository_source === descriptor.repoSource
          : v.repository_source === descriptor.repoSource,
      )
    }
    // YAML campaign S5: server-backed (YAML-onboarded) engines get their
    // row partition from the server descriptor's repository_source —
    // id-keyed dynamic map, no named refs, no local table entry.
    const nextServerVersions = { ...serverVersions.value }
    for (const descriptor of serverBackedEngines(engineDescriptors.value)) {
      const source = descriptor.repository_source || descriptor.id
      nextServerVersions[descriptor.id] = all.filter((v) => v.repository_source === source)
    }
    serverVersions.value = nextServerVersions
  }

  async function checkLlamaCppUpdates() {
    const { data } = await axios.get(nativeGenericApiPath('llama_cpp', 'check-updates'))
    return data
  }

  async function checkIkLlamaUpdates() {
    // Engine travels in the path; the legacy ?source= query is retired.
    const { data } = await axios.get(nativeGenericApiPath('ik_llama', 'check-updates'))
    return data
  }

  async function checkUnslothLlamaUpdates() {
    const { data } = await axios.get(nativeGenericApiPath('unsloth_llama', 'check-updates'))
    return data
  }

  async function installUnslothLlama(params = {}) {
    const { data } = await axios.post(nativeGenericApiPath('unsloth_llama', 'install'), params)
    return data
  }

  async function cancelUnslothLlama(taskId) {
    const { data } = await axios.post(nativeGenericApiPath('unsloth_llama', 'cancel'), {
      task_id: taskId,
    })
    return data
  }

  async function checkLmdeployUpdates() {
    return venvCheckUpdates('lmdeploy')
  }

  async function checkOnecatVllmUpdates() {
    return venvCheckUpdates('1cat_vllm')
  }

  async function checkAudioCppUpdates() {
    const { data } = await axios.get('/api/audio-cpp/check-updates')
    return data
  }

  async function fetchAudioCppStatus() {
    const { data } = await axios.get('/api/audio-cpp/status')
    const value = data || {}
    engineStatusById.value = { ...engineStatusById.value, audio_cpp: value }
    return value
  }

  async function fetchAudioCppBuildSettings() {
    const { data } = await axios.get('/api/audio-cpp/build-settings')
    return data
  }

  async function fetchAudioCppBuildOptions() {
    return fetchBuildOptionsForEngine('audio_cpp')
  }

  /** R95: catalog fetch dispatched on the engine's DECLARED `buildOptionsPath`. */
  async function fetchBuildOptionsForEngine(engineId) {
    const path = buildOptionsPath(engineId)
    if (!path) return fetchBuildOptions(engineId)
    const { data } = await axios.get(path, { params: { engine: engineId } })
    return data
  }

  async function saveAudioCppBuildSettings(settings) {
    const { data } = await axios.put('/api/audio-cpp/build-settings', settings)
    return data
  }

  async function buildAudioCppSource(params = {}) {
    const { data } = await axios.post('/api/audio-cpp/build-source', params)
    return data
  }

  async function updateAudioCpp(params = {}) {
    const { data } = await axios.post('/api/audio-cpp/update', params)
    return data
  }

  async function fetchBuildSettings(engine) {
    if (NATIVE_GENERIC_VERBS[engine]?.includes('build-settings')) {
      const { data } = await axios.get(nativeGenericApiPath(engine, 'build-settings'))
      return data
    }
    const { data } = await axios.get('/api/llama-versions/build-settings', {
      params: { engine },
    })
    return data
  }

  async function fetchBuildOptions(engine) {
    const { data } = await axios.get('/api/llama-versions/build-options', {
      params: engine ? { engine } : {},
    })
    return data
  }

  async function saveBuildSettings(engine, settings) {
    if (NATIVE_GENERIC_VERBS[engine]?.includes('build-settings')) {
      const { data } = await axios.put(nativeGenericApiPath(engine, 'build-settings'), settings)
      return data
    }
    const { data } = await axios.put('/api/llama-versions/build-settings', settings, {
      params: { engine },
    })
    return data
  }

  // Native install/build verbs ride the generic router (same implementations
  // behind verb overrides; byte-equality pinned by
  // backend/tests/test_native_engine_lifecycle.py). Bodies are unchanged —
  // update_engine/build_source still read engine and repository_source from
  // the payload, so the engine id in the path is routing, not data.
  async function updateEngine(engine, params = {}) {
    const { data } = await axios.post(nativeGenericApiPath(engine, 'install'), {
      engine,
      ...params,
    })
    await fetchLlamaVersions()
    return data
  }

  // repository_source ('llama.cpp' | 'ik_llama.cpp' | absent -> llama.cpp)
  // selects the path's engine id; the payload keeps carrying the field.
  async function buildSource(params) {
    const engine = params.repository_source === 'ik_llama.cpp' ? 'ik_llama' : 'llama_cpp'
    const { data } = await axios.post(nativeGenericApiPath(engine, 'install-source'), params)
    return data
  }

  // llama.cpp and ik_llama builds share one BuildTaskManager and one cancel
  // implementation; the engine id in the path is routing, not behavior.
  async function cancelSourceBuild(taskId) {
    const { data } = await axios.post(nativeGenericApiPath('llama_cpp', 'cancel'), {
      task_id: taskId,
    })
    return data
  }

  async function syncVersion(versionId) {
    const { data } = await axios.post('/api/llama-versions/versions/sync', {
      version_id: versionId,
    })
    return data
  }

  async function retryVersion(versionId) {
    const { data } = await axios.post('/api/llama-versions/versions/retry', {
      version_id: versionId,
    })
    await fetchLlamaVersions()
    return data
  }

  async function saveVersionBuildConfig(versionId, buildConfig) {
    const { data } = await axios.put('/api/llama-versions/versions/build-config', {
      version_id: versionId,
      build_config: buildConfig,
    })
    await fetchLlamaVersions()
    return data
  }

  async function scanEngineParams(engine, version = null, options = {}) {
    const body = { engine }
    if (version) body.version = version
    if (options?.modelId) body.model_id = options.modelId
    const { data } = await axios.post('/api/llama-versions/scan-engine-params', body)
    return data
  }

  // Post-activate/post-delete status fan-out — ONE declarative copy of what
  // used to be two hand-rolled if-chains. Byte-preserved: lmdeploy and 1Cat
  // are independent containment checks; the SGLang panel family is
  // exclusive-first-match (sglang_v100 must precede sglang); audio refreshes
  // on activate only. The legacy quirk stays on purpose: 1Cat version ids
  // contain "vllm", so they hit the exclusive chain's vllm branch too and
  // refresh vLLM status as well. Every sequence is pinned in
  // engines.venv.test.js ("version status refresh fan-out").
  const STATUS_REFRESH_INDEPENDENT = [
    { token: 'lmdeploy', fetch: () => fetchLmdeployStatus() },
    { token: '1cat_vllm', fetch: () => fetchOnecatVllmStatus() },
  ]
  const STATUS_REFRESH_EXCLUSIVE = [
    { token: 'sglang_v100', fetch: () => fetchSglangStatus('sglang_v100') },
    { token: 'sglang_sxm2', fetch: () => fetchSglangStatus('sglang_sxm2') },
    { token: 'sglang_sm70', fetch: () => fetchSglangStatus('sglang_sm70') },
    { token: 'sglang', fetch: () => fetchSglangStatus('sglang') },
    { token: 'vllm', fetch: () => fetchSglangStatus('vllm') },
  ]
  const STATUS_REFRESH_ACTIVATE_ONLY = [{ token: 'audio_cpp', fetch: () => fetchAudioCppStatus() }]

  async function refreshStatusesForVersion(versionId, { audio = false } = {}) {
    const text = String(versionId)
    for (const rule of STATUS_REFRESH_INDEPENDENT) {
      if (text.includes(rule.token)) await rule.fetch()
    }
    for (const rule of STATUS_REFRESH_EXCLUSIVE) {
      if (text.includes(rule.token)) {
        await rule.fetch()
        break
      }
    }
    if (audio) {
      for (const rule of STATUS_REFRESH_ACTIVATE_ONLY) {
        if (text.includes(rule.token)) await rule.fetch()
      }
    }
  }

  async function activateVersion(versionId, { onStage } = {}) {
    const { data } = await axios.post('/api/llama-versions/versions/activate', {
      version_id: versionId,
    })
    if (data?.background && data?.operation_id) {
      // audio.cpp activation re-probes binaries in the background; the active
      // switch is already applied. Wait for the scan/refresh to finish so the
      // status refresh below sees fresh capability data.
      const { useProgressStore } = await import('./progress')
      const progress = useProgressStore()
      const unsubscribe = onStage
        ? progress.subscribe('task_updated', (task) => {
            if (task?.task_id === data.operation_id && task?.message) onStage(task.message)
          })
        : null
      try {
        await progress.waitForTask(data.operation_id)
      } finally {
        if (unsubscribe) unsubscribe()
      }
    }
    await fetchLlamaVersions()
    await refreshStatusesForVersion(versionId, { audio: true })
    fetchRuntimePlanState()
  }

  async function deleteVersion(versionId, params = {}) {
    const query = Object.fromEntries(
      Object.entries(params).filter(
        ([, value]) => value !== undefined && value !== null && value !== '',
      ),
    )
    await axios.delete(`/api/llama-versions/${encodeURIComponent(versionId)}`, { params: query })
    await fetchLlamaVersions()
    await refreshStatusesForVersion(versionId)
    fetchRuntimePlanState()
  }

  // --- CUDA ---

  async function fetchCudaStatus() {
    const { data } = await axios.get('/api/llama-versions/cuda-status')
    cudaStatus.value = data
    return data
  }

  async function installCuda(params) {
    const { data } = await axios.post('/api/llama-versions/cuda-install', params)
    return data
  }

  async function uninstallCuda(params = {}) {
    const { data } = await axios.post('/api/llama-versions/cuda-uninstall', params)
    await fetchCudaStatus()
    return data
  }

  // --- venv lifecycle (registry-driven) -----------------------------------
  // The historical per-engine export names below are thin wrappers over
  // these one-implementation helpers; engines/registry.js decides the URL
  // segment and the per-family null-handling divergence.

  /** Generic keyed reads (R82): new consumers take an engine id, not a ref. */
  function versionsForEngine(engineId) {
    return engineVersionsById.value[engineId] ?? []
  }

  function statusForEngine(engineId) {
    return engineStatusById.value[engineId] ?? {}
  }

  async function venvCheckUpdates(engine) {
    const { data } = await axios.get(venvApiPathFor(engine, 'check-updates'))
    return data
  }

  /** URL decision for the shared venv helpers: server-backed (YAML)
   *  engines route through the generic router ONLY; the nine keep their
   *  exact legacy paths (byte-pinned by engines.v3.test.js). Unsupported
   *  verbs 405 server-side — the client mirrors that trust boundary. */
  function venvApiPathFor(engine, verb) {
    if (serverBackedDescriptor(engine, engineDescriptors.value)) {
      return serverBackedApiPath(engine, verb)
    }
    return venvApiPath(engine, verb)
  }

  async function venvFetchStatus(engine) {
    const serverDescriptor = serverBackedDescriptor(engine, engineDescriptors.value)
    if (serverDescriptor) {
      // Server-backed engine: generic router only, dynamic map only.
      const { data } = await axios.get(serverBackedApiPath(engine, 'status'))
      serverStatuses.value = { ...serverStatuses.value, [engine]: data || {} }
      return data || {}
    }
    const descriptor = venvDescriptor(engine)
    const { data } = await axios.get(venvApiPath(engine, 'status'))
    const value = descriptor.nullSafeStatus ? data || {} : data
    // R81: one keyed map (engine id) instead of a per-engine named ref.
    engineStatusById.value = { ...engineStatusById.value, [engine]: value }
    return value
  }

  /** Poll every server-backed engine whose lifecycle serves `status`
   *  (YAML campaign S5: the generic refresh loop is payload-driven). */
  async function fetchServerBackedStatuses() {
    for (const descriptor of serverBackedEngines(engineDescriptors.value)) {
      if (!(descriptor.generic_verbs || []).includes('status')) continue
      try {
        await venvFetchStatus(descriptor.id)
      } catch (error) {
        console.error(`Failed to fetch ${descriptor.id} engine status:`, error)
      }
    }
  }

  async function venvGetBuildSettings(engine) {
    const { data } = await axios.get(venvApiPathFor(engine, 'build-settings'))
    return data
  }

  async function venvSaveBuildSettings(engine, settings) {
    const { data } = await axios.put(venvApiPathFor(engine, 'build-settings'), settings)
    return data
  }

  async function venvInstall(engine, params) {
    const { data } = await axios.post(venvApiPathFor(engine, 'install'), params)
    return data
  }

  async function venvInstallFromSource(engine, params) {
    const { data } = await axios.post(venvApiPathFor(engine, 'install-source'), params)
    return data
  }

  async function venvRemove(engine, params) {
    await axios.post(venvApiPathFor(engine, 'remove'), params)
    await venvFetchStatus(engine)
    await fetchLlamaVersions()
  }

  /** Cancel a running venv/generic operation (server-backed aware). */
  async function venvCancel(engine, taskId) {
    const { data } = await axios.post(venvApiPathFor(engine, 'cancel'), {
      task_id: taskId,
    })
    return data
  }

  // --- LMDeploy ---

  async function fetchLmdeployStatus() {
    return venvFetchStatus('lmdeploy')
  }

  async function fetchLmdeployBuildSettings() {
    return venvGetBuildSettings('lmdeploy')
  }

  async function saveLmdeployBuildSettings(settings) {
    return venvSaveBuildSettings('lmdeploy', settings)
  }

  async function installLmdeploy(params = {}) {
    return venvInstall('lmdeploy', params)
  }

  async function installLmdeployFromSource(params) {
    return venvInstallFromSource('lmdeploy', params)
  }

  async function removeLmdeploy(params = {}) {
    await venvRemove('lmdeploy', params)
  }

  // --- 1Cat-vLLM ---

  async function fetchOnecatVllmStatus() {
    return venvFetchStatus('1cat_vllm')
  }

  async function fetchOnecatVllmBuildSettings() {
    return venvGetBuildSettings('1cat_vllm')
  }

  async function saveOnecatVllmBuildSettings(settings) {
    return venvSaveBuildSettings('1cat_vllm', settings)
  }

  async function installOnecatVllm(params = {}) {
    return venvInstall('1cat_vllm', params)
  }

  async function installOnecatVllmFromSource(params) {
    return venvInstallFromSource('1cat_vllm', params)
  }

  async function removeOnecatVllm(params = {}) {
    await venvRemove('1cat_vllm', params)
  }

  // --- SGLang / SGLang V100 / vLLM (one installer family, three engines) ---

  async function fetchSglangStatus(engine = 'sglang') {
    return venvFetchStatus(engine)
  }

  async function checkSglangUpdates(engine = 'sglang') {
    return venvCheckUpdates(engine)
  }

  async function fetchSglangBuildSettings(engine = 'sglang') {
    return venvGetBuildSettings(engine)
  }

  async function saveSglangBuildSettings(engine = 'sglang', settings = {}) {
    return venvSaveBuildSettings(engine, settings)
  }

  async function installSglang(engine = 'sglang', params = {}) {
    return venvInstall(engine, params)
  }

  async function installSglangFromSource(engine = 'sglang', params = {}) {
    return venvInstallFromSource(engine, params)
  }

  // --- GPU / System ---

  async function fetchGpuInfo() {
    if (gpuInfoRequest) return gpuInfoRequest
    gpuInfoRequest = (async () => {
      const { data } = await axios.get('/api/gpu-info')
      gpuInfo.value = data
      return data
    })()
    try {
      return await gpuInfoRequest
    } finally {
      gpuInfoRequest = null
    }
  }

  /** Lightweight GPU list (cached at server startup) for model-config GPU pinning. */
  async function fetchGpuList() {
    const { data } = await axios.get('/api/gpu-list')
    return data
  }

  async function fetchSystemStatus({ signal } = {}) {
    loading.value = true
    try {
      const [statusRes, gpuData] = await Promise.all([
        signal ? axios.get('/api/status', { signal }) : axios.get('/api/status'),
        fetchGpuInfo(),
      ])
      if (signal?.aborted) return
      systemStatus.value = statusRes.data
      gpuInfo.value = gpuData
    } catch (err) {
      if (!signal?.aborted) console.error('Failed to fetch system status:', err)
      throw err
    } finally {
      loading.value = false
    }
  }

  async function fetchSystemStats() {
    const { data } = await axios.get('/api/system-stats')
    if (data?.system) {
      systemStatus.value = { ...systemStatus.value, system: data.system }
    }
    return data?.system
  }

  async function fetchRuntimePlan() {
    try {
      const { data } = await axios.get('/api/runtime/pending')
      runtimePlan.value = {
        applicable: Boolean(data?.applicable),
        pending: Boolean(data?.pending),
        changes: Array.isArray(data?.changes) ? data.changes : [],
        reason: data?.reason ?? null,
        plan_id: data?.plan_id ?? null,
        launch_manifests: Boolean(data?.launch_manifests),
        requires_proxy_reload: Boolean(data?.requires_proxy_reload),
        migration_required: Boolean(data?.migration_required),
        models: Array.isArray(data?.models) ? data.models : [],
      }
    } catch (err) {
      console.warn('fetchRuntimePlan failed:', err)
    }
    return runtimePlan.value
  }

  function fetchRuntimePlanState() {
    if (runtimePlanStateRequest) return runtimePlanStateRequest
    const requestEpoch = runtimePlanStateEpoch
    const request = (async () => {
      try {
        const { data } = await axios.get('/api/runtime/stale')
        if (requestEpoch === runtimePlanStateEpoch) {
          runtimePlanState.value = {
            applicable: Boolean(data?.applicable),
            stale: Boolean(data?.stale),
          }
        }
      } catch (err) {
        console.warn('fetchRuntimePlanState failed:', err)
      } finally {
        if (runtimePlanStateRequest === request) {
          runtimePlanStateRequest = null
        }
      }
      return runtimePlanState.value
    })()
    runtimePlanStateRequest = request
    return request
  }

  function invalidateRuntimePlan() {
    runtimePlanStateEpoch += 1
    runtimePlanStateRequest = null
    runtimePlanState.value = {
      applicable: true,
      stale: true,
    }
  }

  async function fetchAll() {
    await Promise.allSettled([
      fetchEngineDescriptors(),
      fetchLlamaVersions(),
      fetchCudaStatus(),
      fetchLmdeployStatus(),
      fetchOnecatVllmStatus(),
      ...SG_PANEL_ENGINE_IDS.map((engine) => fetchSglangStatus(engine)),
      fetchAudioCppStatus(),
      fetchSystemStatus(),
      fetchRuntimePlanState(),
    ])
    // YAML campaign S5: payload-driven follow-up — server-backed
    // (YAML-onboarded) engines are polled from the descriptor list, not a
    // hardcoded id list, so onboarding one needs zero frontend edits.
    // (Descriptors must have arrived first; row partitioning re-runs over
    // the shared /api/llama-versions list.)
    if (serverBackedEngines(engineDescriptors.value).length) {
      await Promise.allSettled([fetchLlamaVersions(), fetchServerBackedStatuses()])
    }
  }

  return {
    engineDescriptors,
    fetchEngineDescriptors,
    serverStatuses,
    serverVersions,
    fetchServerBackedStatuses,
    // Generic verb surface, server-backing aware (YAML campaign S5): these
    // drive BOTH the nine and YAML-onboarded engines via one code path.
    venvCheckUpdates,
    venvFetchStatus,
    venvGetBuildSettings,
    venvSaveBuildSettings,
    venvInstall,
    venvInstallFromSource,
    venvRemove,
    venvCancel,
    // R82: keyed reads for generic consumers (views/panels take an engine id).
    versionsForEngine,
    statusForEngine,
    engineVersionsById,
    ...versionAliases,
    engineStatusById,
    cudaStatus,
    gpuInfo,
    systemStatus,
    runtimePlan,
    runtimePlanState,
    loading,

    fetchLlamaVersions,
    verifyLlamaVersion,
    fetchBuildCapabilities,
    checkLlamaCppUpdates,
    checkIkLlamaUpdates,
    checkUnslothLlamaUpdates,
    installUnslothLlama,
    cancelUnslothLlama,
    checkLmdeployUpdates,
    checkOnecatVllmUpdates,
    checkAudioCppUpdates,
    fetchBuildSettings,
    fetchBuildOptions,
    fetchBuildOptionsForEngine,
    saveBuildSettings,
    updateEngine,
    buildSource,
    cancelSourceBuild,
    syncVersion,
    retryVersion,
    saveVersionBuildConfig,
    scanEngineParams,
    activateVersion,
    deleteVersion,

    fetchCudaStatus,
    installCuda,
    uninstallCuda,

    fetchLmdeployStatus,
    fetchLmdeployBuildSettings,
    saveLmdeployBuildSettings,
    installLmdeploy,
    installLmdeployFromSource,
    removeLmdeploy,

    fetchOnecatVllmStatus,
    fetchOnecatVllmBuildSettings,
    saveOnecatVllmBuildSettings,
    installOnecatVllm,
    installOnecatVllmFromSource,
    removeOnecatVllm,

    fetchSglangStatus,
    checkSglangUpdates,
    fetchSglangBuildSettings,
    saveSglangBuildSettings,
    installSglang,
    installSglangFromSource,

    fetchAudioCppStatus,
    fetchAudioCppBuildSettings,
    fetchAudioCppBuildOptions,
    saveAudioCppBuildSettings,
    buildAudioCppSource,
    updateAudioCpp,

    fetchGpuList,
    fetchSystemStats,
    fetchSystemStatus,
    fetchRuntimePlan,
    fetchRuntimePlanState,
    invalidateRuntimePlan,
    fetchAll,
  }
})
