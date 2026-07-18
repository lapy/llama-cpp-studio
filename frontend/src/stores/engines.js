import { ref } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

export const useEnginesStore = defineStore('engines', () => {
  const engineDescriptors = ref([])
  const llamaVersions = ref([])
  const ikLlamaVersions = ref([])
  const lmdeployVersions = ref([])
  const lmdeployStatus = ref({})
  const onecatVllmVersions = ref([])
  const onecatVllmStatus = ref({})
  const audioCppVersions = ref([])
  const audioCppStatus = ref({})
  const cudaStatus = ref({})
  const gpuInfo = ref({})
  const systemStatus = ref({})
  const loading = ref(false)
  /** Full diff from GET /api/llama-swap/pending (expensive; load on demand in the apply dialog). */
  const swapConfigPending = ref({
    applicable: false,
    pending: false,
    changes: [],
    reason: null,
  })
  /** Cheap badge state from GET /api/llama-swap/stale */
  const swapConfigStale = ref({
    applicable: false,
    stale: false,
  })
  let swapConfigStaleRequest = null
  let swapConfigStaleEpoch = 0
  let gpuInfoRequest = null

  // --- llama.cpp versions ---

  async function fetchEngineDescriptors() {
    const { data } = await axios.get('/api/engines')
    engineDescriptors.value = Array.isArray(data?.engines) ? data.engines : []
    return engineDescriptors.value
  }

  async function fetchLlamaVersions() {
    const { data } = await axios.get('/api/llama-versions')
    const all = Array.isArray(data) ? data : []
    llamaVersions.value = all.filter(v => !v.repository_source || v.repository_source === 'llama.cpp')
    ikLlamaVersions.value = all.filter(v => v.repository_source === 'ik_llama.cpp')
    lmdeployVersions.value = all.filter(v => v.repository_source === 'LMDeploy')
    onecatVllmVersions.value = all.filter(v => v.repository_source === '1Cat-vLLM')
    audioCppVersions.value = all.filter(v => v.repository_source === 'audio.cpp')
  }

  async function checkLlamaCppUpdates() {
    const { data } = await axios.get('/api/llama-versions/check-updates')
    return data
  }

  async function checkIkLlamaUpdates() {
    const { data } = await axios.get('/api/llama-versions/check-updates', {
      params: { source: 'ik_llama' },
    })
    return data
  }

  async function checkLmdeployUpdates() {
    const { data } = await axios.get('/api/lmdeploy/check-updates')
    return data
  }

  async function checkOnecatVllmUpdates() {
    const { data } = await axios.get('/api/1cat-vllm/check-updates')
    return data
  }

  async function checkAudioCppUpdates() {
    const { data } = await axios.get('/api/audio-cpp/check-updates')
    return data
  }

  async function fetchAudioCppStatus() {
    const { data } = await axios.get('/api/audio-cpp/status')
    audioCppStatus.value = data || {}
    return audioCppStatus.value
  }

  async function fetchAudioCppBuildSettings() {
    const { data } = await axios.get('/api/audio-cpp/build-settings')
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

  async function cancelAudioCppBuild(taskId) {
    const { data } = await axios.post('/api/audio-cpp/cancel', { task_id: taskId })
    return data
  }

  async function migrateAudioCppDefaults(params = {}) {
    const { data } = await axios.post('/api/audio-cpp/migrate-defaults', params)
    return data
  }

  async function fetchBuildSettings(engine) {
    const { data } = await axios.get('/api/llama-versions/build-settings', {
      params: { engine },
    })
    return data
  }

  async function saveBuildSettings(engine, settings) {
    const { data } = await axios.put('/api/llama-versions/build-settings', settings, {
      params: { engine },
    })
    return data
  }

  async function updateEngine(engine, params = {}) {
    const { data } = await axios.post('/api/llama-versions/update', {
      engine,
      ...params,
    })
    await fetchLlamaVersions()
    return data
  }

  async function buildSource(params) {
    const { data } = await axios.post('/api/llama-versions/build-source', params)
    return data
  }

  async function cancelSourceBuild(taskId) {
    const { data } = await axios.post('/api/llama-versions/build-cancel', {
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

  async function scanEngineParams(engine, version = null, options = {}) {
    const body = { engine }
    if (version) body.version = version
    if (options?.modelId) body.model_id = options.modelId
    const { data } = await axios.post('/api/llama-versions/scan-engine-params', body)
    return data
  }

  async function activateVersion(versionId) {
    await axios.post('/api/llama-versions/versions/activate', { version_id: versionId })
    await fetchLlamaVersions()
    if (String(versionId).includes('lmdeploy')) {
      await fetchLmdeployStatus()
    }
    if (String(versionId).includes('1cat_vllm')) {
      await fetchOnecatVllmStatus()
    }
    if (String(versionId).includes('audio_cpp')) {
      await fetchAudioCppStatus()
    }
    fetchSwapConfigStale()
  }

  async function deleteVersion(versionId) {
    await axios.delete(`/api/llama-versions/${encodeURIComponent(versionId)}`)
    await fetchLlamaVersions()
    if (String(versionId).includes('lmdeploy')) {
      await fetchLmdeployStatus()
    }
    if (String(versionId).includes('1cat_vllm')) {
      await fetchOnecatVllmStatus()
    }
    fetchSwapConfigStale()
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

  async function fetchCudaLogs() {
    const { data } = await axios.get('/api/llama-versions/cuda-logs')
    return data
  }

  // --- LMDeploy ---

  async function fetchLmdeployStatus() {
    const { data } = await axios.get('/api/lmdeploy/status')
    lmdeployStatus.value = data
    return data
  }

  async function installLmdeploy(params = {}) {
    const { data } = await axios.post('/api/lmdeploy/install', params)
    return data
  }

  async function installLmdeployFromSource(params) {
    const { data } = await axios.post('/api/lmdeploy/install-source', params)
    return data
  }

  async function removeLmdeploy() {
    await axios.post('/api/lmdeploy/remove')
    await fetchLmdeployStatus()
    await fetchLlamaVersions()
  }

  // --- 1Cat-vLLM ---

  async function fetchOnecatVllmStatus() {
    const { data } = await axios.get('/api/1cat-vllm/status')
    onecatVllmStatus.value = data
    return data
  }

  async function installOnecatVllm(params = {}) {
    const { data } = await axios.post('/api/1cat-vllm/install', params)
    return data
  }

  async function installOnecatVllmFromSource(params) {
    const { data } = await axios.post('/api/1cat-vllm/install-source', params)
    return data
  }

  async function removeOnecatVllm() {
    await axios.post('/api/1cat-vllm/remove')
    await fetchOnecatVllmStatus()
    await fetchLlamaVersions()
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

  async function fetchSystemStatus() {
    loading.value = true
    try {
      const [statusRes, gpuData] = await Promise.all([
        axios.get('/api/status'),
        fetchGpuInfo(),
      ])
      systemStatus.value = statusRes.data
      gpuInfo.value = gpuData
    } catch (err) {
      console.error('Failed to fetch system status:', err)
      throw err
    } finally {
      loading.value = false
    }
  }

  async function fetchSwapConfigPending() {
    try {
      const { data } = await axios.get('/api/llama-swap/pending')
      swapConfigPending.value = {
        applicable: Boolean(data?.applicable),
        pending: Boolean(data?.pending),
        changes: Array.isArray(data?.changes) ? data.changes : [],
        reason: data?.reason ?? null,
      }
    } catch (err) {
      console.warn('fetchSwapConfigPending failed:', err)
    }
    return swapConfigPending.value
  }

  function fetchSwapConfigStale() {
    if (swapConfigStaleRequest) return swapConfigStaleRequest
    const requestEpoch = swapConfigStaleEpoch
    const request = (async () => {
      try {
        const { data } = await axios.get('/api/llama-swap/stale')
        if (requestEpoch === swapConfigStaleEpoch) {
          swapConfigStale.value = {
            applicable: Boolean(data?.applicable),
            stale: Boolean(data?.stale),
          }
        }
      } catch (err) {
        console.warn('fetchSwapConfigStale failed:', err)
      } finally {
        if (swapConfigStaleRequest === request) {
          swapConfigStaleRequest = null
        }
      }
      return swapConfigStale.value
    })()
    swapConfigStaleRequest = request
    return request
  }

  function markSwapConfigStaleLocal() {
    swapConfigStaleEpoch += 1
    swapConfigStaleRequest = null
    swapConfigStale.value = {
      applicable: true,
      stale: true,
    }
  }

  function clearSwapConfigStaleLocal() {
    swapConfigStaleEpoch += 1
    swapConfigStaleRequest = null
    swapConfigStale.value = {
      applicable: true,
      stale: false,
    }
  }

  async function applySwapConfig() {
    const { data } = await axios.post('/api/llama-swap/apply-config')
    clearSwapConfigStaleLocal()
    swapConfigPending.value = {
      applicable: true,
      pending: false,
      changes: [],
      reason: null,
    }
    void fetchSwapConfigStale()
    void fetchSystemStatus()
    return data
  }

  // --- Bulk fetch ---

  async function fetchAll() {
    await Promise.allSettled([
      fetchEngineDescriptors(),
      fetchLlamaVersions(),
      fetchCudaStatus(),
      fetchLmdeployStatus(),
      fetchOnecatVllmStatus(),
      fetchAudioCppStatus(),
      fetchSystemStatus(),
      fetchSwapConfigStale(),
    ])
  }

  return {
    engineDescriptors,
    fetchEngineDescriptors,
    llamaVersions,
    ikLlamaVersions,
    lmdeployVersions,
    lmdeployStatus,
    onecatVllmVersions,
    onecatVllmStatus,
    audioCppVersions,
    audioCppStatus,
    cudaStatus,
    gpuInfo,
    systemStatus,
    swapConfigPending,
    swapConfigStale,
    loading,

    fetchLlamaVersions,
    checkLlamaCppUpdates,
    checkIkLlamaUpdates,
    checkLmdeployUpdates,
    checkOnecatVllmUpdates,
    checkAudioCppUpdates,
    fetchBuildSettings,
    saveBuildSettings,
    updateEngine,
    buildSource,
    cancelSourceBuild,
    syncVersion,
    scanEngineParams,
    activateVersion,
    deleteVersion,

    fetchCudaStatus,
    installCuda,
    uninstallCuda,
    fetchCudaLogs,

    fetchLmdeployStatus,
    installLmdeploy,
    installLmdeployFromSource,
    removeLmdeploy,

    fetchOnecatVllmStatus,
    installOnecatVllm,
    installOnecatVllmFromSource,
    removeOnecatVllm,

    fetchAudioCppStatus,
    fetchAudioCppBuildSettings,
    saveAudioCppBuildSettings,
    buildAudioCppSource,
    updateAudioCpp,
    cancelAudioCppBuild,
    migrateAudioCppDefaults,

    fetchGpuInfo,
    fetchGpuList,
    fetchSystemStatus,
    fetchSwapConfigPending,
    fetchSwapConfigStale,
    markSwapConfigStaleLocal,
    clearSwapConfigStaleLocal,
    applySwapConfig,
    fetchAll,
  }
})
