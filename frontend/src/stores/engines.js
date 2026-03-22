import { ref } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

export const useEnginesStore = defineStore('engines', () => {
  const llamaVersions = ref([])
  const ikLlamaVersions = ref([])
  const lmdeployVersions = ref([])
  const lmdeployStatus = ref({})
  const cudaStatus = ref({})
  const gpuInfo = ref({})
  const systemStatus = ref({})
  const loading = ref(false)
  /** llama-swap YAML vs DB: { applicable, pending, changes[], reason? } */
  const swapConfigPending = ref({
    applicable: false,
    pending: false,
    changes: [],
    reason: null,
  })

  // --- llama.cpp versions ---

  async function fetchLlamaVersions() {
    const { data } = await axios.get('/api/llama-versions')
    const all = Array.isArray(data) ? data : []
    llamaVersions.value = all.filter(v => !v.repository_source || v.repository_source === 'llama.cpp')
    ikLlamaVersions.value = all.filter(v => v.repository_source === 'ik_llama.cpp')
    lmdeployVersions.value = all.filter(v => v.repository_source === 'LMDeploy')
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

  async function scanEngineParams(engine, version = null) {
    const body = { engine }
    if (version) body.version = version
    const { data } = await axios.post('/api/llama-versions/scan-engine-params', body)
    return data
  }

  async function activateVersion(versionId) {
    await axios.post('/api/llama-versions/versions/activate', { version_id: versionId })
    await fetchLlamaVersions()
    if (String(versionId).includes('lmdeploy')) {
      await fetchLmdeployStatus()
    }
    fetchSwapConfigPending()
  }

  async function deleteVersion(versionId) {
    await axios.delete(`/api/llama-versions/${encodeURIComponent(versionId)}`)
    await fetchLlamaVersions()
    if (String(versionId).includes('lmdeploy')) {
      await fetchLmdeployStatus()
    }
    fetchSwapConfigPending()
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

  // --- GPU / System ---

  async function fetchGpuInfo() {
    const { data } = await axios.get('/api/gpu-info')
    gpuInfo.value = data
    return data
  }

  async function fetchSystemStatus() {
    loading.value = true
    try {
      const [statusRes, gpuRes] = await Promise.all([
        axios.get('/api/status'),
        axios.get('/api/gpu-info'),
      ])
      systemStatus.value = statusRes.data
      gpuInfo.value = gpuRes.data
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

  async function applySwapConfig() {
    const { data } = await axios.post('/api/llama-swap/apply-config')
    await fetchSwapConfigPending()
    await fetchSystemStatus()
    return data
  }

  // --- Bulk fetch ---

  async function fetchAll() {
    await Promise.allSettled([
      fetchLlamaVersions(),
      fetchCudaStatus(),
      fetchLmdeployStatus(),
      fetchSystemStatus(),
      fetchSwapConfigPending(),
    ])
  }

  return {
    llamaVersions,
    ikLlamaVersions,
    lmdeployVersions,
    lmdeployStatus,
    cudaStatus,
    gpuInfo,
    systemStatus,
    swapConfigPending,
    loading,

    fetchLlamaVersions,
    checkLlamaCppUpdates,
    checkIkLlamaUpdates,
    checkLmdeployUpdates,
    fetchBuildSettings,
    saveBuildSettings,
    updateEngine,
    buildSource,
    cancelSourceBuild,
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

    fetchGpuInfo,
    fetchSystemStatus,
    fetchSwapConfigPending,
    applySwapConfig,
    fetchAll,
  }
})
