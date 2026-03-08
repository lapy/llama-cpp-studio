import { ref } from 'vue'
import { defineStore } from 'pinia'
import axios from 'axios'

export const useEnginesStore = defineStore('engines', () => {
  const llamaVersions = ref([])
  const ikLlamaVersions = ref([])
  const lmdeployStatus = ref({})
  const cudaStatus = ref({})
  const gpuInfo = ref({})
  const systemStatus = ref({})
  const loading = ref(false)

  // --- llama.cpp versions ---

  async function fetchLlamaVersions() {
    const { data } = await axios.get('/api/llama-versions')
    const all = Array.isArray(data) ? data : []
    llamaVersions.value = all.filter(v => !v.repository_source || v.repository_source === 'llama.cpp')
    ikLlamaVersions.value = all.filter(v => v.repository_source === 'ik_llama.cpp')
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

  async function fetchReleaseAssets(tagName) {
    const { data } = await axios.get(`/api/llama-versions/releases/${encodeURIComponent(tagName)}/assets`)
    return data
  }

  async function installRelease(params) {
    const { data } = await axios.post('/api/llama-versions/install-release', params)
    await fetchLlamaVersions()
    return data
  }

  async function buildSource(params) {
    const { data } = await axios.post('/api/llama-versions/build-source', params)
    await fetchLlamaVersions()
    return data
  }

  async function activateVersion(versionId) {
    await axios.post('/api/llama-versions/versions/activate', { version_id: versionId })
    await fetchLlamaVersions()
  }

  async function deleteVersion(versionId) {
    await axios.delete(`/api/llama-versions/${versionId}`)
    await fetchLlamaVersions()
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
  }

  async function fetchLmdeployLogs(maxBytes = 8192) {
    const { data } = await axios.get('/api/lmdeploy/logs', { params: { max_bytes: maxBytes } })
    return data
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

  // --- Bulk fetch ---

  async function fetchAll() {
    await Promise.allSettled([
      fetchLlamaVersions(),
      fetchCudaStatus(),
      fetchLmdeployStatus(),
      fetchSystemStatus(),
    ])
  }

  return {
    llamaVersions,
    ikLlamaVersions,
    lmdeployStatus,
    cudaStatus,
    gpuInfo,
    systemStatus,
    loading,

    fetchLlamaVersions,
    checkLlamaCppUpdates,
    checkIkLlamaUpdates,
    checkLmdeployUpdates,
    fetchReleaseAssets,
    installRelease,
    buildSource,
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
    fetchLmdeployLogs,

    fetchGpuInfo,
    fetchSystemStatus,
    fetchAll,
  }
})
