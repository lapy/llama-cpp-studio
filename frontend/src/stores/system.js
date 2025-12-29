import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export const useSystemStore = defineStore('system', () => {
  const systemStatus = ref({})
  const gpuInfo = ref({})
  const llamaVersions = ref([])
  const loading = ref(false)

  const fetchSystemStatus = async () => {
    loading.value = true
    try {
      const [statusResponse, gpuResponse] = await Promise.all([
        axios.get('/api/monitoring/status'),
        axios.get('/api/gpu-info')
      ])
      
      systemStatus.value = statusResponse.data
      gpuInfo.value = gpuResponse.data
    } catch (error) {
      console.error('Failed to fetch system status:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const fetchLlamaVersions = async () => {
    try {
      const response = await axios.get('/api/llama-versions')
      llamaVersions.value = response.data
    } catch (error) {
      console.error('Failed to fetch llama versions:', error)
      throw error
    }
  }

  const checkUpdates = async () => {
    try {
      const response = await axios.get('/api/llama-versions/check-updates')
      return response.data
    } catch (error) {
      console.error('Failed to check updates:', error)
      throw error
    }
  }

  const fetchReleaseAssets = async (tagName) => {
    try {
      const response = await axios.get(`/api/llama-versions/releases/${encodeURIComponent(tagName)}/assets`)
      return response.data
    } catch (error) {
      console.error('Failed to fetch release assets:', error)
      throw error
    }
  }

  const installRelease = async (tagName, assetId) => {
    try {
      const payload = { tag_name: tagName }
      if (assetId !== undefined && assetId !== null) {
        payload.asset_id = assetId
      }
      await axios.post('/api/llama-versions/install-release', payload)
      await fetchLlamaVersions()
    } catch (error) {
      console.error('Failed to install release:', error)
      throw error
    }
  }

  const buildSource = async (commitSha, patches = [], buildConfig = {}, repositorySource = 'llama.cpp', versionSuffix = null) => {
    try {
      const payload = {
        commit_sha: commitSha,
        patches,
        build_config: buildConfig,
        repository_source: repositorySource
      }
      if (versionSuffix) {
        payload.version_suffix = versionSuffix
      }
      await axios.post('/api/llama-versions/build-source', payload)
      await fetchLlamaVersions()
    } catch (error) {
      console.error('Failed to build from source:', error)
      throw error
    }
  }

  const activateVersion = async (versionId) => {
    try {
      await axios.post(`/api/llama-versions/${versionId}/activate`)
      await fetchLlamaVersions()
    } catch (error) {
      console.error('Failed to activate version:', error)
      throw error
    }
  }

  const deleteVersion = async (versionId) => {
    try {
      await axios.delete(`/api/llama-versions/${versionId}`)
      await fetchLlamaVersions()
    } catch (error) {
      console.error('Failed to delete version:', error)
      throw error
    }
  }

  const updateSystemStatus = (status) => {
    systemStatus.value = { ...systemStatus.value, ...status }
  }

  const updateGpuInfo = (gpuData) => {
    gpuInfo.value = { ...gpuInfo.value, ...gpuData }
  }

  const getCudaStatus = async () => {
    try {
      const response = await axios.get('/api/llama-versions/cuda-status')
      return response.data
    } catch (error) {
      console.error('Failed to get CUDA status:', error)
      throw error
    }
  }

  const installCuda = async (version = '12.6') => {
    try {
      await axios.post('/api/llama-versions/cuda-install', { version })
    } catch (error) {
      console.error('Failed to install CUDA:', error)
      throw error
    }
  }

  const getCudaLogs = async () => {
    try {
      const response = await axios.get('/api/llama-versions/cuda-logs')
      return response.data
    } catch (error) {
      console.error('Failed to get CUDA logs:', error)
      throw error
    }
  }

  const uninstallCuda = async (version = null) => {
    try {
      await axios.post('/api/llama-versions/cuda-uninstall', { version })
    } catch (error) {
      console.error('Failed to uninstall CUDA:', error)
      throw error
    }
  }

  return {
    systemStatus,
    gpuInfo,
    llamaVersions,
    loading,
    fetchSystemStatus,
    fetchLlamaVersions,
    checkUpdates,
    fetchReleaseAssets,
    installRelease,
    buildSource,
    activateVersion,
    deleteVersion,
    updateSystemStatus,
    updateGpuInfo,
    getCudaStatus,
    installCuda,
    uninstallCuda,
    getCudaLogs
  }
})
