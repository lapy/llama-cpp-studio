import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'

export const useLmdeployStore = defineStore('lmdeployInstaller', () => {
  let pollTimer = null
  const status = ref(null)
  const loading = ref(false)
  const installing = ref(false)
  const removing = ref(false)
  const logs = ref('')
  const logLoading = ref(false)
  const runtimeLogs = ref('')
  const runtimeLogLoading = ref(false)

  const fetchStatus = async () => {
    loading.value = true
    try {
      const response = await axios.get('/api/lmdeploy/status')
      status.value = response.data
    } catch (error) {
      console.error('Failed to fetch LMDeploy installer status:', error)
      throw error
    } finally {
      loading.value = false
    }
  }

  const fetchLogs = async (maxBytes = 8192) => {
    logLoading.value = true
    try {
      const response = await axios.get('/api/lmdeploy/logs', {
        params: { max_bytes: maxBytes }
      })
      logs.value = response.data?.log || ''
    } catch (error) {
      console.error('Failed to fetch LMDeploy installer logs:', error)
      throw error
    } finally {
      logLoading.value = false
    }
  }

  const fetchRuntimeLogs = async (maxBytes = 8192) => {
    runtimeLogLoading.value = true
    try {
      const response = await axios.get('/api/lmdeploy/runtime-logs', {
        params: { max_bytes: maxBytes }
      })
      runtimeLogs.value = response.data?.log || ''
    } catch (error) {
      console.error('Failed to fetch LMDeploy runtime logs:', error)
      throw error
    } finally {
      runtimeLogLoading.value = false
    }
  }

  const install = async (options = {}) => {
    installing.value = true
    try {
      await axios.post('/api/lmdeploy/install', options)
      await fetchStatus()
    } catch (error) {
      console.error('Failed to start LMDeploy install:', error)
      throw error
    } finally {
      installing.value = false
    }
  }

  const remove = async () => {
    removing.value = true
    try {
      await axios.post('/api/lmdeploy/remove')
      await fetchStatus()
    } catch (error) {
      console.error('Failed to start LMDeploy removal:', error)
      throw error
    } finally {
      removing.value = false
    }
  }

  const startPolling = (intervalMs = 4000) => {
    if (pollTimer) return
    pollTimer = setInterval(() => {
      fetchStatus().catch(() => {})
      fetchLogs().catch(() => {})
      fetchRuntimeLogs().catch(() => {})
    }, intervalMs)
  }

  const stopPolling = () => {
    if (pollTimer) {
      clearInterval(pollTimer)
      pollTimer = null
    }
  }

  return {
    status,
    logs,
    runtimeLogs,
    loading,
    logLoading,
    runtimeLogLoading,
    installing,
    removing,
    fetchStatus,
    fetchLogs,
    fetchRuntimeLogs,
    install,
    remove,
    startPolling,
    stopPolling
  }
})

