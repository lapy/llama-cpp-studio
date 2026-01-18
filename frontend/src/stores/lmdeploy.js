import { defineStore } from 'pinia'
import { ref } from 'vue'
import axios from 'axios'
import { useWebSocketStore } from './websocket'

export const useLmdeployStore = defineStore('lmdeployInstaller', () => {
  const wsStore = useWebSocketStore()
  let statusUnsubscribe = null
  let installLogUnsubscribe = null
  let runtimeLogUnsubscribe = null
  
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

  const startWebSocketSubscriptions = () => {
    // Subscribe to status updates
    if (!statusUnsubscribe) {
      statusUnsubscribe = wsStore.subscribeToLmdeployStatus((data) => {
        status.value = data
      })
    }
    
    // Subscribe to install log lines
    if (!installLogUnsubscribe) {
      installLogUnsubscribe = wsStore.subscribeToLmdeployInstallLog((data) => {
        if (data.line) {
          // Prevent duplicates: check if this line already exists in current logs
          // This handles the case where HTTP fetch and WebSocket might send the same line
          // We check the last 500 chars to avoid checking entire log for performance
          const recentLogs = logs.value.slice(-500)
          if (!recentLogs.includes(data.line)) {
            // Append new log line
            if (logs.value) {
              logs.value += '\n' + data.line
            } else {
              logs.value = data.line
            }
          }
        }
      })
    }
    
    // Subscribe to runtime log lines
    if (!runtimeLogUnsubscribe) {
      runtimeLogUnsubscribe = wsStore.subscribeToLmdeployRuntimeLog((data) => {
        if (data.line) {
          // Prevent duplicates: check if this line already exists in current logs
          // This handles the case where HTTP fetch and WebSocket might send the same line
          // We check the last 500 chars to avoid checking entire log for performance
          const recentLogs = runtimeLogs.value.slice(-500)
          if (!recentLogs.includes(data.line)) {
            // Append new log line
            if (runtimeLogs.value) {
              runtimeLogs.value += '\n' + data.line
            } else {
              runtimeLogs.value = data.line
            }
          }
        }
      })
    }
  }

  const stopWebSocketSubscriptions = () => {
    if (statusUnsubscribe) {
      statusUnsubscribe()
      statusUnsubscribe = null
    }
    if (installLogUnsubscribe) {
      installLogUnsubscribe()
      installLogUnsubscribe = null
    }
    if (runtimeLogUnsubscribe) {
      runtimeLogUnsubscribe()
      runtimeLogUnsubscribe = null
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
    startWebSocketSubscriptions,
    stopWebSocketSubscriptions
  }
})

