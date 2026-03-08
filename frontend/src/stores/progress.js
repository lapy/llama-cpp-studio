/**
 * SSE-based progress and events store. Subscribes to GET /api/events.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const SSE_EVENT_TYPES = [
  'task_created',
  'task_updated',
  'download_progress',
  'download_complete',
  'build_progress',
  'notification',
  'model_status',
  'model_event',
  'unified_monitoring',
  'lmdeploy_status',
  'lmdeploy_runtime_log',
  'lmdeploy_install_status',
  'lmdeploy_install_log',
  'cuda_install_status',
  'cuda_install_progress',
  'cuda_install_log',
  'broadcast'
]

export const useProgressStore = defineStore('progress', () => {
  const tasks = ref({})
  const eventSource = ref(null)
  const connected = ref(false)
  const subscribers = ref(new Map()) // eventType -> Set<callback>

  const activeTasks = computed(() => {
    return Object.values(tasks.value).filter(t => t.status === 'running')
  })

  const connectionStatus = computed(() => (connected.value ? 'connected' : 'disconnected'))
  const isConnected = computed(() => connected.value && eventSource.value?.readyState === EventSource.OPEN)

  function notifySubscribers(eventType, data) {
    const callbacks = subscribers.value.get(eventType)
    if (callbacks) {
      callbacks.forEach(cb => {
        try {
          const result = cb(data)
          if (result && typeof result.then === 'function') result.catch(() => {})
        } catch (_) {}
      })
    }
    const any = subscribers.value.get('*')
    if (any) any.forEach(cb => { try { cb(eventType, data) } catch (_) {} })
  }

  function handleEvent(eventType, rawData) {
    let data = rawData
    try {
      if (typeof rawData === 'string') data = JSON.parse(rawData)
    } catch (_) { return }
    if (eventType === 'task_created' || eventType === 'task_updated') {
      const task = data?.data ?? data
      if (task?.task_id) tasks.value = { ...tasks.value, [task.task_id]: task }
    }
    const payload = data?.data != null ? data.data : data
    notifySubscribers(eventType, payload)
    if (payload?.type && payload.type !== eventType) notifySubscribers(payload.type, payload)
  }

  function connect() {
    if (eventSource.value?.readyState === EventSource.OPEN) return
    // In dev, connect directly to backend to avoid proxy buffering SSE (port must match vite proxy target)
    const base = typeof window !== 'undefined' && window.location?.origin ? window.location.origin : ''
    const isDev = typeof import.meta !== 'undefined' && import.meta.env?.DEV
    const devPort = typeof import.meta !== 'undefined' && import.meta.env?.VITE_API_PORT ? Number(import.meta.env.VITE_API_PORT) : 8081
    const url = isDev ? `http://localhost:${devPort}/api/events` : `${base}/api/events`
    if (isDev) console.log('[SSE] Connecting to', url, '(dev: direct to backend)')
    const es = new EventSource(url)
    es.onopen = () => {
      if (isDev) console.log('[SSE] onopen, readyState=', es.readyState)
      connected.value = true
    }
    es.onerror = (e) => {
      if (isDev) console.warn('[SSE] onerror, readyState=', es.readyState, 'event=', e)
      es.close()
      eventSource.value = null
      connected.value = false
      setTimeout(() => connect(), 3000)
    }
    es.onmessage = (e) => handleEvent('message', e.data)
    SSE_EVENT_TYPES.forEach(type => {
      es.addEventListener(type, (e) => handleEvent(type, e.data))
    })
    eventSource.value = es
  }

  function disconnect() {
    if (eventSource.value) {
      eventSource.value.close()
      eventSource.value = null
    }
    connected.value = false
  }

  function getTask(taskId) {
    return tasks.value[taskId] || null
  }

  function subscribe(eventType, callback) {
    if (!subscribers.value.has(eventType)) subscribers.value.set(eventType, new Set())
    subscribers.value.get(eventType).add(callback)
    return () => {
      const set = subscribers.value.get(eventType)
      if (set) {
        set.delete(callback)
        if (set.size === 0) subscribers.value.delete(eventType)
      }
    }
  }

  const subscribeToDownloadProgress = (cb) => subscribe('download_progress', cb)
  const subscribeToBuildProgress = (cb) => subscribe('build_progress', cb)
  const subscribeToModelStatus = (cb) => subscribe('model_status', cb)
  const subscribeToNotifications = (cb) => subscribe('notification', cb)
  const subscribeToDownloadComplete = (cb) => subscribe('download_complete', cb)
  const subscribeToUnifiedMonitoring = (cb) => subscribe('unified_monitoring', cb)
  const subscribeToModelEvents = (cb) => subscribe('model_event', cb)
  const subscribeToLmdeployStatus = (cb) => subscribe('lmdeploy_status', cb)
  const subscribeToLmdeployInstallLog = (cb) => subscribe('lmdeploy_install_log', cb)
  const subscribeToLmdeployRuntimeLog = (cb) => subscribe('lmdeploy_runtime_log', cb)

  return {
    tasks,
    activeTasks,
    connected,
    connectionStatus,
    isConnected,
    connect,
    disconnect,
    getTask,
    subscribe,
    subscribeToDownloadProgress,
    subscribeToBuildProgress,
    subscribeToModelStatus,
    subscribeToNotifications,
    subscribeToDownloadComplete,
    subscribeToUnifiedMonitoring,
    subscribeToModelEvents,
    subscribeToLmdeployStatus,
    subscribeToLmdeployInstallLog,
    subscribeToLmdeployRuntimeLog
  }
})
