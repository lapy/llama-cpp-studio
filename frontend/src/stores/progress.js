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
  'lmdeploy_install_status',
  'lmdeploy_install_log',
  'cuda_install_status',
  'cuda_install_progress',
  'cuda_install_log',
  'broadcast'
]

export const useProgressStore = defineStore('progress', () => {
  const tasks = ref({})
  const taskLogs = ref({})
  const eventSource = ref(null)
  const connected = ref(false)
  const subscribers = ref(new Map()) // eventType -> Set<callback>
  const CUDA_TASK_ID = 'cuda_operation'
  const LMDEPLOY_TASK_ID = 'lmdeploy_operation'
  const MAX_LOG_LINES = 200
  const MAX_BUILD_LOG_LINES = 15000

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

  function upsertTask(taskId, updates) {
    const existing = tasks.value[taskId] || {}
    tasks.value = {
      ...tasks.value,
      [taskId]: {
        ...existing,
        task_id: taskId,
        ...updates,
      },
    }
  }

  function appendTaskLogs(taskId, lines, options = {}) {
    if (lines == null || taskId == null) return
    const dedupe = options.dedupe !== false
    const cap =
      typeof taskId === 'string' && taskId.startsWith('build_')
        ? MAX_BUILD_LOG_LINES
        : MAX_LOG_LINES
    const entries = Array.isArray(lines) ? lines : [lines]
    const existing = taskLogs.value[taskId] || []
    const next = [...existing]
    const seen = dedupe ? new Set(existing) : null

    entries.forEach((entry) => {
      if (typeof entry !== 'string') return
      entry.split(/\r?\n/).forEach((rawLine) => {
        const line = rawLine.trim()
        if (!line) return
        if (dedupe && seen.has(line)) return
        if (dedupe) seen.add(line)
        next.push(line)
      })
    })

    if (next.length === existing.length) return

    taskLogs.value = {
      ...taskLogs.value,
      [taskId]: next.slice(-cap),
    }
  }

  function syncTaskLogsFromTask(task) {
    if (!task?.task_id) return

    const existing = taskLogs.value[task.task_id] || []
    const metadataLines = Array.isArray(task.metadata?.log_lines) ? task.metadata.log_lines : []

    if (existing.length === 0 && metadataLines.length > 0) {
      appendTaskLogs(task.task_id, metadataLines)
    }

    if (task.message && existing.length === 0) {
      appendTaskLogs(task.task_id, task.message)
    }
  }

  function normalizeCudaTask(eventType, payload) {
    if (!payload || typeof payload !== 'object') return

    if (eventType === 'cuda_install_status') {
      const operation = payload.operation || payload.status || 'install'
      const description = operation === 'uninstall' ? 'Uninstall CUDA' : 'Install CUDA'

      if (payload.status === 'completed' || payload.status === 'failed') {
        const existing = tasks.value[CUDA_TASK_ID] || {}
        upsertTask(CUDA_TASK_ID, {
          type: 'install',
          description,
          progress: payload.status === 'completed' ? 100 : (existing.progress ?? 0),
          status: payload.status,
          message: payload.message || existing.message || '',
          metadata: {
            ...(existing.metadata || {}),
            target: 'cuda',
            operation,
            ended_at: payload.ended_at,
          },
        })
        appendTaskLogs(CUDA_TASK_ID, payload.message)
        return
      }

      upsertTask(CUDA_TASK_ID, {
        type: 'install',
        description,
        progress: 0,
        status: 'running',
        message: payload.message || (operation === 'uninstall' ? 'Preparing CUDA uninstall...' : 'Preparing CUDA install...'),
        metadata: {
          target: 'cuda',
          operation,
          started_at: payload.started_at,
        },
      })
      appendTaskLogs(CUDA_TASK_ID, payload.message)
      return
    }

    if (eventType === 'cuda_install_progress') {
      const existing = tasks.value[CUDA_TASK_ID] || {}
      const operation = existing.metadata?.operation || 'install'
      upsertTask(CUDA_TASK_ID, {
        type: 'install',
        description: operation === 'uninstall' ? 'Uninstall CUDA' : 'Install CUDA',
        progress: Number(payload.progress ?? existing.progress ?? 0),
        status: existing.status === 'failed' ? 'failed' : 'running',
        message: payload.message || existing.message || '',
        metadata: {
          ...(existing.metadata || {}),
          target: 'cuda',
          stage: payload.stage,
          timestamp: payload.timestamp,
        },
      })
    }
  }

  function normalizeLmdeployTask(eventType, payload) {
    if (!payload || typeof payload !== 'object') return

    if (eventType === 'lmdeploy_install_status') {
      const operation = payload.operation || payload.status || 'install'
      const actionMap = {
        install: 'Install LMDeploy',
        install_source: 'Install LMDeploy from Source',
        remove: 'Remove LMDeploy',
      }
      const description = actionMap[operation] || 'Install LMDeploy'

      if (payload.status === 'completed' || payload.status === 'failed') {
        const existing = tasks.value[LMDEPLOY_TASK_ID] || {}
        upsertTask(LMDEPLOY_TASK_ID, {
          type: 'install',
          description,
          progress: payload.status === 'completed' ? 100 : (existing.progress ?? 0),
          status: payload.status,
          message: payload.message || existing.message || '',
          metadata: {
            ...(existing.metadata || {}),
            target: 'lmdeploy',
            operation,
            ended_at: payload.ended_at,
          },
        })
        appendTaskLogs(LMDEPLOY_TASK_ID, payload.message)
        return
      }

      upsertTask(LMDEPLOY_TASK_ID, {
        type: 'install',
        description,
        progress: 10,
        status: 'running',
        message: payload.message || 'Preparing LMDeploy operation...',
        metadata: {
          target: 'lmdeploy',
          operation,
          started_at: payload.started_at,
          log_count: 0,
        },
      })
      appendTaskLogs(LMDEPLOY_TASK_ID, payload.message)
      return
    }

    if (eventType === 'lmdeploy_install_log') {
      const existing = tasks.value[LMDEPLOY_TASK_ID]
      if (!existing || existing.status !== 'running') return
      const logCount = Number(existing.metadata?.log_count || 0) + 1
      const progress = Math.min(90, Math.max(Number(existing.progress || 10), 10 + logCount * 3))
      upsertTask(LMDEPLOY_TASK_ID, {
        type: 'install',
        description: existing.description || 'Install LMDeploy',
        progress,
        status: 'running',
        message: payload.line || existing.message || '',
        metadata: {
          ...(existing.metadata || {}),
          target: 'lmdeploy',
          log_count: logCount,
          timestamp: payload.timestamp,
        },
      })
      appendTaskLogs(LMDEPLOY_TASK_ID, payload.line)
    }
  }

  function handleEvent(eventType, rawData) {
    let data = rawData
    try {
      if (typeof rawData === 'string') data = JSON.parse(rawData)
    } catch (_) { return }
    const payload = data?.data != null ? data.data : data
    if (eventType === 'cuda_install_status' || eventType === 'cuda_install_progress') {
      normalizeCudaTask(eventType, payload)
    }
    if (eventType === 'cuda_install_log') {
      appendTaskLogs(CUDA_TASK_ID, payload?.line)
    }
    if (eventType === 'lmdeploy_install_status' || eventType === 'lmdeploy_install_log') {
      normalizeLmdeployTask(eventType, payload)
    }
    if (eventType === 'build_progress') {
      appendTaskLogs(payload?.task_id, payload?.log_lines, { dedupe: false })
    }
    if (eventType === 'task_created' || eventType === 'task_updated') {
      const task = data?.data ?? data
      if (task?.task_id) {
        tasks.value = { ...tasks.value, [task.task_id]: task }
        syncTaskLogsFromTask(task)
      }
    }
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

  function getTaskLogs(taskId) {
    return taskLogs.value[taskId] || []
  }

  /** Remove a task and its logs from the UI (e.g. user dismissed after completion). */
  function removeTask(taskId) {
    if (!taskId) return
    const { [taskId]: _t, ...restTasks } = tasks.value
    tasks.value = restTasks
    const { [taskId]: _l, ...restLogs } = taskLogs.value
    taskLogs.value = restLogs
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
  const subscribeToLmdeployInstallLog = (cb) => subscribe('lmdeploy_install_log', cb)

  return {
    tasks,
    taskLogs,
    activeTasks,
    connected,
    connectionStatus,
    isConnected,
    connect,
    disconnect,
    getTask,
    getTaskLogs,
    removeTask,
    subscribe,
    subscribeToDownloadProgress,
    subscribeToBuildProgress,
    subscribeToModelStatus,
    subscribeToNotifications,
    subscribeToDownloadComplete,
    subscribeToUnifiedMonitoring,
    subscribeToModelEvents,
    subscribeToLmdeployInstallLog
  }
})
