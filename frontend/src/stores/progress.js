/**
 * SSE-based progress and events store. Subscribes to GET /api/events.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'

const SSE_EVENT_TYPES = [
  'task_created',
  'task_updated',
  'task_log',
  'download_progress',
  'download_complete',
  'build_progress',
  'notification',
  'model_status',
  'model_event',
  'unified_monitoring',
  'lmdeploy_install_status',
  'lmdeploy_install_log',
  'onecat_vllm_install_status',
  'onecat_vllm_install_log',
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

    const metadataLines = Array.isArray(task.metadata?.log_lines) ? task.metadata.log_lines : []
    const isBuildLike =
      task.type === 'build'
      || (typeof task.task_id === 'string' && task.task_id.startsWith('build_'))
    const isDownloadLike =
      task.type === 'download'
      || (typeof task.task_id === 'string' && task.task_id.startsWith('download_'))

    // Build tasks stream live lines via `build_progress`. `metadata.log_lines` is a
    // reconnect snapshot (and send_build_progress also mirrors the current batch into
    // metadata). Appending both channels duplicated every line in the UI.
    if (isBuildLike) {
      const existing = taskLogs.value[task.task_id] || []
      if (existing.length === 0 && metadataLines.length > 0) {
        appendTaskLogs(task.task_id, metadataLines)
      }
      return
    }

    // Download status text changes every tick (MB/s). Keep that on the progress
    // row via download_progress — do not spam the log buffer.
    if (isDownloadLike) {
      return
    }

    // Merge on every task_updated so installers that only stream via
    // metadata.log_lines (no task_log / build_progress) still update live.
    if (metadataLines.length > 0) {
      appendTaskLogs(task.task_id, metadataLines)
      return
    }

    if (task.message) {
      appendTaskLogs(task.task_id, task.message)
    }
  }

  function handleEvent(eventType, rawData) {
    let data = rawData
    try {
      if (typeof rawData === 'string') data = JSON.parse(rawData)
    } catch (_) { return }
    const payload = data?.data != null ? data.data : data

    if (eventType === 'task_log') {
      appendTaskLogs(payload?.task_id, payload?.line)
    }
    if (eventType === 'build_progress') {
      // Dedupe against the existing buffer so mirrored metadata.log_lines from
      // task_updated (seed / reconnect) are not appended a second time.
      appendTaskLogs(payload?.task_id, payload?.log_lines)
      if (payload?.task_id) {
        const existing = tasks.value[payload.task_id] || {}
        const nextProgress = payload.progress ?? existing.progress ?? 0
        tasks.value = {
          ...tasks.value,
          [payload.task_id]: {
            ...existing,
            task_id: payload.task_id,
            type: existing.type || 'build',
            status: existing.status || 'running',
            description: existing.description || payload.stage || 'Build',
            progress: nextProgress,
            message: payload.message || existing.message || '',
            metadata: {
              ...(existing.metadata || {}),
              stage: payload.stage || existing.metadata?.stage,
              log_lines: payload.log_lines || existing.metadata?.log_lines || [],
            },
          },
        }
      }
    }
    if (eventType === 'download_progress' && payload?.task_id) {
      const existing = tasks.value[payload.task_id] || {}
      tasks.value = {
        ...tasks.value,
        [payload.task_id]: {
          ...existing,
          task_id: payload.task_id,
          type: existing.type || 'download',
          status: existing.status || 'running',
          description: existing.description || payload.filename || 'Download',
          progress: payload.progress ?? existing.progress ?? 0,
          message: payload.message || existing.message || '',
          metadata: {
            ...(existing.metadata || {}),
            bytes_downloaded: payload.bytes_downloaded,
            total_bytes: payload.total_bytes,
            speed_mbps: payload.speed_mbps,
            eta_seconds: payload.eta_seconds,
            filename: payload.filename,
            current_filename: payload.current_filename || payload.filename,
            files_completed: payload.files_completed,
            files_total: payload.files_total,
            huggingface_id: payload.huggingface_id,
            model_format: payload.model_format,
          },
        },
      }
    }
    if (
      eventType === 'cuda_install_log'
      || eventType === 'lmdeploy_install_log'
      || eventType === 'onecat_vllm_install_log'
    ) {
      if (payload?.task_id) appendTaskLogs(payload.task_id, payload?.line)
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
  const subscribeToOnecatVllmInstallLog = (cb) => subscribe('onecat_vllm_install_log', cb)

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
    subscribeToLmdeployInstallLog,
    subscribeToOnecatVllmInstallLog,
    handleEvent,
  }
})
