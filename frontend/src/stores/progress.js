/**
 * SSE-based progress and events store. Subscribes to GET /api/events.
 */
import { defineStore } from 'pinia'
import { ref, computed, shallowRef } from 'vue'
import axios from 'axios'
import {
  OFFLINE_AFTER_ATTEMPTS,
  STALE_CHECK_INTERVAL_MS,
  STALE_CONNECTION_MS,
  reconnectDelay,
} from '@/utils/liveUpdates'

export const eventSourceFactory = {
  open(url) {
    return new globalThis.EventSource(url)
  },
}

const SSE_EVENT_TYPES = [
  'task_created',
  'task_updated',
  'task_dismissed',
  'task_snapshot',
  'resync',
  'task_log',
  'download_progress',
  'download_complete',
  'build_progress',
  'notification',
  'broadcast',
]

const DISMISSED_TASKS_KEY = 'llama-studio.activity.dismissed'
const ACTIVE_TASK_STATUSES = new Set(['running', 'queued', 'cancelling'])

function readDismissedTaskIds() {
  if (typeof localStorage === 'undefined') return []
  try {
    const parsed = JSON.parse(localStorage.getItem(DISMISSED_TASKS_KEY) || '[]')
    return Array.isArray(parsed) ? parsed.filter((id) => typeof id === 'string' && id) : []
  } catch (_) {
    return []
  }
}

function writeDismissedTaskIds(ids) {
  if (typeof localStorage === 'undefined') return
  try {
    localStorage.setItem(DISMISSED_TASKS_KEY, JSON.stringify(ids.slice(-400)))
  } catch (_) {}
}

export const useProgressStore = defineStore('progress', () => {
  const tasks = ref({})
  const taskLogs = ref({})
  const hydratingTaskLogs = ref({})
  const hydratedTaskLogs = ref({})
  // The EventSource is an opaque browser handle: shallowRef keeps Vue from
  // deep-proxying it (and lets identity checks protect against stale handlers).
  const eventSource = shallowRef(null)
  const connected = ref(false)
  const reconnectAttempt = ref(0)
  const lastConnectedAt = ref(null)
  const lastEventAt = ref(null)
  const disconnectedSince = ref(null)
  const manuallyDisconnected = ref(false)
  // Durable interrupted/unknown operations from GET /api/operations/attention.
  // Unlike `tasks`, this is a server-read list, not the SSE activity snapshot:
  // it carries the recovery classification and the actions the API supports.
  const attentionOperations = ref([])
  const attentionLoading = ref(false)
  const attentionError = ref('')
  let attentionFetchSeq = 0
  let reconnectTimer = null
  let staleTimer = null
  let closedByUser = false
  // Only true once this connection has delivered a named `heartbeat` event.
  // The current backend sends comment heartbeats instead, so the watchdog
  // normally needs active tasks before it may call a quiet stream stale.
  let namedHeartbeatSeen = false
  const subscribers = ref(new Map()) // eventType -> Set<callback>
  const dismissedTaskIds = new Set(readDismissedTaskIds())

  function rememberDismissed(taskId) {
    if (!taskId || dismissedTaskIds.has(taskId)) return
    dismissedTaskIds.add(taskId)
    writeDismissedTaskIds([...dismissedTaskIds])
  }

  function forgetDismissed(taskId) {
    if (!taskId || !dismissedTaskIds.has(taskId)) return
    dismissedTaskIds.delete(taskId)
    writeDismissedTaskIds([...dismissedTaskIds])
  }

  function keepIncomingTask(task) {
    const taskId = task?.task_id
    if (!taskId) return false
    if (!dismissedTaskIds.has(taskId)) return true
    if (ACTIVE_TASK_STATUSES.has(task.status)) {
      forgetDismissed(taskId)
      return true
    }
    return false
  }

  const MAX_LOG_LINES = 200
  const MAX_INSTALL_LOG_LINES = 15000
  const MAX_BUILD_LOG_LINES = 15000

  const activeTasks = computed(() => {
    return Object.values(tasks.value).filter((t) => t.status === 'running')
  })

  // In-flight work of any kind, including tasks queued behind another task or
  // already being cancelled. The paused banner uses this rather than
  // `activeTasks`, which deliberately keeps its running-only meaning.
  const hasActiveTasks = computed(() =>
    Object.values(tasks.value).some((task) => ACTIVE_TASK_STATUSES.has(task?.status)),
  )

  const connectionStatus = computed(() => (connected.value ? 'connected' : 'disconnected'))
  const isConnected = computed(() => connected.value && eventSource.value?.readyState === 1)
  const connectionState = computed(() => {
    if (isConnected.value) return 'live'
    if (manuallyDisconnected.value) return 'offline'
    if (typeof navigator !== 'undefined' && navigator.onLine === false) return 'offline'
    return reconnectAttempt.value >= OFFLINE_AFTER_ATTEMPTS ? 'offline' : 'connecting'
  })

  function notifySubscribers(eventType, data) {
    const callbacks = subscribers.value.get(eventType)
    if (callbacks) {
      callbacks.forEach((cb) => {
        try {
          const result = cb(data)
          if (result && typeof result.then === 'function') result.catch(() => {})
        } catch (_) {}
      })
    }
    const any = subscribers.value.get('*')
    if (any)
      any.forEach((cb) => {
        try {
          cb(eventType, data)
        } catch (_) {}
      })
  }

  function monotonicProgress(existing, incoming) {
    const current = Number(existing?.progress)
    const next = Number(incoming)
    if (!Number.isFinite(next)) return Number.isFinite(current) ? current : 0
    if (!Number.isFinite(current)) return Math.max(0, next)
    return Math.max(current, next)
  }

  function mergeTaskProgress(existing, task) {
    if (!task || typeof task !== 'object') return task
    return { ...task, progress: monotonicProgress(existing, task.progress) }
  }

  function splitTaskLogEntry(entry) {
    // Match Python's str.splitlines(): preserve leading/trailing whitespace
    // and internal blank lines, but do not invent a trailing blank line after
    // a terminating newline. A literal empty entry is one blank log line.
    if (entry === '') return ['']
    const parts = entry.split(/\r\n|\n|\r/)
    if (parts.length > 1 && parts[parts.length - 1] === '') parts.pop()
    return parts
  }

  function appendWithOverlap(existing, incoming, dedupe) {
    if (!incoming.length) return existing
    let start = 0
    if (dedupe) {
      const limit = Math.min(existing.length, incoming.length)
      for (let size = limit; size > 0; size -= 1) {
        let matches = true
        for (let index = 0; index < size; index += 1) {
          if (existing[existing.length - size + index] !== incoming[index]) {
            matches = false
            break
          }
        }
        if (matches) {
          start = size
          break
        }
      }
    }
    return existing.concat(incoming.slice(start))
  }

  function appendTaskLogs(taskId, lines, options = {}) {
    if (lines == null || taskId == null) return
    const isBuild = typeof taskId === 'string' && taskId.startsWith('build_')
    const isScan =
      (typeof taskId === 'string' && taskId.startsWith('scan_')) ||
      tasks.value[taskId]?.type === 'param_scan'
    const isInstall =
      (typeof taskId === 'string' && taskId.startsWith('install_')) ||
      tasks.value[taskId]?.type === 'install'
    const cap =
      isBuild || isScan ? MAX_BUILD_LOG_LINES : isInstall ? MAX_INSTALL_LOG_LINES : MAX_LOG_LINES
    const entries = Array.isArray(lines) ? lines : [lines]
    const existing = taskLogs.value[taskId] || []
    const incoming = []

    entries.forEach((entry) => {
      if (typeof entry !== 'string') return
      incoming.push(...splitTaskLogEntry(entry))
    })

    const next = appendWithOverlap(existing, incoming, options.dedupe !== false)
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
      task.type === 'build' ||
      task.type === 'param_scan' ||
      (typeof task.task_id === 'string' && task.task_id.startsWith('build_')) ||
      (typeof task.task_id === 'string' && task.task_id.startsWith('scan_'))
    const isDownloadLike =
      task.type === 'download' ||
      (typeof task.task_id === 'string' && task.task_id.startsWith('download_'))

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
    }
    // `task.message` is Studio's progress/status label, not installer output.
    // It stays on the task card; injecting it here duplicates it repeatedly
    // (e.g. "Pip tools ready") into the log buffer.
  }

  function handleEvent(eventType, rawData) {
    // Any framed event (including the snapshot the server sends on connect)
    // proves the stream is alive. Comment heartbeats are invisible to
    // EventSource; the named `heartbeat` listener below covers a future server.
    lastEventAt.value = Date.now()
    let data = rawData
    try {
      if (typeof rawData === 'string') data = JSON.parse(rawData)
    } catch (_) {
      return
    }
    const payload = data?.data != null ? data.data : data

    if (eventType === 'task_log') {
      // task_log is an incremental stream: repeated/blank lines are real
      // output and must never be deduplicated.
      appendTaskLogs(payload?.task_id, payload?.line, { dedupe: false })
    }
    if (
      eventType === 'build_progress' &&
      payload?.task_id &&
      !dismissedTaskIds.has(payload.task_id)
    ) {
      // Dedupe against the existing buffer so mirrored metadata.log_lines from
      // task_updated (seed / reconnect) are not appended a second time.
      appendTaskLogs(payload?.task_id, payload?.log_lines)
      if (payload?.task_id) {
        const existing = tasks.value[payload.task_id] || {}
        const nextProgress = monotonicProgress(existing, payload.progress ?? existing.progress ?? 0)
        const scanLike =
          existing.type === 'param_scan' ||
          (typeof payload.task_id === 'string' && payload.task_id.startsWith('scan_'))
        tasks.value = {
          ...tasks.value,
          [payload.task_id]: {
            ...existing,
            task_id: payload.task_id,
            type: existing.type || (scanLike ? 'param_scan' : 'build'),
            status: existing.status || 'running',
            description:
              existing.description || payload.stage || (scanLike ? 'Parameter scan' : 'Build'),
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
    if (
      eventType === 'download_progress' &&
      payload?.task_id &&
      !dismissedTaskIds.has(payload.task_id)
    ) {
      const existing = tasks.value[payload.task_id] || {}
      tasks.value = {
        ...tasks.value,
        [payload.task_id]: {
          ...existing,
          task_id: payload.task_id,
          type: existing.type || 'download',
          status: existing.status || 'running',
          description: existing.description || payload.filename || 'Download',
          progress: monotonicProgress(existing, payload.progress ?? existing.progress ?? 0),
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
    if (eventType === 'task_dismissed') {
      const taskId = payload?.task_id
      if (taskId) {
        rememberDismissed(taskId)
        const { [taskId]: _t, ...restTasks } = tasks.value
        tasks.value = restTasks
        const { [taskId]: _l, ...restLogs } = taskLogs.value
        taskLogs.value = restLogs
        forgetTaskLogs(taskId)
      }
    }
    if (eventType === 'task_snapshot' || eventType === 'resync') {
      const incoming = Array.isArray(payload?.tasks) ? payload.tasks : []
      const next = {}
      incoming.forEach((task) => {
        if (!keepIncomingTask(task)) return
        const merged = mergeTaskProgress(tasks.value[task.task_id], task)
        next[task.task_id] = merged
        syncTaskLogsFromTask(merged)
      })
      tasks.value = next
    }
    if (eventType === 'task_created' || eventType === 'task_updated') {
      const task = data?.data ?? data
      if (keepIncomingTask(task)) {
        task.progress = monotonicProgress(tasks.value[task.task_id], task.progress)
        tasks.value = { ...tasks.value, [task.task_id]: task }
        syncTaskLogsFromTask(task)
      }
    }
    notifySubscribers(eventType, payload)
    if (payload?.type && payload.type !== eventType) notifySubscribers(payload.type, payload)
  }

  function clearReconnectTimer() {
    if (reconnectTimer) {
      clearTimeout(reconnectTimer)
      reconnectTimer = null
    }
  }

  function stopStaleWatch() {
    if (staleTimer) {
      clearInterval(staleTimer)
      staleTimer = null
    }
  }

  function markDisconnected() {
    if (disconnectedSince.value == null) disconnectedSince.value = Date.now()
  }

  /** Close the active source without letting its handlers schedule more work. */
  function closeSource() {
    const es = eventSource.value
    eventSource.value = null
    if (!es) return
    es.onopen = null
    es.onerror = null
    es.onmessage = null
    try {
      es.close()
    } catch (_) {}
  }

  /**
   * Stale-stream watchdog.
   *
   * The backend writes comment heartbeats (`: heartbeat`) every 15s. The SSE
   * spec discards comments, so EventSource never exposes them to JavaScript:
   * an idle healthy stream and a half-open stream both look silent. We only
   * declare the stream stale while the store knows about in-flight work, which
   * should keep producing task events; 45s of silence then means the channel
   * is wedged. Idle sessions are left alone so a healthy quiet stream is not
   * torn down every 45s. If the server ever emits a named `heartbeat` event
   * (tracked in `openSource`), silence is meaningful even while idle and the
   * watchdog runs without active tasks.
   */
  function startStaleWatch() {
    stopStaleWatch()
    staleTimer = setInterval(() => {
      if (!isConnected.value) {
        stopStaleWatch()
        return
      }
      const baseline = lastEventAt.value ?? lastConnectedAt.value
      if (!baseline) return
      if (!hasActiveTasks.value && !namedHeartbeatSeen) return
      if (Date.now() - baseline < STALE_CONNECTION_MS) return
      reconnectAttempt.value += 1
      markDisconnected()
      closeSource()
      connected.value = false
      stopStaleWatch()
      if (closedByUser) return
      scheduleReconnect()
    }, STALE_CHECK_INTERVAL_MS)
  }

  function scheduleReconnect() {
    clearReconnectTimer()
    if (closedByUser) return
    const delay = reconnectDelay(reconnectAttempt.value)
    reconnectTimer = setTimeout(() => {
      reconnectTimer = null
      if (closedByUser || eventSource.value) return
      openSource()
    }, delay)
  }

  function openSource() {
    // Never keep two streams open: an EventSource in CONNECTING is already a
    // duplicate if connect() is called again by a component or retry.
    if (eventSource.value) return
    // Use the same origin in development and production so HTTPS and session
    // cookies also work through a reverse proxy and the Vite API proxy.
    const isDev = typeof import.meta !== 'undefined' && import.meta.env?.DEV
    const url = '/api/events'
    const es = eventSourceFactory.open(url)
    eventSource.value = es
    namedHeartbeatSeen = false
    es.onopen = () => {
      if (eventSource.value !== es) return
      if (isDev) console.log('[SSE] onopen, readyState=', es.readyState)
      connected.value = true
      reconnectAttempt.value = 0
      lastConnectedAt.value = Date.now()
      // The server sends a snapshot immediately; use open as the watchdog's
      // starting point in case the first frame is delayed.
      lastEventAt.value = lastConnectedAt.value
      disconnectedSince.value = null
      startStaleWatch()
    }
    es.onerror = (e) => {
      if (eventSource.value !== es) return
      if (isDev) console.warn('[SSE] onerror, readyState=', es.readyState, 'event=', e)
      closeSource()
      connected.value = false
      stopStaleWatch()
      markDisconnected()
      if (closedByUser) return
      reconnectAttempt.value += 1
      scheduleReconnect()
    }
    es.onmessage = (e) => {
      if (eventSource.value !== es) return
      handleEvent('message', e.data)
    }
    SSE_EVENT_TYPES.forEach((type) => {
      es.addEventListener(type, (e) => {
        if (eventSource.value !== es) return
        handleEvent(type, e.data)
      })
    })
    es.addEventListener('heartbeat', () => {
      if (eventSource.value !== es) return
      namedHeartbeatSeen = true
      lastEventAt.value = Date.now()
    })
  }

  function connect() {
    closedByUser = false
    manuallyDisconnected.value = false
    clearReconnectTimer()
    openSource()
  }

  /**
   * User-initiated retry: cancel the pending backoff and open a fresh stream
   * now. Resetting the attempt count makes the next automatic retry start at
   * the short end of the backoff again.
   */
  function retryConnection() {
    closedByUser = false
    manuallyDisconnected.value = false
    clearReconnectTimer()
    closeSource()
    stopStaleWatch()
    connected.value = false
    reconnectAttempt.value = 0
    openSource()
  }

  function disconnect() {
    closedByUser = true
    manuallyDisconnected.value = true
    clearReconnectTimer()
    stopStaleWatch()
    closeSource()
    connected.value = false
  }

  function getTask(taskId) {
    return tasks.value[taskId] || null
  }

  function getTaskLogs(taskId) {
    return taskLogs.value[taskId] || []
  }

  function isTaskLogsLoading(taskId) {
    return Boolean(hydratingTaskLogs.value[taskId])
  }

  /**
   * Restore the complete server-retained log for a task after a page refresh.
   * Live SSE events keep arriving while this runs; the server buffer is
   * authoritative for everything already produced.
   */
  async function hydrateTaskLogs(taskId) {
    const id = String(taskId || '')
    if (!id) return []
    if (hydratedTaskLogs.value[id]) return taskLogs.value[id] || []
    if (hydratingTaskLogs.value[id]) return taskLogs.value[id] || []
    hydratingTaskLogs.value = { ...hydratingTaskLogs.value, [id]: true }
    try {
      const { data } = await axios.get(`/api/operations/tasks/${encodeURIComponent(id)}/logs`)
      const serverLines = Array.isArray(data?.lines) ? data.lines : []
      const localLines = taskLogs.value[id] || []
      const next = serverLines.length ? serverLines : localLines
      if (next.length || !localLines.length) {
        taskLogs.value = { ...taskLogs.value, [id]: next }
      }
      return next
    } catch (_) {
      // A task that is no longer retained still keeps whatever streamed live.
      return taskLogs.value[id] || []
    } finally {
      const nextLoading = { ...hydratingTaskLogs.value }
      delete nextLoading[id]
      hydratingTaskLogs.value = nextLoading
      hydratedTaskLogs.value = { ...hydratedTaskLogs.value, [id]: true }
    }
  }

  /**
   * Load interrupted/unknown operations with their recovery classification.
   * The endpoint is read-only; retry/confirm still belongs to the page that
   * owns the original action and the confirmation the supervisor requires.
   */
  async function fetchAttentionOperations() {
    const seq = ++attentionFetchSeq
    attentionLoading.value = true
    attentionError.value = ''
    try {
      const { data } = await axios.get('/api/operations/attention')
      if (seq !== attentionFetchSeq) return attentionOperations.value
      const rows = Array.isArray(data?.operations) ? data.operations : []
      attentionOperations.value = rows
      return rows
    } catch (error) {
      if (seq === attentionFetchSeq) {
        const detail = error?.response?.data?.detail
        attentionError.value =
          (typeof detail === 'string' && detail) ||
          error?.message ||
          'Could not check for interrupted operations.'
      }
      throw error
    } finally {
      if (seq === attentionFetchSeq) attentionLoading.value = false
    }
  }

  /** Drop one acknowledged row locally; the server already forgot the task. */
  function forgetAttentionOperation(operationId) {
    const id = String(operationId || '')
    if (!id) return
    attentionOperations.value = attentionOperations.value.filter(
      (operation) => operation?.operation_id !== id,
    )
  }

  function forgetTaskLogs(taskId) {
    const nextHydrated = { ...hydratedTaskLogs.value }
    delete nextHydrated[taskId]
    hydratedTaskLogs.value = nextHydrated
    const nextLoading = { ...hydratingTaskLogs.value }
    delete nextLoading[taskId]
    hydratingTaskLogs.value = nextLoading
  }

  /** Remove a task and its logs from the UI (e.g. user dismissed after completion). */
  function removeTask(taskId) {
    if (!taskId) return
    rememberDismissed(taskId)
    const { [taskId]: _t, ...restTasks } = tasks.value
    tasks.value = restTasks
    const { [taskId]: _l, ...restLogs } = taskLogs.value
    taskLogs.value = restLogs
    forgetTaskLogs(taskId)
  }

  function undismissTask(taskId) {
    forgetDismissed(taskId)
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

  const subscribeToDownloadComplete = (cb) => subscribe('download_complete', cb)

  /**
   * Resolve when a background operation reaches a terminal state; reject on
   * failure so callers can surface the backend message. Resolves immediately
   * when the task is already finished or unknown (e.g. SSE never connected).
   */
  function waitForTask(taskId, { timeoutMs = 10 * 60 * 1000 } = {}) {
    return new Promise((resolve, reject) => {
      if (!taskId) {
        resolve(null)
        return
      }
      let settled = false
      let unsubscribe = null
      let timer = null
      const finish = (fn, value) => {
        if (settled) return
        settled = true
        if (unsubscribe) unsubscribe()
        if (timer) clearTimeout(timer)
        fn(value)
      }
      const check = (task) => {
        if (!task) return
        if (task.status === 'completed') finish(resolve, task)
        else if (
          task.status === 'failed' ||
          task.status === 'cancelled' ||
          task.status === 'interrupted'
        ) {
          finish(reject, new Error(task.message || 'Operation failed'))
        }
      }
      unsubscribe = subscribe('task_updated', (task) => {
        if (task?.task_id === taskId) check(task)
      })
      check(getTask(taskId))
      timer = setTimeout(() => {
        finish(reject, new Error('Operation timed out'))
      }, timeoutMs)
    })
  }

  return {
    tasks,
    taskLogs,
    activeTasks,
    hasActiveTasks,
    connected,
    connectionStatus,
    connectionState,
    reconnectAttempt,
    lastConnectedAt,
    lastEventAt,
    disconnectedSince,
    isConnected,
    connect,
    disconnect,
    retryConnection,
    getTask,
    getTaskLogs,
    hydrateTaskLogs,
    isTaskLogsLoading,
    removeTask,
    undismissTask,
    subscribe,
    subscribeToDownloadComplete,
    waitForTask,
    handleEvent,
    attentionOperations,
    attentionLoading,
    attentionError,
    fetchAttentionOperations,
    forgetAttentionOperation,
  }
})
