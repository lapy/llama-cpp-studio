<template>
  <Teleport to="body">
    <div class="activity-dock">
      <button
        type="button"
        class="activity-toggle"
        :aria-expanded="panelOpen ? 'true' : 'false'"
        aria-controls="activity-panel"
        @click="panelOpen = !panelOpen"
      >
        Activity
        <span
          v-if="badgeCount"
          class="activity-count"
          :data-kind="badgeKind"
        >{{ badgeCount }}</span>
      </button>
      <section
        v-if="panelOpen"
        id="activity-panel"
        class="activity-panel"
        :class="{ 'activity-panel--alert': panelAlert }"
        role="region"
        aria-label="Activity"
      >
        <header class="activity-panel__bar">
          <span>{{ summaryLabel }}</span>
          <button
            v-if="finishedCount"
            type="button"
            class="activity-clear"
            @click="clearFinished"
          >
            Clear finished
          </button>
        </header>
        <p v-if="!sortedTasks.length" class="activity-empty">No recent activity.</p>
        <div v-else class="activity-panel__list">
          <article
            v-for="task in sortedTasks"
            :key="task.task_id"
            class="task-toast"
            :class="`task-toast--${task.status}`"
          >
            <div class="task-toast__main">
              <button
                type="button"
                class="task-toast__body"
                :aria-expanded="logsExpanded(task) ? 'true' : 'false'"
                :aria-label="logLabel(task)"
                @click="toggleLogs(task)"
              >
                <div class="task-toast__header">
                  <i class="pi pi-spin pi-spinner" v-if="task.status === 'running' || task.status === 'cancelling'" aria-hidden="true" />
                  <i class="pi pi-clock" v-else-if="task.status === 'queued'" aria-hidden="true" />
                  <i class="pi pi-check-circle" v-else-if="task.status === 'completed'" aria-hidden="true" />
                  <i class="pi pi-ban" v-else-if="task.status === 'cancelled' || task.status === 'canceled'" aria-hidden="true" />
                  <i class="pi pi-times-circle" v-else-if="task.status === 'failed'" aria-hidden="true" />
                  <span class="task-toast__title">{{ task.description }}</span>
                  <span class="task-toast__percent">{{ statusLabel(task) }}</span>
                </div>
                <p v-if="showsProgress(task) || detailLine(task)" class="task-toast__message">
                  {{ detailLine(task) || '\u00a0' }}
                </p>
                <ProgressBar
                  v-if="showsProgress(task)"
                  :value="task.progress"
                  :show-value="false"
                  :class="task.status === 'failed' ? 'p-progressbar-danger' : ''"
                />
                <small v-if="showsProgress(task) || downloadSummary(task)" class="task-toast__download-meta">
                  {{ downloadSummary(task) || '\u00a0' }}
                </small>
              </button>
              <div v-if="hasActions(task)" class="task-toast__actions">
                <button
                  v-if="retryableVersionId(task)"
                  type="button"
                  class="task-toast__retry"
                  aria-label="Retry task"
                  :disabled="retryTaskId === task.task_id"
                  @click.stop="retryTask(task)"
                >
                  Retry
                </button>
                <button
                  v-if="canStopTask(task)"
                  type="button"
                  class="task-toast__stop"
                  aria-label="Stop task"
                  :disabled="stopTaskId === task.task_id"
                  @click.stop="requestStopTask(task)"
                >
                  <i :class="stopTaskId === task.task_id ? 'pi pi-spin pi-spinner' : 'pi pi-stop'" aria-hidden="true" />
                </button>
                <button
                  v-if="canDismissTask(task)"
                  type="button"
                  class="task-toast__dismiss"
                  aria-label="Dismiss notification"
                  @click="dismissTaskRow(task.task_id)"
                >
                  <i class="pi pi-times" aria-hidden="true" />
                </button>
              </div>
            </div>
            <div v-if="logsExpanded(task)" class="task-toast__log-view">
              <div class="task-toast__log-bar">
                <button
                  type="button"
                  class="task-toast__log-follow"
                  :aria-pressed="isFollowing(task.task_id) ? 'true' : 'false'"
                  @click.stop="resumeFollow(task.task_id)"
                >
                  {{ isFollowing(task.task_id) ? 'Following' : 'Follow' }}
                </button>
                <button
                  type="button"
                  class="task-toast__log-copy"
                  @click.stop="copyTaskLogs(task)"
                >
                  {{ copyLabel(task.task_id) }}
                </button>
              </div>
              <pre
                :ref="logRef(task.task_id)"
                :data-log-id="task.task_id"
                class="task-toast__logs"
                @scroll="onLogScroll(task.task_id, $event)"
              >{{ getTaskLogs(task).join('\n') }}</pre>
            </div>
          </article>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onUnmounted, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ProgressBar from 'primevue/progressbar'
import { useTaskFilter } from '@/composables/useTaskFilter'
import { retryableVersionId, useTaskActions } from '@/composables/useTaskActions'
import { formatBytes } from '@/utils/formatting'

const ACTIVE_STATUSES = new Set(['running', 'queued', 'cancelling'])
const HIDDEN_ACTIVITY_TYPES = new Set(['param_scan', 'runtime_apply'])
const STATUS_RANK = {
  running: 0,
  cancelling: 1,
  queued: 2,
  failed: 3,
  cancelled: 4,
  canceled: 4,
  completed: 5,
}

const { filteredTasks: visibleTasks } = useTaskFilter({
  type: null,
  showCompleted: true,
})

function belongsInActivity(task) {
  if (HIDDEN_ACTIVITY_TYPES.has(task?.type)) return false
  if (task?.metadata?.recovered && task?.status === 'completed') return false
  return true
}

const activityTasks = computed(() => visibleTasks.value.filter(belongsInActivity))

const panelOpen = ref(false)
const panelAlert = ref(false)
const expandedLogs = ref({})
const seenTaskIds = new Set()
const seenStatus = new Map()
let alertTimer = 0

function announceActivity() {
  panelOpen.value = true
  panelAlert.value = true
  window.clearTimeout(alertTimer)
  alertTimer = window.setTimeout(() => {
    panelAlert.value = false
  }, 2400)
}

onUnmounted(() => {
  window.clearTimeout(alertTimer)
})

const sortedTasks = computed(() => [...activityTasks.value].sort((a, b) => {
  const rank = (STATUS_RANK[a.status] ?? 6) - (STATUS_RANK[b.status] ?? 6)
  if (rank !== 0) return rank
  return String(a.task_id).localeCompare(String(b.task_id))
}))

const activeCount = computed(() => sortedTasks.value.filter((task) => ACTIVE_STATUSES.has(task.status)).length)
const failedCount = computed(() => sortedTasks.value.filter((task) => task.status === 'failed').length)
const finishedCount = computed(() => sortedTasks.value.filter((task) => !ACTIVE_STATUSES.has(task.status)).length)
const badgeCount = computed(() => activeCount.value || failedCount.value)
const badgeKind = computed(() => (activeCount.value ? 'active' : 'failed'))
const summaryLabel = computed(() => {
  if (activeCount.value === 1) return '1 running'
  if (activeCount.value > 1) return `${activeCount.value} running`
  if (sortedTasks.value.length) return 'Recent'
  return 'Activity'
})

watch(activityTasks, (tasks) => {
  let open = false
  for (const task of tasks) {
    const id = task.task_id
    const status = task.status
    const previous = seenStatus.get(id)
    const hot = ACTIVE_STATUSES.has(status) || status === 'failed'
    const arrived = !seenTaskIds.has(id)
    const failedNow = status === 'failed' && previous != null && previous !== 'failed'
    if ((arrived && hot) || failedNow) open = true
    seenTaskIds.add(id)
    seenStatus.set(id, status)
  }
  if (open) announceActivity()
}, { immediate: true })

function statusLabel(task) {
  const status = String(task?.status || '')
  if (status === 'queued') return 'Queued'
  if (status === 'cancelled' || status === 'canceled') return 'Canceled'
  if (status === 'cancelling') return 'Stopping'
  if (status === 'failed') return 'Failed'
  if (status === 'completed') return 'Done'
  return `${Math.round(Number(task?.progress) || 0)}%`
}

function showsProgress(task) {
  return ACTIVE_STATUSES.has(task.status) || task.status === 'failed'
}

function detailLine(task) {
  if (downloadSummary(task)) return ''
  const message = String(task?.error || task?.message || '').trim()
  if (!message || message === String(task?.description || '').trim()) return ''
  return message
}

function hasActions(task) {
  return Boolean(retryableVersionId(task) || canStopTask(task) || canDismissTask(task))
}

function canDismissTask(task) {
  return !ACTIVE_STATUSES.has(task?.status)
}

const { dismissTask, canStopTask, requestStopTask, stopTaskId, retryTask, retryTaskId, getTaskLogs, progressStore } = useTaskActions()
const { taskLogs } = storeToRefs(progressStore)
const logPreEls = {}
const followLog = ref({})
const copiedLog = ref({})
const lastScrollTop = new Map()
const pinLock = new Set()

function downloadSummary(task) {
  const downloaded = Number(task?.metadata?.bytes_downloaded)
  const total = Number(task?.metadata?.total_bytes)
  if (!Number.isFinite(total) || total <= 0) return ''

  const safeDownloaded = Number.isFinite(downloaded) ? Math.min(Math.max(downloaded, 0), total) : 0
  const parts = [
    `${formatBytes(safeDownloaded)} / ${formatBytes(total)}`,
    `${formatBytes(Math.max(total - safeDownloaded, 0))} left`,
  ]
  const fileNumber = Number(task?.metadata?.files_completed)
  const fileCount = Number(task?.metadata?.files_total)
  if (Number.isFinite(fileNumber) && Number.isFinite(fileCount) && fileCount > 1) {
    parts.push(`file ${Math.min(Math.max(fileNumber, 1), fileCount)}/${fileCount}`)
  }
  return parts.join(' · ')
}

function logsExpanded(task) {
  return Boolean(expandedLogs.value[task.task_id]) && getTaskLogs(task).length > 0
}

function logLabel(task) {
  const logs = getTaskLogs(task)
  if (!logs.length) return task.description || 'Activity'
  return logsExpanded(task) ? `Hide logs for ${task.description}` : `Show logs for ${task.description}`
}

function toggleLogs(task) {
  if (!getTaskLogs(task).length) return
  const id = task.task_id
  const nextOpen = !expandedLogs.value[id]
  expandedLogs.value = {
    ...expandedLogs.value,
    [id]: nextOpen,
  }
  if (nextOpen) followLog.value = { ...followLog.value, [id]: true }
}

function isFollowing(taskId) {
  return followLog.value[taskId] !== false
}

function scrollPreToBottom(el) {
  if (!el) return
  const taskId = el.dataset?.logId
  if (taskId) pinLock.add(String(taskId))
  el.scrollTop = el.scrollHeight
  if (taskId) lastScrollTop.set(String(taskId), el.scrollTop)
  queueMicrotask(() => {
    if (taskId) pinLock.delete(String(taskId))
  })
}

const logRefSetters = new Map()

function logRef(taskId) {
  let setter = logRefSetters.get(taskId)
  if (!setter) {
    setter = (el) => setLogPreRef(taskId, el)
    logRefSetters.set(taskId, setter)
  }
  return setter
}

function setLogPreRef(taskId, el) {
  if (el) {
    logPreEls[taskId] = el
    if (isFollowing(taskId)) {
      nextTick(() => requestAnimationFrame(() => scrollPreToBottom(el)))
    }
  } else {
    delete logPreEls[taskId]
  }
}

function onLogScroll(taskId, event) {
  const el = event?.target
  if (!el) return
  const id = String(taskId)
  if (pinLock.has(id)) {
    lastScrollTop.set(id, el.scrollTop)
    return
  }
  const distance = el.scrollHeight - el.scrollTop - el.clientHeight
  const nearBottom = distance < 48
  const previous = lastScrollTop.get(id)
  // New log lines grow scrollHeight without moving scrollTop. That is not
  // the user leaving the bottom; only an upward scroll releases follow.
  const movedUp = previous != null && el.scrollTop < previous - 8
  lastScrollTop.set(id, el.scrollTop)

  if (isFollowing(id)) {
    if (nearBottom || !movedUp) {
      if (!nearBottom) scrollPreToBottom(el)
      return
    }
    followLog.value = { ...followLog.value, [id]: false }
    return
  }
  if (nearBottom) {
    followLog.value = { ...followLog.value, [id]: true }
  }
}

function logElement(taskId) {
  const cached = logPreEls[taskId]
  if (cached?.isConnected) return cached
  if (typeof document === 'undefined') return null
  const el = document.querySelector(`[data-log-id="${CSS.escape(String(taskId))}"]`)
  if (el) logPreEls[taskId] = el
  return el
}

function resumeFollow(taskId) {
  followLog.value = { ...followLog.value, [taskId]: true }
  scrollPreToBottom(logElement(taskId))
}

function scrollFollowingLogs() {
  nextTick(() => {
    requestAnimationFrame(() => {
      for (const task of sortedTasks.value) {
        if (!logsExpanded(task) || !isFollowing(task.task_id)) continue
        scrollPreToBottom(logElement(task.task_id))
      }
    })
  })
}

watch(taskLogs, scrollFollowingLogs, { deep: true })

function copyLabel(taskId) {
  return copiedLog.value[taskId] ? 'Copied' : 'Copy'
}

async function copyTaskLogs(task) {
  const text = getTaskLogs(task).join('\n')
  if (!text) return
  try {
    await navigator.clipboard.writeText(text)
  } catch (_) {
    const area = document.createElement('textarea')
    area.value = text
    area.setAttribute('readonly', '')
    area.style.position = 'fixed'
    area.style.left = '-9999px'
    document.body.appendChild(area)
    area.select()
    document.execCommand('copy')
    document.body.removeChild(area)
  }
  copiedLog.value = { ...copiedLog.value, [task.task_id]: true }
  window.setTimeout(() => {
    if (!copiedLog.value[task.task_id]) return
    const next = { ...copiedLog.value }
    delete next[task.task_id]
    copiedLog.value = next
  }, 1500)
}

function dismissTaskRow(taskId) {
  dismissTask(taskId, expandedLogs, logPreEls)
  const nextFollow = { ...followLog.value }
  delete nextFollow[taskId]
  followLog.value = nextFollow
}

function clearFinished() {
  sortedTasks.value
    .filter((task) => !ACTIVE_STATUSES.has(task.status))
    .forEach((task) => dismissTaskRow(task.task_id))
}
</script>

<style scoped>
.activity-dock {
  position: fixed;
  right: max(1rem, env(safe-area-inset-right));
  bottom: max(3.25rem, calc(env(safe-area-inset-bottom) + 2.5rem));
  /* Above dialogs (~1100), the tour (10000), and tooltips (11000).
     The external-access lock stays higher, at 20000. */
  z-index: 19000;
  display: flex;
  flex-direction: column-reverse;
  align-items: flex-end;
  gap: 0.5rem;
  width: min(40rem, calc(100vw - 1.5rem));
  pointer-events: none;
}

.activity-toggle,
.activity-panel {
  pointer-events: auto;
}

.activity-toggle {
  border: 1px solid var(--border-primary);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-radius: 999px;
  min-height: 2.4rem;
  padding: 0.35rem 0.8rem;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
}

.activity-count {
  display: inline-flex;
  margin-left: 0.4rem;
  min-width: 1.25rem;
  justify-content: center;
  border-radius: 999px;
  background: var(--nav-active-bg, #0e7490);
  color: var(--nav-active-fg, #f8fafc);
  font-size: 0.75rem;
  padding: 0 0.35rem;
}

.activity-count[data-kind="failed"] {
  background: var(--status-error, #b91c1c);
  color: #f8fafc;
}

.activity-panel {
  width: 100%;
  max-height: min(78vh, 40rem);
  overflow: auto;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg, 0.75rem);
  background: var(--bg-secondary);
  padding: 0.45rem;
  box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45);
}

.activity-panel--alert {
  border-color: var(--accent-cyan, #22d3ee);
  box-shadow:
    0 16px 48px rgba(0, 0, 0, 0.45),
    0 0 0 2px var(--accent-cyan, #22d3ee);
  animation: activity-arrive 0.45s ease-out;
}

@keyframes activity-arrive {
  from {
    transform: translateY(0.75rem);
    opacity: 0.35;
  }
  to {
    transform: none;
    opacity: 1;
  }
}

@media (prefers-reduced-motion: reduce) {
  .activity-panel--alert {
    animation: none;
  }
}

.activity-panel__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.15rem 0.35rem 0.4rem;
  color: var(--text-secondary);
  font-size: 0.75rem;
  font-weight: 650;
}

.activity-clear {
  border: 0;
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: 0.75rem;
  cursor: pointer;
  padding: 0.1rem 0.2rem;
}

.activity-panel__list {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.activity-empty {
  margin: 0.25rem;
  color: var(--text-secondary);
  font-size: 0.85rem;
}

.task-toast {
  display: flex;
  flex-direction: column;
  border-radius: 0.55rem;
  border: 1px solid var(--border-primary);
  background: var(--bg-tertiary);
  overflow: hidden;
  position: relative;
}

.task-toast__main {
  display: flex;
  align-items: stretch;
  min-width: 0;
}

.task-toast__actions {
  display: flex;
  flex-shrink: 0;
}

.task-toast::before {
  content: '';
  position: absolute;
  inset: 0 auto 0 0;
  width: 0.28rem;
  background: var(--accent-primary, #3b82f6);
}

.task-toast--running::before {
  background: var(--accent-primary, #3b82f6);
}

.task-toast--failed {
  border-color: color-mix(in srgb, var(--color-error, #ef4444) 55%, transparent);
}

.task-toast--failed::before {
  background: var(--color-error, #ef4444);
}

.task-toast--completed {
  border-color: color-mix(in srgb, #22c55e 45%, transparent);
}

.task-toast--completed::before {
  background: #22c55e;
}

.task-toast__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.3rem;
  padding: 0.55rem 0.7rem 0.55rem 0.85rem;
  border: none;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
  transition: background 0.15s ease;
}

.task-toast__body:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.05));
}

.task-toast__header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  min-width: 0;
}

.task-toast__header .pi {
  flex-shrink: 0;
  font-size: 1rem;
}

.task-toast__header .pi-spinner {
  color: var(--accent-primary, #60a5fa);
}

.task-toast__header .pi-check-circle {
  color: #22c55e;
}

.task-toast__header .pi-times-circle {
  color: #ef4444;
}

.task-toast__title {
  flex: 1;
  min-width: 0;
  font-size: 0.84rem;
  font-weight: 650;
  letter-spacing: 0.01em;
  color: var(--text-primary, #f3f4f6);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-toast__percent {
  flex-shrink: 0;
  width: 4.75rem;
  text-align: right;
  font-size: 0.875rem;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary, #f3f4f6);
}

.task-toast__message,
.task-toast__download-meta {
  display: block;
  height: 1.2em;
  margin: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  line-height: 1.2;
}

.task-toast__message {
  font-size: 0.78rem;
  color: var(--text-secondary, #c4c9d4);
}

.task-toast__download-meta {
  color: var(--text-secondary, #c4c9d4);
  font-size: 0.72rem;
  font-variant-numeric: tabular-nums;
}

.task-toast__log-view {
  display: flex;
  flex-direction: column;
  border-top: 1px solid var(--border-primary);
  background: var(--bg-primary);
}

.task-toast__log-bar {
  display: flex;
  justify-content: flex-end;
  gap: 0.35rem;
  padding: 0.35rem 0.55rem 0;
}

.task-toast__log-follow,
.task-toast__log-copy {
  border: 1px solid var(--border-primary);
  background: transparent;
  color: var(--text-secondary);
  border-radius: 999px;
  min-height: 1.6rem;
  padding: 0.1rem 0.6rem;
  font: inherit;
  font-size: 0.72rem;
  font-weight: 650;
  cursor: pointer;
}

.task-toast__log-follow[aria-pressed='true'] {
  color: var(--text-primary);
  border-color: var(--nav-active-bg, #0e7490);
}

.task-toast__log-follow:hover,
.task-toast__log-copy:hover {
  color: var(--text-primary);
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.06));
}

.task-toast__logs {
  margin: 0.35rem 0.45rem 0.45rem;
  max-height: min(42vh, 22rem);
  min-height: 8rem;
  overflow: auto;
  padding: 0.65rem 0.75rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-md, 0.5rem);
  background: color-mix(in srgb, var(--bg-primary, #11131c) 88%, black);
  color: var(--text-secondary);
  font-size: 0.75rem;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
}

.task-toast__retry,
.task-toast__stop,
.task-toast__dismiss {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2.35rem;
  flex-shrink: 0;
  border: none;
  border-left: 1px solid var(--border-primary, #2a2f45);
  background: transparent;
  color: var(--text-secondary, #9ca3af);
  cursor: pointer;
  transition:
    background 0.15s ease,
    color 0.15s ease;
}

.task-toast__retry {
  width: auto;
  padding: 0 0.55rem;
  font: inherit;
  font-size: 0.75rem;
  font-weight: 650;
}

.task-toast__stop:hover:not(:disabled),
.task-toast__retry:hover:not(:disabled),
.task-toast__dismiss:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.06));
  color: var(--text-primary, #f3f4f6);
}

.task-toast__stop:disabled,
.task-toast__retry:disabled {
  cursor: wait;
  opacity: 0.7;
}

.task-toast__stop .pi-stop {
  color: #ef4444;
}

.task-toast-enter-active,
.task-toast-leave-active {
  transition:
    opacity 0.22s ease,
    transform 0.22s ease;
}

.task-toast-enter-from,
.task-toast-leave-to {
  opacity: 0;
  transform: translateY(0.85rem) scale(0.98);
}

.task-toast-move {
  transition: transform 0.2s ease;
}

:deep(.task-toast .p-progressbar) {
  height: 0.55rem;
  margin: 0;
  flex: none;
  border-radius: 999px;
  background: color-mix(in srgb, var(--bg-primary, #11131c) 70%, white 8%);
  overflow: hidden;
}

:deep(.task-toast .p-progressbar .p-progressbar-value) {
  border-radius: 999px;
  background: linear-gradient(
    90deg,
    color-mix(in srgb, var(--accent-primary, #3b82f6) 80%, white 10%),
    var(--accent-primary, #60a5fa)
  );
}

:deep(.task-toast .p-progressbar-danger .p-progressbar-value) {
  background: linear-gradient(90deg, #dc2626, #f87171);
}
</style>
