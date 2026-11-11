<template>
  <Teleport to="#activity-anchor" defer>
    <div class="activity-dock" @keydown.esc.stop.prevent="closePanel">
      <button
        type="button"
        ref="activityToggle"
        class="activity-toggle"
        aria-label="Activity"
        :aria-expanded="panelOpen ? 'true' : 'false'"
        aria-controls="activity-panel"
        @click="panelOpen = !panelOpen"
      >
        <i class="pi pi-bolt" aria-hidden="true" />
        <span class="activity-toggle__label">Activity</span>
        <span v-if="badgeCount" class="activity-count" :data-kind="badgeKind">{{
          badgeCount
        }}</span>
      </button>
      <p class="sr-only" role="status">{{ activityAnnouncement }}</p>
      <p v-if="recoveryUncertain" class="activity-recovery" role="status">
        {{ recoveryDetail }}
        <button type="button" @click="refreshRecovery">Refresh</button>
      </p>
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
          <button v-if="finishedCount" type="button" class="activity-clear" @click="clearFinished">
            Clear finished
          </button>
          <button
            type="button"
            class="activity-clear"
            aria-label="Close activity"
            @click="closePanel"
          >
            Close
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
                  <i
                    class="pi pi-spin pi-spinner"
                    v-if="task.status === 'running' || task.status === 'cancelling'"
                    aria-hidden="true"
                  />
                  <i class="pi pi-clock" v-else-if="task.status === 'queued'" aria-hidden="true" />
                  <i
                    class="pi pi-check-circle"
                    v-else-if="task.status === 'completed'"
                    aria-hidden="true"
                  />
                  <i
                    class="pi pi-ban"
                    v-else-if="task.status === 'cancelled' || task.status === 'canceled'"
                    aria-hidden="true"
                  />
                  <i
                    class="pi pi-times-circle"
                    v-else-if="task.status === 'failed'"
                    aria-hidden="true"
                  />
                  <span class="task-toast__title">{{ task.description }}</span>
                  <span class="task-toast__percent" aria-hidden="true">{{
                    statusLabel(task)
                  }}</span>
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
                <small
                  v-if="showsProgress(task) || downloadSummary(task)"
                  class="task-toast__download-meta"
                  aria-hidden="true"
                >
                  {{ downloadSummary(task) || '\u00a0' }}
                </small>
              </button>
              <div v-if="hasActions(task)" class="task-toast__actions">
                <button
                  v-if="retryableVersionId(task)"
                  type="button"
                  class="task-toast__retry"
                  :aria-label="`Retry ${task.description || 'task'}`"
                  :disabled="retryTaskId === task.task_id"
                  @click.stop="retryTask(task)"
                >
                  Retry
                </button>
                <button
                  v-if="canStopTask(task)"
                  type="button"
                  class="task-toast__stop"
                  :aria-label="`Stop ${task.description || 'task'}`"
                  :disabled="stopTaskId === task.task_id"
                  @click.stop="requestStopTask(task)"
                >
                  <i
                    :class="stopTaskId === task.task_id ? 'pi pi-spin pi-spinner' : 'pi pi-stop'"
                    aria-hidden="true"
                  />
                </button>
                <button
                  v-if="canDismissTask(task)"
                  type="button"
                  class="task-toast__dismiss"
                  :aria-label="`Dismiss ${task.description || 'notification'}`"
                  @click="dismissTaskRow(task.task_id)"
                >
                  <i class="pi pi-times" aria-hidden="true" />
                </button>
              </div>
            </div>
            <div v-if="logsExpanded(task)" class="task-toast__log-view">
              <LogOutput
                :lines="getTaskLogs(task)"
                :loading="isTaskLogsLoading(task.task_id)"
                :live="task.status === 'running' || task.status === 'cancelling'"
                aria-label="Task log output"
                empty-message="No output yet."
              />
            </div>
          </article>
        </div>
      </section>
    </div>
  </Teleport>
</template>

<script setup>
import { computed, nextTick, onMounted, onUnmounted, ref, watch } from 'vue'
import axios from 'axios'
import ProgressBar from 'primevue/progressbar'
import LogOutput from './LogOutput.vue'
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

const PANEL_OPEN_KEY = 'llama-studio.activity.open'
const EXPANDED_LOGS_KEY = 'llama-studio.activity.expandedLogs'

function readSessionValue(key) {
  if (typeof sessionStorage === 'undefined') return ''
  try {
    return sessionStorage.getItem(key) || ''
  } catch (_) {
    return ''
  }
}

function writeSessionValue(key, value) {
  if (typeof sessionStorage === 'undefined') return
  try {
    sessionStorage.setItem(key, value)
  } catch (_) {}
}

function readExpandedLogs() {
  const raw = readSessionValue(EXPANDED_LOGS_KEY)
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw)
    if (!Array.isArray(parsed)) return {}
    return Object.fromEntries(
      parsed.filter((id) => typeof id === 'string' && id).map((id) => [id, true]),
    )
  } catch (_) {
    return {}
  }
}

const panelOpen = ref(readSessionValue(PANEL_OPEN_KEY) === '1')
const activityToggle = ref(null)
async function closePanel() {
  panelOpen.value = false
  await nextTick()
  activityToggle.value?.focus()
}
const activityAnnouncement = ref('')
const recoveryUncertain = ref(false)
const recoveryDetail = ref(
  'The recovery outcome could not be established. Your edits are still here. Refresh before trying again.',
)
const panelAlert = ref(false)
const expandedLogs = ref(readExpandedLogs())
const seenTaskIds = new Set()
const seenStatus = new Map()
let alertTimer = 0

watch(panelOpen, (open) => writeSessionValue(PANEL_OPEN_KEY, open ? '1' : '0'))
watch(
  expandedLogs,
  (expanded) =>
    writeSessionValue(
      EXPANDED_LOGS_KEY,
      JSON.stringify(
        Object.keys(expanded)
          .filter((id) => expanded[id])
          .slice(-200),
      ),
    ),
  { deep: true },
)

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

const sortedTasks = computed(() =>
  [...activityTasks.value].sort((a, b) => {
    const rank = (STATUS_RANK[a.status] ?? 6) - (STATUS_RANK[b.status] ?? 6)
    if (rank !== 0) return rank
    const stamp = (task) => {
      const raw = task.updated_at || task.completed_at || task.created_at || task.started_at
      return typeof raw === 'number' ? raw * (raw < 1e12 ? 1000 : 1) : Date.parse(raw || '') || 0
    }
    return stamp(b) - stamp(a) || String(a.task_id).localeCompare(String(b.task_id))
  }),
)

const activeCount = computed(
  () => sortedTasks.value.filter((task) => ACTIVE_STATUSES.has(task.status)).length,
)
const failedCount = computed(
  () => sortedTasks.value.filter((task) => task.status === 'failed').length,
)
const finishedCount = computed(
  () => sortedTasks.value.filter((task) => !ACTIVE_STATUSES.has(task.status)).length,
)
const badgeCount = computed(() => activeCount.value || failedCount.value)
const badgeKind = computed(() => (activeCount.value ? 'active' : 'failed'))
const summaryLabel = computed(() => {
  const labels = ['running', 'queued', 'cancelling']
    .map((status) => {
      const count = sortedTasks.value.filter((task) => task.status === status).length
      return count ? `${count} ${status}` : ''
    })
    .filter(Boolean)
  return labels.join(' · ') || (sortedTasks.value.length ? 'Recent activity' : 'Activity')
})

watch(
  activityTasks,
  (tasks) => {
    // A reconnect snapshot can remove every task after an automatic announcement.
    // Do not leave an empty floating panel covering workspace controls.
    if (!tasks.length) {
      panelOpen.value = false
      panelAlert.value = false
      return
    }
    let open = false
    for (const task of tasks) {
      const id = task.task_id
      const status = task.status
      const previous = seenStatus.get(id)
      const hot = ACTIVE_STATUSES.has(status) || status === 'failed'
      const arrived = !seenTaskIds.has(id)
      const failedNow = status === 'failed' && previous != null && previous !== 'failed'
      const finishedNow = status === 'completed' && previous != null && previous !== 'completed'
      if (failedNow) activityAnnouncement.value = `${task.description || 'Activity'} failed`
      else if (finishedNow)
        activityAnnouncement.value = `${task.description || 'Activity'} finished`
      if ((arrived && hot) || failedNow) open = true
      seenTaskIds.add(id)
      seenStatus.set(id, status)
    }
    if (open) announceActivity()
  },
  { immediate: true },
)

function statusLabel(task) {
  const status = String(task?.status || '')
  if (status === 'queued') return 'Queued'
  if (status === 'cancelled' || status === 'canceled') return 'Canceled'
  if (status === 'cancelling') return 'Stopping'
  if (status === 'failed') return 'Failed'
  if (status === 'completed') return 'Done'
  if (status === 'interrupted') return 'Interrupted'
  if (status === 'unknown') return 'Unknown'
  return `${Math.round(Number(task?.progress) || 0)}%`
}

function showUncertainRecovery(detail) {
  recoveryUncertain.value = true
  if (detail) recoveryDetail.value = detail
}

async function refreshRecovery() {
  try {
    const { data } = await axios.post('/api/operations/reconcile')
    if (data?.outcome === 'unknown' || data?.committed === 'unknown') {
      showUncertainRecovery(data?.detail)
      return
    }
    recoveryUncertain.value = false
  } catch (error) {
    showUncertainRecovery(
      error?.response?.data?.detail ||
        'The recovery outcome could not be established. Your edits are still here. Refresh before trying again.',
    )
  }
}

onMounted(async () => {
  try {
    const { data } = await axios.get('/api/operations/recovery')
    if (data?.outcome === 'unknown') showUncertainRecovery(data?.detail)
  } catch (error) {
    showUncertainRecovery(error?.response?.data?.detail)
  }
})

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

const {
  dismissTask,
  canStopTask,
  requestStopTask,
  stopTaskId,
  retryTask,
  retryTaskId,
  getTaskLogs,
  progressStore,
} = useTaskActions()

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

function hasTaskLogs(task) {
  return getTaskLogs(task).length > 0 || Number(task?.log_count) > 0
}

function logsExpanded(task) {
  return (
    Boolean(expandedLogs.value[task.task_id]) &&
    (hasTaskLogs(task) || progressStore.isTaskLogsLoading(task.task_id))
  )
}

function isTaskLogsLoading(taskId) {
  return progressStore.isTaskLogsLoading(taskId)
}

function logLabel(task) {
  if (!hasTaskLogs(task)) return task.description || 'Activity'
  return logsExpanded(task)
    ? `Hide logs for ${task.description}`
    : `Show logs for ${task.description}`
}

function toggleLogs(task) {
  if (!hasTaskLogs(task)) return
  const id = task.task_id
  const nextOpen = !expandedLogs.value[id]
  expandedLogs.value = {
    ...expandedLogs.value,
    [id]: nextOpen,
  }
  if (nextOpen) void progressStore.hydrateTaskLogs(id)
}

function dismissTaskRow(taskId) {
  dismissTask(taskId, expandedLogs)
}

function clearFinished() {
  sortedTasks.value
    .filter((task) => !ACTIVE_STATUSES.has(task.status))
    .forEach((task) => dismissTaskRow(task.task_id))
}

// Persisted expansion and Activity-open state are restored after the SSE
// snapshot arrives; hydrate the retained server log for each visible card.
watch(
  () => sortedTasks.value.map((task) => `${task.task_id}:${task.status}:${task.log_count ?? 0}`),
  () => {
    for (const task of sortedTasks.value) {
      if (expandedLogs.value[task.task_id] && hasTaskLogs(task)) {
        void progressStore.hydrateTaskLogs(task.task_id)
      }
    }
  },
  { immediate: true },
)
</script>

<style scoped>
.activity-dock {
  position: static;
  /* Header stacking keeps this panel above content and below modal dialogs. */
  z-index: 19000;
  display: flex;
  flex-direction: column-reverse;
  align-items: flex-end;
  gap: 0.5rem;
  width: auto;
  pointer-events: none;
}

.activity-recovery {
  position: absolute;
  right: 1rem;
  top: calc(100% + 0.5rem);
  width: min(32rem, calc(100vw - 2rem));
}
.activity-toggle,
.activity-panel,
.activity-recovery {
  pointer-events: auto;
}

.activity-toggle {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  border: 1px solid var(--border-primary);
  background: var(--bg-secondary);
  color: var(--text-primary);
  border-radius: 999px;
  min-height: 2.75rem;
  padding: 0.35rem 0.8rem;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  box-shadow: var(--shadow-md);
}

.activity-toggle .pi {
  color: var(--accent-cyan);
  font-size: var(--text-xs);
}

.activity-count {
  display: inline-flex;
  margin-left: 0.4rem;
  min-width: 1.25rem;
  justify-content: center;
  border-radius: 999px;
  background: var(--nav-active-bg, #0e7490);
  color: var(--nav-active-fg, #f8fafc);
  font-size: var(--text-xs);
  padding: 0 0.35rem;
}

.activity-count[data-kind='failed'] {
  background: var(--status-error, #b91c1c);
  color: #f8fafc;
}

.activity-panel {
  position: absolute;
  top: calc(100% + 0.5rem);
  right: 1rem;
  width: min(40rem, calc(100vw - 2rem));
  max-height: min(calc(100dvh - 12rem), 40rem);
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

@media (max-width: 900px) {
  .activity-dock {
    right: auto;
    bottom: auto;
    width: auto;
  }

  .activity-toggle {
    width: 2.75rem;
    height: 2.75rem;
    justify-content: center;
    padding: 0;
  }

  .activity-toggle__label {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }

  .activity-count {
    position: absolute;
    top: -0.3rem;
    right: -0.3rem;
    margin: 0;
    min-width: 1.1rem;
    padding: 0 0.25rem;
    font-size: 0.62rem;
  }
}

.activity-panel__bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.15rem 0.35rem 0.4rem;
  color: var(--text-secondary);
  font-size: var(--text-xs);
  font-weight: 650;
}

.activity-clear {
  border: 0;
  background: transparent;
  color: var(--text-primary);
  font: inherit;
  font-size: var(--text-xs);
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
  font-size: var(--text-xs);
  color: var(--text-secondary, #c4c9d4);
}

.task-toast__download-meta {
  color: var(--text-secondary, #c4c9d4);
  font-size: var(--text-xs);
  font-variant-numeric: tabular-nums;
}

.task-toast__log-view {
  padding: 0.35rem;
  border-top: 1px solid var(--border-primary);
  background: var(--bg-primary);
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
  font-size: var(--text-xs);
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
