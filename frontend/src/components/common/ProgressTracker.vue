<template>
  <div v-if="activeTasks.length > 0" class="progress-tracker">
    <div v-if="sectionTitle || hasDismissibleTasks" class="progress-tracker__head">
      <div v-if="sectionTitle" class="progress-tracker__section-title">{{ sectionTitle }}</div>
      <button
        v-if="hasDismissibleTasks"
        type="button"
        class="dismiss-all-btn"
        @click="dismissAllFinished"
      >
        Clear finished
      </button>
    </div>
    <div
      v-for="task in activeTasks"
      :key="task.task_id"
      class="progress-item"
      :class="`status-${task.status}`"
    >
      <div class="progress-header">
        <div class="task-info">
          <i class="pi pi-spin pi-spinner" v-if="task.status === 'running'" />
          <i class="pi pi-check-circle text-success" v-else-if="task.status === 'completed'" />
          <i class="pi pi-times-circle text-danger" v-else-if="task.status === 'failed'" />
          <span class="task-description">{{ task.description }}</span>
        </div>
        <div class="progress-meta">
          <Button
            v-if="canStopSourceBuild(task)"
            label="Stop build"
            icon="pi pi-stop"
            severity="danger"
            outlined
            size="small"
            :loading="stopBuildTaskId === task.task_id"
            :disabled="stopBuildTaskId === task.task_id"
            @click.stop="requestStopBuild(task)"
          />
          <button
            v-if="getTaskLogs(task).length > 0"
            type="button"
            class="logs-toggle"
            @click="toggleLogs(task.task_id)"
          >
            {{ isExpanded(task.task_id) ? 'Hide logs' : 'Show logs' }}
          </button>
          <span class="progress-percent">{{ Math.round(task.progress) }}%</span>
          <button
            v-if="isTaskDismissible(task)"
            type="button"
            class="dismiss-task-btn"
            title="Dismiss"
            aria-label="Dismiss this progress entry"
            @click.stop="dismissTaskRow(task.task_id)"
          >
            <i class="pi pi-times" aria-hidden="true" />
          </button>
        </div>
      </div>
      <ProgressBar :value="task.progress" :class="task.status === 'failed' ? 'p-progressbar-danger' : ''" />
      <small v-if="task.message" class="task-message" :class="task.status === 'failed' ? 'text-danger' : 'text-muted'">
        {{ task.message }}
      </small>
      <pre
        v-if="isExpanded(task.task_id) && getTaskLogs(task).length > 0"
        :ref="(el) => setLogPreRef(task.task_id, el)"
        class="task-logs"
      >{{ getTaskLogs(task).join('\n') }}</pre>
    </div>
  </div>
</template>

<script setup>
import { computed, nextTick, ref, watch } from 'vue'
import { storeToRefs } from 'pinia'
import ProgressBar from 'primevue/progressbar'
import Button from 'primevue/button'
import { useToast } from 'primevue/usetoast'
import { useProgressStore } from '@/stores/progress'
import { useEnginesStore } from '@/stores/engines'

const props = defineProps({
  /** Single type string or array of types to show (e.g. ['build', 'install_release']) */
  type: {
    type: [String, Array],
    default: null,
  },
  metadataKey: {
    type: String,
    default: null,
  },
  metadataValue: {
    type: [String, Number, Boolean],
    default: null,
  },
  showCompleted: {
    type: Boolean,
    default: false,
  },
  /** Shown only when there is at least one matching task (same visibility as the tracker). */
  sectionTitle: {
    type: String,
    default: null,
  },
  /** Allow removing completed/failed rows from the store (hides the card). */
  dismissible: {
    type: Boolean,
    default: true,
  },
})

const progressStore = useProgressStore()
const enginesStore = useEnginesStore()
const toast = useToast()
const { taskLogs } = storeToRefs(progressStore)
const expandedLogs = ref({})
const stopBuildTaskId = ref(null)

function canStopSourceBuild(task) {
  return task?.status === 'running' && task?.type === 'build'
}

async function requestStopBuild(task) {
  const id = task?.task_id
  if (!id) return
  stopBuildTaskId.value = id
  try {
    const res = await enginesStore.cancelSourceBuild(id)
    if (res?.ok === false) {
      toast.add({
        severity: 'warn',
        summary: 'Stop build',
        detail: res?.message || 'Could not cancel this build.',
        life: 5000,
      })
      return
    }
    toast.add({
      severity: 'info',
      summary: 'Stop requested',
      detail: 'The compiler process will be terminated shortly.',
      life: 4000,
    })
  } catch (e) {
    toast.add({
      severity: 'error',
      summary: 'Stop build failed',
      detail: e?.response?.data?.detail || e?.message || 'Request failed',
      life: 5000,
    })
  } finally {
    stopBuildTaskId.value = null
  }
}

const activeTasks = computed(() => {
  const allTasks = Object.values(progressStore.tasks)
  const types = props.type == null
    ? null
    : Array.isArray(props.type)
      ? props.type
      : [props.type]
  return allTasks.filter((t) => {
    const typeMatch = !types || types.length === 0 || types.includes(t.type)
    const metadataMatch = !props.metadataKey || t?.metadata?.[props.metadataKey] === props.metadataValue
    const statusMatch = t.status === 'running' || (props.showCompleted && t.status === 'completed') || t.status === 'failed'
    return typeMatch && metadataMatch && statusMatch
  })
})

const hasDismissibleTasks = computed(() => {
  if (!props.dismissible) return false
  return activeTasks.value.some((t) => t.status !== 'running')
})

function isTaskDismissible(task) {
  return props.dismissible && task?.status && task.status !== 'running'
}

function dismissTaskRow(taskId) {
  if (!taskId) return
  progressStore.removeTask(taskId)
  const { [taskId]: _, ...restExp } = expandedLogs.value
  expandedLogs.value = restExp
  delete logPreEls[taskId]
}

function dismissAllFinished() {
  const ids = activeTasks.value.filter((t) => t.status !== 'running').map((t) => t.task_id)
  ids.forEach((id) => dismissTaskRow(id))
}

function getTaskLogs(task) {
  return progressStore.getTaskLogs(task.task_id)
}

function isExpanded(taskId) {
  return Boolean(expandedLogs.value[taskId])
}

/** @type {Record<string, HTMLElement | undefined>} */
const logPreEls = {}

function scrollPreToBottom(el) {
  if (!el) return
  el.scrollTop = el.scrollHeight
}

function setLogPreRef(taskId, el) {
  if (el) {
    logPreEls[taskId] = el
    nextTick(() => {
      requestAnimationFrame(() => scrollPreToBottom(el))
    })
  } else {
    delete logPreEls[taskId]
  }
}

function scrollVisibleExpandedLogsToBottom() {
  nextTick(() => {
    requestAnimationFrame(() => {
      for (const task of activeTasks.value) {
        if (!isExpanded(task.task_id)) continue
        if (getTaskLogs(task).length === 0) continue
        scrollPreToBottom(logPreEls[task.task_id])
      }
    })
  })
}

watch(taskLogs, scrollVisibleExpandedLogsToBottom, { deep: true })
watch(expandedLogs, scrollVisibleExpandedLogsToBottom, { deep: true })

function toggleLogs(taskId) {
  expandedLogs.value = {
    ...expandedLogs.value,
    [taskId]: !expandedLogs.value[taskId],
  }
}
</script>

<style scoped>
.progress-tracker {
  display: flex;
  flex-direction: column;
  gap: var(--spacing-md, 0.75rem);
  margin: var(--spacing-md, 0.75rem) 0;
}

.progress-tracker__head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem;
  flex-wrap: wrap;
  margin: 0 0 -0.2rem;
}

.progress-tracker__section-title {
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin: 0;
}

.dismiss-all-btn {
  border: none;
  background: transparent;
  color: var(--text-secondary, #9ca3af);
  font-size: 0.72rem;
  font-weight: 600;
  cursor: pointer;
  padding: 0.15rem 0.35rem;
  border-radius: var(--radius-sm, 4px);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.dismiss-all-btn:hover {
  color: var(--text-primary, #f3f4f6);
}

.dismiss-task-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.65rem;
  height: 1.65rem;
  padding: 0;
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-sm, 6px);
  background: transparent;
  color: var(--text-secondary, #9ca3af);
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease, border-color 0.15s ease;
}

.dismiss-task-btn:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.06));
  color: var(--text-primary, #f3f4f6);
  border-color: var(--border-hover, #3b4261);
}

.dismiss-task-btn .pi {
  font-size: 0.8rem;
}

.progress-item {
  background: var(--bg-surface, #1e2235);
  border: 1px solid var(--border-primary, #2a2f45);
  border-radius: var(--radius-md, 0.5rem);
  padding: var(--spacing-md, 0.75rem);
  display: flex;
  flex-direction: column;
  gap: var(--spacing-sm, 0.5rem);
}

.progress-item.status-failed {
  border-color: var(--color-error, #ef4444);
  background: rgba(239, 68, 68, 0.05);
}

.progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--spacing-sm, 0.5rem);
}

.progress-meta {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm, 0.5rem);
  flex-shrink: 0;
}

.task-info {
  display: flex;
  align-items: center;
  gap: var(--spacing-sm, 0.5rem);
  flex: 1;
  min-width: 0;
}

.task-description {
  font-weight: 500;
  font-size: 0.875rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.progress-percent {
  font-size: 0.75rem;
  font-weight: 600;
  color: var(--text-secondary, #9ca3af);
  flex-shrink: 0;
}

.logs-toggle {
  border: 1px solid var(--border-primary, #2a2f45);
  background: transparent;
  color: var(--text-secondary, #9ca3af);
  border-radius: 999px;
  padding: 0.2rem 0.55rem;
  font-size: 0.75rem;
  line-height: 1.2;
  cursor: pointer;
  transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease;
}

.logs-toggle:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.04));
  border-color: var(--border-hover, #3b4261);
  color: var(--text-primary, #f3f4f6);
}

.task-message {
  font-size: 0.75rem;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-logs {
  margin: 0;
  padding: 0.75rem;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.03));
  color: var(--text-secondary, #d1d5db);
  font-size: 0.75rem;
  line-height: 1.5;
  white-space: pre-wrap;
  word-break: break-word;
  max-height: min(42vh, 28rem);
  overflow: auto;
}

.text-success { color: #22c55e; }
.text-danger  { color: #ef4444; }
.text-muted   { color: var(--text-secondary, #9ca3af); }
</style>
