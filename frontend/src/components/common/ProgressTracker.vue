<template>
  <div v-if="activeTasks.length > 0" class="progress-tracker">
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
          <button
            v-if="getTaskLogs(task).length > 0"
            type="button"
            class="logs-toggle"
            @click="toggleLogs(task.task_id)"
          >
            {{ isExpanded(task.task_id) ? 'Hide logs' : 'Show logs' }}
          </button>
          <span class="progress-percent">{{ Math.round(task.progress) }}%</span>
        </div>
      </div>
      <ProgressBar :value="task.progress" :class="task.status === 'failed' ? 'p-progressbar-danger' : ''" />
      <small v-if="task.message" class="task-message" :class="task.status === 'failed' ? 'text-danger' : 'text-muted'">
        {{ task.message }}
      </small>
      <pre v-if="isExpanded(task.task_id) && getTaskLogs(task).length > 0" class="task-logs">{{ getTaskLogs(task).join('\n') }}</pre>
    </div>
  </div>
</template>

<script setup>
import { computed, ref } from 'vue'
import ProgressBar from 'primevue/progressbar'
import { useProgressStore } from '@/stores/progress'

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
})

const progressStore = useProgressStore()
const expandedLogs = ref({})

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

function getTaskLogs(task) {
  return progressStore.getTaskLogs(task.task_id)
}

function isExpanded(taskId) {
  return Boolean(expandedLogs.value[taskId])
}

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
  max-height: 14rem;
  overflow: auto;
}

.text-success { color: #22c55e; }
.text-danger  { color: #ef4444; }
.text-muted   { color: var(--text-secondary, #9ca3af); }
</style>
