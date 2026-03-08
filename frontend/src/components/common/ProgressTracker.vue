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
        <span class="progress-percent">{{ Math.round(task.progress) }}%</span>
      </div>
      <ProgressBar :value="task.progress" :class="task.status === 'failed' ? 'p-progressbar-danger' : ''" />
      <small v-if="task.message" class="task-message" :class="task.status === 'failed' ? 'text-danger' : 'text-muted'">
        {{ task.message }}
      </small>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'
import ProgressBar from 'primevue/progressbar'
import { useProgressStore } from '@/stores/progress'

const props = defineProps({
  /** Single type string or array of types to show (e.g. ['build', 'install_release']) */
  type: {
    type: [String, Array],
    default: null,
  },
  showCompleted: {
    type: Boolean,
    default: false,
  },
})

const progressStore = useProgressStore()

const activeTasks = computed(() => {
  const allTasks = Object.values(progressStore.tasks)
  const types = props.type == null
    ? null
    : Array.isArray(props.type)
      ? props.type
      : [props.type]
  return allTasks.filter((t) => {
    const typeMatch = !types || types.length === 0 || types.includes(t.type)
    const statusMatch = t.status === 'running' || (props.showCompleted && t.status === 'completed') || t.status === 'failed'
    return typeMatch && statusMatch
  })
})
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

.task-message {
  font-size: 0.75rem;
  display: block;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.text-success { color: #22c55e; }
.text-danger  { color: #ef4444; }
.text-muted   { color: var(--text-secondary, #9ca3af); }
</style>
