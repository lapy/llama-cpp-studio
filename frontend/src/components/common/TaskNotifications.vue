<template>
  <Teleport to="body">
    <div
      v-if="visibleTasks.length > 0"
      class="task-notifications-tray"
      aria-live="polite"
      aria-label="Task progress notifications"
    >
      <TransitionGroup name="task-toast">
        <article
          v-for="task in visibleTasks"
          :key="task.task_id"
          class="task-toast"
          :class="`task-toast--${task.status}`"
        >
          <button
            type="button"
            class="task-toast__body"
            :aria-label="`View details for ${task.description}`"
            @click="openDetail(task.task_id)"
          >
            <div class="task-toast__header">
              <i class="pi pi-spin pi-spinner" v-if="task.status === 'running'" aria-hidden="true" />
              <i class="pi pi-check-circle" v-else-if="task.status === 'completed'" aria-hidden="true" />
              <i class="pi pi-times-circle" v-else-if="task.status === 'failed'" aria-hidden="true" />
              <span class="task-toast__title">{{ task.description }}</span>
              <span class="task-toast__percent">{{ Math.round(task.progress) }}%</span>
            </div>
            <ProgressBar
              :value="task.progress"
              :show-value="false"
              :class="task.status === 'failed' ? 'p-progressbar-danger' : ''"
            />
          </button>
          <button
            v-if="canStopTask(task)"
            type="button"
            class="task-toast__stop"
            title="Stop task"
            aria-label="Stop task"
            :disabled="stopTaskId === task.task_id"
            @click.stop="requestStopTask(task)"
          >
            <i :class="stopTaskId === task.task_id ? 'pi pi-spin pi-spinner' : 'pi pi-stop'" aria-hidden="true" />
          </button>
          <button
            v-if="task.status !== 'running'"
            type="button"
            class="task-toast__dismiss"
            title="Dismiss"
            aria-label="Dismiss notification"
            @click="dismissTaskRow(task.task_id)"
          >
            <i class="pi pi-times" aria-hidden="true" />
          </button>
        </article>
      </TransitionGroup>
    </div>

    <Dialog
      v-model:visible="detailVisible"
      :header="detailHeader"
      modal
      class="dialog-width-md task-detail-dialog"
      @hide="selectedTaskId = null"
    >
      <TaskDetailPanel
        v-if="selectedTaskId"
        :task-id="selectedTaskId"
        :show-completed="true"
        :dismissible="true"
      />
      <template #footer>
        <Button label="Close" severity="secondary" outlined @click="detailVisible = false" />
      </template>
    </Dialog>
  </Teleport>
</template>

<script setup>
import { computed, ref } from 'vue'
import Dialog from 'primevue/dialog'
import Button from 'primevue/button'
import ProgressBar from 'primevue/progressbar'
import TaskDetailPanel from '@/components/common/TaskDetailPanel.vue'
import { useTaskFilter } from '@/composables/useTaskFilter'
import { useTaskActions } from '@/composables/useTaskActions'

const { filteredTasks: visibleTasks } = useTaskFilter({
  type: null,
  showCompleted: true,
})

const { dismissTask, progressStore, canStopTask, requestStopTask, stopTaskId } = useTaskActions()

const detailVisible = ref(false)
const selectedTaskId = ref(null)

const detailHeader = computed(() => {
  if (!selectedTaskId.value) return 'Task details'
  const task = progressStore.getTask(selectedTaskId.value)
  return task?.description || 'Task details'
})

function openDetail(taskId) {
  selectedTaskId.value = taskId
  detailVisible.value = true
}

function dismissTaskRow(taskId) {
  dismissTask(taskId)
  if (selectedTaskId.value === taskId) {
    detailVisible.value = false
    selectedTaskId.value = null
  }
}
</script>

<style scoped>
.task-notifications-tray {
  position: fixed;
  right: var(--spacing-md, 1rem);
  bottom: var(--spacing-md, 1rem);
  z-index: 1100;
  display: flex;
  flex-direction: column-reverse;
  gap: var(--spacing-sm, 0.5rem);
  width: min(22rem, calc(100vw - 2rem));
  pointer-events: none;
}

.task-toast {
  display: flex;
  align-items: stretch;
  gap: 0.35rem;
  pointer-events: auto;
  border-radius: var(--radius-md, 0.5rem);
  border: 1px solid var(--border-primary, #2a2f45);
  background: var(--bg-surface, #1e2235);
  box-shadow: var(--shadow-lg, 0 10px 25px rgba(0, 0, 0, 0.35));
  overflow: hidden;
}

.task-toast--failed {
  border-color: var(--color-error, #ef4444);
}

.task-toast--completed {
  border-color: rgba(34, 197, 94, 0.45);
}

.task-toast__body {
  flex: 1;
  min-width: 0;
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
  padding: 0.65rem 0.75rem;
  border: none;
  background: transparent;
  color: inherit;
  text-align: left;
  cursor: pointer;
  transition: background 0.15s ease;
}

.task-toast__body:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.04));
}

.task-toast__header {
  display: flex;
  align-items: center;
  gap: 0.45rem;
  min-width: 0;
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
  font-size: 0.8125rem;
  font-weight: 600;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-toast__percent {
  flex-shrink: 0;
  font-size: 0.72rem;
  font-weight: 600;
  color: var(--text-secondary, #9ca3af);
}

.task-toast__stop,
.task-toast__dismiss {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 2rem;
  flex-shrink: 0;
  border: none;
  border-left: 1px solid var(--border-primary, #2a2f45);
  background: transparent;
  color: var(--text-secondary, #9ca3af);
  cursor: pointer;
  transition: background 0.15s ease, color 0.15s ease;
}

.task-toast__stop:hover:not(:disabled),
.task-toast__dismiss:hover {
  background: var(--bg-card-hover, rgba(255, 255, 255, 0.06));
  color: var(--text-primary, #f3f4f6);
}

.task-toast__stop:disabled {
  cursor: wait;
  opacity: 0.7;
}

.task-toast__stop .pi-stop {
  color: #ef4444;
}

.task-toast__dismiss {
  /* inherits shared stop/dismiss chrome */
}

.task-toast-enter-active,
.task-toast-leave-active {
  transition: opacity 0.2s ease, transform 0.2s ease;
}

.task-toast-enter-from,
.task-toast-leave-to {
  opacity: 0;
  transform: translateY(0.75rem);
}

.task-toast-move {
  transition: transform 0.2s ease;
}

:deep(.task-toast .p-progressbar) {
  height: 0.35rem;
}

:deep(.task-detail-dialog .task-detail-panel) {
  margin: 0;
}
</style>
