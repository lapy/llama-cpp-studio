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
            <p v-if="task.message" class="task-toast__message">{{ task.message }}</p>
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
  right: max(1rem, env(safe-area-inset-right));
  bottom: max(1rem, env(safe-area-inset-bottom));
  /* Above PrimeVue dialogs/masks (~1100+), tooltips (11000), and tour highlights. */
  z-index: 20000;
  display: flex;
  flex-direction: column-reverse;
  gap: 0.65rem;
  width: min(26rem, calc(100vw - 1.5rem));
  pointer-events: none;
}

.task-toast {
  display: flex;
  align-items: stretch;
  gap: 0;
  pointer-events: auto;
  border-radius: 0.75rem;
  border: 1px solid color-mix(in srgb, var(--border-primary, #3b4261) 80%, white 20%);
  background:
    linear-gradient(
      180deg,
      color-mix(in srgb, var(--bg-surface, #252b40) 88%, white 12%) 0%,
      var(--bg-surface, #1e2235) 100%
    );
  box-shadow:
    0 0 0 1px rgba(255, 255, 255, 0.04),
    0 18px 40px rgba(0, 0, 0, 0.45),
    0 4px 12px rgba(0, 0, 0, 0.3);
  overflow: hidden;
  position: relative;
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
  gap: 0.5rem;
  padding: 0.85rem 0.9rem 0.85rem 1.05rem;
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
  font-size: 0.9375rem;
  font-weight: 700;
  letter-spacing: 0.01em;
  color: var(--text-primary, #f3f4f6);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.task-toast__percent {
  flex-shrink: 0;
  min-width: 2.75rem;
  text-align: right;
  font-size: 0.875rem;
  font-weight: 750;
  font-variant-numeric: tabular-nums;
  color: var(--text-primary, #f3f4f6);
}

.task-toast__message {
  margin: 0;
  font-size: 0.78rem;
  line-height: 1.35;
  color: var(--text-secondary, #c4c9d4);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

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

.task-toast-enter-active,
.task-toast-leave-active {
  transition: opacity 0.22s ease, transform 0.22s ease;
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

:deep(.task-detail-dialog .task-detail-panel) {
  margin: 0;
}
</style>
