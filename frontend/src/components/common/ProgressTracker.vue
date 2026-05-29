<template>
  <TaskDetailPanel
    :type="type"
    :metadata-key="metadataKey"
    :metadata-value="metadataValue"
    :task-id="taskId"
    :show-completed="showCompleted"
    :section-title="sectionTitle"
    :dismissible="dismissible"
    class="progress-tracker"
  />
</template>

<script setup>
import TaskDetailPanel from '@/components/common/TaskDetailPanel.vue'

defineProps({
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
  taskId: {
    type: [String, Number],
    default: null,
  },
  showCompleted: {
    type: Boolean,
    default: false,
  },
  sectionTitle: {
    type: String,
    default: null,
  },
  dismissible: {
    type: Boolean,
    default: true,
  },
})
</script>

<style scoped>
.progress-tracker {
  margin: var(--spacing-md, 0.75rem) 0;
}
</style>
