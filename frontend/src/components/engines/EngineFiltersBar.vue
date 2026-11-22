<template>
  <PageToolbar class="engine-filters compact-toolbar" label="Engine filters">
    <label class="compact-field">
      <span class="compact-label sr-only">Task</span>
      <select id="engine-task-filter" v-model="task">
        <option value="all">All tasks</option>
        <option value="text">Text</option>
        <option value="embeddings">Embeddings</option>
        <option value="audio">Audio</option>
      </select>
    </label>
    <label class="compact-field">
      <span class="compact-label sr-only">Hardware</span>
      <select id="engine-hardware-filter" v-model="hardware">
        <option value="all">Any hardware</option>
        <option value="cpu">CPU-capable</option>
        <option value="sm70">SM70 / V100</option>
      </select>
    </label>
    <template v-if="summary" #summary>
      <span class="engine-filters__summary">{{ summary }}</span>
    </template>
    <template #actions>
      <Button
        class="engine-filters__reset"
        label="Reset filters"
        icon="pi pi-filter-slash"
        severity="secondary"
        text
        size="small"
        @click="emit('reset')"
      />
    </template>
  </PageToolbar>
</template>

<script setup>
import Button from 'primevue/button'
import PageToolbar from '@/components/common/PageToolbar.vue'

const task = defineModel('task', { type: String, default: 'all' })
const hardware = defineModel('hardware', { type: String, default: 'all' })

defineProps({
  /** Honest result count for the current filters, e.g. "Showing 3 of 11 engines". */
  summary: { type: String, default: '' },
})

const emit = defineEmits(['reset'])
</script>

<style scoped>
/* Layout only: control density and the touched state come from .compact-*. */
.engine-filters .compact-field {
  flex: 0 1 11rem;
  width: 11rem;
}

.engine-filters select {
  display: block;
  width: 100%;
  max-width: 100%;
}

.engine-filters__summary {
  font-size: var(--text-xs);
}

@media (max-width: 640px), (pointer: coarse) {
  .engine-filters .compact-field {
    flex: 1 1 auto;
    width: 100%;
  }
}
</style>
