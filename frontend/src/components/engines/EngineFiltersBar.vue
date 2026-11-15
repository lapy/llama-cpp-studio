<template>
  <div class="engine-filters compact-toolbar">
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
    <button type="button" class="engine-filters__reset" @click="emit('reset')">
      Reset filters
    </button>
  </div>
</template>

<script setup>
const task = defineModel('task', { type: String, default: 'all' })
const hardware = defineModel('hardware', { type: String, default: 'all' })
const emit = defineEmits(['reset'])
</script>

<style scoped>
.engine-filters {
  display: flex;
  flex-wrap: wrap;
  align-items: end;
  gap: 0.5rem;
  min-width: 0;
  margin-bottom: 0.75rem;
  padding: 0.5rem;
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  background: color-mix(in srgb, var(--bg-surface) 74%, transparent);
}

.engine-filters .compact-field {
  flex: 0 1 11rem;
  width: 11rem;
}

.engine-filters select {
  display: block;
  width: 100%;
  min-height: var(--compact-control-height);
  max-width: 100%;
  padding: 0.3rem 1.75rem 0.3rem 0.5rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-sm);
  background: var(--bg-secondary);
  color: var(--text-primary);
  font-size: 0.8125rem;
}

.engine-filters__reset {
  min-height: var(--compact-control-height);
  padding: 0 0.4rem;
  border: none;
  background: transparent;
  color: var(--accent-cyan);
  font: inherit;
  font-size: 0.8125rem;
  cursor: pointer;
}

.engine-filters__reset:hover {
  text-decoration: underline;
}

@media (max-width: 640px), (pointer: coarse) {
  .engine-filters .compact-field {
    flex: 1 1 auto;
    width: 100%;
  }

  .engine-filters select,
  .engine-filters__reset {
    font-size: 0.9375rem;
  }
}
</style>
