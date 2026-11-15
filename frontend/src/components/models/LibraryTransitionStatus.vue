<template>
  <p v-if="entries.length" class="library-transition" role="status" aria-live="polite">
    <i class="pi pi-spin pi-spinner library-transition__spinner" aria-hidden="true" />
    <span class="library-transition__text">{{ summary }}</span>
    <span class="library-transition__hint">{{ hint }}</span>
  </p>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  /** `{ id, label, verb }` for every start/stop in flight. */
  entries: {
    type: Array,
    default: () => [],
  },
})

const starting = computed(() => props.entries.filter((entry) => entry.verb !== 'Stopping'))
const stopping = computed(() => props.entries.filter((entry) => entry.verb === 'Stopping'))

const summary = computed(() => {
  const parts = []
  if (starting.value.length === 1) parts.push(`Starting ${starting.value[0].label}…`)
  if (starting.value.length > 1) parts.push(`Starting ${starting.value.length} models…`)
  if (stopping.value.length === 1) parts.push(`Stopping ${stopping.value[0].label}…`)
  if (stopping.value.length > 1) parts.push(`Stopping ${stopping.value.length} models…`)
  return parts.join(' ')
})

const hint = computed(() => {
  if (starting.value.length && stopping.value.length) {
    return 'Other models keep their current state until their transition finishes.'
  }
  if (starting.value.length)
    return 'Waiting for the library to report Running. Stop cancels a start.'
  return 'Waiting for the library to report Stopped.'
})
</script>

<style scoped>
.library-transition {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem 0.5rem;
  margin: 0 0 0.5rem;
  padding: 0.35rem 0.65rem;
  border: 1px solid var(--border-secondary);
  border-radius: var(--radius-md);
  background: var(--bg-tertiary);
  color: var(--text-secondary);
  font-size: var(--text-sm);
}

.library-transition__spinner {
  color: var(--accent-cyan);
}

.library-transition__text {
  color: var(--text-primary);
  font-weight: 550;
}

.library-transition__hint {
  font-size: var(--text-xs);
}
</style>
