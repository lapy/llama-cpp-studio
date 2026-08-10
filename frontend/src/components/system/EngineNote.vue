<template>
  <div class="engine-note" :class="`engine-note--${severity}`" role="status">
    <i :class="['pi', iconClass]" aria-hidden="true" />
    <div class="engine-note__body">
      <slot />
      <div v-if="$slots.actions" class="engine-note__actions">
        <slot name="actions" />
      </div>
    </div>
  </div>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  severity: {
    type: String,
    default: 'info',
    validator: (v) => ['info', 'warning'].includes(v),
  },
})

const iconClass = computed(() =>
  props.severity === 'warning' ? 'pi-exclamation-triangle' : 'pi-info-circle',
)
</script>

<style scoped>
.engine-note {
  display: flex;
  align-items: flex-start;
  gap: 0.5rem;
  font-size: 0.8125rem;
  line-height: 1.45;
  color: var(--text-secondary);
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: 0.75rem 0.9rem;
}

.engine-note :deep(code) {
  font-size: 0.75rem;
}

.engine-note--warning {
  color: var(--accent-amber, #f59e0b);
  border-color: color-mix(in srgb, var(--accent-amber, #f59e0b) 45%, transparent);
  background: color-mix(in srgb, var(--accent-amber, #f59e0b) 8%, transparent);
}

.engine-note__body {
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
  min-width: 0;
  flex: 1;
}

.engine-note__actions {
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
</style>
