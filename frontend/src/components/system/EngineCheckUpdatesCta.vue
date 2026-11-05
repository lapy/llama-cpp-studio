<template>
  <div class="engine-check-updates-cta">
    <Button
      :label="label"
      icon="pi pi-arrow-up-right"
      icon-pos="left"
      severity="info"
      :loading="loading"
      @click="$emit('check')"
    />
    <span class="engine-check-updates-cta__hint">{{ hint }}</span>
  </div>
</template>

<script setup>
import Button from 'primevue/button'

defineProps({
  loading: { type: Boolean, default: false },
  /** Primary action label (same for all engines). */
  label: { type: String, default: 'Check for updates' },
  /** Supporting line next to the button. */
  hint: {
    type: String,
    default: 'Compare the installed version to the latest upstream release.',
  },
})

defineEmits(['check'])
</script>

<style scoped>
.engine-check-updates-cta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.75rem 1rem;
  padding: 0.85rem 1rem;
  margin-bottom: 0.65rem;
  border-radius: var(--radius-lg);
  border: 1px solid rgba(96, 165, 250, 0.3);
  background: linear-gradient(
    145deg,
    rgba(96, 165, 250, 0.1) 0%,
    rgba(96, 165, 250, 0.03) 55%,
    transparent 100%
  );
}

.engine-check-updates-cta__hint {
  font-size: 0.8125rem;
  line-height: 1.45;
  color: var(--text-secondary);
  flex: 1;
  min-width: 12rem;
}

.engine-check-updates-cta :deep(.p-button) {
  font-weight: 600;
  flex-shrink: 0;
}
</style>
