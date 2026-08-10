<template>
  <div class="engine-install-panel">
    <div class="engine-install-panel__head">
      <span class="engine-install-panel__title">{{ title }}</span>
      <span v-if="subtitle" class="engine-install-panel__subtitle">{{ subtitle }}</span>
    </div>
    <div class="engine-install-panel__actions">
      <slot />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, default: 'Install' },
  subtitle: { type: String, default: '' },
})
</script>

<style scoped>
.engine-install-panel {
  background: var(--bg-surface);
  border: 1px solid var(--border-primary);
  border-radius: var(--radius-lg);
  padding: 1rem 1.15rem;
}

.engine-install-panel__head {
  margin-bottom: 0.85rem;
}

.engine-install-panel__title {
  display: block;
  font-size: 0.7rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--text-secondary);
  margin-bottom: 0.35rem;
}

.engine-install-panel__subtitle {
  display: block;
  font-size: 0.8125rem;
  line-height: 1.45;
  color: var(--text-secondary);
}

.engine-install-panel__actions {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.65rem;
}

@media (max-width: 520px) {
  .engine-install-panel__actions {
    grid-template-columns: 1fr;
  }
}
</style>
