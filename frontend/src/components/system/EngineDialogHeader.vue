<template>
  <div class="engine-dialog-header">
    <div class="engine-dialog-header__main">
      <span v-if="$slots.leading" class="engine-dialog-header__leading">
        <slot name="leading" />
      </span>
      <span class="engine-dialog-header__title">{{ title }}</span>
      <span v-if="$slots.tags" class="engine-dialog-header__tags">
        <slot name="tags" />
      </span>
    </div>
    <div v-if="$slots.actions" class="engine-dialog-header__actions">
      <slot name="actions" />
    </div>
  </div>
</template>

<script setup>
defineProps({
  title: { type: String, required: true },
})
</script>

<style scoped>
/* Fills the header slot so actions sit just before PrimeVue’s maximize/close icons. */
.engine-dialog-header {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
}

.engine-dialog-header__main {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex: 1;
  min-width: 0;
  flex-wrap: wrap;
}

.engine-dialog-header__leading {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  flex-shrink: 0;
}

.engine-dialog-header__leading :deep(.pi) {
  font-size: 1.25rem;
  color: var(--accent-cyan);
}

.engine-dialog-header__title {
  font-size: 1.25rem;
  font-weight: 600;
  line-height: 1.3;
  margin: 0;
  min-width: 0;
}

.engine-dialog-header__tags {
  display: inline-flex;
  align-items: center;
  gap: 0.35rem;
  flex-wrap: wrap;
}

.engine-dialog-header__actions {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-shrink: 0;
  /* Visually group with PrimeVue maximize/close (they use margin-right: 0.5rem between icons). */
  margin-right: 0.5rem;
}

/* Icon-only text buttons: global styles in styles/_components.css */
</style>
